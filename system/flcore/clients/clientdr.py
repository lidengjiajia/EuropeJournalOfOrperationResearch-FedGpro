import torch
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client


class clientDR(Client):
    """
    FedDR+: Federated Douglas-Rachford+
    An improved federated optimization algorithm based on Douglas-Rachford splitting.
    Category: Traditional FL (tFL) - Advanced Optimization
    
    Key Features:
    - Uses Douglas-Rachford splitting for better convergence
    - Maintains dual variables for constrained optimization
    - Better handles non-convex objectives and heterogeneous data
    - Improved version with adaptive penalty parameter
    
    Reference: "FedDR - Randomized Douglas-Rachford Splitting Algorithms for 
                Nonconvex Federated Composite Optimization" (NeurIPS 2021)
    """
    
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # Douglas-Rachford penalty parameter (rho)
        self.rho = args.dr_rho if hasattr(args, 'dr_rho') else 0.1
        
        # Adaptive rho adjustment
        self.adaptive_rho = args.dr_adaptive_rho if hasattr(args, 'dr_adaptive_rho') else True
        self.rho_increase_factor = 2.0
        self.rho_decrease_factor = 0.5
        
        # Dual variables (Lagrange multipliers) for consensus constraints
        self.dual_variables = []
        for param in self.model.parameters():
            dual_var = torch.zeros_like(param.data)
            self.dual_variables.append(dual_var)
        
        # Store global model parameters for DR splitting
        self.global_params = [torch.zeros_like(p.data) for p in self.model.parameters()]
        
        # Store auxiliary variables for DR iterations
        self.auxiliary_vars = [torch.zeros_like(p.data) for p in self.model.parameters()]
        
        # Track primal and dual residuals for adaptive rho
        self.primal_residual = 0.0
        self.dual_residual = 0.0

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                
                output = self.model(x)
                loss = self.loss(output, y)
                
                # Add Douglas-Rachford penalty terms
                # Augmented Lagrangian: L(w) + <lambda, w - z> + (rho/2)||w - z||^2
                dr_penalty = 0.0
                for param, global_param, dual_var in zip(self.model.parameters(), 
                                                         self.global_params, 
                                                         self.dual_variables):
                    # Consensus constraint: w should be close to global model z
                    diff = param - global_param
                    
                    # Dual variable term: <lambda, w - z>
                    dr_penalty += torch.sum(dual_var * diff)
                    
                    # Quadratic penalty: (rho/2)||w - z||^2
                    dr_penalty += (self.rho / 2.0) * torch.sum(diff ** 2)
                
                total_loss = loss + dr_penalty
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
        
        # Douglas-Rachford auxiliary variable update
        # z^{k+1} = prox_{g}(2*w^{k+1} - z^k - (1/rho)*lambda^k)
        self._update_auxiliary_variables()
        
        # Dual variable update
        # lambda^{k+1} = lambda^k + rho*(w^{k+1} - z^{k+1})
        self._update_dual_variables()
        
        # Compute residuals for adaptive rho
        self._compute_residuals()
        
        # Adaptive rho adjustment
        if self.adaptive_rho:
            self._adjust_rho()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def _update_auxiliary_variables(self):
        """
        Douglas-Rachford auxiliary variable update.
        This represents the "splitting" in Douglas-Rachford.
        """
        for aux_var, param, global_param, dual_var in zip(self.auxiliary_vars,
                                                          self.model.parameters(),
                                                          self.global_params,
                                                          self.dual_variables):
            # z = prox(2*w - z_old - (1/rho)*lambda)
            # For simple case, use soft update toward current parameter
            update_target = 2 * param.data - aux_var.data - (1.0 / self.rho) * dual_var.data
            
            # Soft update (proximal operator approximation)
            aux_var.data = 0.5 * (aux_var.data + update_target)

    def _update_dual_variables(self):
        """
        Update dual variables (Lagrange multipliers).
        Enforces consensus constraint between local and global models.
        """
        for dual_var, param, aux_var in zip(self.dual_variables,
                                           self.model.parameters(),
                                           self.auxiliary_vars):
            # lambda = lambda + rho * (w - z)
            residual = param.data - aux_var.data
            dual_var.data += self.rho * residual

    def _compute_residuals(self):
        """
        Compute primal and dual residuals for convergence monitoring.
        """
        # Primal residual: ||w - z||
        primal_res_sq = 0.0
        for param, aux_var in zip(self.model.parameters(), self.auxiliary_vars):
            primal_res_sq += torch.sum((param.data - aux_var.data) ** 2).item()
        self.primal_residual = np.sqrt(primal_res_sq)
        
        # Dual residual: ||rho * (z_{k+1} - z_k)||
        dual_res_sq = 0.0
        for aux_var, global_param in zip(self.auxiliary_vars, self.global_params):
            dual_res_sq += torch.sum((aux_var.data - global_param.data) ** 2).item()
        self.dual_residual = self.rho * np.sqrt(dual_res_sq)

    def _adjust_rho(self):
        """
        Adaptive penalty parameter adjustment based on residuals.
        Balance primal and dual residuals for better convergence.
        """
        # If primal residual is much larger than dual residual, increase rho
        if self.primal_residual > 10 * self.dual_residual:
            self.rho *= self.rho_increase_factor
            # Scale dual variables to maintain constraint
            for dual_var in self.dual_variables:
                dual_var.data /= self.rho_increase_factor
        
        # If dual residual is much larger than primal residual, decrease rho
        elif self.dual_residual > 10 * self.primal_residual:
            self.rho *= self.rho_decrease_factor
            # Scale dual variables to maintain constraint
            for dual_var in self.dual_variables:
                dual_var.data /= self.rho_decrease_factor
        
        # Clamp rho to reasonable range
        self.rho = np.clip(self.rho, 0.01, 10.0)

    def set_parameters(self, model):
        """
        Override to update global parameters for DR splitting.
        """
        # Update local model
        super().set_parameters(model)
        
        # Update global model reference
        for global_param, new_param in zip(self.global_params, model.parameters()):
            global_param.data = new_param.data.clone()
        
        # Update auxiliary variables toward new global model
        for aux_var, global_param in zip(self.auxiliary_vars, self.global_params):
            aux_var.data = global_param.data.clone()

    def get_dr_diagnostics(self):
        """
        Return Douglas-Rachford diagnostic information.
        """
        dual_norm = sum([torch.norm(dv).item() for dv in self.dual_variables])
        
        return {
            'primal_residual': self.primal_residual,
            'dual_residual': self.dual_residual,
            'rho': self.rho,
            'dual_norm': dual_norm
        }

    def train_metrics(self):
        """
        Override to include DR penalty in training loss.
        """
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                
                # Add DR penalty
                dr_penalty = 0.0
                for param, global_param, dual_var in zip(self.model.parameters(),
                                                        self.global_params,
                                                        self.dual_variables):
                    diff = param - global_param
                    dr_penalty += torch.sum(dual_var * diff)
                    dr_penalty += (self.rho / 2.0) * torch.sum(diff ** 2)
                
                total_loss = loss + dr_penalty
                
                train_num += y.shape[0]
                losses += total_loss.item() * y.shape[0]

        return losses, train_num
