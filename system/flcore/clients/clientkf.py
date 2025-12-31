import torch
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client


class clientKF(Client):
    """
    FedKF: Federated Kalman Filter
    Uses Kalman filtering for more robust parameter updates in federated learning.
    Category: Traditional FL (tFL) - Robust Aggregation
    
    Key Features:
    - Maintains state covariance matrix for uncertainty quantification
    - Applies Kalman gain for adaptive parameter updates
    - Better handles noisy gradients and non-IID data
    
    Reference: "Federated Kalman Filter for Secure Cooperative Learning" (2021)
    """
    
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # Kalman filter hyperparameters
        self.process_noise = args.kf_process_noise if hasattr(args, 'kf_process_noise') else 0.01
        self.measurement_noise = args.kf_measurement_noise if hasattr(args, 'kf_measurement_noise') else 0.1
        
        # Initialize state covariance matrix P for each parameter
        self.state_covariances = []
        for param in self.model.parameters():
            # Initialize with identity matrix scaled by process noise
            cov = torch.ones_like(param.data) * self.process_noise
            self.state_covariances.append(cov)
        
        # Store previous model for change detection
        self.previous_model = copy.deepcopy(self.model)
        
        # Initialize Kalman gain storage
        self.kalman_gains = []
        for param in self.model.parameters():
            self.kalman_gains.append(torch.zeros_like(param.data))

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
                self.optimizer.zero_grad()
                loss.backward()
                
                # Apply Kalman filtering to gradients
                self._apply_kalman_filter()
                
                self.optimizer.step()

        # Update state covariance after training
        self._update_covariance()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def _apply_kalman_filter(self):
        """
        Apply Kalman filter to gradients before optimization step.
        This reduces noise and provides more stable updates.
        """
        for param, cov, kalman_gain in zip(self.model.parameters(), 
                                           self.state_covariances, 
                                           self.kalman_gains):
            if param.grad is not None:
                # Prediction step: P_k = P_{k-1} + Q (process noise)
                cov.data += self.process_noise
                
                # Compute Kalman gain: K = P / (P + R)
                # where R is measurement noise
                kalman_gain.data = cov.data / (cov.data + self.measurement_noise)
                
                # Update step: gradient is the "measurement"
                # Filter the gradient: grad_filtered = K * grad
                param.grad.data = kalman_gain.data * param.grad.data
                
                # Update covariance: P = (1 - K) * P
                cov.data = (1 - kalman_gain.data) * cov.data

    def _update_covariance(self):
        """
        Update state covariance based on parameter changes.
        Larger changes indicate higher uncertainty.
        """
        for param, prev_param, cov in zip(self.model.parameters(), 
                                          self.previous_model.parameters(), 
                                          self.state_covariances):
            # Compute parameter change
            param_change = (param.data - prev_param.data).abs()
            
            # Update covariance based on change magnitude
            # Larger changes -> higher uncertainty -> larger covariance
            cov.data += param_change * self.process_noise
            
            # Clamp covariance to prevent numerical instability
            cov.data = torch.clamp(cov.data, min=1e-6, max=1.0)
        
        # Update previous model
        for param, prev_param in zip(self.model.parameters(), 
                                    self.previous_model.parameters()):
            prev_param.data = param.data.clone()

    def set_parameters(self, model):
        """
        Override to also update Kalman filter state when receiving global model.
        """
        # Standard parameter update
        super().set_parameters(model)
        
        # Reset previous model to current state
        for param, prev_param in zip(self.model.parameters(), 
                                    self.previous_model.parameters()):
            prev_param.data = param.data.clone()
        
        # Slightly increase covariance after receiving global model
        # (reflects uncertainty from aggregation)
        for cov in self.state_covariances:
            cov.data += self.process_noise * 2.0

    def get_kalman_diagnostics(self):
        """
        Return diagnostic information about Kalman filter state.
        Useful for monitoring convergence.
        """
        avg_covariance = torch.mean(torch.stack([cov.mean() for cov in self.state_covariances]))
        avg_kalman_gain = torch.mean(torch.stack([kg.mean() for kg in self.kalman_gains]))
        
        return {
            'avg_covariance': avg_covariance.item(),
            'avg_kalman_gain': avg_kalman_gain.item()
        }
