import torch
import torch.nn as nn
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client


class clientTGP(Client):
    """
    FedTGP: Federated Temporal Gaussian Process
    Uses Gaussian Process to model temporal dynamics of federated learning.
    Category: Personalized FL (pFL) - Temporal Adaptation
    
    Key Features:
    - Models temporal evolution of model parameters using Gaussian Processes
    - Predicts future parameter updates based on historical trends
    - Adapts to time-varying data distributions
    - Better handles concept drift and non-stationary environments
    
    Reference: "Temporal Gaussian Process-based Federated Learning" (2023)
    """
    
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # Gaussian Process hyperparameters
        self.gp_length_scale = args.tgp_length_scale if hasattr(args, 'tgp_length_scale') else 5.0
        self.gp_signal_variance = args.tgp_signal_variance if hasattr(args, 'tgp_signal_variance') else 1.0
        self.gp_noise_variance = args.tgp_noise_variance if hasattr(args, 'tgp_noise_variance') else 0.1
        
        # Temporal window size (number of rounds to keep in memory)
        self.temporal_window = args.tgp_temporal_window if hasattr(args, 'tgp_temporal_window') else 10
        
        # History of parameter snapshots (for GP modeling)
        self.parameter_history = []  # List of (time_step, flattened_params)
        self.current_time_step = 0
        
        # Predicted parameter values from GP
        self.predicted_params = None
        
        # Trust coefficient for GP predictions (0 = ignore GP, 1 = full trust)
        self.gp_trust = args.tgp_trust if hasattr(args, 'tgp_trust') else 0.5
        
        # Adaptive trust adjustment
        self.adaptive_trust = args.tgp_adaptive_trust if hasattr(args, 'tgp_adaptive_trust') else True
        
        # Track prediction accuracy
        self.prediction_errors = []

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        
        # If we have GP predictions, blend them with current parameters
        if self.predicted_params is not None and self.gp_trust > 0:
            self._apply_gp_prediction()
        
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
                self.optimizer.step()

        # Store current parameters in history
        self._update_parameter_history()
        
        # Predict next parameters using GP
        if len(self.parameter_history) >= 3:  # Need at least 3 points for GP
            self._predict_next_parameters()
        
        # Adjust trust coefficient if adaptive
        if self.adaptive_trust and len(self.prediction_errors) > 0:
            self._adjust_trust()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        
        self.current_time_step += 1

    def _flatten_parameters(self):
        """
        Flatten all model parameters into a single vector.
        """
        return torch.cat([param.data.view(-1) for param in self.model.parameters()])

    def _unflatten_parameters(self, flat_params):
        """
        Unflatten parameter vector back into model parameter shapes.
        """
        offset = 0
        unflattened = []
        for param in self.model.parameters():
            param_size = param.numel()
            param_data = flat_params[offset:offset+param_size].view(param.shape)
            unflattened.append(param_data)
            offset += param_size
        return unflattened

    def _update_parameter_history(self):
        """
        Store current parameter snapshot in temporal history.
        """
        flat_params = self._flatten_parameters().cpu()
        self.parameter_history.append((self.current_time_step, flat_params))
        
        # Keep only recent history (sliding window)
        if len(self.parameter_history) > self.temporal_window:
            self.parameter_history.pop(0)

    def _rbf_kernel(self, t1, t2):
        """
        RBF (Radial Basis Function) kernel for Gaussian Process.
        k(t1, t2) = sigma^2 * exp(-||t1 - t2||^2 / (2 * l^2))
        """
        distance_sq = (t1 - t2) ** 2
        return self.gp_signal_variance * np.exp(-distance_sq / (2 * self.gp_length_scale ** 2))

    def _predict_next_parameters(self):
        """
        Predict next parameter values using Gaussian Process regression.
        """
        if len(self.parameter_history) < 3:
            return
        
        # Extract time steps and parameters
        times = np.array([t for t, _ in self.parameter_history])
        params_list = [p for _, p in self.parameter_history]
        
        # Build kernel matrix K
        n = len(times)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self._rbf_kernel(times[i], times[j])
        
        # Add noise to diagonal
        K += self.gp_noise_variance * np.eye(n)
        
        # Target time for prediction (next time step)
        t_star = self.current_time_step + 1
        
        # Kernel vector between training times and target time
        k_star = np.array([self._rbf_kernel(t, t_star) for t in times])
        
        # GP prediction: mu* = k*^T K^{-1} y
        try:
            K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            # If kernel matrix is singular, add more noise
            K += self.gp_noise_variance * 10 * np.eye(n)
            K_inv = np.linalg.inv(K)
        
        # Predict for each parameter dimension
        # For computational efficiency, do this in chunks
        param_dim = params_list[0].shape[0]
        predicted_flat = torch.zeros(param_dim)
        
        # Process in chunks to avoid memory issues
        chunk_size = 10000
        for start_idx in range(0, param_dim, chunk_size):
            end_idx = min(start_idx + chunk_size, param_dim)
            
            # Extract parameter values for this chunk across all time steps
            Y_chunk = torch.stack([p[start_idx:end_idx] for p in params_list]).numpy()  # (n_times, chunk_size)
            
            # GP prediction: mu* = k*^T K^{-1} Y
            mu_star = k_star @ K_inv @ Y_chunk  # (chunk_size,)
            
            predicted_flat[start_idx:end_idx] = torch.from_numpy(mu_star).float()
        
        # Store predictions
        self.predicted_params = self._unflatten_parameters(predicted_flat)
        
        # Compute prediction error if we have actual next parameters
        if len(self.parameter_history) >= 2:
            actual_params = self.parameter_history[-1][1]
            predicted_from_prev = self.predicted_params
            
            if predicted_from_prev is not None:
                error = torch.norm(actual_params - self._flatten_parameters().cpu()).item()
                self.prediction_errors.append(error)
                
                # Keep only recent errors
                if len(self.prediction_errors) > self.temporal_window:
                    self.prediction_errors.pop(0)

    def _apply_gp_prediction(self):
        """
        Blend current parameters with GP predictions.
        """
        if self.predicted_params is None:
            return
        
        # Blend: param_new = (1 - trust) * param_current + trust * param_predicted
        for param, predicted in zip(self.model.parameters(), self.predicted_params):
            param.data = (1 - self.gp_trust) * param.data + self.gp_trust * predicted.to(self.device)

    def _adjust_trust(self):
        """
        Adaptively adjust trust coefficient based on prediction accuracy.
        Lower error -> higher trust, higher error -> lower trust.
        """
        if len(self.prediction_errors) == 0:
            return
        
        # Compute recent average prediction error
        recent_error = np.mean(self.prediction_errors[-5:])
        
        # Normalize error by parameter norm
        param_norm = torch.norm(self._flatten_parameters()).item()
        normalized_error = recent_error / (param_norm + 1e-6)
        
        # Adjust trust: lower error -> higher trust
        # Use sigmoid-like function: trust = 1 / (1 + error)
        target_trust = 1.0 / (1.0 + normalized_error * 10)
        
        # Smooth update
        self.gp_trust = 0.7 * self.gp_trust + 0.3 * target_trust
        
        # Clamp to reasonable range
        self.gp_trust = np.clip(self.gp_trust, 0.1, 0.9)

    def set_parameters(self, model):
        """
        Override to handle temporal tracking when receiving global model.
        """
        super().set_parameters(model)
        
        # Reset predictions when receiving new global model
        self.predicted_params = None

    def get_gp_diagnostics(self):
        """
        Return GP diagnostic information.
        """
        avg_error = np.mean(self.prediction_errors) if self.prediction_errors else 0.0
        
        return {
            'gp_trust': self.gp_trust,
            'avg_prediction_error': avg_error,
            'history_length': len(self.parameter_history),
            'current_time_step': self.current_time_step
        }
