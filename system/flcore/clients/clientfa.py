import torch
import torch.nn as nn
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client


class clientFA(Client):
    """
    FedFA: Federated Focal Average
    Uses focal loss and adaptive weighting to focus on hard samples.
    Category: Traditional FL (tFL) - Loss-based Enhancement
    
    Key Features:
    - Focal loss to emphasize hard samples during training
    - Adaptive sample weighting based on loss magnitude
    - Better handles class imbalance and difficult examples
    
    Reference: "FedFA: Federated Feature Augmentation" (2022)
    """
    
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # Focal loss hyperparameters
        self.focal_alpha = args.fa_alpha if hasattr(args, 'fa_alpha') else 0.25
        self.focal_gamma = args.fa_gamma if hasattr(args, 'fa_gamma') else 2.0
        
        # Adaptive weighting parameters
        self.use_adaptive_weight = args.fa_adaptive_weight if hasattr(args, 'fa_adaptive_weight') else True
        self.weight_momentum = args.fa_weight_momentum if hasattr(args, 'fa_weight_momentum') else 0.9
        
        # Replace standard loss with focal loss
        self.loss = self._create_focal_loss()
        
        # Track sample difficulties (running average)
        self.sample_difficulties = None
        
        # Store per-class weights for imbalanced data
        self.class_weights = None

    def _create_focal_loss(self):
        """
        Create focal loss function.
        Focal Loss = -alpha * (1-pt)^gamma * log(pt)
        where pt is the predicted probability for the correct class.
        """
        def focal_loss(inputs, targets):
            ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)  # Probability for correct class
            focal_weight = (1 - pt) ** self.focal_gamma
            
            # Apply alpha weighting if specified
            if self.focal_alpha is not None:
                alpha_t = self.focal_alpha * torch.ones_like(targets, dtype=torch.float)
                focal_loss = alpha_t * focal_weight * ce_loss
            else:
                focal_loss = focal_weight * ce_loss
            
            return focal_loss.mean()
        
        return focal_loss

    def _compute_class_weights(self, trainloader):
        """
        Compute class weights based on inverse frequency.
        Helps handle class imbalance.
        """
        class_counts = torch.zeros(self.num_classes)
        
        for x, y in trainloader:
            for label in y:
                class_counts[label.item()] += 1
        
        # Inverse frequency weighting
        total_samples = class_counts.sum()
        class_weights = total_samples / (self.num_classes * class_counts + 1e-6)
        
        # Normalize weights
        class_weights = class_weights / class_weights.sum() * self.num_classes
        
        return class_weights.to(self.device)

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        
        # Compute class weights if not already done
        if self.class_weights is None:
            self.class_weights = self._compute_class_weights(trainloader)
            print(f"Client {self.id} - Class weights: {self.class_weights.cpu().numpy()}")
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        # Track batch losses for adaptive weighting
        batch_losses = []

        for epoch in range(max_local_epochs):
            epoch_losses = []
            
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                
                output = self.model(x)
                
                # Compute focal loss
                loss = self.loss(output, y)
                
                # Apply class weighting
                if self.class_weights is not None:
                    # Recompute with class weights
                    ce_loss = nn.functional.cross_entropy(output, y, weight=self.class_weights, reduction='none')
                    pt = torch.exp(-ce_loss)
                    focal_weight = (1 - pt) ** self.focal_gamma
                    loss = (self.focal_alpha * focal_weight * ce_loss).mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Apply adaptive gradient clipping based on loss magnitude
                if self.use_adaptive_weight:
                    self._apply_adaptive_clipping(loss.item())
                
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
            
            batch_losses.append(np.mean(epoch_losses))

        # Update sample difficulties (exponential moving average)
        if self.sample_difficulties is None:
            self.sample_difficulties = np.mean(batch_losses)
        else:
            self.sample_difficulties = (self.weight_momentum * self.sample_difficulties + 
                                       (1 - self.weight_momentum) * np.mean(batch_losses))

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def _apply_adaptive_clipping(self, loss_value):
        """
        Apply adaptive gradient clipping based on loss magnitude.
        Prevents gradient explosion on hard samples.
        """
        # Compute gradient norm
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Adaptive clipping threshold based on loss
        # Higher loss -> more lenient clipping
        clip_threshold = 1.0 + loss_value
        
        if total_norm > clip_threshold:
            clip_coef = clip_threshold / (total_norm + 1e-6)
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)

    def get_sample_difficulty(self):
        """
        Return current sample difficulty estimate.
        Can be used by server for adaptive aggregation.
        """
        return self.sample_difficulties if self.sample_difficulties is not None else 0.0

    def train_metrics(self):
        """
        Override to use focal loss in evaluation.
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
                
                # Use focal loss
                loss = self.loss(output, y)
                
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
