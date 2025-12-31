import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client


class clientDrPlus(Client):
    """
    FedDr+: Federated Distillation with Prototype Enhancement
    Combines prototype-based knowledge distillation for better personalization.
    Category: Personalized FL (pFL) - Knowledge Distillation
    
    Key Features:
    - Maintains class prototypes (centroids) for each client
    - Uses distillation loss to align local model with global knowledge
    - Prototype enhancement for better handling of data heterogeneity
    
    Reference: "FedDr+: Enhanced Federated Distillation with Prototype Regularization" (2023)
    """
    
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # Distillation parameters
        self.distill_alpha = args.distill_alpha if hasattr(args, 'distill_alpha') else 0.5
        self.distill_temperature = args.distill_temperature if hasattr(args, 'distill_temperature') else 3.0
        self.prototype_lambda = args.prototype_lambda if hasattr(args, 'prototype_lambda') else 0.1
        
        # Store global model for distillation
        self.global_model = None
        
        # Initialize prototypes: one prototype per class
        # Will be computed during training
        self.class_prototypes = {}
        self.prototype_counts = {}
        
        # Feature extractor (all layers except final classification layer)
        self.feature_dim = self._get_feature_dim()


    def _get_feature_dim(self):
        """Get dimension of feature space (before final classification layer)"""
        # Assume the model has a structure where last layer is classifier
        # This works for most CNN architectures (e.g., ResNet, VGG)
        if hasattr(self.model, 'fc'):  # ResNet-like
            return self.model.fc.in_features
        elif hasattr(self.model, 'classifier'):  # VGG-like or MobileNet
            if isinstance(self.model.classifier, nn.Sequential):
                for layer in reversed(self.model.classifier):
                    if isinstance(layer, nn.Linear):
                        return layer.in_features
            elif isinstance(self.model.classifier, nn.Linear):
                return self.model.classifier.in_features
        elif hasattr(self.model, 'head'):  # Vision Transformer
            return self.model.head.in_features
        else:
            # Default: try to infer from a forward pass
            try:
                dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
                features = self.get_features(dummy_input)
                return features.shape[1]
            except:
                return 512  # Default fallback
        return 512


    def get_features(self, x):
        """Extract features before final classification layer"""
        if hasattr(self.model, 'fc'):  # ResNet-like
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            if hasattr(self.model, 'layer4'):
                x = self.model.layer4(x)
            
            x = self.model.avgpool(x)
            features = torch.flatten(x, 1)
        else:
            # For simpler models, use feature extraction before last layer
            features = x
            for name, module in self.model.named_children():
                if 'fc' in name or 'classifier' in name or 'head' in name:
                    break
                features = module(features)
            features = torch.flatten(features, 1)
        
        return features


    def compute_prototypes(self, trainloader):
        """Compute class prototypes (centroids) from training data"""
        self.model.eval()
        
        # Initialize prototype accumulators
        prototype_sums = {i: torch.zeros(self.feature_dim).to(self.device) for i in range(self.num_classes)}
        prototype_counts = {i: 0 for i in range(self.num_classes)}
        
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                # Extract features
                features = self.get_features(x)
                
                # Accumulate features for each class
                for i in range(len(y)):
                    class_id = y[i].item()
                    prototype_sums[class_id] += features[i]
                    prototype_counts[class_id] += 1
        
        # Compute centroids
        for class_id in range(self.num_classes):
            if prototype_counts[class_id] > 0:
                self.class_prototypes[class_id] = prototype_sums[class_id] / prototype_counts[class_id]
                self.prototype_counts[class_id] = prototype_counts[class_id]
            else:
                # No samples for this class, use zero vector
                self.class_prototypes[class_id] = torch.zeros(self.feature_dim).to(self.device)
                self.prototype_counts[class_id] = 0


    def prototype_loss(self, features, labels):
        """Compute prototype alignment loss"""
        if len(self.class_prototypes) == 0:
            return torch.tensor(0.0).to(self.device)
        
        loss = 0.0
        count = 0
        for i in range(len(labels)):
            class_id = labels[i].item()
            if class_id in self.class_prototypes and self.prototype_counts[class_id] > 0:
                # L2 distance between feature and prototype
                prototype = self.class_prototypes[class_id]
                loss += F.mse_loss(features[i], prototype.detach())
                count += 1
        
        return loss / max(count, 1)


    def distillation_loss(self, student_logits, teacher_logits, labels, temperature):
        """Compute knowledge distillation loss"""
        # Soft target loss (KL divergence)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction='batchmean'
        ) * (temperature ** 2)
        
        # Hard target loss (standard cross-entropy)
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Weighted combination
        return self.distill_alpha * soft_loss + (1 - self.distill_alpha) * hard_loss


    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        
        start_time = time.time()

        # Compute prototypes before training
        self.compute_prototypes(trainloader)

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
                
                # Forward pass - local model
                features = self.get_features(x)
                output = self.model(x)
                
                # Compute loss
                if self.global_model is not None:
                    # Knowledge distillation loss
                    with torch.no_grad():
                        teacher_output = self.global_model(x)
                    loss = self.distillation_loss(output, teacher_output, y, self.distill_temperature)
                else:
                    # Standard cross-entropy loss
                    loss = self.loss(output, y)
                
                # Add prototype regularization
                if len(self.class_prototypes) > 0:
                    proto_loss = self.prototype_loss(features, y)
                    loss += self.prototype_lambda * proto_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_parameters(self, model):
        """Set global model and store for distillation"""
        super().set_parameters(model)
        # Store a copy of global model for distillation
        self.global_model = copy.deepcopy(model)
        self.global_model.eval()
        for param in self.global_model.parameters():
            param.requires_grad = False


    def get_prototypes(self):
        """Return class prototypes for server aggregation"""
        return self.class_prototypes, self.prototype_counts
