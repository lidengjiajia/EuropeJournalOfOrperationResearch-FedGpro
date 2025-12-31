"""
Credit Scoring Neural Network Models for Federated Learning

This module implements specialized neural network architectures for credit scoring tasks,
specifically designed for UCI Credit Card (Taiwan) and Xinwang Credit datasets.

Design principles based on:
1. "Deep Learning for Credit Scoring" (He et al., 2019)
2. "Credit Risk Prediction with Deep Learning" (Barboza et al., 2017)
3. "Attention Mechanisms for Tabular Data" (Arik & Pfister, 2021)

Key considerations for credit scoring:
- Tabular data requires different architecture than images
- Class imbalance handling through architecture design
- Feature interaction modeling
- Interpretability through attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ====================================================================================================================
# Auxiliary Modules (ResidualBlock and FeatureAttention)
# ====================================================================================================================

class ResidualBlock(nn.Module):
    """
    Residual Block for tabular data.
    
    Structure:
        Input -> FC -> BatchNorm -> ReLU -> Dropout -> FC -> BatchNorm -> (+Input) -> ReLU -> Output
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        dropout: Dropout probability
    """
    def __init__(self, in_features, out_features, dropout=0.3):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features)
        )
        
        # Shortcut connection if dimensions don't match
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
        
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        identity = self.shortcut(x)
        out = self.block(x)
        out = out + identity
        out = self.relu(out)
        return out


class FeatureAttention(nn.Module):
    """
    Feature Attention Mechanism for tabular data.
    
    Learns importance weights for each input feature, allowing the model
    to focus on more relevant features for credit scoring.
    
    Based on "Squeeze-and-Excitation Networks" (Hu et al., 2018)
    
    Args:
        input_dim: Number of input features
        reduction_ratio: Reduction ratio for attention bottleneck (default: 4)
    """
    def __init__(self, input_dim, reduction_ratio=4):
        super(FeatureAttention, self).__init__()
        
        reduced_dim = max(input_dim // reduction_ratio, 1)
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, reduced_dim),
            nn.ReLU(),
            nn.Linear(reduced_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Apply feature attention.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Attention-weighted tensor of shape (batch_size, input_dim)
        """
        # Compute attention weights
        attention_weights = self.attention(x)
        
        # Apply attention weights
        return x * attention_weights


# ====================================================================================================================
# Credit Scoring Models
# ====================================================================================================================

class UciCreditNet(nn.Module):
    """
    Neural Network for UCI Credit Card Dataset (23 features, 2 classes)
    
    Architecture: Lightweight MLP optimized for federated learning
    - Input: 23 features (after removing ID column)
    - Output: 2 classes (default=0, default=1)
    - Label distribution: ~4:1 (imbalanced)
    
    Optimized for FL (每客户端~3000样本):
    - 超轻量级架构避免客户端过拟合
    - Strong regularization with dropout=0.4
    
    Network Structure:
    Input(23) -> FC(32) -> FC(16) -> Output(2)
    Total parameters: ~1,200 (优化后)
    """
    
    def __init__(self, input_dim=23, hidden_dims=[32, 16], num_classes=2, dropout=0.4):
        super(UciCreditNet, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Simple architecture for FL
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dims[1], num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Xavier/He initialization for better convergence.
        Critical for federated learning with limited local data.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 23)
        
        Returns:
            Output logits of shape (batch_size, 2)
        """
        # Handle 4D input (batch, channels, height, width) - flatten to 2D
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        
        # Forward pass
        x = self.layers(x)
        out = self.fc(x)
        
        return out


class XinwangCreditNet(nn.Module):
    """
    Neural Network for Xinwang Credit Dataset (38 features after toad filtering, 2 classes)
    
    Architecture: Ultra-lightweight MLP with Feature Attention
    - Input: 38 features (after toad feature selection from original 100)
    - Output: 2 classes (good credit, bad credit)
    - Label distribution: ~9:1 (imbalanced)
    
    Optimized for FL (每客户端~1800样本):
    - 超轻量级架构: 总参数~2K
    - Feature attention保留可解释性
    - Strong regularization
    
    Network Structure:
    Input(38) -> Feature Attention -> FC layers -> Output(2)
    Total parameters: ~2,000+ (depends on hidden_dims)
    """
    
    def __init__(self, input_dim=38, hidden_dims=[128, 64, 32], num_classes=2, dropout=0.3):
        super(XinwangCreditNet, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Feature attention mechanism (lightweight)
        self.feature_attention = FeatureAttention(input_dim, reduction_ratio=8)
        
        # Build hidden layers dynamically based on hidden_dims
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            # Apply dropout with decreasing rate for deeper layers
            dropout_rate = dropout if i == 0 else dropout * 0.5
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.hidden = nn.Sequential(*layers)
        
        # Output layer
        self.fc = nn.Linear(hidden_dims[-1], num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Xavier/He initialization with proper BatchNorm initialization.
        
        Critical for preventing NaN in eval mode before training:
        - BatchNorm running_mean initialized to 0 (default is correct)
        - BatchNorm running_var initialized to 1 (default is correct)
        - But we need to ensure momentum is set properly
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # Ensure running stats are properly initialized
                nn.init.constant_(m.running_mean, 0)
                nn.init.constant_(m.running_var, 1)
                m.momentum = 0.1  # Default momentum
                m.eps = 1e-5  # Prevent division by zero
    
    def forward(self, x):
        """
        Forward pass with feature attention.
        
        Args:
            x: Input tensor of shape (batch_size, 38)
        
        Returns:
            Output logits of shape (batch_size, 2)
        """
        # Handle 4D input - flatten to 2D
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        
        # Apply feature attention
        x_attended = self.feature_attention(x)
        
        # Forward pass
        hidden = self.hidden(x_attended)
        out = self.fc(hidden)
        
        return out


class GiveMeSomeCreditNet(nn.Module):
    """
    Neural Network for GiveMeSomeCredit Dataset (10 features, 2 classes)
    
    Architecture: Lightweight MLP
    - Input: 10 features (age, debt ratio, income, credit lines, etc.)
    - Output: 2 classes (normal=0, serious_delinquency=1)
    - Default rate: ~6.68%
    
    Optimized for large dataset (150K samples):
    - Simple architecture to prevent overfitting
    - Strong regularization
    
    Network Structure:
    Input(10) -> FC(32) -> FC(16) -> Output(2)
    Total parameters: ~1,000 (reduced from 20K)
    """
    
    def __init__(self, input_dim=10, num_classes=2, dropout=0.3):
        super(GiveMeSomeCreditNet, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Simple architecture
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Output layer
        self.fc = nn.Linear(16, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        
        x = self.layers(x)
        out = self.fc(x)
        
        return out


class EuropeCreditCardFraudNet(nn.Module):
    """
    Neural Network for European Credit Card Fraud Detection (30 features, 2 classes)
    
    Architecture: Deep MLP for Extreme Imbalance
    - Input: 30 features (Time + V1-V28[PCA] + Amount)
    - Output: 2 classes (normal=0, fraud=1)
    - Fraud rate: 0.172% (extreme imbalance)
    
    Design for extreme imbalance:
    - Deeper network to learn complex fraud patterns
    - Strong regularization (dropout)
    - Batch normalization for stable training
    
    Network Structure:
    Input(30) -> FC(128) -> FC(64) -> FC(64) -> FC(32) -> Output(2)
    """
    
    def __init__(self, input_dim=30, num_classes=2, dropout=0.4):
        super(EuropeCreditCardFraudNet, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Deep architecture for fraud detection
        self.layers = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 3
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 4
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout/2)
        )
        
        # Output layer
        self.fc = nn.Linear(32, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        
        x = self.layers(x)
        out = self.fc(x)
        
        return out


def get_credit_model(dataset_name, input_dim=None):
    """
    Factory function to get appropriate credit scoring model.
    
    Args:
        dataset_name: Dataset name ('Uci', 'Xinwang', 'GiveMeSomeCredit', 'EuropeCreditCardFraud')
        input_dim: Override default input dimension (from config.json)
    
    Returns:
        Neural network model instance
    
    Supported datasets:
        - GiveMeSomeCredit: 10 features
        - Uci: 23 features
        - Xinwang: 38 features (after toad filtering)
        - EuropeCreditCardFraud: 30 features (PCA fraud detection)
    
    Example:
        # Use default dimensions
        model = get_credit_model('Uci')
        
        # Override input dimension from config
        model = get_credit_model('Xinwang', input_dim=38)
    """
    dataset_name_lower = dataset_name.lower()
    
    # GiveMeSomeCredit (10 features)
    if 'givemesomecredit' in dataset_name_lower:
        dim = input_dim if input_dim is not None else 10
        return GiveMeSomeCreditNet(input_dim=dim)
    
    # UCI Credit Card (23 features)
    elif 'uci' in dataset_name_lower:
        dim = input_dim if input_dim is not None else 23
        return UciCreditNet(input_dim=dim)
    
    # Xinwang (38 features after toad filtering)
    elif 'xinwang' in dataset_name_lower:
        dim = input_dim if input_dim is not None else 38
        return XinwangCreditNet(input_dim=dim)
    
    # European Credit Card Fraud Detection (30 features, PCA)
    elif 'europe' in dataset_name_lower or 'fraud' in dataset_name_lower:
        dim = input_dim if input_dim is not None else 30
        return EuropeCreditCardFraudNet(input_dim=dim)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: 'Uci', 'Xinwang', 'GiveMeSomeCredit', 'EuropeCreditCardFraud'.")


# ====================================================================================================================
# Model information
# ====================================================================================================================

def print_model_info(model, dataset_name):
    """Print model architecture and parameter count."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print(f"Model: {model.__class__.__name__} for {dataset_name} dataset")
    print(f"{'='*60}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")
    print(f"{'='*60}\n")
