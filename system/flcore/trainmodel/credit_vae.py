"""
Variational Autoencoder (VAE) for Credit Scoring Data

Designed for generating synthetic tabular data in federated learning scenarios.
Supports both UCI Credit Card (23 features) and Xinwang Credit (100 features) datasets.

Key Features:
- Reparameterization trick for stable training
- Reconstruction + KL divergence losses
- Specialized for tabular financial data

References:
- Kingma & Welling (2014). "Auto-Encoding Variational Bayes". ICLR 2014.
- Xu et al. (2019). "Modeling Tabular data using Conditional GAN". NeurIPS 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CreditVAE(nn.Module):
    """
    Variational Autoencoder for Credit Scoring Tabular Data
    
    Supports two configurations:
    - UCI Credit Card: input_dim=23, latent_dim=16
    - Xinwang Credit: input_dim=100, latent_dim=32
    
    Architecture:
    Encoder: input → hidden layers → (mu, logvar)
    Decoder: latent → hidden layers → reconstructed input
    
    Loss = Reconstruction Loss + β × KL Divergence
    """
    
    def __init__(self, input_dim, latent_dim=16, hidden_dims=None, beta=1.0):
        """
        Args:
            input_dim (int): Number of input features (23 for UCI, 100 for Xinwang)
            latent_dim (int): Dimension of latent space
            hidden_dims (list): Hidden layer dimensions. If None, auto-configured
            beta (float): Weight for KL divergence term (β-VAE)
        """
        super(CreditVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Auto-configure hidden dimensions based on input size
        # 优化：增加VAE容量以提高生成质量
        if hidden_dims is None:
            if input_dim <= 30:  # UCI-like small datasets
                hidden_dims = [128, 64, 32]  # 原[64, 32] → 增加容量
            else:  # Xinwang-like larger datasets
                hidden_dims = [256, 128, 64]  # 原[128, 64] → 增加容量
        
        self.hidden_dims = hidden_dims
        
        # Build Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(0.2))
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space projection (mu and logvar)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Build Decoder (reverse of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(0.2))
            prev_dim = h_dim
        
        # Final reconstruction layer (no activation for continuous features)
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Xavier initialization for better gradient flow"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def encode(self, x):
        """
        Encode input to latent distribution parameters
        
        Args:
            x (torch.Tensor): Input features [batch_size, input_dim]
        
        Returns:
            mu (torch.Tensor): Mean of latent distribution [batch_size, latent_dim]
            logvar (torch.Tensor): Log variance [batch_size, latent_dim]
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * epsilon
        
        Args:
            mu (torch.Tensor): Mean [batch_size, latent_dim]
            logvar (torch.Tensor): Log variance [batch_size, latent_dim]
        
        Returns:
            z (torch.Tensor): Sampled latent code [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        Decode latent code to reconstructed input
        
        Args:
            z (torch.Tensor): Latent code [batch_size, latent_dim]
        
        Returns:
            recon_x (torch.Tensor): Reconstructed input [batch_size, input_dim]
        """
        # 确保z的类型与decoder权重一致（修复dtype mismatch）
        model_dtype = next(self.decoder.parameters()).dtype
        if z.dtype != model_dtype:
            z = z.to(model_dtype)
        return self.decoder(z)
    
    def forward(self, x):
        """
        Full forward pass: encode → reparameterize → decode
        
        Args:
            x (torch.Tensor): Input features [batch_size, input_dim]
        
        Returns:
            recon_x (torch.Tensor): Reconstructed input
            mu (torch.Tensor): Latent mean
            logvar (torch.Tensor): Latent log variance
        """
        # 确保输入类型与模型权重一致（解决dtype mismatch问题）
        model_dtype = next(self.encoder.parameters()).dtype
        if x.dtype != model_dtype:
            x = x.to(model_dtype)
        
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        """
        VAE loss = Reconstruction Loss + β × KL Divergence
        
        Reconstruction Loss: MSE (for continuous tabular data)
        KL Divergence: KL(N(mu, sigma) || N(0, 1))
        
        Args:
            recon_x (torch.Tensor): Reconstructed input
            x (torch.Tensor): Original input
            mu (torch.Tensor): Latent mean
            logvar (torch.Tensor): Latent log variance
        
        Returns:
            loss (torch.Tensor): Total VAE loss
            recon_loss (torch.Tensor): Reconstruction component
            kl_loss (torch.Tensor): KL divergence component
        """
        # Reconstruction loss (MSE for tabular data)
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss with β weighting
        loss = recon_loss + self.beta * kl_loss
        
        return loss, recon_loss, kl_loss
    
    def sample(self, num_samples, device='cuda'):
        """
        Generate synthetic samples from the learned distribution
        
        Args:
            num_samples (int): Number of samples to generate
            device (str): Device to generate samples on
        
        Returns:
            samples (torch.Tensor): Generated samples [num_samples, input_dim]
        """
        self.eval()
        with torch.no_grad():
            # Sample from standard normal distribution
            z = torch.randn(num_samples, self.latent_dim).to(device)
            # Decode to feature space
            samples = self.decode(z)
        return samples
    
    def reconstruct(self, x):
        """
        Reconstruct input (for quality assessment)
        
        Args:
            x (torch.Tensor): Input features
        
        Returns:
            recon_x (torch.Tensor): Reconstructed input
        """
        self.eval()
        with torch.no_grad():
            recon_x, _, _ = self.forward(x)
        return recon_x


class ConditionalCreditVAE(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) for Credit Scoring Data
    
    Key Improvement: 添加类别条件，实现可控生成
    - 解决类别不平衡问题（可指定生成少数类样本）
    - 提升生成质量（类别信息引导生成）
    - 适合联邦学习异质性场景
    
    Architecture:
    Encoder: (input + class_embedding) → hidden → (mu, logvar)
    Decoder: (latent + class_embedding) → hidden → reconstructed input
    
    Usage:
        cvae.encode(x, y)  # 编码时加入类别条件
        cvae.decode(z, y)  # 解码时指定目标类别
    """
    
    def __init__(self, input_dim, num_classes, latent_dim=16, hidden_dims=None, 
                 beta=1.0, class_embedding_dim=10):
        """
        Args:
            input_dim (int): 输入特征维度 (23 for UCI, 100 for Xinwang)
            num_classes (int): 类别数量 (通常为2，二分类信用评分)
            latent_dim (int): 潜在空间维度
            hidden_dims (list): 隐藏层维度
            beta (float): KL损失权重（β-VAE）
            class_embedding_dim (int): 类别嵌入维度
        """
        super(ConditionalCreditVAE, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.beta = beta
        self.class_embedding_dim = class_embedding_dim
        
        # Auto-configure hidden dimensions
        if hidden_dims is None:
            if input_dim <= 30:
                hidden_dims = [128, 64, 32]  # UCI
            else:
                hidden_dims = [256, 128, 64]  # Xinwang
        self.hidden_dims = hidden_dims
        
        # === 类别嵌入层 ===
        self.class_embedding = nn.Embedding(num_classes, class_embedding_dim)
        
        # === Build Encoder (input + class_embedding) ===
        encoder_layers = []
        prev_dim = input_dim + class_embedding_dim  # 拼接输入和类别嵌入
        
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(0.2))
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space projection
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # === Build Decoder (latent + class_embedding) ===
        decoder_layers = []
        prev_dim = latent_dim + class_embedding_dim  # 拼接潜在变量和类别嵌入
        
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(0.2))
            prev_dim = h_dim
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def encode(self, x, y):
        """
        条件编码：输入特征 + 类别标签 → 潜在分布参数
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            y: 类别标签 [batch_size] (整数)
        
        Returns:
            mu, logvar: 潜在分布参数
        """
        # 类别嵌入
        class_emb = self.class_embedding(y)  # [batch_size, class_embedding_dim]
        
        # 拼接输入和类别嵌入
        x_cond = torch.cat([x, class_emb], dim=1)  # [batch_size, input_dim + emb_dim]
        
        # 编码
        h = self.encoder(x_cond)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, y):
        """
        条件解码：潜在变量 + 类别标签 → 重构特征
        
        Args:
            z: 潜在变量 [batch_size, latent_dim]
            y: 类别标签 [batch_size] (整数)
        
        Returns:
            recon_x: 重构特征 [batch_size, input_dim]
        """
        # 类别嵌入
        class_emb = self.class_embedding(y)  # [batch_size, class_embedding_dim]
        
        # 拼接潜在变量和类别嵌入
        z_cond = torch.cat([z, class_emb], dim=1)  # [batch_size, latent_dim + emb_dim]
        
        # 确保dtype一致
        model_dtype = next(self.decoder.parameters()).dtype
        if z_cond.dtype != model_dtype:
            z_cond = z_cond.to(model_dtype)
        
        # 解码
        return self.decoder(z_cond)
    
    def forward(self, x, y):
        """
        完整前向传播：编码 → 重参数化 → 解码
        
        Args:
            x: 输入特征
            y: 类别标签
        
        Returns:
            recon_x, mu, logvar
        """
        model_dtype = next(self.encoder.parameters()).dtype
        if x.dtype != model_dtype:
            x = x.to(model_dtype)
        
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, y)
        return recon_x, mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        """计算VAE损失（与标准VAE相同）"""
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.beta * kl_loss
        return loss, recon_loss, kl_loss
    
    def sample(self, num_samples, class_id, device='cuda'):
        """
        生成指定类别的样本（CVAE核心功能）
        
        Args:
            num_samples: 生成样本数量
            class_id: 目标类别（0或1）
            device: 设备
        
        Returns:
            samples: 生成的样本 [num_samples, input_dim]
        """
        self.eval()
        with torch.no_grad():
            # 从标准正态分布采样
            z = torch.randn(num_samples, self.latent_dim).to(device)
            
            # 创建类别标签张量
            y = torch.full((num_samples,), class_id, dtype=torch.long).to(device)
            
            # 条件解码
            samples = self.decode(z, y)
        return samples


class CreditVAEWithClassifier(nn.Module):
    """
    Joint VAE + Classifier for FedVPS Phase 1
    
    Combines VAE for virtual data generation with a classifier for 
    supervised learning. Used in Phase 1 of FedVPS algorithm.
    
    Training flow:
    1. Real data → VAE → Virtual data
    2. Virtual data → Classifier → Classification loss
    3. Virtual features → Prototype loss
    4. Total loss = Classification + Reconstruction + KL + Prototype
    """
    
    def __init__(self, vae, classifier):
        """
        Args:
            vae (CreditVAE): Pre-initialized VAE
            classifier (nn.Module): Classification model from credit.py
        """
        super(CreditVAEWithClassifier, self).__init__()
        
        self.vae = vae
        self.classifier = classifier
        
        # Loss weights (will be set from args)
        self.lambda_cls = 1.0
        self.lambda_recon = 1.0
        self.lambda_kl = 0.01
        self.lambda_proto = 0.1
    
    def set_loss_weights(self, lambda_cls=1.0, lambda_recon=1.0, 
                        lambda_kl=0.01, lambda_proto=0.1):
        """Set loss combination weights"""
        self.lambda_cls = lambda_cls
        self.lambda_recon = lambda_recon
        self.lambda_kl = lambda_kl
        self.lambda_proto = lambda_proto
    
    def forward(self, x):
        """
        Forward pass for training
        
        Args:
            x (torch.Tensor): Input features [batch_size, input_dim]
        
        Returns:
            virtual_x (torch.Tensor): Generated virtual features
            cls_output (torch.Tensor): Classification logits
            mu, logvar: VAE latent parameters
        """
        # VAE forward: generate virtual data
        virtual_x, mu, logvar = self.vae(x)
        
        # Classify virtual data
        cls_output = self.classifier(virtual_x)
        
        return virtual_x, cls_output, mu, logvar
    
    def compute_loss(self, x, y, prototypes=None):
        """
        Compute joint loss for Phase 1 training
        
        Args:
            x (torch.Tensor): Input features
            y (torch.Tensor): Labels
            prototypes (dict): Class prototypes {class_id: prototype_tensor}
        
        Returns:
            total_loss, cls_loss, recon_loss, kl_loss, proto_loss
        """
        # Forward pass
        virtual_x, cls_output, mu, logvar = self.forward(x)
        
        # 1. Classification loss
        cls_loss = F.cross_entropy(cls_output, y)
        
        # 2. VAE reconstruction loss
        recon_loss = F.mse_loss(virtual_x, x, reduction='mean')
        
        # 3. KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 4. Prototype loss (if prototypes provided)
        proto_loss = torch.tensor(0.0, device=x.device)
        if prototypes is not None and len(prototypes) > 0:
            # Virtual features should be close to class prototypes
            for i, label in enumerate(y):
                class_id = label.item()
                if class_id in prototypes:
                    # Distance between virtual feature and prototype
                    proto_loss += F.mse_loss(virtual_x[i], prototypes[class_id])
            proto_loss = proto_loss / len(y)
        
        # Total weighted loss
        total_loss = (self.lambda_cls * cls_loss + 
                     self.lambda_recon * recon_loss + 
                     self.lambda_kl * kl_loss + 
                     self.lambda_proto * proto_loss)
        
        return total_loss, cls_loss, recon_loss, kl_loss, proto_loss
    
    def generate_virtual_data(self, num_samples, device='cuda'):
        """Generate virtual data for sharing"""
        return self.vae.sample(num_samples, device)


def create_credit_vae(input_dim, latent_dim=None, dataset_name='UCI', 
                     num_classes=2, use_conditional=True):
    """
    Factory function to create appropriate VAE for dataset
    
    Args:
        input_dim (int): Number of features
        latent_dim (int): Latent dimension (auto-configured if None)
        dataset_name (str): 'UCI' or 'Xinwang'
        num_classes (int): 类别数量（默认2，二分类）
        use_conditional (bool): 是否使用条件VAE（推荐True）
    
    Returns:
        vae (CreditVAE or ConditionalCreditVAE): Configured VAE model
    """
    if latent_dim is None:
        # Auto-configure latent dimension
        latent_dim = 16 if input_dim <= 30 else 32
    
    # 🔥 优先使用CVAE（条件VAE）
    if use_conditional:
        if dataset_name.lower() == 'uci' or input_dim <= 30:
            vae = ConditionalCreditVAE(
                input_dim=input_dim,
                num_classes=num_classes,
                latent_dim=latent_dim,
                hidden_dims=[128, 64, 32],  # 增强容量
                beta=1.0,  # 初始β值（会被动态调度器覆盖）
                class_embedding_dim=10
            )
        else:
            vae = ConditionalCreditVAE(
                input_dim=input_dim,
                num_classes=num_classes,
                latent_dim=latent_dim,
                hidden_dims=[256, 128, 64],  # Xinwang增强容量
                beta=1.0,
                class_embedding_dim=10
            )
    else:
        # 保留标准VAE（向后兼容）
        if dataset_name.lower() == 'uci' or input_dim <= 30:
            vae = CreditVAE(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dims=[64, 32],
                beta=1.0
            )
        else:
            vae = CreditVAE(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dims=[128, 64],
                beta=1.0
            )
    
    return vae
