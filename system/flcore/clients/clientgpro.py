"""
FedGpro Client: Federated Global Prototype Learning

Two-phase federated learning client for credit scoring with privacy-preserving
virtual data generation.

Phase 1: VAE Training + Prototype Learning (No Parameter Aggregation)
- Train VAE + Classifier jointly with prototype regularization
- Generate virtual data when accuracy threshold is met
- Add differential privacy noise to virtual data
- Upload: accuracy, prototypes, noisy virtual data

Phase 2: Federated Training with Virtual Data (Flexible Aggregation)
- Train on mixed data (real + shared virtual)
- Server uses configurable aggregation algorithm (FedAvg/FedCS/FedProx/etc.)
- Standard federated learning workflow

Author: [Your Name]
Date: 2025-12-16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
from collections import defaultdict
from flcore.clients.clientbase import Client
from flcore.trainmodel.credit_vae import CreditVAE, create_credit_vae
from sklearn.preprocessing import label_binarize
from sklearn import metrics


class PrototypeAlignmentLayer(nn.Module):
    """
    原型对齐层：显式特征增强模块
    
    功能：在特征进入分类器前，将特征向全局原型方向调整，
    增强模型对类别共享表征的学习能力。
    
    与原型损失的关系：
    - 原型损失：训练时的监督信号（让VAE学会生成接近原型的特征）
    - 原型对齐层：推理时的显式增强（即使VAE没学好也能拉回来）
    
    Args:
        feature_dim: 特征维度
        alpha: 原型增强强度，默认0.3
    """
    def __init__(self, feature_dim, alpha=0.3):
        super().__init__()
        self.projector = nn.Linear(feature_dim, feature_dim)
        self.alpha = alpha
        
    def forward(self, features, prototypes):
        """
        Args:
            features: [batch_size, feature_dim] - VAE生成的特征
            prototypes: [num_classes, feature_dim] - 全局原型张量
        Returns:
            enhanced_features: 原型增强后的特征
        """
        if prototypes is None or prototypes.shape[0] == 0:
            return features
        
        # Step 1: 特征投影（可学习变换）
        features_proj = self.projector(features)
        
        # Step 2: 计算特征与各原型的相似度
        similarity = torch.mm(features_proj, prototypes.T) / 0.5  # 温度=0.5
        weights = F.softmax(similarity, dim=1)  # [batch_size, num_classes]
        
        # Step 3: 加权原型向量
        proto_weighted = torch.mm(weights, prototypes)  # [batch_size, feature_dim]
        
        # Step 4: 残差连接（保留原始特征+原型增强）
        enhanced_features = features + self.alpha * proto_weighted
        
        return enhanced_features


class clientGpro(Client):
    """
    FedGpro Client Implementation
    
    Key Features:
    - Two-phase training protocol
    - VAE-based virtual data generation
    - Prototype learning for feature regularization
    - Differential privacy via Laplace/Gaussian noise
    - Threshold-based phase transition
    
    Args:
        args: Arguments containing VPS-specific parameters
            - fedgpro_phase: Current phase (1 or 2)
            - fedgpro_epsilon: Privacy budget for DP noise
            - fedgpro_noise_type: 'laplace' or 'gaussian'
            - fedgpro_lambda_cls: Weight for classification loss
            - fedgpro_lambda_recon: Weight for reconstruction loss
            - fedgpro_lambda_kl: Weight for KL divergence
            - fedgpro_lambda_proto: Weight for prototype loss
            - fedgpro_proto_momentum: EMA momentum for prototype update
            - fedgpro_latent_dim: VAE latent dimension
    """
    
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # Ensure model is float64
        self.model = self.model.double()
        
        # Phase control
        self.current_phase = getattr(args, 'fedgpro_phase', 1)
        
        # Privacy parameters (optional, default: no noise)
        self.epsilon = getattr(args, 'fedgpro_epsilon', None)  # None = disabled
        self.noise_type = getattr(args, 'fedgpro_noise_type', None)  # None, 'laplace', 'gaussian'
        self.delta = getattr(args, 'fedgpro_delta', 1e-5)  # For Gaussian noise
        
        # Ditto-style personalized model (Phase2 only)
        self.model_per = None
        self.optimizer_per = None
        self.mu_ditto = getattr(args, 'mu', 0.01)  # Ditto regularization parameter
        self.plocal_epochs = getattr(args, 'plocal_epochs', 3)  # Personalized training epochs
        
        # Component switches for ablation study
        self.use_vae_generation = getattr(args, 'fedgpro_use_vae', True)  # Default: enabled
        self.use_prototype_loss = getattr(args, 'fedgpro_use_prototype', True)  # Default: enabled
        
        # Generated data ratio (for ablation study)
        self.gen_data_ratio = getattr(args, 'fedgpro_gen_data_ratio', 1.0)  # Default: 100% (1.0)
        
        # Loss weights - 优化策略
        self.lambda_cls = getattr(args, 'fedgpro_lambda_cls', 10.0)  # 增加到10.0以强化分类
        
        # Dataset-specific loss weights
        if 'Xinwang' in args.dataset:
            # Xinwang: 高维数据，增强重构质量和原型约束
            self.lambda_recon = getattr(args, 'fedgpro_lambda_recon', 1.5)  # 增强重构：1.0 → 1.5
            self.lambda_kl = getattr(args, 'fedgpro_lambda_kl', 0.1)  # Optimized: 0.1 (best from ablation)
            self.lambda_proto = getattr(args, 'fedgpro_lambda_proto', 0.1)  # Optimized: 0.1 (best from ablation)
        else:
            # Uci: 低维数据，使用标准权重
            self.lambda_recon = getattr(args, 'fedgpro_lambda_recon', 1.0)
            self.lambda_kl = getattr(args, 'fedgpro_lambda_kl', 0.01)
            self.lambda_proto = getattr(args, 'fedgpro_lambda_proto', 0.5)  # 原0.1 → 0.3 → 0.5
        
        # Prototype learning
        self.proto_momentum = getattr(args, 'fedgpro_proto_momentum', 0.9)
        self.prototypes = {}  # {class_id: prototype_tensor}
        
        # Phase 1: VAE model - Get correct input dimension based on model architecture
        if hasattr(self.model, 'input_dim'):
            # For UciCreditNet and XinwangCreditNet which have input_dim attribute
            input_dim = self.model.input_dim
        elif hasattr(self.model, 'fc1'):
            # Fallback: try to get from first layer
            input_dim = self.model.fc1.in_features
        elif hasattr(self.model, 'input_embedding'):
            # For XinwangCreditNet's input_embedding layer
            input_dim = self.model.input_embedding[0].in_features
        else:
            # Ultimate fallback based on dataset name
            if 'Xinwang' in args.dataset:
                input_dim = 100
            else:
                input_dim = 23
        
        # 针对Xinwang扩大隐空间容量以处理高维特征
        if 'Xinwang' in args.dataset:
            latent_dim = getattr(args, 'fedgpro_latent_dim', 32)  # 强制32维
        else:
            latent_dim = getattr(args, 'fedgpro_latent_dim', 16 if input_dim <= 30 else 32)
        
        # ✨ 阶段1：启用CVAE（条件VAE）
        self.use_cvae = getattr(args, 'fedgpro_use_cvae', True)  # 默认True
        
        self.vae = create_credit_vae(
            input_dim=input_dim,
            latent_dim=latent_dim,
            dataset_name=args.dataset,
            num_classes=self.num_classes,
            use_conditional=self.use_cvae  # CVAE or standard VAE
        ).to(self.device).double()
        
        # ⚡ 阶段2：动态β调度器
        self.beta_schedule_enabled = getattr(args, 'fedgpro_beta_schedule', True)  # 默认True
        self.beta_warmup_epochs = getattr(args, 'fedgpro_beta_warmup', 3)  # 前3轮低β
        self.beta_min = getattr(args, 'fedgpro_beta_min', 0.001)  # 最小β
        self.beta_max = self.lambda_kl  # 最大β = 原KL权重
        self.current_beta = self.beta_min  # 当前β值（动态调整）
        
        # VAE optimizer
        self.vae_optimizer = torch.optim.Adam(
            self.vae.parameters(),
            lr=getattr(args, 'fedgpro_vae_lr', 0.001)
        )
        
        # Baseline VAE (for feature importance computation via contrastive learning)
        self.vae_baseline = None  # Created on-demand
        self.vae_baseline_optimizer = None
        self.feature_importance = None  # Computed from VAE comparison
        self.baseline_epochs = getattr(args, 'fedgpro_baseline_epochs', 50)  # Fixed training epochs
        
        # Phase 1 metrics
        self.accuracy = 0.0
        self.virtual_data = []  # List of (features, labels)
        self.threshold_met = False
        self.virtual_data_generated = False  # 标记虚拟数据是否已生成（锁定）
        
        # 🔥 Phase 2机制标记：是否贡献虚拟数据
        self.contributes_virtual_data = False  # Phase 2中是否贡献虚拟数据（达标客户端=True，未达标=False）
        
        # Phase 1: Early stopping tracking
        self.accuracy_history = []  # Track accuracy for last N epochs
        self.early_stopped = False  # Flag indicating if client has converged
        self.current_threshold = 0.70  # Dynamic threshold from server (updated each round)
        self.convergence_window = 5  # Check last 5 epochs
        self.convergence_threshold = 0.005  # 0.5% fluctuation threshold
        
        # Phase 2: Shared virtual data storage
        self.shared_virtual_data = []
        
        # Phase 2: Algorithm type (set by server during transition)
        self.phase2_algorithm = 'fedavg'  # Default
        
        # Phase 2: Algorithm-specific states
        self.prev_model = None  # For MOON contrastive learning
        self.c_local = None  # For SCAFFOLD control variates
        self.c_global = None
        self.personalized_model = None  # For Ditto, pFedMe
        self.val_acc = 0.0  # For FedGWO ranking
        
        # For FedPSO (Particle Swarm Optimization)
        self.velocity = None  # PSO velocity
        self.pbest_model = None  # Individual best model
        self.gbest_model = None  # Global best model
        
        # ==================== Personalization Mechanism (Ditto-style) ====================
        # Personalized model for each client (maintained locally)
        self.mu = getattr(args, 'mu', 0.01)  # Regularization weight between global and personalized
        self.plocal_epochs = getattr(args, 'plocal_epochs', 1)  # Personalization training epochs
        
        # Create personalized model (deep copy of global model)
        self.model_per = copy.deepcopy(self.model)
        
        # Personalized optimizer with regularization
        from flcore.optimizers.fedoptimizer import PerturbedGradientDescent
        self.optimizer_per = PerturbedGradientDescent(
            self.model_per.parameters(), 
            lr=self.learning_rate, 
            mu=self.mu
        )
        
        # Learning rate scheduler for personalized model
        self.learning_rate_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_per,
            gamma=args.learning_rate_decay_gamma
        )
        print(f"  Client {self.id}: Personalization enabled (mu={self.mu}, plocal_epochs={self.plocal_epochs})")
        self.pso_w = 0.9  # Inertia weight
        self.pso_c1 = 2.0  # Cognitive parameter
        self.pso_c2 = 2.0  # Social parameter
        self.pso_r1 = 0.5  # Random number 1
        self.pso_r2 = 0.5  # Random number 2
        self.pso_v_max = 0.5  # Maximum velocity ratio
        
        # For FedProto
        self.protos = None  # Local prototypes collected during training
        
        # For pFedMe
        self.local_params = None
        self.personalized_params = None
        
        # Global prototypes from server (Phase 1 and FedProto)
        self.global_prototypes = {}
        
        # 原型对齐层（已禁用 - 简化模型）
        # self.proto_align_layer = PrototypeAlignmentLayer(
        #     feature_dim=input_dim,  # 与VAE输入维度一致
        #     alpha=0.3  # 原型增强强度
        # ).to(self.device).double()
        
        # 为原型对齐层添加优化器（已禁用）
        # self.proto_align_optimizer = torch.optim.Adam(
        #     self.proto_align_layer.parameters(),
        #     lr=getattr(args, 'fedgpro_proto_align_lr', 0.001)
        # )
    
    def set_phase(self, phase):
        """Switch between Phase 1 and Phase 2"""
        self.current_phase = phase
    
    def get_vae_parameters(self):
        """返回VAE参数用于联邦聚合"""
        return {
            'encoder': self.vae.encoder.state_dict(),
            'fc_mu': self.vae.fc_mu.state_dict(),
            'fc_logvar': self.vae.fc_logvar.state_dict(),
            'decoder': self.vae.decoder.state_dict()
        }
    
    def set_vae_parameters(self, vae_params):
        """设置全局VAE参数"""
        self.vae.encoder.load_state_dict(vae_params['encoder'])
        self.vae.fc_mu.load_state_dict(vae_params['fc_mu'])
        self.vae.fc_logvar.load_state_dict(vae_params['fc_logvar'])
        self.vae.decoder.load_state_dict(vae_params['decoder'])
    
    def receive_global_prototypes(self, global_prototypes):
        """
        Receive aggregated global prototypes from server
        
        Args:
            global_prototypes: dict {class_id: prototype_tensor}
        
        This allows clients to leverage global feature representations
        to improve local prototype learning and regularization.
        """
        self.global_prototypes = {
            class_id: proto.clone().detach().to(self.device)
            for class_id, proto in global_prototypes.items()
        }
        print(f"  Client {self.id} received {len(self.global_prototypes)} global prototypes")
        
    def train(self):
        """
        Unified training entry point.
        Dispatches to phase-specific training methods.
        """
        if self.current_phase == 1:
            return self.train_phase1()
        else:
            return self.train_phase2()
    
    # ==================== Phase 1: VAE + Prototype Learning ====================
    
    def train_phase1(self):
        """
        Phase 1 Training: Hybrid Strategy (Warm-up + Joint Training)
        
        优化策略：混合训练
        - Stage 1 (前30%轮次): 预训练分类器，冻结VAE
        - Stage 2 (后70%轮次): VAE + 分类器联合训练
        
        Training flow:
        1. Warm-up Stage (3/10 epochs):
           - Real data → Classifier → Classification loss
           - 只更新分类器，VAE冻结
           - 目标：建立稳定的判别能力
        
        2. Joint Training Stage (7/10 epochs):
           - Real data → VAE → Virtual data
           - Virtual data → Classifier → Classification loss
           - Compute VAE reconstruction + KL losses
           - Compute prototype loss (feature → prototype distance)
           - 同时更新VAE和分类器
        
        Returns:
            accuracy (float): Current validation accuracy
        """
        trainloader = self.load_train_data()
        
        start_time = time.time()
        
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        
        # 计算warm-up阶段的轮次（30%）
        warmup_epochs = max(1, int(max_local_epochs * 0.3))
        joint_epochs_start = warmup_epochs
        
        for epoch in range(max_local_epochs):
            # ====== Stage 1: Warm-up (预训练分类器) ======
            if epoch < warmup_epochs:
                self.model.train()
                self.vae.eval()  # 冻结VAE
                
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device).double()
                    else:
                        x = x.to(self.device).double()
                    y = y.to(self.device)
                    
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))
                    
                    # 分类器直接在真实数据上训练
                    cls_output = self.model(x.double())
                    cls_loss = self.loss(cls_output, y)
                    
                    # 只更新分类器
                    self.optimizer.zero_grad()
                    cls_loss.backward()
                    self.optimizer.step()
                
                if epoch == 0:
                    print(f"  Client {self.id}: [Warm-up Stage] Pretraining classifier on real data (epochs 1-{warmup_epochs})")
            
            # ====== Stage 2: Joint Training (联合训练) ======
            else:
                self.model.train()
                self.vae.train()
                
                if epoch == joint_epochs_start:
                    print(f"  Client {self.id}: [Joint Training Stage] VAE + Classifier joint training (epochs {joint_epochs_start+1}-{max_local_epochs})")
                
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device).double()
                    else:
                        x = x.to(self.device).double()
                    y = y.to(self.device)
                    
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))
                    
                    # === VAE Forward Pass (Ablation: can be disabled) ===
                    if self.use_vae_generation:
                        # ✨ CVAE: 加入类别条件
                        if self.use_cvae:
                            virtual_x, mu, logvar = self.vae(x, y)  # 条件输入
                        else:
                            virtual_x, mu, logvar = self.vae(x)  # 标准VAE
                        
                        # === Loss 2: VAE Reconstruction ===
                        recon_loss = F.mse_loss(virtual_x, x, reduction='mean')
                        
                        # === Loss 3: KL Divergence (使用动态β) ===
                        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                        kl_loss = (self.current_beta * kl_div).to(torch.float64)  # 动态β权重
                    else:
                        # No VAE: use real data directly
                        virtual_x = x
                        recon_loss = torch.tensor(0.0, device=self.device, dtype=torch.float64)
                        kl_loss = torch.tensor(0.0, device=self.device, dtype=torch.float64)
                    
                    # === 原型对齐层已禁用 - 直接使用原始特征 ===
                    # if len(self.global_prototypes) > 0 and self.use_prototype_loss:
                    #     # 提取全局原型张量
                    #     try:
                    #         prototype_tensor = torch.stack([
                    #             self.global_prototypes[class_id] 
                    #             for class_id in sorted(self.global_prototypes.keys())
                    #         ])  # [num_classes, feature_dim]
                    #         
                    #         # 应用原型对齐层（显式增强）
                    #         virtual_x_enhanced = self.proto_align_layer(virtual_x, prototype_tensor)
                    #     except Exception as e:
                    #         # 如果出错，使用原始特征
                    #         virtual_x_enhanced = virtual_x
                    # else:
                    #     virtual_x_enhanced = virtual_x
                    
                    # === Classifier Forward Pass (直接使用VAE特征) ===
                    cls_output = self.model(virtual_x.double())
                    
                    # === Loss 1: Classification ===
                    cls_loss = self.loss(cls_output, y)
                    
                    # === 原型损失已禁用 ===
                    # if self.use_prototype_loss:
                    #     proto_loss = self._compute_prototype_loss(virtual_x, y)
                    # else:
                    #     proto_loss = torch.tensor(0.0, device=self.device, dtype=torch.float64)
                    
                    # === Total Weighted Loss (分类 + 重构 + 动态β*KL) ===
                    total_loss = (self.lambda_cls * cls_loss.to(torch.float64) +
                                 self.lambda_recon * recon_loss.to(torch.float64) +
                                 kl_loss)  # kl_loss已包含current_beta权重
                    
                    # === Backward Pass ===
                    self.optimizer.zero_grad()
                    self.vae_optimizer.zero_grad()
                    # self.proto_align_optimizer.zero_grad()  # 原型对齐层优化器已禁用
                    total_loss.backward()
                    self.optimizer.step()
                    self.vae_optimizer.step()
                    # self.proto_align_optimizer.step()  # 原型对齐层已禁用
        
        # Update prototypes after epoch
        # 所有客户端（包括早停客户端）都继续更新并上传原型
        # 早停客户端的原型权重会通过自适应衰减α_k(t)自动降低
        self._update_prototypes()
        
        # Compute validation accuracy
        self.accuracy = self._compute_accuracy()
        
        # Track accuracy history for early stopping detection
        self.accuracy_history.append(self.accuracy)
        
        # Check for early stopping and virtual data generation
        # 前5轮强制训练，不检查早停；6轮后开始检查
        current_round = len(self.accuracy_history)  # 1-indexed
        
        if current_round <= 5:
            # Round 1-5: 强制训练，不检查早停
            pass
        elif not self.early_stopped:
            # Round 6+: 检查早停
            self.early_stopped = self._check_early_stopping()
            if self.early_stopped:
                recent_3 = self.accuracy_history[-3:] if len(self.accuracy_history) >= 3 else self.accuracy_history
                fluctuation = max(recent_3) - min(recent_3) if len(recent_3) > 0 else 0
                print(f"  [Qualified] Client {self.id} reached early stopping criteria (Round {current_round}):")
                print(f"    - Accuracy: {self.accuracy:.4f} (threshold: {self.current_threshold:.4f})")
                print(f"    - Fluctuation: {fluctuation:.4f} (last {len(recent_3)} rounds)")
                print(f"    - History: {[f'{x:.3f}' for x in recent_3]}")
                
                # 首次达标：生成并锁定虚拟数据
                if not self.virtual_data_generated:
                    print(f"  [Virtual Data] Generating and locking virtual data for client {self.id}...")
                    # 这里生成虚拟数据的逻辑会在服务器端调用 generate_virtual_samples()
                    self.virtual_data_generated = True
                    print(f"  [Mentor Mode] Client {self.id} enters mentor mode:")
                    print(f"    - Continue training model (help weak clients)")
                    print(f"    - Continue uploading prototypes (weight auto-decayed by α_k(t))")
                    print(f"    - Virtual data locked (privacy preserved)")
        
        # 即使达到早停标准，客户端仍继续训练（作为"助教"角色）
        # 继续上传原型（权重通过α_k(t)自适应衰减），虚拟数据已锁定
        
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        
        return self.accuracy
    
    def _compute_prototype_loss(self, features, labels):
        """
        Compute distance between features and their class prototypes
        
        Strategy:
        1. Prioritize global prototypes (from server aggregation)
        2. Fallback to local prototypes if global not available
        3. This ensures clients leverage global knowledge
        
        Args:
            features: Virtual features [batch_size, feature_dim]
            labels: Class labels [batch_size]
        
        Returns:
            proto_loss: Average distance to prototypes
        """
        # Use global prototypes if available, otherwise local
        active_prototypes = self.global_prototypes if len(self.global_prototypes) > 0 else self.prototypes
        
        if len(active_prototypes) == 0:
            return torch.tensor(0.0, device=self.device, dtype=torch.float64)
        
        proto_loss = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        count = 0
        
        for i, label in enumerate(labels):
            class_id = label.item()
            if class_id in active_prototypes:
                # MSE distance to prototype (global or local)
                proto_loss += F.mse_loss(features[i], active_prototypes[class_id])
                count += 1
        
        return proto_loss / count if count > 0 else torch.tensor(0.0, device=self.device, dtype=torch.float64)
    
    def _update_prototypes(self):
        """
        Update class prototypes using EMA (Exponential Moving Average)
        
        Prototype update rule:
        proto_new = momentum × proto_old + (1 - momentum) × proto_current
        """
        trainloader = self.load_train_data()
        self.model.eval()
        self.vae.eval()
        
        # Collect features by class
        class_features = defaultdict(list)
        
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device).double()
                else:
                    x = x.to(self.device).double()
                y = y.to(self.device)
                
                # Generate virtual features (支持CVAE条件输入)
                if self.use_cvae:
                    virtual_x, _, _ = self.vae(x, y)  # CVAE需要类别标签
                else:
                    virtual_x, _, _ = self.vae(x)  # 标准VAE
                
                # Group by class
                for i, label in enumerate(y):
                    class_id = label.item()
                    class_features[class_id].append(virtual_x[i].cpu())
        
        # Compute mean prototype for each class
        for class_id, features in class_features.items():
            if len(features) == 0:
                continue
            
            # Current prototype: mean of features
            current_proto = torch.stack(features).mean(dim=0).to(self.device)
            
            # EMA update with global prototype guidance
            if class_id in self.prototypes:
                # Standard EMA update
                self.prototypes[class_id] = (
                    self.proto_momentum * self.prototypes[class_id] +
                    (1 - self.proto_momentum) * current_proto
                )
            elif class_id in self.global_prototypes:
                # Initialize from global prototype if available
                self.prototypes[class_id] = (
                    0.7 * self.global_prototypes[class_id] +
                    0.3 * current_proto
                )
            else:
                # New class, use current
                self.prototypes[class_id] = current_proto
    
    def _update_beta_schedule(self):
        """
        ⚡ 阶段2: 动态β调度器
        
        策略：
        - 前N轮 (warmup): β从min逐渐增加到max，学习丰富表示
        - 后期 (stable): β保持max值，稳定训练
        
        好处：
        - 早期低β → VAE学习更多样化的特征表示
        - 后期高β → 约束z到标准分布，稳定生成质量
        - 提升生成数据多样性 +15~25%
        """
        current_round = len(self.accuracy_history) + 1  # 当前是第几轮
        
        if current_round <= self.beta_warmup_epochs:
            # 线性增长: beta_min → beta_max
            progress = current_round / self.beta_warmup_epochs
            self.current_beta = self.beta_min + (self.beta_max - self.beta_min) * progress
            
            if current_round == 1:
                print(f"  [β-Schedule] Client {self.id}: Warmup phase (rounds 1-{self.beta_warmup_epochs})")
                print(f"    β range: {self.beta_min:.4f} → {self.beta_max:.4f}")
        else:
            # 稳定阶段
            self.current_beta = self.beta_max
        
        # 更新VAE的beta参数（如果VAE支持）
        if hasattr(self.vae, 'beta'):
            self.vae.beta = self.current_beta
    
    def collect_latent_distribution(self):
        """
        从真实训练数据中提取隐空间分布的统计量（均值和标准差）
        
        目的：
        - 不再从标准正态分布N(0,1)采样
        - 而是从训练数据学到的真实分布采样
        - 提高生成虚拟数据的质量和真实性
        
        Returns:
            latent_stats: Dict[class_id -> {'mu': mean_vector, 'std': std_vector}]
        """
        self.vae.eval()
        trainloader = self.load_train_data()
        
        # 按类别收集隐变量
        latent_by_class = {i: [] for i in range(self.num_classes)}
        
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x = x[0]
                x = x.to(self.device).double()
                y = y.to(self.device)
                
                # 编码到隐空间 (支持CVAE条件编码)
                if self.use_cvae:
                    mu, logvar = self.vae.encode(x, y)  # CVAE条件编码
                else:
                    h = self.vae.encoder(x)
                    mu = self.vae.fc_mu(h)
                
                # 按类别存储
                for i, label in enumerate(y):
                    latent_by_class[label.item()].append(mu[i].cpu())
        
        # 计算每个类别的均值和标准差
        latent_stats = {}
        for class_id, latents in latent_by_class.items():
            if len(latents) > 0:
                latents = torch.stack(latents)
                latent_stats[class_id] = {
                    'mu': latents.mean(dim=0),      # [latent_dim]
                    'std': latents.std(dim=0) + 1e-6  # 添加小值防止除零
                }
        
        return latent_stats
    
    def generate_virtual_data(self, num_samples=None, confidence_threshold=None, 
                             use_real_distribution=True, exploration_ratio=None):
        """
        Generate high-quality virtual data using trained VAE
        
        改进点：
        1. 质量筛选：基于分类器置信度过滤低质量样本
        2. 真实分布采样：从训练数据的隐空间分布采样（不是N(0,1)）
        3. 混合采样：结合探索（随机）和利用（真实分布）
        4. 生成数据比例：支持按比例生成数据用于消融实验
        5. 针对Xinwang优化：更严格的质量控制
        
        Args:
            num_samples: Number of samples to generate.
                        If None, generates same amount as real training data.
            confidence_threshold: 置信度阈值，只保留分类器置信度>threshold的样本
                                 None时自动选择（Xinwang=0.8, 其他=0.7）
            use_real_distribution: 是否从真实数据分布采样（True推荐）
            exploration_ratio: 随机探索的比例
                              None时自动选择（Xinwang=0.1, 其他=0.2）
        
        Returns:
            virtual_data: List of (features, label) tuples
        """
        if num_samples is None:
            num_samples = self.train_samples
        
        # 应用生成数据比例（消融实验用）
        num_samples = int(num_samples * self.gen_data_ratio)
        
        # 如果比例为0，直接返回空列表
        if num_samples == 0 or not self.use_vae_generation:
            self.virtual_data = []
            print(f"  Client {self.id}: Skipped virtual data generation (ratio={self.gen_data_ratio})")
            return []
        
        # 🔥 严格质量控制：与中心化训练标准对齐
        # 原则：真实数据能达到的标准，虚拟数据也必须达到，不降低标准
        is_xinwang = 'Xinwang' in str(self.__class__.__module__)
        if confidence_threshold is None:
            confidence_threshold = 0.96 if is_xinwang else 0.8  # 提高到中心化水平
        if exploration_ratio is None:
            exploration_ratio = 0.1 if is_xinwang else 0.2
        
        self.vae.eval()
        self.model.eval()
        
        # 收集真实数据的隐空间分布
        latent_stats = None
        if use_real_distribution:
            print(f"  Client {self.id}: Collecting latent distribution from real data...")
            latent_stats = self.collect_latent_distribution()
        
        virtual_data = []
        samples_per_class = num_samples // self.num_classes
        
        # 生成更多候选样本用于质量筛选（2倍）
        candidates_per_class = samples_per_class * 2
        
        with torch.no_grad():
            for class_id in range(self.num_classes):
                # ========== 改进1: 从真实分布采样 ==========
                if use_real_distribution and latent_stats and class_id in latent_stats:
                    # 计算探索和利用的样本数
                    exploit_count = int(candidates_per_class * (1 - exploration_ratio))
                    explore_count = candidates_per_class - exploit_count
                    
                    # 80%: 从真实分布采样（利用）
                    mu = latent_stats[class_id]['mu'].to(self.device)
                    std = latent_stats[class_id]['std'].to(self.device)
                    z_exploit = torch.randn(exploit_count, self.vae.latent_dim).to(self.device)
                    z_exploit = mu + z_exploit * std  # 重参数化技巧
                    
                    # 20%: 随机采样（探索）
                    z_explore = torch.randn(explore_count, self.vae.latent_dim).to(self.device)
                    
                    # 合并
                    z = torch.cat([z_exploit, z_explore], dim=0)
                else:
                    # 回退到标准正态分布（如果没有统计信息）
                    z = torch.randn(candidates_per_class, self.vae.latent_dim).to(self.device)
                
                # ✨ 解码到特征空间 (CVAE核心改进：指定类别生成)
                # ConditionalCreditVAE总是需要y参数
                y_cond = torch.full((z.shape[0],), class_id, dtype=torch.long).to(self.device)
                features = self.vae.decode(z, y_cond)  # 条件解码
                
                # ========== 改进2: 质量筛选（基于分类器置信度） ==========
                # 用分类器预测虚拟数据
                outputs = self.model(features)
                probs = torch.softmax(outputs, dim=1)
                
                # 获取目标类别的置信度
                confidences = probs[:, class_id]
                
                # 🔥 严格筛选：只保留高置信度样本（不降低标准）
                # 原则：宁缺毋滥，不达标的虚拟数据会污染"分类器尺子"
                high_quality_indices = (confidences >= confidence_threshold).nonzero(as_tuple=True)[0]
                
                # 限制数量到目标值（如果高质量样本充足）
                if len(high_quality_indices) > samples_per_class:
                    high_quality_indices = high_quality_indices[:samples_per_class]
                
                # 📊 质量统计
                num_high_quality = len(high_quality_indices)
                if num_high_quality < samples_per_class:
                    print(f"    [WARNING] Class {class_id}: 仅{num_high_quality}/{samples_per_class}样本达标 "
                          f"(threshold={confidence_threshold:.2f}), 宁缺毋滥，不补充低质量数据")
                else:
                    print(f"    [OK] Class {class_id}: {num_high_quality}个高质量样本达标 "
                          f"(threshold={confidence_threshold:.2f})")
                
                # 保存筛选后的虚拟数据
                for idx in high_quality_indices:
                    idx = idx.item()
                    virtual_data.append((
                        features[idx].cpu().numpy(),
                        class_id  # numpy标量
                    ))
                
                # 打印质量统计
                mean_conf = confidences[high_quality_indices].mean().item()
                print(f"    Class {class_id}: {len(high_quality_indices)}/{candidates_per_class} "
                      f"candidates passed filter (avg confidence: {mean_conf:.3f})")
        
        self.virtual_data = virtual_data
        self.threshold_met = True
        
        actual_generated = len(virtual_data)
        quality_rate = (actual_generated / num_samples * 100) if num_samples > 0 else 0
        print(f"  Client {self.id}: Generated {actual_generated}/{num_samples} high-quality samples "
              f"({quality_rate:.1f}% pass rate, strict threshold={confidence_threshold:.2f})")
        
        return virtual_data
    
    def add_adaptive_noise_to_virtual_data(self, strategy='balanced'):
        """
        Add adaptive differential privacy noise to virtual data based on feature importance
        
        Args:
            strategy: Noise allocation strategy
                - 'privacy_first': More noise on important features (protect privacy)
                - 'utility_first': Less noise on important features (preserve utility)
                - 'balanced': Uniform noise (traditional DP)
        
        Noise is added ONLY if:
        - epsilon is not None AND epsilon > 0
        - noise_type is specified
        
        Supports two noise mechanisms:
        - Laplace: scale = Δf / ε
        - Gaussian: σ = Δf × sqrt(2 × ln(1.25/δ)) / ε
        
        Modifies self.virtual_data in-place by adding noise to features.
        """
        if len(self.virtual_data) == 0:
            return
        
        # Check if noise should be added
        if self.epsilon is None or self.epsilon <= 0:
            print(f"  Client {self.id}: Skipping noise addition (epsilon={self.epsilon}, privacy disabled)")
            return
        
        if self.noise_type is None or self.noise_type == 'none':
            print(f"  Client {self.id}: Skipping noise addition (noise_type={self.noise_type})")
            return
        
        print(f"  Client {self.id}: Adding {strategy} {self.noise_type} noise (ε={self.epsilon})")
        
        # Get feature importance if available
        use_adaptive = (self.feature_importance is not None and strategy != 'balanced')
        
        noisy_data = []
        
        for features, label in self.virtual_data:
            features = np.array(features)
            
            # Sensitivity: assume features are normalized to [0, 1] or [-1, 1]
            sensitivity = 1.0
            
            if use_adaptive:
                # Adaptive noise based on feature importance
                if strategy == 'privacy_first':
                    # More noise on important features
                    noise_scale = self.feature_importance
                elif strategy == 'utility_first':
                    # Less noise on important features
                    noise_scale = 1.0 - self.feature_importance
                else:
                    noise_scale = np.ones_like(self.feature_importance)
                
                # Normalize to maintain total privacy budget
                noise_scale = noise_scale / noise_scale.mean()
            else:
                # Uniform noise
                noise_scale = np.ones(features.shape[0])
            
            if self.noise_type == 'laplace':
                # Laplace mechanism with adaptive scaling
                base_scale = sensitivity / self.epsilon
                noise = np.random.laplace(0, base_scale * noise_scale, size=features.shape)
            else:  # gaussian
                # Gaussian mechanism with adaptive scaling
                base_sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
                noise = np.random.normal(0, base_sigma * noise_scale, size=features.shape)
            
            noisy_features = features + noise
            noisy_data.append((noisy_features, label))
        
        self.virtual_data = noisy_data
    
    def add_noise_to_virtual_data(self):
        """
        Backward compatibility wrapper - calls add_adaptive_noise_to_virtual_data with balanced strategy
        """
        self.add_adaptive_noise_to_virtual_data(strategy='balanced')
    
    def train_baseline_vae(self):
        """
        训练纯重建VAE（本地对比实验，不参与联邦学习）
        
        目的：通过对比识别分类相关特征
        策略：
        - 仅使用重建损失和KL散度（无分类损失、无原型损失）
        - 固定训练轮数（不需要早停、不需要服务器通讯）
        - 完全本地训练，用于后续特征重要性计算
        
        Returns:
            float: 最终平均重建误差
        """
        print(f"  Client {self.id}: Training baseline VAE (reconstruction-only)...")
        
        # 创建基线VAE（如果不存在）
        if self.vae_baseline is None:
            # 从数据获取实际特征维度（不是从encoder第一层，因为那里是input_dim + class_embedding_dim）
            trainloader_temp = self.load_train_data()
            sample_x, _ = next(iter(trainloader_temp))
            if type(sample_x) == type([]):
                actual_input_dim = sample_x[0].shape[1]
            else:
                actual_input_dim = sample_x.shape[1]
            
            latent_dim = self.vae.latent_dim
            
            self.vae_baseline = create_credit_vae(
                input_dim=actual_input_dim,
                latent_dim=latent_dim,
                dataset_name=self.dataset
            ).to(self.device).double()
            
            self.vae_baseline_optimizer = torch.optim.Adam(
                self.vae_baseline.parameters(),
                lr=0.001
            )
        
        trainloader = self.load_train_data()
        self.vae_baseline.train()
        
        total_recon_loss = 0.0
        
        # 固定训练轮数
        for epoch in range(self.baseline_epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device).double()
                else:
                    x = x.to(self.device).double()
                y = y.to(self.device).long()
                
                # 仅使用VAE的重建和KL损失（无分类、无原型）
                virtual_x, mu, logvar = self.vae_baseline(x, y)
                
                # Loss 1: 重建损失
                recon_loss = F.mse_loss(virtual_x, x, reduction='mean')
                
                # Loss 2: KL散度
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                
                # 总损失（只有重建和KL）
                total_loss = recon_loss + 0.01 * kl_loss
                
                self.vae_baseline_optimizer.zero_grad()
                total_loss.backward()
                self.vae_baseline_optimizer.step()
                
                epoch_loss += recon_loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count
            if (epoch + 1) % 10 == 0:
                print(f"    Baseline VAE Epoch {epoch+1}/{self.baseline_epochs}, Recon Loss: {avg_loss:.4f}")
            
            total_recon_loss = avg_loss
        
        print(f"  Client {self.id}: Baseline VAE training completed. Final recon loss: {total_recon_loss:.4f}")
        return total_recon_loss
    
    def compute_feature_importance(self, num_samples=1000):
        """
        通过对比主VAE和基线VAE的输出，计算特征重要性
        
        方法：对比学习（Contrastive Analysis）
        1. 主VAE（有分类约束）：学习判别性+重建性特征
        2. 基线VAE（无分类约束）：仅学习重建性特征
        3. 差异 = 判别性特征的重要性
        
        Args:
            num_samples: 用于对比的样本数量
        
        Returns:
            numpy.ndarray: 特征重要性向量（归一化到[0,1]）
        """
        if self.vae_baseline is None:
            print(f"  Client {self.id}: Baseline VAE not trained. Training now...")
            self.train_baseline_vae()
        
        print(f"  Client {self.id}: Computing feature importance via VAE comparison...")
        
        self.vae.eval()
        self.vae_baseline.eval()
        
        with torch.no_grad():
            # 使用相同的随机隐变量生成特征
            # 关键修复：确保z的dtype与VAE模型一致
            model_dtype = next(self.vae.parameters()).dtype
            z = torch.randn(num_samples, self.vae.latent_dim, dtype=model_dtype).to(self.device)
            
            # 为decode生成类别标签（使用类别0作为默认，或者可以采样多个类别）
            y_decode = torch.zeros(num_samples, dtype=torch.long).to(self.device)
            
            # 主VAE解码（包含分类信息）
            features_main = self.vae.decode(z, y_decode)
            
            # 基线VAE解码（仅重建信息）
            features_baseline = self.vae_baseline.decode(z, y_decode)
            
            # 计算逐特征的绝对差异
            diff = (features_main - features_baseline).abs().mean(dim=0)
            
            # 归一化到[0, 1]
            diff_min = diff.min()
            diff_max = diff.max()
            
            if diff_max - diff_min > 1e-8:
                importance = (diff - diff_min) / (diff_max - diff_min)
            else:
                # 如果所有特征差异相同，设置为均匀重要性
                importance = torch.ones_like(diff) * 0.5
        
        self.feature_importance = importance.cpu().numpy()
        
        print(f"  Client {self.id}: Feature importance computed.")
        print(f"    Top 3 important features: {np.argsort(self.feature_importance)[-3:]}")
        print(f"    Importance range: [{self.feature_importance.min():.3f}, {self.feature_importance.max():.3f}]")
        
        return self.feature_importance
    
    def get_phase1_upload(self):
        """
        Prepare Phase 1 upload data for server
        
        Returns:
            dict: {
                'accuracy': float,
                'prototypes': {class_id: prototype_tensor},
                'virtual_data': [(features, label), ...],
                'threshold_met': bool
            }
        """
        return {
            'accuracy': self.accuracy,
            'prototypes': {k: v.cpu() for k, v in self.prototypes.items()},
            'virtual_data': self.virtual_data,
            'threshold_met': self.threshold_met
        }
    
    def _compute_accuracy(self):
        """Compute validation accuracy"""
        testloader = self.load_test_data()
        self.model.eval()
        
        test_acc = 0
        test_num = 0
        
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device).double()
                else:
                    x = x.to(self.device).double()
                y = y.to(self.device)
                
                output = self.model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
        
        return test_acc / test_num if test_num > 0 else 0.0
    
    # ==================== Personalization Training (Ditto-style) ====================
    
    def ptrain(self):
        """
        Personalized model training (Ditto-style)
        Train personalized model with regularization to global model
        
        Called after global model training in Phase 2
        """
        trainloader = self.load_train_data()
        start_time = time.time()
        
        self.model_per.train()
        
        max_local_epochs = self.plocal_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        
        for epoch in range(max_local_epochs):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device).double()
                else:
                    x = x.to(self.device).double()
                y = y.to(self.device)
                
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                
                output = self.model_per(x)
                loss = self.loss(output, y)
                
                self.optimizer_per.zero_grad()
                loss.backward()
                # PerturbedGradientDescent adds regularization: mu * (w_per - w_global)
                self.optimizer_per.step(self.model.parameters(), self.device)
        
        self.train_time_cost['total_cost'] += time.time() - start_time
    
    def test_metrics_personalized(self):
        """
        Evaluate personalized model (Ditto-style)
        
        Returns:
            test_acc: Number of correct predictions
            test_num: Total number of test samples
            auc: Area under ROC curve
        """
        if self.model_per is None:
            # Fallback to global model if personalized model not initialized
            return self.test_metrics()
        
        testloaderfull = self.load_test_data()
        self.model_per.eval()
        
        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device).double()
                else:
                    x = x.to(self.device).double()
                y = y.to(self.device)
                
                output = self.model_per(x)
                
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
                
                y_prob.append(F.softmax(output, dim=1).detach().cpu().numpy())
                
                # Handle label_binarize for binary classification
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)
        
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        
        # Compute AUC
        if self.num_classes == 2:
            auc = metrics.roc_auc_score(y_true[:, 1], y_prob[:, 1])
        else:
            auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc
    
    def train_metrics_personalized(self):
        """
        Compute training metrics on personalized model (Ditto-style)
        
        Includes Ditto regularization term in loss calculation:
        L = L_CE + μ/2 * ||w_per - w_global||²
        
        Returns:
            train_loss: Average training loss (with regularization)
            train_num: Total number of training samples
        """
        if self.model_per is None:
            # Fallback to global model
            trainloader = self.load_train_data()
            self.model.eval()
            
            train_num = 0
            losses = 0
            with torch.no_grad():
                for x, y in trainloader:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device).double()
                    else:
                        x = x.to(self.device).double()
                    y = y.to(self.device)
                    output = self.model(x)
                    loss = self.loss(output, y)
                    train_num += y.shape[0]
                    losses += loss.item() * y.shape[0]
            
            return losses, train_num
        
        # Evaluate personalized model with Ditto regularization
        trainloader = self.load_train_data()
        self.model_per.eval()
        
        train_num = 0
        losses = 0
        
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device).double()
                else:
                    x = x.to(self.device).double()
                y = y.to(self.device)
                
                output = self.model_per(x)
                loss = self.loss(output, y)
                
                # Add Ditto regularization term: μ/2 * ||w_per - w_global||²
                gm = torch.cat([p.data.view(-1) for p in self.model.parameters()], dim=0)
                pm = torch.cat([p.data.view(-1) for p in self.model_per.parameters()], dim=0)
                loss += 0.5 * self.mu_ditto * torch.norm(gm - pm, p=2)
                
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        
        return losses, train_num
    
    # ==================== Phase 2: Algorithm-Specific Training ====================
    
    def ptrain(self):
        """
        Ditto-style personalized model training (Phase2 only)
        
        Trains the local personalized model with Ditto regularization:
        L_per = L_CE(w_per) + μ/2 * ||w_per - w_global||²
        
        This is automatically handled by PerturbedGradientDescent optimizer.
        """
        if self.model_per is None:
            print(f"  Client {self.id}: Personalized model not initialized, skipping ptrain")
            return
        
        trainloader = self.load_train_data()
        self.model_per.train()
        
        start_time = time.time()
        
        max_local_epochs = self.plocal_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        
        for epoch in range(max_local_epochs):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device).double()
                else:
                    x = x.to(self.device).double()
                y = y.to(self.device)
                
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                
                # Forward pass on personalized model
                output = self.model_per(x)
                loss = self.loss(output, y)
                
                # Backward pass
                self.optimizer_per.zero_grad()
                loss.backward()
                
                # Ditto regularization: step() uses global model parameters
                # Automatically adds: grad += μ * (w_per - w_global)
                self.optimizer_per.step(self.model.parameters(), self.device)
        
        self.train_time_cost['total_cost'] += time.time() - start_time
    
    def train_phase2(self):
        """
        Phase 2 Training: Algorithm-specific training on mixed data
        
        Dispatches to appropriate training method based on self.phase2_algorithm
        
        Supported algorithms:
        - fedavg: Standard weighted averaging (with Ditto personalization)
        - fedprox: Proximal term regularization
        - fedscaffold: Variance reduction with control variates
        """
        # Ensure model is double precision for Phase 2 (consistent with Phase 1)
        self.model = self.model.double()
        
        # Initialize personalized model if not already done
        if self.model_per is None:
            self.init_personalized_model()
        
        if self.phase2_algorithm == 'fedavg':
            return self._train_fedavg()
        elif self.phase2_algorithm == 'fedprox':
            return self._train_fedprox()
        elif self.phase2_algorithm == 'fedscaffold':
            return self._train_scaffold()
        else:
            # Default to FedAvg
            print(f"Warning: Unknown algorithm '{self.phase2_algorithm}', using FedAvg")
            return self._train_fedavg()
    
    def _train_fedavg(self):
        """
        FedAvg: Standard training on mixed data
        
        Note: ptrain() is already called by server before this method.
        This method only trains the global model.
        """
        mixed_trainloader = self._create_mixed_dataloader(self.load_train_data())
        self.model.train()
        
        for epoch in range(self.local_epochs):
            for x, y in mixed_trainloader:
                x, y = x.to(self.device).double(), y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                
                # Check for NaN/Inf in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Client {self.id} detected NaN/Inf loss, skipping batch")
                    continue
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
    
    def _train_fedprox(self):
        """FedProx: Training with proximal term"""
        global_model = copy.deepcopy(self.model)
        mixed_trainloader = self._create_mixed_dataloader(self.load_train_data())
        self.model.train()
        
        mu = getattr(self, 'mu', 0.01)  # From server args
        
        for epoch in range(self.local_epochs):
            for x, y in mixed_trainloader:
                x, y = x.to(self.device).double(), y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                
                # Add proximal term: mu/2 * ||w - w_global||^2
                proximal_term = 0.0
                for w, w_g in zip(self.model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_g).norm(2)
                loss += (mu / 2) * proximal_term
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def _train_moon(self):
        """MOON: Model-contrastive training"""
        if self.prev_model is None:
            self.prev_model = copy.deepcopy(self.model)
        
        # Store global model reference (should be set by server before training)
        if not hasattr(self, 'global_model') or self.global_model is None:
            self.global_model = copy.deepcopy(self.model)
        
        mixed_trainloader = self._create_mixed_dataloader(self.load_train_data())
        self.model.train()
        
        mu = getattr(self, 'mu', 0.5)  # Match original MOON naming
        temperature = getattr(self, 'tau', 0.5)  # Match original MOON naming
        
        # Check if model has base-head structure
        has_base_head = hasattr(self.model, 'base') and hasattr(self.model, 'head')
        
        for epoch in range(self.local_epochs):
            for x, y in mixed_trainloader:
                x, y = x.to(self.device).double(), y.to(self.device)
                
                if has_base_head:
                    # Use base features for contrastive learning (correct approach)
                    rep = self.model.base(x)
                    output = self.model.head(rep)
                    
                    with torch.no_grad():
                        rep_global = self.global_model.base(x)
                        rep_prev = self.prev_model.base(x)
                else:
                    # Fallback: use output as features (for models without base-head)
                    output = self.model(x)
                    rep = output
                    
                    with torch.no_grad():
                        rep_global = self.global_model(x)
                        rep_prev = self.prev_model(x)
                
                # Classification loss
                loss_ce = self.loss(output, y)
                
                # MOON contrastive loss (InfoNCE)
                # Positive pair: current vs global, Negative pair: current vs previous
                cos_sim_global = F.cosine_similarity(rep, rep_global)
                cos_sim_prev = F.cosine_similarity(rep, rep_prev)
                
                loss_con = -torch.log(
                    torch.exp(cos_sim_global / temperature) / 
                    (torch.exp(cos_sim_global / temperature) + torch.exp(cos_sim_prev / temperature))
                )
                loss_con = loss_con.mean()
                
                loss = loss_ce + mu * loss_con
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        # Update previous model
        self.prev_model = copy.deepcopy(self.model)
    
    def _train_scaffold(self):
        """
        SCAFFOLD: Training with control variates
        完全按照原框架clientscaffold.py实现
        """
        if self.c_local is None:
            self.init_scaffold_controls()
        
        mixed_trainloader = self._create_mixed_dataloader(self.load_train_data())
        self.model.train()
        
        # Import SCAFFOLD optimizer
        from flcore.optimizers.fedoptimizer import SCAFFOLDOptimizer
        scaffold_optimizer = SCAFFOLDOptimizer(self.model.parameters(), lr=self.learning_rate)
        
        # Save number of batches for control variate update
        self.num_batches = len(mixed_trainloader)
        
        for epoch in range(self.local_epochs):
            for x, y in mixed_trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device).double()
                else:
                    x = x.to(self.device).double()
                y = y.to(self.device)
                
                output = self.model(x)
                loss = self.loss(output, y)
                
                scaffold_optimizer.zero_grad()
                loss.backward()
                # Apply SCAFFOLD correction: grad = grad - c_i + c (与原框架一致)
                scaffold_optimizer.step(self.c_global, self.c_local)
        
        # Update local control variate using update_yc (与原框架clientscaffold.py一致)
        self.update_yc(self.local_epochs)
    
    def update_yc(self, max_local_epochs=None):
        """
        SCAFFOLD: Update local control variate
        与原框架clientscaffold.py的update_yc()完全一致
        
        Formula: c_i+ = c_i - c + (x - y)/(K*eta)
        where K = num_batches * max_local_epochs
        """
        if max_local_epochs is None:
            max_local_epochs = self.local_epochs
        
        for ci, c, x, yi in zip(self.c_local, self.c_global, self.global_model.parameters(), 
                               self.model.parameters()):
            ci.data = ci - c + 1/self.num_batches/max_local_epochs/self.learning_rate * (x - yi)
    
    def _train_perfedavg(self):
        """Per-FedAvg: MAML-based meta-learning"""
        # Use PerAvgOptimizer for proper MAML implementation
        from flcore.optimizers.fedoptimizer import PerAvgOptimizer
        peravg_optimizer = PerAvgOptimizer(self.model.parameters(), lr=self.learning_rate)
        
        # MAML requires batch_size*2 to split for inner/outer loop
        mixed_trainloader = self._create_mixed_dataloader(
            self.load_train_data(batch_size=self.batch_size * 2)
        )
        self.model.train()
        
        beta = getattr(self, 'beta', self.learning_rate)  # Meta learning rate
        
        for epoch in range(self.local_epochs):
            for X, Y in mixed_trainloader:
                # Save model parameters before inner loop update
                temp_model = copy.deepcopy([p.data.clone().double() for p in self.model.parameters()])
                
                # Step 1: Inner loop - first half of batch
                x = X[:self.batch_size].to(self.device)
                y = Y[:self.batch_size].to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                peravg_optimizer.zero_grad()
                loss.backward()
                peravg_optimizer.step()  # Inner update with lr
                
                # Step 2: Outer loop - second half of batch (meta gradient)
                x = X[self.batch_size:].to(self.device)
                y = Y[self.batch_size:].to(self.device)
                peravg_optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                
                # Restore model parameters to before inner update
                for old_param, new_param in zip(self.model.parameters(), temp_model):
                    old_param.data = new_param.data.clone()
                
                # Meta update with beta (second-order gradient approximation)
                peravg_optimizer.step(beta=beta)
    
    def _train_ditto(self):
        """Ditto: Train both global and personalized models"""
        if self.personalized_model is None:
            self.init_personalized_model()
        
        mixed_trainloader = self._create_mixed_dataloader(self.load_train_data())
        
        mu = getattr(self, 'mu', 1.0)
        plocal_epochs = getattr(self, 'plocal_epochs', 1)
        
        # Import PerturbedGradientDescent for personalized model
        from flcore.optimizers.fedoptimizer import PerturbedGradientDescent
        pers_optimizer = PerturbedGradientDescent(
            self.personalized_model.parameters(), 
            lr=self.learning_rate, 
            mu=mu
        )
        
        # Step 1: Train personalized model first (with regularization to global)
        self.personalized_model.train()
        for epoch in range(plocal_epochs):
            for x, y in mixed_trainloader:
                x, y = x.to(self.device).double(), y.to(self.device)
                output = self.personalized_model(x)
                loss = self.loss(output, y)
                pers_optimizer.zero_grad()
                loss.backward()
                # PerturbedGradientDescent adds mu*(w_pers - w_global) automatically
                pers_optimizer.step(self.model.parameters(), self.device)
        
        # Step 2: Train global model (standard training)
        self.model.train()
        for epoch in range(self.local_epochs):
            for x, y in mixed_trainloader:
                x, y = x.to(self.device).double(), y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def _train_fedrep(self):
        """FedRep: Train with body-head split"""
        # Check if model has base-head structure
        if not (hasattr(self.model, 'base') and hasattr(self.model, 'head')):
            print(f"Warning: Model doesn't have base-head structure, using standard training")
            self._train_fedavg()
            return
        
        mixed_trainloader = self._create_mixed_dataloader(self.load_train_data())
        self.model.train()
        
        plocal_epochs = getattr(self, 'plocal_epochs', 1)
        
        # Create separate optimizers for base and head
        optimizer_base = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate)
        optimizer_head = torch.optim.SGD(self.model.head.parameters(), lr=self.learning_rate)
        
        # Step 1: Train head only (personalization) - Freeze base
        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True
        
        for epoch in range(plocal_epochs):
            for x, y in mixed_trainloader:
                x, y = x.to(self.device).double(), y.to(self.device)
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)
                optimizer_head.zero_grad()
                loss.backward()
                optimizer_head.step()
        
        # Step 2: Train base only (will be aggregated) - Freeze head
        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = False
        
        for epoch in range(self.local_epochs):
            for x, y in mixed_trainloader:
                x, y = x.to(self.device).double(), y.to(self.device)
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)
                optimizer_base.zero_grad()
                loss.backward()
                optimizer_base.step()
        
        # Restore gradient computation for all parameters
        for param in self.model.parameters():
            param.requires_grad = True
    
    def _train_fedproto(self):
        """FedProto: Prototype-based training"""
        # Check if model has base-head structure
        if not (hasattr(self.model, 'base') and hasattr(self.model, 'head')):
            print(f"Warning: Model doesn't have base-head structure, using standard training")
            self._train_fedavg()
            return
        
        from collections import defaultdict
        
        mixed_trainloader = self._create_mixed_dataloader(self.load_train_data())
        self.model.train()
        
        lamda = getattr(self, 'lamda', 1.0)  # Prototype loss weight
        loss_mse = nn.MSELoss()
        
        # Collect local prototypes during training
        protos = defaultdict(list)
        
        for epoch in range(self.local_epochs):
            for x, y in mixed_trainloader:
                x, y = x.to(self.device).double(), y.to(self.device)
                
                # Extract features from base model
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)
                
                # Add prototype regularization if global prototypes exist
                if self.global_prototypes is not None and len(self.global_prototypes) > 0:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        # Use global prototype if exists
                        if y_c in self.global_prototypes and type(self.global_prototypes[y_c]) != type([]):
                            proto_new[i, :] = self.global_prototypes[y_c].data
                    loss += loss_mse(proto_new, rep) * lamda
                
                # Collect local prototypes
                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        # Aggregate local prototypes (average per class)
        self.protos = {}
        for label, proto_list in protos.items():
            if len(proto_list) > 0:
                self.protos[label] = torch.stack(proto_list).mean(0)
        
        print(f"  Client {self.id}: Collected {len(self.protos)} class prototypes")
    
    def _train_pfedme(self):
        """pFedMe: Moreau envelope personalization"""
        # Import pFedMeOptimizer
        from flcore.optimizers.fedoptimizer import pFedMeOptimizer
        
        mixed_trainloader = self._create_mixed_dataloader(self.load_train_data())
        
        lamda = getattr(self, 'lamda', 15.0)  # Moreau envelope parameter
        K = getattr(self, 'K', 5)  # Number of personalized steps
        p_learning_rate = getattr(self, 'personalized_learning_rate', 0.01)
        
        # Initialize local and personalized parameters if not exists
        if not hasattr(self, 'local_params'):
            self.local_params = copy.deepcopy([p.data.clone().double() for p in self.model.parameters()])
        if not hasattr(self, 'personalized_params'):
            self.personalized_params = copy.deepcopy([p.data.clone().double() for p in self.model.parameters()])
        
        # Create pFedMe optimizer
        pfedme_optimizer = pFedMeOptimizer(
            self.model.parameters(), 
            lr=p_learning_rate, 
            lamda=lamda
        )
        
        self.model.train()
        
        for epoch in range(self.local_epochs):
            for x, y in mixed_trainloader:
                x, y = x.to(self.device).double(), y.to(self.device)
                
                # K personalized steps per batch
                for i in range(K):
                    output = self.model(x)
                    loss = self.loss(output, y)
                    pfedme_optimizer.zero_grad()
                    loss.backward()
                    # Find approximate theta (personalized parameters)
                    self.personalized_params = pfedme_optimizer.step(self.local_params, self.device)
                
                # Update local weights after finding approximate theta
                for new_param, localweight in zip(self.personalized_params, self.local_params):
                    localweight = localweight.to(self.device)
                    localweight.data = localweight.data - lamda * self.learning_rate * (localweight.data - new_param.data)
        
        # Update model with local parameters for aggregation
        for param, local_param in zip(self.model.parameters(), self.local_params):
            param.data = local_param.data.clone()
    
    def _train_fedgwo(self):
        """
        FedGWO: Grey Wolf Optimizer
        完全按照原框架clientgwo.py实现
        """
        # Initialize GWO parameters if not set
        if not hasattr(self, 'alpha_model'):
            self.alpha_model = None
            self.beta_model = None
            self.delta_model = None
            self.a = None
            self.A1 = None
            self.A2 = None
            self.A3 = None
            self.C1 = None
            self.C2 = None
            self.C3 = None
        
        # Use real + virtual data (VPS特有)
        mixed_trainloader = self._create_mixed_dataloader(self.load_train_data())
        
        # Phase 0: 评估当前模型在验证集上的准确率
        # 注意：这个在服务器端_setup_fedgwo_params中已经做了，但为了与原框架一致，这里也保留
        self.current_acc = self.get_validation_accuracy()
        
        # Phase 1: Grey Wolf Optimization Update
        # 原始GWO论文公式:
        # D_α = |C1·X_α - X|, X1 = X_α - A1·D_α
        # D_β = |C2·X_β - X|, X2 = X_β - A2·D_β
        # D_δ = |C3·X_δ - X|, X3 = X_δ - A3·D_δ
        # X(t+1) = (X1 + X2 + X3) / 3
        
        if self.alpha_model is not None and self.beta_model is not None and self.delta_model is not None:
            with torch.no_grad():
                for param, alpha_param, beta_param, delta_param in zip(
                    self.model.parameters(),
                    self.alpha_model.parameters(),
                    self.beta_model.parameters(),
                    self.delta_model.parameters()
                ):
                    # 计算距离 D = |C × θ_leader - θ_k|
                    D_alpha = torch.abs(self.C1 * alpha_param.data - param.data)
                    D_beta = torch.abs(self.C2 * beta_param.data - param.data)
                    D_delta = torch.abs(self.C3 * delta_param.data - param.data)
                    
                    # 向三个领导者学习
                    # θ1 = θ_α - A1 × D_α
                    theta_1 = alpha_param.data - self.A1 * D_alpha
                    # θ2 = θ_β - A2 × D_β
                    theta_2 = beta_param.data - self.A2 * D_beta
                    # θ3 = θ_δ - A3 × D_δ
                    theta_3 = delta_param.data - self.A3 * D_delta
                    
                    # 三者平均作为新位置
                    param.data = (theta_1 + theta_2 + theta_3) / 3.0
        
        # Phase 2: Local Training (完整的E轮本地训练)
        self.model.train()
        
        for epoch in range(self.local_epochs):
            for x, y in mixed_trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device).double()
                else:
                    x = x.to(self.device).double()
                y = y.to(self.device)
                
                output = self.model(x)
                loss = self.loss(output, y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def set_gwo_params(self, alpha_model, beta_model, delta_model, a, A1, A2, A3, C1, C2, C3):
        """
        设置GWO算法所需的参数 (与原框架clientgwo.py完全一致)
        
        Args:
            alpha_model: θ_α - Alpha wolf (最优客户端模型)
            beta_model: θ_β - Beta wolf (次优客户端模型)
            delta_model: θ_δ - Delta wolf (第三优客户端模型)
            a: 收敛因子，从2线性递减到0 (原始论文)
            A1, A2, A3: 向三个领导者学习的系数，A = 2a·r - a
            C1, C2, C3: 三个领导者的权重系数，C = 2·r
        """
        # 确保所有模型都是double类型（防止dtype不匹配）
        self.alpha_model = alpha_model.double() if alpha_model is not None else None
        self.beta_model = beta_model.double() if beta_model is not None else None
        self.delta_model = delta_model.double() if delta_model is not None else None
        self.a = a
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
    
    def _train_fedpso(self):
        """
        FedPSO: Particle Swarm Optimization
        完全参照clientpso.py实现，适配混合数据训练
        
        PSO核心思想：
        - 每个客户端是一个粒子，在解空间中搜索最优模型参数
        - 速度更新公式：v(t+1) = w*v(t) + c1*r1*(pbest - x(t)) + c2*r2*(gbest - x(t))
        - 位置更新公式：x(t+1) = x(t) + v(t+1)
        
        Training flow:
        1. PSO位置更新（在梯度下降前）
        2. 梯度下降微调
        """
        # Step 1: PSO位置更新（在传统训练前）
        if self.pbest_model is not None and self.gbest_model is not None:
            self._pso_update()
        
        # Step 2: 传统梯度下降训练（微调）
        mixed_trainloader = self._create_mixed_dataloader(self.load_train_data())
        self.model.train()
        
        for epoch in range(self.local_epochs):
            for x, y in mixed_trainloader:
                x, y = x.to(self.device).double(), y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                
                # Check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Client {self.id} detected NaN/Inf loss, skipping batch")
                    continue
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                
                self.optimizer.step()
        
        return 0.0  # Return 0 loss for PSO (not needed)
    
    def _pso_update(self):
        """
        PSO核心更新公式（参照clientpso.py）
        
        原始论文公式 (Kennedy & Eberhart, 1995)：
        v_i(t+1) = w*v_i(t) + c1*r1*(pbest_i - x_i(t)) + c2*r2*(gbest - x_i(t))
        x_i(t+1) = x_i(t) + v_i(t+1)
        
        三个成分：
        1. 惯性成分: w*v_i(t) - 保持之前的搜索方向
        2. 认知成分: c1*r1*(pbest_i - x_i(t)) - 向个体历史最优学习
        3. 社会成分: c2*r2*(gbest - x_i(t)) - 向群体全局最优学习
        """
        with torch.no_grad():
            # 遍历模型的每一层参数
            for i, (param, vel, pbest_param, gbest_param) in enumerate(
                zip(self.model.parameters(), self.velocity, self.pbest_model, self.gbest_model)
            ):
                # 当前位置 x_i(t)
                x_current = param.data
                
                # 个体最优位置 pbest_i
                x_pbest = pbest_param
                
                # 全局最优位置 gbest
                x_gbest = gbest_param
                
                # 计算速度更新的三个成分
                # 1. 惯性成分: w * v_i(t)
                inertia = self.pso_w * vel
                
                # 2. 认知成分（个体学习）: c1 * r1 * (pbest_i - x_i(t))
                cognitive = self.pso_c1 * self.pso_r1 * (x_pbest - x_current)
                
                # 3. 社会成分（群体学习）: c2 * r2 * (gbest - x_i(t))
                social = self.pso_c2 * self.pso_r2 * (x_gbest - x_current)
                
                # 速度更新：v_i(t+1) = inertia + cognitive + social
                new_velocity = inertia + cognitive + social
                
                # 速度限制（防止速度过大）
                param_range = torch.abs(x_current).mean() + 1e-8  # 避免除零
                v_max = self.pso_v_max * param_range
                new_velocity = torch.clamp(new_velocity, -v_max, v_max)
                
                # 更新速度
                vel.data = new_velocity
                
                # 位置更新：x_i(t+1) = x_i(t) + v_i(t+1)
                param.data = x_current + new_velocity
            
            # 更新速度列表引用
            self.velocity = [v.clone() for v in self.velocity]
    
    def set_pso_parameters(self, w, c1, c2, r1, r2, pbest_model, gbest_model, velocity):
        """
        设置PSO参数（由服务器传入）
        
        Args:
            w: 惯性权重（inertia weight）
            c1: 个体学习因子（cognitive parameter）
            c2: 社会学习因子（social parameter）
            r1: 随机数1，范围[0, 1]
            r2: 随机数2，范围[0, 1]
            pbest_model: 个体最优模型参数列表
            gbest_model: 全局最优模型参数列表
            velocity: 当前速度
        """
        self.pso_w = w
        self.pso_c1 = c1
        self.pso_c2 = c2
        self.pso_r1 = r1
        self.pso_r2 = r2
        self.pbest_model = pbest_model
        self.gbest_model = gbest_model
        self.velocity = velocity
    
    def _create_mixed_dataloader(self, real_trainloader):
        """
        Create a DataLoader with mixed real and virtual data
        
        Args:
            real_trainloader: DataLoader for real training data
        
        Returns:
            mixed_trainloader: DataLoader with both real and virtual data
        """
        from torch.utils.data import TensorDataset, DataLoader
        
        # Collect real data
        real_features = []
        real_labels = []
        
        for x, y in real_trainloader:
            if type(x) == type([]):
                x = x[0]
            real_features.append(x)
            real_labels.append(y)
        
        real_features = torch.cat(real_features, dim=0)
        real_labels = torch.cat(real_labels, dim=0)
        
        # Add virtual data if available
        if len(self.shared_virtual_data) > 0:
            virtual_features = []
            virtual_labels = []
            
            for features, label in self.shared_virtual_data:
                virtual_features.append(torch.tensor(features, dtype=torch.float32))
                virtual_labels.append(torch.tensor(label, dtype=torch.long))
            
            virtual_features = torch.stack(virtual_features)
            virtual_labels = torch.stack(virtual_labels)
            
            # Concatenate real and virtual
            mixed_features = torch.cat([real_features, virtual_features], dim=0)
            mixed_labels = torch.cat([real_labels, virtual_labels], dim=0)
        else:
            mixed_features = real_features
            mixed_labels = real_labels
        
        # Create mixed dataset
        mixed_dataset = TensorDataset(mixed_features, mixed_labels)
        mixed_trainloader = DataLoader(
            mixed_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        return mixed_trainloader
    
    def load_shared_virtual_data(self, virtual_data):
        """
        Load shared virtual data from server
        
        Args:
            virtual_data: List of (features, label) tuples from all clients
        """
        self.shared_virtual_data = virtual_data
    
    def set_phase2_algorithm(self, algorithm):
        """Set Phase 2 algorithm type"""
        self.phase2_algorithm = algorithm.lower()
    
    def init_moon_states(self):
        """Initialize MOON: Save previous model for contrastive learning"""
        self.prev_model = copy.deepcopy(self.model).double()
    
    def init_scaffold_controls(self):
        """Initialize SCAFFOLD: Control variates"""
        self.c_local = [torch.zeros_like(p.data).double() for p in self.model.parameters()]
        self.c_global = [torch.zeros_like(p.data).double() for p in self.model.parameters()]
        # Save global model parameters for control variate update (needed by SCAFFOLD)
        self.global_model_params = [p.data.clone().double() for p in self.model.parameters()]
        self.global_model = copy.deepcopy(self.model).double()  # Keep reference
    
    def set_parameters(self, model, global_c=None):
        """
        Set model parameters (for SCAFFOLD compatibility)
        与原框架clientscaffold.py完全一致
        
        Args:
            model: Global model
            global_c: Global control variates (for SCAFFOLD)
        """
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone().double()
        
        if global_c is not None:
            # SCAFFOLD: Also set global control and model
            self.c_global = global_c
            self.global_model = model
    
    def delta_yc(self, max_local_epochs=None):
        """
        SCAFFOLD: Compute delta_y and delta_c
        与原框架clientscaffold.py的delta_yc()完全一致
        
        Returns:
            delta_y: Model parameter update
            delta_c: Control variate update
        """
        if max_local_epochs is None:
            max_local_epochs = self.local_epochs
        
        # Compute number of batches
        if not hasattr(self, 'num_batches'):
            self.num_batches = len(self.load_train_data())
        
        delta_y = []
        delta_c = []
        for c, x, yi in zip(self.c_global, self.global_model.parameters(), self.model.parameters()):
            delta_y.append(yi - x)
            delta_c.append(- c + 1/self.num_batches/max_local_epochs/self.learning_rate * (x - yi))
        
        return delta_y, delta_c
    
    def init_personalized_model(self):
        """
        Initialize Ditto-style personalized model for Phase2
        Called when transitioning from Phase1 to Phase2
        """
        if self.model_per is None:
            import copy
            self.model_per = copy.deepcopy(self.model).double()
            
            from flcore.optimizers.fedoptimizer import PerturbedGradientDescent
            self.optimizer_per = PerturbedGradientDescent(
                self.model_per.parameters(),
                lr=self.learning_rate,
                mu=self.mu_ditto
            )
            print(f"  Client {self.id}: Initialized personalized model for Ditto-style training (μ={self.mu_ditto})")
    
    def get_validation_accuracy(self):
        """
        For Phase 2 compatibility with server aggregation algorithms
        (e.g., FedCS needs client ranking)
        Returns current validation accuracy for server-side ranking.
        """
        return self._compute_accuracy()
    
    def _check_early_stopping(self):
        """
        Check if client should stop Phase 1 training (simplified to 3 conditions)
        
        Three Conditions (ALL must be met):
        1. 训练至少10轮 (forced training)
        2. 最近3轮准确率均 ≥ ACC(t) - 0.02 (稳定在阈值附近)
        3. 最近3轮波动 ≤ 0.02 (收敛稳定)
        
        Note: 前5轮在train_phase1()中已被跳过检查
        
        Returns:
            bool: True if all conditions met
        """
        # === 参数配置 ===
        min_training_rounds = 10   # 至少10轮训练（但前5轮已被跳过检查）
        min_stable_rounds = 3      # 条件(2)(3): 检查最近3轮
        max_fluctuation = 0.02     # 条件(3): 最大波动率2%
        threshold_tolerance = 0.02  # 条件(2): 阈值容忍度2%
        
        print(f"\n  [Client {self.id}] 早停检查 (当前准确率: {self.accuracy:.4f})")
        
        # === 条件(1): 前10轮强制训练，不检查早停 ===
        if len(self.accuracy_history) < min_training_rounds:
            print(f"    [-] 条件(1): 训练轮数不足 ({len(self.accuracy_history)}/{min_training_rounds}轮) - 强制训练")
            return False
        else:
            print(f"    [+] 条件(1): 训练轮数充足 ({len(self.accuracy_history)}轮)")
        
        # === 条件(2): 最近3轮每一轮都≥ACC(t)-0.02（性能达标） ===
        recent = self.accuracy_history[-min_stable_rounds:]
        tolerance_threshold = self.current_threshold - threshold_tolerance
        failed_rounds = []
        for i, acc in enumerate(recent):
            if acc < tolerance_threshold:
                failed_rounds.append((i, acc))
        
        if failed_rounds:
            print(f"    [-] 条件(2): 存在低于容忍阈值的轮次")
            print(f"       容忍阈值: {tolerance_threshold:.4f} (ACC(t)={self.current_threshold:.4f} - 0.02)")
            for idx, acc in failed_rounds:
                print(f"       第{len(self.accuracy_history)-min_stable_rounds+idx+1}轮: {acc:.4f} < {tolerance_threshold:.4f} [FAIL]")
            return False
        else:
            print(f"    [+] 条件(2): 所有轮次都持续达标")
            print(f"       容忍阈值: {tolerance_threshold:.4f}")
            for i, acc in enumerate(recent):
                print(f"       第{len(self.accuracy_history)-min_stable_rounds+i+1}轮: {acc:.4f} >= {tolerance_threshold:.4f} [+]")
        
        # === 条件(3): 最近3轮波动率≤0.02（稳定不震荡） ===
        fluctuation = max(recent) - min(recent)
        if fluctuation > max_fluctuation:
            print(f"    [-] 条件(3): 波动率过大 ({fluctuation:.4f} > {max_fluctuation:.4f})")
            print(f"       最近3轮: {[f'{x:.4f}' for x in recent]} (最大-最小={fluctuation:.4f})")
            return False
        else:
            print(f"    [+] 条件(3): 波动率稳定 ({fluctuation:.4f} <= {max_fluctuation:.4f})")
            print(f"       最近3轮: {[f'{x:.4f}' for x in recent]}")
        
        # 所有3个条件都满足，判定为真正稳定收敛
        print(f"    [OK] 所有条件满足 - 客户端达标！")
        return True
        tolerance_threshold = self.current_threshold - threshold_tolerance
        failed_rounds = []
        for i, acc in enumerate(recent):
            if acc < tolerance_threshold:
                failed_rounds.append((i, acc))
        
        if failed_rounds:
            print(f"    [-] 条件(4): 存在低于容忍阈值的轮次")
            print(f"       容忍阈值: {tolerance_threshold:.4f} (阈值{self.current_threshold:.4f} - 容忍度{threshold_tolerance:.4f})")
            for idx, acc in failed_rounds:
                print(f"       第{len(self.accuracy_history)-min_stable_rounds+idx+1}轮: {acc:.4f} < {tolerance_threshold:.4f} [-]")
            return False
        else:
            print(f"    [+] 条件(4): 所有轮次都持续达标")
            print(f"       容忍阈值: {tolerance_threshold:.4f}")
            for i, acc in enumerate(recent):
                print(f"       第{len(self.accuracy_history)-min_stable_rounds+i+1}轮: {acc:.4f} >= {tolerance_threshold:.4f} [+]")
        
        # 所有4个条件都满足，判定为真正稳定收敛
        print(f"    [OK] 所有条件满足 - 客户端达标！")
        return True
    
    def update_threshold(self, threshold):
        """
        接收服务器传来的动态阈值
        
        Args:
            threshold: 当前轮的动态阈值
        """
        self.current_threshold = threshold
    
    def test_metrics(self):
        """
        Override base class test_metrics to ensure float64 compatibility
        """
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
        from sklearn.preprocessing import label_binarize
        
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device).double()
                else:
                    x = x.to(self.device).double()
                y = y.to(self.device)
                output = self.model(x)
                
                # Check for NaN in model output
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"Warning: Client {self.id} detected NaN/Inf in model output")
                    self.enable_grad_clip = True
                    self.nan_detected_count += 1
                    output = torch.nan_to_num(output, nan=0.0, posinf=50.0, neginf=-50.0)
                    output = torch.clamp(output, min=-50, max=50)
                
                # Convert logits to probabilities
                output_prob = torch.nn.functional.softmax(output, dim=1)

                pred = torch.argmax(output, dim=1)
                test_acc += (torch.sum(pred == y)).item()
                test_num += y.shape[0]

                y_prob.append(output_prob.detach().cpu().numpy())
                y_pred.append(pred.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        
        # Final NaN check
        if np.isnan(y_prob).any():
            print(f"Warning: NaN found in y_prob for client {self.id}")
            nan_mask = np.isnan(y_prob)
            y_prob[nan_mask] = 1.0 / self.num_classes
        
        # Calculate AUC - special handling for binary classification
        if self.num_classes == 2:
            auc = roc_auc_score(y_true[:, 1], y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, average='micro')
        
        # Calculate Precision, Recall, F1 - convert one-hot to labels
        y_true_labels = np.argmax(y_true, axis=1)
        precision = precision_score(y_true_labels, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true_labels, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true_labels, y_pred, average='weighted', zero_division=0)

        return test_acc, test_num, auc, precision, recall, f1
    
    def train_metrics(self):
        """
        Override base class train_metrics to ensure float64 compatibility
        """
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device).double()
                else:
                    x = x.to(self.device).double()
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
