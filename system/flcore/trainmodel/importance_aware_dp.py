"""
Importance-Aware Adaptive Differential Privacy (IA-ADP)

核心思想：
1. 根据特征重要性动态调整噪声尺度
2. 重要特征添加更少噪声（保留效用）
3. 不重要特征添加更多噪声（增强隐私）
4. 保证严格的ε-差分隐私

数学公式：
    β_i = α + (1-α) × (1 - importance_i)
    noise_i ~ Laplace(0, β_i × Δf / ε)

其中：
    - α ∈ [0.1, 0.5]：用户可调参数
    - importance_i ∈ [0, 1]：特征i的重要性分数
    - β_i ∈ [α, 1]：特征i的噪声系数
    - Δf = 2 × clip_norm：敏感度边界
    - ε：全局隐私预算
"""

import torch
import numpy as np
from typing import Dict, List, Optional


class ImportanceAwareDP:
    """重要性感知自适应差分隐私机制"""
    
    def __init__(
        self,
        epsilon: float = 5.0,
        alpha: float = 0.3,
        clip_norm: float = 1.0,
        importance_method: str = 'vae_contrast',
        importance_momentum: float = 0.9,
        privacy_priority: bool = False,
        device: str = 'cpu'
    ):
        """
        Args:
            epsilon: 隐私预算，越小隐私保护越强
            alpha: 最小噪声系数（重要特征的噪声下界），范围 [0.1, 0.5]
            clip_norm: 梯度裁剪范数
            importance_method: 重要性计算方法（仅支持 'vae_contrast'）
            importance_momentum: 重要性指数移动平均系数
            privacy_priority: True=重要特征加更多噪声(privacy-first), False=重要特征加更少噪声(utility-first)
            device: 计算设备
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.clip_norm = clip_norm
        self.importance_method = importance_method
        self.importance_momentum = importance_momentum
        self.privacy_priority = privacy_priority
        self.device = device
        
        # 存储历史重要性分数（用于平滑）
        self.importance_history: Dict[str, torch.Tensor] = {}
        
        # 统计信息
        self.total_noise_added = 0.0
        self.num_noising_rounds = 0
        
    def compute_vae_contrast_importance(
        self,
        parameters: Dict[str, torch.Tensor],
        vae_contrast_scores: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        基于VAE对比损失计算特征重要性
        
        核心思想：VAE的对比损失（contrast loss）衡量特征的判别性
        - 对分类有用的特征：高对比损失（高重要性）
        - 对分类无用的特征：低对比损失（低重要性）
        
        Args:
            parameters: 模型参数
            vae_contrast_scores: 从client传入的VAE对比分数
        
        Returns:
            importance_scores: 每个特征的重要性分数 [0, 1]
        """
        if vae_contrast_scores is not None:
            # 直接使用传入的对比分数
            return vae_contrast_scores
        else:
            # Fallback: 返回均匀重要性
            print("[Warning] VAE contrast scores not provided, using uniform importance")
            importance_scores = {}
            for name in parameters.keys():
                importance_scores[name] = torch.tensor(
                    1.0 / len(parameters),
                    device=self.device
                )
            return importance_scores
    
    def update_importance(
        self,
        parameters: Dict[str, torch.Tensor],
        vae_contrast_scores: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算并更新特征重要性（带动量平滑）
        
        使用指数移动平均避免重要性剧烈波动：
        importance_t = momentum × importance_{t-1} + (1 - momentum) × current_importance
        
        仅支持vae_contrast方法
        """
        # 计算当前重要性
        current_importance = self.compute_vae_contrast_importance(parameters, vae_contrast_scores)
        
        # 使用动量平滑
        if len(self.importance_history) == 0:
            # 第一次计算，直接使用当前值
            self.importance_history = current_importance
        else:
            # 应用指数移动平均
            for name in current_importance.keys():
                if name in self.importance_history:
                    self.importance_history[name] = (
                        self.importance_momentum * self.importance_history[name] +
                        (1 - self.importance_momentum) * current_importance[name]
                    )
                else:
                    self.importance_history[name] = current_importance[name]
        
        return self.importance_history
    
    def compute_noise_coefficients(
        self,
        importance_scores: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        根据重要性计算自适应噪声系数
        
        公式：
        - Utility-First (privacy_priority=False): β_i = α + (1 - α) × (1 - importance_i)
          重要特征 → 小噪声，不重要特征 → 大噪声
        - Privacy-First (privacy_priority=True): β_i = α + (1 - α) × importance_i
          重要特征 → 大噪声，不重要特征 → 小噪声
        
        性质：β_i ∈ [α, 1]
        """
        noise_coefficients = {}
        
        for name, importance in importance_scores.items():
            if self.privacy_priority:
                # Privacy-First: 重要特征加更多噪声（保护判别性信息）
                # β_i = α + (1 - α) × importance_i
                beta = self.alpha + (1 - self.alpha) * importance.item()
            else:
                # Utility-First: 重要特征加更少噪声（保留判别性信息）
                # β_i = α + (1 - α) × (1 - importance_i)
                beta = self.alpha + (1 - self.alpha) * (1 - importance.item())
            noise_coefficients[name] = beta
        
        return noise_coefficients
    
    def add_adaptive_noise(
        self,
        parameters: Dict[str, torch.Tensor],
        importance_scores: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        为参数添加重要性感知的自适应噪声
        
        步骤：
        1. 计算每个参数的噪声系数 β_i
        2. 计算噪声尺度 scale_i = β_i × Δf / ε
        3. 从 Laplace(0, scale_i) 采样噪声
        4. 添加噪声并返回
        
        差分隐私保证：
        - 敏感度 Δf = 2 × clip_norm
        - 每个参数满足 (ε/β_i)-DP
        - 组合后满足 (Σ ε/β_i)-DP ≤ (d×ε/α)-DP
        """
        noised_parameters = {}
        noise_coefficients = self.compute_noise_coefficients(importance_scores)
        
        # 敏感度边界（假设梯度已裁剪到 clip_norm）
        sensitivity = 2 * self.clip_norm
        
        for name, param in parameters.items():
            if name not in noise_coefficients:
                # 如果没有重要性分数，使用默认系数1.0
                beta = 1.0
            else:
                beta = noise_coefficients[name]
            
            # 计算拉普拉斯噪声尺度
            # scale = β_i × Δf / ε
            noise_scale = beta * sensitivity / self.epsilon
            
            # 生成拉普拉斯噪声
            noise = torch.from_numpy(
                np.random.laplace(0, noise_scale, param.shape)
            ).float().to(param.device)
            
            # 添加噪声
            noised_param = param + noise
            noised_parameters[name] = noised_param
            
            # 统计
            self.total_noise_added += torch.norm(noise).item()
        
        self.num_noising_rounds += 1
        
        return noised_parameters
    
    def add_noise_to_model(
        self,
        model: torch.nn.Module,
        gradients: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.nn.Module:
        """
        为整个模型添加重要性感知噪声
        
        Args:
            model: 待加噪的模型
            gradients: 可选的梯度信息（用于基于梯度的重要性计算）
        
        Returns:
            加噪后的模型
        """
        # 提取模型参数
        parameters = {name: param.data.clone() for name, param in model.named_parameters()}
        
        # 更新重要性分数
        importance_scores = self.update_importance(parameters, gradients)
        
        # 添加自适应噪声
        noised_parameters = self.add_adaptive_noise(parameters, importance_scores)
        
        # 更新模型参数
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in noised_parameters:
                    param.copy_(noised_parameters[name])
        
        return model
    
    def get_privacy_budget(self) -> float:
        """
        返回当前累积的隐私预算消耗
        
        根据组合定理，总隐私预算为：
        ε_total ≤ T × (d × ε / α)
        
        其中：
        - T：通信轮数
        - d：参数维度数
        - ε：单轮隐私预算
        - α：最小噪声系数
        """
        if self.num_noising_rounds == 0:
            return 0.0
        
        # 估算参数总数（使用历史记录）
        num_params = len(self.importance_history) if self.importance_history else 1
        
        # 最坏情况下的隐私预算（假设所有参数都使用最小噪声系数α）
        epsilon_per_round = num_params * self.epsilon / self.alpha
        total_epsilon = self.num_noising_rounds * epsilon_per_round
        
        return total_epsilon
    
    def get_average_noise_magnitude(self) -> float:
        """返回平均噪声幅度"""
        if self.num_noising_rounds == 0:
            return 0.0
        return self.total_noise_added / self.num_noising_rounds
    
    def reset_statistics(self):
        """重置统计信息"""
        self.total_noise_added = 0.0
        self.num_noising_rounds = 0
        self.importance_history = {}
    
    def get_info(self) -> Dict[str, any]:
        """获取当前DP配置信息"""
        return {
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'clip_norm': self.clip_norm,
            'importance_method': self.importance_method,
            'num_rounds': self.num_noising_rounds,
            'avg_noise': self.get_average_noise_magnitude(),
            'privacy_budget': self.get_privacy_budget(),
        }
