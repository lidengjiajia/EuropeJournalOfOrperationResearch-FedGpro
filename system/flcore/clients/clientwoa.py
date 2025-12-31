# FedWOA: Federated Whale Optimization Algorithm
# Based on: Mirjalili, S., & Lewis, A. (2016). 
# The whale optimization algorithm. Advances in engineering software, 95, 51-67.
# Citations: 12,000+ (Google Scholar)

import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientWOA(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # WOA算法参数（源自原始论文）
        self.best_position = None  # X* - 最优鲸鱼位置（最优客户端模型）
        
        # WOA控制参数
        self.a = None  # 线性递减参数 a: 2 -> 0
        self.A = None  # A = 2a·r - a (r是[0,1]随机数)
        self.C = None  # C = 2·r
        self.b = 1     # 螺旋形状常数（论文中b=1）
        self.l = None  # 随机数 [-1, 1]，用于螺旋更新
        
        self.current_fitness = float('inf')  # 当前适应度（损失）

    def train(self):
        """
        WOA客户端训练：包裹猎物、泡泡网攻击、搜索猎物三种行为
        """
        trainloader = self.load_train_data()
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        # WOA位置更新（在收到全局参数后）
        if self.best_position is not None and self.A is not None:
            self.whale_position_update()

    def whale_position_update(self):
        """
        鲸鱼位置更新（融合三种行为）
        基于论文公式：
        1. 包裹猎物：|A| < 1
        2. 螺旋更新：泡泡网攻击
        3. 探索：|A| >= 1
        """
        p = np.random.rand()  # 随机数决定使用哪种策略
        
        with torch.no_grad():
            for param, best_param in zip(self.model.parameters(), self.best_position):
                if abs(self.A) < 1:
                    if p < 0.5:
                        # 包裹猎物（Encircling prey）
                        # D = |C·X* - X|
                        # X(t+1) = X* - A·D
                        D = torch.abs(self.C * best_param - param)
                        param.data = best_param - self.A * D
                    else:
                        # 螺旋更新（Bubble-net attacking）
                        # D' = |X* - X|
                        # X(t+1) = D'·e^(bl)·cos(2πl) + X*
                        D_prime = torch.abs(best_param - param)
                        param.data = D_prime * torch.exp(torch.tensor(self.b * self.l)) * \
                                   torch.cos(torch.tensor(2 * np.pi * self.l)) + best_param
                else:
                    # 搜索猎物（Search for prey）- 探索阶段
                    # 随机选择一个客户端模型作为参考
                    # X(t+1) = X_rand - A·D
                    # 这里使用当前模型加随机扰动来模拟
                    D = torch.abs(self.C * best_param - param)
                    param.data = best_param - self.A * D + torch.randn_like(param) * 0.01

    def set_woa_parameters(self, a, best_position, current_round, max_rounds):
        """
        设置WOA参数（由服务器传入）
        
        Args:
            a: 线性递减参数，从2递减到0
            best_position: 当前最优位置（最优客户端模型）
            current_round: 当前轮次
            max_rounds: 最大轮次
        """
        self.a = a
        
        # A = 2a·r - a，其中r∈[0,1]
        r1 = np.random.rand()
        self.A = 2 * self.a * r1 - self.a
        
        # C = 2·r
        r2 = np.random.rand()
        self.C = 2 * r2
        
        # l ∈ [-1, 1] 用于螺旋更新
        self.l = np.random.uniform(-1, 1)
        
        # 保存最优位置
        self.best_position = [param.clone().detach() for param in best_position]
        
    def evaluate_fitness(self):
        """
        评估当前模型的适应度（使用验证损失）
        """
        self.model.eval()
        trainloader = self.load_train_data()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.model(x)
                loss = self.loss(output, y)
                total_loss += loss.item()
                num_batches += 1
                
        self.current_fitness = total_loss / max(num_batches, 1)
        return self.current_fitness
