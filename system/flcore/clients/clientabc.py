# FedABC: Federated Artificial Bee Colony
# Based on: Karaboga, D., & Basturk, B. (2007). 
# A powerful and efficient algorithm for numerical function optimization: 
# artificial bee colony (ABC) algorithm. Journal of global optimization, 39(3), 459-471.
# Citations: 10,000+ (Google Scholar)

import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientABC(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # ABC算法参数（源自原始论文）
        self.food_source = None  # 当前蜜源位置（客户端模型）
        self.fitness = float('inf')  # 适应度值（负损失）
        self.trial_counter = 0  # 尝试计数器
        self.limit = 10  # 放弃蜜源的阈值（论文中一般为SN*D，这里简化）
        
        # 蜂群角色
        self.role = 'employed'  # 'employed', 'onlooker', 'scout'
        
        # 邻居蜜源（用于生成新解）
        self.neighbor_source = None

    def train(self):
        """
        ABC客户端训练：包括雇佣蜂、观察蜂和侦查蜂阶段
        """
        trainloader = self.load_train_data()
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        
        # 标准训练过程
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

    def employed_bee_phase(self):
        """
        雇佣蜂阶段：在当前蜜源附近搜索
        公式：v_ij = x_ij + φ_ij(x_ij - x_kj)
        其中 φ_ij ∈ [-1, 1]
        """
        if self.neighbor_source is None:
            return
        
        # 生成新候选解
        new_model = []
        phi = np.random.uniform(-1, 1)  # φ参数
        
        with torch.no_grad():
            for param, neighbor_param in zip(self.model.parameters(), 
                                            self.neighbor_source):
                # v = x + φ(x - x_k)
                new_param = param + phi * (param - neighbor_param)
                new_model.append(new_param.clone())
        
        # 评估新解
        new_fitness = self.evaluate_fitness_with_params(new_model)
        
        # 贪婪选择（如果新解更好则接受）
        if new_fitness < self.fitness:
            for param, new_param in zip(self.model.parameters(), new_model):
                param.data = new_param.data.clone()
            self.fitness = new_fitness
            self.trial_counter = 0  # 重置计数器
        else:
            self.trial_counter += 1  # 增加失败计数

    def onlooker_bee_phase(self, probability):
        """
        观察蜂阶段：根据概率选择蜜源
        选择概率：P_i = fitness_i / Σfitness
        """
        if np.random.rand() < probability:
            # 被选中的观察蜂执行与雇佣蜂相同的搜索
            self.employed_bee_phase()

    def scout_bee_phase(self):
        """
        侦查蜂阶段：如果蜜源超过limit次未改进，则放弃并随机生成新蜜源
        """
        if self.trial_counter >= self.limit:
            # 随机初始化新位置
            with torch.no_grad():
                for param in self.model.parameters():
                    # 在当前位置附近随机搜索
                    param.data += torch.randn_like(param) * 0.1
            
            self.trial_counter = 0
            self.fitness = self.evaluate_fitness()
            print(f"🔍 侦查蜂 {self.id} 发现新蜜源，适应度: {self.fitness:.4f}")

    def set_neighbor_source(self, neighbor_model):
        """
        设置邻居蜜源（由服务器随机分配）
        """
        self.neighbor_source = [param.clone().detach() for param in neighbor_model]

    def evaluate_fitness(self):
        """
        评估当前模型的适应度
        适应度定义：fitness = 1/(1+loss) 或 -loss
        这里使用负损失值，损失越小适应度越高
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
        
        avg_loss = total_loss / max(num_batches, 1)
        self.fitness = -avg_loss  # 负损失作为适应度（最大化问题）
        return self.fitness

    def evaluate_fitness_with_params(self, model_params):
        """
        评估给定参数的适应度（不修改当前模型）
        """
        # 临时保存当前参数
        original_params = [param.clone() for param in self.model.parameters()]
        
        # 设置新参数
        with torch.no_grad():
            for param, new_param in zip(self.model.parameters(), model_params):
                param.data = new_param.data.clone()
        
        # 评估
        fitness = self.evaluate_fitness()
        
        # 恢复原始参数
        with torch.no_grad():
            for param, orig_param in zip(self.model.parameters(), original_params):
                param.data = orig_param.data.clone()
        
        return fitness

    def get_fitness_value(self):
        """
        返回当前适应度（用于计算选择概率）
        """
        return self.fitness
