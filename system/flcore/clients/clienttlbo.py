# FedTLBO: Federated Teaching-Learning-Based Optimization
# Teaching-Learning-Based Optimization (TLBO) for Federated Learning
# Original TLBO: Rao, R., Savsani, V., & Vakharia, D. (2011). 
# Teaching-learning-based optimization: a novel method for constrained mechanical design optimization problems.

import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientTLBO(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # 存储从服务器接收的TLBO参数
        self.best_model = None  # θ_best: 最优客户端的模型
        self.mean_model = None  # θ̅: 平均模型
        self.partner_model = None  # θ_u: 随机伙伴客户端的模型
        self.partner_acc = None  # acc_u: 伙伴客户端的准确率
        self.TF = None  # Teaching Factor ∈ {1, 2}
        self.r_teacher = None  # Teacher phase的随机数 r1
        self.r_student = None  # Student phase的随机数 r2
        
        self.current_acc = 0.0  # 当前客户端的准确率

    def train(self):
        trainloader = self.load_train_data()
        
        # Phase 0: 评估当前模型在验证集上的准确率
        self.current_acc = self.get_validation_accuracy()
        
        # Phase 1: Teacher Phase
        # θ' = θ_k + r1 × (θ_best - TF × θ̅)
        if self.best_model is not None and self.mean_model is not None:
            with torch.no_grad():
                for param, best_param, mean_param in zip(
                    self.model.parameters(), 
                    self.best_model.parameters(),
                    self.mean_model.parameters()
                ):
                    # difference_teacher = θ_best - TF × θ̅
                    difference_teacher = best_param.data - self.TF * mean_param.data
                    # θ' = θ_k + r1 × difference_teacher
                    param.data = param.data + self.r_teacher * difference_teacher
        
        # Phase 2: Student Phase (基于准确率比较)
        # if acc_k > acc_u: θ'' = θ' + r2 × (θ' - θ_u)  (好学生远离差学生)
        # else: θ'' = θ' + r2 × (θ_u - θ')  (差学生向好学生学习)
        if self.partner_model is not None and self.partner_acc is not None:
            with torch.no_grad():
                for param, partner_param in zip(
                    self.model.parameters(),
                    self.partner_model.parameters()
                ):
                    if self.current_acc > self.partner_acc:
                        # 当前客户端更优秀，远离伙伴
                        difference_student = param.data - partner_param.data
                    else:
                        # 当前客户端较差，向伙伴学习
                        difference_student = partner_param.data - param.data
                    
                    # θ'' = θ' + r2 × difference_student
                    param.data = param.data + self.r_student * difference_student
        
        # Phase 3: Local Training (完整的E轮本地训练)
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
                self.optimizer.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_tlbo_params(self, best_model, mean_model, partner_model, partner_acc, TF, r_teacher, r_student):
        """
        设置TLBO算法所需的参数
        
        Args:
            best_model: θ_best - 最优客户端的模型
            mean_model: θ̅ - 平均模型
            partner_model: θ_u - 随机伙伴客户端的模型
            partner_acc: acc_u - 伙伴客户端的准确率
            TF: Teaching Factor ∈ {1, 2}
            r_teacher: Teacher phase的随机数 r1 ∈ [0, 1]
            r_student: Student phase的随机数 r2 ∈ [0, 1]
        """
        self.best_model = best_model
        self.mean_model = mean_model
        self.partner_model = partner_model
        self.partner_acc = partner_acc
        self.TF = TF
        self.r_teacher = r_teacher
        self.r_student = r_student

    def get_validation_accuracy(self):
        """
        在完整验证集/测试集上评估模型准确率
        
        Returns:
            accuracy: 准确率 (0~1之间的浮点数)
        """
        self.model.eval()
        
        # 使用测试数据作为验证集
        test_acc = 0
        test_num = 0
        
        with torch.no_grad():
            for x, y in self.load_test_data():
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
        
        accuracy = test_acc / test_num if test_num > 0 else 0.0
        
        return accuracy

    def train_metrics(self):
        """
        训练指标统计
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
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
