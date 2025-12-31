# FedTLBO: Federated Teaching-Learning-Based Optimization
# Server implementation for Teaching-Learning-Based Optimization

import time
import numpy as np
import copy
from flcore.clients.clienttlbo import clientTLBO
from flcore.servers.serverbase import Server
from threading import Thread


class FedTLBO(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # 选择客户端
        self.set_slow_clients()
        self.set_clients(clientTLBO)

        print(f"\n加入 {self.num_clients} 个客户端!")
        print("=" * 50)
        print(f"算法: FedTLBO (Federated Teaching-Learning-Based Optimization)")
        print(f"总轮数: {self.global_rounds}")
        print(f"每轮参与客户端数: {self.num_join_clients}")
        print(f"客户端本地训练轮数: {self.local_epochs}")
        print(f"客户端学习率: {self.learning_rate}")
        print(f"TF策略: 每轮随机选择 {1, 2} (原始TLBO)")
        print(f"准确率评估: 完整验证集评估")
        print(f"伙伴选择: 从所有 {self.num_clients} 个客户端随机选择")
        print("=" * 50)
        
        self.Budget = []
        
        # TLBO相关记录
        self.client_accuracies = {}  # 记录每个客户端的准确率
        self.best_client_id = None  # 最优客户端ID
        self.best_client_acc = 0.0  # 最优客户端准确率
        
        self.Budget = []  # 记录每轮的时间开销

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            
            # Step 1: 发送全局模型给所有参与客户端
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n------------- Round {i} -------------")
                print("评估全局模型...")
                self.evaluate()

            # Step 2: 所有参与客户端评估当前准确率(在Teacher phase之前)
            print(f"Round {i}: 收集客户端准确率...")
            for client in self.selected_clients:
                client.current_acc = client.get_validation_accuracy()
                self.client_accuracies[client.id] = client.current_acc
            
            # Step 3: 找出最优客户端
            self.find_best_client()
            
            # Step 4: 计算平均模型（用于Teacher phase）
            mean_model = self.compute_mean_model()
            
            # Step 5: 随机生成TF ∈ {1, 2}
            TF = np.random.choice([1, 2])
            print(f"Round {i}: TF={TF}, 最优客户端ID={self.best_client_id}, 准确率={self.best_client_acc:.4f}")
            
            # Step 6: 为每个客户端分配TLBO参数并训练
            for client in self.selected_clients:
                # 6.1 随机选择伙伴客户端u（从所有n个客户端中选择，排除自己）
                available_partners = [c for c in self.clients if c.id != client.id]
                partner_client = np.random.choice(available_partners)
                partner_acc = self.client_accuracies.get(partner_client.id, 0.0)
                
                # 6.2 生成随机数 r1, r2 ∈ [0, 1]
                r_teacher = np.random.uniform(0, 1)
                r_student = np.random.uniform(0, 1)
                
                # 6.3 获取最优客户端的模型
                best_client = self.clients[self.best_client_id]
                best_model = copy.deepcopy(best_client.model)
                
                # 6.4 获取伙伴客户端的模型
                partner_model = copy.deepcopy(partner_client.model)
                
                # 6.5 设置TLBO参数
                client.set_tlbo_params(
                    best_model=best_model,
                    mean_model=copy.deepcopy(mean_model),
                    partner_model=partner_model,
                    partner_acc=partner_acc,
                    TF=TF,
                    r_teacher=r_teacher,
                    r_student=r_student
                )
            
            # Step 7: 执行TLBO训练（Teacher + Student + Local Training）
            print(f"Round {i}: 客户端执行TLBO训练...")
            for client in self.selected_clients:
                client.train()

            # Step 8: 接收客户端模型
            self.receive_models()
            
            # Step 9: 聚合模型（使用标准FedAvg）
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print(f"Round {i} 完成，耗时: {self.Budget[-1]:.2f}s")
            print('-' * 50)

        print("\n训练完成!")
        print("最终评估...")
        self.evaluate()

        self.save_results()
        self.save_global_model()

    def find_best_client(self):
        """
        找出准确率最高的客户端
        """
        best_acc = -1
        best_id = 0
        
        for client_id, acc in self.client_accuracies.items():
            if acc > best_acc:
                best_acc = acc
                best_id = client_id
        
        self.best_client_id = best_id
        self.best_client_acc = best_acc

    def compute_mean_model(self):
        """
        计算所有参与客户端的平均模型 θ̅
        用于Teacher phase
        
        Returns:
            mean_model: 平均模型
        """
        # 创建一个空模型用于存储平均参数
        mean_model = copy.deepcopy(self.selected_clients[0].model)
        
        # 初始化为0
        for param in mean_model.parameters():
            param.data = torch.zeros_like(param.data)
        
        # 累加所有客户端的参数
        for client in self.selected_clients:
            for mean_param, client_param in zip(mean_model.parameters(), client.model.parameters()):
                mean_param.data += client_param.data
        
        # 除以客户端数量得到平均
        num_clients = len(self.selected_clients)
        for param in mean_model.parameters():
            param.data /= num_clients
        
        return mean_model

    def receive_models(self):
        """
        接收客户端模型，并记录准确率
        """
        assert (len(self.selected_clients) > 0)

        active_clients = np.random.choice(self.selected_clients, 
                                         int((1 - self.client_drop_rate) * self.current_num_join_clients), 
                                         replace=False)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        """
        聚合客户端模型参数（标准FedAvg加权平均）
        """
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)


import torch  # 补充导入

