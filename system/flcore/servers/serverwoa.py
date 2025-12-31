# FedWOA: Federated Whale Optimization Algorithm
# Based on: Mirjalili, S., & Lewis, A. (2016). 
# The whale optimization algorithm. Advances in engineering software, 95, 51-67.
# Citations: 12,000+ (Google Scholar)

import time
import numpy as np
from flcore.clients.clientwoa import clientWOA
from flcore.servers.serverbase import Server
from threading import Thread


class FedWOA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # 选择慢速客户端
        self.set_slow_clients()
        self.set_clients(clientWOA)

        print(f"\n加入联邦学习训练的客户端数量 / 总客户端数量: {self.num_join_clients} / {self.num_clients}")
        print("训练完成后, 所有 {} 个客户端的模型参数将被保存.\n".format(self.num_clients))
        
        # WOA特定参数（基于原始论文）
        self.a_max = 2.0  # a的初始值
        self.a_min = 0.0  # a的最终值
        self.current_a = self.a_max
        
        # 记录最优客户端
        self.best_client_id = None
        self.best_fitness = float('inf')
        self.best_model = None
        
        # 记录所有客户端的适应度
        self.fitness_history = []

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            
            # 更新WOA参数 a: 线性从2递减到0
            # a = 2 - t * (2 / max_iter)
            self.current_a = self.a_max - i * (self.a_max - self.a_min) / self.global_rounds
            
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------第 {i}轮 全局训练-------------")
                print(f"WOA参数 a = {self.current_a:.4f}")
                print("\n评估全局模型")
                self.evaluate()

            # 让客户端评估自己的适应度
            for client in self.selected_clients:
                client.evaluate_fitness()
            
            # 找到当前最优客户端
            current_best_client = min(self.selected_clients, 
                                    key=lambda c: c.current_fitness)
            
            # 更新全局最优
            if current_best_client.current_fitness < self.best_fitness:
                self.best_fitness = current_best_client.current_fitness
                self.best_client_id = current_best_client.id
                self.best_model = [param.clone().detach() 
                                 for param in current_best_client.model.parameters()]
                print(f"✨ 发现更优解！客户端 {self.best_client_id}, 适应度: {self.best_fitness:.4f}")
            
            # 传递最优位置给所有客户端
            for client in self.selected_clients:
                if self.best_model is not None:
                    client.set_woa_parameters(
                        self.current_a, 
                        self.best_model,
                        i,
                        self.global_rounds
                    )

            # 客户端训练
            for client in self.selected_clients:
                client.train()

            # 接收模型但不直接平均聚合
            # WOA通过位置更新机制自然产生聚合效果
            if i % self.eval_gap == 0:
                self.receive_models()
                # 使用最优客户端模型作为全局模型
                if self.best_model is not None:
                    for global_param, best_param in zip(self.global_model.parameters(), 
                                                       self.best_model):
                        global_param.data = best_param.data.clone()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\n最优客户端:", self.best_client_id)
        print("最优适应度:", self.best_fitness)
        print("\n总预算 (s):", self.Budget)
        print(self.num_clients, "客户端总训练所有轮次花费时间:")
        time_cost = sum([c.train_time_cost['total_cost'] / c.train_time_cost['num_rounds'] 
                        for c in self.clients])
        print(f"总时间成本 {time_cost:.2f}s 平均每轮 {time_cost / self.global_rounds:.2f}s")

        self.save_results()
        self.save_global_model()

    def send_models(self):
        """
        发送全局模型给选中的客户端
        """
        assert (len(self.selected_clients) > 0)

        for client in self.selected_clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        """
        接收客户端模型
        WOA通过位置更新自然聚合，这里主要用于收集信息
        """
        assert (len(self.selected_clients) > 0)

        active_clients = []
        
        for client in self.selected_clients:
            # 检查客户端是否掉线
            if client.train_slow:
                continue
            active_clients.append(client)
        
        # 记录适应度历史
        fitness_values = [c.current_fitness for c in active_clients]
        self.fitness_history.append({
            'round': len(self.fitness_history),
            'best': min(fitness_values),
            'worst': max(fitness_values),
            'mean': np.mean(fitness_values),
            'std': np.std(fitness_values)
        })
        
        return active_clients

    def add_parameters(self, w, client_model):
        """
        辅助函数：参数累加
        """
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w
