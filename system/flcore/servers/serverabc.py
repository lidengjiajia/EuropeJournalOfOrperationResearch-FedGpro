# FedABC: Federated Artificial Bee Colony
# Based on: Karaboga, D., & Basturk, B. (2007). 
# A powerful and efficient algorithm for numerical function optimization: 
# artificial bee colony (ABC) algorithm. Journal of global optimization, 39(3), 459-471.
# Citations: 10,000+ (Google Scholar)

import time
import numpy as np
from flcore.clients.clientabc import clientABC
from flcore.servers.serverbase import Server
from threading import Thread


class FedABC(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # 选择慢速客户端
        self.set_slow_clients()
        self.set_clients(clientABC)

        print(f"\n加入联邦学习训练的客户端数量 / 总客户端数量: {self.num_join_clients} / {self.num_clients}")
        print("训练完成后, 所有 {} 个客户端的模型参数将被保存.\n".format(self.num_clients))
        
        # ABC特定参数（基于原始论文）
        self.num_employed_bees = self.num_join_clients  # 雇佣蜂数量 = 蜜源数量
        self.num_onlooker_bees = self.num_join_clients  # 观察蜂数量
        self.limit = 10  # 放弃阈值（论文推荐 SN*D，简化为固定值）
        
        # 记录最优解
        self.best_client_id = None
        self.best_fitness = float('-inf')  # 注意：fitness越大越好（负损失）
        self.best_model = None
        
        # 适应度历史
        self.fitness_history = []

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------第 {i}轮 全局训练-------------")
                print("\n评估全局模型")
                self.evaluate()

            # === 雇佣蜂阶段 ===
            print("🐝 雇佣蜂阶段...")
            for client in self.selected_clients:
                # 随机选择一个不同的客户端作为邻居
                neighbor = np.random.choice([c for c in self.selected_clients if c.id != client.id])
                client.set_neighbor_source(neighbor.model.parameters())
                
                # 训练
                client.train()
                # 雇佣蜂搜索
                client.employed_bee_phase()
            
            # 评估所有客户端的适应度
            for client in self.selected_clients:
                client.evaluate_fitness()
            
            # 计算选择概率
            fitness_values = np.array([max(c.fitness, 1e-10) for c in self.selected_clients])
            fitness_sum = np.sum(fitness_values)
            probabilities = fitness_values / fitness_sum if fitness_sum > 0 else \
                           np.ones(len(self.selected_clients)) / len(self.selected_clients)
            
            # === 观察蜂阶段 ===
            print("👀 观察蜂阶段...")
            for client, prob in zip(self.selected_clients, probabilities):
                client.onlooker_bee_phase(prob)
            
            # 重新评估适应度
            for client in self.selected_clients:
                client.evaluate_fitness()
            
            # === 侦查蜂阶段 ===
            print("🔍 侦查蜂阶段...")
            for client in self.selected_clients:
                client.scout_bee_phase()
            
            # 找到当前最优客户端
            current_best_client = max(self.selected_clients, 
                                    key=lambda c: c.fitness)
            
            # 更新全局最优
            if current_best_client.fitness > self.best_fitness:
                self.best_fitness = current_best_client.fitness
                self.best_client_id = current_best_client.id
                self.best_model = [param.clone().detach() 
                                 for param in current_best_client.model.parameters()]
                print(f"✨ 发现更优解！客户端 {self.best_client_id}, "
                      f"适应度: {self.best_fitness:.4f} (损失: {-self.best_fitness:.4f})")
            
            # 聚合：使用最优模型作为全局模型
            if self.best_model is not None:
                for global_param, best_param in zip(self.global_model.parameters(), 
                                                   self.best_model):
                    global_param.data = best_param.data.clone()

            # 记录适应度统计
            fitness_values = [c.fitness for c in self.selected_clients]
            self.fitness_history.append({
                'round': i,
                'best': max(fitness_values),
                'worst': min(fitness_values),
                'mean': np.mean(fitness_values),
                'std': np.std(fitness_values)
            })

            self.Budget.append(time.time() - s_t)
            print('-'*50, f"耗时: {self.Budget[-1]:.2f}s")
            print(f"当前轮最优适应度: {max(fitness_values):.4f}, "
                  f"平均适应度: {np.mean(fitness_values):.4f}")

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\n" + "="*70)
        print("🏆 ABC优化完成")
        print("="*70)
        print(f"最优客户端: {self.best_client_id}")
        print(f"最优适应度: {self.best_fitness:.4f} (对应损失: {-self.best_fitness:.4f})")
        print("\n总预算 (s):", sum(self.Budget))
        print(f"{self.num_clients} 客户端总训练所有轮次花费时间:")
        time_cost = sum([c.train_time_cost['total_cost'] / c.train_time_cost['num_rounds'] 
                        for c in self.clients])
        print(f"总时间成本 {time_cost:.2f}s 平均每轮 {time_cost / self.global_rounds:.2f}s")
        print("="*70)

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
