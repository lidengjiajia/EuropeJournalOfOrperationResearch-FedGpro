import time
import numpy as np
import torch
from flcore.clients.clientfa import clientFA
from flcore.servers.serverbase import Server
from threading import Thread


class FedFA(Server):
    """
    FedFA Server: Federated Focal Average
    
    Server-side implementation for FedFA algorithm.
    Supports difficulty-aware adaptive aggregation where clients
    with harder samples get higher weights.
    
    Category: Traditional FL (tFL) - Loss-based Enhancement
    """
    
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # FedFA parameters
        if not hasattr(args, 'fa_alpha'):
            args.fa_alpha = 0.25
        if not hasattr(args, 'fa_gamma'):
            args.fa_gamma = 2.0
        if not hasattr(args, 'fa_adaptive_weight'):
            args.fa_adaptive_weight = True
        if not hasattr(args, 'fa_weight_momentum'):
            args.fa_weight_momentum = 0.9
        
        # Difficulty-aware aggregation
        self.use_difficulty_weighting = args.fa_use_difficulty_weighting if hasattr(args, 'fa_use_difficulty_weighting') else True
        
        # Set up clients
        self.set_slow_clients()
        self.set_clients(clientFA)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"FedFA Parameters: alpha={args.fa_alpha}, gamma={args.fa_gamma}, adaptive_weight={args.fa_adaptive_weight}")
        print(f"Difficulty-aware aggregation: {self.use_difficulty_weighting}")
        print("Finished creating server and clients.")

        self.Budget = []
        
        # Track client difficulties
        self.client_difficulties = {}

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
                
                # Print difficulty statistics
                if self.client_difficulties:
                    avg_difficulty = np.mean(list(self.client_difficulties.values()))
                    print(f"Average client difficulty: {avg_difficulty:.4f}")

            # Train selected clients
            for client in self.selected_clients:
                client.train()
                
                # Collect difficulty information
                if hasattr(client, 'get_sample_difficulty'):
                    difficulty = client.get_sample_difficulty()
                    self.client_difficulties[client.id] = difficulty

            # Receive models with optional difficulty-based weighting
            if self.use_difficulty_weighting:
                self.receive_models_with_difficulty()
            else:
                self.receive_models()
            
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            
            # Aggregate with weighted averaging
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientFA)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def receive_models_with_difficulty(self):
        """
        Receive models and adjust aggregation weights based on client difficulties.
        Clients with harder samples (higher difficulty) get higher weights.
        """
        assert (len(self.selected_clients) > 0)

        import random
        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        
        # Collect models and difficulties
        client_data = []
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
                
            if client_time_cost <= self.time_threthold:
                difficulty = self.client_difficulties.get(client.id, 1.0)
                client_data.append({
                    'id': client.id,
                    'samples': client.train_samples,
                    'model': client.model,
                    'difficulty': difficulty
                })
        
        if len(client_data) == 0:
            return
        
        # Compute difficulty-aware weights
        # weight = samples * sqrt(difficulty)
        # Intuition: clients with harder samples should have more influence
        total_weighted = 0.0
        for data in client_data:
            # Use square root to avoid over-emphasizing difficulty
            difficulty_factor = np.sqrt(max(data['difficulty'], 0.1))
            weighted_samples = data['samples'] * difficulty_factor
            total_weighted += weighted_samples
        
        # Normalize and store
        for data in client_data:
            difficulty_factor = np.sqrt(max(data['difficulty'], 0.1))
            weighted_samples = data['samples'] * difficulty_factor
            weight = weighted_samples / total_weighted
            
            self.uploaded_ids.append(data['id'])
            self.uploaded_weights.append(weight)
            self.uploaded_models.append(data['model'])
        
        # Print weight distribution
        print(f"Aggregation weights (difficulty-aware): {[f'{w:.3f}' for w in self.uploaded_weights]}")
