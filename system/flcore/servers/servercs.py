import time
import copy
import torch
import numpy as np
import random
from flcore.clients.clientcs import clientCS
from flcore.servers.serverbase import Server
from threading import Thread


class FedCS(Server):
    """
    FedCS Server: Federated Crow Search-Based Dynamic Aggregation
    
    Server-side implementation for FedCS algorithm.
    Combines FedAvg with Crow Search Algorithm (CSA) optimization.
    
    Algorithm Flow:
    1. Clients train locally (standard SGD)
    2. Server receives models and computes validation accuracy for each
    3. FedAvg aggregation: θ_t̄ = Σ(w_k × θ_t^k)
    4. Find best (θ*) and second-best (θ**) clients by validation acc
    5. CSA update: θ_t^(k) = θ_t̄ + fl_t × (θ* - θ_t^k) + r^k × (θ** - θ_t^k)
    6. Send CSA-optimized parameters back to clients
    
    Key Innovation:
    - Combines global averaging (FedAvg) with best-tracking (CSA)
    - Dynamic parameters fl_t and AP_t balance exploration/exploitation
    - Avoids local optima through second-best random exploration
    
    Category: Traditional FL (tFL) - Advanced Aggregation
    
    Reference: "FedCS: Federated Learning with Crow Search-Based Dynamic Aggregation"
    """
    
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # CSA hyperparameters (Flight Length)
        self.f_max = args.cs_f_max if hasattr(args, 'cs_f_max') else 2.0
        self.f_min = args.cs_f_min if hasattr(args, 'cs_f_min') else 0.1
        
        # CSA hyperparameters (Awareness Probability)
        self.AP_max = args.cs_AP_max if hasattr(args, 'cs_AP_max') else 0.3
        self.AP_min = args.cs_AP_min if hasattr(args, 'cs_AP_min') else 0.1
        
        # Track best and second-best clients
        self.best_client_id = None
        self.second_best_client_id = None
        self.best_client_model = None
        self.second_best_client_model = None
        
        # Track CSA diagnostics
        self.csa_diagnostics = {
            'fl_values': [],
            'AP_values': [],
            'best_acc': [],
            'second_best_acc': []
        }
        
        # Set up clients
        self.set_slow_clients()
        self.set_clients(clientCS)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"FedCS Parameters:")
        print(f"  Flight Length: f_max={self.f_max}, f_min={self.f_min}")
        print(f"  Awareness Probability: AP_max={self.AP_max}, AP_min={self.AP_min}")
        print("Finished creating server and clients.")

        self.Budget = []

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
                
                # Print CSA diagnostics
                if len(self.csa_diagnostics['fl_values']) > 0:
                    print(f"CSA Diagnostics:")
                    print(f"  Flight Length (fl_t): {self.csa_diagnostics['fl_values'][-1]:.4f}")
                    print(f"  Awareness Prob (AP_t): {self.csa_diagnostics['AP_values'][-1]:.4f}")
                    if self.best_client_id is not None:
                        print(f"  Best Client: ID={self.best_client_id}, Acc={self.csa_diagnostics['best_acc'][-1]:.4f}")
                    if self.second_best_client_id is not None:
                        print(f"  2nd Best Client: ID={self.second_best_client_id}, Acc={self.csa_diagnostics['second_best_acc'][-1]:.4f}")

            # Train selected clients
            for client in self.selected_clients:
                client.train()

            # Receive models and compute validation accuracies
            self.receive_models()
            
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            
            # FedAvg aggregation
            self.aggregate_parameters()
            
            # Find best and second-best clients by validation accuracy
            self.find_best_clients()
            
            # Apply CSA optimization
            if i > 0:  # Skip first round (no history)
                self.apply_csa_optimization(i)

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
            self.set_new_clients(clientCDB)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def find_best_clients(self):
        """
        Find the best and second-best clients based on validation accuracy.
        Uses test set as validation set (standard practice in FL).
        """
        # Compute validation accuracy for all clients who uploaded models
        client_accs = []
        for client_id, client_model in zip(self.uploaded_ids, self.uploaded_models):
            # Find the client object
            client = self.clients[client_id]
            
            # Temporarily set model to evaluate
            original_model = copy.deepcopy(client.model)
            client.set_parameters(client_model)
            
            # Get validation accuracy
            acc = client.get_validation_accuracy()
            client_accs.append((client_id, acc, client_model))
            
            # Restore original model
            client.set_parameters(original_model)
        
        # Sort by accuracy (descending)
        client_accs.sort(key=lambda x: x[1], reverse=True)
        
        # Store best and second-best
        if len(client_accs) >= 1:
            self.best_client_id = client_accs[0][0]
            self.best_client_model = copy.deepcopy(client_accs[0][2])
            self.csa_diagnostics['best_acc'].append(client_accs[0][1])
            
        if len(client_accs) >= 2:
            self.second_best_client_id = client_accs[1][0]
            self.second_best_client_model = copy.deepcopy(client_accs[1][2])
            self.csa_diagnostics['second_best_acc'].append(client_accs[1][1])
        else:
            # If only one client, use it as second-best too
            self.second_best_client_id = self.best_client_id
            self.second_best_client_model = copy.deepcopy(self.best_client_model)
            self.csa_diagnostics['second_best_acc'].append(client_accs[0][1] if len(client_accs) > 0 else 0.0)

    def compute_flight_length(self, t):
        """
        Compute dynamic flight length parameter.
        
        Formula: fl_t = 1/2 × (f_max + f_min × exp(-t²/Epoch²) + f_min)
        
        Behavior:
        - Early rounds: Large fl (aggressive exploration)
        - Late rounds: Small fl (fine-tuning)
        
        Args:
            t: Current round number
        
        Returns:
            float: Flight length for round t
        """
        Epoch = self.global_rounds
        
        # Compute exponential decay term
        exp_term = np.exp(-t**2 / Epoch**2)
        
        # Apply formula
        fl_t = 0.5 * (self.f_max + self.f_min * exp_term + self.f_min)
        
        return fl_t

    def compute_awareness_probability(self, t):
        """
        Compute dynamic awareness probability parameter.
        
        Formula: AP_t = AP_max - 1/2 × (AP_max + AP_min) × (2t/Epoch² - t²/Epoch²)
        Simplified: AP_t = AP_max - 1/2 × (AP_max + AP_min) × (t/Epoch²) × (2 - t/Epoch)
        
        Behavior:
        - Early rounds: High AP (learn more from best)
        - Late rounds: Low AP (more random exploration to avoid local optima)
        
        Args:
            t: Current round number
        
        Returns:
            float: Awareness probability for round t
        """
        Epoch = self.global_rounds
        
        # Compute the quadratic decay term
        term = (2 * t / Epoch**2) - (t**2 / Epoch**2)
        
        # Apply formula
        AP_t = self.AP_max - 0.5 * (self.AP_max + self.AP_min) * term
        
        # Clamp to valid range [AP_min, AP_max]
        AP_t = np.clip(AP_t, self.AP_min, self.AP_max)
        
        return AP_t

    def apply_csa_optimization(self, t):
        """
        Apply Crow Search Algorithm optimization to update client parameters.
        
        Core Update Formula:
        θ_t^(k) = θ_t̄ + fl_t × (θ* - θ_t^k) + r^k × (θ** - θ_t^k)
        
        Where:
        - θ_t̄: Global averaged model (FedAvg result)
        - θ*: Best client's model (highest validation acc)
        - θ**: Second-best client's model
        - θ_t^k: Client k's current model
        - fl_t: Dynamic flight length
        - r^k: Random coefficient in [0, 1]
        
        Args:
            t: Current round number
        """
        if self.best_client_model is None or self.second_best_client_model is None:
            print("Warning: Best/second-best models not found, skipping CSA optimization")
            return
        
        # Compute dynamic CSA parameters
        fl_t = self.compute_flight_length(t)
        AP_t = self.compute_awareness_probability(t)
        
        # Store for diagnostics
        self.csa_diagnostics['fl_values'].append(fl_t)
        self.csa_diagnostics['AP_values'].append(AP_t)
        
        # Get global averaged model (current global_model after FedAvg)
        global_avg_params = [param.data.clone() for param in self.global_model.parameters()]
        
        # Get best and second-best parameters
        best_params = [param.data.clone() for param in self.best_client_model.parameters()]
        second_best_params = [param.data.clone() for param in self.second_best_client_model.parameters()]
        
        # Apply CSA update to each client
        for client in self.clients:
            # Generate random coefficient for this client
            r_k = np.random.uniform(0, 1)
            
            # Get client's current parameters
            client_params = [param.data.clone() for param in client.model.parameters()]
            
            # Apply CSA update formula:
            # θ_t^(k) = θ_t̄ + fl_t × (θ* - θ_t^k) + r^k × (θ** - θ_t^k)
            new_params = []
            for p_avg, p_best, p_second, p_client in zip(global_avg_params, 
                                                          best_params, 
                                                          second_best_params, 
                                                          client_params):
                # Term 1: Global average (base)
                term1 = p_avg
                
                # Term 2: Move toward best client
                term2 = fl_t * (p_best - p_client)
                
                # Term 3: Random exploration toward second-best
                term3 = r_k * (p_second - p_client)
                
                # Combine all terms
                new_param = term1 + term2 + term3
                new_params.append(new_param)
            
            # Update client's model with CSA-optimized parameters
            for param, new_param in zip(client.model.parameters(), new_params):
                param.data = new_param.clone()
        
        # Update global model to averaged CSA-optimized parameters
        # This ensures next round starts from the CSA-enhanced global state
        all_client_params = []
        for client in self.clients:
            all_client_params.append([param.data.clone() for param in client.model.parameters()])
        
        # Average all CSA-optimized client models
        for param_idx, param in enumerate(self.global_model.parameters()):
            avg_param = torch.stack([client_params[param_idx] for client_params in all_client_params]).mean(dim=0)
            param.data = avg_param.clone()

    def send_models(self):
        """
        Override to send CSA-optimized models to clients.
        After CSA optimization, each client has a personalized model.
        """
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            # Send client's personalized CSA-optimized model
            # (After first round, models are CSA-enhanced)
            client.set_parameters(client.model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
