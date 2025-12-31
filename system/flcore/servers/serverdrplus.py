import copy
import time
import torch
import torch.nn.functional as F
from flcore.clients.clientdrplus import clientDrPlus
from flcore.servers.serverbase import Server
from threading import Thread


class FedDrPlus(Server):
    """
    FedDr+: Federated Distillation with Prototype Enhancement
    Server coordinates distillation-based learning and aggregates client prototypes.
    
    Key Features:
    - Aggregates client models with FedAvg
    - Collects and aggregates class prototypes across clients
    - Enhanced personalization through prototype knowledge
    
    Reference: "FedDr+: Enhanced Federated Distillation with Prototype Regularization" (2023)
    """
    
    def __init__(self, args, times):
        super().__init__(args, times)

        # Set slow clients
        self.set_slow_clients()
        self.set_clients(clientDrPlus)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        
        # Global prototypes: aggregated from all clients
        self.global_prototypes = {}
        self.prototype_weights = {}


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # Receive models and prototypes
            self.receive_models()
            self.receive_prototypes()
            
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            
            # Aggregate models and prototypes
            self.aggregate_parameters()
            self.aggregate_prototypes()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAveraged time per iteration.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientDrPlus)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


    def receive_prototypes(self):
        """Collect class prototypes from selected clients"""
        assert (len(self.selected_clients) > 0)

        self.uploaded_prototypes = []
        self.uploaded_prototype_weights = []
        
        for client in self.selected_clients:
            prototypes, counts = client.get_prototypes()
            self.uploaded_prototypes.append(prototypes)
            self.uploaded_prototype_weights.append(counts)


    def aggregate_prototypes(self):
        """Aggregate class prototypes using weighted averaging"""
        if len(self.uploaded_prototypes) == 0:
            return
        
        # Get number of classes from first client
        num_classes = len(self.uploaded_prototypes[0])
        
        # Initialize global prototypes
        self.global_prototypes = {}
        self.prototype_weights = {}
        
        for class_id in range(num_classes):
            # Collect prototypes and weights for this class
            class_prototypes = []
            class_weights = []
            
            for client_prototypes, client_counts in zip(self.uploaded_prototypes, self.uploaded_prototype_weights):
                if class_id in client_prototypes and client_counts[class_id] > 0:
                    class_prototypes.append(client_prototypes[class_id])
                    class_weights.append(client_counts[class_id])
            
            if len(class_prototypes) > 0:
                # Weighted average of prototypes
                total_weight = sum(class_weights)
                weighted_prototype = torch.zeros_like(class_prototypes[0])
                
                for proto, weight in zip(class_prototypes, class_weights):
                    weighted_prototype += proto * (weight / total_weight)
                
                self.global_prototypes[class_id] = weighted_prototype
                self.prototype_weights[class_id] = total_weight


    def send_prototypes(self):
        """Send global prototypes to clients"""
        assert (len(self.selected_clients) > 0)

        for client in self.selected_clients:
            if hasattr(client, 'set_global_prototypes'):
                client.set_global_prototypes(self.global_prototypes)


    def add_parameters(self, w, client_model):
        """Add weighted client parameters to accumulator"""
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w


    def aggregate_parameters(self):
        """Aggregate client models using weighted averaging"""
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
        
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)
