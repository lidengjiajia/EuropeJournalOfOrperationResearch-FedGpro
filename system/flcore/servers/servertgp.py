import time
import numpy as np
from flcore.clients.clienttgp import clientTGP
from flcore.servers.serverbase import Server
from threading import Thread


class FedTGP(Server):
    """
    FedTGP Server: Federated Temporal Gaussian Process
    
    Server-side implementation for FedTGP algorithm.
    Clients use Gaussian Processes to model temporal dynamics.
    Server performs standard aggregation.
    
    Category: Personalized FL (pFL) - Temporal Adaptation
    """
    
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # Gaussian Process parameters
        if not hasattr(args, 'tgp_length_scale'):
            args.tgp_length_scale = 5.0
        if not hasattr(args, 'tgp_signal_variance'):
            args.tgp_signal_variance = 1.0
        if not hasattr(args, 'tgp_noise_variance'):
            args.tgp_noise_variance = 0.1
        if not hasattr(args, 'tgp_temporal_window'):
            args.tgp_temporal_window = 10
        if not hasattr(args, 'tgp_trust'):
            args.tgp_trust = 0.5
        if not hasattr(args, 'tgp_adaptive_trust'):
            args.tgp_adaptive_trust = True
        
        # Set up clients with temporal GP
        self.set_slow_clients()
        self.set_clients(clientTGP)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"FedTGP Parameters:")
        print(f"  Length Scale: {args.tgp_length_scale}")
        print(f"  Signal Variance: {args.tgp_signal_variance}")
        print(f"  Noise Variance: {args.tgp_noise_variance}")
        print(f"  Temporal Window: {args.tgp_temporal_window}")
        print(f"  Initial Trust: {args.tgp_trust}")
        print(f"  Adaptive Trust: {args.tgp_adaptive_trust}")
        print("Finished creating server and clients.")

        self.Budget = []
        
        # Track GP diagnostics across clients
        self.gp_diagnostics = []

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
                
                # Print GP diagnostics
                if self.gp_diagnostics:
                    avg_trust = np.mean([d['gp_trust'] for d in self.gp_diagnostics])
                    avg_error = np.mean([d['avg_prediction_error'] for d in self.gp_diagnostics])
                    avg_history = np.mean([d['history_length'] for d in self.gp_diagnostics])
                    
                    print(f"GP Diagnostics:")
                    print(f"  Avg Trust Coefficient: {avg_trust:.4f}")
                    print(f"  Avg Prediction Error: {avg_error:.6f}")
                    print(f"  Avg History Length: {avg_history:.1f}")
                    
                    self.gp_diagnostics = []

            # Train selected clients with temporal GP
            for client in self.selected_clients:
                client.train()
                
                # Collect GP diagnostics
                if hasattr(client, 'get_gp_diagnostics'):
                    diagnostics = client.get_gp_diagnostics()
                    self.gp_diagnostics.append(diagnostics)

            self.receive_models()
            
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            
            # Standard weighted averaging aggregation
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
            self.set_new_clients(clientTGP)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
