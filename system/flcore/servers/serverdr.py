import time
import numpy as np
from flcore.clients.clientdr import clientDR
from flcore.servers.serverbase import Server
from threading import Thread


class FedDR(Server):
    """
    FedDR+ Server: Federated Douglas-Rachford+
    
    Server-side implementation for FedDR algorithm.
    Uses standard averaging but clients apply Douglas-Rachford splitting
    for better convergence in non-convex federated optimization.
    
    Category: Traditional FL (tFL) - Advanced Optimization
    
    Reference: "FedDR - Randomized Douglas-Rachford Splitting Algorithms for 
                Nonconvex Federated Composite Optimization" (NeurIPS 2021)
    """
    
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # Douglas-Rachford parameters
        if not hasattr(args, 'dr_rho'):
            args.dr_rho = 0.1
        if not hasattr(args, 'dr_adaptive_rho'):
            args.dr_adaptive_rho = True
        
        # Set up clients with DR splitting
        self.set_slow_clients()
        self.set_clients(clientDR)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"FedDR+ Parameters: rho={args.dr_rho}, adaptive_rho={args.dr_adaptive_rho}")
        print("Finished creating server and clients.")

        self.Budget = []
        
        # Track DR diagnostics
        self.dr_diagnostics = []

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
                
                # Print DR diagnostics
                if self.dr_diagnostics:
                    avg_primal = np.mean([d['primal_residual'] for d in self.dr_diagnostics])
                    avg_dual = np.mean([d['dual_residual'] for d in self.dr_diagnostics])
                    avg_rho = np.mean([d['rho'] for d in self.dr_diagnostics])
                    avg_dual_norm = np.mean([d['dual_norm'] for d in self.dr_diagnostics])
                    
                    print(f"DR Diagnostics:")
                    print(f"  Primal Residual: {avg_primal:.6f}")
                    print(f"  Dual Residual: {avg_dual:.6f}")
                    print(f"  Avg Rho: {avg_rho:.4f}")
                    print(f"  Dual Norm: {avg_dual_norm:.4f}")
                    
                    self.dr_diagnostics = []

            # Train selected clients with DR splitting
            for client in self.selected_clients:
                client.train()
                
                # Collect DR diagnostics
                if hasattr(client, 'get_dr_diagnostics'):
                    diagnostics = client.get_dr_diagnostics()
                    self.dr_diagnostics.append(diagnostics)

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
            self.set_new_clients(clientDR)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
