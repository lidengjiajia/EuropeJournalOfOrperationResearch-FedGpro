import time
import copy
import torch
from flcore.clients.clientkf import clientKF
from flcore.servers.serverbase import Server
from threading import Thread


class FedKF(Server):
    """
    FedKF Server: Federated Kalman Filter
    
    Server-side implementation for FedKF algorithm.
    Aggregates client models using standard weighted averaging.
    The Kalman filtering is applied on the client side.
    
    Category: Traditional FL (tFL) - Robust Aggregation
    """
    
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # Kalman filter parameters (passed to clients)
        if not hasattr(args, 'kf_process_noise'):
            args.kf_process_noise = 0.01
        if not hasattr(args, 'kf_measurement_noise'):
            args.kf_measurement_noise = 0.1
        
        # Set up clients with Kalman filter
        self.set_slow_clients()
        self.set_clients(clientKF)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"FedKF Parameters: process_noise={args.kf_process_noise}, measurement_noise={args.kf_measurement_noise}")
        print("Finished creating server and clients.")

        self.Budget = []
        
        # Track Kalman diagnostics across rounds
        self.kalman_diagnostics = []

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
                
                # Print Kalman diagnostics
                if self.kalman_diagnostics:
                    avg_cov = sum([d['avg_covariance'] for d in self.kalman_diagnostics]) / len(self.kalman_diagnostics)
                    avg_gain = sum([d['avg_kalman_gain'] for d in self.kalman_diagnostics]) / len(self.kalman_diagnostics)
                    print(f"Kalman Diagnostics - Avg Covariance: {avg_cov:.6f}, Avg Gain: {avg_gain:.6f}")
                    self.kalman_diagnostics = []

            # Train selected clients
            for client in self.selected_clients:
                client.train()
                
                # Collect Kalman diagnostics
                if hasattr(client, 'get_kalman_diagnostics'):
                    diagnostics = client.get_kalman_diagnostics()
                    self.kalman_diagnostics.append(diagnostics)

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
            self.set_new_clients(clientKF)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
