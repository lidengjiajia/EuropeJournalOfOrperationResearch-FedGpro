"""
FedGpro Server: Federated Global Prototype Learning

Two-phase federated learning server for privacy-preserving virtual data sharing.

Phase 1: Statistical Collection & Virtual Data Aggregation (No Parameter Aggregation)
- Collect client accuracies and prototypes
- Monitor dynamic threshold for phase transition
- Aggregate virtual data from all clients
- Distribute shared virtual data pool

Phase 2: Federated Learning with Flexible Aggregation
- Standard parameter aggregation with configurable algorithm
- Supports: FedAvg, FedCS, FedProx, and other algorithms
- Clients train on mixed data (real + virtual)
- CSA-optimized parameter updates

Author: [Your Name]
Date: 2025-12-16
"""

import time
import copy
import torch
import numpy as np
import random
import os
from collections import defaultdict
from flcore.clients.clientgpro import clientGpro
from flcore.servers.serverbase import Server


class FedGpro(Server):
    """
    FedGpro Server Implementation
    
    Coordinates two-phase federated learning:
    - Phase 1: No parameter aggregation, collect virtual data
    - Phase 2: Flexible aggregation algorithm (FedAvg/FedCS/FedProx/etc.)
    
    Key Features:
    - Dynamic threshold for phase transition
    - Global prototype aggregation
    - Virtual data pool management
    - Threshold warmup mechanism (default 5 rounds)
    
    Args:
        args: Arguments containing VPS-specific parameters
            - fedgpro_threshold_warmup: Rounds before threshold check (default 5)
            - fedgpro_threshold_min: Minimum threshold value (default 0.60)
            - fedgpro_phase: Initial phase (1 or 2)
            - fedgpro_phase2_agg: Phase 2 aggregation algorithm (fedavg/fedcs/fedprox/etc.)
            - cs_*: FedCS parameters for Phase 2
    """
    
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # Phase control
        self.current_phase = getattr(args, 'fedgpro_phase', 1)
        
        # Save original algorithm name (before Phase 2 modifies it)
        self.original_algorithm = self.algorithm
        
        # Threshold parameters (clients force first 5 rounds training, no server warmup)
        self.threshold_min = getattr(args, 'fedgpro_threshold_min', 0.60)
        self.current_threshold = self.threshold_min  # 初始为最小阈值
        
        # Phase transition threshold
        self.phase_transition_threshold = getattr(args, 'fedgpro_phase_transition_threshold', 0.8)
        
        # Phase control: round tracking
        self.phase2_rounds = getattr(args, 'fedgpro_phase2_rounds', None)  # Max rounds for Phase 2 (None=unlimited)
        self.phase2_start_round = None  # Track when Phase 2 started
        self.phase2_current_round = 0  # Counter for Phase 2 rounds
        
        # Phase 1: Statistics collection
        self.client_accuracies = {}  # {client_id: accuracy}
        self.client_prototypes = {}  # {client_id: {class_id: prototype}}
        self.global_prototypes = {}  # {class_id: prototype}
        self.virtual_data_pool = []  # Aggregated virtual data
        
        # Phase 1: Threshold monitoring
        self.best_client_acc = 0.0
        self.global_avg_acc = 0.0
        self.global_avg_acc_history = []  # Track global average accuracy over rounds
        self.clients_met_threshold = set()  # Track which clients met threshold
        
        # Phase 2: Aggregation algorithm selection
        # Supported: fedavg, fedprox, fedscaffold
        self.phase2_aggregation = getattr(args, 'fedgpro_phase2_agg', 'fedavg').lower()
        
        # Validate Phase 2 algorithm
        supported_algorithms = ['fedavg', 'fedprox', 'fedscaffold']
        if self.phase2_aggregation not in supported_algorithms:
            print(f"Warning: '{self.phase2_aggregation}' not supported. Using 'fedavg'.")
            self.phase2_aggregation = 'fedavg'
        
        # Phase 2: Algorithm-specific parameters
        # FedProx
        self.mu = getattr(args, 'mu', 0.01)  # Proximal term coefficient
        
        # MOON
        self.model_momentum = getattr(args, 'model_momentum', 0.5)  # Contrastive weight
        self.temperature = getattr(args, 'temperature', 0.5)  # Temperature for contrastive loss
        
        # SCAFFOLD (control variates will be initialized in clients)
        
        # Per-FedAvg (MAML)
        self.beta = getattr(args, 'beta', 0.001)  # Meta learning rate
        self.lamda = getattr(args, 'lamda', 15)  # Regularization weight
        
        # Ditto
        self.plocal_steps = getattr(args, 'plocal_steps', 1)  # Personalization steps
        # mu already defined for FedProx
        
        # FedRep
        # No special server parameters needed
        
        # FedProto
        # lamda already defined
        
        # pFedMe
        self.K = getattr(args, 'K', 5)  # Personalization steps
        self.personalized_learning_rate = getattr(args, 'personalized_learning_rate', 0.01)
        # beta, lamda already defined
        
        # FedGWO (Grey Wolf Optimizer)
        self.gwo_a = 2.0  # Will linearly decrease to 0
        self.gwo_alpha_decay = getattr(args, 'gwo_alpha_decay', 0.01)
        
        # FedPSO (Particle Swarm Optimization)
        self.pso_w_max = getattr(args, 'pso_w_max', 0.9)  # Maximum inertia weight
        self.pso_w_min = getattr(args, 'pso_w_min', 0.4)  # Minimum inertia weight
        self.pso_c1 = getattr(args, 'pso_c1', 2.0)  # Cognitive parameter
        self.pso_c2 = getattr(args, 'pso_c2', 2.0)  # Social parameter
        self.pso_v_max = getattr(args, 'pso_v_max', 0.5)  # Maximum velocity ratio
        # PSO state (will be initialized in Phase 2)
        self.pso_gbest_model = None  # Global best model
        self.pso_gbest_acc = 0.0  # Global best accuracy
        self.pso_pbest_models = {}  # {client_id: pbest_model}
        self.pso_pbest_accs = {}  # {client_id: pbest_acc}
        self.pso_velocities = {}  # {client_id: velocity}
        
        # Phase 2: FedCS parameters (deprecated, kept for backward compatibility)
        self.f_max = getattr(args, 'cs_f_max', 2.0)
        self.f_min = getattr(args, 'cs_f_min', 0.1)
        self.AP_max = getattr(args, 'cs_AP_max', 0.3)
        self.AP_min = getattr(args, 'cs_AP_min', 0.1)
        
        # Phase 2: FedCS state (deprecated)
        self.best_client_id = None
        self.second_best_client_id = None
        self.best_client_model = None
        self.second_best_client_model = None
        
        # Phase 2: Adaptive weight decay for early-stopped clients
        self.client_stop_times = {}  # {client_id: t_s(k)} - round when client stopped
        self.client_accuracies_history = defaultdict(list)  # {client_id: [acc_0, acc_1, ...]}  
        self.active_training_clients = set()  # C_t - clients still training
        self.decay_lambda = getattr(args, 'fedgpro_decay_lambda', 0.5)  # λ: decay strength
        self.client_decay_weights = {}  # {client_id: α_k(t)}
        
        # Generalization experiment: reserved clients
        self.reserved_client_ids = getattr(args, 'reserved_clients', [])  # e.g., [0, 1, 2]
        self.reserved_clients = []  # Reserved client objects (not participating in training)
        if len(self.reserved_client_ids) > 0:
            print(f"\n[Generalization Experiment] Reserved {len(self.reserved_client_ids)} clients: {self.reserved_client_ids}")
            print(f"  These clients will NOT participate in training.")
            print(f"  They will be used for generalization testing after training completes.")
        
        # Set up clients
        self.set_slow_clients()
        self.set_clients(clientGpro)
        
        print(f"\n{'='*60}")
        print(f"FedGpro: Federated Global Prototype Learning")
        print(f"{'='*60}")
        print(f"Join ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"\nPhase 1 Parameters (Max 20 rounds):")
        print(f"  Minimum threshold: {self.threshold_min}")
        print(f"  Threshold formula:")
        print(f"    ACC(t-1) = (best_acc(t-1) + avg_acc(t-1)) / 2")
        print(f"    threshold(r) = ACC(t-1),                                       if r <= 9")
        print(f"    threshold(r) = max(0.66, ACC(t-1)*(1-0.05*floor((r-10)/5))), if r >= 10")
        print(f"  Training stages:")
        print(f"    - Round 1-10: Forced training (no early stop check, ensure VAE quality)")
        print(f"    - Round 11+: Early stop check + dynamic threshold with time decay")
        print(f"  Phase transition: 70% clients qualified OR 25 rounds reached")
        print(f"\nPhase 2 Parameters:")
        print(f"  Aggregation: {self.phase2_aggregation}")
        print(f"  Supported algorithms: fedavg, fedprox, fedscaffold")
        if self.phase2_aggregation == 'fedprox':
            print(f"  - mu (proximal): {self.mu}")
        elif self.phase2_aggregation == 'fedscaffold':
            print(f"  - Control variates enabled")
        print(f"{'='*60}\n")
        print("Finished creating server and clients.")
        
        self.Budget = []
    
    def set_clients(self, clientObj):
        """
        Override parent set_clients to support reserved clients for generalization experiments
        
        Reserved clients:
        - Do NOT participate in training
        - Used for generalization testing after training completes
        - Stored separately in self.reserved_clients
        """
        from utils.data_utils import read_client_data
        
        for i, train_slow, send_slow in zip(range(self.num_clients), 
                                             self.train_slow_clients, 
                                             self.send_slow_clients):
            
            if i in self.reserved_client_ids:
                # Reserved client: create but don't add to training clients
                train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
                test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
                client = clientObj(self.args, 
                                id=i, 
                                train_samples=len(train_data), 
                                test_samples=len(test_data), 
                                train_slow=train_slow, 
                                send_slow=send_slow)
                self.reserved_clients.append(client)
                print(f"  Reserved Client {i}: {len(train_data)} train samples, {len(test_data)} test samples")
            else:
                # Normal training client
                train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
                test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
                client = clientObj(self.args, 
                                id=i, 
                                train_samples=len(train_data), 
                                test_samples=len(test_data), 
                                train_slow=train_slow, 
                                send_slow=send_slow)
                self.clients.append(client)
    
    def evaluate_reserved_clients(self):
        """
        Evaluate generalization performance on reserved clients
        
        Returns:
            dict: Statistics about reserved clients' performance
        """
        if len(self.reserved_clients) == 0:
            print("  No reserved clients to evaluate")
            return {}
        
        print(f"\n{'='*80}")
        print(f"[Generalization Test] Evaluating {len(self.reserved_clients)} Reserved Clients")
        print(f"{'='*80}")
        
        reserved_accuracies = []
        reserved_losses = []
        reserved_aucs = []
        reserved_precisions = []
        reserved_recalls = []
        reserved_f1s = []
        
        for client in self.reserved_clients:
            # Set global model parameters
            client.set_parameters(self.global_model)
            
            # Test on reserved client's local data
            test_acc, test_num, test_auc, test_precision, test_recall, test_f1 = client.test_metrics()
            test_loss = client.train_metrics()[1]  # Get train loss
            
            reserved_accuracies.append(test_acc)
            reserved_losses.append(test_loss)
            reserved_aucs.append(test_auc)
            reserved_precisions.append(test_precision)
            reserved_recalls.append(test_recall)
            reserved_f1s.append(test_f1)
            
            print(f"  Reserved Client {client.id}: Acc={test_acc:.4f}, Prec={test_precision:.4f}, Rec={test_recall:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}")
        
        # Compute statistics
        stats = {
            'avg_accuracy': np.mean(reserved_accuracies),
            'std_accuracy': np.std(reserved_accuracies),
            'avg_precision': np.mean(reserved_precisions),
            'std_precision': np.std(reserved_precisions),
            'avg_recall': np.mean(reserved_recalls),
            'std_recall': np.std(reserved_recalls),
            'avg_f1': np.mean(reserved_f1s),
            'std_f1': np.std(reserved_f1s),
            'avg_auc': np.mean(reserved_aucs),
            'std_auc': np.std(reserved_aucs),
            'avg_loss': np.mean(reserved_losses),
            'individual_accuracies': reserved_accuracies,
            'individual_precisions': reserved_precisions,
            'individual_recalls': reserved_recalls,
            'individual_f1s': reserved_f1s,
            'individual_aucs': reserved_aucs,
        }
        
        print(f"\n  [Summary] Reserved Clients Performance:")
        print(f"    Average Accuracy:  {stats['avg_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
        print(f"    Average Precision: {stats['avg_precision']:.4f} ± {stats['std_precision']:.4f}")
        print(f"    Average Recall:    {stats['avg_recall']:.4f} ± {stats['std_recall']:.4f}")
        print(f"    Average F1-Score:  {stats['avg_f1']:.4f} ± {stats['std_f1']:.4f}")
        print(f"    Average AUC:       {stats['avg_auc']:.4f} ± {stats['std_auc']:.4f}")
        print(f"    Average Loss:      {stats['avg_loss']:.4f}")
        print(f"{'='*80}\n")
        
        return stats
    
    def train(self):
        """
        Main training loop with phase-aware dispatching
        
        Phase control logic:
        - Phase 1: Runs until all clients meet threshold (dynamic)
        - Phase 2: Runs for fedgpro_phase2_rounds (if specified) or until global_rounds
        """
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            
            # Check if Phase 2 should stop
            if self.current_phase == 2 and self.phase2_rounds is not None:
                if self.phase2_current_round >= self.phase2_rounds:
                    print(f"\n{'='*60}")
                    print(f"Phase 2 completed: {self.phase2_current_round}/{self.phase2_rounds} rounds")
                    print(f"{'='*60}\n")
                    # CRITICAL: Perform final evaluation before breaking (unconditionally)
                    print(f"\n{'='*60}")
                    print(f"Final Evaluation - Round {i} | Phase {self.current_phase}")
                    print(f"{'='*60}")
                    print("\nEvaluate global model")
                    self.evaluate()
                    break
            
            # Phase-specific training
            if self.current_phase == 1:
                self.train_phase1(i)
            else:
                self.train_phase2(i)
                self.phase2_current_round += 1  # Increment Phase 2 counter
            
            # Evaluation
            if i % self.eval_gap == 0:
                print(f"\n{'='*60}")
                print(f"Round {i} | Phase {self.current_phase}")
                print(f"{'='*60}")
                print("\nEvaluate global model")
                self.evaluate()
                
                # Evaluate personalized model (Phase 2 only)
                if self.current_phase == 2:
                    print("\nEvaluate personalized models")
                    self.evaluate_personalized()
            
            self.Budget.append(time.time() - s_t)
            print(f"-" * 60)
            print(f"Round {i} time cost: {time.time() - s_t:.2f}s")
            print(f"-" * 60)
            
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        
        print("\nBest accuracy:")
        if self.rs_test_acc:
            print(max(self.rs_test_acc))
        else:
            print("No evaluation results available")
        print("\nAverage time cost:")
        if len(self.Budget) > 1:
            print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        else:
            print("No timing data available")
        
        # Generalization test on reserved clients (if any)
        if len(self.reserved_clients) > 0:
            print("\n" + "="*80)
            print("Training completed. Starting generalization test on reserved clients...")
            print("="*80)
            reserved_stats = self.evaluate_reserved_clients()
            
            # Save reserved client statistics for analysis
            if reserved_stats:
                print(f"\n[Generalization Summary]")
                print(f"  Average Accuracy on Reserved Clients: {reserved_stats['avg_accuracy']:.4f}")
                print(f"  Average AUC on Reserved Clients: {reserved_stats['avg_auc']:.4f}")
                print(f"  Average Loss on Reserved Clients: {reserved_stats['avg_loss']:.4f}")
        
        self.save_results()
    
    # ==================== Personalized Model Evaluation (Ditto-style) ====================
    
    def test_metrics_personalized(self):
        """
        Evaluate personalized models across all clients
        
        Returns:
            ids: Client IDs
            num_samples: Number of test samples per client
            tot_correct: Total correct predictions per client
            tot_auc: Total AUC score per client
        """
        num_samples = []
        tot_correct = []
        tot_auc = []
        
        for c in self.clients:
            ct, ns, auc = c.test_metrics_personalized()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)
        
        ids = [c.id for c in self.clients]
        
        return ids, num_samples, tot_correct, tot_auc
    
    def train_metrics_personalized(self):
        """
        Compute personalized model training metrics
        
        Returns:
            ids: Client IDs
            num_samples: Number of training samples per client
            losses: Training losses per client
        """
        num_samples = []
        losses = []
        
        for c in self.clients:
            cl, ns = c.train_metrics_personalized()
            num_samples.append(ns)
            losses.append(cl * 1.0)
        
        ids = [c.id for c in self.clients]
        
        return ids, num_samples, losses
    
    def evaluate_personalized(self, acc=None, loss=None):
        """
        Evaluate personalized models with detailed statistics
        
        Args:
            acc: Optional list to append accuracy
            loss: Optional list to append loss
        """
        stats = self.test_metrics_personalized()
        stats_train = self.train_metrics_personalized()
        
        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)
        
        print("Averaged Train Loss (Personalized): {:.4f}".format(train_loss))
        print("Averaged Test Accuracy (Personalized): {:.4f}".format(test_acc))
        print("Averaged Test AUC (Personalized): {:.4f}".format(test_auc))
        print("Std Test Accuracy (Personalized): {:.4f}".format(np.std(accs)))
        print("Std Test AUC (Personalized): {:.4f}".format(np.std(aucs)))
    
    # ==================== Phase 1: Virtual Data Collection ====================
    
    def train_phase1(self, round_num):
        """
        Phase 1 Training: No Parameter Aggregation
        
        Workflow:
        1. Clients train VAE + Classifier locally
        2. Collect accuracies and prototypes (no model parameters)
        3. Check for early-stopped clients and compute adaptive weights
        4. Compute dynamic threshold
        5. If threshold met: clients generate virtual data
        6. Aggregate virtual data pool (with adaptive weights)
        7. Check phase transition condition
        """
        print(f"\n[Phase 1] Statistical Collection (Round {round_num})")
        
        # Step 1: Clients train locally
        for client in self.selected_clients:
            accuracy = client.train()  # Returns accuracy
            self.client_accuracies[client.id] = accuracy
            
            # Track accuracy history for performance change calculation
            self.client_accuracies_history[client.id].append(accuracy)
            
            # Check if client just stopped (early stopping)
            if client.early_stopped and client.id not in self.client_stop_times:
                self.client_stop_times[client.id] = round_num
                print(f"  [Adaptive Decay] Client {client.id} early stopped at round {round_num}")
            
            # Track active training clients (not early stopped)
            if not client.early_stopped:
                self.active_training_clients.add(client.id)
        
        # Step 2: Compute dynamic threshold BEFORE updating statistics (use ACC(t-1))
        # 关键修改：先计算阈值，使用上一轮的best_client_acc和global_avg_acc
        self.current_threshold = self.compute_dynamic_threshold(round_num)
        print(f"  Current threshold (based on ACC(t-1)): {self.current_threshold:.4f}")
        
        # Step 3: Compute global statistics (update ACC(t) for next round)
        accuracies = list(self.client_accuracies.values())
        self.best_client_acc = max(accuracies) if accuracies else 0.0
        self.global_avg_acc = np.mean(accuracies) if accuracies else 0.0
        
        # Record global average accuracy history for adaptive decay
        self.global_avg_acc_history.append(self.global_avg_acc)
        
        print(f"  Best client accuracy (ACC(t)): {self.best_client_acc:.4f}")
        print(f"  Global average accuracy (ACC(t)): {self.global_avg_acc:.4f}")
        
        # Step 4: Compute adaptive decay weights for early-stopped clients
        # 关键：必须在global_avg_acc更新后计算，确保使用最新的全局准确率
        self._compute_adaptive_decay_weights(round_num)
        
        # Report early stopping statistics
        if len(self.client_stop_times) > 0:
            print(f"  Early-stopped clients: {len(self.client_stop_times)}/{self.num_clients}")
            print(f"  Active training clients: {len(self.active_training_clients)}")
        
        # Step 3.5: Update threshold for all clients (for early stopping judgment)
        for client in self.selected_clients:
            client.update_threshold(self.current_threshold)
        
        # Step 4: Check which clients meet threshold
        newly_met = []
        for client in self.selected_clients:
            if client.id not in self.clients_met_threshold:
                if self.client_accuracies[client.id] >= self.current_threshold:
                    # Client meets threshold - generate virtual data
                    print(f"  Client {client.id} meets threshold! Generating virtual data...")
                    
                    # Check if differential privacy is enabled
                    use_dp = (client.epsilon is not None and client.epsilon > 0 and 
                             client.noise_type is not None and client.noise_type != 'none')
                    
                    # Step 4.1-4.2: Train baseline VAE and compute feature importance (only if DP enabled)
                    if use_dp:
                        client.train_baseline_vae()
                        client.compute_feature_importance()
                    else:
                        print(f"  Client {client.id}: Privacy disabled, skipping baseline VAE training.")
                    
                    # Step 4.3: Generate virtual data
                    virtual_data = client.generate_virtual_data()
                    
                    # Step 4.4: Add adaptive differential privacy noise (only if DP enabled)
                    if use_dp:
                        # Use adaptive noise addition (based on feature importance)
                        client.add_adaptive_noise_to_virtual_data(strategy='privacy_first')
                    else:
                        print(f"  Client {client.id}: No noise added (privacy disabled).")
                    
                    # Collect for aggregation
                    self.clients_met_threshold.add(client.id)
                    newly_met.append(client.id)
                    
                    # Add to virtual data pool
                    self.virtual_data_pool.extend(client.virtual_data)
        
        if newly_met:
            print(f"  Newly qualified clients: {newly_met}")
            print(f"  Total qualified clients: {len(self.clients_met_threshold)}/{self.num_clients}")
            print(f"  Virtual data pool size: {len(self.virtual_data_pool)}")
        
        # Step 5: 联邦聚合VAE参数（优化：每轮聚合以提升生成质量）
        self.aggregate_vae_parameters()
        
        # Step 6: Aggregate prototypes (optional, for monitoring)
        self.aggregate_prototypes()
        
        # Step 7: Check phase transition
        if self.check_phase_transition(round_num):
            self.transition_to_phase2()
        
        # Clear active training set for next round
        self.active_training_clients.clear()
    
    def compute_dynamic_threshold(self, round_num):
        """
        Compute dynamic threshold with time decay strategy
        
        数学公式:
        
        ACC(t-1) = (best_acc(t-1) + avg_acc(t-1)) / 2
        
        threshold(round) = {
            ACC(t-1),                                                          if round ≤ 9
            max(threshold_min × 1.1, ACC(t-1) × (1 - 0.05 × ⌊(round-10)/5⌋)), if 10 ≤ round ≤ 20
        }
        
        策略说明:
        - 使用ACC(t-1): 上一轮的统计数据（在本函数调用前未更新）
        - Round 1-5: 强制训练，不检查早停
        - Round 6-9: 可检查早停，严格阈值 ACC(t-1)
        - Round 10+: 时间衰减，每5轮降低5%
        - Round 20+: Phase 1结束，强制转Phase 2
        - 保底质量: threshold_min × 1.1 = 0.60 × 1.1 = 0.66
        
        Args:
            round_num: Current training round (0-indexed)
        
        Returns:
            threshold: Current threshold value with decay
        """
        # 基础阈值: ACC(t) = (best(t-1) + avg(t-1)) / 2
        # 注意：此时best_client_acc和global_avg_acc还是上一轮的值（未更新）
        base_threshold = (self.best_client_acc + self.global_avg_acc) / 2
        
        # Round 1-9: 严格阈值（不衰减）
        if round_num < 10:
            threshold = base_threshold
            decay_info = "严格阈值"
        else:
            # Round 10+: 时间衰减策略
            # decay_steps = ⌊(round - 10) / 5⌋
            rounds_after_10 = round_num - 10
            decay_steps = rounds_after_10 // 5  # 每5轮一个衰减步
            decay_rate = 0.05 * decay_steps  # 每步5%: 0.05, 0.10, 0.15, ...
            
            # 应用衰减: threshold = ACC(t) × (1 - decay_rate)
            threshold = base_threshold * (1 - decay_rate)
            decay_info = f"衰减 {decay_rate*100:.0f}% (第{decay_steps}步)"
        
        # 保底阈值: max(threshold_min × 1.1, threshold)
        min_allowed = self.threshold_min * 1.1
        final_threshold = max(min_allowed, threshold)
        
        # 打印衰减信息（每5轮打印一次，从第10轮开始）
        if round_num >= 10 and (round_num - 10) % 5 == 0:
            print(f"  [Threshold Decay] Round {round_num+1}: {decay_info}")
            print(f"    Base: {base_threshold:.4f} → Decayed: {threshold:.4f} → Final: {final_threshold:.4f}")
        
        return final_threshold
    
    def _compute_adaptive_decay_weights(self, round_num):
        """
        计算早停客户端的自适应权重衰减
        
        Algorithm: Adaptive decay based on accumulated global accuracy changes
        For each early-stopped client k (stopped at round t_s(k)):
        1. Compute global accuracy change from stop time to current:
           Δ_global(t) = acc_global(t) - acc_global(t_s(k))
        2. Apply exponential decay weight:
           α_k(t) = exp(-λ · max(0, Δ_global(t)))
        
        Where:
        - lambda (decay_lambda): Decay strength hyperparameter, controls decay speed
        - Δ_global(t): 全局模型性能的累积提升
        - max(0, ...): Decay only when global performance improves, keep weight when performance decreases
        
        Intuition:
        - 全局准确率大幅提升 → 活跃客户端学到新知识 → 快速衰减早停客户端权重
        - 全局准确率停滞/下降 → 保留早停客户端的稳定原型 → 减缓衰减
        
        Args:
            round_num: 当前训练轮数
        """
        # If no early stopped clients, return directly
        if len(self.client_stop_times) == 0:
            return
        
        # If global accuracy history is not enough (first round), skip
        if len(self.global_avg_acc_history) == 0:
            return
        
        # 打印衰减计算详情（每5轮打印一次）
        should_print = (round_num % 5 == 0) or (round_num >= 10 and round_num <= 15)
        if should_print and len(self.client_stop_times) > 0:
            print(f"\n  [Adaptive Decay] 权重衰减计算详情 (Round {round_num+1}):")
            print(f"  当前全局准确率: {self.global_avg_acc:.4f}")
            print(f"  衰减强度λ: {self.decay_lambda}")
        
        # 为每个早停客户端计算衰减权重
        for client_id, stop_round in self.client_stop_times.items():
            # 确保停止轮数在历史记录范围内
            if stop_round >= len(self.global_avg_acc_history):
                # Just stopped client, weight temporarily kept at 1.0
                self.client_decay_weights[client_id] = 1.0
                if should_print:
                    print(f"    Client {client_id}: 刚达标 → α_k=1.0000 (初始)")
                continue
            
            # 获取客户端停止时的全局准确率
            acc_at_stop = self.global_avg_acc_history[stop_round]
            
            # 计算从停止时刻到现在的全局准确率累积变化
            delta_global = self.global_avg_acc - acc_at_stop
            
            # Apply exponential decay formula: alpha_k(t) = exp(-lambda * max(0, delta_global))
            delta_clamped = max(0, delta_global)
            alpha = np.exp(-self.decay_lambda * delta_clamped)
            
            # 存储权重
            self.client_decay_weights[client_id] = alpha
            
            # 打印详细信息
            if should_print:
                print(f"    Client {client_id}: 第{stop_round+1}轮达标")
                print(f"      达标时准确率: {acc_at_stop:.4f}")
                print(f"      准确率提升Δ: {delta_global:+.4f} → max(0,Δ)={delta_clamped:.4f}")
                print(f"      衰减权重α_k: exp(-{self.decay_lambda}×{delta_clamped:.4f}) = {alpha:.4f}")
    
    def aggregate_vae_parameters(self):
        """
        Phase 1优化：联邦聚合VAE参数
        
        策略：
        1. 收集所有客户端的VAE参数
        2. FedAvg加权平均（权重=样本数量）
        3. 分发全局VAE回所有客户端
        
        目的：
        - 提升VAE泛化能力（学习全局数据分布）
        - 提高少数类生成质量（从75样本→1125样本的等效学习）
        - 预期虚拟样本置信度从30.5%提升到62%+
        """
        print(f"\n  [VAE Aggregation] 联邦聚合VAE参数...")
        
        # 收集所有客户端的VAE参数和权重
        vae_params_list = []
        weights = []
        
        for client in self.selected_clients:
            vae_params_list.append(client.get_vae_parameters())
            weights.append(client.train_samples)
        
        # 归一化权重
        total_samples = sum(weights)
        normalized_weights = [w / total_samples for w in weights]
        
        # FedAvg聚合：对每个参数进行加权平均
        global_vae_params = {}
        
        # 遍历每个组件（encoder, fc_mu, fc_logvar, decoder）
        for component_name in vae_params_list[0].keys():
            global_vae_params[component_name] = {}
            
            # 遍历该组件的每个参数
            for param_name in vae_params_list[0][component_name].keys():
                # 加权平均
                weighted_sum = None
                for client_params, weight in zip(vae_params_list, normalized_weights):
                    param_value = client_params[component_name][param_name].clone()
                    if weighted_sum is None:
                        weighted_sum = param_value * weight
                    else:
                        weighted_sum += param_value * weight
                
                global_vae_params[component_name][param_name] = weighted_sum
        
        # 分发全局VAE参数到所有客户端
        for client in self.selected_clients:
            client.set_vae_parameters(global_vae_params)
        
        print(f"  [OK] VAE参数聚合完成，已分发到 {len(self.selected_clients)} 个客户端")
        print(f"  参数规模: Encoder={len(global_vae_params['encoder'])} layers, "
              f"Decoder={len(global_vae_params['decoder'])} layers")
    
    def aggregate_prototypes(self):
        """
        Aggregate class prototypes from all clients with adaptive decay weights
        
        Phase 1 Strategy:
        1. Collect prototypes from all active clients
        2. For inactive/dropout clients: use their last uploaded prototypes
        3. Apply adaptive decay weights to early-stopped clients
        4. Perform weighted averaging (weights normalized to sum=1 per class)
        5. Distribute global prototypes back to clients for next round
        
        Weight Calculation:
        - Base weight: w_base = train_samples (数据量)
        - Decay weight: α_k(t) = exp(-λ·max(0, Δ_global)) (早停衰减)
        - Combined weight: w_k = w_base × α_k(t)
        - Normalized weight: w_k_norm = w_k / Σw_k (归一化为1)
        
        This ensures continuity and helps clients learn from global knowledge.
        """
        class_prototypes = defaultdict(list)
        class_weights = defaultdict(list)
        
        # 用于打印详细权重信息
        client_weight_info = []
        
        # Collect prototypes from all clients in Phase 2
        # 所有客户端均在Phase 2，都可以贡献原型
        for client in self.clients:
            if len(client.prototypes) > 0:
                # Base weight: sample size
                base_weight = client.train_samples
                
                # Apply adaptive decay weight if client has early stopped
                decay_weight = self.client_decay_weights.get(client.id, 1.0)
                combined_weight = base_weight * decay_weight
                
                # 记录权重信息用于打印
                is_early_stopped = client.id in self.client_stop_times
                client_weight_info.append({
                    'id': client.id,
                    'base': base_weight,
                    'decay': decay_weight,
                    'combined': combined_weight,
                    'early_stopped': is_early_stopped
                })
                
                for class_id, prototype in client.prototypes.items():
                    class_prototypes[class_id].append(prototype.to(self.device))
                    class_weights[class_id].append(combined_weight)
        
        # If no new prototypes this round, keep previous global prototypes
        if len(class_prototypes) == 0:
            print("  Warning: No prototypes collected this round. Using previous global prototypes.")
            return
        
        # 打印权重分解详情（每5轮打印一次）
        current_round = len(self.Budget)
        if current_round % 5 == 0 or len(self.client_stop_times) > 0:
            print(f"\n  [Prototype Aggregation] 权重分解详情 (Round {current_round}):")
            print(f"  {'='*80}")
            
            # 计算总组合权重（归一化前）
            total_combined_weight = sum([info['combined'] for info in client_weight_info])
            
            print(f"  {'客户端':<8} {'数据量':<10} {'衰减α_k':<12} {'组合权重':<14} {'归一化权重':<14} {'状态':<10}")
            print(f"  {'-'*80}")
            
            for info in client_weight_info:
                normalized_weight = info['combined'] / total_combined_weight
                status = "早停" if info['early_stopped'] else "活跃"
                
                print(f"  Client {info['id']:<2}  "
                      f"{info['base']:<10.0f}  "
                      f"{info['decay']:<12.4f}  "
                      f"{info['combined']:<14.2f}  "
                      f"{normalized_weight:<14.4f}  "
                      f"{status:<10}")
            
            print(f"  {'-'*80}")
            print(f"  总组合权重: {total_combined_weight:.2f}")
            print(f"  归一化检查: Σw_k_norm = {sum([info['combined'] / total_combined_weight for info in client_weight_info]):.6f} (应为1.0)")
            print(f"  {'='*80}\n")
        
        # Weighted average
        new_global_prototypes = {}
        for class_id, prototypes in class_prototypes.items():
            if len(prototypes) > 0:
                weights = torch.tensor(class_weights[class_id], device=self.device)
                weights = weights / weights.sum()  # 归一化：确保权重和为1
                
                # Stack and weighted sum
                proto_stack = torch.stack(prototypes)
                new_global_prototypes[class_id] = (proto_stack * weights.view(-1, 1)).sum(dim=0)
        
        # Update global prototypes (preserves old classes if not updated)
        self.global_prototypes.update(new_global_prototypes)
        
        # Log adaptive decay weights for stopped clients
        stopped_clients = [cid for cid in range(self.num_clients) if cid in self.client_stop_times]
        if len(stopped_clients) > 0:
            print(f"  [Adaptive Decay] Prototype aggregation weights: ", end="")
            for cid in stopped_clients[:5]:  # Show first 5
                weight = self.client_decay_weights.get(cid, 1.0)
                print(f"C{cid}={weight:.3f} ", end="")
            if len(stopped_clients) > 5:
                print(f"... ({len(stopped_clients)} total)", end="")
            print()
        
        # Distribute global prototypes to all clients
        if len(self.global_prototypes) > 0:
            for client in self.clients:
                client.receive_global_prototypes(self.global_prototypes)
    
    def check_phase_transition(self, round_num):
        """
        Check if Phase 1 should transition to Phase 2
        
        Logic:
        1. First 10 rounds are forced Phase 1 (no early stop check)
        2. Round 11-25: Check if 70% clients are qualified
        3. Round 25: Force Phase 1 to finalize (close virtual data contribution window)
        
        Design Rationale:
        - Force 10 rounds to ensure VAE quality and basic learning
        - 70% threshold is relaxed for faster Phase 2 entry
        - Max 25 rounds Phase 1 to give more training time
        - Time decay helps weak clients qualify
        
        Returns:
            bool: True if should transition to Phase 2
        """
        # Force Phase 1 for first 10 rounds
        min_phase1_rounds = 10
        max_phase1_rounds = 25  # Phase 1最多25轮
        
        if round_num < min_phase1_rounds:
            print(f"\n[Phase Transition Check] Round {round_num+1}: 前{min_phase1_rounds}轮强制Phase 1（共同训练，确保VAE质量）")
            return False
        
        # Round 25: 强制转Phase 2
        if round_num >= max_phase1_rounds:
            print(f"\n[Phase Transition Check] Round {round_num+1}: 达到Phase 1最大轮次({max_phase1_rounds})，强制转Phase 2")
            print(f"  当前达标: {len(self.clients_met_threshold)}/{self.num_clients} 客户端")
            print(f"  注意：第{max_phase1_rounds}轮后未达标客户端将永久不贡献虚拟数据")
            return True
        
        # ========== 70%阈值策略（6-19轮检查）==========
        transition_threshold = 0.7  # 70%客户端达标即可
        required_clients = int(self.num_clients * transition_threshold)
        
        # 打印详细的客户端达标状态
        print(f"\n{'='*80}")
        print(f"[Phase Transition Check] Round {round_num+1} (Phase 1: 最多{max_phase1_rounds}轮)")
        print(f"{'='*80}")
        print(f"达标要求: {required_clients}/{self.num_clients} 客户端 ({int(transition_threshold*100)}%)")
        print(f"当前状态: {len(self.clients_met_threshold)}/{self.num_clients} 客户端达标")
        print(f"\n客户端详细状态:")
        
        # 显示每个客户端的准确率和达标状态
        qualified_clients = []
        unqualified_clients = []
        
        for client_id in range(self.num_clients):
            acc = self.client_accuracies.get(client_id, 0.0)
            is_qualified = client_id in self.clients_met_threshold
            
            if is_qualified:
                qualified_clients.append((client_id, acc))
            else:
                unqualified_clients.append((client_id, acc))
        
        # 打印已达标客户端
        if qualified_clients:
            print(f"\n[+] 已达标客户端 ({len(qualified_clients)}个):")
            for cid, acc in sorted(qualified_clients, key=lambda x: x[1], reverse=True):
                print(f"  Client {cid}: {acc:.4f} [+]")
        
        # 打印未达标客户端
        if unqualified_clients:
            print(f"\n[-] 未达标客户端 ({len(unqualified_clients)}个):")
            for cid, acc in sorted(unqualified_clients, key=lambda x: x[1], reverse=True):
                print(f"  Client {cid}: {acc:.4f} (仍在训练中...)")
        
        print(f"\n当前阈值: {self.current_threshold:.4f}")
        print(f"平均准确率: {self.global_avg_acc:.4f}")
        print(f"最佳准确率: {self.best_client_acc:.4f}")
        print(f"{'='*80}")
        
        if len(self.clients_met_threshold) >= required_clients:
            print(f"\n[>>] Phase Transition Triggered! (70% Threshold Strategy)")
            print(f"   Qualified clients: {len(self.clients_met_threshold)}/{self.num_clients}")
            print(f"   Required: {required_clients} ({int(transition_threshold*100)}%)")
            print(f"   Remaining: {self.num_clients - len(self.clients_met_threshold)} clients will join dynamically in Phase 2")
            print(f"   Phase 1 completed: {round_num + 1} rounds (min: {min_phase1_rounds}, max: {max_phase1_rounds})")
            print(f"{'='*80}\n")
            return True
        else:
            shortage = required_clients - len(self.clients_met_threshold)
            remaining_rounds = max_phase1_rounds - round_num
            print(f"\n[CONTINUE] 继续Phase 1: 还需要 {shortage} 个客户端达标 (剩余{remaining_rounds}轮)")
            print(f"{'='*80}\n")
        
        return False
    
    def transition_to_phase2(self):
        """
        Transition from Phase 1 to Phase 2
        
        Actions:
        1. Switch phase flag
        2. Record transition round
        3. Distribute virtual data pool to all clients
        4. Initialize algorithm-specific states
        """
        print(f"\n{'='*60}")
        print(f"PHASE TRANSITION: Phase 1 → Phase 2")
        print(f"{'='*60}")
        print(f"Virtual data pool size: {len(self.virtual_data_pool)}")
        print(f"Phase 2 Algorithm: {self.phase2_aggregation.upper()}")
        if self.phase2_rounds is not None:
            print(f"Phase 2 Max Rounds: {self.phase2_rounds}")
        print(f"{'='*60}\n")
        
        # Switch phase and record transition
        self.current_phase = 2
        self.phase2_start_round = len(self.Budget)  # Current round number
        self.phase2_current_round = 0  # Reset Phase 2 counter
        
        # 🔥 FIX: 更新算法名称以包含Phase2聚合方法（用于结果保存）
        # 将 "FedGpro" 改为 "FedGpro-{phase2_agg}"
        if '-' not in self.algorithm:  # 避免重复添加
            phase2_method = self.phase2_aggregation.capitalize()
            if self.phase2_aggregation == 'fedavg':
                phase2_method = 'FedAvg'
            elif self.phase2_aggregation == 'fedprox':
                phase2_method = 'FedProx'
            elif self.phase2_aggregation == 'ditto':
                phase2_method = 'Ditto'
            elif self.phase2_aggregation == 'fedgwo':
                phase2_method = 'FedGwo'
            elif self.phase2_aggregation == 'fedpso':
                phase2_method = 'FedPso'
            # 可以添加更多...
            
            original_algorithm = self.algorithm
            self.algorithm = f"{original_algorithm}-{phase2_method}"
            print(f"  [Algorithm Name] Updated: {original_algorithm} → {self.algorithm}")
        
        # 🔥 关键设计：70%达标后，所有客户端（100%）都进入Phase 2
        # Distribute virtual data pool to ALL clients (100%)
        # - Qualified clients (70%): Already contributed virtual data, participate in Phase 2
        # - Unqualified clients (30%): Can still contribute virtual data (until round 20), participate in Phase 2
        # 
        # Phase 1在后台并行：未达标客户端继续尝试达标并贡献虚拟数据（直到第20轮）
        for client in self.clients:
            # 所有客户端都进入Phase 2
            client.current_phase = 2
            client.set_phase2_algorithm(self.phase2_aggregation)
            
            # 所有客户端都加载虚拟数据池（用于混合训练）
            client.load_shared_virtual_data(self.virtual_data_pool)
            
            if client.id in self.clients_met_threshold:
                # 已达标客户端：已贡献虚拟数据
                client.contributes_virtual_data = True
                print(f"  Client {client.id}: Phase 2 (already contributed virtual data)")
            else:
                # 未达标客户端：尚未贡献，但可在第20轮前继续尝试
                client.contributes_virtual_data = False
                print(f"  Client {client.id}: Phase 2 (can still contribute virtual data until round 20)")
        
        # Initialize algorithm-specific states for ALL clients (100%)
        # 所有客户端都参与Phase 2，因此都需要初始化算法状态
        
        if self.phase2_aggregation == 'moon':
            # MOON: Initialize previous models for contrastive learning
            for client in self.clients:
                client.init_moon_states()
        elif self.phase2_aggregation == 'scaffold':
            # SCAFFOLD: Initialize control variates
            for client in self.clients:
                client.init_scaffold_controls()
        elif self.phase2_aggregation == 'ditto':
            # Ditto: Initialize personalized models
            for client in self.clients:
                client.init_personalized_model()
        elif self.phase2_aggregation == 'fedgwo':
            # FedGWO: Initialize wolf positions
            self.gwo_alpha_pos = None  # Alpha wolf (best)
            self.gwo_beta_pos = None   # Beta wolf (2nd best)
            self.gwo_delta_pos = None  # Delta wolf (3rd best)
        elif self.phase2_aggregation == 'fedpso':
            # FedPSO: Initialize particle swarm for ALL clients
            print(f"  [FedPSO] Initializing particle swarm for all clients...")
            for client in self.clients:
                # 评估初始适应度
                client.current_acc = client.get_validation_accuracy()
                # 初始化pbest
                self.pso_pbest_models[client.id] = copy.deepcopy([p.data.clone().double() for p in client.model.parameters()])
                self.pso_pbest_accs[client.id] = client.current_acc
                # 初始化速度为0
                self.pso_velocities[client.id] = [torch.zeros_like(p.data).double() for p in client.model.parameters()]
            
            # 初始化gbest（从所有客户端中选择）
            if len(self.clients) > 0:
                best_client = max(self.clients, key=lambda c: self.pso_pbest_accs[c.id])
                self.pso_gbest_acc = self.pso_pbest_accs[best_client.id]
                self.pso_gbest_model = copy.deepcopy([p.data.clone().double() for p in best_client.model.parameters()])
                print(f"  [FedPSO] Initial gbest: Client {best_client.id}, acc={self.pso_gbest_acc:.4f}")
        
        print(f"[OK] Phase 2 initialized successfully!")
        print(f"   Algorithm: {self.phase2_aggregation.upper()}")
        print(f"   All clients enter Phase 2: {len(self.clients)} (100%)")
        print(f"   - Already contributed virtual data: {len(self.clients_met_threshold)} ({len(self.clients_met_threshold)*100//self.num_clients}%)")
        print(f"   - Can still contribute (until round 20): {self.num_clients - len(self.clients_met_threshold)} ({(self.num_clients-len(self.clients_met_threshold))*100//self.num_clients}%)")
        print(f"   Virtual data pool size: {len(self.virtual_data_pool)} samples")
        print(f"   Phase 1 will continue in parallel until round 20\n")
    
    def _check_late_arrivals(self, round_num):
        """
        Check if any unqualified clients become qualified during Phase 2 (dynamic virtual data contribution)
        
        Key Design:
        - Phase 2已启动，所有客户端均参与Phase 2训练
        - Phase 1在后台并行：未达标客户端继续尝试达标
        - 服务器每轮检查未达标客户端的准确率
        - 一旦达标（在20轮内）：
          1. 生成虚拟数据并加入虚拟数据池
          2. 标记contributes_virtual_data=True
          3. 重新分发虚拟数据池给所有客户端
        - 第20轮后：
          - Phase 1强制结束
          - 未达标客户端永久不贡献虚拟数据
          - 但仍然继续参与Phase 2训练
        
        This ensures:
        - 所有客户端都参与Phase 2训练（100%）
        - 弱客户端有20轮的时间窗口达标并贡献虚拟数据
        - 20轮后避免无限等待，虚拟数据池固定
        """
        if len(self.clients_met_threshold) >= self.num_clients:
            return  # All clients already qualified
        
        # 第20轮：Phase 1强制结束，未达标客户端永久不贡献虚拟数据
        if round_num >= 20:
            self._finalize_phase1(round_num)
            return  # Phase 1训练窗口已关闭
        
        newly_qualified = []
        
        for client in self.clients:
            # 跳过已达标客户端
            if client.id in self.clients_met_threshold:
                continue
            
            # 评估未达标客户端的准确率
            # Note: In Phase 2, clients will continue training (using mixed data)
            # 我们检查其在验证集上的表现
            client_acc = self.client_accuracies.get(client.id, 0.0)
            
            # 检查是否达标
            if client_acc >= self.current_threshold:
                print(f"\n[>>] Late Arrival Detected! Client {client.id} qualified (was in Phase 1)")
                print(f"   Round: {round_num}")
                print(f"   Accuracy: {client_acc:.4f} (threshold: {self.current_threshold:.4f})")
                
                # 只在首次达标时生成虚拟数据
                if not client.virtual_data_generated:
                    # Generate virtual data (using improved method)
                    print(f"   Generating and locking virtual data for Client {client.id}...")
                    
                    # 检查是否启用差分隐私
                    use_dp = (client.epsilon is not None and client.epsilon > 0 and 
                             client.noise_type is not None and client.noise_type != 'none')
                    
                    if use_dp:
                        client.train_baseline_vae()
                        client.compute_feature_importance()
                    
                    # Generate virtual data (with quality filtering and real distribution sampling)
                    virtual_data = client.generate_virtual_data(
                        confidence_threshold=0.7,
                        use_real_distribution=True,
                        exploration_ratio=0.2
                    )
                    
                    if use_dp:
                        client.add_adaptive_noise_to_virtual_data(strategy='privacy_first')
                    
                    # 加入虚拟数据池
                    self.virtual_data_pool.extend(client.virtual_data)
                    client.virtual_data_generated = True  # 标记虚拟数据已生成并锁定
                    
                    print(f"   [OK] Generated {len(virtual_data)} virtual samples (LOCKED)")
                    print(f"   [OK] Virtual pool size: {len(self.virtual_data_pool)}")
                
                # 首次达标：加入达标集合，开始贡献虚拟数据
                if client.id not in self.clients_met_threshold:
                    self.clients_met_threshold.add(client.id)
                    newly_qualified.append(client.id)
                    
                    # 客户端已在Phase 2，现在标记为贡献虚拟数据
                    client.contributes_virtual_data = True
                    
                    # 为FedPSO初始化pbest（如果需要）
                    if self.phase2_aggregation == 'fedpso' and client.id not in self.pso_pbest_models:
                        client.current_acc = client.get_validation_accuracy()
                        self.pso_pbest_models[client.id] = copy.deepcopy([p.data.clone().double() for p in client.model.parameters()])
                        self.pso_pbest_accs[client.id] = client.current_acc
                        self.pso_velocities[client.id] = [torch.zeros_like(p.data).double() for p in client.model.parameters()]
                    
                    print(f"   [OK] Client {client.id}: NOW contributes virtual data (was non-contributor)")
        
        # If new clients qualified, redistribute virtual data pool to ALL clients
        if len(newly_qualified) > 0:
            print(f"\n📢 Redistributing Virtual Data Pool to All Clients...")
            print(f"   Newly qualified (now contribute virtual data): {newly_qualified}")
            print(f"   Total contributors: {len(self.clients_met_threshold)}/{self.num_clients}")
            print(f"   New virtual data pool size: {len(self.virtual_data_pool)}")
            
            # 重新分发给所有客户端（全员都在Phase 2）
            for client in self.clients:
                client.load_shared_virtual_data(self.virtual_data_pool)
            
            print(f"   [OK] All clients updated with new virtual data pool")
            print(f"   [OK] Non-contributors ({self.num_clients - len(self.clients_met_threshold)}): can still contribute until round 20")
            
            if len(self.clients_met_threshold) == self.num_clients:
                print(f"\n🎉 All clients now contribute virtual data! Full virtual data sharing achieved!\n")
    
    def _finalize_phase1(self, round_num):
        """
        Finalize Phase 1 at round 20 - close the virtual data contribution window
        
        Remaining unqualified clients will:
        - NOT generate virtual data (永久不贡献虚拟数据)
        - Continue participating in Phase 2 training (仍然参与Phase 2训练)
        - Use existing virtual data pool for training (使用现有虚拟数据池)
        
        This ensures Phase 1 training window closes at round 20.
        """
        unqualified_clients = [c for c in self.clients if c.id not in self.clients_met_threshold]
        
        if len(unqualified_clients) == 0:
            return  # 所有客户端已达标
        
        print(f"\n{'='*80}")
        print(f"[Round {round_num}] Phase 1 Training Window CLOSED (20 rounds limit reached)")
        print(f"{'='*80}")
        print(f"Finalizing Phase 1: {len(unqualified_clients)} clients did not contribute virtual data")
        print(f"Note: These clients will continue Phase 2 training but NOT contribute to virtual data pool\n")
        
        for client in unqualified_clients:
            print(f"  Client {client.id}: contributes_virtual_data permanently set to False")
            # 客户端已在Phase 2，仅确认contributes_virtual_data=False
            client.contributes_virtual_data = False
        
        print(f"\n[OK] Phase 1 finalized!")
        print(f"   - Virtual data contributors: {len(self.clients_met_threshold)}/{self.num_clients}")
        print(f"   - Non-contributors (permanent): {len(unqualified_clients)}")
        print(f"   - Virtual data pool size (final): {len(self.virtual_data_pool)}")
        print(f"{'='*80}\n")
    
    # ==================== Phase 2: Training & Aggregation ====================
    
    def train_phase2(self, round_num):
        """
        Phase 2 Training: Algorithm-specific training and aggregation
        
        Key Design: Dual-track training with personalization
        1. Phase 2 clients: Train personalized + global model with virtual data pool
        2. Phase 1 clients: Continue local VAE+classifier training (waiting to qualify)
        3. Check Phase 1 clients each round for qualification → dynamic join Phase 2
        4. Redistribute virtual data pool when new clients join
        
        Supported algorithms:
        1. FedAvg - Standard weighted averaging
        2. FedProx - Proximal term regularization
        3. FedScaffold - Variance reduction with control variates
        """
        phase2_count = len(self.clients_met_threshold)
        phase1_count = self.num_clients - phase2_count
        print(f"\n[Phase 2 - Round {round_num}] Algorithm: {self.phase2_aggregation.upper()}")
        print(f"  Phase 2 clients: {phase2_count} | Phase 1 clients (training): {phase1_count}")
        
        # Step 1: Phase 1客户端继续本地训练（直到25轮或达标）
        phase1_clients = [c for c in self.clients if c.current_phase == 1]
        if len(phase1_clients) > 0:
            print(f"\n  [Phase 1 Parallel Training] {len(phase1_clients)} clients continue local training...")
            for client in phase1_clients:
                # Continue Phase 1 local VAE+classifier training
                accuracy = client.train()  # train_phase1
                self.client_accuracies[client.id] = accuracy
                if client.prototypes:
                    self.client_prototypes[client.id] = client.prototypes
        
        # Step 2: Check for newly qualified clients (Phase 1→Phase 2 dynamic join)
        self._check_late_arrivals(round_num)
        
        # Step 3: Phase 2客户端训练
        # Send global model to Phase 2 clients only
        phase2_clients = [c for c in self.clients if c.current_phase == 2]
        for client in phase2_clients:
            client.set_parameters(self.global_model)
        
        # Select Phase 2 clients for this round
        self.selected_clients = [c for c in self.select_clients() if c.current_phase == 2]
        
        # Update current_num_join_clients to reflect actual Phase 2 participants
        # This prevents sampling errors in receive_models()
        self.current_num_join_clients = len(self.selected_clients)
        
        # Clients train with algorithm-specific logic
        for client in self.selected_clients:
            # Step 1: Train personalized model first (if enabled)
            client.ptrain()
            
            # Step 2: Train global model
            client.train_phase2()
        
        # Receive updated models
        self.receive_models()
        
        # Algorithm-specific aggregation
        if self.phase2_aggregation == 'fedavg':
            self._aggregate_fedavg()
        elif self.phase2_aggregation == 'fedprox':
            self._aggregate_fedprox()
        elif self.phase2_aggregation == 'fedscaffold':
            self._aggregate_scaffold()
        else:
            print(f"Warning: Unknown algorithm '{self.phase2_aggregation}', using FedAvg")
            self._aggregate_fedavg()
        
        print(f"[Phase 2] Aggregation complete: {self.phase2_aggregation.upper()}")
    
    # ==================== Phase 2: Algorithm-Specific Aggregation Methods ====================
    
    def _aggregate_fedavg(self):
        """FedAvg: Standard weighted averaging"""
        self.aggregate_parameters()
        print(f"  [FedAvg] Weighted average aggregation")
    
    def _aggregate_fedprox(self):
        """
        FedProx: Same as FedAvg on server side
        (Proximal term is added in client training)
        """
        self.aggregate_parameters()
        print(f"  [FedProx] Weighted average aggregation (proximal term in clients)")
    
    def _aggregate_moon(self):
        """
        MOON: Standard aggregation
        (Contrastive loss is computed in client training)
        """
        self.aggregate_parameters()
        print(f"  [MOON] Weighted average aggregation (contrastive learning in clients)")
    
    def _aggregate_scaffold(self):
        """
        SCAFFOLD: Aggregation with control variates
        完全按照原框架serverscaffold.py实现
        
        Following original SCAFFOLD implementation:
        - Aggregate model parameters with server learning rate
        - Update server control variates
        """
        # Initialize server control if not exists
        if not hasattr(self, 'c_global'):
            self.c_global = [torch.zeros_like(p.data).double() for p in self.global_model.parameters()]
        
        # Get server learning rate (default 1.0)
        server_lr = getattr(self, 'server_learning_rate', 1.0)
        
        # Use original SCAFFOLD aggregation logic from serverscaffold.py
        # "save GPU memory" version
        global_model = copy.deepcopy(self.global_model).double()
        global_c = copy.deepcopy(self.c_global)
        
        # Compute delta_y and delta_c for each uploaded client (与原框架完全一致)
        for client in self.selected_clients:
            # Use client's delta_yc() method (原框架方式)
            dy, dc = client.delta_yc()
            
            # Aggregate parameters: θ = θ + Σ(dy) * η_s / m
            for server_param, client_param in zip(global_model.parameters(), dy):
                server_param.data += client_param.data.clone() / self.num_join_clients * server_lr
            
            # Aggregate control variates: c = c + Σ(dc) / N
            for server_c, client_delta_c in zip(global_c, dc):
                server_c.data += client_delta_c.data.clone() / self.num_clients
        
        self.global_model = global_model
        self.c_global = global_c
        
        print(f"  [SCAFFOLD] Aggregated with control variates (server_lr={server_lr})")
    
    def _aggregate_perfedavg(self, round_num):
        """
        Per-FedAvg: MAML-based meta-learning aggregation
        
        In theory, Per-FedAvg uses Hessian-vector products.
        Here we use a simplified version with standard aggregation.
        """
        self.aggregate_parameters()
        print(f"  [Per-FedAvg] Meta-learning aggregation (simplified)")
    
    def _aggregate_ditto(self):
        """
        Ditto: Aggregates global model only
        (Personalized models stay local)
        """
        self.aggregate_parameters()
        print(f"  [Ditto] Global model aggregation (personalized models local)")
    
    def _aggregate_fedrep(self):
        """
        FedRep: Only aggregate representation layers (body)
        Head layers stay local
        """
        # Check if model has base-head structure
        if hasattr(self.global_model, 'base') and hasattr(self.global_model, 'head'):
            # Aggregate only base (representation) layers
            assert (len(self.uploaded_models) > 0)
            
            # Weighted averaging of base layers only
            for param in self.global_model.base.parameters():
                param.data.zero_()
            
            for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
                for server_param, client_param in zip(self.global_model.base.parameters(), 
                                                      client_model.base.parameters()):
                    server_param.data += client_param.data.clone() * w
            
            print(f"  [FedRep] Aggregated base layers only (head stays local)")
        else:
            # Fallback: aggregate all parameters
            self.aggregate_parameters()
            print(f"  [FedRep] Model doesn't have base-head, aggregating all parameters")
    
    def _aggregate_fedproto(self):
        """
        FedProto: Aggregate class prototypes (no model parameter aggregation)
        
        Following original FedProto paper
        """
        # Collect prototypes from clients
        uploaded_protos = []
        for client in self.selected_clients:
            if hasattr(client, 'protos') and client.protos is not None:
                uploaded_protos.append(client.protos)
        
        if len(uploaded_protos) == 0:
            print(f"  [FedProto] No prototypes collected, skipping aggregation")
            return
        
        # Aggregate prototypes by class (average)
        global_protos = {}
        all_classes = set()
        for protos in uploaded_protos:
            all_classes.update(protos.keys())
        
        for class_id in all_classes:
            class_protos = []
            for protos in uploaded_protos:
                if class_id in protos:
                    class_protos.append(protos[class_id])
            
            if len(class_protos) > 0:
                global_protos[class_id] = torch.stack(class_protos).mean(0)
        
        # Send global prototypes to all clients
        for client in self.clients:
            client.global_prototypes = global_protos
        
        print(f"  [FedProto] Aggregated {len(global_protos)} class prototypes (no model aggregation)")
    
    def _aggregate_pfedme(self):
        """
        pFedMe: Moreau envelope-based personalization
        完全按照原框架serverpFedMe.py实现
        
        Server aggregates θ (global), then applies beta smoothing with previous model
        Clients maintain personalized ω
        """
        # Save previous global model
        if not hasattr(self, 'previous_global_model'):
            self.previous_global_model = copy.deepcopy(list(self.global_model.parameters()))
        else:
            self.previous_global_model = copy.deepcopy(list(self.global_model.parameters()))
        
        # Standard aggregation first
        self.aggregate_parameters()
        
        # pFedMe specific: Beta aggregation with previous model
        # θ_new = (1-β)*θ_old + β*θ_aggregated
        beta = getattr(self, 'beta', 0.001)
        for pre_param, param in zip(self.previous_global_model, self.global_model.parameters()):
            param.data = (1 - beta)*pre_param.data + beta*param.data
        
        print(f"  [pFedMe] Moreau envelope aggregation with beta smoothing (β={beta})")
    
    def _setup_fedpso_params(self, round_num):
        """
        FedPSO: Setup PSO parameters for clients BEFORE training
        完全按照原框架serverpso.py的逻辑实现
        
        PSO公式 (Kennedy & Eberhart, 1995):
        v_i(t+1) = w*v_i(t) + c1*r1*(pbest_i - x_i(t)) + c2*r2*(gbest - x_i(t))
        x_i(t+1) = x_i(t) + v_i(t+1)
        
        Steps:
        1. 计算当前轮的惯性权重w（线性递减）
        2. 为每个客户端生成随机数r1, r2
        3. 设置PSO参数到客户端
        """
        print(f"  [FedPSO] Setting up PSO parameters (Round {round_num})...")
        
        # Step 1: 计算惯性权重w（线性递减从w_max到w_min）
        # w(t) = w_max - (w_max - w_min) * t / T
        current_w = self.pso_w_max - (self.pso_w_max - self.pso_w_min) * round_num / self.global_rounds
        
        print(f"  [FedPSO] w={current_w:.4f}, c1={self.pso_c1}, c2={self.pso_c2}")
        print(f"  [FedPSO] Current gbest_acc={self.pso_gbest_acc:.4f}")
        
        # Step 2: 为每个Phase 2客户端设置PSO参数
        phase2_clients = [c for c in self.selected_clients if c.current_phase == 2]
        
        for client in phase2_clients:
            # 生成随机数
            r1 = np.random.rand()
            r2 = np.random.rand()
            
            # 设置PSO参数（客户端会在训练前执行PSO更新）
            client.set_pso_parameters(
                w=current_w,
                c1=self.pso_c1,
                c2=self.pso_c2,
                r1=r1,
                r2=r2,
                pbest_model=self.pso_pbest_models[client.id],
                gbest_model=self.pso_gbest_model,
                velocity=self.pso_velocities[client.id]
            )
    
    def _aggregate_fedpso(self, round_num):
        """
        FedPSO: Update pbest, gbest and aggregate
        
        Steps:
        1. 评估每个客户端的新适应度（准确率）
        2. 更新个体最优pbest
        3. 更新全局最优gbest
        4. 保存更新后的速度
        5. 将gbest设置为全局模型
        """
        print(f"  [FedPSO] Evaluating fitness and updating best positions...")
        
        phase2_clients = [c for c in self.selected_clients if c.current_phase == 2]
        
        fitness_values = []
        for client in phase2_clients:
            # Step 1: 评估适应度
            client.current_acc = client.get_validation_accuracy()
            fitness_values.append(client.current_acc)
            
            # Step 2: 更新个体最优pbest
            if client.current_acc > self.pso_pbest_accs[client.id]:
                self.pso_pbest_accs[client.id] = client.current_acc
                self.pso_pbest_models[client.id] = copy.deepcopy([p.data.clone().double() for p in client.model.parameters()])
                print(f"    [+] Client {client.id} updated pbest: {client.current_acc:.4f}")
            
            # Step 3: 更新全局最优gbest
            if client.current_acc > self.pso_gbest_acc:
                self.pso_gbest_acc = client.current_acc
                self.pso_gbest_model = copy.deepcopy([p.data.clone().double() for p in client.model.parameters()])
                print(f"    🌟 New gbest! Client {client.id}, acc: {self.pso_gbest_acc:.4f}")
            
            # Step 4: 保存更新后的速度
            self.pso_velocities[client.id] = copy.deepcopy([v.data.clone().double() for v in client.velocity])
        
        # Step 5: 使用gbest作为全局模型
        if self.pso_gbest_model is not None:
            for global_param, gbest_param in zip(self.global_model.parameters(), self.pso_gbest_model):
                global_param.data = gbest_param.clone()
        
        if len(fitness_values) > 0:
            print(f"  [FedPSO] Fitness stats - best: {max(fitness_values):.4f}, worst: {min(fitness_values):.4f}, mean: {np.mean(fitness_values):.4f}")
        print(f"  [FedPSO] Global model updated with gbest (acc={self.pso_gbest_acc:.4f})")
    
    def _setup_fedgwo_params(self, round_num):
        """
        FedGWO: Setup GWO parameters for clients BEFORE training
        完全按照原框架servergwo.py的train()方法实现
        
        This method is called before client training in train_phase2()
        
        Steps (与servergwo.py完全一致):
        1. 所有参与客户端评估当前准确率
        2. 找出α, β, δ三个领导者 (top-3客户端)
        3. 计算收敛因子 a (从2线性递减到0)
        4. 为每个客户端分配GWO参数
        """
        print(f"  [FedGWO] Setting up GWO parameters (Round {round_num})...")
        
        # Step 1: 所有参与客户端评估当前准确率
        if not hasattr(self, 'client_accuracies'):
            self.client_accuracies = {}
        
        for client in self.selected_clients:
            client.current_acc = client.get_validation_accuracy()
            self.client_accuracies[client.id] = client.current_acc
        
        # Step 2: 找出α, β, δ三个领导者 (top-3客户端)
        self.find_leaders()
        
        # Step 3: 计算收敛因子 a (原始论文: 从2线性递减到0)
        a = 2.0 - 2.0 * round_num / self.global_rounds
        
        print(f"  [FedGWO] a={a:.4f}")
        print(f"    α客户端: ID={self.alpha_id}, acc={self.alpha_acc:.4f}")
        print(f"    β客户端: ID={self.beta_id}, acc={self.beta_acc:.4f}")
        print(f"    δ客户端: ID={self.delta_id}, acc={self.delta_acc:.4f}")
        
        # Step 4: 为每个客户端分配GWO参数
        for client in self.selected_clients:
            # 4.1 获取α, β, δ三个领导者的模型
            alpha_client = self.clients[self.alpha_id]
            beta_client = self.clients[self.beta_id]
            delta_client = self.clients[self.delta_id]
            
            alpha_model = copy.deepcopy(alpha_client.model).double()
            beta_model = copy.deepcopy(beta_client.model).double()
            delta_model = copy.deepcopy(delta_client.model).double()
            
            # 4.2 生成随机系数 (原始论文公式)
            r1, r2 = np.random.rand(), np.random.rand()
            A1 = 2.0 * a * r1 - a  # A ∈ [-a, a]
            C1 = 2.0 * r2          # C ∈ [0, 2]
            
            r1, r2 = np.random.rand(), np.random.rand()
            A2 = 2.0 * a * r1 - a
            C2 = 2.0 * r2
            
            r1, r2 = np.random.rand(), np.random.rand()
            A3 = 2.0 * a * r1 - a
            C3 = 2.0 * r2
            
            # 4.3 设置GWO参数
            client.set_gwo_params(
                alpha_model=alpha_model,
                beta_model=beta_model,
                delta_model=delta_model,
                a=a,
                A1=A1, A2=A2, A3=A3,
                C1=C1, C2=C2, C3=C3
            )
    
    def find_leaders(self):
        """
        找出准确率最高的前三个客户端作为α, β, δ领导者
        与原框架servergwo.py的find_leaders()完全一致
        原始GWO论文: Alpha (最优), Beta (次优), Delta (第三优)
        """
        # 按准确率排序
        sorted_clients = sorted(self.client_accuracies.items(), 
                               key=lambda x: x[1], reverse=True)
        
        if len(sorted_clients) >= 3:
            self.alpha_id, self.alpha_acc = sorted_clients[0]
            self.beta_id, self.beta_acc = sorted_clients[1]
            self.delta_id, self.delta_acc = sorted_clients[2]
        elif len(sorted_clients) == 2:
            self.alpha_id, self.alpha_acc = sorted_clients[0]
            self.beta_id, self.beta_acc = sorted_clients[1]
            self.delta_id, self.delta_acc = sorted_clients[1]  # Beta和Delta相同
        elif len(sorted_clients) == 1:
            self.alpha_id, self.alpha_acc = sorted_clients[0]
            self.beta_id, self.beta_acc = sorted_clients[0]
            self.delta_id, self.delta_acc = sorted_clients[0]  # 全部相同
        else:
            # 默认使用第一个客户端
            self.alpha_id = self.beta_id = self.delta_id = 0
            self.alpha_acc = self.beta_acc = self.delta_acc = 0.0
    
    def _aggregate_fedgwo(self, round_num):
        """
        FedGWO: Standard FedAvg aggregation
        
        Note: GWO position update is done in client training phase via _setup_fedgwo_params().
        Server just does standard weighted averaging here.
        """
        self.aggregate_parameters()
        print(f"  [FedGWO] Standard aggregation (GWO position update done in clients)")

    # ==================== Utility Methods ====================

    def send_models(self):
        """Send global model to all clients (Phase 2 only)"""
        if self.current_phase == 2:
            super().send_models()
            
            # For MOON: Set global_model reference in clients
            if self.phase2_aggregation == 'moon':
                for client in self.clients:
                    client.global_model = copy.deepcopy(self.global_model).double()
            
            # For SCAFFOLD: Set control variates and global model (与原框架serverscaffold.py一致)
            if self.phase2_aggregation == 'scaffold':
                if not hasattr(self, 'c_global'):
                    self.c_global = [torch.zeros_like(p.data).double() for p in self.global_model.parameters()]
                
                for client in self.selected_clients:
                    # Set parameters using client's set_parameters method
                    # This sets both model and control variates (与原框架一致)
                    client.set_parameters(self.global_model, self.c_global)
    
    def receive_models(self):
        """Receive models from clients (Phase 2 only)"""
        if self.current_phase == 2:
            super().receive_models()
    
    def save_results(self):
        """Save training results"""
        print(f"\n[FedGpro] Saving results...")
        print(f"  rs_test_acc entries: {len(self.rs_test_acc)}")
        print(f"  rs_test_auc entries: {len(self.rs_test_auc)}")
        print(f"  rs_train_loss entries: {len(self.rs_train_loss)}")
        
        if len(self.rs_test_acc) == 0:
            print("  WARNING: No evaluation results to save! Training may not have included evaluation rounds.")
            print("  Please check: 1) eval_gap setting, 2) whether evaluation was performed, 3) training logs for errors")
            return
        
        if len(self.rs_test_acc):
            # 创建子目录结构: system/results/{dataset}_{algorithm}_{goal}/
            # CRITICAL: Use original algorithm name (without Phase 2 suffix like "-Ditto")
            algo_folder = f"{self.dataset}_{self.original_algorithm}_{self.goal}"
            result_path = os.path.join("system", "results", algo_folder)
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            result_path = result_path + os.sep
            
            # Save in h5 format (standard format for average_data)
            import h5py
            algo_h5 = f"{self.dataset}_{self.original_algorithm}_{self.goal}_{self.times}"
            file_path = result_path + "{}.h5".format(algo_h5)
            print(f"  Saving to: {file_path}")
            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_test_precision', data=self.rs_test_precision)
                hf.create_dataset('rs_test_recall', data=self.rs_test_recall)
                hf.create_dataset('rs_test_f1', data=self.rs_test_f1)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
            print(f"  [+] Saved h5 file successfully")
            
            # Save training process data to CSV file (for consistency with other algorithms)
            import csv
            csv_file_path = result_path + "{}_training_process.csv".format(algo_h5)
            print(f"  Saving CSV to: {csv_file_path}")
            
            with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['round', 'test_accuracy', 'test_auc', 'test_precision', 'test_recall', 'test_f1', 'train_loss']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for i in range(len(self.rs_test_acc)):
                    writer.writerow({
                        'round': i * self.eval_gap,
                        'test_accuracy': self.rs_test_acc[i],
                        'test_auc': self.rs_test_auc[i] if i < len(self.rs_test_auc) else 0.0,
                        'test_precision': self.rs_test_precision[i] if i < len(self.rs_test_precision) else 0.0,
                        'test_recall': self.rs_test_recall[i] if i < len(self.rs_test_recall) else 0.0,
                        'test_f1': self.rs_test_f1[i] if i < len(self.rs_test_f1) else 0.0,
                        'train_loss': self.rs_train_loss[i] if i < len(self.rs_train_loss) else 0.0
                    })
            print(f"  [+] Saved CSV file successfully")
            
            # Also save in npy format for convenience
            np.save(result_path + "{}_test_acc.npy".format(algo_h5), np.array(self.rs_test_acc))
            np.save(result_path + "{}_test_auc.npy".format(algo_h5), np.array(self.rs_test_auc))
            np.save(result_path + "{}_train_loss.npy".format(algo_h5), np.array(self.rs_train_loss))
            print(f"  [+] Saved npy files successfully")
            
            # Save phase transition info
            phase_info = {
                'transition_round': None,
                'virtual_data_size': len(self.virtual_data_pool),
                'threshold_final': self.current_threshold,
                'best_acc': max(self.rs_test_acc) if self.rs_test_acc else 0.0
            }
            
            # Find transition round
            for i, budget in enumerate(self.Budget):
                if i > 0 and self.current_phase == 2:
                    phase_info['transition_round'] = i
                    break
            
            import json
            with open(result_path + "{}_phase_info.json".format(algo_h5), 'w') as f:
                json.dump(phase_info, f, indent=2)



