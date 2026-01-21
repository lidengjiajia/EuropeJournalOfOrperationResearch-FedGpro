#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import torch
from torch import nn
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging

# 基线算法导入
from flcore.servers.serveravg import FedAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.serverproto import FedProto
from flcore.servers.servergpro import FedGpro
from flcore.servers.servergen import FedGen
from flcore.servers.servermoon import MOON
from flcore.servers.serverrep import FedRep
from flcore.servers.serverscaffold import SCAFFOLD
from flcore.servers.serverperavg import PerAvg
from flcore.servers.servergwo import FedGWO
from flcore.servers.serverpso import FedPSO

from flcore.trainmodel.models import *
from flcore.trainmodel.credit import *

from utils.result_utils import average_data
try:
    from utils.mem_utils import MemReporter
except ImportError:
    MemReporter = None
    print("Warning: MemReporter requires calmsize. Install with: pip install calmsize")

try:
    from utils.plot_utils import plot_training_results
except ImportError:
    plot_training_results = None
    print("Warning: plot_utils requires matplotlib. Install with: pip install matplotlib")

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)


def run(args):

    time_list = []
    reporter = MemReporter() if MemReporter else None
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Auto-configure num_classes for binary classification datasets
        credit_datasets = ['Uci', 'Xinwang', 'GiveMeSomeCredit', 'EuropeCreditCardFraud']
        if args.dataset in credit_datasets and args.num_classes != 2:
            print(f"Warning: Dataset {args.dataset} requires 2 classes. Auto-correcting from {args.num_classes} to 2.")
            args.num_classes = 2

        # Auto-configure num_clients from config.json
        dataset_normalized = args.dataset.capitalize()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        config_path = os.path.join(project_root, 'dataset', dataset_normalized, 'config.json')
        if os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'num_clients' in config and args.num_clients != config['num_clients']:
                        print(f"Auto-correcting num_clients from {args.num_clients} to {config['num_clients']} (from config.json)")
                        args.num_clients = config['num_clients']
            except Exception as e:
                print(f"[WARNING] Failed to read num_clients from config.json: {e}")

        # Generate args.model
        if model_str == "MLR": # convex
            if "MNIST" in args.dataset:
                args.model = Mclr_Logistic(1*28*28, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = Mclr_Logistic(3*32*32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

        elif model_str == "CNN": # non-convex
            if "MNIST" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "Omniglot" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
                # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
            elif "Digit5" in args.dataset:
                args.model = Digit5CNN().to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        elif model_str == "DNN": # non-convex
            if "MNIST" in args.dataset:
                args.model = DNN(1*28*28, 100, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = DNN(3*32*32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)
        
        elif model_str == "ResNet18":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = torchvision.models.resnet18(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
            # args.model = resnet18(num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)
        
        elif model_str == "ResNet10":
            args.model = resnet10(num_classes=args.num_classes).to(args.device)
        
        elif model_str == "ResNet34":
            args.model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "AlexNet":
            args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = alexnet(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
        elif model_str == "GoogleNet":
            args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False, 
                                                      num_classes=args.num_classes).to(args.device)
            
            # args.model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "MobileNet":
            args.model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = mobilenet_v2(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
        elif model_str == "LSTM":
            args.model = LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes).to(args.device)

        elif model_str == "BiLSTM":
            args.model = BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.feature_dim, 
                                                   output_size=args.num_classes, num_layers=1, 
                                                   embedding_dropout=0, lstm_dropout=0, attention_dropout=0, 
                                                   embedding_length=args.feature_dim).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes).to(args.device)

        elif model_str == "TextCNN":
            args.model = TextCNN(hidden_dim=args.feature_dim, max_len=args.max_len, vocab_size=args.vocab_size, 
                                 num_classes=args.num_classes).to(args.device)

        elif model_str == "Transformer":
            args.model = TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2, 
                                          num_classes=args.num_classes, max_len=args.max_len).to(args.device)
        
        elif model_str == "AmazonMLP":
            args.model = AmazonMLP().to(args.device)

        elif model_str == "HARCNN":
            if args.dataset == 'HAR':
                args.model = HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9), 
                                    pool_kernel_size=(1, 2)).to(args.device)
            elif args.dataset == 'PAMAP2':
                args.model = HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9), 
                                    pool_kernel_size=(1, 2)).to(args.device)

        elif model_str == "credit_uci":
            args.model = UciCreditNet(num_classes=args.num_classes).to(args.device)

        elif model_str == "credit_xinwang":
            # Load actual feature dimension from config.json (after toad filtering)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            config_path = os.path.join(project_root, 'dataset', 'Xinwang', 'config.json')
            xinwang_input_dim = 38  # Default: 38 features after TOAD filtering
            
            if os.path.exists(config_path):
                try:
                    import json
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        if 'feature_dim' in config:
                            xinwang_input_dim = config['feature_dim']
                            print(f"[INFO] Loaded Xinwang feature_dim from config: {xinwang_input_dim}")
                        else:
                            print(f"[WARNING] feature_dim not found in config. Using default: {xinwang_input_dim}")
                except Exception as e:
                    print(f"[WARNING] Failed to read config.json: {e}. Using default input_dim={xinwang_input_dim}")
            else:
                print(f"[WARNING] Config file not found. Using default input_dim={xinwang_input_dim}")
            
            args.model = XinwangCreditNet(input_dim=xinwang_input_dim, num_classes=args.num_classes).to(args.device)

        elif model_str == "credit_givemesomecredit":
            args.model = GiveMeSomeCreditNet(input_dim=10, num_classes=args.num_classes).to(args.device)

        elif model_str == "credit_europecreditcardfraud":
            args.model = EuropeCreditCardFraudNet(input_dim=30, num_classes=args.num_classes).to(args.device)

        elif model_str == "credit":
            # Auto-select credit model based on dataset
            if args.dataset.lower() == "uci":
                args.model = UciCreditNet(num_classes=args.num_classes).to(args.device)
                print(f"Auto-selected UciCreditNet for dataset {args.dataset}")
            elif args.dataset.lower() == "xinwang":
                # Load actual feature dimension from config.json (after toad filtering)
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(script_dir)
                config_path = os.path.join(project_root, 'dataset', 'Xinwang', 'config.json')
                xinwang_input_dim = 38  # Default: 38 features after TOAD filtering
                
                if os.path.exists(config_path):
                    try:
                        import json
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            if 'feature_dim' in config:
                                xinwang_input_dim = config['feature_dim']
                                print(f"Loaded Xinwang feature_dim={xinwang_input_dim} from config.json")
                    except Exception as e:
                        print(f"[WARNING] Failed to read config.json: {e}. Using default input_dim={xinwang_input_dim}")
                else:
                    print(f"[WARNING] Config file not found. Using default input_dim={xinwang_input_dim}")
                
                args.model = XinwangCreditNet(input_dim=xinwang_input_dim, num_classes=args.num_classes).to(args.device)
                print(f"Auto-selected XinwangCreditNet for dataset {args.dataset}")
            elif args.dataset.lower() == "givemesomecredit":
                args.model = GiveMeSomeCreditNet(input_dim=10, num_classes=args.num_classes).to(args.device)
                print(f"Auto-selected GiveMeSomeCreditNet for dataset {args.dataset}")
            elif args.dataset.lower() == "europecreditcardfraud":
                args.model = EuropeCreditCardFraudNet(input_dim=30, num_classes=args.num_classes).to(args.device)
                print(f"Auto-selected EuropeCreditCardFraudNet for dataset {args.dataset}")
            else:
                raise NotImplementedError(f"Model 'credit' does not support dataset '{args.dataset}'. Supported: uci, xinwang, givemesomecredit, europecreditcardfraud")

        else:
            raise NotImplementedError

        print(args.model)

        # Extract base algorithm name (handle suffixes like _feature, _label, _iid, _quantity)
        # Example: FedAvg_feature -> FedAvg, FedGpro_label -> FedGpro, FedAvg_quantity -> FedAvg
        # Also handle ablation experiment names: FedGpro_HP-Baseline_feature -> FedGpro
        base_algorithm = args.algorithm
        if '_' in args.algorithm:
            # Check if suffix is heterogeneity type (feature, label, iid, quantity)
            parts = args.algorithm.rsplit('_', 1)
            if len(parts) == 2 and parts[1] in ['feature', 'label', 'iid', 'quantity']:
                base_algorithm = parts[0]
                # Handle ablation experiment format: FedGpro_VariantName_hetero -> FedGpro
                # Example: FedGpro_HP-Baseline_feature -> FedGpro_HP-Baseline -> FedGpro
                if '_' in base_algorithm and base_algorithm.startswith('FedGpro_'):
                    base_algorithm = 'FedGpro'
        
        # Algorithm name mapping (for backward compatibility)
        algorithm_mapping = {}
        if base_algorithm in algorithm_mapping:
            base_algorithm = algorithm_mapping[base_algorithm]
        
        # Select algorithm (using base_algorithm for matching)
        if base_algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)

        elif base_algorithm == "FedProx":
            server = FedProx(args, i)

        elif base_algorithm == "FedProto":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedProto(args, i)

        elif base_algorithm == "FedGpro" or base_algorithm.startswith("FedGpro-"):
            server = FedGpro(args, i)

        elif base_algorithm == "FedGen":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGen(args, i)

        elif base_algorithm == "FedMoon" or base_algorithm == "MOON":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = MOON(args, i)

        elif base_algorithm == "FedRep":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedRep(args, i)

        elif base_algorithm == "FedScaffold" or base_algorithm == "SCAFFOLD":
            server = SCAFFOLD(args, i)

        elif base_algorithm == "Per-FedAvg" or base_algorithm == "PerAvg":
            server = PerAvg(args, i)

        elif base_algorithm == "FedGwo":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGWO(args, i)

        elif base_algorithm == "FedPso":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPSO(args, i)

        else:
            raise NotImplementedError(f"Algorithm '{base_algorithm}' is not implemented")

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    # Auto-generate visualization plots
    if plot_training_results:
        print("\nGenerating training result visualizations...")
        result_subdir = f"{args.dataset}_{args.algorithm}_{args.goal}"
        for i in range(args.times):
            result_filename = f"{args.dataset}_{args.algorithm}_{args.goal}_{i}.h5"
            try:
                plot_training_results(result_filename, result_subdir=result_subdir, show_plot=False)
            except Exception as e:
                print(f"[WARNING] Plotting failed ({result_filename}): {e}")
    else:
        print("\n[WARNING] Skipping visualization (matplotlib not installed)")

    print("\nAll done!")

    if reporter:
        reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="MNIST")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="CNN")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-tc', "--top_cnt", type=int, default=100, 
                        help="For auto_break")
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=80, 
                        help="Set this for text tasks. 80 for Shakespeare. 32000 for AG_News and SogouNews.")
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP / GPFL / FedCAC
    parser.add_argument('-bt', "--beta", type=float, default=0.0)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.0)
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # FedMTL
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0, 
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # APFL / FedCross
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_epochs", type=int, default=1)
    # MOON / FedCAC / FedLC
    parser.add_argument('-tau', "--tau", type=float, default=1.0)
    # FedBABU
    parser.add_argument('-fte', "--fine_tuning_epochs", type=int, default=10)
    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)
    # SCAFFOLD / FedGH
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-graind than its original paper.")
    # FedKD
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.005)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)
    # FedDBE
    parser.add_argument('-mo', "--momentum", type=float, default=0.1)
    parser.add_argument('-klw', "--kl_weight", type=float, default=0.0)

    # FedCross
    parser.add_argument('-fsb', "--first_stage_bound", type=int, default=0)
    parser.add_argument('-ca', "--fedcross_alpha", type=float, default=0.99)
    parser.add_argument('-cmss', "--collaberative_model_select_strategy", type=int, default=1)

    # FedGpro: Federated Global Prototype Learning
    parser.add_argument('--fedgpro_phase', type=int, default=1,
                        help='Current phase (1 or 2)')
    parser.add_argument('--fedgpro_use_vae', type=lambda x: str(x).lower() == 'true', default=True,
                        help='Enable VAE generation (True/False, for ablation study)')
    parser.add_argument('--fedgpro_use_prototype', type=lambda x: str(x).lower() == 'true', default=True,
                        help='Enable prototype loss (True/False, for ablation study)')
    parser.add_argument('--fedgpro_epsilon', type=float, default=None,
                        help='Differential privacy budget per client (None=disabled, >0=enabled)')
    parser.add_argument('--fedgpro_noise_type', type=str, default=None, 
                        choices=[None, 'laplace', 'gaussian', 'none'],
                        help='Type of DP noise (None or "none"=disabled, "laplace", "gaussian")')
    parser.add_argument('--fedgpro_delta', type=float, default=1e-5,
                        help='Delta parameter for Gaussian DP')
    parser.add_argument('--fedgpro_adaptive_noise', type=lambda x: str(x).lower() == 'true', default=False,
                        help='Use adaptive noise based on feature importance (True/False, for ablation study)')
    parser.add_argument('--fedgpro_adaptive_strategy', type=str, default='balanced',
                        choices=['privacy_first', 'utility_first', 'balanced', 'hybrid'],
                        help='Adaptive noise strategy: privacy_first, utility_first, balanced, hybrid (for ablation study)')
    parser.add_argument('--fedgpro_utility_weight', type=float, default=1.0,
                        help='Utility weight for utility_first strategy [0.1-1.0]. Controls how much feature importance affects noise: 0.1=weak utility preservation (near-uniform noise), 1.0=strong utility preservation (full importance-based adjustment)')
    
    # FedGpro: Importance-Aware Adaptive Differential Privacy (IA-ADP)
    parser.add_argument('--fedgpro_use_iadp', type=lambda x: str(x).lower() == 'true', default=False,
                        help='Enable Importance-Aware Adaptive DP (True/False). 重要性感知自适应差分隐私')
    parser.add_argument('--fedgpro_iadp_alpha', type=float, default=0.3,
                        help='IA-ADP alpha parameter [0.1-0.5]. α=0.1: aggressive utility preservation; α=0.5: balanced')
    parser.add_argument('--fedgpro_iadp_importance_method', type=str, default='vae_contrast',
                        choices=['vae_contrast', 'gradient', 'weight', 'hybrid'],
                        help='Feature importance method: vae_contrast (VAE comparison, recommended), gradient, weight, hybrid')
    parser.add_argument('--fedgpro_iadp_importance_momentum', type=float, default=0.9,
                        help='Exponential moving average momentum for importance smoothing [0-1]')
    parser.add_argument('--fedgpro_iadp_privacy_priority', type=lambda x: str(x).lower() == 'true', default=False,
                        help='Privacy-First vs Utility-First. True: important features get MORE noise (privacy-first); False: important features get LESS noise (utility-first, default)')
    
    parser.add_argument('--fedgpro_lambda_cls', type=float, default=10.0,
                        help='Weight for classification loss (Optimized: 1.0→10.0, stronger focus on classification)')
    parser.add_argument('--fedgpro_lambda_recon', type=float, default=1.0,
                        help='Weight for VAE reconstruction loss')
    parser.add_argument('--fedgpro_lambda_kl', type=float, default=0.1,
                        help='Weight for KL divergence loss (Optimized: 0.01 to 0.1, +2.73 percent F1)')
    parser.add_argument('--fedgpro_lambda_proto', type=float, default=0.1,
                        help='Weight for prototype loss (Optimized: kept at 0.1)')
    parser.add_argument('--fedgpro_proto_momentum', type=float, default=0.95,
                        help='EMA momentum for prototype update (Optimized: 0.9→0.95, better stability)')
    parser.add_argument('--fedgpro_latent_dim', type=int, default=None,
                        help='VAE latent dimension (auto if None)')
    parser.add_argument('--fedgpro_vae_lr', type=float, default=0.001,
                        help='Learning rate for VAE optimizer')
    parser.add_argument('--fedgpro_proto_align_lr', type=float, default=0.001,
                        help='Learning rate for prototype alignment layer optimizer')
    parser.add_argument('--fedgpro_threshold_min', type=float, default=0.60,
                        help='Minimum threshold value (client forced training for first 10 rounds)')
    parser.add_argument('--fedgpro_phase_transition_threshold', type=float, default=0.70,
                        help='Percentage of clients that must meet threshold to trigger phase transition (default: 0.70 = 70 percent, Phase 1 max 20 rounds)')
    parser.add_argument('--fedgpro_phase2_agg', type=str, default='fedpso',
                        choices=['fedavg', 'fedcs', 'fedprox', 'fedgwo', 'fedpso', 'fedwoa', 'fedproto', 'moon', 'scaffold', 'perfedavg', 'ditto', 'fedrep', 'pfedme'],
                        help='Phase 2 aggregation algorithm: fedpso (default), fedavg, fedcs, fedprox, fedgwo, fedwoa, fedproto, moon, scaffold, perfedavg, ditto, fedrep, pfedme')
    parser.add_argument('--fedgpro_phase2_rounds', type=int, default=50,
                        help='Max rounds for Phase 2 (default: 50 rounds, Phase 1 max 20 rounds, total ~70 rounds)')
    parser.add_argument('--reserved_clients', type=str, default='',
                        help='Comma-separated list of client IDs to reserve for generalization testing (e.g., "0,1,2")')
    
    # FedGWO (Grey Wolf Optimizer) parameters for Phase 2
    parser.add_argument('--gwo_alpha_decay', type=float, default=0.015,
                        help='GWO alpha decay rate (Optimized: 0.01→0.015, balanced exploration)')
    
    # FedPSO (Particle Swarm Optimization) parameters for Phase 2
    parser.add_argument('--pso_w_max', type=float, default=0.9,
                        help='PSO maximum inertia weight (default: 0.9)')
    parser.add_argument('--pso_w_min', type=float, default=0.4,
                        help='PSO minimum inertia weight (default: 0.4)')
    parser.add_argument('--pso_c1', type=float, default=2.0,
                        help='PSO cognitive parameter (default: 2.0)')
    parser.add_argument('--pso_c2', type=float, default=2.0,
                        help='PSO social parameter (default: 2.0)')
    parser.add_argument('--pso_v_max', type=float, default=0.5,
                        help='PSO maximum velocity ratio (default: 0.5)')
    
    # FedCS parameters for Phase 2 (only used if fedgpro_phase2_agg=fedcs)
    parser.add_argument('--cs_f_max', type=float, default=2.0,
                        help='CSA max flight length')
    parser.add_argument('--cs_f_min', type=float, default=0.1,
                        help='CSA min flight length')
    parser.add_argument('--cs_AP_max', type=float, default=0.3,
                        help='CSA max awareness probability')
    parser.add_argument('--cs_AP_min', type=float, default=0.1,
                        help='CSA min awareness probability')


    args = parser.parse_args()
    
    # Parse reserved_clients string to list
    if args.reserved_clients:
        args.reserved_clients = [int(x.strip()) for x in args.reserved_clients.split(',') if x.strip()]
    else:
        args.reserved_clients = []

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    for arg in vars(args):
        print(arg, '=',getattr(args, arg))
    print("=" * 50)

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA],
    #     profile_memory=True, 
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    #     ) as prof:
    # with torch.autograd.profiler.profile(profile_memory=True) as prof:
    run(args)

    
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")
