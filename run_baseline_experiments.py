#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联邦学习基线实验运行器

功能说明:
- 支持多GPU并行执行，每GPU支持2个并发任务
- 自动检测缺失实验文件并补充运行
- 实时监控实验进度，每20轮输出一次日志
- 运行完成后自动生成Excel汇总报告

支持的算法（可运行）:
- FedAvg: 经典联邦平均算法
- FedProx: 带近端项正则化的联邦学习
- FedProto: 基于原型学习的联邦学习
- FedGpro: 本文提出的方法

已有结果的算法（仅分析）:
- Centralized: 集中式训练（性能上界）
- FedGen: 生成式联邦学习
- FedMoon: 模型对比联邦学习
- FedRep: 表示学习联邦学习
- FedScaffold: 方差缩减联邦学习
- Per-FedAvg: 个性化联邦平均
- FedGwo: 灰狼优化联邦学习
- FedPso: 粒子群优化联邦学习

使用方法:
    python run_baseline_experiments.py           # 运行所有缺失实验
    python run_baseline_experiments.py --check   # 仅检查缺失实验
    python run_baseline_experiments.py --analyze # 仅生成分析报告（含全部12种算法）
"""

import os
import sys
import time
import subprocess
import threading
import re
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty

# 项目根目录
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR / 'system'))

# =============================================================================
# 配置参数
# =============================================================================

# 数据集列表
DATASETS = ['Uci', 'Xinwang']

# 异质性类型
HETEROGENEITY_TYPES = {
    'feature': '特征异质性',
    'label': '标签异质性',
    'quantity': '数量异质性',
    'iid': '独立同分布'
}

# 基线算法列表（完整版，包含所有已有结果的算法）
# 可运行算法：所有联邦学习算法（服务器实现已恢复）
# 仅分析算法：Centralized（使用 centralized_training.py 单独运行）
ALGORITHMS_RUNNABLE = [
    'FedAvg',        # 经典联邦平均
    'FedProx',       # 近端项正则化
    'FedProto',      # 基于原型学习
    'FedGen',        # 生成式联邦学习
    'FedMoon',       # 模型对比联邦学习
    'FedRep',        # 表示学习联邦学习
    'FedScaffold',   # 方差缩减联邦学习
    'Per-FedAvg',    # 个性化联邦平均
    'FedGwo',        # 灰狼优化联邦学习
    'FedPso',        # 粒子群优化联邦学习
    'FedGpro',       # 本文提出的方法
]
ALGORITHMS_ALL = [
    'Centralized',   # 集中式训练（上界）
    'FedAvg',        # 经典联邦平均
    'FedProx',       # 近端项正则化
    'FedProto',      # 基于原型学习
    'FedGen',        # 生成式联邦学习
    'FedMoon',       # 模型对比联邦学习
    'FedRep',        # 表示学习联邦学习
    'FedScaffold',   # 方差缩减联邦学习
    'Per-FedAvg',    # 个性化联邦平均
    'FedGwo',        # 灰狼优化联邦学习
    'FedPso',        # 粒子群优化联邦学习
    'FedGpro',       # 本文提出的方法
]
ALGORITHMS = ALGORITHMS_RUNNABLE  # 默认只运行有实现的算法

# 训练参数
GLOBAL_ROUNDS = 100
LOCAL_EPOCHS = 5

# GPU配置（自动检测）
try:
    import torch
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        GPU_IDS = list(range(num_gpus))
        print(f"[INFO] 检测到 {num_gpus} 个GPU: {GPU_IDS}")
        for i in range(num_gpus):
            print(f"       GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        GPU_IDS = [0]
        print("[WARN] 未检测到CUDA，使用默认GPU 0")
except:
    GPU_IDS = [0]
    print("[WARN] 无法检测GPU，使用默认配置")

SLOTS_PER_GPU = 2

# 默认超参数（用于没有特定配置的算法）
DEFAULT_HYPERPARAMS = {
    'Uci': {'lr': 0.005, 'batch_size': 64, 'local_epochs': 5},
    'Xinwang': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
}

# 超参数配置（完整版，包含所有算法）
HYPERPARAMETERS = {
    'Uci': {
        'feature': {
            # 经典联邦学习算法
            'FedAvg': {'lr': 0.005, 'batch_size': 64, 'local_epochs': 5},
            'FedProx': {'lr': 0.005, 'batch_size': 64, 'local_epochs': 5, 'mu': 0.2},
            'FedProto': {'lr': 0.005, 'batch_size': 64, 'local_epochs': 5, 'lamda': 15},
            # 生成式与对比学习
            'FedGen': {'lr': 0.005, 'batch_size': 64, 'local_epochs': 5, 'generator_lr': 0.001},
            'FedMoon': {'lr': 0.005, 'batch_size': 64, 'local_epochs': 5, 'temperature': 0.5, 'mu': 1.0},
            # 个性化联邦学习
            'FedRep': {'lr': 0.005, 'batch_size': 64, 'local_epochs': 5, 'rep_layers': 2},
            'Per-FedAvg': {'lr': 0.005, 'batch_size': 64, 'local_epochs': 5, 'beta': 0.001, 'lamda': 15},
            # 方差缩减
            'FedScaffold': {'lr': 0.005, 'batch_size': 64, 'local_epochs': 5},
            # 元启发式优化
            'FedGwo': {'lr': 0.005, 'batch_size': 64, 'local_epochs': 5, 'wolves': 5, 'iterations': 10},
            'FedPso': {'lr': 0.005, 'batch_size': 64, 'local_epochs': 5, 'particles': 10, 'w': 0.7, 'c1': 1.5, 'c2': 1.5},
            # 本文方法
            'FedGpro': {'lr': 0.005, 'batch_size': 64, 'local_epochs': 5, 'mu': 0.2, 
                       'plocal_epochs': 3, 'fedgpro_phase2_agg': 'ditto', 
                       'fedgpro_phase2_rounds': 50, 'fedgpro_phase_transition_threshold': 0.70},
        },
        'label': {
            'FedAvg': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5},
            'FedProx': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'mu': 0.1},
            'FedProto': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'lamda': 20},
            'FedGen': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'generator_lr': 0.001},
            'FedMoon': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'temperature': 0.5, 'mu': 1.0},
            'FedRep': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'rep_layers': 2},
            'Per-FedAvg': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'beta': 0.001, 'lamda': 15},
            'FedScaffold': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5},
            'FedGwo': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'wolves': 5, 'iterations': 10},
            'FedPso': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'particles': 10, 'w': 0.7, 'c1': 1.5, 'c2': 1.5},
            'FedGpro': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'mu': 0.1, 
                       'plocal_epochs': 4, 'fedgpro_phase2_agg': 'ditto', 
                       'fedgpro_phase2_rounds': 50, 'fedgpro_phase_transition_threshold': 0.70},
        },
        'quantity': {
            'FedAvg': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5},
            'FedProx': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'mu': 0.05},
            'FedProto': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'lamda': 18},
            'FedGen': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'generator_lr': 0.001},
            'FedMoon': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'temperature': 0.5, 'mu': 1.0},
            'FedRep': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'rep_layers': 2},
            'Per-FedAvg': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'beta': 0.001, 'lamda': 15},
            'FedScaffold': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5},
            'FedGwo': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'wolves': 5, 'iterations': 10},
            'FedPso': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'particles': 10, 'w': 0.7, 'c1': 1.5, 'c2': 1.5},
            'FedGpro': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'mu': 0.05, 
                       'plocal_epochs': 3, 'fedgpro_phase2_agg': 'ditto', 
                       'fedgpro_phase2_rounds': 50, 'fedgpro_phase_transition_threshold': 0.70},
        },
        'iid': {
            'FedAvg': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5},
            'FedProx': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'mu': 0.01},
            'FedProto': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'lamda': 15},
            'FedGen': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'generator_lr': 0.001},
            'FedMoon': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'temperature': 0.5, 'mu': 1.0},
            'FedRep': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'rep_layers': 2},
            'Per-FedAvg': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'beta': 0.001, 'lamda': 15},
            'FedScaffold': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5},
            'FedGwo': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'wolves': 5, 'iterations': 10},
            'FedPso': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'particles': 10, 'w': 0.7, 'c1': 1.5, 'c2': 1.5},
            'FedGpro': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'mu': 0.01, 
                       'plocal_epochs': 2, 'fedgpro_phase2_agg': 'ditto', 
                       'fedgpro_phase2_rounds': 50, 'fedgpro_phase_transition_threshold': 0.70},
        },
    },
    'Xinwang': {
        'feature': {
            'FedAvg': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedProx': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'mu': 0.12},
            'FedProto': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'lamda': 18},
            'FedGen': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'generator_lr': 0.001},
            'FedMoon': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'temperature': 0.5, 'mu': 1.0},
            'FedRep': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'rep_layers': 2},
            'Per-FedAvg': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'beta': 0.001, 'lamda': 15},
            'FedScaffold': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedGwo': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'wolves': 5, 'iterations': 10},
            'FedPso': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'particles': 10, 'w': 0.7, 'c1': 1.5, 'c2': 1.5},
            'FedGpro': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'mu': 0.12, 
                       'plocal_epochs': 3, 'fedgpro_phase2_agg': 'ditto', 
                       'fedgpro_phase2_rounds': 50, 'fedgpro_phase_transition_threshold': 0.70},
        },
        'label': {
            'FedAvg': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedProx': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'mu': 0.08},
            'FedProto': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'lamda': 25},
            'FedGen': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'generator_lr': 0.001},
            'FedMoon': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'temperature': 0.5, 'mu': 1.0},
            'FedRep': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'rep_layers': 2},
            'Per-FedAvg': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'beta': 0.001, 'lamda': 15},
            'FedScaffold': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedGwo': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'wolves': 5, 'iterations': 10},
            'FedPso': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'particles': 10, 'w': 0.7, 'c1': 1.5, 'c2': 1.5},
            'FedGpro': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'mu': 0.08, 
                       'plocal_epochs': 4, 'fedgpro_phase2_agg': 'ditto', 
                       'fedgpro_phase2_rounds': 50, 'fedgpro_phase_transition_threshold': 0.70},
        },
        'quantity': {
            'FedAvg': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedProx': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'mu': 0.08},
            'FedProto': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'lamda': 20},
            'FedGen': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'generator_lr': 0.001},
            'FedMoon': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'temperature': 0.5, 'mu': 1.0},
            'FedRep': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'rep_layers': 2},
            'Per-FedAvg': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'beta': 0.001, 'lamda': 15},
            'FedScaffold': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedGwo': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'wolves': 5, 'iterations': 10},
            'FedPso': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'particles': 10, 'w': 0.7, 'c1': 1.5, 'c2': 1.5},
            'FedGpro': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'mu': 0.08, 
                       'plocal_epochs': 3, 'fedgpro_phase2_agg': 'ditto', 
                       'fedgpro_phase2_rounds': 50, 'fedgpro_phase_transition_threshold': 0.70},
        },
        'iid': {
            'FedAvg': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedProx': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'mu': 0.01},
            'FedProto': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'lamda': 15},
            'FedGen': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'generator_lr': 0.001},
            'FedMoon': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'temperature': 0.5, 'mu': 1.0},
            'FedRep': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'rep_layers': 2},
            'Per-FedAvg': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'beta': 0.001, 'lamda': 15},
            'FedScaffold': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedGwo': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'wolves': 5, 'iterations': 10},
            'FedPso': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'particles': 10, 'w': 0.7, 'c1': 1.5, 'c2': 1.5},
            'FedGpro': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'mu': 0.01, 
                       'plocal_epochs': 2, 'fedgpro_phase2_agg': 'ditto', 
                       'fedgpro_phase2_rounds': 50, 'fedgpro_phase_transition_threshold': 0.70},
        },
    },
}

# =============================================================================
# 全局状态
# =============================================================================

progress_lock = threading.Lock()
gpu_status = {}
completed_count = 0
failed_count = 0
total_experiments = 0
task_queue = None
results_list = []
results_lock = threading.Lock()

for gpu_id in GPU_IDS:
    for slot_id in range(SLOTS_PER_GPU):
        gpu_status[(gpu_id, slot_id)] = None

# =============================================================================
# 工具函数
# =============================================================================

def _ts():
    """获取时间戳"""
    return datetime.now().strftime('%H:%M:%S')


def check_missing_experiments():
    """检查缺失的实验"""
    print("\n" + "="*80)
    print("检查缺失的实验...")
    print("="*80)
    
    missing = []
    for dataset in DATASETS:
        for hetero in HETEROGENEITY_TYPES.keys():
            for algo in ALGORITHMS:
                results_dir = BASE_DIR / 'system' / 'results' / f"{dataset}_{algo}_{hetero}"
                file_prefix = f"{dataset}_{algo}_{hetero}"
                
                if not results_dir.exists():
                    missing.append((dataset, hetero, algo, 0))
                    continue
                
                completed_files = list(results_dir.glob(f"{file_prefix}_*.h5"))
                if len(completed_files) < 5:
                    missing.append((dataset, hetero, algo, len(completed_files)))
    
    if missing:
        print(f"\n缺失实验数: {len(missing)}")
        print(f"{'数据集':<10} {'异质性':<10} {'算法':<20} {'进度'}")
        print("-" * 60)
        for dataset, hetero, algo, completed in missing:
            print(f"{dataset:<10} {hetero:<10} {algo:<20} {completed}/5")
    else:
        print("\n[INFO] 所有实验已完成！")
    
    print("="*80 + "\n")
    return missing


def build_command(dataset, algorithm, hetero_type, gpu_id):
    """构建运行命令"""
    params = HYPERPARAMETERS[dataset][hetero_type][algorithm]
    goal_name = hetero_type
    save_folder = f'system/models/{dataset}_{algorithm}_{hetero_type}'
    
    cmd = [
        'python', '-u', 'system/main.py',
        '-data', dataset, '-m', 'credit', '-algo', algorithm,
        '-did', str(gpu_id), '-gr', str(GLOBAL_ROUNDS),
        '-nc', '10', '-ls', str(params.get('local_epochs', LOCAL_EPOCHS)),
        '-lr', str(params['lr']), '-lbs', str(params['batch_size']),
        '-t', '5', '-go', goal_name, '-sfn', save_folder,
    ]
    
    if 'mu' in params:
        cmd.extend(['-mu', str(params['mu'])])
    if 'plocal_epochs' in params:
        cmd.extend(['-pls', str(params['plocal_epochs'])])
    if 'lamda' in params:
        cmd.extend(['-lam', str(params['lamda'])])
    if 'fedgpro_phase2_agg' in params:
        cmd.extend(['--fedgpro_phase2_agg', params['fedgpro_phase2_agg']])
    if 'fedgpro_phase2_rounds' in params:
        cmd.extend(['--fedgpro_phase2_rounds', str(params['fedgpro_phase2_rounds'])])
    if 'fedgpro_phase_transition_threshold' in params:
        cmd.extend(['--fedgpro_phase_transition_threshold', str(params['fedgpro_phase_transition_threshold'])])
    
    return cmd


def worker_thread(gpu_id, slot_id):
    """工作线程"""
    global completed_count, failed_count, task_queue
    
    while True:
        try:
            task = task_queue.get(timeout=1)
            if task is None:
                break
            
            dataset, hetero, algo, exp_id, current_time = task
            
            with progress_lock:
                gpu_status[(gpu_id, slot_id)] = {
                    'dataset': dataset, 'hetero': hetero, 'algo': algo,
                    'start_time': time.time(), 'last_round': 0, 'current_time': current_time
                }
            
            print(f"[{_ts()}] [START] GPU{gpu_id}-{slot_id}: {dataset}-{hetero}-{algo} [第{current_time+1}次]")
            
            cmd = build_command(dataset, algo, hetero, gpu_id)
            start_time = time.time()
            success = False
            elapsed = 0
            
            try:
                logs_dir = BASE_DIR / 'logs'
                logs_dir.mkdir(exist_ok=True)
                log_file = logs_dir / f"{dataset}_{algo}_{hetero}.log"
                
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding='utf-8', errors='replace'
                )
                
                with open(log_file, 'w', encoding='utf-8') as f:
                    last_round = -1
                    for line in process.stdout:
                        f.write(line)
                        f.flush()
                        
                        match = re.search(r'Round[:\s]+(\d+)', line, re.IGNORECASE)
                        if match:
                            round_num = int(match.group(1))
                            with progress_lock:
                                if gpu_status.get((gpu_id, slot_id)):
                                    gpu_status[(gpu_id, slot_id)]['last_round'] = round_num
                            
                            if round_num > 0 and round_num % 20 == 0 and round_num != last_round:
                                last_round = round_num
                                elapsed_min = (time.time() - start_time) / 60
                                print(f"  [{_ts()}] {dataset}-{hetero}-{algo} | Round {round_num}/{GLOBAL_ROUNDS} | {elapsed_min:.1f}min")
                
                returncode = process.wait(timeout=7200)
                elapsed = time.time() - start_time
                
                if returncode == 0:
                    success = True
                    with progress_lock:
                        completed_count += 1
                    print(f"[{_ts()}] [DONE] {dataset}-{hetero}-{algo} [第{current_time+1}次] ({elapsed/60:.1f}min)")
                else:
                    with progress_lock:
                        failed_count += 1
                    print(f"[{_ts()}] [FAIL] {dataset}-{hetero}-{algo} [第{current_time+1}次]")
            
            except subprocess.TimeoutExpired:
                process.kill()
                elapsed = time.time() - start_time
                with progress_lock:
                    failed_count += 1
                print(f"[{_ts()}] [TIMEOUT] {dataset}-{hetero}-{algo}")
            except Exception as e:
                with progress_lock:
                    failed_count += 1
                print(f"[{_ts()}] [ERROR] {dataset}-{hetero}-{algo}: {e}")
            
            with progress_lock:
                gpu_status[(gpu_id, slot_id)] = None
            
            with results_lock:
                results_list.append({'dataset': dataset, 'hetero': hetero, 'algo': algo, 'success': success})
            
            task_queue.task_done()
        
        except Empty:
            continue


def run_experiments():
    """运行实验"""
    global total_experiments, task_queue
    
    print("\n" + "="*80)
    print("联邦学习基线实验")
    print("="*80)
    print(f"并发数: {len(GPU_IDS)} GPU x {SLOTS_PER_GPU} = {len(GPU_IDS) * SLOTS_PER_GPU}")
    print(f"数据集: {', '.join(DATASETS)}")
    print(f"算法: {', '.join(ALGORITHMS)}")
    print("="*80)
    
    missing = check_missing_experiments()
    if not missing:
        print("所有实验已完成！")
        return
    
    task_queue = Queue()
    for i, (dataset, hetero, algo, completed) in enumerate(missing):
        task_queue.put((dataset, hetero, algo, i, completed))
    
    total_experiments = len(missing)
    print(f"\n待运行实验: {total_experiments}\n")
    
    threads = []
    for gpu_id in GPU_IDS:
        for slot_id in range(SLOTS_PER_GPU):
            t = threading.Thread(target=worker_thread, args=(gpu_id, slot_id), daemon=True)
            t.start()
            threads.append(t)
    
    start_time = time.time()
    while True:
        time.sleep(10)
        with progress_lock:
            running = sum(1 for s in gpu_status.values() if s)
            comp, fail = completed_count, failed_count
            remain = total_experiments - comp - fail
        
        if remain == 0 and running == 0:
            break
        
        print(f"\n[{_ts()}] 完成:{comp} 失败:{fail} 运行:{running} 剩余:{remain}")
    
    for _ in range(len(GPU_IDS) * SLOTS_PER_GPU):
        task_queue.put(None)
    for t in threads:
        t.join()
    
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print(f"完成! 耗时:{total_time/3600:.2f}h 成功:{completed_count}/{total_experiments}")
    print("="*80)
    
    analyze_baseline_results()


def analyze_baseline_results():
    """生成分析报告"""
    try:
        from system.utils.analyze_results import main as analyze_main
        analyze_main()
        print("\n[DONE] 分析完成! Excel: system/results/汇总/")
    except Exception as e:
        print(f"\n[ERROR] 分析失败: {e}")


def check_centralized_completed():
    """检查中心化训练完成状态"""
    for dataset in DATASETS:
        results_dir = BASE_DIR / 'system' / 'results' / f"{dataset}_Centralized"
        if not results_dir.exists() or len(list(results_dir.glob("*.h5"))) < 5:
            return False
    return True


def run_centralized_training():
    """运行中心化训练"""
    print("\n" + "="*80)
    print("步骤 1: 中心化训练")
    print("="*80)
    
    if check_centralized_completed():
        print("[INFO] 已完成，跳过")
        return True
    
    script = BASE_DIR / 'system' / 'centralized_training.py'
    if script.exists():
        return subprocess.run(['python', str(script)], cwd=str(BASE_DIR)).returncode == 0
    print("[WARN] 脚本不存在，跳过")
    return True


def main():
    """主入口"""
    import argparse
    parser = argparse.ArgumentParser(description='联邦学习基线实验')
    parser.add_argument('--check', action='store_true', help='检查缺失实验')
    parser.add_argument('--analyze', action='store_true', help='仅生成报告')
    parser.add_argument('--skip-centralized', action='store_true', help='跳过中心化训练')
    args = parser.parse_args()
    
    if args.check:
        check_missing_experiments()
    elif args.analyze:
        analyze_baseline_results()
    else:
        if not args.skip_centralized:
            run_centralized_training()
        print("\n" + "="*80)
        print("步骤 2: 联邦学习实验")
        print("="*80)
        run_experiments()


if __name__ == '__main__':
    main()
