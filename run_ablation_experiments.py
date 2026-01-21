#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联邦学习消融实验运行器

功能说明:
- 支持多GPU并行执行，每GPU支持2个并发任务
- 自动检测缺失实验并补充运行
- 实时监控进度，每20轮输出日志
- 运行完成后自动生成Excel报告

消融配置:
1. 组件消融: 测试VAE生成数据和原型学习的贡献
2. 隐私消融: 测试不同隐私预算和自适应加密策略
3. 泛化消融: 测试模型在新客户端的泛化能力

使用方法:
    python run_ablation_experiments.py           # 运行所有缺失实验
    python run_ablation_experiments.py --check   # 仅检查缺失实验
    python run_ablation_experiments.py --analyze # 仅生成分析报告
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

DATASETS = ['Uci', 'Xinwang']
HETEROGENEITY_TYPES = {'feature': '特征异质性', 'label': '标签异质性',
                       'quantity': '数量异质性', 'iid': '独立同分布'}

GLOBAL_ROUNDS = 100
LOCAL_EPOCHS = 5

# GPU配置
try:
    import torch
    gpu_count = torch.cuda.device_count()
    if gpu_count > 0:
        GPU_IDS = list(range(gpu_count))
        print(f"[INFO] 检测到 {gpu_count} 个GPU")
    else:
        GPU_IDS = [0]
        print("[WARN] 未检测到GPU")
except:
    GPU_IDS = [0]

SLOTS_PER_GPU = 2

# 超参数
HYPERPARAMETERS = {
    'Uci': {
        'feature': {'batch_size': 64, 'learning_rate': 0.005, 'num_clients': 10},
        'label': {'batch_size': 64, 'learning_rate': 0.007, 'num_clients': 10},
        'quantity': {'batch_size': 64, 'learning_rate': 0.007, 'num_clients': 10},
        'iid': {'batch_size': 64, 'learning_rate': 0.007, 'num_clients': 10},
    },
    'Xinwang': {
        'feature': {'batch_size': 128, 'learning_rate': 0.006, 'num_clients': 10},
        'label': {'batch_size': 128, 'learning_rate': 0.006, 'num_clients': 10},
        'quantity': {'batch_size': 128, 'learning_rate': 0.006, 'num_clients': 10},
        'iid': {'batch_size': 128, 'learning_rate': 0.006, 'num_clients': 10},
    },
}

# 消融配置
ABLATION_CONFIGS = {
    # 组件消融
    'Full_Model': {
        'fedgpro_use_vae': 'True',
        'fedgpro_use_prototype': 'True',
        'fedgpro_phase2_agg': 'fedavg',
        'fedgpro_lambda_proto': '0.3',
        'fedgpro_phase_transition_threshold': '0.70',
        'fedgpro_phase2_rounds': '50'
    },
    'No_VAE_Generation': {
        'fedgpro_use_vae': 'False',
        'fedgpro_use_prototype': 'True',
        'fedgpro_phase2_agg': 'fedavg',
        'fedgpro_lambda_proto': '0.3',
        'fedgpro_phase_transition_threshold': '0.70',
        'fedgpro_phase2_rounds': '50'
    },
    'No_Prototype': {
        'fedgpro_use_vae': 'True',
        'fedgpro_use_prototype': 'False',
        'fedgpro_phase2_agg': 'fedavg',
        'fedgpro_phase_transition_threshold': '0.70',
        'fedgpro_phase2_rounds': '50'
    },
    
    # 隐私预算消融
    'Privacy_Epsilon_1.0': {
        'fedgpro_use_vae': 'True',
        'fedgpro_use_prototype': 'True',
        'fedgpro_phase2_agg': 'fedavg',
        'fedgpro_lambda_proto': '0.3',
        'fedgpro_epsilon': '1.0',
        'fedgpro_noise_type': 'laplace',
        'fedgpro_use_iadp': 'False',
        'fedgpro_phase_transition_threshold': '0.70',
        'fedgpro_phase2_rounds': '50'
    },
    'Privacy_Epsilon_10.0': {
        'fedgpro_use_vae': 'True',
        'fedgpro_use_prototype': 'True',
        'fedgpro_phase2_agg': 'fedavg',
        'fedgpro_lambda_proto': '0.3',
        'fedgpro_epsilon': '10.0',
        'fedgpro_noise_type': 'laplace',
        'fedgpro_use_iadp': 'False',
        'fedgpro_phase_transition_threshold': '0.70',
        'fedgpro_phase2_rounds': '50'
    },
    
    # 自适应加密策略消融
    'Privacy_Utility_First': {
        'fedgpro_use_vae': 'True',
        'fedgpro_use_prototype': 'True',
        'fedgpro_phase2_agg': 'fedavg',
        'fedgpro_lambda_proto': '0.3',
        'fedgpro_epsilon': '10.0',
        'fedgpro_noise_type': 'laplace',
        'fedgpro_use_iadp': 'True',
        'fedgpro_iadp_alpha': '0.3',
        'fedgpro_iadp_importance_method': 'vae_contrast',
        'fedgpro_iadp_privacy_priority': 'False',
        'fedgpro_phase_transition_threshold': '0.70',
    },
    'Privacy_Privacy_First': {
        'fedgpro_use_vae': 'True',
        'fedgpro_use_prototype': 'True',
        'fedgpro_phase2_agg': 'fedavg',
        'fedgpro_lambda_proto': '0.3',
        'fedgpro_epsilon': '10.0',
        'fedgpro_noise_type': 'laplace',
        'fedgpro_use_iadp': 'True',
        'fedgpro_iadp_alpha': '0.3',
        'fedgpro_iadp_importance_method': 'vae_contrast',
        'fedgpro_iadp_privacy_priority': 'True',
        'fedgpro_phase_transition_threshold': '0.70',
        'fedgpro_phase2_rounds': '50'
    },
    
    # 泛化能力消融
    'Generalization_Reserve_2': {
        'fedgpro_use_vae': 'True',
        'fedgpro_use_prototype': 'True',
        'fedgpro_phase2_agg': 'fedavg',
        'fedgpro_lambda_proto': '0.3',
        'fedgpro_phase_transition_threshold': '0.70',
        'fedgpro_phase2_rounds': '50',
        'reserved_clients': '8,9'
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

for gpu_id in GPU_IDS:
    for slot_id in range(SLOTS_PER_GPU):
        gpu_status[(gpu_id, slot_id)] = None

# =============================================================================
# 工具函数
# =============================================================================

def _ts():
    return datetime.now().strftime('%H:%M:%S')


def check_missing_experiments():
    """检查缺失实验"""
    print("\n" + "="*80)
    print("检查消融实验进度")
    print("="*80)
    
    config_stats = {}
    total_needed = 0
    total_completed = 0
    
    for dataset in DATASETS:
        for hetero in HETEROGENEITY_TYPES.keys():
            for config_name in ABLATION_CONFIGS.keys():
                total_needed += 5
                
                if config_name not in config_stats:
                    config_stats[config_name] = {'total': 0, 'completed': 0, 'details': []}
                
                config_stats[config_name]['total'] += 5
                
                results_dir = BASE_DIR / 'system' / 'results' / f"{dataset}_FedGpro_Ablation_{config_name}_{hetero}"
                completed = len(list(results_dir.glob('*.h5'))) if results_dir.exists() else 0
                
                config_stats[config_name]['completed'] += completed
                total_completed += completed
                config_stats[config_name]['details'].append((dataset, hetero, completed))
    
    print(f"\n总进度: {total_completed}/{total_needed} ({total_completed/total_needed*100:.1f}%)")
    print(f"\n{'配置':<30} {'进度':<12} {'状态'}")
    print("-" * 60)
    
    for config in sorted(config_stats.keys()):
        stats = config_stats[config]
        status = "[DONE]" if stats['completed'] == stats['total'] else "[...]"
        print(f"{config:<30} {stats['completed']:3d}/{stats['total']:3d}    {status}")
    
    missing = [(d, h, c, comp) for c in config_stats 
               for d, h, comp in config_stats[c]['details'] if comp < 5]
    
    if not missing:
        print("\n[INFO] 所有消融实验已完成！")
    
    print("="*80 + "\n")
    return missing


def build_command(dataset, hetero_type, config_name, gpu_id):
    """构建运行命令"""
    params = HYPERPARAMETERS[dataset][hetero_type]
    config = ABLATION_CONFIGS[config_name]
    
    goal_name = f'Ablation_{config_name}_{hetero_type}'
    save_folder = f'system/models/{dataset}_FedGpro_Ablation_{config_name}_{hetero_type}'
    
    cmd = [
        'python', '-u', 'system/main.py',
        '-data', dataset, '-m', 'credit', '-algo', 'FedGpro',
        '-did', str(gpu_id), '-gr', str(GLOBAL_ROUNDS),
        '-nc', str(params['num_clients']),
        '-ls', str(LOCAL_EPOCHS),
        '-lr', str(params['learning_rate']),
        '-lbs', str(params['batch_size']),
        '-t', '5', '-go', goal_name, '-sfn', save_folder,
    ]
    
    for key, value in config.items():
        cmd.extend([f'--{key}', str(value)])
    
    return cmd


def worker_thread(gpu_id, slot_id, task_queue, results_list, results_lock):
    """工作线程"""
    global completed_count, failed_count
    
    while True:
        try:
            task = task_queue.get(timeout=1)
            if task is None:
                break
            
            dataset, hetero, config_name, exp_id = task
            
            with progress_lock:
                gpu_status[(gpu_id, slot_id)] = {
                    'dataset': dataset, 'hetero': hetero, 'config': config_name,
                    'start_time': time.time(), 'last_round': 0
                }
            
            print(f"[{_ts()}] [START] GPU{gpu_id}-{slot_id}: {dataset}-{hetero}-{config_name}")
            
            cmd = build_command(dataset, hetero, config_name, gpu_id)
            start_time = time.time()
            success = False
            
            try:
                logs_dir = BASE_DIR / 'logs'
                logs_dir.mkdir(exist_ok=True)
                log_file = logs_dir / f"Ablation_{config_name}_{dataset}_{hetero}.log"
                
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
                                elapsed = (time.time() - start_time) / 60
                                print(f"  [{_ts()}] {dataset}-{config_name} | Round {round_num}/{GLOBAL_ROUNDS} | {elapsed:.1f}min")
                
                returncode = process.wait(timeout=7200)
                elapsed = time.time() - start_time
                
                if returncode == 0:
                    success = True
                    with progress_lock:
                        completed_count += 1
                    print(f"[{_ts()}] [DONE] {dataset}-{hetero}-{config_name} ({elapsed/60:.1f}min)")
                else:
                    with progress_lock:
                        failed_count += 1
                    print(f"[{_ts()}] [FAIL] {dataset}-{hetero}-{config_name}")
            
            except subprocess.TimeoutExpired:
                process.kill()
                with progress_lock:
                    failed_count += 1
                print(f"[{_ts()}] [TIMEOUT] {dataset}-{hetero}-{config_name}")
            except Exception as e:
                with progress_lock:
                    failed_count += 1
                print(f"[{_ts()}] [ERROR] {dataset}-{hetero}-{config_name}: {e}")
            
            with progress_lock:
                gpu_status[(gpu_id, slot_id)] = None
            
            with results_lock:
                results_list.append({'dataset': dataset, 'config': config_name, 'success': success})
            
            task_queue.task_done()
        
        except Empty:
            continue


def run_experiments():
    """运行消融实验"""
    global total_experiments
    
    print("\n" + "="*80)
    print("联邦学习消融实验")
    print("="*80)
    print(f"并发数: {len(GPU_IDS)} GPU x {SLOTS_PER_GPU}")
    print(f"配置数: {len(ABLATION_CONFIGS)}")
    print("="*80)
    
    missing = check_missing_experiments()
    if not missing:
        print("所有实验已完成！")
        return
    
    task_queue = Queue()
    for i, (dataset, hetero, config, completed) in enumerate(missing):
        task_queue.put((dataset, hetero, config, i))
    
    total_experiments = len(missing)
    print(f"\n待运行: {total_experiments}\n")
    
    threads = []
    results_list = []
    results_lock = threading.Lock()
    
    for gpu_id in GPU_IDS:
        for slot_id in range(SLOTS_PER_GPU):
            t = threading.Thread(target=worker_thread, 
                               args=(gpu_id, slot_id, task_queue, results_list, results_lock), 
                               daemon=True)
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
    
    analyze_ablation_results()


def analyze_ablation_results():
    """生成分析报告"""
    try:
        from system.utils.analyze_results import main as analyze_main
        analyze_main()
        print("\n[DONE] 分析完成! Excel: system/results/汇总/")
    except Exception as e:
        print(f"\n[ERROR] 分析失败: {e}")


def main():
    """主入口"""
    import argparse
    parser = argparse.ArgumentParser(description='联邦学习消融实验')
    parser.add_argument('--check', action='store_true', help='检查缺失实验')
    parser.add_argument('--analyze', action='store_true', help='仅生成报告')
    args = parser.parse_args()
    
    if args.check:
        check_missing_experiments()
    elif args.analyze:
        analyze_ablation_results()
    else:
        run_experiments()


if __name__ == '__main__':
    main()
