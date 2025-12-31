"""
联邦学习基准算法批量实验运行脚本 (优化版)
特性:
- 并发执行: 每个GPU同时运行2个实验
- 智能检测: 打印缺失的实验文件
- 简化日志: 每20轮打印一次进度
- 实时监控: 显示每个实验的进度
"""

import os
import sys
import time
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from typing import Dict, List, Optional, Any
from collections import defaultdict
import re

# 添加system目录到路径
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR / 'system'))

# =============================================================================
# 配置常量
# =============================================================================

DATASETS = ['Uci', 'Xinwang']
HETEROGENEITY_TYPES = {'feature': '特征异质性', 'label': '标签异质性', 
                       'quantity': '样本数量异质性', 'iid': 'IID均匀分布'}
ALGORITHMS = ['FedAvg', 'FedProx', 'FedScaffold', 'FedMoon', 'FedGen',
              'Per-FedAvg', 'FedDitto', 'FedRep', 'FedProto', 'FedPso', 'FedGwo',
              'FedGpro']

GLOBAL_ROUNDS = 100
LOCAL_EPOCHS = 5

# 自动检测GPU并配置
try:
    import torch
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        GPU_IDS = list(range(num_gpus))  # 自动使用所有GPU
        print(f"✅ 检测到 {num_gpus} 个GPU: {GPU_IDS}")
        for i in range(num_gpus):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        GPU_IDS = [0]
        print("⚠️ 未检测到CUDA，使用默认配置 GPU 0")
except:
    GPU_IDS = [0]
    print("⚠️ 无法检测GPU，使用默认配置 GPU 0")

SLOTS_PER_GPU = 2

# 超参数配置
HYPERPARAMETERS = {
    'Uci': {
        'feature': {
            'FedAvg': {'lr': 0.005, 'batch_size': 64, 'local_epochs': 5},
            'FedProx': {'lr': 0.005, 'batch_size': 64, 'local_epochs': 5, 'mu': 0.2},
            'FedScaffold': {'lr': 0.005, 'batch_size': 64, 'local_epochs': 5},
            'FedMoon': {'lr': 0.005, 'batch_size': 64, 'local_epochs': 5, 'mu': 1.0},
            'FedGen': {'lr': 0.005, 'batch_size': 64, 'local_epochs': 5},
            'Per-FedAvg': {'lr': 0.005, 'batch_size': 64, 'local_epochs': 5, 'beta': 0.003},
            'FedDitto': {'lr': 0.005, 'batch_size': 64, 'local_epochs': 5, 'mu': 0.2, 'plocal_epochs': 3},
            'FedRep': {'lr': 0.005, 'batch_size': 64, 'local_epochs': 5},
            'FedProto': {'lr': 0.005, 'batch_size': 64, 'local_epochs': 5, 'lamda': 15},
            'FedPso': {'lr': 0.005, 'batch_size': 64, 'local_epochs': 5},
            'FedGwo': {'lr': 0.005, 'batch_size': 64, 'local_epochs': 5},
            'FedGpro': {'lr': 0.005, 'batch_size': 64, 'local_epochs': 5, 'mu': 0.2, 'plocal_epochs': 3, 'fedgpro_phase2_agg': 'ditto', 'fedgpro_phase2_rounds': 50, 'fedgpro_phase_transition_threshold': 0.70},
        },
        'label': {
            'FedAvg': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5},
            'FedProx': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'mu': 0.1},
            'FedScaffold': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5},
            'FedMoon': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'mu': 1.2},
            'FedGen': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5},
            'Per-FedAvg': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'beta': 0.005},
            'FedDitto': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'mu': 0.1, 'plocal_epochs': 4},
            'FedRep': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5},
            'FedProto': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'lamda': 20},
            'FedPso': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5},
            'FedGwo': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5},
            'FedGpro': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'mu': 0.1, 'plocal_epochs': 4, 'fedgpro_phase2_agg': 'ditto', 'fedgpro_phase2_rounds': 50, 'fedgpro_phase_transition_threshold': 0.70},
        },
        'quantity': {
            'FedAvg': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5},
            'FedProx': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'mu': 0.05},
            'FedScaffold': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5},
            'FedMoon': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'mu': 1.0},
            'FedGen': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5},
            'Per-FedAvg': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'beta': 0.004},
            'FedDitto': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'mu': 0.05, 'plocal_epochs': 3},
            'FedRep': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5},
            'FedProto': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'lamda': 18},
            'FedPso': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5},
            'FedGwo': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5},
            'FedGpro': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'mu': 0.05, 'plocal_epochs': 3, 'fedgpro_phase2_agg': 'ditto', 'fedgpro_phase2_rounds': 50, 'fedgpro_phase_transition_threshold': 0.70},
        },
        'iid': {
            'FedAvg': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5},
            'FedProx': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'mu': 0.01},
            'FedScaffold': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5},
            'FedMoon': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'mu': 1.0},
            'FedGen': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5},
            'Per-FedAvg': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'beta': 0.003},
            'FedDitto': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'mu': 0.01, 'plocal_epochs': 2},
            'FedRep': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5},
            'FedProto': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'lamda': 15},
            'FedPso': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5},
            'FedGwo': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5},
            'FedGpro': {'lr': 0.007, 'batch_size': 64, 'local_epochs': 5, 'mu': 0.01, 'plocal_epochs': 2, 'fedgpro_phase2_agg': 'ditto', 'fedgpro_phase2_rounds': 50, 'fedgpro_phase_transition_threshold': 0.70},
        },
    },
    'Xinwang': {
        'feature': {
            'FedAvg': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedProx': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'mu': 0.12},
            'FedScaffold': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedMoon': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'mu': 1.5},
            'FedGen': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'Per-FedAvg': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'beta': 0.002},
            'FedDitto': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'mu': 0.12, 'plocal_epochs': 3},
            'FedRep': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedProto': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'lamda': 18},
            'FedPso': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedGwo': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedGpro': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'mu': 0.12, 'plocal_epochs': 3, 'fedgpro_phase2_agg': 'ditto', 'fedgpro_phase2_rounds': 50, 'fedgpro_phase_transition_threshold': 0.70},
        },
        'label': {
            'FedAvg': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedProx': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'mu': 0.08},
            'FedScaffold': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedMoon': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'mu': 1.3},
            'FedGen': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'Per-FedAvg': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'beta': 0.003},
            'FedDitto': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'mu': 0.08, 'plocal_epochs': 4},
            'FedRep': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedProto': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'lamda': 25},
            'FedPso': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedGwo': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedGpro': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'mu': 0.08, 'plocal_epochs': 4, 'fedgpro_phase2_agg': 'ditto', 'fedgpro_phase2_rounds': 50, 'fedgpro_phase_transition_threshold': 0.70},
        },
        'quantity': {
            'FedAvg': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedProx': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'mu': 0.08},
            'FedScaffold': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedMoon': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'mu': 1.0},
            'FedGen': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'Per-FedAvg': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'beta': 0.003},
            'FedDitto': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'mu': 0.08, 'plocal_epochs': 3},
            'FedRep': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedProto': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'lamda': 20},
            'FedPso': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedGwo': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedGpro': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'mu': 0.08, 'plocal_epochs': 3, 'fedgpro_phase2_agg': 'ditto', 'fedgpro_phase2_rounds': 50, 'fedgpro_phase_transition_threshold': 0.70},
        },
        'iid': {
            'FedAvg': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedProx': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'mu': 0.01},
            'FedScaffold': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedMoon': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'mu': 1.0},
            'FedGen': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'Per-FedAvg': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'beta': 0.002},
            'FedDitto': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'mu': 0.01, 'plocal_epochs': 2},
            'FedRep': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedProto': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'lamda': 15},
            'FedPso': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedGwo': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5},
            'FedGpro': {'lr': 0.006, 'batch_size': 128, 'local_epochs': 5, 'mu': 0.01, 'plocal_epochs': 2, 'fedgpro_phase2_agg': 'ditto', 'fedgpro_phase2_rounds': 50, 'fedgpro_phase_transition_threshold': 0.70},
        },
    },
}

progress_lock = threading.Lock()
gpu_status = {}  # {(gpu_id, slot_id): task_info}
completed_count = 0
failed_count = 0
total_experiments = 0
task_queue = None  # 将在运行时初始化
results_list = []
results_lock = threading.Lock()

for gpu_id in GPU_IDS:
    for slot_id in range(SLOTS_PER_GPU):
        gpu_status[(gpu_id, slot_id)] = None

# =============================================================================
# 工具函数
# =============================================================================

def _ts():
    """时间戳"""
    return datetime.now().strftime('%H:%M:%S')

def check_missing_experiments():
    """检查缺失的实验文件"""
    print("\n" + "="*80)
    print("检查缺失的实验文件...")
    print("="*80)
    
    missing = []
    for dataset in DATASETS:
        for hetero in HETEROGENEITY_TYPES.keys():
            for algo in ALGORITHMS:
                # 基线实验目录结构: 每个算法配置一个独立目录
                # 目录名格式: {dataset}_{algo}_{hetero}
                results_dir = BASE_DIR / 'system' / 'results' / f"{dataset}_{algo}_{hetero}"
                # 文件名格式: {dataset}_{algo}_{hetero}_*.h5 (不含test前缀)
                file_prefix = f"{dataset}_{algo}_{hetero}"
                
                if not results_dir.exists():
                    missing.append((dataset, hetero, algo, 0))
                    continue
                
                # 查找实际的文件模式（每个目录应该只有5个文件）
                completed_files = list(results_dir.glob(f"{file_prefix}_*.h5"))
                completed = len(completed_files)
                
                if completed < 5:
                    missing.append((dataset, hetero, algo, completed))
    
    if missing:
        print(f"\n缺失实验数: {len(missing)}")
        print(f"{'数据集':<10} {'异质性':<10} {'算法':<20} {'已完成/需要'}")
        print("-" * 80)
        for dataset, hetero, algo, completed in missing:
            print(f"{dataset:<10} {hetero:<10} {algo:<20} {completed}/5")
    else:
        print("\n✅ 所有实验均已完成！")
    
    print("="*80 + "\n")
    return missing

def build_command(dataset, algorithm, hetero_type, gpu_id):
    """构建运行命令"""
    params = HYPERPARAMETERS[dataset][hetero_type][algorithm]
    model_name = 'credit'
    # 目标名称（简洁格式）: {hetero_type}
    goal_name = hetero_type
    # 模型保存目录：每个实验配置一个独立目录
    save_folder = f'system/models/{dataset}_{algorithm}_{hetero_type}'
    
    cmd = [
        'python', '-u', 'system/main.py',
        '-data', dataset, '-m', model_name, '-algo', algorithm,
        '-did', str(gpu_id), '-gr', str(GLOBAL_ROUNDS),
        '-nc', '10', '-ls', str(params.get('local_epochs', LOCAL_EPOCHS)),
        '-lr', str(params['lr']), '-lbs', str(params['batch_size']),
        '-t', '5', '-go', goal_name,
        '-sfn', save_folder,
    ]
    
    # 添加额外参数
    if 'mu' in params:
        cmd.extend(['-mu', str(params['mu'])])
    if 'plocal_epochs' in params:
        cmd.extend(['-pls', str(params['plocal_epochs'])])
    if 'beta' in params:
        cmd.extend(['-bt', str(params['beta'])])
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
    """Worker线程 - 从队列获取任务并执行"""
    global completed_count, failed_count, task_queue, results_list, results_lock
    
    while True:
        try:
            task = task_queue.get(timeout=1)
            if task is None:
                break
            
            dataset, hetero, algo, exp_id, current_time = task
            
            # 更新GPU状态
            with progress_lock:
                gpu_status[(gpu_id, slot_id)] = {
                    'dataset': dataset, 'hetero': hetero, 'algo': algo,
                    'exp_id': exp_id, 'start_time': time.time(),
                    'last_round': 0, 'current_time': current_time
                }
            
            print(f"[{_ts()}] 🚀 GPU{gpu_id}-槽位{slot_id}: {dataset}-{hetero}-{algo} [第{current_time+1}次/共5次]")
            
            cmd = build_command(dataset, algo, hetero, gpu_id)
            start_time = time.time()
            success = False
            
            try:
                # 创建logs目录
                logs_dir = BASE_DIR / 'logs'
                logs_dir.mkdir(exist_ok=True)
                log_file_path = logs_dir / f"{dataset}_{algo}_{hetero}.log"
                
                # stderr=subprocess.STDOUT 将stderr合并到stdout
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding='utf-8', errors='replace',
                    bufsize=1, universal_newlines=True
                )
                
                # 打开日志文件并实时写入
                with open(log_file_path, 'w', encoding='utf-8') as log_file:
                    last_printed_round = -1  # 记录上次打印的round，避免重复
                    # 每20轮打印一次
                    for line in process.stdout:
                        # 写入日志文件
                        log_file.write(line)
                        log_file.flush()
                        
                        line = line.strip()
                        if not line:
                            continue
                        
                        # 增强正则：匹配 "Round 5", "Round number: 5", "Round 5 |", "----- Round 5 -----"
                        match = re.search(r'Round[:\s]+(?:number[:\s]+)?(\d+)', line, re.IGNORECASE)
                        if match:
                            round_num = int(match.group(1))
                            with progress_lock:
                                if (gpu_id, slot_id) in gpu_status and gpu_status[(gpu_id, slot_id)]:
                                    gpu_status[(gpu_id, slot_id)]['last_round'] = round_num
                            
                            # 排除round 0，且避免重复打印同一轮
                            if round_num > 0 and round_num % 20 == 0 and round_num != last_printed_round:
                                last_printed_round = round_num
                                elapsed_min = (time.time() - start_time) / 60
                                print(f"  [{_ts()}] {dataset}-{hetero}-{algo} [第{current_time+1}次] | Round {round_num}/{GLOBAL_ROUNDS} | {elapsed_min:.1f}分钟")
                
                returncode = process.wait(timeout=7200)
                elapsed = time.time() - start_time
                
                if returncode == 0:
                    success = True
                    with progress_lock:
                        completed_count += 1
                    print(f"[{_ts()}] ✅ {dataset}-{hetero}-{algo} [第{current_time+1}次] 完成 ({elapsed/60:.1f}分钟)")
                else:
                    with progress_lock:
                        failed_count += 1
                    print(f"[{_ts()}] ❌ {dataset}-{hetero}-{algo} [第{current_time+1}次] 失败")
            
            except subprocess.TimeoutExpired:
                process.kill()
                elapsed = time.time() - start_time
                with progress_lock:
                    failed_count += 1
                print(f"[{_ts()}] ⏱️ {dataset}-{hetero}-{algo} [第{current_time+1}次] 超时 ({elapsed/60:.1f}分钟)")
            except Exception as e:
                elapsed = time.time() - start_time
                with progress_lock:
                    failed_count += 1
                print(f"[{_ts()}] 💥 {dataset}-{hetero}-{algo} [第{current_time+1}次] 异常: {str(e)}")
                import traceback
                print(f"  详细错误: {traceback.format_exc()}")
            
            # 清除GPU状态
            with progress_lock:
                gpu_status[(gpu_id, slot_id)] = None
            
            with results_lock:
                results_list.append({
                    'dataset': dataset, 'hetero': hetero, 'algo': algo,
                    'success': success, 'elapsed': elapsed
                })
            
            task_queue.task_done()
        
        except Empty:
            continue

# =============================================================================
# 主函数
# =============================================================================

def run_experiments():
    """运行所有实验"""
    global total_experiments, task_queue, results_list, results_lock
    
    print("\n" + "="*80)
    print("联邦学习基准算法批量实验 (优化版)")
    print("="*80)
    print(f"GPU配置: {len(GPU_IDS)}个GPU × {SLOTS_PER_GPU}槽位 = {len(GPU_IDS) * SLOTS_PER_GPU}并发")
    print(f"数据集: {', '.join(DATASETS)}")
    print(f"算法数: {len(ALGORITHMS)}")
    print(f"异质性: {', '.join(HETEROGENEITY_TYPES.keys())}")
    print("="*80)
    
    # 检查缺失文件
    missing = check_missing_experiments()
    
    if not missing:
        print("所有实验已完成，无需运行！")
        return
    
    # 生成任务队列
    task_queue = Queue()
    exp_id = 0
    for dataset, hetero, algo, completed in missing:
        # completed表示已完成的次数，下一次运行就是completed次（从0开始）
        task_queue.put((dataset, hetero, algo, exp_id, completed))
        exp_id += 1
    
    total_experiments = exp_id
    print(f"\n需要运行的实验数: {total_experiments}\n")
    
    # 启动worker线程
    threads = []
    
    for gpu_id in GPU_IDS:
        for slot_id in range(SLOTS_PER_GPU):
            t = threading.Thread(
                target=worker_thread,
                args=(gpu_id, slot_id),
                daemon=True
            )
            t.start()
            threads.append(t)
    
    # 监控进度
    start_time = time.time()
    while True:
        time.sleep(10)
        
        with progress_lock:
            running = sum(1 for s in gpu_status.values() if s is not None)
            comp = completed_count
            fail = failed_count
            remain = total_experiments - comp - fail
        
        if remain == 0 and running == 0:
            break
        
        print(f"\n[{_ts()}] 进度: 完成{comp} | 失败{fail} | 运行中{running} | 剩余{remain}")
        
        with progress_lock:
            for (gpu_id, slot_id), info in gpu_status.items():
                if info:
                    elapsed_min = (time.time() - info['start_time']) / 60
                    r = info.get('last_round', 0)
                    ct = info.get('current_time', 0)
                    print(f"  GPU{gpu_id}-槽位{slot_id}: {info['dataset']}-{info['hetero']}-{info['algo']} [第{ct+1}次] | Round {r}/{GLOBAL_ROUNDS} | {elapsed_min:.1f}分钟")
    
    # 等待所有任务完成
    for gpu_id in GPU_IDS:
        for slot_id in range(SLOTS_PER_GPU):
            task_queue.put(None)
    
    for t in threads:
        t.join()
    
    # 打印总结
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("实验完成！")
    print(f"总耗时: {total_time/3600:.2f}小时")
    print(f"完成: {completed_count}/{total_experiments}")
    print(f"失败: {failed_count}/{total_experiments}")
    print(f"成功率: {completed_count/total_experiments*100:.1f}%")
    print("="*80 + "\n")

if __name__ == '__main__':
    run_experiments()
