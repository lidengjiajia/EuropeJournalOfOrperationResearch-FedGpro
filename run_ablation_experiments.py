"""
联邦学习消融实验批量运行脚本 (优化版)
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
from typing import Dict, List
import re

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR / 'system'))

# =============================================================================
# 配置常量
# =============================================================================

DATASETS = ['Uci', 'Xinwang']
HETEROGENEITY_TYPES = {'feature': '特征异质性', 'label': '标签异质性',
                       'quantity': '样本数量异质性', 'iid': 'IID均匀分布'}

GLOBAL_ROUNDS = 100
LOCAL_EPOCHS = 5

# GPU自动检测
try:
    import torch
    gpu_count = torch.cuda.device_count()
    if gpu_count > 0:
        GPU_IDS = list(range(gpu_count))
        print(f"检测到 {gpu_count} 个GPU")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        GPU_IDS = [0]
        print("未检测到GPU，使用CPU")
except:
    GPU_IDS = [0]
    print("无法检测GPU，使用默认配置")

SLOTS_PER_GPU = 2

# 超参数配置
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
# 根据论文图5-9设计消融实验：
# 图5: 生成数据作用、原型作用、元启发式算法作用
# 图6: 隐私预算（不同ε值对比 + Hybrid策略）
# 图7: 泛化能力（新客户端测试 - reserved_clients）
# 图8: 客户端数目影响（5, 6, 7, 8, 9, 10个客户端）
# 图9: 超参数敏感性分析（lambda_proto, lambda_kl, latent_dim, proto_momentum, phase_transition_threshold, gwo_alpha_decay）

ABLATION_CONFIGS = {
    # ========== 图5: 组件消融（生成数据、原型、不同Phase2聚合算法） ==========
    # 完整模型（baseline，Phase2使用FedAvg）
    'Full_Model': {
        'fedgpro_use_vae': 'True',
        'fedgpro_use_prototype': 'True',
        'fedgpro_phase2_agg': 'fedavg',      # Phase2使用FedAvg
        'fedgpro_lambda_proto': '0.3',       # 优化后的原型损失权重
        'fedgpro_phase_transition_threshold': '0.70',
        'fedgpro_phase2_rounds': '50'
    },
    
    # 无VAE生成数据（图5a: 测试生成数据作用）
    'No_VAE_Generation': {
        'fedgpro_use_vae': 'False',          # 禁用VAE生成虚拟数据
        'fedgpro_use_prototype': 'True',
        'fedgpro_phase2_agg': 'fedavg',
        'fedgpro_lambda_proto': '0.3',
        'fedgpro_phase_transition_threshold': '0.70',
        'fedgpro_phase2_rounds': '50'
    },
    
    # 无原型学习（图5b: 测试原型作用）
    'No_Prototype': {
        'fedgpro_use_vae': 'True',
        'fedgpro_use_prototype': 'False',    # 禁用原型学习损失
        'fedgpro_phase2_agg': 'fedavg',
        'fedgpro_phase_transition_threshold': '0.70',
        'fedgpro_phase2_rounds': '50'
    },
    
    # Phase2使用FedProx（图5c: 测试不同聚合算法）
    'Phase2_FedProx': {
        'fedgpro_use_vae': 'True',
        'fedgpro_use_prototype': 'True',
        'fedgpro_phase2_agg': 'fedprox',
        'mu': '0.1',                         # FedProx需要mu参数
        'fedgpro_lambda_proto': '0.3',
        'fedgpro_phase_transition_threshold': '0.70',
        'fedgpro_phase2_rounds': '50'
    },
    
    # Phase2使用Scaffold（图5d: 测试Scaffold聚合算法）
    'Phase2_Scaffold': {
        'fedgpro_use_vae': 'True',
        'fedgpro_use_prototype': 'True',
        'fedgpro_phase2_agg': 'scaffold',
        'fedgpro_lambda_proto': '0.3',
        'fedgpro_phase_transition_threshold': '0.70',
        'fedgpro_phase2_rounds': '50'
    },
    
    # ========== 图6a: 隐私预算消融（3个ε值，均使用常规加密） ==========
    
    # ε=1.0（严格隐私保护）
    'Privacy_Epsilon_1.0': {
        'fedgpro_use_vae': 'True',
        'fedgpro_use_prototype': 'True',
        'fedgpro_phase2_agg': 'fedavg',
        'fedgpro_lambda_proto': '0.3',
        'fedgpro_epsilon': '1.0',            # 隐私预算=1.0
        'fedgpro_noise_type': 'laplace',
        'fedgpro_use_iadp': 'False',         # 常规加密
        'fedgpro_phase_transition_threshold': '0.70',
        'fedgpro_phase2_rounds': '50'
    },
    
    # ε=5.0（中等隐私保护）
    'Privacy_Epsilon_5.0': {
        'fedgpro_use_vae': 'True',
        'fedgpro_use_prototype': 'True',
        'fedgpro_phase2_agg': 'fedavg',
        'fedgpro_lambda_proto': '0.3',
        'fedgpro_epsilon': '5.0',            # 隐私预算=5.0
        'fedgpro_noise_type': 'laplace',
        'fedgpro_use_iadp': 'False',         # 常规加密
        'fedgpro_phase_transition_threshold': '0.70',
        'fedgpro_phase2_rounds': '50'
    },
    
    # ε=10.0（宽松隐私保护）
    'Privacy_Epsilon_10.0': {
        'fedgpro_use_vae': 'True',
        'fedgpro_use_prototype': 'True',
        'fedgpro_phase2_agg': 'fedavg',
        'fedgpro_lambda_proto': '0.3',
        'fedgpro_epsilon': '10.0',           # 隐私预算=10.0
        'fedgpro_noise_type': 'laplace',
        'fedgpro_use_iadp': 'False',         # 常规加密
        'fedgpro_phase_transition_threshold': '0.70',
        'fedgpro_phase2_rounds': '50'
    },
    
    # ========== 图6b: 基于特征重要性的自适应加密（ε=10，3种策略） ==========
    
    # 策略1: 常规加密（传统DP，均匀噪声，作为baseline）
    'Privacy_Conventional': {
        'fedgpro_use_vae': 'True',
        'fedgpro_use_prototype': 'True',
        'fedgpro_phase2_agg': 'fedavg',
        'fedgpro_lambda_proto': '0.3',
        'fedgpro_epsilon': '10.0',           # 隐私预算=10
        'fedgpro_noise_type': 'laplace',
        'fedgpro_use_iadp': 'False',         # 不使用自适应加密
        'fedgpro_phase_transition_threshold': '0.70',
        'fedgpro_phase2_rounds': '50'
    },
    
    # 策略2: 效用优先（基于特征重要性排序，重要特征低噪声）
    # 使用VAE对比损失度量特征重要性：对分类有用的特征具有高判别性
    'Privacy_Utility_First': {
        'fedgpro_use_vae': 'True',
        'fedgpro_use_prototype': 'True',
        'fedgpro_phase2_agg': 'fedavg',
        'fedgpro_lambda_proto': '0.3',
        'fedgpro_epsilon': '10.0',           # 隐私预算=10
        'fedgpro_noise_type': 'laplace',
        'fedgpro_use_iadp': 'True',          # 使用自适应DP
        'fedgpro_iadp_alpha': '0.3',
        'fedgpro_iadp_importance_method': 'vae_contrast',  # 使用VAE对比损失
        'fedgpro_iadp_privacy_priority': 'False',  # 效用优先
        'fedgpro_phase_transition_threshold': '0.70',
        'fedgpro_phase2_rounds': '50'
    },
    
    # 策略3: 隐私优先（基于特征重要性排序，重要特征高噪声）
    # 使用VAE对比损失度量特征重要性：高判别性特征添加更多噪声保护隐私
    'Privacy_Privacy_First': {
        'fedgpro_use_vae': 'True',
        'fedgpro_use_prototype': 'True',
        'fedgpro_phase2_agg': 'fedavg',
        'fedgpro_lambda_proto': '0.3',
        'fedgpro_epsilon': '10.0',           # 隐私预算=10
        'fedgpro_noise_type': 'laplace',
        'fedgpro_use_iadp': 'True',          # 使用自适应DP
        'fedgpro_iadp_alpha': '0.3',
        'fedgpro_iadp_importance_method': 'vae_contrast',  # 使用VAE对比损失
        'fedgpro_iadp_privacy_priority': 'True',  # 隐私优先
        'fedgpro_phase_transition_threshold': '0.70',
        'fedgpro_phase2_rounds': '50'
    },
    
        # ========== 图7: 泛化能力（新客户端测试） ==========
    # 保留20%客户端测试泛化能力
    'Generalization_Reserve_2': {
        'fedgpro_use_vae': 'True',
        'fedgpro_use_prototype': 'True',
        'fedgpro_phase2_agg': 'fedavg',
        'fedgpro_lambda_proto': '0.3',
        'fedgpro_phase_transition_threshold': '0.70',
        'fedgpro_phase2_rounds': '50',
        'reserved_clients': '8,9'            # 保留客户端8,9用于泛化测试（20%）
    },
    
    # 保留30%客户端测试泛化能力
    'Generalization_Reserve_3': {
        'fedgpro_use_vae': 'True',
        'fedgpro_use_prototype': 'True',
        'fedgpro_phase2_agg': 'fedavg',
        'fedgpro_lambda_proto': '0.3',
        'fedgpro_phase_transition_threshold': '0.70',
        'fedgpro_phase2_rounds': '50',
        'reserved_clients': '7,8,9'          # 保留客户端7,8,9用于泛化测试（30%）
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

def get_algorithm_name_for_config(config_name):
    """根据消融配置获取实际的算法名称
    
    servergpro.py会在save_results时使用self.original_algorithm，
    这个值在__init__时就从args.algorithm复制过来
    所以，无论fedgpro_phase2_agg是什么，保存的目录名都基于传入的算法名
    
    因此，如果build_command传的是'FedGpro'，结果目录就会是 FedGpro
    如果传的是'FedGpro-FedAvg'，结果目录就会是 FedGpro-FedAvg
    """
    # 返回build_command中传递的算法名
    return 'FedGpro'

def check_missing_experiments():
    """检查缺失的实验文件"""
    print("\n" + "="*80)
    print("检查缺失的消融实验文件...")
    print("="*80)
    
    missing = []
    for dataset in DATASETS:
        for hetero in HETEROGENEITY_TYPES.keys():
            for config_name in ABLATION_CONFIGS.keys():
                # 新的目录结构: 每个实验配置一个独立目录
                # 目录名格式: {dataset}_FedGpro_Ablation_{config_name}_{hetero}
                # 注意: servergpro.py在save_results时使用self.original_algorithm（初始值），
                # 无论fedgpro_phase2_agg是什么，所以目录名总是基于传入的算法名'FedGpro'
                algo_name = get_algorithm_name_for_config(config_name)
                results_dir = BASE_DIR / 'system' / 'results' / f"{dataset}_{algo_name}_Ablation_{config_name}_{hetero}"
                # 文件名格式: {dataset}_FedGpro_Ablation_{config_name}_{hetero}_*.h5
                file_prefix = f"{dataset}_{algo_name}_Ablation_{config_name}_{hetero}"
                
                if not results_dir.exists():
                    missing.append((dataset, hetero, config_name, 0))
                    continue
                
                # 查找实际的文件模式
                completed_files = list(results_dir.glob(f"{file_prefix}_*.h5"))
                completed = len(completed_files)
                
                if completed < 5:
                    missing.append((dataset, hetero, config_name, completed))
    
    if missing:
        print(f"\n缺失消融实验数: {len(missing)}")
        print(f"{'数据集':<10} {'异质性':<10} {'配置':<20} {'已完成/需要'}")
        print("-" * 80)
        for dataset, hetero, config, completed in missing:
            print(f"{dataset:<10} {hetero:<10} {config:<20} {completed}/5")
    else:
        print("\n✅ 所有消融实验均已完成！")
    
    print("="*80 + "\n")
    return missing

def build_command(dataset, hetero_type, config_name, gpu_id):
    """构建运行命令"""
    params = HYPERPARAMETERS[dataset][hetero_type]
    config = ABLATION_CONFIGS[config_name]
    
    # 新的目录结构: 简洁格式
    # 🔥 关键修复: 传入 'FedGpro' 而不是 'FedGpro-FedAvg'
    # servergpro.py会在save_results时使用self.original_algorithm（初始值），
    # 所以如果传'FedGpro'，保存的目录就是 FedGpro_xxx，不会包含FedAvg/FedProx等
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
        '-t', '5',
        '-go', goal_name,
        '-sfn', save_folder,
    ]
    
    for key, value in config.items():
        cmd.extend([f'--{key}', str(value)])
    
    return cmd

# =============================================================================
# Worker线程
# =============================================================================

def worker_thread(gpu_id, slot_id, task_queue, results_list, results_lock):
    """Worker线程"""
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
                    'exp_id': exp_id, 'start_time': time.time(), 'last_round': 0
                }
            
            print(f"[{_ts()}] 🚀 GPU{gpu_id}-槽位{slot_id}: {dataset}-{hetero}-{config_name}")
            
            cmd = build_command(dataset, hetero, config_name, gpu_id)
            start_time = time.time()
            success = False
            
            try:
                # 创建logs目录
                logs_dir = BASE_DIR / 'logs'
                logs_dir.mkdir(exist_ok=True)
                log_file_path = logs_dir / f"Ablation_{config_name}_{dataset}_{hetero}.log"
                
                # stderr=subprocess.STDOUT 将stderr合并到stdout
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding='utf-8', errors='replace',
                    bufsize=1, universal_newlines=True
                )
                
                # 打开日志文件并实时写入
                with open(log_file_path, 'w', encoding='utf-8') as log_file:
                    last_printed_round = -1  # 记录上次打印的round，避免重复
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
                                print(f"  [{_ts()}] {dataset}-{hetero}-{config_name} | Round {round_num}/{GLOBAL_ROUNDS} | {elapsed_min:.1f}分钟")
                
                returncode = process.wait(timeout=7200)
                elapsed = time.time() - start_time
                
                if returncode == 0:
                    success = True
                    with progress_lock:
                        completed_count += 1
                    print(f"[{_ts()}] ✅ {dataset}-{hetero}-{config_name} 完成 ({elapsed/60:.1f}分钟)")
                else:
                    with progress_lock:
                        failed_count += 1
                    print(f"[{_ts()}] ❌ {dataset}-{hetero}-{config_name} 失败")
            
            except subprocess.TimeoutExpired:
                process.kill()
                elapsed = time.time() - start_time
                with progress_lock:
                    failed_count += 1
                print(f"[{_ts()}] ⏱️ {dataset}-{hetero}-{config_name} 超时 ({elapsed/60:.1f}分钟)")
            except Exception as e:
                elapsed = time.time() - start_time
                with progress_lock:
                    failed_count += 1
                print(f"[{_ts()}] 💥 {dataset}-{hetero}-{config_name} 异常: {str(e)}")
                import traceback
                print(f"  详细错误: {traceback.format_exc()}")
            
            with progress_lock:
                gpu_status[(gpu_id, slot_id)] = None
            
            with results_lock:
                results_list.append({
                    'dataset': dataset, 'hetero': hetero, 'config': config_name,
                    'success': success, 'elapsed': elapsed
                })
            
            task_queue.task_done()
        
        except Empty:
            continue

# =============================================================================
# 主函数
# =============================================================================

def run_experiments():
    """运行所有消融实验"""
    global total_experiments
    
    print("\n" + "="*80)
    print("联邦学习消融实验批量运行 (优化版)")
    print("="*80)
    print(f"GPU配置: {len(GPU_IDS)}个GPU × {SLOTS_PER_GPU}槽位 = {len(GPU_IDS) * SLOTS_PER_GPU}并发")
    print(f"数据集: {', '.join(DATASETS)}")
    print(f"消融配置: {', '.join(ABLATION_CONFIGS.keys())}")
    print("="*80)
    
    missing = check_missing_experiments()
    
    if not missing:
        print("所有消融实验已完成，无需运行！")
        return
    
    task_queue = Queue()
    exp_id = 0
    for dataset, hetero, config_name, completed in missing:
        task_queue.put((dataset, hetero, config_name, exp_id))
        exp_id += 1
    
    total_experiments = exp_id
    print(f"\n需要运行的消融实验数: {total_experiments}\n")
    
    threads = []
    results_list = []
    results_lock = threading.Lock()
    
    for gpu_id in GPU_IDS:
        for slot_id in range(SLOTS_PER_GPU):
            t = threading.Thread(
                target=worker_thread,
                args=(gpu_id, slot_id, task_queue, results_list, results_lock),
                daemon=True
            )
            t.start()
            threads.append(t)
    
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
                    print(f"  GPU{gpu_id}-槽位{slot_id}: {info['dataset']}-{info['hetero']}-{info['config']} | Round {r}/{GLOBAL_ROUNDS} | {elapsed_min:.1f}分钟")
    
    for gpu_id in GPU_IDS:
        for slot_id in range(SLOTS_PER_GPU):
            task_queue.put(None)
    
    for t in threads:
        t.join()
    
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("消融实验完成！")
    print(f"总耗时: {total_time/3600:.2f}小时")
    print(f"完成: {completed_count}/{total_experiments}")
    print(f"失败: {failed_count}/{total_experiments}")
    print(f"成功率: {completed_count/total_experiments*100:.1f}%")
    print("="*80 + "\n")

if __name__ == '__main__':
    run_experiments()
