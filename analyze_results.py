"""
Federated Learning Experiment Results Analysis

This script processes experimental results from federated learning experiments,
computing statistics across multiple runs and generating comparative visualizations.

Key Features:
- Processes results from system/results directory
- Groups by dataset (Uci/Xinwang), algorithm, and heterogeneity type
- Computes mean±std across 5 repeated runs
- Extracts metrics: accuracy, precision, recall, F1-score
- Generates Excel reports and convergence plots
"""

import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# Core baseline algorithms for comparison
VALID_BASELINE_ALGORITHMS = {
    'FedAvg',    # Classic federated averaging
    'FedProx',   # Proximal term regularization
    'FedProto',  # Prototype-based learning
    'FedGpro',   # Proposed method
}

# Ablation study configurations
ABLATION_CONFIG_MAPPING = {}

VALID_ABLATION_CONFIGS = {
    'Full_Model',
    'No_Prototype',
    'No_VAE_Generation',
    'Privacy_Epsilon1',
    'Phase2_FedProx',
    'Phase2_Scaffold',
    'Generalization_Reserve_2',
    'Generalization_Reserve_3',
}


def read_h5_metrics(h5_file_path):
    """Read final round test metrics from h5 file"""
    try:
        with h5py.File(h5_file_path, 'r') as f:
            metrics = {}
            
            if 'rs_test_acc' in f:
                test_acc = f['rs_test_acc'][:]
                if len(test_acc) > 0:
                    metrics['accuracy'] = float(test_acc[-1])
            
            for key in ['rs_test_precision', 'test_precision', 'precision']:
                if key in f:
                    precision = f[key][:]
                    if len(precision) > 0:
                        metrics['precision'] = float(precision[-1])
                    break
            
            for key in ['rs_test_recall', 'test_recall', 'recall']:
                if key in f:
                    recall = f[key][:]
                    if len(recall) > 0:
                        metrics['recall'] = float(recall[-1])
                    break
            
            for key in ['rs_test_f1', 'test_f1', 'f1']:
                if key in f:
                    f1 = f[key][:]
                    if len(f1) > 0:
                        metrics['f1'] = float(f1[-1])
                    break
            
            if 'precision' in metrics and 'recall' in metrics and 'f1' not in metrics:
                p, r = metrics['precision'], metrics['recall']
                if p + r > 0:
                    metrics['f1'] = 2 * p * r / (p + r)
            
            if 'accuracy' not in metrics:
                return None
            
            return metrics
    
    except Exception as e:
        return None


def parse_folder_name(folder_name):
    """
    解析文件夹名称，提取数据集、算法、异质性类型、是否为消融实验
    
    Args:
        folder_name: 例如 
            "Uci_FedAvg_feature" (基线)
            "Uci_FedGpro_Ablation_Full_Model_feature" (消融)
    
    Returns:
        tuple: (dataset, algorithm, heterogeneity, is_ablation, ablation_config) 或 None
    """
    parts = folder_name.split('_')
    
    if len(parts) < 3:
        return None
    
    dataset = parts[0]  # Uci 或 Xinwang
    
    # 过滤掉无效的异质性类型
    valid_heterogeneities = ['feature', 'label', 'quantity', 'iid']
    heterogeneity = parts[-1]  # feature, label, quantity, iid
    if heterogeneity not in valid_heterogeneities:
        return None
    
    # 过滤掉中心化训练结果
    if 'Centralized' in folder_name:
        return None
    
    # 判断是否为消融实验
    is_ablation = 'Ablation' in folder_name
    
    if is_ablation:
        # 消融实验: Uci_FedGpro_Ablation_Full_Model_feature
        # parts: ['Uci', 'FedGpro', 'Ablation', 'Full', 'Model', 'feature']
        # 算法总是 FedGpro，消融配置 = 'Full_Model'
        
        if len(parts) < 5:  # 至少要有 Dataset, FedGpro, Ablation, Config, Hetero
            return None
        
        algorithm = 'FedGpro'
        
        # 消融配置名 = Ablation 后面到 Hetero 前面的所有部分
        ablation_config = '_'.join(parts[3:-1])  # e.g., "Full_Model"
        
        # 合并重复的消融配置名
        if ablation_config in ABLATION_CONFIG_MAPPING:
            ablation_config = ABLATION_CONFIG_MAPPING[ablation_config]
        
        # 过滤掉无效的消融配置
        if ablation_config not in VALID_ABLATION_CONFIGS:
            return None
        
        return dataset, algorithm, heterogeneity, is_ablation, ablation_config
    else:
        # 基线实验: Uci_FedAvg_feature
        # 或者: Uci_FedGpro_feature
        algorithm = '_'.join(parts[1:-1])
        
        # 过滤掉无效的基线算法（如与本文无关的 FedGwo, FedPso）
        if algorithm not in VALID_BASELINE_ALGORITHMS:
            return None
        
        return dataset, algorithm, heterogeneity, is_ablation, None


def collect_all_results(results_dir):
    """
    收集所有实验结果，分别存储基线实验和消融实验
    
    Args:
        results_dir: system/results目录路径
    
    Returns:
        tuple: (baseline_results, ablation_results)
            baseline_results: {(dataset, algorithm, heterogeneity): [metrics_dict_1, ...]}
            ablation_results: {(dataset, ablation_config, heterogeneity): [metrics_dict_1, ...]}
    """
    baseline_results = defaultdict(list)
    ablation_results = defaultdict(list)
    
    # 遍历所有文件夹
    for folder_name in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        # 解析文件夹名称
        parsed = parse_folder_name(folder_name)
        if parsed is None:
            continue
        
        dataset, algorithm, heterogeneity, is_ablation, ablation_config = parsed
        
        # 读取该文件夹下的所有h5文件
        h5_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.h5')])
        
        if len(h5_files) == 0:
            continue
        
        # 读取每个重复实验的结果
        for h5_file in h5_files:
            h5_path = os.path.join(folder_path, h5_file)
            metrics = read_h5_metrics(h5_path)
            
            if metrics is not None:
                if is_ablation:
                    print(f"Ablation: {dataset} | {ablation_config} | {heterogeneity} | Files: {len(h5_files)}")
                    ablation_results[(dataset, ablation_config, heterogeneity)].append(metrics)
                else:
                    print(f"Baseline: {dataset} | {algorithm} | {heterogeneity} | Files: {len(h5_files)}")
                    baseline_results[(dataset, algorithm, heterogeneity)].append(metrics)
    
    return baseline_results, ablation_results


def compute_statistics(metrics_list):
    """
    计算指标的均值和标准差
    
    Args:
        metrics_list: [{'accuracy': ..., 'precision': ..., ...}, ...]
    
    Returns:
        dict: {'accuracy_mean': ..., 'accuracy_std': ..., ...}
    """
    if len(metrics_list) == 0:
        return None
    
    stats = {}
    
    # 收集所有可能的指标键
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())
    
    # 对每个指标计算均值和标准差
    for key in all_keys:
        values = [m.get(key, np.nan) for m in metrics_list]
        values = [v for v in values if not np.isnan(v)]  # 过滤NaN
        
        if len(values) > 0:
            stats[f'{key}_mean'] = np.mean(values)
            stats[f'{key}_std'] = np.std(values, ddof=1) if len(values) > 1 else 0.0
            stats[f'{key}_count'] = len(values)
    
    return stats


def format_mean_std(mean, std):
    """
    格式化为 "均值±标准差" 字符串
    
    Args:
        mean: 均值
        std: 标准差
    
    Returns:
        str: "0.8532±0.0123"
    """
    return f"{mean:.4f}±{std:.4f}"


def generate_ablation_tables(ablation_results):
    """
    为消融实验的每个数据集和异质性生成单独的表格
    
    Args:
        ablation_results: {(dataset, ablation_config, heterogeneity): [metrics_dict_1, ...]}
    
    Returns:
        dict: {(dataset, heterogeneity): pd.DataFrame}
              包含消融配置作为行，指标作为列
    """
    tables = {}
    
    # 获取所有数据集和异质性类型
    datasets = sorted(set(k[0] for k in ablation_results.keys()))
    heterogeneities = sorted(set(k[2] for k in ablation_results.keys()))
    
    for dataset in datasets:
        for heterogeneity in heterogeneities:
            # 过滤当前数据集和异质性的结果
            filtered_results = {
                k: v for k, v in ablation_results.items()
                if k[0] == dataset and k[2] == heterogeneity
            }
            
            if len(filtered_results) == 0:
                continue
            
            rows = []
            for (_, ablation_config, _), metrics_list in sorted(filtered_results.items()):
                stats = compute_statistics(metrics_list)
                
                if stats is None:
                    continue
                
                row = {
                    'Ablation_Config': ablation_config,
                    'Runs': stats.get('accuracy_count', 0)
                }
                
                # 添加各指标
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    mean_key = f'{metric}_mean'
                    std_key = f'{metric}_std'
                    
                    if mean_key in stats:
                        mean_val = stats[mean_key]
                        std_val = stats.get(std_key, 0.0)
                        row[metric.capitalize()] = format_mean_std(mean_val, std_val)
                    else:
                        row[metric.capitalize()] = 'N/A'
                
                rows.append(row)
            
            if len(rows) > 0:
                df = pd.DataFrame(rows)
                df = df.sort_values('Ablation_Config')
                tables[(dataset, heterogeneity)] = df
    
    return tables


def generate_ablation_tables_by_config(ablation_results):
    """
    按消融配置类型分组生成表格（而不是按数据集）
    
    Args:
        ablation_results: {(dataset, ablation_config, heterogeneity): [metrics_dict_1, ...]}
    
    Returns:
        dict: {ablation_config: pd.DataFrame}
              包含数据集和异质性作为行，指标作为列
    """
    tables = {}
    
    # 获取所有消融配置
    ablation_configs = sorted(set(k[1] for k in ablation_results.keys()))
    
    for config in ablation_configs:
        # 过滤当前配置的所有结果（不管数据集和异质性）
        filtered_results = {
            k: v for k, v in ablation_results.items()
            if k[1] == config
        }
        
        if len(filtered_results) == 0:
            continue
        
        rows = []
        for (dataset, _, heterogeneity), metrics_list in sorted(filtered_results.items()):
            stats = compute_statistics(metrics_list)
            
            if stats is None:
                continue
            
            row = {
                'Dataset': dataset,
                'Heterogeneity': heterogeneity,
                'Runs': stats.get('accuracy_count', 0)
            }
            
            # 添加各指标
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                mean_key = f'{metric}_mean'
                std_key = f'{metric}_std'
                
                if mean_key in stats:
                    mean_val = stats[mean_key]
                    std_val = stats.get(std_key, 0.0)
                    row[metric.capitalize()] = format_mean_std(mean_val, std_val)
                else:
                    row[metric.capitalize()] = 'N/A'
            
            rows.append(row)
        
        if len(rows) > 0:
            df = pd.DataFrame(rows)
            # 按数据集和异质性排序
            df = df.sort_values(['Dataset', 'Heterogeneity'])
            tables[config] = df
    
    return tables


def generate_dataset_heterogeneity_tables(baseline_results):
    """
    为每个数据集的每种异质性生成单独的表格
    
    Args:
        baseline_results: {(dataset, algorithm, heterogeneity): [metrics_dict_1, ...]}
    
    Returns:
        dict: {(dataset, heterogeneity): pd.DataFrame}
    """
    tables = {}
    
    # 获取所有数据集和异质性类型
    datasets = sorted(set(k[0] for k in baseline_results.keys()))
    heterogeneities = sorted(set(k[2] for k in baseline_results.keys()))
    
    for dataset in datasets:
        for heterogeneity in heterogeneities:
            # 过滤当前数据集和异质性的结果
            filtered_results = {
                k: v for k, v in baseline_results.items()
                if k[0] == dataset and k[2] == heterogeneity
            }
            
            if len(filtered_results) == 0:
                continue
            
            rows = []
            for (_, algorithm, _), metrics_list in sorted(filtered_results.items()):
                stats = compute_statistics(metrics_list)
                
                if stats is None:
                    continue
                
                row = {
                    'Algorithm': algorithm,
                    'Runs': stats.get('accuracy_count', 0)
                }
                
                # 添加各指标
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    mean_key = f'{metric}_mean'
                    std_key = f'{metric}_std'
                    
                    if mean_key in stats:
                        mean_val = stats[mean_key]
                        std_val = stats.get(std_key, 0.0)
                        row[metric.capitalize()] = format_mean_std(mean_val, std_val)
                    else:
                        row[metric.capitalize()] = 'N/A'
                
                rows.append(row)
            
            if len(rows) > 0:
                df = pd.DataFrame(rows)
                df = df.sort_values('Algorithm')
                tables[(dataset, heterogeneity)] = df
    
    return tables


def read_training_history(h5_file_path):
    """
    从h5文件读取训练过程中每轮的准确率
    
    Args:
        h5_file_path: h5文件路径
    
    Returns:
        numpy.ndarray: 每轮的测试准确率，如果读取失败返回None
    """
    try:
        with h5py.File(h5_file_path, 'r') as f:
            if 'rs_test_acc' in f:
                test_acc = f['rs_test_acc'][:]
                if len(test_acc) > 0:
                    return test_acc
    except Exception as e:
        print(f"  [WARNING] Failed to read training history from {os.path.basename(h5_file_path)}: {e}")
    return None


def plot_baseline_iid_comparison(baseline_results, base_dir):
    """
    Plot baseline algorithm comparison on IID scenario
    
    Args:
        baseline_results: Dictionary of baseline experiment results
        base_dir: Base directory to save figures
    """
    figures_dir = base_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # Extract IID results for both datasets
    datasets = ['Uci', 'Xinwang']
    algorithms = sorted(VALID_BASELINE_ALGORITHMS)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#2E86AB', '#06A77D', '#A23B72', '#F18F01']
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        data_means = []
        data_stds = []
        
        for algo in algorithms:
            key = (dataset, algo, 'iid')
            if key in baseline_results:
                stats = compute_statistics(baseline_results[key])
                if stats:
                    data_means.append(stats.get('accuracy_mean', 0) * 100)
                    data_stds.append(stats.get('accuracy_std', 0) * 100)
                else:
                    data_means.append(0)
                    data_stds.append(0)
            else:
                data_means.append(0)
                data_stds.append(0)
        
        x = np.arange(len(algorithms))
        bars = ax.bar(x, data_means, yerr=data_stds, capsize=5, 
                     alpha=0.85, color=colors, edgecolor='black', linewidth=1.2)
        
        ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{dataset} Dataset (IID)', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=0, ha='center', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, 100])
        
        # Add value labels
        for bar, mean, std in zip(bars, data_means, data_stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                   f'{mean:.2f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_file = figures_dir / 'baseline_iid_comparison'
    plt.savefig(f'{output_file}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_file}.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  [Figure] IID comparison saved: {output_file}.png")


def plot_ablation_convergence(ablation_results, base_dir, results_dir):
    """
    Plot convergence curves for ablation experiments (accuracy vs. communication rounds)
    
    Args:
        ablation_results: Dictionary of ablation experiment results
        base_dir: Base directory to save figures
        results_dir: Directory containing raw result files
    """
    figures_dir = base_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    target_configs = [
        'Full_Model',
        'No_Prototype', 
        'No_VAE_Generation',
        'Phase2_Scaffold',
        'Privacy_Epsilon1',
        'Privacy_Epsilon5',
        'Privacy_Epsilon10',
    ]
    
    config_labels = {
        'Full_Model': 'Full Model',
        'No_Prototype': 'Phase1-NoPrototype',
        'No_VAE_Generation': 'Phase1-NoVAE',
        'Phase2_Scaffold': 'Phase2-Scaffold',
        'Privacy_Epsilon1': 'Privacy-DP(ε=1)',
        'Privacy_Epsilon5': 'Privacy-DP(ε=5)',
        'Privacy_Epsilon10': 'Privacy-DP(ε=10)',
    }
    
    # Process both datasets
    for dataset in ['Uci', 'Xinwang']:
        convergence_data = {}
        
        for config in target_configs:
            folder_name = f"{dataset}_FedGpro_Ablation_{config}_feature"
            folder_path = results_dir / folder_name
            
            if not folder_path.exists():
                continue
            
            h5_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.h5')])
            histories = []
            
            for h5_file in h5_files:
                h5_path = folder_path / h5_file
                history = read_training_history(h5_path)
                if history is not None:
                    histories.append(history)
            
            if len(histories) > 0:
                min_length = min(len(h) for h in histories)
                histories_array = np.array([h[:min_length] for h in histories])
                mean_history = np.mean(histories_array, axis=0)
                std_history = np.std(histories_array, axis=0)
                
                convergence_data[config] = {
                    'mean': mean_history,
                    'std': std_history,
                    'label': config_labels.get(config, config)
                }
        
        if len(convergence_data) == 0:
            continue
        
        # Plot convergence curves
        fig, ax = plt.subplots(figsize=(10, 6))
    
        colors = {
            'Full_Model': '#2E86AB',
            'No_Prototype': '#A23B72',
            'No_VAE_Generation': '#F18F01',
            'Phase2_Scaffold': '#06A77D',
            'Privacy_Epsilon1': '#C73E1D',
            'Privacy_Epsilon5': '#E63946',
            'Privacy_Epsilon10': '#F77F00',
        }
        
        linestyles = {
            'Full_Model': '-',
            'No_Prototype': '--',
            'No_VAE_Generation': '--',
            'Phase2_Scaffold': '-.',
            'Privacy_Epsilon1': ':',
            'Privacy_Epsilon5': ':',
            'Privacy_Epsilon10': ':',
        }
        
        for config in target_configs:
            if config not in convergence_data:
                continue
            
            data = convergence_data[config]
            mean_acc = data['mean']
            std_acc = data['std']
            label = data['label']
            
            rounds = np.arange(1, len(mean_acc) + 1)
            
            ax.plot(rounds, mean_acc, 
                    label=label,
                    color=colors.get(config, '#000000'),
                    linestyle=linestyles.get(config, '-'),
                    linewidth=2.5 if config == 'Full_Model' else 2,
                    alpha=0.9)
            
            if config in ['Full_Model', 'No_Prototype', 'No_VAE_Generation']:
                ax.fill_between(rounds[::5], 
                               (mean_acc - std_acc)[::5], 
                               (mean_acc + std_acc)[::5],
                               color=colors.get(config, '#000000'),
                               alpha=0.15)
        
        ax.set_xlabel('Communication Rounds', fontsize=13, fontweight='bold')
        ax.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
        ax.set_title(f'Ablation Study Convergence ({dataset} Dataset - Feature Heterogeneity)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(left=1)
        
        plt.tight_layout()
        output_file = figures_dir / f'ablation_convergence_{dataset.lower()}'
        plt.savefig(f'{output_file}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_file}.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"  [Figure] Convergence curve saved: {output_file}.png")




def main():
    """主函数"""
    print("="*80)
    print("联邦学习实验结果统计分析")
    print("="*80)
    
    # 设置路径
    base_dir = Path(__file__).parent
    results_dir = base_dir / 'system' / 'results'
    
    if not results_dir.exists():
        print(f"[ERROR] Results directory not found: {results_dir}")
        return
    
    print(f"\n[INFO] Reading results from: {results_dir}")
    print(f"[INFO] Expected structure: Dataset_Algorithm_Heterogeneity/*.h5")
    print(f"[INFO] Valid baseline algorithms: {sorted(VALID_BASELINE_ALGORITHMS)}")
    print(f"[INFO] Valid ablation configs: {sorted(VALID_ABLATION_CONFIGS)}\n")
    
    # 收集所有结果（分基线和消融）
    print("[STEP 1] Collecting all experiment results...")
    baseline_results, ablation_results = collect_all_results(str(results_dir))
    
    print(f"\n[INFO] Baseline configurations: {len(baseline_results)}")
    print(f"[INFO] Ablation configurations: {len(ablation_results)}")
    
    if len(baseline_results) == 0 and len(ablation_results) == 0:
        print("[ERROR] No valid results found!")
        return
    
    # 生成按数据集和异质性分组的表格
    print("\n[STEP 2] Generating baseline experiment tables...")
    baseline_tables = generate_dataset_heterogeneity_tables(baseline_results)
    print(f"  Generated {len(baseline_tables)} baseline tables")
    
    print("\n[STEP 3] Generating ablation experiment tables...")
    ablation_tables = generate_ablation_tables(ablation_results)
    print(f"  Generated {len(ablation_tables)} ablation tables")
    
    # 保存为两个独立的Excel文件
    print("\n[STEP 4] Generating separate Excel files...")
    
    # ========== 基线实验Excel文件 ==========
    baseline_excel = base_dir / 'baseline_experiments.xlsx'
    if len(baseline_tables) > 0:
        try:
            with pd.ExcelWriter(baseline_excel, engine='openpyxl') as writer:
                for (dataset, heterogeneity), df in sorted(baseline_tables.items()):
                    sheet_name = f'{dataset}_{heterogeneity}'
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"  [Baseline] Added sheet: {sheet_name} ({len(df)} algorithms)")
            
            print(f"\n[SUCCESS] Baseline Excel saved to: {baseline_excel}")
            print(f"[INFO] Total baseline sheets: {len(baseline_tables)}")
        except Exception as e:
            print(f"\n[ERROR] Failed to create baseline Excel: {e}")
            print("[INFO] You may need to install openpyxl: pip install openpyxl")
    
    # ========== 消融实验Excel文件 ==========
    ablation_excel = base_dir / 'ablation_experiments.xlsx'
    if len(ablation_tables) > 0:
        try:
            with pd.ExcelWriter(ablation_excel, engine='openpyxl') as writer:
                for (dataset, heterogeneity), df in sorted(ablation_tables.items()):
                    sheet_name = f'{dataset}_{heterogeneity}'
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"  [Ablation] Added sheet: {sheet_name} ({len(df)} configurations)")
            
            print(f"\n[SUCCESS] Ablation Excel saved to: {ablation_excel}")
            print(f"[INFO] Total ablation sheets: {len(ablation_tables)}")
        except Exception as e:
            print(f"\n[ERROR] Failed to create ablation Excel: {e}")
            print("[INFO] You may need to install openpyxl: pip install openpyxl")
    
    # ========== STEP 5: Generate figures ==========
    print("\n[STEP 5] Generating experimental figures...")
    if len(baseline_results) > 0:
        plot_baseline_iid_comparison(baseline_results, base_dir)
    if len(ablation_results) > 0:
        plot_ablation_convergence(ablation_results, base_dir, results_dir)
    
    # 显示统计摘要
    print("\n" + "="*80)
    print("统计摘要")
    print("="*80)
    
    # 基线实验统计
    if len(baseline_results) > 0:
        baseline_df = pd.concat([df for df in baseline_tables.values()], ignore_index=True)
        
        print("\n【基线实验 (Baseline Experiments)】")
        datasets = sorted(set(k[0] for k in baseline_results.keys()))
        heterogeneities = sorted(set(k[2] for k in baseline_results.keys()))
        algorithms = sorted(set(k[1] for k in baseline_results.keys()))
        
        print(f"  数据集: {', '.join(datasets)}")
        print(f"  异质性类型: {', '.join(heterogeneities)}")
        print(f"  算法数: {len(algorithms)}")
        print(f"  总配置数: {len(baseline_results)}")
    
    # 消融实验统计
    if len(ablation_results) > 0:
        ablation_df = pd.concat([df for df in ablation_tables.values()], ignore_index=True)
        
        print("\n【消融实验 (Ablation Experiments)】")
        datasets = sorted(set(k[0] for k in ablation_results.keys()))
        heterogeneities = sorted(set(k[2] for k in ablation_results.keys()))
        configs = sorted(set(k[1] for k in ablation_results.keys()))
        
        print(f"  数据集: {', '.join(datasets)}")
        print(f"  异质性类型: {', '.join(heterogeneities)}")
        print(f"  消融配置数: {len(configs)}")
        print(f"  总配置数: {len(ablation_results)}")
    
    # 按配置类型分组的消融实验统计
    print("\n【消融实验按配置类型分类】")
    ablation_by_config = generate_ablation_tables_by_config(ablation_results)
    if len(ablation_by_config) > 0:
        print(f"  检测到 {len(ablation_by_config)} 种消融配置:")
        for config in sorted(ablation_by_config.keys()):
            num_rows = len(ablation_by_config[config])
            print(f"    - {config:<30} ({num_rows} 种数据集/异质性组合)")
    
    # 显示样本数据
    if len(baseline_tables) > 0:
        print("\n" + "="*80)
        print("基线实验示例 (前10行)")
        print("="*80)
        first_table = list(baseline_tables.values())[0]
        print(first_table.head(10).to_string(index=False))
    
    if len(ablation_tables) > 0:
        print("\n" + "="*80)
        print("消融实验示例 (按数据集/异质性分组 - 前10行)")
        print("="*80)
        first_table = list(ablation_tables.values())[0]
        print(first_table.head(10).to_string(index=False))
    
    # 按配置类型分组的消融实验统计
    if len(ablation_results) > 0:
        ablation_tables_by_config = generate_ablation_tables_by_config(ablation_results)
        
        print("\n" + "="*80)
        print("消融实验按配置类型分类统计")
        print("="*80)
        
        print(f"\n【消融配置统计】")
        print(f"  总消融配置数: {len(ablation_tables_by_config)}")
        print(f"  每个配置包含: 2个数据集 × 4种异质性 = 8个(数据集,异质性)组合")
        print(f"  每个组合5次重复 = 40个实验 × 18个配置 = 720次")
        
        print(f"\n【各配置实验数据】")
        print(f"{'配置名称':<30} {'覆盖范围':<40} {'行数'}")
        print("-" * 80)
        
        for config in sorted(ablation_tables_by_config.keys()):
            df = ablation_tables_by_config[config]
            datasets = sorted(set(df['Dataset']))
            heterogeneities = sorted(set(df['Heterogeneity']))
            coverage = f"{len(datasets)}数据集×{len(heterogeneities)}异质性"
            print(f"{config:<30} {coverage:<40} {len(df):3d}行")
        
        # 显示某个配置的样本数据
        if len(ablation_tables_by_config) > 0:
            print(f"\n【Full_Model配置示例】")
            if 'Full_Model' in ablation_tables_by_config:
                sample_df = ablation_tables_by_config['Full_Model']
                print(sample_df.head(10).to_string(index=False))
    
    print("\n" + "="*80)
    print("Analysis Complete")
    print("="*80)
    print(f"\n生成的文件:")
    print(f"\n  [1] baseline_experiments.xlsx")
    print(f"      包含 {len(baseline_tables)} 个工作表 (按数据集和异质性分组)")
    print(f"      内容: 4 种核心基线算法的性能对比")
    print(f"      工作表示例: Uci_feature, Uci_iid, Xinwang_label, ...")
    print(f"\n  [2] ablation_experiments.xlsx")
    print(f"      包含 {len(ablation_tables)} 个工作表 (按数据集和异质性分组)")
    print(f"      内容: 8 种消融配置的性能分析")
    print(f"      工作表示例: Uci_feature, Uci_iid, Xinwang_label, ...")
    print(f"\n  [3] figures/")
    print(f"      Experimental figures")
    print(f"      - baseline_iid_comparison.png: IID scenario performance")
    print(f"      - ablation_convergence_uci.png: Uci ablation convergence")
    print(f"      - ablation_convergence_xinwang.png: Xinwang ablation convergence")
    print()


if __name__ == '__main__':
    main()
