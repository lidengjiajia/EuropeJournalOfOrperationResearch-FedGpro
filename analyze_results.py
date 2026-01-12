"""
统计联邦学习实验结果脚本

功能：
1. 读取system/results目录下所有h5文件
2. 按数据集(Uci/Xinwang)、算法、异质性类型(feature/label/quantity/iid)分组
3. 每个实验5次重复，计算均值±标准差
4. 提取指标：accuracy, precision, recall, f1
5. 输出CSV文件

作者：AI Assistant
日期：2025-12-30
"""

import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def read_h5_metrics(h5_file_path):
    """
    从h5文件读取最终轮次的测试指标
    
    Args:
        h5_file_path: h5文件路径
    
    Returns:
        dict: {'accuracy': float, 'precision': float, 'recall': float, 'f1': float}
              如果读取失败返回None
    """
    try:
        with h5py.File(h5_file_path, 'r') as f:
            # 读取所有可能的指标键
            metrics = {}
            
            # 尝试读取测试集指标
            if 'rs_test_acc' in f:
                test_acc = f['rs_test_acc'][:]
                if len(test_acc) > 0:
                    metrics['accuracy'] = float(test_acc[-1])  # 最后一轮的准确率
            
            # 尝试读取precision（可能的键名）
            for key in ['rs_test_precision', 'test_precision', 'precision']:
                if key in f:
                    precision = f[key][:]
                    if len(precision) > 0:
                        metrics['precision'] = float(precision[-1])
                    break
            
            # 尝试读取recall
            for key in ['rs_test_recall', 'test_recall', 'recall']:
                if key in f:
                    recall = f[key][:]
                    if len(recall) > 0:
                        metrics['recall'] = float(recall[-1])
                    break
            
            # 尝试读取f1
            for key in ['rs_test_f1', 'test_f1', 'f1']:
                if key in f:
                    f1 = f[key][:]
                    if len(f1) > 0:
                        metrics['f1'] = float(f1[-1])
                    break
            
            # 如果没有找到某些指标，尝试从其他指标计算
            if 'precision' in metrics and 'recall' in metrics and 'f1' not in metrics:
                p, r = metrics['precision'], metrics['recall']
                if p + r > 0:
                    metrics['f1'] = 2 * p * r / (p + r)
            
            # 检查是否至少有accuracy
            if 'accuracy' not in metrics:
                print(f"  [WARNING] No accuracy found in {os.path.basename(h5_file_path)}")
                return None
            
            return metrics
    
    except Exception as e:
        print(f"  [ERROR] Failed to read {os.path.basename(h5_file_path)}: {e}")
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
        
        return dataset, algorithm, heterogeneity, is_ablation, ablation_config
    else:
        # 基线实验: Uci_FedAvg_feature
        # 或者: Uci_FedGpro_feature
        algorithm = '_'.join(parts[1:-1])
        
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


def generate_summary_table(results):
    """
    生成汇总表格
    
    Args:
        results: {(dataset, algorithm, heterogeneity): [metrics_dict_1, ...]}
    
    Returns:
        pd.DataFrame: 汇总表格
    """
    rows = []
    
    for (dataset, algorithm, heterogeneity), metrics_list in sorted(results.items()):
        stats = compute_statistics(metrics_list)
        
        if stats is None:
            continue
        
        row = {
            'Dataset': dataset,
            'Algorithm': algorithm,
            'Heterogeneity': heterogeneity,
            'Runs': stats.get('accuracy_count', 0)
        }
        
        # 添加各指标的均值±标准差
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            mean_key = f'{metric}_mean'
            std_key = f'{metric}_std'
            
            if mean_key in stats:
                mean_val = stats[mean_key]
                std_val = stats.get(std_key, 0.0)
                row[metric.capitalize()] = format_mean_std(mean_val, std_val)
                
                # 同时保存原始数值（用于排序）
                row[f'{metric}_mean_raw'] = mean_val
                row[f'{metric}_std_raw'] = std_val
            else:
                row[metric.capitalize()] = 'N/A'
                row[f'{metric}_mean_raw'] = np.nan
                row[f'{metric}_std_raw'] = np.nan
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # 按数据集、异质性、算法排序
    df = df.sort_values(['Dataset', 'Heterogeneity', 'Algorithm'])
    
    # 移除原始数值列（仅用于排序）
    display_columns = ['Dataset', 'Algorithm', 'Heterogeneity', 'Runs', 
                      'Accuracy', 'Precision', 'Recall', 'F1']
    df_display = df[display_columns].copy()
    
    return df_display


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



    """
    为每个数据集的每种异质性生成单独的表格
    
    Args:
        results: {(dataset, algorithm, heterogeneity): [metrics_dict_1, ...]}
    
    Returns:
        dict: {(dataset, heterogeneity): pd.DataFrame}
    """
    tables = {}
    
    # 获取所有数据集和异质性类型
    datasets = sorted(set(k[0] for k in results.keys()))
    heterogeneities = sorted(set(k[2] for k in results.keys()))
    
    for dataset in datasets:
        for heterogeneity in heterogeneities:
            # 过滤当前数据集和异质性的结果
            filtered_results = {
                k: v for k, v in results.items()
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
    print(f"[INFO] Expected structure: Dataset_Algorithm_Heterogeneity/*.h5\n")
    
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
    
    # 保存为一个Excel文件，包含多个工作表
    print("\n[STEP 4] Generating consolidated Excel file...")
    excel_file = base_dir / 'experiment_results_summary.xlsx'
    
    try:
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # ========== 基线实验工作表 ==========
            if len(baseline_tables) > 0:
                for (dataset, heterogeneity), df in sorted(baseline_tables.items()):
                    sheet_name = f'Baseline_{dataset}_{heterogeneity}'
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"  - Added sheet: {sheet_name} ({len(df)} algorithms)")
            
            # ========== 消融实验工作表 ==========
            if len(ablation_tables) > 0:
                for (dataset, heterogeneity), df in sorted(ablation_tables.items()):
                    sheet_name = f'Ablation_{dataset}_{heterogeneity}'
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"  - Added sheet: {sheet_name} ({len(df)} configurations)")
        
        print(f"\n[SUCCESS] Excel file saved to: {excel_file}")
        print(f"[INFO] Total sheets: {len(baseline_tables) + len(ablation_tables)} " +
              f"({len(baseline_tables)} baseline + {len(ablation_tables)} ablation)")
    except Exception as e:
        print(f"\n[ERROR] Failed to create Excel file: {e}")
        print("[INFO] You may need to install openpyxl: pip install openpyxl")
    
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
    
    # 显示样本数据
    if len(baseline_tables) > 0:
        print("\n" + "="*80)
        print("基线实验示例 (前10行)")
        print("="*80)
        first_table = list(baseline_tables.values())[0]
        print(first_table.head(10).to_string(index=False))
    
    if len(ablation_tables) > 0:
        print("\n" + "="*80)
        print("消融实验示例 (前10行)")
        print("="*80)
        first_table = list(ablation_tables.values())[0]
        print(first_table.head(10).to_string(index=False))
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)
    print(f"\n生成的文件:")
    print(f"  experiment_results_summary.xlsx")
    print(f"     包含 {len(baseline_tables) + len(ablation_tables)} 个工作表:")
    print(f"       - 基线实验: {len(baseline_tables)} 个 (按数据集和异质性分组)")
    print(f"       - 消融实验: {len(ablation_tables)} 个 (按数据集和异质性分组)")
    print()


if __name__ == '__main__':
    main()
