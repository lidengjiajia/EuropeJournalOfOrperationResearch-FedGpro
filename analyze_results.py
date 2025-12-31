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
    解析文件夹名称，提取数据集、算法、异质性类型
    
    Args:
        folder_name: 例如 "Uci_FedAvg_feature"
    
    Returns:
        tuple: (dataset, algorithm, heterogeneity) 或 None
    """
    parts = folder_name.split('_')
    
    if len(parts) < 3:
        return None
    
    dataset = parts[0]  # Uci 或 Xinwang
    
    # 处理算法名称可能包含多个单词的情况（如 Per-FedAvg）
    heterogeneity = parts[-1]  # feature, label, quantity, iid
    
    # 算法名称是中间所有部分
    algorithm = '_'.join(parts[1:-1])
    
    # 过滤掉无效的异质性类型
    valid_heterogeneities = ['feature', 'label', 'quantity', 'iid']
    if heterogeneity not in valid_heterogeneities:
        return None
    
    # 过滤掉中心化训练结果
    if 'Centralized' in algorithm:
        return None
    
    return dataset, algorithm, heterogeneity


def collect_all_results(results_dir):
    """
    收集所有实验结果
    
    Args:
        results_dir: system/results目录路径
    
    Returns:
        dict: {(dataset, algorithm, heterogeneity): [metrics_dict_1, ..., metrics_dict_5]}
    """
    results = defaultdict(list)
    
    # 遍历所有文件夹
    for folder_name in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        # 解析文件夹名称
        parsed = parse_folder_name(folder_name)
        if parsed is None:
            continue
        
        dataset, algorithm, heterogeneity = parsed
        
        # 读取该文件夹下的所有h5文件
        h5_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.h5')])
        
        if len(h5_files) == 0:
            continue
        
        print(f"Processing: {dataset} | {algorithm} | {heterogeneity} | Files: {len(h5_files)}")
        
        # 读取每个重复实验的结果
        for h5_file in h5_files:
            h5_path = os.path.join(folder_path, h5_file)
            metrics = read_h5_metrics(h5_path)
            
            if metrics is not None:
                results[(dataset, algorithm, heterogeneity)].append(metrics)
    
    return results


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


def generate_dataset_heterogeneity_tables(results):
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
    
    # 收集所有结果
    print("[STEP 1] Collecting all experiment results...")
    results = collect_all_results(str(results_dir))
    
    print(f"\n[INFO] Total experiment configurations: {len(results)}")
    
    if len(results) == 0:
        print("[ERROR] No valid results found!")
        return
    
    # 生成汇总表格
    print("\n[STEP 2] Generating summary table...")
    summary_df = generate_summary_table(results)
    
    # 不再保存CSV文件，只保存XLSX
    print(f"\n[INFO] Total rows: {len(summary_df)}")
    
    # 生成按数据集和异质性分组的表格
    print("\n[STEP 3] Generating dataset-specific tables...")
    dataset_tables = generate_dataset_heterogeneity_tables(results)
    
    # 保存为一个Excel文件，包含多个工作表（按数据集和异质性分组）
    print("\n[STEP 4] Generating consolidated Excel file...")
    excel_file = base_dir / 'experiment_results_summary.xlsx'
    
    try:
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # 首先添加汇总表（所有结果）
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            print(f"  - Added sheet: Summary ({len(summary_df)} configurations)")
            
            # 然后添加按数据集和异质性分组的工作表
            for (dataset, heterogeneity), df in sorted(dataset_tables.items()):
                sheet_name = f'{dataset}_{heterogeneity}'
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"  - Added sheet: {sheet_name} ({len(df)} algorithms)")
        
        print(f"\n[SUCCESS] Excel file saved to: {excel_file}")
        print(f"[INFO] Total sheets: {len(dataset_tables) + 1} (1 summary + {len(dataset_tables)} detail sheets)")
    except Exception as e:
        print(f"\n[ERROR] Failed to create Excel file: {e}")
        print("[INFO] You may need to install openpyxl: pip install openpyxl")
    
    # 显示统计摘要
    print("\n" + "="*80)
    print("统计摘要")
    print("="*80)
    
    datasets = summary_df['Dataset'].unique()
    heterogeneities = summary_df['Heterogeneity'].unique()
    algorithms = summary_df['Algorithm'].unique()
    
    print(f"数据集: {', '.join(datasets)} ({len(datasets)} total)")
    print(f"异质性类型: {', '.join(heterogeneities)} ({len(heterogeneities)} total)")
    print(f"算法: {len(algorithms)} total")
    print(f"总实验配置: {len(summary_df)}")
    
    # 检查重复次数
    runs_counts = summary_df['Runs'].value_counts()
    print(f"\n重复次数分布:")
    for runs, count in runs_counts.items():
        print(f"  {runs}次重复: {count} 个配置")
    
    # 显示前10行示例
    print("\n" + "="*80)
    print("结果示例 (前10行)")
    print("="*80)
    print(summary_df.head(10).to_string(index=False))
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)
    print(f"\n生成的文件:")
    print(f"  experiment_results_summary.xlsx - Excel文件（{len(dataset_tables) + 1}个工作表）")
    print(f"     包含以下工作表:")
    print(f"       - Summary (所有{len(summary_df)}个配置)")
    for dataset in datasets:
        for heterogeneity in heterogeneities:
            if (dataset, heterogeneity) in [(d, h) for d, h in dataset_tables.keys()]:
                print(f"       - {dataset}_{heterogeneity}")
    print()


if __name__ == '__main__':
    main()
