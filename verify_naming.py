#!/usr/bin/env python
"""全面检查命名规范和逻辑"""

from pathlib import Path
import re

results_dir = Path('system/results')

print("="*100)
print("1. 检查文件夹命名规范")
print("="*100)

# 预期的命名规则
baseline_pattern = r'^([A-Za-z0-9]+)_([A-Za-z0-9]+)_(feature|iid|label|quantity)$'
ablation_pattern = r'^([A-Za-z0-9]+)_FedGpro_Ablation_([A-Za-z0-9_]+)_(feature|iid|label|quantity)$'

all_folders = sorted([f for f in results_dir.iterdir() if f.is_dir()])

# 统计
baseline_folders = []
ablation_folders = []
other_folders = []
wrong_folders = []

for folder in all_folders:
    name = folder.name
    
    # 检查是否是旧名称（包含 FedGpro-FedAvg）
    if 'FedGpro-FedAvg' in name:
        wrong_folders.append((name, '❌ 旧名称 (包含FedGpro-FedAvg)'))
        continue
    
    # 检查是否匹配消融实验模式
    if re.match(ablation_pattern, name):
        ablation_folders.append((name, '✅ 消融实验'))
    # 检查是否匹配基线实验模式
    elif re.match(baseline_pattern, name):
        baseline_folders.append((name, '✅ 基线实验'))
    else:
        other_folders.append((name, '⚠️ 特殊文件夹'))

print(f"\n【命名规范检查】")
print(f"  基线实验文件夹: {len(baseline_folders)} 个")
print(f"  消融实验文件夹: {len(ablation_folders)} 个")
print(f"  特殊文件夹:   {len(other_folders)} 个")
print(f"  ❌ 错误命名:   {len(wrong_folders)} 个")

if wrong_folders:
    print(f"\n【发现错误命名的文件夹】")
    for name, status in wrong_folders[:10]:
        print(f"  {name:<50} {status}")
    if len(wrong_folders) > 10:
        print(f"  ... 还有 {len(wrong_folders) - 10} 个")

print(f"\n【基线实验样本（前5个）】")
for name, status in baseline_folders[:5]:
    h5_count = len(list((results_dir / name).glob('*.h5')))
    print(f"  {name:<50} {status} ({h5_count} files)")

print(f"\n【消融实验样本（前5个）】")
for name, status in ablation_folders[:5]:
    h5_count = len(list((results_dir / name).glob('*.h5')))
    print(f"  {name:<50} {status} ({h5_count} files)")

# ===================================================================
print("\n" + "="*100)
print("2. 检查.h5文件内部的命名规范")
print("="*100)

# 抽取几个文件夹检查内部的.h5文件名
sample_folders_to_check = [
    'Uci_FedGpro_Ablation_Full_Model_feature',
    'Uci_FedGpro_Ablation_No_VAE_Generation_feature',
    'Uci_FedAvg_feature',
    'Xinwang_FedGpro_feature'
]

print(f"\n【样本文件夹内部的.h5文件检查】\n")

for folder_name in sample_folders_to_check:
    folder_path = results_dir / folder_name
    if not folder_path.exists():
        print(f"{folder_name}: ❌ 不存在")
        continue
    
    h5_files = sorted(list(folder_path.glob('*.h5')))
    print(f"\n{folder_name}:")
    print(f"  总文件数: {len(h5_files)}")
    
    if h5_files:
        # 检查文件名规范
        old_pattern_count = 0
        new_pattern_count = 0
        
        for h5_file in h5_files:
            if 'FedGpro-FedAvg' in h5_file.name:
                old_pattern_count += 1
            else:
                new_pattern_count += 1
        
        print(f"  旧格式文件: {old_pattern_count} 个 (❌ FedGpro-FedAvg)")
        print(f"  新格式文件: {new_pattern_count} 个 (✅ FedGpro)")
        
        # 显示前2个文件
        for h5_file in h5_files[:2]:
            print(f"    - {h5_file.name}")
        if len(h5_files) > 2:
            print(f"    ... 还有 {len(h5_files) - 2} 个文件")

# ===================================================================
print("\n" + "="*100)
print("3. 检查analyze_results.py的parse_folder_name逻辑")
print("="*100)

# 导入parse_folder_name函数
import sys
sys.path.insert(0, str(Path.cwd()))

try:
    from analyze_results import parse_folder_name
    
    test_folders = [
        'Uci_FedAvg_feature',
        'Xinwang_FedGpro_feature',
        'Uci_FedGpro_Ablation_Full_Model_feature',
        'Xinwang_FedGpro_Ablation_Privacy_Epsilon_1.0_label',
        'Uci_FedDitto_label',
        'Xinwang_Per-FedAvg_quantity'
    ]
    
    print(f"\n【parse_folder_name函数测试】\n")
    
    for folder_name in test_folders:
        try:
            result = parse_folder_name(folder_name)
            if result:
                if len(result) == 5:  # 消融实验
                    dataset, algo, hetero, is_ablation, ablation_config = result
                    print(f"✅ {folder_name}")
                    print(f"   → Dataset: {dataset}, Algorithm: {algo}, Heterogeneity: {hetero}")
                    print(f"   → Is_Ablation: {is_ablation}, Config: {ablation_config}")
                elif len(result) == 3:  # 基线实验
                    dataset, algo, hetero = result
                    print(f"✅ {folder_name}")
                    print(f"   → Dataset: {dataset}, Algorithm: {algo}, Heterogeneity: {hetero}")
            else:
                print(f"❌ {folder_name} → 解析失败 (返回None)")
        except Exception as e:
            print(f"❌ {folder_name} → 解析异常: {e}")
    
except ImportError as e:
    print(f"❌ 无法导入analyze_results: {e}")

# ===================================================================
print("\n" + "="*100)
print("4. 检查run_ablation_experiments.py的命名逻辑")
print("="*100)

try:
    from run_ablation_experiments import ABLATION_CONFIGS, DATASETS, HETEROGENEITY_TYPES
    
    print(f"\n【消融实验配置信息】")
    print(f"  数据集: {DATASETS}")
    print(f"  异质性: {list(HETEROGENEITY_TYPES.keys())}")
    print(f"  消融配置数: {len(ABLATION_CONFIGS)}")
    
    print(f"\n【预期生成的文件夹名称示例】")
    
    # 按照新标准生成文件夹名
    for dataset in DATASETS[:1]:  # 只取第一个数据集
        for hetero in list(HETEROGENEITY_TYPES.keys())[:1]:  # 只取第一个异质性
            for config_name in list(ABLATION_CONFIGS.keys())[:3]:  # 只取前3个配置
                expected_folder = f"{dataset}_FedGpro_Ablation_{config_name}_{hetero}"
                print(f"  {expected_folder}")
    
except ImportError as e:
    print(f"❌ 无法导入run_ablation_experiments: {e}")

print("\n" + "="*100)
