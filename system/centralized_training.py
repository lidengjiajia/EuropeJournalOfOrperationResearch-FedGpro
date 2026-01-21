#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中心化训练脚本 - 作为联邦学习的性能上界参考

功能说明:
- 使用全部数据进行集中式训练
- 测试多种网络配置找到最优
- 为每个数据集生成性能上界

使用方法:
    python centralized_training.py           # 训练所有数据集
    python centralized_training.py --dataset Uci    # 仅训练UCI
    python centralized_training.py --dataset Xinwang  # 仅训练Xinwang
"""

import os
import sys
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime
import argparse

# 项目根目录
BASE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BASE_DIR))

# =============================================================================
# 配置参数
# =============================================================================

DATASETS = ['Uci', 'Xinwang']

# 各配置的超参数
CONFIGS = {
    'Default': {
        'lr': 0.01,
        'batch_size': 64,
        'weight_decay': 0.0001,
        'hidden_layers': [64, 32],
        'dropout': 0.3,
        'lr_step': 20,
        'lr_gamma': 0.9,
    },
    'LowLR': {
        'lr': 0.005,
        'batch_size': 64,
        'weight_decay': 0.0001,
        'hidden_layers': [32, 16],
        'dropout': 0.4,
        'lr_step': 20,
        'lr_gamma': 0.9,
    },
    'HighLR': {
        'lr': 0.02,
        'batch_size': 64,
        'weight_decay': 0.0001,
        'hidden_layers': [64, 32],
        'dropout': 0.3,
        'lr_step': 15,
        'lr_gamma': 0.85,
    },
    'LargeBatch': {
        'lr': 0.006,
        'batch_size': 128,
        'weight_decay': 0.0001,
        'hidden_layers': [128, 64, 32],
        'dropout': 0.3,
        'lr_step': 20,
        'lr_gamma': 0.9,
    },
    'Wide': {
        'lr': 0.008,
        'batch_size': 64,
        'weight_decay': 0.0001,
        'hidden_layers': [128, 64],
        'dropout': 0.3,
        'lr_step': 20,
        'lr_gamma': 0.9,
    },
    'Deep3Layer': {
        'lr': 0.008,
        'batch_size': 64,
        'weight_decay': 0.0001,
        'hidden_layers': [64, 32, 16],
        'dropout': 0.3,
        'lr_step': 20,
        'lr_gamma': 0.9,
    },
    'Deep4Layer': {
        'lr': 0.008,
        'batch_size': 64,
        'weight_decay': 0.0001,
        'hidden_layers': [128, 64, 32, 16],
        'dropout': 0.35,
        'lr_step': 20,
        'lr_gamma': 0.9,
    },
    'WideDeep': {
        'lr': 0.006,
        'batch_size': 64,
        'weight_decay': 0.0001,
        'hidden_layers': [256, 128, 64, 32],
        'dropout': 0.4,
        'lr_step': 25,
        'lr_gamma': 0.9,
    },
}

EPOCHS = 100
NUM_REPEATS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# 模型定义
# =============================================================================

class CentralizedMLP(nn.Module):
    """可配置的多层感知机"""
    
    def __init__(self, input_dim, hidden_layers, num_classes=2, dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# =============================================================================
# 数据加载
# =============================================================================

def load_centralized_data(dataset_name):
    """加载数据集的所有客户端数据合并为中心化数据"""
    
    # 使用label异质性的数据（任意一种都可以，因为合并后是一样的）
    data_dir = BASE_DIR / 'dataset' / 'dataset' / dataset_name / 'label'
    
    if not data_dir.exists():
        # 尝试备用路径
        data_dir = BASE_DIR / 'dataset' / dataset_name / 'label'
    
    if not data_dir.exists():
        print(f"[ERROR] 数据目录不存在: {data_dir}")
        return None, None, None, None
    
    # 加载所有客户端数据
    train_data_list = []
    train_label_list = []
    test_data_list = []
    test_label_list = []
    
    for npz_file in sorted(data_dir.glob('*.npz')):
        data = np.load(npz_file, allow_pickle=True)
        
        if 'train_data' in data:
            train_data_list.append(data['train_data'])
            train_label_list.append(data['train_label'])
        
        if 'test_data' in data:
            test_data_list.append(data['test_data'])
            test_label_list.append(data['test_label'])
    
    if len(train_data_list) == 0:
        print(f"[ERROR] 未找到训练数据文件")
        return None, None, None, None
    
    # 合并所有数据
    train_data = np.concatenate(train_data_list, axis=0)
    train_label = np.concatenate(train_label_list, axis=0)
    test_data = np.concatenate(test_data_list, axis=0)
    test_label = np.concatenate(test_label_list, axis=0)
    
    print(f"  [INFO] 加载数据: 训练集 {train_data.shape}, 测试集 {test_data.shape}")
    
    return train_data, train_label, test_data, test_label


# =============================================================================
# 训练与评估
# =============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)
    
    return total_loss / total, correct / total


def evaluate(model, test_loader, device):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            correct += pred.eq(target).sum().item()
            total += data.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = correct / total
    
    # 计算precision, recall, f1
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    return accuracy, precision, recall, f1


def train_centralized(dataset_name, config_name, config, repeat_id, save_dir):
    """执行一次中心化训练"""
    
    # 加载数据
    train_data, train_label, test_data, test_label = load_centralized_data(dataset_name)
    if train_data is None:
        return None
    
    # 转换为张量
    train_dataset = TensorDataset(
        torch.FloatTensor(train_data),
        torch.LongTensor(train_label)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(test_data),
        torch.LongTensor(test_label)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 创建模型
    input_dim = train_data.shape[1]
    num_classes = len(np.unique(train_label))
    model = CentralizedMLP(
        input_dim=input_dim,
        hidden_layers=config['hidden_layers'],
        num_classes=num_classes,
        dropout=config['dropout']
    ).to(DEVICE)
    
    # 优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step'], gamma=config['lr_gamma'])
    
    # 训练记录
    train_acc_history = []
    test_acc_history = []
    test_precision_history = []
    test_recall_history = []
    test_f1_history = []
    best_acc = 0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, DEVICE)
        scheduler.step()
        
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        test_precision_history.append(test_prec)
        test_recall_history.append(test_rec)
        test_f1_history.append(test_f1)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d}: Train={train_acc:.4f}, Test={test_acc:.4f}, Best={best_acc:.4f}")
    
    # 保存结果
    save_path = save_dir / f"Centralized_{config_name}_{repeat_id}.h5"
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('rs_train_acc', data=np.array(train_acc_history))
        f.create_dataset('rs_test_acc', data=np.array(test_acc_history))
        f.create_dataset('rs_test_precision', data=np.array(test_precision_history))
        f.create_dataset('rs_test_recall', data=np.array(test_recall_history))
        f.create_dataset('rs_test_f1', data=np.array(test_f1_history))
        f.attrs['config_name'] = config_name
        f.attrs['best_accuracy'] = best_acc
        f.attrs['final_accuracy'] = test_acc_history[-1]
    
    return {
        'best_acc': best_acc,
        'final_acc': test_acc_history[-1],
        'final_f1': test_f1_history[-1],
    }


def run_centralized_experiments(dataset_name):
    """运行指定数据集的所有中心化实验"""
    
    print(f"\n{'='*80}")
    print(f"中心化训练: {dataset_name}")
    print(f"{'='*80}")
    
    # 创建保存目录
    save_dir = BASE_DIR / 'system' / 'results' / f'{dataset_name}_Centralized'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已完成
    existing_files = list(save_dir.glob('*.h5'))
    expected_files = len(CONFIGS) * NUM_REPEATS
    
    if len(existing_files) >= expected_files:
        print(f"[INFO] 已完成 ({len(existing_files)}/{expected_files} 文件)，跳过")
        return
    
    results = {}
    
    for config_name, config in CONFIGS.items():
        print(f"\n[CONFIG] {config_name}")
        print(f"  lr={config['lr']}, batch={config['batch_size']}, layers={config['hidden_layers']}")
        
        config_results = []
        
        for repeat in range(NUM_REPEATS):
            # 设置随机种子
            torch.manual_seed(42 + repeat)
            np.random.seed(42 + repeat)
            
            result = train_centralized(dataset_name, config_name, config, repeat, save_dir)
            if result:
                config_results.append(result)
                print(f"  [Repeat {repeat}] Best={result['best_acc']:.4f}, Final={result['final_acc']:.4f}")
        
        if config_results:
            best_accs = [r['best_acc'] for r in config_results]
            final_accs = [r['final_acc'] for r in config_results]
            results[config_name] = {
                'best_mean': np.mean(best_accs),
                'best_std': np.std(best_accs),
                'final_mean': np.mean(final_accs),
                'final_std': np.std(final_accs),
            }
            
            # 保存配置汇总
            summary_file = save_dir / f"Centralized_{config_name}_summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Configuration: {config_name}\n")
                f.write(f"Parameters: {config}\n")
                f.write(f"Best Accuracy: {np.mean(best_accs):.4f} +/- {np.std(best_accs):.4f}\n")
                f.write(f"Final Accuracy: {np.mean(final_accs):.4f} +/- {np.std(final_accs):.4f}\n")
    
    # 找出最佳配置并保存推荐
    if results:
        best_config = max(results.keys(), key=lambda k: results[k]['best_mean'])
        
        recommendation_file = save_dir / f"{dataset_name}_BEST_CONFIG_RECOMMENDATION.txt"
        with open(recommendation_file, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write(f"中心化训练最佳配置推荐 - {dataset_name}\n")
            f.write("="*100 + "\n\n")
            f.write(f"【推荐配置】\n")
            f.write(f"  配置名称: {best_config}\n")
            f.write(f"  最佳准确率: {results[best_config]['best_mean']:.4f} ± {results[best_config]['best_std']:.4f}\n")
            f.write(f"  最终准确率: {results[best_config]['final_mean']:.4f} ± {results[best_config]['final_std']:.4f}\n\n")
            f.write(f"【配置参数】\n")
            for k, v in CONFIGS[best_config].items():
                f.write(f"  {k}: {v}\n")
            f.write(f"\n【性能排名前5】\n")
            sorted_configs = sorted(results.keys(), key=lambda k: results[k]['best_mean'], reverse=True)
            f.write(f"{'排名':<6} {'配置名称':<20} {'最佳准确率':<30} {'最终准确率':<30}\n")
            f.write("-"*100 + "\n")
            for i, cfg in enumerate(sorted_configs[:5], 1):
                f.write(f"{i:<6} {cfg:<20} {results[cfg]['best_mean']:.4f} ± {results[cfg]['best_std']:.4f}   ")
                f.write(f"{results[cfg]['final_mean']:.4f} ± {results[cfg]['final_std']:.4f}\n")
            f.write("\n" + "="*100 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*100 + "\n")
        
        print(f"\n[BEST] {best_config}: {results[best_config]['best_mean']:.4f} ± {results[best_config]['best_std']:.4f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='中心化训练')
    parser.add_argument('--dataset', type=str, choices=['Uci', 'Xinwang', 'all'], 
                       default='all', help='要训练的数据集')
    args = parser.parse_args()
    
    print("="*80)
    print("中心化训练 - 联邦学习性能上界参考")
    print("="*80)
    print(f"设备: {DEVICE}")
    print(f"配置数量: {len(CONFIGS)}")
    print(f"重复次数: {NUM_REPEATS}")
    print(f"训练轮数: {EPOCHS}")
    
    if args.dataset == 'all':
        datasets = DATASETS
    else:
        datasets = [args.dataset]
    
    for dataset in datasets:
        run_centralized_experiments(dataset)
    
    print("\n" + "="*80)
    print("中心化训练完成!")
    print("="*80)


if __name__ == '__main__':
    main()
