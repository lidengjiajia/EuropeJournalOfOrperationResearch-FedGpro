"""
中心化训练脚本（自动批量运行版本）
功能：
1. 聚合所有客户端的数据集
2. 使用聚合后的数据进行中心化训练
3. 自动运行多个数据集、多种参数组合、多次重复实验
4. 结果保存到results/{Dataset}_Centralized目录
5. 模型命名：Centralized_{参数配置}

使用方法：
python centralized_training.py                    # 运行所有实验
python centralized_training.py --quick_test       # 快速测试（每个1次）
python centralized_training.py --datasets Uci     # 只运行指定数据集
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime
import h5py

# 添加system目录到路径
BASE_DIR = Path(__file__).parent.resolve()
SYSTEM_DIR = BASE_DIR / 'system'
sys.path.insert(0, str(SYSTEM_DIR))

from flcore.trainmodel.credit import UciCreditNet, XinwangCreditNet
from utils.data_utils import read_data

# =============================================================================
# 配置
# =============================================================================

DATASETS = ['Uci', 'Xinwang']
RESULTS_DIR = BASE_DIR / 'system' / 'results'
MODELS_DIR = BASE_DIR / 'system' / 'models'

# 参数配置组合
PARAM_CONFIGS = {
    'Uci': {
        'default': {
            'name': 'Default',
            'lr': 0.01,
            'batch_size': 64,
            'weight_decay': 1e-4,
            'lr_decay_step': 20,
            'lr_decay_gamma': 0.9,
            'hidden_dims': [32, 16],
            'dropout': 0.4
        },
        'shallow': {
            'name': 'Shallow',
            'lr': 0.01,
            'batch_size': 64,
            'weight_decay': 1e-4,
            'lr_decay_step': 20,
            'lr_decay_gamma': 0.9,
            'hidden_dims': [32],
            'dropout': 0.4
        },
        'deep3': {
            'name': 'Deep3Layer',
            'lr': 0.01,
            'batch_size': 64,
            'weight_decay': 1e-4,
            'lr_decay_step': 20,
            'lr_decay_gamma': 0.9,
            'hidden_dims': [64, 32, 16],
            'dropout': 0.4
        },
        'deep4': {
            'name': 'Deep4Layer',
            'lr': 0.01,
            'batch_size': 64,
            'weight_decay': 1e-4,
            'lr_decay_step': 20,
            'lr_decay_gamma': 0.9,
            'hidden_dims': [64, 48, 32, 16],
            'dropout': 0.4
        },
        'wide': {
            'name': 'Wide',
            'lr': 0.01,
            'batch_size': 64,
            'weight_decay': 1e-4,
            'lr_decay_step': 20,
            'lr_decay_gamma': 0.9,
            'hidden_dims': [64, 32],
            'dropout': 0.4
        },
        'wide_deep': {
            'name': 'WideDeep',
            'lr': 0.01,
            'batch_size': 64,
            'weight_decay': 1e-4,
            'lr_decay_step': 20,
            'lr_decay_gamma': 0.9,
            'hidden_dims': [128, 64, 32],
            'dropout': 0.4
        },
        'high_lr': {
            'name': 'HighLR',
            'lr': 0.02,
            'batch_size': 64,
            'weight_decay': 1e-4,
            'lr_decay_step': 20,
            'lr_decay_gamma': 0.9,
            'hidden_dims': [32, 16],
            'dropout': 0.4
        },
        'low_lr': {
            'name': 'LowLR',
            'lr': 0.005,
            'batch_size': 64,
            'weight_decay': 1e-4,
            'lr_decay_step': 20,
            'lr_decay_gamma': 0.9,
            'hidden_dims': [32, 16],
            'dropout': 0.4
        },
        'large_batch': {
            'name': 'LargeBatch',
            'lr': 0.01,
            'batch_size': 128,
            'weight_decay': 1e-4,
            'lr_decay_step': 20,
            'lr_decay_gamma': 0.9,
            'hidden_dims': [32, 16],
            'dropout': 0.4
        },
    },
    'Xinwang': {
        'default': {
            'name': 'Default',
            'lr': 0.006,
            'batch_size': 64,
            'weight_decay': 1e-4,
            'lr_decay_step': 20,
            'lr_decay_gamma': 0.9,
            'hidden_dims': [128, 64, 32],
            'dropout': 0.3
        },
        'shallow': {
            'name': 'Shallow',
            'lr': 0.006,
            'batch_size': 64,
            'weight_decay': 1e-4,
            'lr_decay_step': 20,
            'lr_decay_gamma': 0.9,
            'hidden_dims': [64],
            'dropout': 0.3
        },
        'medium2': {
            'name': 'Medium2Layer',
            'lr': 0.006,
            'batch_size': 64,
            'weight_decay': 1e-4,
            'lr_decay_step': 20,
            'lr_decay_gamma': 0.9,
            'hidden_dims': [64, 32],
            'dropout': 0.3
        },
        'deep4': {
            'name': 'Deep4Layer',
            'lr': 0.006,
            'batch_size': 64,
            'weight_decay': 1e-4,
            'lr_decay_step': 20,
            'lr_decay_gamma': 0.9,
            'hidden_dims': [128, 96, 64, 32],
            'dropout': 0.3
        },
        'deep5': {
            'name': 'Deep5Layer',
            'lr': 0.006,
            'batch_size': 64,
            'weight_decay': 1e-4,
            'lr_decay_step': 20,
            'lr_decay_gamma': 0.9,
            'hidden_dims': [128, 96, 64, 48, 32],
            'dropout': 0.3
        },
        'wide': {
            'name': 'Wide',
            'lr': 0.006,
            'batch_size': 64,
            'weight_decay': 1e-4,
            'lr_decay_step': 20,
            'lr_decay_gamma': 0.9,
            'hidden_dims': [256, 128],
            'dropout': 0.3
        },
        'wide_deep': {
            'name': 'WideDeep',
            'lr': 0.006,
            'batch_size': 64,
            'weight_decay': 1e-4,
            'lr_decay_step': 20,
            'lr_decay_gamma': 0.9,
            'hidden_dims': [256, 128, 64],
            'dropout': 0.3
        },
        'high_lr': {
            'name': 'HighLR',
            'lr': 0.01,
            'batch_size': 64,
            'weight_decay': 1e-4,
            'lr_decay_step': 20,
            'lr_decay_gamma': 0.9,
            'hidden_dims': [128, 64, 32],
            'dropout': 0.3
        },
        'low_lr': {
            'name': 'LowLR',
            'lr': 0.003,
            'batch_size': 64,
            'weight_decay': 1e-4,
            'lr_decay_step': 20,
            'lr_decay_gamma': 0.9,
            'hidden_dims': [128, 64, 32],
            'dropout': 0.3
        },
        'large_batch': {
            'name': 'LargeBatch',
            'lr': 0.006,
            'batch_size': 128,
            'weight_decay': 1e-4,
            'lr_decay_step': 20,
            'lr_decay_gamma': 0.9,
            'hidden_dims': [128, 64, 32],
            'dropout': 0.3
        },
    }
}

# =============================================================================
# 数据加载和聚合
# =============================================================================

def load_all_client_data(dataset, num_clients):
    """加载并聚合所有客户端数据"""
    print(f"\n正在加载数据集: {dataset}")
    print(f"客户端数量: {num_clients}")
    
    # 聚合训练数据
    all_train_X = []
    all_train_y = []
    
    for client_id in range(num_clients):
        try:
            train_data = read_data(dataset, client_id, is_train=True)
            
            if 'x' in train_data and 'y' in train_data:
                X = train_data['x']
                y = train_data['y']
                
                all_train_X.append(X)
                all_train_y.append(y)
                
                print(f"  客户端 {client_id}: {len(X)} 样本")
        except Exception as e:
            print(f"  警告: 无法加载客户端 {client_id} 数据: {e}")
    
    # 合并所有数据
    train_X = np.vstack(all_train_X)
    train_y = np.concatenate(all_train_y)
    
    # 确保标签是一维的
    if len(train_y.shape) > 1:
        train_y = train_y.flatten()
    
    print(f"\n训练数据总计: {len(train_X)} 样本 (来自{num_clients}个客户端)")
    print(f"特征维度: {train_X.shape[1]}")
    
    # 加载并聚合所有客户端的测试数据
    all_test_X = []
    all_test_y = []
    
    try:
        for client_id in range(num_clients):
            test_data = read_data(dataset, client_id, is_train=False)
            if 'x' in test_data and 'y' in test_data:
                all_test_X.append(test_data['x'])
                all_test_y.append(test_data['y'])
        
        # 合并所有测试数据
        test_X = np.vstack(all_test_X)
        test_y = np.concatenate(all_test_y)
        
        # 确保标签是一维的
        if len(test_y.shape) > 1:
            test_y = test_y.flatten()
            
        print(f"测试数据总计: {len(test_X)} 样本 (来自{num_clients}个客户端的测试集)")
    except Exception as e:
        # 如果没有统一测试集，从训练集划分20%
        print(f"警告: 无法加载测试集 ({e})，从训练集划分20%作为测试集")
        n_test = int(len(train_X) * 0.2)
        indices = np.random.permutation(len(train_X))
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        test_X = train_X[test_indices]
        test_y = train_y[test_indices]
        train_X = train_X[train_indices]
        train_y = train_y[train_indices]
        
        print(f"训练集: {len(train_X)} 样本")
        print(f"测试集: {len(test_X)} 样本")
    
    # 转换为Tensor
    train_X = torch.FloatTensor(train_X)
    train_y = torch.LongTensor(train_y)
    test_X = torch.FloatTensor(test_X)
    test_y = torch.LongTensor(test_y)
    
    return train_X, train_y, test_X, test_y

# =============================================================================
# 训练和评估
# =============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    # 计算精确率、召回率、F1
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# =============================================================================
# 主训练流程
# =============================================================================

def train_centralized(dataset, param_config, times, epochs=100, gpu_id=0, num_classes=2, print_every=10):
    """中心化训练主流程
    
    Args:
        dataset: 数据集名称 ('Uci' 或 'Xinwang')
        param_config: 参数配置字典
        times: 实验重复编号 (0-4)
        epochs: 训练轮数
        gpu_id: GPU编号
        num_classes: 类别数
        print_every: 打印间隔
    """
    config_name = param_config['name']
    print("="*80)
    print(f"中心化训练 - {dataset} - {config_name} - 第{times+1}次")
    print("="*80)
    
    # 设置设备
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    
    # 获取客户端数量
    dataset_dir = BASE_DIR / 'dataset' / dataset
    config_path = dataset_dir / 'config.json'
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
            num_clients = config.get('num_clients', 10)
    else:
        num_clients = 10
        print(f"警告: 未找到config.json，使用默认客户端数量: {num_clients}")
    
    # 加载数据
    train_X, train_y, test_X, test_y = load_all_client_data(dataset, num_clients)
    
    # 创建数据加载器
    batch_size = param_config['batch_size']
    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    print(f"\nBatch size: {batch_size}")
    print(f"训练批次: {len(train_loader)}")
    print(f"测试批次: {len(test_loader)}")
    
    # 创建模型（根据数据集和参数配置选择）
    input_dim = train_X.shape[1]
    hidden_dims = param_config['hidden_dims']
    dropout = param_config['dropout']
    
    if dataset == 'Uci':
        model = UciCreditNet(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout
        ).to(device)
    elif dataset == 'Xinwang':
        model = XinwangCreditNet(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout
        ).to(device)
    else:
        # 默认使用Uci架构
        model = UciCreditNet(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout
        ).to(device)
    
    print(f"\n模型配置:")
    print(f"  数据集: {dataset}")
    print(f"  配置名称: {config_name}")
    print(f"  输入维度: {input_dim}")
    print(f"  隐藏层: {hidden_dims}")
    print(f"  Dropout: {dropout}")
    print(f"  输出类别: {num_classes}")
    print(f"  参数总数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器和损失函数
    lr = param_config['lr']
    weight_decay = param_config['weight_decay']
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=param_config['lr_decay_step'],
        gamma=param_config['lr_decay_gamma']
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # 训练记录
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    test_precisions = []
    test_recalls = []
    test_f1s = []
    
    best_acc = 0.0
    best_epoch = 0
    
    # 训练循环
    print(f"\n开始训练 (总轮数: {epochs})")
    print("-"*80)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 评估
        test_metrics = evaluate(model, test_loader, criterion, device)
        
        # 记录
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_metrics['loss'])
        test_accs.append(test_metrics['accuracy'])
        test_precisions.append(test_metrics['precision'])
        test_recalls.append(test_metrics['recall'])
        test_f1s.append(test_metrics['f1'])
        
        # 更新最佳模型
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            best_epoch = epoch
            
            # 保存最佳模型（按dataset和config_name分类）
            model_dir = MODELS_DIR / dataset
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"Centralized_{config_name}_server.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': best_acc,
                'config': param_config,
                'dataset': dataset
            }, model_path)
        
        # 学习率衰减
        scheduler.step()
        
        # 打印进度
        epoch_time = time.time() - epoch_start
        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Test  Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.4f}")
            print(f"  Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
            print(f"  Best Acc: {best_acc:.4f} (Epoch {best_epoch+1})")
    
    total_time = time.time() - start_time
    print("-"*80)
    print(f"训练完成! 总耗时: {total_time/60:.2f}分钟")
    print(f"最佳测试准确率: {best_acc:.4f} (Epoch {best_epoch+1})")
    
    # 保存结果到子目录结构: system/results/{dataset}_Centralized/
    result_dir = RESULTS_DIR / f"{dataset}_Centralized"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 文件命名: Centralized_{config_name}_{times}.h5
    result_file = result_dir / f"Centralized_{config_name}_{times}.h5"
    
    with h5py.File(result_file, 'w') as f:
        f.create_dataset('rs_train_loss', data=np.array(train_losses))
        f.create_dataset('rs_train_acc', data=np.array(train_accs))
        f.create_dataset('rs_test_loss', data=np.array(test_losses))
        f.create_dataset('rs_test_acc', data=np.array(test_accs))
        f.create_dataset('rs_test_precision', data=np.array(test_precisions))
        f.create_dataset('rs_test_recall', data=np.array(test_recalls))
        f.create_dataset('rs_test_f1', data=np.array(test_f1s))
        
        # 保存超参数
        f.attrs['dataset'] = dataset
        f.attrs['config_name'] = config_name
        f.attrs['epochs'] = epochs
        f.attrs['batch_size'] = batch_size
        f.attrs['lr'] = lr
        f.attrs['weight_decay'] = weight_decay
        f.attrs['hidden_dims'] = str(hidden_dims)
        f.attrs['dropout'] = dropout
        f.attrs['best_acc'] = best_acc
        f.attrs['best_epoch'] = best_epoch
        f.attrs['total_time'] = total_time
        f.attrs['times'] = times
    
    # 保存CSV文件
    csv_file = result_dir / f"Centralized_{config_name}_{times}_training_process.csv"
    import csv
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['round', 'test_accuracy', 'test_auc', 'test_precision', 'test_recall', 'test_f1', 'train_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(test_accs)):
            writer.writerow({
                'round': i + 1,
                'test_accuracy': test_accs[i],
                'test_auc': 0.0,  # 中心化训练暂未计算AUC
                'test_precision': test_precisions[i],
                'test_recall': test_recalls[i],
                'test_f1': test_f1s[i],
                'train_loss': train_losses[i]
            })
    
    print(f"\n结果已保存: {result_file}")
    print(f"CSV文件已保存: {csv_file}")
    print(f"模型已保存: {MODELS_DIR / dataset / f'Centralized_{config_name}_server.pt'}")
    
    # 输出最终统计
    print("\n"+"="*80)
    print("训练统计")
    print("="*80)
    print(f"数据集: {dataset}")
    print(f"配置: {config_name}")
    print(f"训练样本: {len(train_X)}")
    print(f"测试样本: {len(test_X)}")
    print(f"总轮数: {epochs}")
    print(f"批大小: {batch_size}")
    print(f"学习率: {lr}")
    print(f"最佳准确率: {best_acc:.4f} (Epoch {best_epoch+1})")
    print(f"最终测试指标:")
    print(f"  Accuracy:  {test_accs[-1]:.4f}")
    print(f"  Precision: {test_precisions[-1]:.4f}")
    print(f"  Recall:    {test_recalls[-1]:.4f}")
    print(f"  F1 Score:  {test_f1s[-1]:.4f}")
    print("="*80)
    
    return {
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'total_time': total_time,
        'final_metrics': {
            'accuracy': test_accs[-1],
            'precision': test_precisions[-1],
            'recall': test_recalls[-1],
            'f1': test_f1s[-1]
        }
    }

# =============================================================================
# 批量实验和结果汇总
# =============================================================================

def generate_summary(dataset, config_name, num_times=5):
    """
    生成单个配置的汇总txt文件
    
    Args:
        dataset: 数据集名称
        config_name: 配置名称
        num_times: 重复实验次数
    """
    result_dir = RESULTS_DIR / f"{dataset}_Centralized"
    
    if not result_dir.exists():
        print(f"警告: 结果目录不存在: {result_dir}")
        return
    
    # 收集所有重复实验的结果
    all_metrics = {
        'test_acc': [],
        'test_precision': [],
        'test_recall': [],
        'test_f1': [],
        'train_loss': [],
        'best_acc': [],
        'best_epoch': []
    }
    
    for times in range(num_times):
        result_file = result_dir / f"Centralized_{config_name}_{times}.h5"
        if not result_file.exists():
            print(f"警告: 结果文件不存在: {result_file}")
            continue
        
        try:
            with h5py.File(result_file, 'r') as f:
                # 取最后一个epoch的指标
                all_metrics['test_acc'].append(f['rs_test_acc'][-1])
                all_metrics['test_precision'].append(f['rs_test_precision'][-1])
                all_metrics['test_recall'].append(f['rs_test_recall'][-1])
                all_metrics['test_f1'].append(f['rs_test_f1'][-1])
                all_metrics['train_loss'].append(f['rs_train_loss'][-1])
                all_metrics['best_acc'].append(f.attrs['best_acc'])
                all_metrics['best_epoch'].append(f.attrs['best_epoch'])
        except Exception as e:
            print(f"警告: 读取文件失败 {result_file}: {e}")
            continue
    
    if not all_metrics['test_acc']:
        print(f"错误: 没有找到任何有效的结果文件")
        return
    
    # 计算平均值和标准差
    summary_file = result_dir / f"Centralized_{config_name}_summary.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"中心化训练结果汇总 - {dataset} - {config_name}\n")
        f.write("="*80 + "\n\n")
        f.write(f"重复次数: {len(all_metrics['test_acc'])}/{num_times}\n\n")
        f.write("-"*80 + "\n")
        f.write("指标统计 (平均值 ± 标准差)\n")
        f.write("-"*80 + "\n\n")
        
        # 计算并写入每个指标
        metrics_display = [
            ('Best Test Accuracy', 'best_acc', 4),
            ('Final Test Accuracy', 'test_acc', 4),
            ('Final Test Precision', 'test_precision', 4),
            ('Final Test Recall', 'test_recall', 4),
            ('Final Test F1 Score', 'test_f1', 4),
            ('Final Train Loss', 'train_loss', 6),
            ('Best Epoch', 'best_epoch', 1)
        ]
        
        for display_name, metric_key, decimals in metrics_display:
            values = np.array(all_metrics[metric_key])
            mean_val = np.mean(values)
            std_val = np.std(values)
            if metric_key == 'best_epoch':
                f.write(f"{display_name:25s}: {mean_val:.{decimals}f} ± {std_val:.{decimals}f}\n")
            else:
                f.write(f"{display_name:25s}: {mean_val:.{decimals}f} ± {std_val:.{decimals}f}\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("各次实验详细结果\n")
        f.write("-"*80 + "\n\n")
        
        # 写入每次实验的详细结果
        for i in range(len(all_metrics['test_acc'])):
            f.write(f"实验 {i}:\n")
            for display_name, metric_key, decimals in metrics_display:
                if metric_key == 'best_epoch':
                    f.write(f"  {display_name:25s}: {int(all_metrics[metric_key][i])}\n")
                else:
                    f.write(f"  {display_name:25s}: {all_metrics[metric_key][i]:.{decimals}f}\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"\n汇总文件已保存: {summary_file}")
    
    # 返回汇总结果
    return {
        'config_name': config_name,
        'best_acc_mean': np.mean(all_metrics['best_acc']),
        'best_acc_std': np.std(all_metrics['best_acc']),
        'final_acc_mean': np.mean(all_metrics['test_acc']),
        'final_acc_std': np.std(all_metrics['test_acc'])
    }

def generate_dataset_summary(dataset, num_times=5):
    """生成整个数据集所有配置的汇总对比"""
    result_dir = RESULTS_DIR / f"{dataset}_Centralized"
    
    if not result_dir.exists():
        print(f"警告: 结果目录不存在: {result_dir}")
        return
    
    # 收集所有配置的汇总结果
    all_configs_summary = []
    for config_key in PARAM_CONFIGS[dataset].keys():
        config_name = PARAM_CONFIGS[dataset][config_key]['name']
        summary = generate_summary(dataset, config_name, num_times)
        if summary:
            all_configs_summary.append(summary)
    
    if not all_configs_summary:
        print("警告: 没有找到任何配置的汇总结果")
        return
    
    # 生成总汇总文件
    summary_file = result_dir / f"{dataset}_Centralized_ALL_CONFIGS_SUMMARY.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write(f"中心化训练全配置对比汇总 - {dataset}\n")
        f.write("="*100 + "\n\n")
        f.write(f"重复次数: {num_times}\n")
        f.write(f"配置数量: {len(all_configs_summary)}\n\n")
        f.write("-"*100 + "\n")
        f.write(f"{'配置名称':<20s} | {'最佳准确率(平均)':<20s} | {'最终准确率(平均)':<20s}\n")
        f.write("-"*100 + "\n")
        
        # 按最佳准确率排序
        all_configs_summary.sort(key=lambda x: x['best_acc_mean'], reverse=True)
        
        for summary in all_configs_summary:
            f.write(f"{summary['config_name']:<20s} | "
                   f"{summary['best_acc_mean']:.4f} ± {summary['best_acc_std']:.4f}   | "
                   f"{summary['final_acc_mean']:.4f} ± {summary['final_acc_std']:.4f}\n")
        
        f.write("="*100 + "\n")
        f.write(f"\n最佳配置: {all_configs_summary[0]['config_name']} "
               f"(最佳准确率: {all_configs_summary[0]['best_acc_mean']:.4f})\n")
        f.write("="*100 + "\n")
    
    print(f"\n总汇总文件已保存: {summary_file}")
    
    # 生成"最佳配置推荐"文档
    best_config = all_configs_summary[0]
    best_config_name = best_config['config_name']
    
    # 找到对应的参数配置
    best_param_config = None
    for config_key, param_config in PARAM_CONFIGS[dataset].items():
        if param_config['name'] == best_config_name:
            best_param_config = param_config
            break
    
    # 创建最佳配置推荐文档
    recommendation_file = result_dir / f"{dataset}_BEST_CONFIG_RECOMMENDATION.txt"
    
    with open(recommendation_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write(f"中心化训练最佳配置推荐 - {dataset}\n")
        f.write("="*100 + "\n\n")
        
        f.write("【推荐配置】\n")
        f.write(f"  配置名称: {best_config_name}\n")
        f.write(f"  最佳准确率: {best_config['best_acc_mean']:.4f} ± {best_config['best_acc_std']:.4f}\n")
        f.write(f"  最终准确率: {best_config['final_acc_mean']:.4f} ± {best_config['final_acc_std']:.4f}\n\n")
        
        if best_param_config:
            f.write("【配置参数】\n")
            f.write(f"  学习率: {best_param_config['lr']}\n")
            f.write(f"  批大小: {best_param_config['batch_size']}\n")
            f.write(f"  权重衰减: {best_param_config['weight_decay']}\n")
            f.write(f"  隐藏层结构: {best_param_config['hidden_dims']}\n")
            f.write(f"  Dropout: {best_param_config['dropout']}\n")
            f.write(f"  学习率衰减步长: {best_param_config['lr_decay_step']}\n")
            f.write(f"  学习率衰减系数: {best_param_config['lr_decay_gamma']}\n\n")
        
        f.write("【性能排名前5】\n")
        f.write(f"{'排名':<6s} {'配置名称':<20s} {'最佳准确率':<25s} {'最终准确率':<25s}\n")
        f.write("-"*100 + "\n")
        for i, summary in enumerate(all_configs_summary[:5], 1):
            f.write(f"{i:<6d} {summary['config_name']:<20s} "
                   f"{summary['best_acc_mean']:.4f} ± {summary['best_acc_std']:.4f}   "
                   f"{summary['final_acc_mean']:.4f} ± {summary['final_acc_std']:.4f}\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("【使用建议】\n\n")
        
        # 根据配置名称给出建议
        if 'Shallow' in best_config_name:
            f.write("最佳配置为浅层网络，建议：\n")
            f.write("  - 数据量相对较小，浅层网络避免了过拟合\n")
            f.write("  - 适合联邦学习场景，参数少、通信开销小\n")
            f.write("  - 训练速度快，易于收敛\n")
        elif 'Deep' in best_config_name:
            f.write("最佳配置为深层网络，建议：\n")
            f.write("  - 数据量充足，深层网络能学习更复杂的特征\n")
            f.write("  - 适合中心化训练或数据量大的联邦学习\n")
            f.write("  - 需要更多的训练时间和计算资源\n")
        elif 'Wide' in best_config_name:
            f.write("最佳配置为宽层网络，建议：\n")
            f.write("  - 每层神经元数多，特征表达能力强\n")
            f.write("  - 适合特征维度较高的数据\n")
            f.write("  - 参数量大，需要注意过拟合\n")
        elif 'HighLR' in best_config_name:
            f.write("最佳配置使用较高学习率，建议：\n")
            f.write("  - 模型收敛速度快\n")
            f.write("  - 适合初期快速探索\n")
            f.write("  - 注意监控训练稳定性\n")
        elif 'LowLR' in best_config_name:
            f.write("最佳配置使用较低学习率，建议：\n")
            f.write("  - 训练更稳定，收敛更平滑\n")
            f.write("  - 适合精调和后期优化\n")
            f.write("  - 可能需要更多训练轮数\n")
        elif 'LargeBatch' in best_config_name:
            f.write("最佳配置使用大批量，建议：\n")
            f.write("  - 训练更稳定，梯度估计更准确\n")
            f.write("  - 可以充分利用GPU并行能力\n")
            f.write("  - 适合数据量大的场景\n")
        else:  # Default
            f.write("最佳配置为默认配置，建议：\n")
            f.write("  - 配置平衡，适应性强\n")
            f.write("  - 适合作为基线配置\n")
            f.write("  - 可以在此基础上进行微调\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*100 + "\n")
    
    print(f"\n★ 最佳配置推荐文档已保存: {recommendation_file}")
    
    # 打印到终端
    print("\n" + "="*100)
    print(f"中心化训练全配置对比汇总 - {dataset}")
    print("="*100)
    print(f"{'配置名称':<20s} | {'最佳准确率(平均)':<20s} | {'最终准确率(平均)':<20s}")
    print("-"*100)
    for summary in all_configs_summary:
        print(f"{summary['config_name']:<20s} | "
              f"{summary['best_acc_mean']:.4f} ± {summary['best_acc_std']:.4f}   | "
              f"{summary['final_acc_mean']:.4f} ± {summary['final_acc_std']:.4f}")
    print("="*100)
    print(f"★ 最佳配置: {best_config_name} (最佳准确率: {best_config['best_acc_mean']:.4f})")
    print("="*100)

def run_all_experiments(datasets=None, num_times=5, epochs=100, gpu_id=0):
    """运行所有实验
    
    Args:
        datasets: 数据集列表，None表示运行所有数据集
        num_times: 每个配置重复运行次数
        epochs: 训练轮数
        gpu_id: GPU编号
    """
    if datasets is None:
        datasets = DATASETS
    elif isinstance(datasets, str):
        datasets = [datasets]
    
    print("\n" + "="*100)
    print("中心化训练批量实验")
    print("="*100)
    print(f"数据集: {datasets}")
    print(f"每个配置重复次数: {num_times}")
    print(f"训练轮数: {epochs}")
    print(f"GPU: {gpu_id}")
    print("="*100 + "\n")
    
    total_experiments = sum(len(PARAM_CONFIGS[ds]) for ds in datasets) * num_times
    completed = 0
    
    start_time = time.time()
    
    for dataset in datasets:
        print(f"\n{'#'*100}")
        print(f"# 数据集: {dataset}")
        print(f"{'#'*100}\n")
        
        configs = PARAM_CONFIGS[dataset]
        
        for config_key, param_config in configs.items():
            config_name = param_config['name']
            
            print(f"\n{'-'*100}")
            print(f"配置: {config_name} ({config_key})")
            print(f"{'-'*100}\n")
            
            for times in range(num_times):
                print(f"\n{'='*80}")
                print(f"实验进度: {completed+1}/{total_experiments} - {dataset}/{config_name}/第{times+1}次")
                print(f"{'='*80}")
                
                try:
                    result = train_centralized(
                        dataset=dataset,
                        param_config=param_config,
                        times=times,
                        epochs=epochs,
                        gpu_id=gpu_id,
                        num_classes=2,
                        print_every=10
                    )
                    
                    print(f"\n✓ 完成: {dataset}/{config_name}/第{times+1}次")
                    print(f"  最佳准确率: {result['best_acc']:.4f}")
                    print(f"  最佳轮数: {result['best_epoch']+1}")
                    print(f"  耗时: {result['total_time']/60:.2f}分钟")
                    
                except Exception as e:
                    print(f"\n✗ 失败: {dataset}/{config_name}/第{times+1}次")
                    print(f"  错误: {e}")
                    import traceback
                    traceback.print_exc()
                
                completed += 1
                elapsed = time.time() - start_time
                avg_time = elapsed / completed
                remaining = (total_experiments - completed) * avg_time
                
                print(f"\n总进度: {completed}/{total_experiments} ({completed/total_experiments*100:.1f}%)")
                print(f"已用时: {elapsed/3600:.2f}小时")
                print(f"预计剩余: {remaining/3600:.2f}小时")
        
        # 生成该数据集的汇总
        print(f"\n{'='*100}")
        print(f"生成 {dataset} 汇总文件...")
        print(f"{'='*100}")
        generate_dataset_summary(dataset, num_times)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*100)
    print("所有实验完成!")
    print("="*100)
    print(f"总实验数: {total_experiments}")
    print(f"总耗时: {total_time/3600:.2f}小时")
    print(f"平均每个实验: {total_time/total_experiments/60:.2f}分钟")
    print("="*100)

# =============================================================================
# 命令行参数
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='中心化训练批量实验')
    
    # 数据集配置
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        choices=DATASETS + [None],
                        help='数据集名称列表 (默认: 所有数据集)')
    
    # 实验配置
    parser.add_argument('--num_times', type=int, default=5,
                        help='每个配置重复运行次数 (默认: 5)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数 (默认: 100)')
    
    # GPU配置
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID (默认: 0)')
    
    # 快速测试
    parser.add_argument('--quick_test', action='store_true',
                        help='快速测试模式 (每个配置只运行1次，10轮)')
    
    # 生成汇总
    parser.add_argument('--summary_only', action='store_true',
                        help='只生成汇总文件，不运行训练')
    
    return parser.parse_args()

# =============================================================================
# 主函数
# =============================================================================

def main():
    args = parse_args()
    
    # 打印配置
    print("\n" + "="*100)
    print("中心化训练批量实验配置")
    print("="*100)
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("="*100)
    
    # 只生成汇总
    if args.summary_only:
        datasets = args.datasets if args.datasets else DATASETS
        for dataset in datasets:
            print(f"\n生成 {dataset} 汇总...")
            generate_dataset_summary(dataset, args.num_times)
        return
    
    # 快速测试模式
    if args.quick_test:
        print("\n⚠️  快速测试模式: 每个配置只运行1次，10轮训练")
        num_times = 1
        epochs = 10
    else:
        num_times = args.num_times
        epochs = args.epochs
    
    # 运行所有实验
    run_all_experiments(
        datasets=args.datasets,
        num_times=num_times,
        epochs=epochs,
        gpu_id=args.gpu_id
    )

if __name__ == '__main__':
    main()
