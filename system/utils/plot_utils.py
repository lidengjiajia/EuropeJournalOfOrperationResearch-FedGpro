"""
绘图工具函数
用于训练完成后自动生成结果可视化图表
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_training_results(result_file, result_subdir=None, show_plot=False):
    """
    自动绘制训练结果并保存到results目录
    
    Args:
        result_file: 结果文件名，如 'Uci_FedGWO_test_0.h5'
        result_subdir: 结果子目录，如 'Uci_FedGWO_feature'
        show_plot: 是否显示图表（服务器环境建议False）
    
    Returns:
        output_path: 保存的图片路径
    """
    # 获取正确的结果目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_base = os.path.join(os.path.dirname(current_dir), "results")
    
    # 构建完整路径（包含子目录）
    if result_subdir:
        result_path = os.path.join(results_base, result_subdir, result_file)
    else:
        # 兼容旧的调用方式
        result_path = os.path.join(results_base, result_file)
    
    if not os.path.exists(result_path):
        print(f"⚠️  结果文件不存在: {result_path}")
        return None
    
    try:
        # 读取数据
        with h5py.File(result_path, 'r') as f:
            test_acc = np.array(f['rs_test_acc'])
            test_auc = np.array(f.get('rs_test_auc', []))
            train_loss = np.array(f['rs_train_loss'])
        
        # 检查数据有效性
        if len(test_acc) == 0:
            print(f"⚠️  数据为空: {result_path}")
            return None
        
        has_auc = len(test_auc) > 0
        
        # 创建图表
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. 准确率曲线
        ax1 = axes[0]
        ax1.plot(test_acc, linewidth=2, color='#2E86AB')
        ax1.set_title('Test Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=test_acc.max(), color='r', linestyle='--', alpha=0.5, 
                    label=f'Max: {test_acc.max():.4f}')
        ax1.legend()
        
        # 2. 损失曲线
        ax2 = axes[1]
        ax2.plot(train_loss, linewidth=2, color='#F18F01')
        ax2.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Round', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=train_loss.min(), color='r', linestyle='--', alpha=0.5,
                    label=f'Min: {train_loss.min():.4f}')
        ax2.legend()
        
        # 3. 收敛分析（适用于GWO等优化算法）
        ax3 = axes[2]
        rounds = np.arange(len(test_acc))
        
        # 如果是FedGWO，绘制收敛因子a
        if 'GWO' in result_file or 'gwo' in result_file.lower():
            a_values = 2 - 2 * rounds / max(len(test_acc) - 1, 1)
            ax3.plot(a_values, linewidth=2, color='#6A994E')
            ax3.set_title('GWO Convergence Factor (a)', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Factor a', fontsize=12)
            ax3.axhline(y=1.0, color='orange', linestyle='--', alpha=0.5,
                        label='Exploration/Exploitation (a=1)')
            ax3.fill_between(rounds, 0, a_values, where=(a_values > 1), 
                             alpha=0.2, color='blue', label='Exploration')
            ax3.fill_between(rounds, 0, a_values, where=(a_values <= 1), 
                             alpha=0.2, color='green', label='Exploitation')
        else:
            # 其他算法绘制准确率提升曲线
            acc_improvement = np.diff(test_acc, prepend=test_acc[0])
            ax3.plot(acc_improvement, linewidth=2, color='#A23B72')
            ax3.set_title('Accuracy Improvement', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Δ Accuracy', fontsize=12)
            ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        ax3.set_xlabel('Round', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        
        # 生成输出文件名（保存在同一子目录下）
        base_name = os.path.splitext(result_file)[0]
        if result_subdir:
            output_path = os.path.join(results_base, result_subdir, f'{base_name}_plot.png')
        else:
            output_path = os.path.join(results_base, f'{base_name}_plot.png')
        
        # 保存图表
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ 训练结果可视化已保存: {output_path}")
        
        # 打印统计信息
        print("\n" + "="*70)
        print("📊 训练结果统计")
        print("="*70)
        print(f"  最大准确率: {test_acc.max():.4f} @ Round {test_acc.argmax()}")
        if has_auc:
            print(f"  最大AUC: {test_auc.max():.4f} @ Round {test_auc.argmax()}")
        print(f"  最小损失: {train_loss.min():.4f} @ Round {train_loss.argmin()}")
        print(f"  最终准确率: {test_acc[-1]:.4f}")
        if has_auc:
            print(f"  最终AUC: {test_auc[-1]:.4f}")
        print(f"  最终损失: {train_loss[-1]:.4f}")
        print(f"  总训练轮数: {len(test_acc)}")
        print("="*70)
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return output_path
        
    except Exception as e:
        print(f"❌ 绘图失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_algorithms(result_files, algorithms, output_path='results/algorithm_comparison.png'):
    """
    对比多个算法的性能
    
    Args:
        result_files: 结果文件列表
        algorithms: 算法名称列表
        output_path: 输出图片路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E', '#C73E1D']
    
    for i, (file, algo) in enumerate(zip(result_files, algorithms)):
        if not os.path.exists(file):
            print(f"⚠️  文件不存在: {file}")
            continue
            
        try:
            with h5py.File(file, 'r') as f:
                test_acc = np.array(f['rs_test_acc'])
                test_auc = np.array(f.get('rs_test_auc', []))
                train_loss = np.array(f['rs_train_loss'])
            
            color = colors[i % len(colors)]
            
            # 准确率对比
            axes[0].plot(test_acc, linewidth=2, color=color, label=algo)
            
            # AUC对比（如果有）
            if len(test_auc) > 0:
                axes[1].plot(test_auc, linewidth=2, color=color, label=algo)
            
            # 损失对比
            axes[2].plot(train_loss, linewidth=2, color=color, label=algo)
            
        except Exception as e:
            print(f"⚠️  读取失败 {file}: {e}")
            continue
    
    axes[0].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Round', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('Test AUC Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Round', fontsize=12)
    axes[1].set_ylabel('AUC', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Round', fontsize=12)
    axes[2].set_ylabel('Loss', fontsize=12)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 算法对比图已保存: {output_path}")
    plt.close()
    
    return output_path
