"""
Modern academic-style plotting utilities for federated learning experiments

Features:
- Consistent color schemes across all plots
- Clean, publication-ready visualizations
- Automatic result analysis and visualization
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Modern academic style configuration
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.8

# Consistent color palette (matching analyze_results.py)
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#F18F01',    # Orange
    'tertiary': '#A23B72',     # Purple
    'quaternary': '#6A994E',   # Green
    'quinary': '#D32F2F',      # Red
    'success': '#06A77D',      # Teal
    'danger': '#C73E1D',       # Dark Red
}


def plot_training_results(result_file, result_subdir=None, show_plot=False, output_dir='figures'):
    """
    Generate publication-quality training result visualizations
    
    Args:
        result_file: Result filename, e.g., 'Uci_FedGpro_iid_0.h5'
        result_subdir: Result subdirectory, e.g., 'Uci_FedGpro_iid'
        show_plot: Whether to display the plot (recommended False for servers)
        output_dir: Output directory name ('figures' or 'results'), default: 'figures'
    
    Returns:
        output_path: Path to the saved figure
    
    Example:
        >>> plot_training_results('Uci_FedGpro_iid_0.h5', 'Uci_FedGpro_iid')
        # Saves to: figures/Uci_FedGpro_iid_0_plot.png
    """
    # Construct result path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    system_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(system_dir)
    results_base = os.path.join(system_dir, "results")
    
    if result_subdir:
        result_path = os.path.join(results_base, result_subdir, result_file)
    else:
        result_path = os.path.join(results_base, result_file)
    
    if not os.path.exists(result_path):
        print(f"[WARNING] Result file not found: {result_path}")
        return None
    
    try:
        # Load data
        with h5py.File(result_path, 'r') as f:
            test_acc = np.array(f['rs_test_acc'])
            test_auc = np.array(f.get('rs_test_auc', []))
            train_loss = np.array(f['rs_train_loss'])
        
        # Validate data
        if len(test_acc) == 0:
            print(f"[WARNING] Empty data: {result_path}")
            return None
        
        has_auc = len(test_auc) > 0
        rounds = np.arange(1, len(test_acc) + 1)
        
        # Create figure with modern style
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        fig.patch.set_facecolor('white')
        
        # ========== Plot 1: Test Accuracy ==========
        ax1 = axes[0]
        ax1.plot(rounds, test_acc, linewidth=2.5, color=COLORS['primary'], 
                alpha=0.9, label='Test Accuracy')
        
        # Highlight max accuracy
        max_idx = test_acc.argmax()
        ax1.scatter([rounds[max_idx]], [test_acc[max_idx]], 
                   color=COLORS['danger'], s=100, zorder=5, 
                   label=f'Peak: {test_acc.max():.4f} (R{rounds[max_idx]})')
        
        # Add confidence band (simple smoothing)
        if len(test_acc) > 10:
            window = min(5, len(test_acc) // 10)
            smoothed = np.convolve(test_acc, np.ones(window)/window, mode='same')
            ax1.fill_between(rounds, test_acc, smoothed, alpha=0.1, color=COLORS['primary'])
        
        ax1.set_xlabel('Communication Rounds', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
        ax1.set_title('Test Accuracy Convergence', fontsize=14, fontweight='bold', pad=15)
        ax1.legend(loc='lower right', fontsize=10, framealpha=0.9, edgecolor='#CCCCCC')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(left=1)
        
        # ========== Plot 2: Training Loss ==========
        ax2 = axes[1]
        ax2.plot(rounds, train_loss, linewidth=2.5, color=COLORS['secondary'], 
                alpha=0.9, label='Training Loss')
        
        # Highlight min loss
        min_idx = train_loss.argmin()
        ax2.scatter([rounds[min_idx]], [train_loss[min_idx]], 
                   color=COLORS['success'], s=100, zorder=5,
                   label=f'Min: {train_loss.min():.4f} (R{rounds[min_idx]})')
        
        ax2.set_xlabel('Communication Rounds', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Training Loss', fontsize=13, fontweight='bold')
        ax2.set_title('Training Loss Convergence', fontsize=14, fontweight='bold', pad=15)
        ax2.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='#CCCCCC')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim(left=1)
        
        # ========== Plot 3: Performance Metrics ==========
        ax3 = axes[2]
        
        # Calculate improvement rate (smoothed derivative)
        acc_improvement = np.diff(test_acc, prepend=test_acc[0])
        
        # Plot improvement
        ax3.plot(rounds, acc_improvement, linewidth=2, color=COLORS['tertiary'], 
                alpha=0.8, label='Accuracy Δ')
        ax3.axhline(y=0, color='#666666', linestyle='-', alpha=0.4, linewidth=1)
        
        # Highlight convergence regions
        ax3.fill_between(rounds, 0, acc_improvement, 
                        where=(acc_improvement > 0), 
                        alpha=0.2, color=COLORS['success'], 
                        label='Improving')
        ax3.fill_between(rounds, 0, acc_improvement, 
                        where=(acc_improvement <= 0), 
                        alpha=0.2, color=COLORS['danger'], 
                        label='Declining')
        
        ax3.set_xlabel('Communication Rounds', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Accuracy Change (Δ)', fontsize=13, fontweight='bold')
        ax3.set_title('Learning Dynamics', fontsize=14, fontweight='bold', pad=15)
        ax3.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='#CCCCCC')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_xlim(left=1)
        
        plt.tight_layout()
        
        # Save figure to specified directory
        base_name = os.path.splitext(result_file)[0]
        
        if output_dir == 'figures':
            # Save to project figures directory
            figures_dir = os.path.join(project_root, 'figures')
            os.makedirs(figures_dir, exist_ok=True)
            output_path = os.path.join(figures_dir, f'{base_name}_plot.png')
            output_pdf = os.path.join(figures_dir, f'{base_name}_plot.pdf')
        else:
            # Save to results subdirectory (original behavior)
            if result_subdir:
                output_path = os.path.join(results_base, result_subdir, f'{base_name}_plot.png')
            else:
                output_path = os.path.join(results_base, f'{base_name}_plot.png')
            output_pdf = None
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        if output_pdf:
            plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
            print(f"[SUCCESS] Training visualization saved: {output_path} & {output_pdf}")
        else:
            print(f"[SUCCESS] Training visualization saved: {output_path}")
        
        # Print statistics
        print("\n" + "="*70)
        print("Training Results Summary")
        print("="*70)
        print(f"  Peak Accuracy:   {test_acc.max():.4f} @ Round {test_acc.argmax() + 1}")
        if has_auc:
            print(f"  Peak AUC:        {test_auc.max():.4f} @ Round {test_auc.argmax() + 1}")
        print(f"  Min Loss:        {train_loss.min():.4f} @ Round {train_loss.argmin() + 1}")
        print(f"  Final Accuracy:  {test_acc[-1]:.4f}")
        if has_auc:
            print(f"  Final AUC:       {test_auc[-1]:.4f}")
        print(f"  Final Loss:      {train_loss[-1]:.4f}")
        print(f"  Total Rounds:    {len(test_acc)}")
        print("="*70)
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return output_path
        
    except Exception as e:
        print(f"[ERROR] Plotting failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_algorithms(result_files, algorithms, output_name='algorithm_comparison', output_dir='figures'):
    """
    Generate publication-quality algorithm comparison visualizations
    
    Args:
        result_files: List of result file paths (full paths to .h5 files)
        algorithms: List of algorithm names
        output_name: Output filename (without extension), default: 'algorithm_comparison'
        output_dir: Output directory ('figures' or specify custom path)
    
    Returns:
        output_path: Path to the saved comparison figure
    
    Example:
        >>> files = [
        ...     'system/results/Uci_FedAvg_iid/Uci_FedAvg_iid_0.h5',
        ...     'system/results/Uci_FedProx_iid/Uci_FedProx_iid_0.h5',
        ...     'system/results/Uci_FedGpro_iid/Uci_FedGpro_iid_0.h5'
        ... ]
        >>> compare_algorithms(files, ['FedAvg', 'FedProx', 'FedGpro'])
        # Saves to: figures/algorithm_comparison.png
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor('white')
    
    # Consistent color palette
    color_palette = [
        COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], 
        COLORS['quaternary'], COLORS['quinary'], COLORS['success'], 
        COLORS['danger']
    ]
    
    # Line styles for better distinction
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
    
    for i, (file, algo) in enumerate(zip(result_files, algorithms)):
        if not os.path.exists(file):
            print(f"[WARNING] File not found: {file}")
            continue
            
        try:
            with h5py.File(file, 'r') as f:
                test_acc = np.array(f['rs_test_acc'])
                test_auc = np.array(f.get('rs_test_auc', []))
                train_loss = np.array(f['rs_train_loss'])
            
            rounds = np.arange(1, len(test_acc) + 1)
            color = color_palette[i % len(color_palette)]
            linestyle = line_styles[i % len(line_styles)]
            
            # Accuracy comparison
            axes[0].plot(rounds, test_acc, linewidth=2.5, color=color, 
                        linestyle=linestyle, label=algo, alpha=0.85)
            
            # AUC comparison (if available)
            if len(test_auc) > 0:
                axes[1].plot(rounds, test_auc, linewidth=2.5, color=color, 
                           linestyle=linestyle, label=algo, alpha=0.85)
            
            # Loss comparison
            axes[2].plot(rounds, train_loss, linewidth=2.5, color=color, 
                        linestyle=linestyle, label=algo, alpha=0.85)
            
        except Exception as e:
            print(f"[WARNING] Failed to read {file}: {e}")
            continue
    
    # ========== Configure Accuracy Plot ==========
    axes[0].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
    axes[0].set_xlabel('Communication Rounds', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=10, framealpha=0.9, edgecolor='#CCCCCC')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_xlim(left=1)
    
    # ========== Configure AUC Plot ==========
    axes[1].set_title('Test AUC Comparison', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xlabel('Communication Rounds', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Test AUC', fontsize=13, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=10, framealpha=0.9, edgecolor='#CCCCCC')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_xlim(left=1)
    
    # ========== Configure Loss Plot ==========
    axes[2].set_title('Training Loss Comparison', fontsize=14, fontweight='bold', pad=15)
    axes[2].set_xlabel('Communication Rounds', fontsize=13, fontweight='bold')
    axes[2].set_ylabel('Training Loss', fontsize=13, fontweight='bold')
    axes[2].legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='#CCCCCC')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].set_xlim(left=1)
    
    plt.tight_layout()
    
    # Determine output path
    if output_dir == 'figures':
        current_dir = os.path.dirname(os.path.abspath(__file__))
        system_dir = os.path.dirname(current_dir)
        project_root = os.path.dirname(system_dir)
        figures_dir = os.path.join(project_root, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        output_path = os.path.join(figures_dir, f'{output_name}.png')
        output_pdf = os.path.join(figures_dir, f'{output_name}.pdf')
    else:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{output_name}.png')
        output_pdf = os.path.join(output_dir, f'{output_name}.pdf')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    print(f"[SUCCESS] Algorithm comparison saved: {output_path} & {output_pdf}")
    plt.close()
    
    return output_path
