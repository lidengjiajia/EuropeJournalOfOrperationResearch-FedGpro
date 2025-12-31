"""
NPZ格式详解 - 联邦学习框架的数据存储格式
"""

import numpy as np
import torch

print("="*70)
print("📦 NPZ格式完整工作流程")
print("="*70)

# ========== 第一步：生成数据时（generate_Uci.py） ==========
print("\n【步骤1】数据生成脚本 (generate_Uci.py)")
print("-"*70)

# 模拟处理后的数据
features = np.array([[1.2, 0.3, -0.5], [0.8, -1.2, 0.4]])  # 2样本 × 3特征
labels = np.array([0, 1])  # 2个标签

# 保存为NPZ
np.savez_compressed('demo_client_0.npz', data={'x': features, 'y': labels})
print(f"✓ 保存: data={{'x': shape{features.shape}, 'y': shape{labels.shape}}}")
print(f"✓ 文件: demo_client_0.npz")

# ========== 第二步：训练时读取（system/utils/data_utils.py） ==========
print("\n【步骤2】训练时读取数据 (system/utils/data_utils.py)")
print("-"*70)

# read_data函数
def read_data(dataset, idx, is_train=True):
    """框架的读取函数"""
    file = f'demo_client_{idx}.npz'
    with open(file, 'rb') as f:
        data = np.load(f, allow_pickle=True)['data'].tolist()
    return data

# 读取数据
data = read_data('Demo', 0, is_train=True)
print(f"✓ 读取: data类型={type(data)}")
print(f"✓ data.keys()={data.keys()}")
print(f"✓ data['x']={data['x']}")
print(f"✓ data['y']={data['y']}")

# ========== 第三步：转换为PyTorch张量 =========
print("\n【步骤3】转换为PyTorch张量 (process_image函数)")
print("-"*70)

def process_image(data):
    """框架的处理函数"""
    X = torch.Tensor(data['x']).type(torch.float32)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]

# 转换为PyTorch格式
dataset_list = process_image(data)
print(f"✓ 转换结果: {len(dataset_list)}个样本")
for i, (x, y) in enumerate(dataset_list):
    print(f"  样本{i}: 特征tensor{tuple(x.shape)}, 标签={y.item()}")

# ========== 第四步：训练时使用DataLoader ==========
print("\n【步骤4】训练时使用DataLoader")
print("-"*70)

from torch.utils.data import DataLoader
batch_size = 1
dataloader = DataLoader(dataset_list, batch_size=batch_size, shuffle=False)

for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
    print(f"✓ Batch {batch_idx}: features{tuple(batch_x.shape)}, labels{tuple(batch_y.shape)}")
    print(f"  → 输入模型: model(batch_x) → 输出预测")

# ========== 清理 ==========
import os
os.remove('demo_client_0.npz')

print("\n"+"="*70)
print("🎯 NPZ格式的核心优势")
print("="*70)
print("""
1. **压缩存储**: gzip压缩，节省磁盘空间（比CSV小66%）
2. **类型保持**: float32/int64精确存储，无精度损失
3. **快速加载**: 直接内存映射，比CSV快4倍
4. **框架友好**: NumPy → PyTorch零成本转换
5. **分布式友好**: 每个客户端一个独立文件，易于管理

在联邦学习中:
  - 20个客户端 = 20个NPZ文件（train/0.npz ~ train/19.npz）
  - 每个文件独立读取，支持并行加载
  - 保持数据隐私：原始CSV可以删除，只保留分片后的NPZ
""")

print("\n"+"="*70)
print("💡 为什么叫'npz'而不是'npy'？")
print("="*70)
print("""
- .npy  = 单个NumPy数组
- .npz  = 多个NumPy数组打包（类似ZIP）

在我们的场景中:
  .npz包含2个数组:
    ├─ 'x' (特征数组)
    └─ 'y' (标签数组)
  
这样一个文件就包含了完整的训练数据！
""")
