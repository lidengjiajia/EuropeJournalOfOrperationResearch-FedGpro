# FedGpro: 面向信用评分的联邦学习框架

## 项目简介

FedGpro是一个面向信用评分场景的隐私保护联邦学习框架，结合VAE生成数据增强、原型学习和自适应差分隐私技术，在保护用户隐私的同时提升模型性能。

## 目录结构

```
FedGpro/
├── run_baseline_experiments.py    # 基线实验运行器
├── run_ablation_experiments.py    # 消融实验运行器
├── dataset/                       # 数据集生成和处理
│   ├── generate_all_datasets_auto.py  # 一键生成所有数据集
│   ├── generate_Uci.py            # UCI数据集生成
│   ├── generate_Xinwang.py        # Xinwang数据集生成
│   └── utils/                     # 数据处理工具
├── system/                        # 核心系统代码
│   ├── main.py                    # 主入口
│   ├── flcore/                    # 联邦学习核心模块
│   │   ├── servers/               # 服务端实现
│   │   │   ├── serveravg.py       # FedAvg服务端
│   │   │   ├── serverprox.py      # FedProx服务端
│   │   │   ├── serverproto.py     # FedProto服务端
│   │   │   └── servergpro.py      # FedGpro服务端
│   │   ├── clients/               # 客户端实现
│   │   │   ├── clientavg.py       # FedAvg客户端
│   │   │   ├── clientprox.py      # FedProx客户端
│   │   │   ├── clientproto.py     # FedProto客户端
│   │   │   └── clientgpro.py      # FedGpro客户端
│   │   └── trainmodel/            # 模型定义
│   │       ├── credit.py          # 信用评分MLP模型
│   │       ├── credit_vae.py      # VAE生成模型
│   │       └── importance_aware_dp.py  # 自适应差分隐私
│   ├── utils/                     # 工具模块
│   │   └── analyze_results.py     # 结果分析和报告生成
│   └── results/                   # 实验结果
│       └── 汇总/                  # 汇总报告和图表
└── docs/                          # 文档
```

## 快速开始

### 环境配置

```bash
# 创建conda环境
conda env create -f env_cuda_latest.yaml
conda activate fedgpro

# 或手动安装
pip install torch torchvision numpy pandas h5py matplotlib scipy openpyxl
```

### 数据集生成

```bash
# 一键生成所有数据集（UCI和Xinwang，4种异质性模式）
python dataset/generate_all_datasets_auto.py
```

### 运行实验

```bash
# 运行基线实验（FedAvg, FedProx, FedProto, FedGpro）
python run_baseline_experiments.py

# 仅检查缺失实验
python run_baseline_experiments.py --check

# 仅生成分析报告
python run_baseline_experiments.py --analyze

# 运行消融实验
python run_ablation_experiments.py
```

## 支持的算法

| 算法 | 说明 |
|------|------|
| FedAvg | 经典联邦平均算法 |
| FedProx | 带近端项正则化的联邦学习 |
| FedProto | 基于原型学习的联邦学习 |
| FedGpro | 本文提出的方法（VAE + 原型学习 + 自适应DP） |

## 消融配置

| 配置 | 说明 |
|------|------|
| Full_Model | 完整模型（基准） |
| No_VAE_Generation | 禁用VAE生成数据 |
| No_Prototype | 禁用原型学习 |
| Privacy_Epsilon_1.0 | 隐私预算ε=1.0 |
| Privacy_Epsilon_10.0 | 隐私预算ε=10.0 |
| Privacy_Utility_First | 效用优先的自适应加密 |
| Privacy_Privacy_First | 隐私优先的自适应加密 |
| Generalization_Reserve_2 | 保留20%客户端测试泛化 |

## 数据集

| 数据集 | 样本数 | 特征数 | 正负比例 |
|--------|--------|--------|----------|
| UCI | 30,000 | 23 | 22%:78% |
| Xinwang | 50,000+ | 37 | 约4%:96% |

## 异质性模式

- **Feature**: 特征分布异质性（不同客户端特征分布不同）
- **Label**: 标签分布异质性（不同客户端正负样本比例不同）
- **Quantity**: 数量异质性（不同客户端样本数量不同）
- **IID**: 独立同分布（作为对照组）

## 输出文件

实验完成后，结果保存在 `system/results/汇总/`：

```
汇总/
├── baseline_experiments.xlsx      # 基线实验结果
├── ablation_experiments.xlsx      # 消融实验结果
├── figures/                       # 收敛曲线和消融图
│   ├── ablation_convergence_*.png
│   └── generalization_ablation_*.png
└── heterogeneity_plots/           # 异质性分布图
    ├── label_heterogeneity.png
    ├── feature_heterogeneity.png
    └── quantity_heterogeneity.png
```

## 参数说明

### 基线实验参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| -gr | 全局通信轮数 | 100 |
| -ls | 本地训练轮数 | 5 |
| -nc | 客户端数量 | 10 |
| -lr | 学习率 | 0.005-0.007 |
| -lbs | 批次大小 | 64/128 |
| -t | 重复次数 | 5 |

### FedGpro特有参数

| 参数 | 说明 |
|------|------|
| --fedgpro_use_vae | 是否使用VAE生成数据 |
| --fedgpro_use_prototype | 是否使用原型学习 |
| --fedgpro_phase2_agg | Phase2聚合策略 |
| --fedgpro_epsilon | 差分隐私预算 |
| --fedgpro_use_iadp | 是否使用自适应DP |

## 许可证

MIT License
