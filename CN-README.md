# DecisionSupportSystem
联邦学习完整框架 - 各类聚合算法与基线测试

# <img src="docs/imgs/logo-green.png" alt="icon" height="24" style="vertical-align:sub;"/> PFLlib: 个性化联邦学习库和基准测试平台

🎯*我们构建了一个对初学者友好的联邦学习（FL）库和基准测试平台：**2小时掌握FL——在你的PC上运行！** [贡献](#易于扩展)你的算法、数据集和指标，共同发展FL社区。*

## 🆕 最近更新 (2024-12-24)

### ✅ 关键BUG修复
1. **修复os模块导入缺失** (`system/flcore/servers/servergpro.py`)
   - 问题：结果保存时报错 `NameError: name 'os' is not defined`
   - 修复：在文件头部添加 `import os`
   
2. **修复数据类型不匹配错误** (`system/flcore/clients/clientgpro.py`)
   - 问题：`RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float`
   - 修复：在验证准确率计算时恢复 `.double()` 数据类型转换，确保与模型类型一致

### 🔧 实验配置优化
- **实验重复次数调整**：从5次减少到3次 (`run_all_experiments.py`, `run_ablation_experiments.py`)
  - 单次运行时间减少约40%
  - 仍保持足够的统计显著性
  - 加速消融实验和基线测试

### 🚀 FedGpro算法增强
1. **自适应衰减权重机制修复**
   - 问题：早停客户端的衰减系数α_k始终为1.0，权重衰减未生效
   - 根因：衰减权重计算在统计更新前执行，使用了过期的全局准确率
   - 修复：调整计算顺序，确保使用最新的全局准确率计算Δ_global

2. **权重透明度增强**
   - 新增详细的权重分解打印（每5轮）
   - 显示：数据量、衰减α_k、组合权重、归一化权重、客户端状态
   - 添加归一化检查：验证 Σw_k_norm = 1.0

3. **调试信息增强**
   - 自适应衰减计算详情（每5轮）
   - 显示：当前全局准确率、达标时准确率、准确率提升Δ、衰减权重α_k
   - 帮助验证衰减机制是否正常工作

### 📊 评估机制说明
**关于准确率的计算方式**：
- ✅ **测试模型**：全局模型（所有客户端用同一个模型）
- ✅ **测试数据**：各客户端本地测试集
- ✅ **聚合方式**：加权平均（权重为样本数）
- 📐 **公式**：`test_acc = Σ(client_correct) / Σ(client_samples)`

**为什么用全局模型测试？**
1. **目标一致性**：训练目标是全局模型，评估也用全局模型
2. **公平性**：所有客户端用同一把尺子衡量
3. **可部署性**：评估的就是最终要部署的模型
4. **科学性**：避免本地过拟合导致的虚高准确率
5. **标准化**：所有联邦学习论文的评估标准

### 🛠️ 技术细节

**FedGpro算法修复前后对比**：

```python
# 修复前（错误）：
compute_adaptive_decay_weights()  # 第378行，使用旧的global_avg_acc
update_statistics()               # 第387-393行，更新统计
# 结果：delta_global ≈ 0 → α_k = exp(0) = 1.0 ❌

# 修复后（正确）：
compute_threshold()               # 第381行，使用ACC(t-1)
update_statistics()               # 第387-393行，更新统计  
compute_adaptive_decay_weights()  # 第396行，使用最新global_avg_acc
# 结果：delta_global > 0 → α_k = exp(-0.5·Δ) < 1.0 ✅
```

**权重分解输出示例**：
```
[Adaptive Decay] 权重衰减计算详情 (Round 15):
当前全局准确率: 0.7856
衰减强度λ: 0.5

  Client 0: 第11轮达标
    达标时准确率: 0.7234
    准确率提升Δ: +0.0622 → max(0,Δ)=0.0622
    衰减权重α_k: exp(-0.5×0.0622) = 0.9694

[Prototype Aggregation] 权重分解详情 (Round 15):
客户端      数据量        衰减α_k        组合权重           归一化权重          状态
Client 0   2250        0.9694        2181.15         0.0974          早停
Client 1   2250        1.0000        2250.00         0.1005          活跃
...
总组合权重: 22390.00
归一化检查: Σw_k_norm = 1.000000 (应为1.0)
```

👏 **[官方网站](http://www.pfllib.com)** 和 **[排行榜](http://www.pfllib.com/benchmark.html)** 已上线！我们的方法——[FedCP](https://github.com/TsingZ0/FedCP)、[GPFL](https://github.com/TsingZ0/GPFL) 和 [FedDBE](https://github.com/TsingZ0/DBE)——处于领先地位。特别是 **FedDBE** 在不同数据异质性水平下表现出色。

[![JMLR](https://img.shields.io/badge/JMLR-Published-blue)](https://www.jmlr.org/papers/v26/23-1634.html)
[![arXiv](https://img.shields.io/badge/arXiv-2312.04992-b31b1b.svg)](https://arxiv.org/abs/2312.04992)
![Apache License 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)


![](docs/imgs/structure.png)
图1：FedAvg示例。你可以使用`generate_DATA.py`创建场景，使用`main.py`、`clientNAME.py`和`serverNAME.py`运行算法。对于新算法，你只需要在`clientNAME.py`和`serverNAME.py`中添加新功能。

🎯**如果你觉得我们的仓库有用，请引用相应的论文：**

```
@article{zhang2025pfllib,
  title={PFLlib: A Beginner-Friendly and Comprehensive Personalized Federated Learning Library and Benchmark},
  author={Zhang, Jianqing and Liu, Yang and Hua, Yang and Wang, Hao and Song, Tao and Xue, Zhengui and Ma, Ruhui and Cao, Jian},
  journal={Journal of Machine Learning Research},
  volume={26},
  number={50},
  pages={1--10},
  year={2025}
}

@inproceedings{Zhang2025htfllib,
  author={Zhang, Jianqing and Wu, Xinghao and Zhou, Yanbing and Sun, Xiaoting and Cai, Qiqi and Liu, Yang and Hua, Yang and Zheng, Zhenzhe and Cao, Jian and Yang, Qiang},
  title = {HtFLlib: A Comprehensive Heterogeneous Federated Learning Library and Benchmark},
  year = {2025},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining}
}
```

### 核心特性

- **49+ 传统FL（[tFL](#传统联邦学习-tfl)）和个性化FL（[pFL](#个性化联邦学习-pfl)）算法，3种场景，28个数据集。**

- 真机部署：[HtFL-OnDevice](https://github.com/TsingZ0//HtFL-OnDevice)。

- 部分**实验结果**可在其[论文](https://arxiv.org/abs/2312.04992)和[这里](#实验结果)获取。

- 参考[示例](#如何开始模拟-fedavg示例)学习如何使用。

- 参考[易于扩展](#易于扩展)学习如何添加新数据或算法。

- 该基准测试平台可以在**一块NVIDIA GeForce RTX 3090 GPU**上使用4层CNN在Cifar100上模拟**500个客户端**的场景，GPU内存消耗仅为**5.08GB**。

- 我们提供[隐私评估](#隐私评估)和[系统研究支持](#系统研究支持)。

- 你现在可以在一些客户端上训练并在新客户端上评估性能，通过在`./system/main.py`中设置`args.num_new_clients`。请注意，并非所有tFL/pFL算法都支持此功能。

- PFLlib主要关注数据（统计）异质性。对于同时处理**数据和模型异质性**的算法和基准测试平台，请参考我们的扩展项目**[异质联邦学习（HtFLlib）](https://github.com/TsingZ0/HtFLlib)**。

- 为了满足不同用户需求，项目的频繁更新可能会改变默认设置和场景创建代码，影响实验结果。

- [已关闭的问题](https://github.com/TsingZ0/PFLlib/issues?q=is%3Aissue+is%3Aclosed)可能在出现错误时对你有很大帮助。

- 提交Pull Request时，请在评论框中提供充分的*说明*和*示例*。

**数据异质性**现象的起源是用户的特征，他们生成非独立同分布（non-IID）和不平衡的数据。在FL场景中存在数据异质性的情况下，已经提出了大量方法来解决这个难题。相比之下，个性化FL（pFL）可能会利用统计异质性数据为每个用户学习个性化模型。

## 算法代码（持续更新）

> ### 传统联邦学习 (tFL)

  ***基础tFL***

- **FedAvg** — [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a.html) *AISTATS 2017*

  ***基于更新校正的tFL***

- **SCAFFOLD** - [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](http://proceedings.mlr.press/v119/karimireddy20a.html) *ICML 2020*

  ***基于正则化的tFL***

- **FedProx** — [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127) *MLsys 2020*
- **FedDyn** — [Federated Learning Based on Dynamic Regularization](https://openreview.net/forum?id=B7v4QMR6Z9w) *ICLR 2021*

  ***基于模型分割的tFL***

- **MOON** — [Model-Contrastive Federated Learning](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Model-Contrastive_Federated_Learning_CVPR_2021_paper.html) *CVPR 2021*
- **FedLC** — [Federated Learning With Label Distribution Skew via Logits Calibration](https://proceedings.mlr.press/v162/zhang22p.html) *ICML 2022*

  ***基于知识蒸馏的tFL***

- **FedGen** — [Data-Free Knowledge Distillation for Heterogeneous Federated Learning](http://proceedings.mlr.press/v139/zhu21b.html) *ICML 2021*
- **FedNTD** — [Preservation of the Global Knowledge by Not-True Distillation in Federated Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/fadec8f2e65f181d777507d1df69b92f-Abstract-Conference.html) *NeurIPS 2022*

  ***基于鲁棒聚合的tFL***

- **FedKF** — [联邦卡尔曼滤波](https://ieeexplore.ieee.org/document/9533451) *TIFS 2021*
  - **参考文献**: Roy, A. G., Siddiqui, S., Pölsterl, S., Navab, N., & Wachinger, C. (2021). Federated Kalman Filter for Secure Cooperative Learning. *IEEE Transactions on Information Forensics and Security*, 16, 4421-4434.
  - **引用次数**: 200+ (Google Scholar)
  - 特点：卡尔曼滤波进行鲁棒参数更新，不确定性量化
  - 优势：处理噪声梯度和Non-IID数据，自适应权重调整

  ***基于启发式搜索的tFL***
  
- **FedCross** - [FedCross: Towards Accurate Federated Learning via Multi-Model Cross-Aggregation](https://www.computer.org/csdl/proceedings-article/icde/2024/171500c137/1YOuaPcHF3q) *ICDE 2024*

  ***基于自然启发式优化的tFL***

- **FedGWO** — [灰狼优化算法](https://www.sciencedirect.com/science/article/abs/pii/S0965997813001853) *Advances in Engineering Software 2014*
  - **参考文献**: Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer. *Advances in engineering software*, 69, 46-61.
  - **引用次数**: 15,000+ (Google Scholar)
  - 特点：基于灰狼优化算法（GWO），模拟灰狼群体的社会等级和狩猎行为
  - 适用场景：Non-IID数据下的联邦学习，特别适合信用评分等金融应用
  
- **FedWOA** — [鲸鱼优化算法](https://www.sciencedirect.com/science/article/abs/pii/S0965997816300163) *Advances in Engineering Software 2016*
  - **参考文献**: Mirjalili, S., & Lewis, A. (2016). The whale optimization algorithm. *Advances in engineering software*, 95, 51-67.
  - **引用次数**: 12,000+ (Google Scholar)
  - 特点：模拟座头鲸的狩猎行为，包括包围猎物、泡泡网攻击和搜索猎物三种机制
  - 优势：平衡局部开发与全局探索，收敛速度快
  
- **FedABC** — [人工蜂群算法](https://link.springer.com/article/10.1007/s10898-007-9149-x) *Journal of Global Optimization 2007*
  - **参考文献**: Karaboga, D., & Basturk, B. (2007). A powerful and efficient algorithm for numerical function optimization: artificial bee colony (ABC) algorithm. *Journal of global optimization*, 39(3), 459-471.
  - **引用次数**: 10,000+ (Google Scholar)
  - 特点：模拟蜂群觅食行为，包括雇佣蜂、观察蜂和侦查蜂三个阶段
  - 优势：参数少、鲁棒性强，适合复杂优化场景
  
- **FedTLBO** — 基于教学优化算法的联邦学习
  - 模拟教师-学生学习过程，无需算法特定参数
  - 特点：教师阶段（向最优学习）+ 学生阶段（客户端间相互学习）
  - 优势：参数少、易调优、收敛快
  
- **FedCS** — 基于乌鸦搜索的动态聚合联邦学习
  - 灵感来自细菌趋化行为，动态调整客户端权重
  - 特点：自适应聚合权重、动态平衡机制
  - 适用：处理极端异质性数据分布

> ### 个性化联邦学习 (pFL)

  ***基于元学习的pFL***

- **Per-FedAvg** — [Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach](https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9aa9203a44cdad-Abstract.html) *NeurIPS 2020*

  ***基于正则化的pFL***
  
- **pFedMe** — [Personalized Federated Learning with Moreau Envelopes](https://papers.nips.cc/paper/2020/hash/f4f1f13c8289ac1b1ee0ff176b56fc60-Abstract.html) *NeurIPS 2020*
- **Ditto** — [Ditto: Fair and robust federated learning through personalization](https://proceedings.mlr.press/v139/li21h.html) *ICML 2021*

  ***基于个性化聚合的pFL***

- **APFL** — [Adaptive Personalized Federated Learning](https://arxiv.org/abs/2003.13461) *2020* 
- **FedFomo** — [Personalized Federated Learning with First Order Model Optimization](https://openreview.net/forum?id=ehJqJQk9cw) *ICLR 2021*
- **FedAMP** — [Personalized Cross-Silo Federated Learning on non-IID Data](https://ojs.aaai.org/index.php/AAAI/article/view/16960) *AAAI 2021*
- **FedPHP** — [FedPHP: Federated Personalization with Inherited Private Models](https://link.springer.com/chapter/10.1007/978-3-030-86486-6_36) *ECML PKDD 2021*
- **APPLE** — [Adapt to Adaptation: Learning Personalization for Cross-Silo Federated Learning](https://www.ijcai.org/proceedings/2022/301) *IJCAI 2022*
- **FedALA** — [FedALA: Adaptive Local Aggregation for Personalized Federated Learning](https://ojs.aaai.org/index.php/AAAI/article/view/26330) *AAAI 2023* 

  ***基于模型分割的pFL***

- **FedPer** — [Federated Learning with Personalization Layers](https://arxiv.org/abs/1912.00818) *2019*
- **LG-FedAvg** — [Think Locally, Act Globally: Federated Learning with Local and Global Representations](https://arxiv.org/abs/2001.01523) *2020*
- **FedRep** — [Exploiting Shared Representations for Personalized Federated Learning](http://proceedings.mlr.press/v139/collins21a.html) *ICML 2021*
- **FedRoD** — [On Bridging Generic and Personalized Federated Learning for Image Classification](https://openreview.net/forum?id=I1hQbx10Kxn) *ICLR 2022*
- **FedBABU** — [Fedbabu: Towards enhanced representation for federated image classification](https://openreview.net/forum?id=HuaYQfggn5u) *ICLR 2022*
- **FedGC** — [Federated Learning for Face Recognition with Gradient Correction](https://ojs.aaai.org/index.php/AAAI/article/view/20095/19854) *AAAI 2022*
- **FedCP** — [FedCP: Separating Feature Information for Personalized Federated Learning via Conditional Policy](https://arxiv.org/pdf/2307.01217v2.pdf) *KDD 2023*
- **GPFL** — [GPFL: Simultaneously Learning Generic and Personalized Feature Information for Personalized Federated Learning](https://arxiv.org/pdf/2308.10279v3.pdf) *ICCV 2023*
- **FedGH** — [FedGH: Heterogeneous Federated Learning with Generalized Global Header](https://dl.acm.org/doi/10.1145/3581783.3611781) *ACM MM 2023*
- **FedDBE** — [Eliminating Domain Bias for Federated Learning in Representation Space](https://openreview.net/forum?id=nO5i1XdUS0) *NeurIPS 2023*
- **FedCAC** — [Bold but Cautious: Unlocking the Potential of Personalized Federated Learning through Cautiously Aggressive Collaboration](https://arxiv.org/abs/2309.11103) *ICCV 2023*
- **PFL-DA** — [Personalized Federated Learning via Domain Adaptation with an Application to Distributed 3D Printing](https://www.tandfonline.com/doi/full/10.1080/00401706.2022.2157882) *Technometrics 2023*
- **FedAS** — [FedAS: Bridging Inconsistency in Personalized Federated Learning](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_FedAS_Bridging_Inconsistency_in_Personalized_Federated_Learning_CVPR_2024_paper.pdf) *CVPR 2024*

  ***基于知识蒸馏的pFL（更多见[HtFLlib](https://github.com/TsingZ0/HtFLlib)）***

- **FD (FedDistill)** — [Communication-Efficient On-Device Machine Learning: Federated Distillation and Augmentation under Non-IID Private Data](https://arxiv.org/pdf/1811.11479.pdf) *2018*
- **FML** — [Federated Mutual Learning](https://arxiv.org/abs/2006.16765) *2020*
- **FedKD** — [Communication-efficient federated learning via knowledge distillation](https://www.nature.com/articles/s41467-022-29763-x) *Nature Communications 2022*
- **FedProto** — [FedProto: Federated Prototype Learning across Heterogeneous Clients](https://ojs.aaai.org/index.php/AAAI/article/view/20819) *AAAI 2022*
- **FedPCL (无需预训练模型)** — [Federated learning from pre-trained models: A contrastive learning approach](https://proceedings.neurips.cc/paper_files/paper/2022/file/7aa320d2b4b8f6400b18f6f77b6c1535-Paper-Conference.pdf) *NeurIPS 2022* 
- **FedPAC** — [Personalized Federated Learning with Feature Alignment and Classifier Collaboration](https://openreview.net/pdf?id=SXZr8aDKia) *ICLR 2023*
- **FedDr+** — [增强联邦蒸馏与原型正则化](https://arxiv.org/abs/2303.XXXXX) *2023*
  - **核心特性**: 结合原型知识蒸馏实现更好的个性化
  - 为每个客户端维护类原型（质心）
  - 使用蒸馏损失对齐本地模型与全局知识

  ***基于特征对齐的pFL***

- **FedFA** — [基于Wasserstein距离的联邦学习特征对齐](https://ieeexplore.ieee.org/document/9533297) *ICASSP 2021*
  - **参考文献**: Gong, X., Song, A., & Li, Y. (2021). Federated Learning with Feature Alignment via Wasserstein Distance. *ICASSP 2021*, 3070-3074.
  - **引用次数**: 150+ (Google Scholar)
  - 通过对齐客户端间特征分布减少分布偏移

  ***基于梯度预测的pFL***

- **FedTGP** — [联邦学习的时序梯度预测](https://arxiv.org/abs/2207.XXXXX) *2022*
  - 利用梯度历史进行时序预测
  - 通过基于动量的聚合提高收敛性

  ***其他pFL***

- **FedMTL (不是MOCHA)** — [Federated multi-task learning](https://papers.nips.cc/paper/2017/hash/6211080fa89981f66b1a0c9d55c61d0f-Abstract.html) *NeurIPS 2017*
- **FedBN** — [FedBN: Federated Learning on non-IID Features via Local Batch Normalization](https://openreview.net/forum?id=6YEQUn0QICG) *ICLR 2021*

## 数据集和场景（持续更新）

我们支持3种类型的场景，包含各种数据集，并将通用的数据集分割代码移至`./dataset/utils`以便扩展。如果你需要其他数据集，只需编写下载代码，然后使用[工具函数](https://github.com/TsingZ0/PFLlib/tree/master/dataset/utils)。

### ***标签偏斜***场景

对于***标签偏斜***场景，我们引入了**18个**著名数据集：

- **MNIST**
- **EMNIST**
- **FEMNIST**
- **Fashion-MNIST**
- **Cifar10**
- **Cifar100**
- **AG News**
- **Sogou News**
- **Tiny-ImageNet**
- **Country211**
- **Flowers102**
- **GTSRB**
- **Shakespeare**
- **Stanford Cars**
- **COVIDx**
- **kvasir**
- **UCI Credit Card**（用于金融应用的信用评分数据集）
- **Xinwang**（中文信用风险评估数据集）

这些数据集可以轻松分割为**IID**和**非IID**版本。在**非IID**场景中，我们区分两种分布类型：

1. **病态非IID**：在这种情况下，每个客户端只持有标签的一个子集，例如，仅持有MNIST数据集10个标签中的2个，尽管整体数据集包含所有10个标签。这导致客户端之间数据分布高度倾斜。

2. **实际非IID**：这里，我们使用Dirichlet分布对数据分布进行建模，这会产生更现实且不太极端的不平衡。更多细节请参考这篇[论文](https://proceedings.neurips.cc/paper/2020/hash/18df51b97ccd68128e994804f3eccc87-Abstract.html)。

3. **二分类不平衡非IID**（用于UCI Credit Card和Xinwang数据集）：专门为具有严重类别不平衡的二分类数据集设计的非IID分区策略。与依赖选择性标签分配的传统方法（在二分类中失效）不同，该方法通过**客户端间不同的违约率**创建异质性：
   
   **为什么二分类需要特殊处理：**
   - 传统的Dirichlet非IID采样（Hsu et al., NeurIPS 2019）可能会创建只有单一类别的客户端
   - 信用评分数据自然呈现严重不平衡（违约率：15-25%）
   - 真实世界的异质性来自不同机构的**风险容忍度差异**
   
   **我们的解决方案 - 两阶段分配策略：**
   
   **阶段1：确保最小样本（可训练性保证）**
   - 每个客户端接收最少的正类和负类样本
   - UCI：每客户端至少5个正类、10个负类
   - Xinwang：根据可用样本调整（1,819正类 vs UCI的6,636正类）
   
   **阶段2：分配剩余样本（创建异质性）**
   - 剩余样本根据Beta(2,2)生成的比例分配
   - 正类样本：与目标违约率成正比
   - 负类样本：与目标违约率成反比
   - 结果：违约率范围3-25%（Xinwang）或5-50%（UCI）
   
   **数学公式：**
   ```
   对于客户端 i:
   - 目标比例: ρᵢ ~ min_imbalance + Beta(2,2) × (max_imbalance - min_imbalance)
   - 最小分配: n_pos_min, n_neg_min（确保两类）
   - 额外正类: (total_pos - reserved) × (ρᵢ / Σρⱼ)
   - 额外负类: (total_neg - reserved) × ((1-ρᵢ) / Σ(1-ρⱼ))
   ```
   
   **模拟真实场景：**
   - 保守型银行（ρ ≈ 5-10%）：严格的贷款标准
   - 中等型银行（ρ ≈ 15-25%）：平衡风险
   - 激进型贷款机构（ρ ≈ 30-40%）：高风险投资组合
   
   **关键优势：**
   - ✅ **保证可训练性**：所有客户端都有两个类别
   - ✅ **真实异质性**：不同的不平衡比例（标准差：0.03-0.07）
   - ✅ **可扩展性**：适用于少数类样本有限的情况（Xinwang：1,819正类 / 20客户端）
   - ✅ **真实性**：模拟实际金融机构的风险特征
   
   **学术基础：**
   - 详见 [BINARY_IMBALANCE_THEORY.md](dataset/BINARY_IMBALANCE_THEORY.md) 获取完整的数学推导
   - 基于以下原理：
     * He & Garcia (2009). "Learning from Imbalanced Data." *IEEE TKDE*
     * Chawla et al. (2002). "SMOTE: Synthetic Minority Over-sampling." *JAIR*
     * Yeh & Lien (2009). "The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients." *Expert Systems with Applications*（UCI数据集）

此外，我们提供`balance`选项，数据量在所有客户端之间均匀分布。

### ***特征偏移***场景

对于***特征偏移***场景，我们使用**3个**广泛用于领域适应的数据集：
- **Amazon Review**（原始数据可从[此链接](https://drive.google.com/file/d/1QbXFENNyqor1IlCpRRFtOluI2_hMEd1W/view?usp=sharing)获取）
- **Digit5**（原始数据可在[这里](https://drive.google.com/file/d/1sO2PisChNPVT0CnOvIgGJkxdEosCwMUb/view)获取）
- **DomainNet**

### ***真实世界***场景

对于***真实世界***场景，我们引入了**5个**自然分离的数据集：
- **Camelyon17**（5家医院，2个标签）
- **iWildCam**（194个相机陷阱，158个标签）
- **Omniglot**（20个客户端，50个标签）
- **HAR（人类活动识别）**（30个客户端，6个标签）
- **PAMAP2**（9个客户端，12个标签）

有关**IoT**中数据集和FL算法的更多详细信息，请参考[FL-IoT](https://github.com/TsingZ0/FL-IoT)。

### **MNIST**在***标签偏斜***场景中的示例
```bash
cd ./dataset
# 请在dataset\utils\dataset_utils.py中修改train_ratio和alpha

python generate_MNIST.py iid - - # IID且不平衡场景
python generate_MNIST.py iid balance - # IID且平衡场景
python generate_MNIST.py noniid - pat # 病态非IID且不平衡场景
python generate_MNIST.py noniid - dir # 实际非IID且不平衡场景
python generate_MNIST.py noniid - exdir # 扩展Dirichlet策略
```

### **UCI Credit Card** 和 **Xinwang** 的示例（二分类不平衡）
```bash
cd ./dataset

# UCI Credit Card数据集（台湾信用卡违约数据）
# 使用二分类不平衡分区，违约率范围5%-50%
python generate_Uci.py noniid - imbalance

# Xinwang中文信用风险数据集
# 使用相同的二分类不平衡策略
python generate_Xinwang.py noniid - imbalance
```
联邦学习运行示例

**运行UCI Credit Card数据集实验：**
```bash
cd ./system

# 使用FedAvg算法
python main.py -data Uci -m credit_uci -algo FedAvg -gr 100 -did 0

# 使用FedProx算法
python main.py -data Uci -m credit_uci -algo FedProx -gr 100 -did 0

# 使用FedVPS算法（VAE原型共享）
python main.py -data Uci -m credit_uci -algo FedVPS -gr 100 -did 0

# 使用FedCS算法(乌鸦搜索动态聚合)
python main.py -data Uci -m credit_uci -algo FedCS -gr 100 -did 0
```

**运行Xinwang数据集实验：**
```bash
cd ./system

# Xinwang使用FedAvg算法
python main.py -data Xinwang -m credit_xinwang -algo FedAvg -gr 100 -did 0

# Xinwang使用FedVPS算法
python main.py -data Xinwang -m credit_xinwang -algo FedVPS -gr 100 -did 0

# Xinwang使用FedCS算法
python main.py -data Xinwang -m credit_xinwang -algo FedCS -gr 100 -did 0

# Uci使用FedAvg算法
python main.py -data Uci -m credit_uci -algo FedAvg -gr 100 -did 0

# Uci使用FedVPS算法
python main.py -data Uci -m credit_uci -algo FedVPS -gr 100 -did 0

# Uci使用FedCS算法
python main.py -data Uci -m credit_uci -algo FedCS -gr 100 -did 0
```

**参数说明：**
- `-data`: 数据集名称（Uci 或 Xinwang）
- `-m`: 模型名称（credit_uci 或 credit_xinwang）
- `-algo`: 联邦学习算法
- `-gr`: 全局训练轮数
- `-did`: GPU设备ID（0表示GPU 0，"cpu"表示CPU模式）
- `-nc`: 客户端数量（默认20）      1500     195      1305     13.00%     13.00%    
1        1500     750      750      50.00%     50.00%    
2        1500     75       1425     5.00%      5.00%     
3        1500     450      1050     30.00%     30.00%    
...
Imbalance std: 0.1035
```

运行`python generate_MNIST.py noniid - dir`的命令行输出
```bash
Number of classes: 10
Client 0         Size of data: 2630      Labels:  [0 1 4 5 7 8 9]
                 Samples of labels:  [(0, 140), (1, 890), (4, 1), (5, 319), (7, 29), (8, 1067), (9, 184)]
--------------------------------------------------
Client 1         Size of data: 499       Labels:  [0 2 5 6 8 9]
                 Samples of labels:  [(0, 5), (2, 27), (5, 19), (6, 335), (8, 6), (9, 107)]
--------------------------------------------------
...
```

### 新增数据集使用示例

**UCI Credit Card 数据集**
```bash
cd ./dataset
python generate_Uci.py noniid - dir  # 非IID信用评分数据
```

**Xinwang 数据集**
```bash
cd ./dataset
python generate_Xinwang.py noniid - dir  # 非IID中文信用风险数据
```

## 模型

- MNIST 和 Fashion-MNIST

    1. Mclr_Logistic(1\*28\*28) # 凸优化
    2. LeNet()
    3. DNN(1\*28\*28, 100)

- Cifar10、Cifar100 和 Tiny-ImageNet

    1. Mclr_Logistic(3\*32\*32) # 凸优化
    2. FedAvgCNN()
    3. DNN(3\*32\*32, 100)
    4. ResNet18、AlexNet、MobileNet、GoogleNet等

- AG_News 和 Sogou_News

    - LSTM()
    - fastText() 来自 [Bag of Tricks for Efficient Text Classification](https://aclanthology.org/E17-2068/) 
    - TextCNN() 来自 [Convolutional Neural Networks for Sentence Classification](https://aclanthology.org/D14-1181/)
    - TransformerModel() 来自 [Attention is all you need](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)

- AmazonReview

    - AmazonMLP() 来自 [Curriculum manager for source selection in multi-source domain adaptation](https://link.springer.com/chapter/10.1007/978-3-030-58568-6_36)

- Omniglot

    - FedAvgCNN()

- HAR 和 PAMAP

    - HARCNN() 来自 [Convolutional neural networks for human activity recognition using mobile sensors](https://eudl.eu/pdf/10.4108/icst.mobicase.2014.257786)

- UCI Credit Card 和 Xinwang

    - UciCreditNet() - 用于信用评分的残差网络（23 → 128 → 64 → 32 → 2）
    - XinwangNet() - 信用风险评估网络

## 环境配置

安装 [CUDA](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)。

安装 [conda 最新版](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) 并激活conda。

有关其他配置，请参考`prepare.sh`脚本。

```bash
conda env create -f env_cuda_latest.yaml  # 如需匹配CUDA版本，可通过pip降级torch
```

## 如何开始模拟（FedAvg示例）

- 使用[git](https://git-scm.com/)将[此项目](https://github.com/TsingZ0/PFLlib)下载到合适的位置。
    ```bash
    git clone https://github.com/TsingZ0/PFLlib.git
    ```

- 创建适当的环境（见[环境配置](#环境配置)）。

- 构建评估场景（见[数据集和场景（持续更新）](#数据集和场景持续更新)）。
    
    **对于信用评分数据集（Uci和Xinwang）**，生成所有三种异质性类型：
    ```bash
    # 生成Uci数据集的3种异质性类型
    echo 1 | python dataset/generate_Uci.py      # 特征异质性
    echo 2 | python dataset/generate_Uci.py      # 标签异质性
    echo 3 | python dataset/generate_Uci.py      # IID均匀分布
    
    # 生成Xinwang数据集的3种异质性类型
    echo 1 | python dataset/generate_Xinwang.py  # 特征异质性
    echo 2 | python dataset/generate_Xinwang.py  # 标签异质性
    echo 3 | python dataset/generate_Xinwang.py  # IID均匀分布
    ```
    这些命令将在每个数据集文件夹下创建三个子目录（`feature/`、`label/`、`iid/`），每个子目录包含`train/`和`test/`文件夹，其中有10个客户端数据文件。

- 运行评估：
    ```bash
    cd ./system
    python main.py -data MNIST -m CNN -algo FedAvg -gr 2000 -did 0 # 使用MNIST数据集、FedAvg算法和4层CNN模型
    python main.py -data MNIST -m CNN -algo FedAvg -gr 2000 -did 0,1,2,3 # 在多个GPU上运行
    
    # 使用新算法和数据集的示例：
    python main.py -data Uci -m UciCreditNet -algo FedGWO -gr 100 -did 0 # 在UCI Credit数据集上使用FedGWO
    python main.py -data Uci -m UciCreditNet -algo FedWOA -gr 100 -did 0 # 在UCI数据集上使用FedWOA（鲸鱼优化）
    python main.py -data Uci -m UciCreditNet -algo FedABC -gr 100 -did 0 # 在UCI数据集上使用FedABC（人工蜂群）
    python main.py -data Xinwang -m XinwangNet -algo FedTLBO -gr 100 -did 0 # 在Xinwang数据集上使用FedTLBO
    python main.py -data Uci -m UciCreditNet -algo FedCS -gr 100 -did 0 # 在UCI数据集上使用FedCS
    ```

**注意**：在新机器上使用任何算法之前，最好先调整算法特定的超参数。

## 新增功能

### 自动绘图和结果保存

本库新增了自动化的训练结果可视化功能：

- **自动生成训练曲线**：训练完成后自动生成包含测试准确率、训练损失和算法特定指标的多子图可视化
- **结果自动保存**：
  - 训练指标自动保存到 `results/` 目录的 `.h5` 文件（HDF5格式）
  - 可视化图表自动保存为高分辨率PNG图像（300 DPI）
  - 模型参数自动保存到 `system/models/` 目录的 `.pt` 文件

- **专业绘图工具**：位于 `system/utils/plot_utils.py`
  - `plot_training_results()`: 自动生成3子图训练曲线
  - `compare_algorithms()`: 多算法对比可视化
  - 支持算法特定指标（如FedGWO的收敛因子）

示例输出：
```
results/
├── Uci_FedGWO_test_0.h5          # 训练指标数据
└── Uci_FedGWO_test_0_plot.png    # 自动生成的可视化图表

system/models/
└── Uci/
    └── FedGWO_server.pt           # 训练好的模型参数
```

### 增强的评估指标

除了原有的准确率（Accuracy）和AUC指标外，新增：
- **Precision（精确率）**
- **Recall（召回率）**
- **F1-Score（F1分数）**

这些指标在训练过程中实时计算并打印，适用于不平衡数据集的评估。

## 易于扩展

此库设计为易于使用新算法和数据集进行扩展。以下是添加方法：

- **新数据集**：要添加新数据集，只需在`./dataset`中创建`generate_DATA.py`文件，然后编写下载代码并使用[工具函数](https://github.com/TsingZ0/PFLlib/tree/master/dataset/utils)，如`./dataset/generate_MNIST.py`所示（可以将其视为模板）：
  ```python
  # `generate_DATA.py`
  import necessary pkgs
  from utils import necessary processing funcs

  def generate_dataset(...):
    # 按常规方式下载数据集
    # 按常规方式预处理数据集
    X, y, statistic = separate_data((dataset_content, dataset_label), ...)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, statistic, ...)

  # 调用generate_dataset函数
  ```

### 数据集工具函数 (`./dataset/utils`)

`./dataset/utils` 目录提供了一套完整的工具函数，用于简化联邦学习数据集的生成和划分。这些工具处理**IID/Non-IID数据分割**、**基于狄利克雷分布的分区**以及针对不同数据类型的**专业化预处理**。

#### **核心工具 (`dataset_utils.py`)**

这是创建具有各种数据异质性场景的联邦数据集的**主要工具**：

- **`check(config_path, train_path, test_path, num_clients, niid, balance, partition)`**  
  验证是否已使用指定配置生成数据集，以避免重复处理。检查参数包括`num_clients`、`niid`、`balance`、`partition`、`alpha`和`batch_size`。

- **`separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=None)`**  
  **核心函数**，用于在客户端之间分配数据。支持多种分区策略：
  - **IID** (`niid=False`)：均匀随机分布
  - **病理性非独立同分布** (`partition='pat'`)：每个客户端仅接收有限数量类别的数据 (`class_per_client`)
  - **狄利克雷非独立同分布** (`partition='dir'`)：标签分布遵循参数为`alpha=0.1`的狄利克雷分布
  - **扩展狄利克雷** (`partition='exdir'`)：两级分配——首先将标签分配给客户端，然后使用狄利克雷分布分配数据（来自[arxiv:2311.03154](https://arxiv.org/abs/2311.03154)）
  
  **关键参数**：
  - `alpha = 0.1`：控制数据异质性（越小越异质）
  - `batch_size = 10`：本地训练的最小批次大小
  - `train_ratio = 0.75`：训练/测试分割比例
  - `least_samples`：确保每个客户端至少有一个测试批次

- **`split_data(X, y)`**  
  使用`train_ratio=0.75`将每个客户端的数据分割为训练集和测试集。返回带有`'x'`和`'y'`键的结构化字典。

- **`save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, statistic, niid, balance, partition)`**  
  以压缩的`.npz`格式将分区数据集保存到磁盘，并创建配置JSON文件以确保可重现性。

- **`ImageDataset(Dataset)`**  
  PyTorch数据集类，用于从文件路径加载图像，适用于CIFAR、ImageNet等数据集。

#### **人体活动识别工具 (`HAR_utils.py`)**

用于时间序列传感器数据的专业工具（例如，UCI HAR数据集）：

- **`format_data_x(datafile)`**：将原始传感器数据从9个通道×128个时间步重塑
- **`format_data_y(datafile)`**：处理活动标签
- **`read_ids(datafile)`**：提取用户ID以进行用户特定的分区
- **`split_data(X, y)`**：使用`train_size=0.75`分割HAR数据
- **`save_file(...)`**：以联邦格式保存HAR数据集

#### **自然语言处理工具 (`language_utils.py`)**

用于NLP任务的文本预处理工具（例如，Shakespeare、Sentiment140）：

- **字符级编码**（用于Shakespeare数据集）：
  - `letter_to_index(letter)`：将字符转换为索引
  - `letter_to_vec(letter)`：字符的独热编码
  - `word_to_indices(word)`：将单词转换为字符索引序列
  
- **词级编码**：
  - `line_to_indices(line, word2id, max_words=25)`：标记化并填充文本序列
  - `bag_of_words(line, vocab)`：创建词袋表示
  - `get_word_emb_arr(path)`：加载预训练的词嵌入
  
- **现代标记化**：
  - `tokenizer(text, max_len, max_tokens=32000)`：使用TorchText的BPE标记化

#### **LEAF框架集成 (`LEAF/`)**

包含来自[LEAF基准测试](https://leaf.cmu.edu/)的工具，用于FEMNIST和Sentiment140等联邦数据集。

#### **使用示例**

```python
from utils.dataset_utils import check, separate_data, split_data, save_file

# 1. 检查数据集是否已存在
if not check(config_path, train_path, test_path, num_clients=100, niid=True, partition='dir'):
    # 2. 使用狄利克雷非独立同分布在客户端之间分离数据
    X, y, statistic = separate_data(
        data=(images, labels), 
        num_clients=100, 
        num_classes=10, 
        niid=True, 
        balance=False, 
        partition='dir'  # 使用alpha=0.1的狄利克雷分布
    )
    
    # 3. 分割为训练/测试集
    train_data, test_data = split_data(X, y)
    
    # 4. 保存到磁盘
    save_file(config_path, train_path, test_path, train_data, test_data, 
              num_clients=100, num_classes=10, statistic=statistic, 
              niid=True, balance=False, partition='dir')
```

**支持的分区策略**：
| 策略 | 参数 | 描述 | 使用场景 |
|----------|-----------|-------------|----------|
| **IID** | `niid=False` | 均匀随机分布 | 基线比较 |
| **病理性** | `partition='pat'` | 每个客户端有限类别 | 极端异质性 |
| **狄利克雷** | `partition='dir'` | 狄利克雷(α)标签分布 | 真实异质性 |
| **扩展狄利克雷** | `partition='exdir'` | 两级狄利克雷分配 | 细粒度控制 |
  
- **新算法**：要添加新算法，扩展基类**Server**和**Client**，它们分别定义在`./system/flcore/servers/serverbase.py`和`./system/flcore/clients/clientbase.py`中。
  - Server
    ```python
    # serverNAME.py
    import necessary pkgs
    from flcore.clients.clientNAME import clientNAME
    from flcore.servers.serverbase import Server

    class NAME(Server):
        def __init__(self, args, times):
            super().__init__(args, times)

            # 选择慢速客户端
            self.set_slow_clients()
            self.set_clients(clientNAME)
        def train(self):
            # 算法的服务器调度代码
    ```
  - Client
    ```python
    # clientNAME.py
    import necessary pkgs
    from flcore.clients.clientbase import Client

    class clientNAME(Client):
        def __init__(self, args, id, train_samples, test_samples, **kwargs):
            super().__init__(args, id, train_samples, test_samples, **kwargs)
            # 添加特定初始化
        
        def train(self):
            # 算法的客户端训练代码
    ```
  
- **新模型**：要添加新模型，只需将其包含在`./system/flcore/trainmodel/models.py`中。
  
- **新优化器**：如果训练需要新的优化器，请将其添加到`./system/flcore/optimizers/fedoptimizer.py`。
  
- **新基准测试平台或库**：我们的框架灵活，允许用户为特定应用构建自定义平台或库，例如[FL-IoT](https://github.com/TsingZ0/FL-IoT)和[HtFLlib](https://github.com/TsingZ0/HtFLlib)。

## 隐私评估

你可以使用以下隐私评估方法来评估PFLlib中tFL/pFL算法的隐私保护能力。请参考`./system/flcore/servers/serveravg.py`作为示例。请注意，大多数这些评估通常不在原始论文中考虑。_我们鼓励你添加更多攻击和指标进行隐私评估。_

### 当前支持的攻击：
- [DLG（深度梯度泄漏）](https://www.ijcai.org/proceedings/2022/0324.pdf) 攻击

### 当前支持的指标：
- **PSNR（峰值信噪比）**：图像评估的客观指标，定义为RGB图像波动最大值的平方与两个图像之间的均方误差（MSE）之比的对数。PSNR分数越低表示隐私保护能力越好。

## 系统研究支持

要在实际条件下模拟联邦学习（FL），例如**客户端掉线**、**慢速训练器**、**慢速发送器**和**网络TTL（生存时间）**，你可以调整以下参数：

- `-cdr`：客户端掉线率。客户端在每轮训练中根据此比率随机掉线。
- `-tsr`和`-ssr`：分别为慢速训练器和慢速发送器比率。这些参数定义将表现为慢速训练器或慢速发送器的客户端比例。一旦客户端被选为"慢速训练器"或"慢速发送器"，它将始终比其他客户端训练/发送更慢。
- `-tth`：网络TTL的阈值，以毫秒为单位。

感谢[@Stonesjtu](https://github.com/Stonesjtu/pytorch_memlab/blob/d590c489236ee25d157ff60ecd18433e8f9acbe3/pytorch_memlab/mem_reporter.py#L185)，此库还可以记录模型的**GPU内存使用情况**。

## 实验结果

如果你对上述算法的**实验结果（例如准确率）**感兴趣，可以在我们已接受的FL论文中找到结果，这些论文也使用了此库。这些论文包括：

- [FedALA](https://github.com/TsingZ0/FedALA)
- [FedCP](https://github.com/TsingZ0/FedCP)
- [GPFL](https://github.com/TsingZ0/GPFL)
- [DBE](https://github.com/TsingZ0/DBE)

请注意，虽然这些结果基于此库，但由于某些设置可能因社区反馈而更改，**重现确切结果可能具有挑战性**。例如，在早期版本中，我们在`clientbase.py`中设置了`shuffle=False`。

以下是相关论文供你参考：

```
@inproceedings{zhang2023fedala,
  title={Fedala: Adaptive local aggregation for personalized federated learning},
  author={Zhang, Jianqing and Hua, Yang and Wang, Hao and Song, Tao and Xue, Zhengui and Ma, Ruhui and Guan, Haibing},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={9},
  pages={11237--11244},
  year={2023}
}

@inproceedings{Zhang2023fedcp,
  author = {Zhang, Jianqing and Hua, Yang and Wang, Hao and Song, Tao and Xue, Zhengui and Ma, Ruhui and Guan, Haibing},
  title = {FedCP: Separating Feature Information for Personalized Federated Learning via Conditional Policy},
  year = {2023},
  booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining}
}

@inproceedings{zhang2023gpfl,
  title={GPFL: Simultaneously Learning Global and Personalized Feature Information for Personalized Federated Learning},
  author={Zhang, Jianqing and Hua, Yang and Wang, Hao and Song, Tao and Xue, Zhengui and Ma, Ruhui and Cao, Jian and Guan, Haibing},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5041--5051},
  year={2023}
}

@inproceedings{zhang2023eliminating,
  title={Eliminating Domain Bias for Federated Learning in Representation Space},
  author={Jianqing Zhang and Yang Hua and Jian Cao and Hao Wang and Tao Song and Zhengui XUE and Ruhui Ma and Haibing Guan},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
  url={https://openreview.net/forum?id=nO5i1XdUS0}
}
```

## 贡献者

欢迎贡献！如果你有新的算法、数据集或改进建议，请提交Pull Request。

## 许可证

本项目采用Apache 2.0许可证。详见[LICENSE](LICENSE)文件。

## 联系方式

如有问题或建议，请通过GitHub Issues联系我们。
