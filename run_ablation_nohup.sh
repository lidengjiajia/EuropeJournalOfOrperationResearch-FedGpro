#!/bin/bash
# ===================================================================================
# FedGpro 消融实验批量运行脚本（nohup后台运行）
# ===================================================================================
# 
# 功能说明:
#   - 运行25个消融实验配置（图5-8对应实验）
#   - 每个配置在2数据集×4异构度下测试 = 200个实验组
#   - 每组重复5次（-t 5参数）= 1000个结果文件
#   - 自动检测GPU并并发运行（每GPU 2个槽位）
#   - 智能跳过已完成实验（已有5个结果文件）
# 
# 使用方法:
#   bash run_ablation_nohup.sh
# 
# 查看实时日志:
#   tail -f nohup_ablation.out                              # 总体进度
#   tail -f logs/Ablation_Full_Model_Uci_feature.log       # 单个实验详细日志
# 
# 结果文件位置:
#   system/results/{Dataset}_FedGpro-FedGwo_{Heterogeneity}/
#   └── Ablation_{ConfigName}_{Dataset}_{Heterogeneity}_test_0.h5
#   └── Ablation_{ConfigName}_{Dataset}_{Heterogeneity}_test_1.h5
#   └── Ablation_{ConfigName}_{Dataset}_{Heterogeneity}_test_2.h5
# ===================================================================================

echo ""
echo "======================================================================================================"
echo "                       FedGpro 消融实验 - 后台运行启动                                               "
echo "======================================================================================================"
echo ""
echo "⏰ 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "📁 工作目录: $(pwd)"
echo "🐍 Python版本: $(python --version 2>&1)"
echo ""
echo "======================================================================================================"
echo "                                    实验配置信息                                                      "
echo "======================================================================================================"
echo ""
echo "📊 数据集:"
echo "   • Uci (UCI信用评分数据集)"
echo "   • Xinwang (新旺信用数据集)"
echo ""
echo "🔀 异质性类型 (4种):"
echo "   • feature   (特征异质性)"
echo "   • label     (标签异质性)"
echo "   • quantity  (样本数量异质性)"
echo "   • iid       (IID均匀分布)"
echo ""
echo "======================================================================================================"
echo "                              消融实验配置 (27个配置)                                                 "
echo "======================================================================================================"
echo ""
echo "📌 图5: 组件消融实验 (6个配置)"
echo "   • Full_Model              - 完整模型（基准）"
echo "   • No_VAE_Generation       - 无VAE生成数据（测试生成数据作用）"
echo "   • No_Prototype            - 无原型学习（测试原型作用）"
echo "   • Phase2_FedGwo           - Phase 2使用FedGwo"
echo "   • Phase2_FedPso           - Phase 2使用FedPso"
echo "   • Phase2_FedAvg           - Phase 2使用FedAvg（无元启发式）"
echo ""
echo "📌 图6: 隐私策略消融实验 (5个配置)"
echo "   • Privacy_None            - 无隐私保护（基准）"
echo "   • Privacy_First           - 隐私优先（重要特征加更多噪声）"
echo "   • Utility_First           - 效用优先（重要特征加更少噪声）"
echo "   • Balanced_Privacy        - 平衡策略"
echo "   • Hybrid_Privacy          - 混合策略（我们的方案）"
echo ""
echo "📌 图7: 泛化能力实验 (2个配置)"
echo "   • Generalization_Reserve_2 - 保留20%客户端（8,9）测试泛化"
echo "   • Generalization_Reserve_3 - 保留30%客户端（7,8,9）测试泛化"
echo ""
echo "📌 图8: 损失权重优化实验 (12个配置)"
echo "   λ_cls（分类损失）:"
echo "     • Lambda_cls_0.5, Lambda_cls_1.0（默认）, Lambda_cls_2.0"
echo "   λ_recon（VAE重建损失）:"
echo "     • Lambda_recon_0.5, Lambda_recon_1.0（默认）, Lambda_recon_2.0"
echo "   λ_kl（KL散度损失）:"
echo "     • Lambda_kl_0.005, Lambda_kl_0.01（默认）, Lambda_kl_0.02"
echo "   λ_proto（原型损失）:"
echo "     • Lambda_proto_0.05, Lambda_proto_0.1（默认）, Lambda_proto_0.2"
echo ""
echo "   💡 总损失公式: L = λ_cls*L_cls + λ_recon*L_recon + λ_kl*L_kl + λ_proto*L_proto"
echo ""
echo "======================================================================================================"
echo "                                    实验规模统计                                                      "
echo "======================================================================================================"
echo ""
echo "📈 消融配置: 27个"
echo "📊 数据集: 2个 (Uci, Xinwang)"
echo "🔀 异构度: 4种 (feature, label, quantity, iid)"
echo "🔢 每组重复: 3次"
echo ""
echo "📋 总实验组数: 27配置 × 2数据集 × 4异构度 = 216组"
echo "📄 总结果文件: 216组 × 3次 = 648个 h5文件"
echo ""
echo "🖥️ 硬件配置:"
echo "   • GPU: 自动检测所有可用GPU"
echo "   • 并发策略: 每GPU运行2个实验（2个槽位）"
echo ""
echo "⚙️ 训练参数:"
echo "   • 全局轮数: 100轮"
echo "   • 本地训练轮数: 5轮"
echo "   • 学习率: Uci=0.005, Xinwang=0.006"
echo "   • 批量大小: Uci=64, Xinwang=128"
echo ""
echo "======================================================================================================"
echo ""

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ 错误: 未找到Python解释器"
    exit 1
fi

# 检查脚本文件
if [ ! -f "run_ablation_experiments.py" ]; then
    echo "❌ 错误: 找不到 run_ablation_experiments.py 文件"
    exit 1
fi

# 创建必要目录
mkdir -p logs
mkdir -p system/results
mkdir -p system/models

# 后台运行消融实验
echo "🚀 正在启动消融实验..."
echo ""
nohup python -u run_ablation_experiments.py > nohup_ablation.out 2>&1 &

# 获取进程ID
PID=$!

# 等待进程启动
sleep 2

# 检查进程是否成功启动
if ps -p $PID > /dev/null; then
    echo "✅ 消融实验已成功在后台启动！"
else
    echo "❌ 进程启动失败，请检查 nohup_ablation.out"
    exit 1
fi

echo ""
echo "======================================================================================================"
echo "                                    运行状态信息                                                      "
echo "======================================================================================================"
echo ""
echo "🆔 进程ID (PID): $PID"
echo "📄 标准输出文件: nohup_ablation.out"
echo "📂 详细日志目录: logs/Ablation_*.log"
echo "💾 结果保存位置: system/results/{Dataset}_FedGpro-FedGwo_{Heterogeneity}/"
echo ""
echo "======================================================================================================"
echo "                                    实用监控命令                                                      "
echo "======================================================================================================"
echo ""
echo "📊 查看实时总体进度:"
echo "   tail -f nohup_ablation.out"
echo ""
echo "🔍 查看单个实验详细日志:"
echo "   tail -f logs/Ablation_Full_Model_Uci_feature.log"
echo "   tail -f logs/Ablation_Lambda_cls_1.0_Xinwang_label.log"
echo "   tail -f logs/Ablation_Privacy_First_Uci_quantity.log"
echo ""
echo "📁 查看所有消融日志文件:"
echo "   ls -lh logs/Ablation_*.log"
echo "   ls logs/Ablation_*.log | wc -l    # 统计日志文件数量"
echo ""
echo "🔎 查看进程状态:"
echo "   ps aux | grep run_ablation_experiments.py"
echo "   ps -p $PID -o pid,etime,%cpu,%mem,cmd    # 详细信息"
echo ""
echo "🖥️ 查看GPU使用情况:"
echo "   watch -n 2 nvidia-smi"
echo "   nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv"
echo ""
echo "📈 查看结果文件生成进度:"
echo "   find system/results -name 'Ablation_*_test_*.h5' | wc -l"
echo "   ls system/results/*/Ablation_*.h5 | wc -l"
echo ""
echo "🔍 按配置查看结果:"
echo "   ls system/results/*/Ablation_Full_Model_*.h5"
echo "   ls system/results/*/Ablation_Lambda_cls_*.h5"
echo "   ls system/results/*/Ablation_Privacy_*.h5"
echo ""
echo "⏹️ 停止进程:"
echo "   kill $PID              # 正常终止"
echo "   kill -9 $PID           # 强制终止"
echo "   pkill -f run_ablation_experiments.py    # 按名称终止"
echo ""
echo "======================================================================================================"
echo "                                文件命名规则说明                                                      "
echo "======================================================================================================"
echo ""
echo "📂 结果目录结构:"
echo "   system/results/{Dataset}_FedGpro-FedGwo_{Heterogeneity}/"
echo "   例如: system/results/Uci_FedGpro-FedGwo_feature/"
echo ""
echo "📄 结果文件命名:"
echo "   Ablation_{ConfigName}_{Dataset}_{Heterogeneity}_test_{0-4}.h5"
echo ""
echo "📝 文件名示例:"
echo "   • Ablation_Full_Model_Uci_feature_test_0.h5"
echo "   • Ablation_Full_Model_Uci_feature_test_1.h5"
echo "   • Ablation_Full_Model_Uci_feature_test_2.h5"
echo "   • Ablation_Full_Model_Uci_feature_test_3.h5"
echo "   • Ablation_Full_Model_Uci_feature_test_4.h5"
echo "   • Ablation_No_VAE_Generation_Xinwang_label_test_0.h5"
echo "   • Ablation_Lambda_cls_1.0_Uci_quantity_test_0.h5"
echo "   • Ablation_Privacy_First_Xinwang_iid_test_1.h5"
echo ""
echo "✅ 完成判定逻辑:"
echo "   每个实验组需要5个文件（*_test_0.h5, *_test_1.h5, *_test_2.h5, *_test_3.h5, *_test_4.h5）"
echo "   • 0-4个文件 → 不完整，会重新执行5次（覆盖）"
echo "   • 5个文件 → ✅ 已完成，跳过"
echo ""
echo "======================================================================================================"
echo ""
echo "⏳ 实验运行中，请勿关闭终端..."
echo "⏱️ 预计总耗时: 根据GPU性能和数据集大小，约需 5-12 小时"
echo ""
echo "💡 提示: 可以安全关闭终端，实验会继续在后台运行"
echo "💡 重新连接后使用 'tail -f nohup_ablation.out' 查看进度"
echo ""
echo "======================================================================================================"
echo ""

echo "✅ 消融实验已在后台启动!"
echo ""
echo "进程ID (PID): $PID"
echo "标准输出: $OUTPUT_FILE"
echo "详细日志: logs/ablation_${EXPERIMENT_GROUP}_*.log"
echo ""
echo "======================================================================================================"
echo "实用命令:"
echo "======================================================================================================"
echo "# 查看实时输出（控制台日志）"
echo "tail -f $OUTPUT_FILE"
echo ""
echo "# 查看详细日志（包含看板信息）"
echo "tail -f logs/ablation_${EXPERIMENT_GROUP}_*.log"
echo ""
echo "# 查看进程状态"
echo "ps aux | grep run_ablation_experiments.py"
echo ""
echo "# 查看GPU使用情况"
echo "watch -n 2 nvidia-smi"
echo ""
echo "# 终止进程"
echo "kill $PID"
echo "# 或强制终止: kill -9 $PID"
echo "======================================================================================================"
echo ""
echo "实验运行中，请勿关闭终端..."
echo "预计总耗时: 根据实验数量和模型复杂度，约需数小时至十几小时"
echo ""
