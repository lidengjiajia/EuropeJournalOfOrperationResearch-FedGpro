#!/bin/bash
# ===================================================================================
# FedGpro 基线实验批量运行脚本（nohup后台运行）
# ===================================================================================
# 
# 功能说明:
#   - 运行128个基线对比实验（16算法 × 2数据集 × 4异构度）
#   - 每个实验重复5次（-t 5参数）
#   - 自动检测GPU并并发运行（每GPU 2个槽位）
#   - 智能跳过已完成实验（已有5个结果文件）
# 
# 使用方法:
#   bash run_baseline_nohup.sh
# 
# 查看实时日志:
#   tail -f nohup_baseline.out                      # 总体进度
#   tail -f logs/Uci_FedAvg_feature.log             # 单个实验详细日志
# 
# 结果文件位置:
#   system/results/{Dataset}_{Algorithm}_{Heterogeneity}/
#   └── {Dataset}_{Algorithm}_{Heterogeneity}_test_0.h5
#   └── {Dataset}_{Algorithm}_{Heterogeneity}_test_1.h5
#   ├── {Dataset}_{Algorithm}_{Heterogeneity}_test_2.h5
#   ├── {Dataset}_{Algorithm}_{Heterogeneity}_test_3.h5
#   └── {Dataset}_{Algorithm}_{Heterogeneity}_test_4.h5
# ===================================================================================

echo ""
echo "======================================================================================================"
echo "                       FedGpro 基线对比实验 - 后台运行启动                                           "
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
echo "🤖 算法列表 (16个):"
echo "   基础算法: FedAvg, FedProx, FedScaffold, FedMoon, FedGen"
echo "   个性化: Per-FedAvg, FedDitto, FedRep, FedProto"
echo "   元启发式: FedPso, FedGwo"
echo "   FedGpro变体: FedGpro-FedAvg, FedGpro-FedProx, FedGpro-FedScaffold,"
echo "                FedGpro-FedGwo, FedGpro-FedPso"
echo ""
echo "🔀 异质性类型 (4种):"
echo "   • feature   (特征异质性)"
echo "   • label     (标签异质性)"
echo "   • quantity  (样本数量异质性)"
echo "   • iid       (IID均匀分布)"
echo ""
echo "📈 实验规模:"
echo "   • 总实验组数: 128 (16算法 × 2数据集 × 4异构度)"
echo "   • 每组重复次数: 5次"
echo "   • 总结果文件数: 640个 h5文件"
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
if [ ! -f "run_all_experiments.py" ]; then
    echo "❌ 错误: 找不到 run_all_experiments.py 文件"
    exit 1
fi

# 创建必要目录
mkdir -p logs
mkdir -p system/results
mkdir -p system/models

# 后台运行基线实验
echo "🚀 正在启动基线实验..."
echo ""
nohup python -u run_all_experiments.py > nohup_baseline.out 2>&1 &

# 获取进程ID
PID=$!

# 等待进程启动
sleep 2

# 检查进程是否成功启动
if ps -p $PID > /dev/null; then
    echo "✅ 基线实验已成功在后台启动！"
else
    echo "❌ 进程启动失败，请检查 nohup_baseline.out"
    exit 1
fi

echo ""
echo "======================================================================================================"
echo "                                    运行状态信息                                                      "
echo "======================================================================================================"
echo ""
echo "🆔 进程ID (PID): $PID"
echo "📄 标准输出文件: nohup_baseline.out"
echo "📂 详细日志目录: logs/"
echo "💾 结果保存位置: system/results/"
echo ""
echo "======================================================================================================"
echo "                                    实用监控命令                                                      "
echo "======================================================================================================"
echo ""
echo "📊 查看实时总体进度:"
echo "   tail -f nohup_baseline.out"
echo ""
echo "🔍 查看单个实验详细日志:"
echo "   tail -f logs/Uci_FedAvg_feature.log"
echo "   tail -f logs/Xinwang_FedGpro-FedGwo_label.log"
echo ""
echo "📁 查看所有日志文件列表:"
echo "   ls -lh logs/"
echo "   ls logs/*.log | wc -l    # 统计日志文件数量"
echo ""
echo "🔎 查看进程状态:"
echo "   ps aux | grep run_all_experiments.py"
echo "   ps -p $PID -o pid,etime,%cpu,%mem,cmd    # 详细信息"
echo ""
echo "🖥️ 查看GPU使用情况:"
echo "   watch -n 2 nvidia-smi"
echo "   nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv"
echo ""
echo "📈 查看结果文件生成进度:"
echo "   find system/results -name '*_test_*.h5' | wc -l"
echo "   ls -lh system/results/*/  # 查看所有结果目录"
echo ""
echo "⏹️ 停止进程:"
echo "   kill $PID              # 正常终止"
echo "   kill -9 $PID           # 强制终止"
echo "   pkill -f run_all_experiments.py    # 按名称终止"
echo ""
echo "======================================================================================================"
echo ""
echo "⏳ 实验运行中，请勿关闭终端..."
echo "⏱️ 预计总耗时: 根据GPU性能和数据集大小，约需 3-8 小时"
echo ""
echo "💡 提示: 可以安全关闭终端，实验会继续在后台运行"
echo "💡 重新连接后使用 'tail -f nohup_baseline.out' 查看进度"
echo ""
echo "======================================================================================================"
echo ""
