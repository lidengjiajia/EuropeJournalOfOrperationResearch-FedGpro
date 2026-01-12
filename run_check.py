#!/usr/bin/env python
"""运行check_missing_experiments并显示结果"""

import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path.cwd()))

from run_ablation_experiments import check_missing_experiments

# 运行检查
missing = check_missing_experiments()

print("\n" + "="*100)
if missing:
    print(f"总共需要补充的实验: {len(missing)}")
    print("=" * 100)
else:
    print("✅ 所有实验均已完成！")
    print("=" * 100)
