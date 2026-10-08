"""network 正则诊断：分离「schema 难」与「分类器弱」。

    python tools/run_network_eval.py

三组数：
1. `v1` 基线规则（3 分支、依赖 tier、永不输出 observation）
2. `v2` 改进规则（4 类词汇、不依赖 tier），只在 dev 上改，holdout 只跑一次
3. **schema 上限** = 1 − 无正确类别的用例占比（与分类器无关）
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow.network_eval import (  # noqa: E402
    compare_network_rules,
    evaluate_network,
    format_confusion,
    load_network_cases,
    rule_variants,
)

if __name__ == "__main__":
    print("规则对照（dev 24 条 / holdout 24 条）")
    print(compare_network_rules())

    engines = rule_variants()
    for split in ("dev", "holdout"):
        for name, engine in engines.items():
            report = evaluate_network(load_network_cases(split=split), engine)
            print(f"\n=== {split} · {name} ===")
            print(format_confusion(report))
            print(report.summary())
