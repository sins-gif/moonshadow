"""留出验证：开发集上得出的 floor 结论是否为过拟合。

    python tools/run_holdout.py

开发集（`eval/rank`，11 条）用来提出假设，留出集（`eval/holdout`，10 条）用来检验假设。
留出用例在跑之前就已写完，且期望顺序由一条事前定好的原则（相关性差距 vs 新旧差距）推导，
与 floor 取值无关——避免又在测试集上调参。
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow.rank_eval import (  # noqa: E402
    compare_policies,
    cross_validate_windows,
    evaluate_intent,
    load_holdout_cases,
    load_rank_cases,
)


def main() -> int:
    development = load_rank_cases()
    holdout = load_holdout_cases()
    print(f"开发集 {len(development)} 条 / 留出集 {len(holdout)} 条\n")
    print(compare_policies(development, holdout))
    print()
    print(cross_validate_windows(development, holdout))
    print()
    print("留出集上的时效性识别（真值标签为 needs_fresh，无歧义）：")
    print(evaluate_intent(holdout).summary())
    # 留出验证产出的是证据而非关卡：结论由人看完数据后下。
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
