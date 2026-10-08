"""用例集与硬门规则的一致性审计。

技术报告定义 6 规定：T4–T5 需 `Δ ≤ 30`，T6–T7 需 `Δ ≤ 7`，T0–T3 无卡龄窗口。
用例文件把 `candidates` 声明为「应被召回的候选」，因此其中每一张卡都应满足其层级的窗口。
本脚本检查这一点，并列出违规项。
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow.rank_eval import load_holdout_cases, load_rank_cases  # noqa: E402
from moonshadow.scoring import FRESH_WINDOW, RECENT_WINDOW  # noqa: E402

WINDOWS = {
    "T0": None, "T1": None, "T2": None, "T3": None,
    "T4": RECENT_WINDOW, "T5": RECENT_WINDOW,
    "T6": FRESH_WINDOW, "T7": FRESH_WINDOW,
    "T8": None, "T9": None,
}


def audit(name: str, cases) -> list[str]:
    violations: list[str] = []
    for case in cases:
        for candidate in case.candidates:
            window = WINDOWS[candidate.tier]
            if window is not None and candidate.age_days > window:
                violations.append(
                    f"{case.id:<42} {candidate.id:<26} {candidate.tier} "
                    f"Δ={candidate.age_days:g} > 窗口 {window:g}"
                )
            # 门前置：T8/T9 只在显式回溯时允许进入候选集（定义 6）
            if candidate.tier in ("T8", "T9") and not case.explicit_recall:
                violations.append(
                    f"{case.id:<42} {candidate.id:<26} {candidate.tier} "
                    f"作为正例但未声明 explicit_recall"
                )
    print(f"\n{name}：{len(cases)} 条用例，{sum(len(c.candidates) for c in cases)} 张正例候选")
    if violations:
        print(f"  违规：{len(violations)} 张")
        for line in violations:
            print(f"    ✗ {line}")
    else:
        print("  无违规（卡龄窗口与 T8/T9 门前置均满足）")
    return violations


def main() -> int:
    dev = audit("开发集 eval/rank", load_rank_cases())
    held = audit("留出集 eval/holdout", load_holdout_cases())
    total = len(dev) + len(held)
    print(f"\n合计违规 {total} 张。")
    if total:
        print("含义：这些候选按定义 6 不会被硬门放行，因此相应用例的排序期望")
        print("在真实流水线中不会被执行（负例与正例都会先被窗口挡掉）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
