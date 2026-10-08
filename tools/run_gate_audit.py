"""硬门一致性审计：每个负例都必须真的被门挡住，每个正例都必须真的过门（v1.2 决策 4）。

    python tools/run_gate_audit.py

**为什么需要这个工具**：用例文件里的 `exclude` 此前只被 `θ` 标定读过，而且只读
`reason="score"` 的那些——`reason="window"` 与 `reason="tier"` 的负例**从来没有被任何代码验证过**。
于是「这条噪声该被层级规则挡掉」只是注释，不是断言。

本工具用**生产门**（`scoring.passes_gate`）逐张验证，用的正是生产配置：
采纳方案 `ADOPTED_SCHEME`、采纳下限 `TIME_FLOOR`、标定阈值 `THETA` / `THETA_HIGH`。

- 正例（`candidates`）必须 `passes_gate == True`；
- 负例（`exclude`）必须 `passes_gate == False`，且**挡掉它的理由要与文件声明一致**：
  `window` 声明必须由卡龄窗口挡住（把分数设成 1.0 也应被挡），
  `tier` 声明必须由层级规则挡住（同样与分数无关），
  `score` 声明则由 `θ` 挡住。

退出码：无违规为 `0`，否则为 `1`。
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow.rank_eval import (  # noqa: E402
    load_holdout_cases,
    load_rank_cases,
    score_of,
)
from moonshadow.scoring import (  # noqa: E402
    ADOPTED_SCHEME,
    THETA,
    THETA_HIGH,
    TIME_FLOOR,
    passes_gate,
)


def blocked_by_rule_alone(candidate, *, score: float) -> bool:
    """把分数设成满分后是否仍被挡住——若是，挡它的就不是分数。"""
    return not passes_gate(
        candidate.tier,
        delta_days=candidate.age_days,
        score=score,
        explicit_recall=False,
        theta=THETA,
        theta_high=THETA_HIGH,
    )


def audit(name: str, cases) -> list[str]:
    violations: list[str] = []
    for case in cases:
        for candidate in case.candidates:
            value = score_of(case, candidate, ADOPTED_SCHEME)
            if not passes_gate(
                candidate.tier,
                delta_days=candidate.age_days,
                score=value,
                explicit_recall=case.explicit_recall,
                theta=THETA,
                theta_high=THETA_HIGH,
            ):
                violations.append(
                    f"{case.id:<44} 正例 {candidate.id:<22} {candidate.tier} "
                    f"未过门（q={value:.4f} Δ={candidate.age_days:g}）"
                )
        for candidate in case.exclude:
            value = score_of(case, candidate, ADOPTED_SCHEME)
            passed = passes_gate(
                candidate.tier,
                delta_days=candidate.age_days,
                score=value,
                explicit_recall=case.explicit_recall,
                theta=THETA,
                theta_high=THETA_HIGH,
            )
            if passed:
                violations.append(
                    f"{case.id:<44} 负例 {candidate.id:<22} {candidate.tier} "
                    f"**没有被挡住**（q={value:.4f} reason={candidate.reason}）"
                )
                continue
            # 声明 reason="window" / "tier" 的负例，必须与分数无关地被挡住。
            if candidate.reason in ("window", "tier") and not blocked_by_rule_alone(
                candidate, score=1.0
            ):
                violations.append(
                    f"{case.id:<44} 负例 {candidate.id:<22} 声明 reason={candidate.reason}，"
                    f"但它在 q=1.0 时能被放行——挡住它的其实是分数"
                )
    print(f"\n{name}：{len(cases)} 条用例，"
          f"{sum(len(c.candidates) for c in cases)} 正例 / "
          f"{sum(len(c.exclude) for c in cases)} 负例")
    if violations:
        print(f"  违规：{len(violations)} 张")
        for line in violations:
            print(f"    ✗ {line}")
    else:
        print("  无违规（正例全部过门，负例全部被挡且理由与声明一致）")
    return violations


def main() -> int:
    print(f"生产配置：方案={ADOPTED_SCHEME}  f={TIME_FLOOR}  θ={THETA}  θ′={THETA_HIGH}")
    dev = audit("开发集 eval/rank", load_rank_cases())
    held = audit("留出集 eval/holdout", load_holdout_cases())
    total = len(dev) + len(held)
    print(f"\n合计违规 {total} 张。")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
