"""附录 A.3 / A.4 的数值实例，原封不动落成单元测试。

写法刻意与 `docs/v1.1-weights.md` 保持一致：用例用紧凑字典表示（`tier / i / delta / s / mu`），
断言里直接写附录里的六位小数。

紧凑字典 → 真实入参的翻译由 `_card()` 完成，并且**翻译结果会被反向校验**：
附录里的 `μ`（实体 / 任务匹配度）在产品代码里由「实体 Jaccard」与「是否命中未完成事项」算出，
若构造方式与附录不符，`_assert_match()` 立即失败——否则测试会在悄悄验证另一个东西。
"""

from __future__ import annotations

import pathlib
import sys
import unittest
from datetime import timedelta

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow.rank_eval import NOW  # noqa: E402
from moonshadow.scoring import (  # noqa: E402
    RECENT_WINDOW,
    SCHEME_ADDITIVE,
    SCHEME_MULTIPLICATIVE,
    SCHEME_RELEVANCE,
    TIME_FLOOR,
    match_score,
    score_card,
    time_modifier,
)
from moonshadow.tiers import half_life_for  # noqa: E402

#: 占位实体名。真实代码按实体集合的 Jaccard 计算 μ，这里用单个实体构造出附录要求的 μ 值。
PROBE = "E"


def _card(case: dict) -> dict:
    """把附录的紧凑字典翻译成打分器认识的卡结构。

    映射关系：`s → query_cos`、`i → importance`、`delta → age_days`、`tier → half_life_days`。
    `mu` 不能直接传入，须由实体集合与 task_hit 构造。
    """
    if case["mu"] == 0.5:
        entities: tuple[str, ...] = (PROBE,)      # Jaccard = 1，且无未完成事项 ⇒ μ = 0.5
    elif case["mu"] == 0.0:
        entities = ()                              # Jaccard = 0 ⇒ μ = 0
    else:
        raise ValueError(f"本测试只支持附录用到的 μ 取值 0 与 0.5，收到 {case['mu']}")
    return {
        "id": case.get("id", "unit"),
        "tier": case["tier"],
        "importance": case["i"],
        "entities": list(entities),
        "observed_at": (NOW - timedelta(days=case["delta"])).isoformat(timespec="seconds"),
        "half_life_days": half_life_for(case["tier"]),
        "status": "active",
    }


def _query_entities(case: dict) -> list[str]:
    return [PROBE]


def _assert_match(test: unittest.TestCase, case: dict) -> None:
    """反向校验：代码算出的 μ 必须等于附录里写的 μ。"""
    card = _card(case)
    actual = match_score(_query_entities(case), card["entities"], False)
    test.assertAlmostEqual(actual, case["mu"], places=12, msg=f"{case.get('id')} 的 μ 构造有误")


def calculate_additive(case: dict) -> float:
    """定义 4（加性融合）：`a = 0.45σ + 0.25δ + 0.20(i/10) + 0.10μ`。"""
    return score_card(
        _card(case),
        now=NOW,
        query_cos=case["s"],
        query_entities=_query_entities(case),
        task_hit=False,
        scheme=SCHEME_ADDITIVE,
    )


def breakdown(case: dict, floor: float = TIME_FLOOR) -> dict[str, float]:
    """定义 1–3 的逐步分解：返回 `r`（相关性）、`m`（修正）、`q`（分数）。

    衰减因子 `δ` 不在这里返回——它由 `_decay_factor()` 从 `m` 反解，
    以避免在产品代码之外另写一份衰减公式。
    """
    card = _card(case)
    kwargs = {
        "now": NOW,
        "query_cos": case["s"],
        "query_entities": _query_entities(case),
        "task_hit": False,
    }
    relevance = score_card(card, scheme=SCHEME_RELEVANCE, **kwargs)
    modifier = time_modifier(case["delta"], card["half_life_days"], floor=floor)
    return {"r": relevance, "m": modifier, "q": relevance * modifier}


# --------------------------------------------------------------------------- 附录 A.3

#: 附录 A.3 的反例。**两条候选都必须满足自身层级的卡龄窗口**（定义 6）：
#: 旧版见证 `P = (T5, Δ=120)` 超窗（T5 窗口 30 天），窗口本身就会挡掉它，
#: 因此它不是 θ 的证据——用超窗卡证明「θ 不可行」是无效的。
#: 现行见证取自用例集：`N` 来自 `09-gate-noise-and-window`（用例自己声明它只该被 θ 挡住），
#: `P` 来自 `17-explicit-recall-admits-t8`。两条分属不同用例——这比同用例更强，
#: 因为 θ 是**全局**阈值。
NEGATIVE = {"id": "N(weak-t3)", "tier": "T3", "i": 2, "delta": 10, "s": 0.15, "mu": 0.0}
POSITIVE = {"id": "P(paraphrase)", "tier": "T5", "i": 3, "delta": 30, "s": 0.72, "mu": 0.0}


class AppendixA3Test(unittest.TestCase):
    def test_witness_cards_exist_in_the_case_set_and_are_window_legal(self) -> None:
        """见证对必须是**真实用例里的、窗口合法的**正负例，否则定理 5 的证明无效。

        切分重划后两条分属**不同集合**（`09` 在开发集、`17` 在留出集）——这没有关系，
        因为 `θ` 是**全局**阈值：召回 P 与排除 N 用的是同一个 `θ`。
        """
        from moonshadow.rank_eval import load_holdout_cases, load_rank_cases

        cases = {case.id: case for case in load_rank_cases() + load_holdout_cases()}
        n_case = cases["09-gate-noise-and-window"]
        p_case = cases["17-explicit-recall-admits-t8"]
        n_card = next(k for k in n_case.exclude if k.id == "weak-t3")
        p_card = next(k for k in p_case.candidates if k.id == "paraphrase")

        self.assertEqual(n_card.reason, "score", "N 必须只能被 θ 挡住，否则它不是 θ 的证据")
        self.assertIn(p_card.id, p_case.expect_order, "P 必须在期望次序里（即必须被召回）")
        self.assertEqual(n_card.tier, "T3", "T3 无卡龄窗口，N 不受窗口约束")
        self.assertLessEqual(p_card.age_days, RECENT_WINDOW, "T5 的卡龄必须在窗口内")

        self.assertEqual(
            (n_card.tier, n_card.importance, n_card.age_days, n_card.cos),
            (NEGATIVE["tier"], NEGATIVE["i"], NEGATIVE["delta"], NEGATIVE["s"]),
        )
        self.assertEqual(
            (p_card.tier, p_card.importance, p_card.age_days, p_card.cos),
            (POSITIVE["tier"], POSITIVE["i"], POSITIVE["delta"], POSITIVE["s"]),
        )

    def test_theorem_5(self) -> None:
        """定理 5 的反例：加性打分下负例分数高于正例，故不存在可分离阈值。"""
        for case in (NEGATIVE, POSITIVE):
            _assert_match(self, case)

        a_N = calculate_additive(NEGATIVE)
        a_P = calculate_additive(POSITIVE)

        self.assertAlmostEqual(a_N, 0.530219, places=6)
        self.assertAlmostEqual(a_P, 0.503608, places=6)
        self.assertGreater(a_N, a_P)

        # 可分离性等价于区间 (max a_N, min a_P) 非空；此处为空集。
        lower, upper = max(a_N, a_P), min(a_N, a_P)
        self.assertGreater(lower, upper, "区间必须为空，否则定理 5 不成立")

        print(f"定理5 反例验证通过：a_N = {a_N:.6f}，a_P = {a_P:.6f}，差值 {a_N - a_P:+.6f}")
        print(f"  召回 P 要求 θ < {a_P:.6f}；排除 N 要求 θ > {a_N:.6f} —— 两式互斥")

    def test_theorem_5_threshold_search_finds_nothing(self) -> None:
        """穷举阈值：在 [0,1] 上以 0.0001 为步长，不存在同时满足两个要求的 θ。"""
        a_N = calculate_additive(NEGATIVE)
        a_P = calculate_additive(POSITIVE)
        feasible = [
            step / 10000
            for step in range(10001)
            if (step / 10000) < a_P and (step / 10000) > a_N
        ]
        self.assertEqual(feasible, [])
        print(f"  阈值穷举：在 [0,1] 上以 0.0001 为步长扫描 10001 个取值，可行解 {len(feasible)} 个")


# --------------------------------------------------------------------------- 附录 A.4

#: 附录 A.4 的算例：查询「缓存层换用什么方案了？」，两个候选均命中查询实体，故 μ = 0.5。
#: 输入经修订：让新决定的重要度高于旧文档（i = 4 对 5）。
CANDIDATE_A = {"id": "A(旧方案文档)", "tier": "T4", "i": 4, "delta": 55, "s": 0.95, "mu": 0.5}
CANDIDATE_B = {"id": "B(新方案决定)", "tier": "T5", "i": 5, "delta": 1.5, "s": 0.60, "mu": 0.5}

#: 附录 A.4 表格里的字面值，逐格对照。**两组 floor**：
#: `TIME_FLOOR`（现行采纳值，见 `scoring.py`）与 `0.62`（`experiments/user_a4_candidate_*.py`
#: 硬编码的值，那两个脚本不得修改，因此附录保留这一列以便与其逐格对齐）。
#: `r` 与 `δ` 与 floor 无关，`m`/`q` 随 floor 变——这正是「换下限只动时间修正」的体现。
A4_TABLE = {
    0.5925: {
        "A(旧方案文档)": {"r": 0.758000, "delta_factor": 0.280616, "m": 0.706851, "q": 0.535793},
        "B(新方案决定)": {"r": 0.680000, "delta_factor": 0.928425, "m": 0.970833, "q": 0.660167},
    },
    0.62: {
        "A(旧方案文档)": {"r": 0.758000, "delta_factor": 0.280616, "m": 0.726634, "q": 0.550788},
        "B(新方案决定)": {"r": 0.680000, "delta_factor": 0.928425, "m": 0.972801, "q": 0.661505},
    },
}

#: 各 floor 下 `m_B/m_A` 与**比值余量**（`m_B/m_A − r_A/r_B`，`r_A/r_B = 1.114706` 与 floor 无关）。
#: 注意区分两个不同的「余量」：比值余量（判定式两侧之差）与**分数余量**（`q_B − q_A`）。
#: `f=0.5925` 时后者是 `0.124374`，前者是 `0.258757`——两者不可互换引用。
A4_RATIOS = {0.5925: (1.373463, 0.258757), 0.62: (1.338778, 0.224072)}

#: 两个 floor 下的分数余量 `q_B − q_A`。
A4_SCORE_MARGIN = {0.5925: 0.124374, 0.62: 0.110717}


def _decay_factor(case: dict, floor: float = TIME_FLOOR) -> float:
    """定义 2：由修正系数反解衰减因子 `δ = (m − f) / (1 − f)`，避免另写一份公式。"""
    modifier = time_modifier(case["delta"], half_life_for(case["tier"]), floor=floor)
    return (modifier - floor) / (1.0 - floor)


class AppendixA4Test(unittest.TestCase):
    def test_appendix_a4_multiplicative_table(self) -> None:
        """附录 A.4：两列 floor 下的逐步分解，逐格对照六个小数。"""
        for floor, table in A4_TABLE.items():
            for case in (CANDIDATE_A, CANDIDATE_B):
                _assert_match(self, case)
                row = table[case["id"]]
                stats = breakdown(case, floor=floor)
                self.assertAlmostEqual(stats["r"], row["r"], places=6, msg=f"floor={floor}")
                self.assertAlmostEqual(
                    _decay_factor(case, floor=floor), row["delta_factor"], places=6
                )
                self.assertAlmostEqual(stats["m"], row["m"], places=6, msg=f"floor={floor}")
                self.assertAlmostEqual(stats["q"], row["q"], places=6, msg=f"floor={floor}")

            stats_a, stats_b = breakdown(CANDIDATE_A, floor=floor), breakdown(CANDIDATE_B, floor=floor)
            ratio_m = stats_b["m"] / stats_a["m"]
            ratio_r = stats_a["r"] / stats_b["r"]
            expected_m, expected_margin = A4_RATIOS[floor]
            self.assertAlmostEqual(ratio_m, expected_m, places=6, msg=f"floor={floor}")
            self.assertAlmostEqual(ratio_r, 1.114706, places=6)
            self.assertGreater(ratio_m, ratio_r)
            self.assertGreater(stats_b["q"], stats_a["q"], "定理 3 判定：B 胜出")
            self.assertAlmostEqual(ratio_m - ratio_r, expected_margin, places=6)
            self.assertAlmostEqual(
                stats_b["q"] - stats_a["q"], A4_SCORE_MARGIN[floor], places=6,
                msg="分数余量与比值余量是两个量，须分别核对",
            )

        # `r` 与 `δ` 与 floor 无关——换下限只动 m/q。
        for case in (CANDIDATE_A, CANDIDATE_B):
            self.assertAlmostEqual(
                breakdown(case, floor=TIME_FLOOR)["r"], breakdown(case, floor=0.62)["r"], places=12
            )
            self.assertAlmostEqual(
                _decay_factor(case, floor=TIME_FLOOR), _decay_factor(case, floor=0.62), places=12
            )

        print(f"\n附录 A.4（现行 f = {TIME_FLOOR}，附 f = 0.62 对照）：")
        print(f"  {'候选':<16}{'r':>10}{'δ':>10}{'m':>10}{'q':>10}")
        for case in (CANDIDATE_A, CANDIDATE_B):
            stats = breakdown(case)
            print(
                f"  {case['id']:<16}{stats['r']:>10.6f}{_decay_factor(case):>10.6f}"
                f"{stats['m']:>10.6f}{stats['q']:>10.6f}"
            )
        print(
            f"  定理 3 判定式（f={TIME_FLOOR}）：m_B/m_A = 1.373463 > r_A/r_B = 1.114706 ⇒ B 胜出"
        )
        print("  比值余量 = +0.258757；分数余量 q_B−q_A = +0.124374")
        print("  同一对候选在 f=0.62 下：比值余量 +0.224072，分数余量 +0.110717")

    def test_appendix_a4_critical_floor(self) -> None:
        """由 `m_B/m_A = r_A/r_B` 解出这对候选的**翻转点** `f*`。

        比「试一个 `0.70` 看是否反转」更精确：它给出的是次序反转的临界 floor。
        修订输入（`i = 4/5`）后，`f = 0.70` 不再反转这对候选；翻转需要 `f > 0.842939`。
        """
        card_a, card_b = _card(CANDIDATE_A), _card(CANDIDATE_B)
        stats_a, stats_b = breakdown(CANDIDATE_A), breakdown(CANDIDATE_B)
        ratio_r = stats_a["r"] / stats_b["r"]

        # f=0.70 不再反转：这是输入修订带来的实质变化，必须显式记录
        m_a70 = time_modifier(CANDIDATE_A["delta"], card_a["half_life_days"], floor=0.70)
        m_b70 = time_modifier(CANDIDATE_B["delta"], card_b["half_life_days"], floor=0.70)
        self.assertGreater(m_b70 / m_a70, ratio_r, "f=0.70 时仍是 B 胜出")

        # 解析解：m = δ + (1−δ) f，解 (δ_B + (1−δ_B) f) / (δ_A + (1−δ_A) f) = r_A/r_B
        delta_a = _decay_factor(CANDIDATE_A)
        delta_b = _decay_factor(CANDIDATE_B)
        f_star = (delta_b - ratio_r * delta_a) / (ratio_r * (1 - delta_a) - (1 - delta_b))
        self.assertAlmostEqual(f_star, 0.842939, places=6)

        # 验证 f* 处两侧比值相等
        self.assertAlmostEqual(
            (delta_b + (1 - delta_b) * f_star) / (delta_a + (1 - delta_a) * f_star),
            ratio_r, places=9,
        )
        # 且 f* 高于两个被讨论过的取值
        self.assertGreater(f_star, 0.70)
        self.assertGreater(f_star, TIME_FLOOR)

        print("\n附录 A.4 的临界 floor：")
        print(f"  r_A/r_B = {ratio_r:.6f}   m_B/m_A(f=0.62) = {stats_b['m'] / stats_a['m']:.6f}")
        print(f"  f=0.70：m_B/m_A = {m_b70 / m_a70:.6f} > {ratio_r:.6f} ⇒ 仍为 B 胜出（不发生反转）")
        print(f"  翻转点 f* = {f_star:.6f}（floor 超过它，这对候选的次序才反转）")


if __name__ == "__main__":
    unittest.main(verbosity=2)
