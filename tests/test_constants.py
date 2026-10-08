"""附录 A 的守卫：常量取值与定理。

两件事：
1. **常量守卫**——把 `v1.1-weights.md` 附录 A.1 里出现的每个字面值钉住。
   文档里已经出现过「留出集验证了 0.62」这类错话，而**数字漂移比措辞错误更难发现**。
   失败不代表「常量不能改」，而代表「改了就要去改文档」。
2. **定理守卫**——技术报告正文 §4 与附录 A.2–A.4 里以「定理 / 推论 / 数值实例」形式
   给出的结论必须可检验。写进报告的定理若与代码不符，比写错散文更严重。
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
    FRESH_WINDOW,
    HYBRID_ALPHA,
    RECENT_WINDOW,
    RELEVANCE_WEIGHTS,
    RRF_K,
    RRF_TIME_WEIGHT_RANGE,
    SCHEME_ADDITIVE,
    SCHEME_MULTIPLICATIVE,
    SCHEME_RELEVANCE,
    THETA,
    THETA_HIGH,
    TIME_FLOOR,
    TIME_FLOOR_MAX,
    TIME_FLOOR_MIN,
    WEIGHTS,
    score_card,
    time_modifier,
)
from moonshadow.tiers import TIER_HALF_LIFE  # noqa: E402


def card(tier: str, importance: int, age_days: float, half_life: float, entities=()) -> dict:
    return {
        "id": "unit",
        "tier": tier,
        "importance": importance,
        "entities": list(entities),
        "observed_at": (NOW - timedelta(days=age_days)).isoformat(timespec="seconds"),
        "half_life_days": half_life,
        "status": "active",
    }


class ConstantGuardTest(unittest.TestCase):
    def test_scoring_weights_are_frozen(self) -> None:
        self.assertEqual(WEIGHTS, {"sim": 0.45, "decay": 0.25, "importance": 0.20, "match": 0.10})
        self.assertEqual(
            RELEVANCE_WEIGHTS, {"sim": 0.60, "importance": 0.27, "match": 0.13}
        )
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0, places=12)
        self.assertAlmostEqual(sum(RELEVANCE_WEIGHTS.values()), 1.0, places=12)

    def test_time_floor_is_frozen(self) -> None:
        # 0.5925 = 加入 3 条边界用例后，「开发集 ∩ 留出集」可行窗 [0.5225, 0.6600] 内的
        # minimax 取值；它落在采纳网格上（0.5925/0.0025 = 237）。
        # 注意中点**依赖步长**（0.005→0.5925、0.0025→0.5913、0.0005→0.5922），不是精确值。
        self.assertEqual(TIME_FLOOR, 0.5925)
        self.assertEqual(TIME_FLOOR_MIN, 0.35)
        self.assertEqual(TIME_FLOOR_MAX, TIME_FLOOR)

    def test_gate_thresholds_and_windows_are_frozen(self) -> None:
        # θ′ = θ：T6/T7 不再设更高的分位门槛（开发集只有 1 张 T6/T7 正例，撑不起；
        # 且取其中点会被冻结留出集否决）。两类层级的区别只由卡龄窗口表达。
        self.assertEqual(THETA, 0.3941)
        self.assertEqual(THETA_HIGH, THETA)
        self.assertEqual(RECENT_WINDOW, 30.0)
        self.assertEqual(FRESH_WINDOW, 7.0)

    def test_half_lives_are_frozen(self) -> None:
        expected = {
            "T0": 0.0, "T1": 365.0, "T2": 180.0, "T3": 90.0, "T4": 30.0,
            "T5": 14.0, "T6": 7.0, "T7": 3.0, "T8": 1.0, "T9": 1.0 / 24.0,
        }
        self.assertEqual(set(TIER_HALF_LIFE), set(expected))
        for tier, value in expected.items():
            self.assertAlmostEqual(TIER_HALF_LIFE[tier], value, places=12, msg=tier)

    def test_experiment_only_constants_are_frozen(self) -> None:
        self.assertEqual(RRF_K, 60)
        self.assertEqual(RRF_TIME_WEIGHT_RANGE, (0.4, 0.6))
        self.assertEqual(HYBRID_ALPHA, 0.5)

    def test_overturn_bound_matches_the_documented_fractions(self) -> None:
        """附录 A.1 写的 `1/f` 必须与常量一致。`f = 0.5925 = 237/400`，故 `1/f = 400/237`。"""
        self.assertAlmostEqual(1.0 / TIME_FLOOR, 400 / 237, places=12)
        self.assertAlmostEqual(1.0 / 0.70, 10 / 7, places=12)
        self.assertGreater(1.0 / TIME_FLOOR, 1.0 / 0.70)

    def test_modifier_range_matches_the_documented_interval(self) -> None:
        """附录 A.1 写的 `modifier ∈ [0.5925, 1.00]` 必须成立，且两端都取得到。"""
        self.assertAlmostEqual(time_modifier(0.0, 30.0), 1.00, places=12)
        self.assertAlmostEqual(time_modifier(10**9, 30.0), TIME_FLOOR, places=12)
        self.assertEqual(time_modifier(10**9, 0.0), 1.0, "T0 不衰减，与 floor 无关")


class TheoremGuardTest(unittest.TestCase):
    """技术报告 §4 的定理与附录 A.2–A.4 的证明/实例，逐条对照代码。"""

    def test_theorem_1_ranges(self) -> None:
        """定理 1：σ∈[0,1]、r∈[0.027,1]、m∈(f,1]、q∈(0.027f,1]。

        注意 r = 1 需要 μ = 1，而 μ 的一半来自「命中未完成事项」——只命中实体只能得 0.5。
        """
        lowest = card("T5", 1, 0.0, 14.0)
        self.assertAlmostEqual(
            score_card(
                lowest, now=NOW, query_cos=-1.0, query_entities=["nope"],
                task_hit=False, scheme=SCHEME_RELEVANCE,
            ),
            0.027, places=12,
        )
        best = card("T5", 10, 0.0, 14.0, entities=("nope",))
        self.assertAlmostEqual(
            score_card(
                best, now=NOW, query_cos=1.0, query_entities=["nope"],
                task_hit=True, scheme=SCHEME_RELEVANCE,
            ),
            1.0, places=12,
        )
        self.assertAlmostEqual(time_modifier(0.0, 30.0), 1.0, places=12)
        self.assertAlmostEqual(
            time_modifier(1000.0, 30.0),
            TIME_FLOOR + (1 - TIME_FLOOR) * 0.5 ** (1000 / 30),
            places=15,
        )

    def test_theorem_3_reversal_criterion(self) -> None:
        """定理 3：q_B > q_A ⟺ r_A/r_B < m_B/m_A。用附录 A.4 的两个候选核对。"""
        kwargs = {"now": NOW, "query_entities": ["缓存层"], "task_hit": False}
        old = card("T4", 7, 55.0, 30.0, entities=("缓存层",))
        new = card("T5", 4, 1.5, 14.0, entities=("缓存层",))

        r_old = score_card(old, query_cos=0.95, scheme=SCHEME_RELEVANCE, **kwargs)
        r_new = score_card(new, query_cos=0.60, scheme=SCHEME_RELEVANCE, **kwargs)
        q_old = score_card(old, query_cos=0.95, scheme=SCHEME_MULTIPLICATIVE, **kwargs)
        q_new = score_card(new, query_cos=0.60, scheme=SCHEME_MULTIPLICATIVE, **kwargs)

        m_old = time_modifier(55.0, 30.0)
        m_new = time_modifier(1.5, 14.0)
        self.assertGreater(q_new, q_old, "例题 1：B 胜出")
        self.assertLess(r_old / r_new, m_new / m_old, "命题 3 的判定式必须成立")
        self.assertAlmostEqual(q_old, r_old * m_old, places=12)
        self.assertAlmostEqual(q_new, r_new * m_new, places=12)

    def test_example_2_flips_the_verdict_at_070(self) -> None:
        """例题 2：只把 m 的 floor 换成 0.70，判定式反向。"""
        kwargs = {"now": NOW, "query_entities": ["缓存层"], "task_hit": False}
        old = card("T4", 7, 55.0, 30.0, entities=("缓存层",))
        new = card("T5", 4, 1.5, 14.0, entities=("缓存层",))
        r_old = score_card(old, query_cos=0.95, scheme=SCHEME_RELEVANCE, **kwargs)
        r_new = score_card(new, query_cos=0.60, scheme=SCHEME_RELEVANCE, **kwargs)

        self.assertGreater(
            r_old / r_new,
            time_modifier(1.5, 14.0, floor=0.70) / time_modifier(55.0, 30.0, floor=0.70),
            "floor=0.70 时判定式反向，A 胜出",
        )

    def test_theorem_4_t0_is_immune_to_the_floor(self) -> None:
        """定理 4：h = 0 ⇒ q = r，与 f 和 Δ 无关。"""
        kwargs = {"now": NOW, "query_cos": 0.5, "query_entities": [], "task_hit": False}
        t0 = card("T0", 5, 100_000.0, 0.0)
        self.assertAlmostEqual(
            score_card(t0, scheme=SCHEME_MULTIPLICATIVE, **kwargs),
            score_card(t0, scheme=SCHEME_RELEVANCE, **kwargs),
            places=12,
        )
        fresh = card("T0", 5, 0.0, 0.0)
        self.assertAlmostEqual(
            score_card(t0, scheme=SCHEME_MULTIPLICATIVE, **kwargs),
            score_card(fresh, scheme=SCHEME_MULTIPLICATIVE, **kwargs),
            places=12,
        )

    def test_corollary_1_supremum_is_400_over_237(self) -> None:
        self.assertAlmostEqual(1.0 / TIME_FLOOR, 400 / 237, places=12)

    def test_corollary_2_supremum_is_not_attained(self) -> None:
        """推论 2：有限卡龄下 m > f 严格成立（浮点下溢到 1e9 天时才会取等）。"""
        self.assertGreater(time_modifier(1000.0, 30.0), TIME_FLOOR)
        self.assertEqual(time_modifier(10**9, 30.0), TIME_FLOOR, "极端卡龄下浮点下溢，取等")

    def test_theorem_5_additive_cannot_be_separated(self) -> None:
        """定理 5 的反例（附录 A.3）：加性打分下负例分数高于正例，故任何 θ 都无法分离。

        负例 = 用例 `09-gate-noise-and-window` 的 `weak-t3`（T3 无窗口）；
        正例 = 用例 `17-explicit-recall-admits-t8` 的 `paraphrase`（T5，Δ=30 恰在窗口上界）。
        两条都**窗口合法**，且分属不同用例——θ 是全局阈值，跨用例比同用例更强。
        """
        negative = card("T3", 2, 10.0, 90.0)                  # weak-t3
        positive = card("T5", 3, 30.0, 14.0)                 # paraphrase

        a_negative = score_card(
            negative, now=NOW, query_cos=0.15, query_entities=["张三"],
            task_hit=False, scheme=SCHEME_ADDITIVE,
        )
        a_positive = score_card(
            positive, now=NOW, query_cos=0.72, query_entities=["张三"],
            task_hit=False, scheme=SCHEME_ADDITIVE,
        )
        self.assertAlmostEqual(a_negative, 0.530219, places=6)
        self.assertAlmostEqual(a_positive, 0.503608, places=6)
        self.assertGreater(
            a_negative, a_positive,
            "召回正例要求 θ < 0.503608，排除负例要求 θ > 0.530219，两式互斥",
        )


if __name__ == "__main__":
    unittest.main()
