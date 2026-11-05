"""压力测试门的数学：`summarize` / `limit_ms` / `verdict` / `marginal_rate` 是纯函数。

为什么单独测这四个：它们是**判据本身**。工具跑一次要几十秒，但判据算错的话，
跑得再快也是错的。这一组测试不建库、不跑召回，毫秒级完成。

重点钉住三件事：
1. 门限是**仿射**的（`A + B·n`），不是比例的——比例门在小库必然误杀；
2. 速度门用**中位数**判，稳定性门用 **IQR/中位**判（稳健），`max/min` 只做诊断；
3. 边际门抓的是**台阶**——同一个读数，绝对门放行、边际门必须拦下。
"""

from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from stress_recall import (  # noqa: E402
    BUDGET_MS_PER_1K,
    FLOOR_MS,
    IQR_RATIO_LIMIT,
    VERIFIED_UP_TO,
    limit_ms,
    marginal_rate,
    summarize,
    verdict,
)


def row(cards: int, samples: list[float], *, tokens_used: int = 100, budget_tokens: int = 4000,
        over_budget: bool = False, reserved_share: float = 0.1, deterministic: bool = True) -> dict:
    """构造一条测量记录（只有判据需要的字段）。"""
    measured = summarize(samples)
    measured.update(
        {
            "cards": cards,
            "tokens_used": tokens_used,
            "budget_tokens": budget_tokens,
            "over_budget": over_budget,
            "reserved_share": reserved_share,
            "deterministic": deterministic,
        }
    )
    return measured


class SummarizeTest(unittest.TestCase):
    def test_interval_and_robust_dispersion(self) -> None:
        stats = summarize([88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 96.0])
        self.assertEqual(stats["n"], 7)
        self.assertEqual(stats["min_ms"], 88.0)
        self.assertEqual(stats["max_ms"], 96.0)
        self.assertEqual(stats["median_ms"], 91.0)
        # max/min 是极端统计量，只做诊断；IQR/中位 才是判门用的稳健量
        self.assertAlmostEqual(stats["max_ratio"], round(96.0 / 88.0, 3))
        self.assertGreater(stats["iqr_ratio"], 0)

    def test_single_sample_does_not_crash(self) -> None:
        stats = summarize([42.0])
        self.assertEqual((stats["n"], stats["min_ms"], stats["max_ms"], stats["median_ms"]), (1, 42.0, 42.0, 42.0))


class LimitTest(unittest.TestCase):
    def test_limit_is_affine_not_proportional(self) -> None:
        """这是本工具最重要的一条数学性质：小库要有固定开销项 A。

        纯比例门（`B·n`）在 10 张卡时只给 `3 ms`，而实测要 `5 ms`——必然误杀。
        """
        self.assertEqual(limit_ms(1000), FLOOR_MS + BUDGET_MS_PER_1K)
        self.assertEqual(limit_ms(10), FLOOR_MS + BUDGET_MS_PER_1K * 0.01)
        # 比例项只有 3 ms，而被测实际 5 ms；加上 A 之后门限才成立
        self.assertLess(BUDGET_MS_PER_1K * 10 / 1000, 5.0)
        self.assertGreater(limit_ms(10), 5.0)

    def test_limit_grows_monotonically(self) -> None:
        caps = [limit_ms(cards) for cards in (10, 100, 1000, 10000)]
        self.assertEqual(caps, sorted(caps))


class VerdictTest(unittest.TestCase):
    def test_speed_gate_uses_median_not_max(self) -> None:
        """一条离群慢样本不该让速度门红——中位数对它稳健。"""
        cap = limit_ms(1000)
        raw = [cap - 200, cap - 100, cap - 50, cap - 40, cap + 800]   # 中位低于门限，max 超过
        result = verdict(row(1000, raw))
        self.assertTrue(result["hard"]["G2a_speed"])
        self.assertGreater(result["stats"]["max_ms"] if "stats" in result else max(raw), cap)

    def test_speed_gate_fails_when_median_exceeds(self) -> None:
        cap = limit_ms(1000)
        result = verdict(row(1000, [cap + 10, cap + 20, cap + 30, cap + 40, cap + 50]))
        self.assertFalse(result["hard"]["G2a_speed"])
        self.assertFalse(result["passed"])

    def test_overrun_must_be_visible(self) -> None:
        """超预算可以被看见，但不能被藏起来：G3 判的是「有没有标记」。"""
        silent = verdict(row(1000, [100.0] * 5, tokens_used=5000, over_budget=False))
        self.assertFalse(silent["hard"]["G3_no_silent_overrun"])
        declared = verdict(row(1000, [100.0] * 5, tokens_used=5000, over_budget=True))
        self.assertTrue(declared["hard"]["G3_no_silent_overrun"])

    def test_stability_gate_uses_iqr_ratio(self) -> None:
        tight = verdict(row(1000, [100.0, 101.0, 102.0, 103.0, 104.0]))
        self.assertTrue(tight["soft"]["G5_stability"])
        loose = verdict(row(1000, [50.0, 80.0, 100.0, 140.0, 200.0]))
        self.assertFalse(loose["soft"]["G5_stability"])
        self.assertIn("G5_stability", loose["warnings"])

    def test_reserved_share_is_a_warning_not_a_hard_gate(self) -> None:
        result = verdict(row(1000, [100.0] * 5, reserved_share=4.47))
        self.assertFalse(result["soft"]["G4_reserved_share"])
        self.assertTrue(result["passed"], "常驻区超线是已知缺陷（#19），不能当硬门")

    def test_limit_floor_is_the_constant(self) -> None:
        self.assertEqual(verdict(row(1000, [100.0] * 5))["limit_ms"], round(limit_ms(1000), 1))
        self.assertEqual(IQR_RATIO_LIMIT, 0.25)

    def test_beyond_verified_range_is_reported_but_not_judged(self) -> None:
        """超过已验证范围时，速度门降级为警告：门限本身是把曲线延长出来的。

        这条防的是「用外推值给人判红」——外部证据只有 10–20000 卡，
        再往上三种同样拟合良好的函数形式分歧 35%（见 docs/v1.2-perf-model.md §7）。
        """
        slow = [VERIFIED_UP_TO * 2 * 0.5] * 5          # 50000 卡、每卡 0.5 ms → 25000 ms，远超门限
        measured = summarize(slow)
        measured.update(
            {
                "cards": VERIFIED_UP_TO * 2,
                "tokens_used": 100,
                "budget_tokens": 4000,
                "over_budget": False,
                "reserved_share": 0.1,
                "deterministic": True,
            }
        )
        relaxed = verdict(measured)
        self.assertTrue(relaxed["extrapolated"])
        self.assertNotIn("G2a_speed", relaxed["hard"])
        self.assertIn("G2a_speed_extrapolated", relaxed["soft"])
        self.assertFalse(relaxed["soft"]["G2a_speed_extrapolated"])   # 仍然照实报「超了」
        self.assertTrue(relaxed["passed"], "外推区间不该判红")

        strict = verdict(measured, strict_extrapolated=True)
        self.assertIn("G2a_speed", strict["hard"])
        self.assertFalse(strict["passed"], "显式要求按外推硬判时才红")

    def test_within_verified_range_is_judged_normally(self) -> None:
        self.assertFalse(verdict(row(20000, [100.0] * 5))["extrapolated"])
        self.assertIn("G2a_speed", verdict(row(20000, [100.0] * 5))["hard"])


class MarginalTest(unittest.TestCase):
    def test_step_is_caught_by_marginal_even_when_absolute_passes(self) -> None:
        """1000 → 1500 的实测台阶：绝对门放行、边际门必须拦下。"""
        lower = row(1000, [93.0] * 5)
        upper = row(1500, [382.5] * 5)
        rate = marginal_rate(lower, upper)
        self.assertIsNotNone(rate)
        self.assertAlmostEqual(rate, 579.0, delta=1.0)
        self.assertGreater(rate, BUDGET_MS_PER_1K)
        # 绝对门在 1500 卡时要求 ≤ 465 ms，382.5 通过——所以只看绝对门会漏
        self.assertTrue(verdict(upper)["hard"]["G2a_speed"])
        self.assertLess(upper["median_ms"], verdict(upper)["limit_ms"])

    def test_flat_marginal_passes(self) -> None:
        rate = marginal_rate(row(1000, [100.0] * 5), row(3000, [600.0] * 5))
        self.assertIsNotNone(rate)
        self.assertAlmostEqual(rate, 250.0, delta=1.0)
        self.assertLess(rate, BUDGET_MS_PER_1K)

    def test_non_increasing_cards_returns_none(self) -> None:
        self.assertIsNone(marginal_rate(row(1000, [100.0] * 5), row(1000, [200.0] * 5)))


if __name__ == "__main__":
    unittest.main()
