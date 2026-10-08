"""推论 1 的守卫：有界乘性方案的反转不得越过 `1/f`。

这是全项目里**唯一一条与数据无关的保证**——它不依赖用例集、不依赖参数漂移，
只由融合形式本身决定。因此必须独立于 21 条用例被检验。

对照：加性方案在**同一网格**上会在 `r_P/r_N ≥ 1/f` 的区域里大量反转（见 `moonshadow.drift`），
故该项保证是乘性形式独有的。
"""

from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow.drift import measure_drift  # noqa: E402
from moonshadow.scoring import TIME_FLOOR  # noqa: E402


class ReversalBoundGuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = measure_drift()

    def test_grid_is_non_trivial(self) -> None:
        self.assertGreater(self.report.total, 1000)
        self.assertGreater(self.report.additive_total, 0, "加性方案应当在该网格上出现反转")

    def test_multiplicative_never_breaks_its_own_bound(self) -> None:
        """推论 1：`r_P/r_N ≥ 1/f` 时乘性方案不得反转。"""
        offenders = self.report.multiplicative_over_bound
        self.assertEqual(
            [pair.label for pair in offenders], [],
            f"出现越过上界 {self.report.bound:.6f} 的反转",
        )
        self.assertLess(self.report.worst_multiplicative_ratio, self.report.bound)

    def test_additive_has_no_such_bound(self) -> None:
        """加性方案在乘性方案有保证的区域内同样反转——它没有任何比值上界。"""
        over = self.report.additive_over_bound
        self.assertGreater(
            len(over), 0,
            "加性方案若也守 1/f 上界，则「有界」这一性质就不是乘性形式独有的",
        )
        self.assertGreater(self.report.worst_additive_ratio, self.report.bound)

    def test_bound_is_exactly_one_over_floor(self) -> None:
        self.assertAlmostEqual(self.report.bound, 1.0 / TIME_FLOOR, places=12)

    def test_report_renders(self) -> None:
        from moonshadow.drift import format_drift_report

        text = format_drift_report(self.report)
        self.assertIn("乘性有保证的区域", text)
        self.assertIn("推论 1", text)
        # 无违规时报告应给出「全部落在上界之内」的分支，而不是「违反」
        self.assertIn("全部落在推论 1 的上界之内", text)


if __name__ == "__main__":
    unittest.main()
