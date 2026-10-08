"""Phase 2.5 验收：评测关卡必须能判「没通过」，否则它就不是关卡。

关键用例是那个故意丢字段的抽取器：如果连它都能 PASS，说明指标是装饰品。
"""

from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow.compress import BaselineExtractor  # noqa: E402
from moonshadow.eval import (  # noqa: E402
    DEFAULT_MIN_RECALL,
    GoldCase,
    load_gold,
    run_case,
    run_eval,
)


class LossyExtractor:
    """故意不可用：摘要丢掉所有细节，用来验证关卡真的会失败。"""

    name = "lossy"

    def __call__(self, messages):  # type: ignore[no-untyped-def]
        for message in messages:
            yield {
                "tier": "T6",
                "importance": 2,
                "network": "observation",
                "summary": "已省略细节。",
                "source_ids": [str(message["id"])],
            }


class GoldLoadingTest(unittest.TestCase):
    def test_gold_set_loads_and_is_sorted(self) -> None:
        cases = load_gold()
        self.assertGreaterEqual(len(cases), 3)
        self.assertEqual([case.id for case in cases], sorted(case.id for case in cases))
        for case in cases:
            self.assertTrue(case.messages)
            self.assertTrue(case.day)

    def test_case_without_expected_fields_still_loads(self) -> None:
        case = GoldCase.from_json({"id": "x", "messages": ["a"]})
        self.assertEqual(case.expected_key_fields, [])
        self.assertEqual(case.expected_tiers, [])


class EvalGateTest(unittest.TestCase):
    def test_baseline_passes_the_gate(self) -> None:
        report = run_eval(extractor=BaselineExtractor())
        self.assertTrue(report.ok, report.summary())
        self.assertGreaterEqual(report.key_field_recall, DEFAULT_MIN_RECALL)
        self.assertGreater(report.compression_ratio, 0.0)
        self.assertIn("PASS", report.summary())

    def test_lossy_extractor_fails_the_gate(self) -> None:
        report = run_eval(extractor=LossyExtractor())
        self.assertFalse(report.ok)
        self.assertLess(report.key_field_recall, 1.0)
        self.assertIn("FAIL", report.summary())
        failed = [r for r in report.results if not r.ok]
        self.assertTrue(failed)
        self.assertTrue(any(r.lost for r in failed), "必须指明丢了哪些字段")

    def test_missing_expected_field_is_reported(self) -> None:
        case = GoldCase(
            id="unit-missing-field",
            messages=["预算 12 万。"],
            expected_key_fields=["999 万"],
        )
        result = run_case(case, BaselineExtractor())
        self.assertFalse(result.ok)
        self.assertEqual(result.missing_expected, ["999 万"])

    def test_missing_expected_tier_is_reported(self) -> None:
        case = GoldCase(
            id="unit-missing-tier",
            messages=["决定采用方案B。"],
            expected_tiers=["T9"],
        )
        result = run_case(case, BaselineExtractor())
        self.assertFalse(result.ok)
        self.assertEqual(result.missing_tiers, ["T9"])

    def test_expanding_extractor_is_flagged(self) -> None:
        """压缩率 < 1 必须明确告警：PASS 不等于压缩有效。"""
        report = run_eval(extractor=BaselineExtractor())
        self.assertTrue(report.ok)
        self.assertLess(report.compression_ratio, 1.0)
        self.assertIn("膨胀", report.summary())

    def test_case_exception_is_contained(self) -> None:
        def broken(messages):  # type: ignore[no-untyped-def]
            raise RuntimeError("抽取器炸了")
            yield  # pragma: no cover

        case = GoldCase(id="unit-broken", messages=["有实质内容的一句话。"])
        result = run_case(case, broken)  # type: ignore[arg-type]
        self.assertFalse(result.ok)
        self.assertIsNotNone(result.error)
        self.assertIn("RuntimeError", str(result.error))


if __name__ == "__main__":
    unittest.main()
