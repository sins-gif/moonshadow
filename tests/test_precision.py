"""Phase 2.5 验收：关键字段丢失必须被判失败，而不是靠模型「感觉没问题」。

这里测的是硬关卡（正则），因为它是唯一能在 CI 里稳定回归的部分。
"""

from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow.verify import (  # noqa: E402
    Report,
    card_text,
    extract_key_fields,
    normalize,
    verify_no_silent_loss,
)


def card(**fields: object) -> dict:
    base = {"summary": ""}
    base.update(fields)
    return base


class ExtractTest(unittest.TestCase):
    def test_extracts_hard_fields(self) -> None:
        found = extract_key_fields("预算 12 万，截止 2026-10-15，见 https://x.io/a，跑 `pytest -q`")
        self.assertEqual(found["date"], ["2026-10-15"])
        self.assertEqual(found["amount"], ["12 万"])
        self.assertEqual(found["url"], ["https://x.io/a"])
        self.assertEqual(found["code"], ["`pytest -q`"])

    def test_deduplicates_in_order(self) -> None:
        found = extract_key_fields("截止 2026-01-01，再次确认 2026-01-01，另加 2026-02-02")
        self.assertEqual(found["date"], ["2026-01-01", "2026-02-02"])

    def test_normalize_ignores_spacing(self) -> None:
        self.assertEqual(normalize("12 万"), normalize("12万"))


class SilentLossTest(unittest.TestCase):
    RAW = "预算 12 万，截止 2026-10-15，必须兼容旧版 API。"

    def test_pass_when_fields_are_kept(self) -> None:
        report = verify_no_silent_loss(
            self.RAW,
            [card(summary="预算12万，截止2026-10-15，兼容旧版API。", constraints=["必须兼容旧版 API"])],
        )
        self.assertTrue(report.ok)
        self.assertIn("PASS", report.summary())

    def test_fail_when_amount_is_lost(self) -> None:
        report = verify_no_silent_loss(self.RAW, [card(summary="截止2026-10-15。")])
        self.assertFalse(report.ok)
        self.assertEqual(report.missing["amount"], ["12 万"])
        self.assertIn("FAIL", report.summary())

    def test_fields_may_be_split_across_cards(self) -> None:
        """12 万进 T0 卡、日期进 T2 卡，整体没丢，就不该报错。"""
        report = verify_no_silent_loss(
            self.RAW,
            [card(summary="预算12万。"), card(summary="截止 2026-10-15 交付。")],
        )
        self.assertTrue(report.ok)

    def test_negation_only_warns(self) -> None:
        raw = "预算 12 万，截止 2026-10-15，除非客户端显式升级，否则不得破坏兼容。"
        report = verify_no_silent_loss(raw, [card(summary="截止2026-10-15，预算12万。")])
        self.assertTrue(report.ok)
        self.assertTrue(report.warnings)
        self.assertIn("negation", "".join(report.warnings))

    def test_empty_report_is_ok(self) -> None:
        report = Report()
        self.assertTrue(report.ok)
        self.assertEqual(report.summary(), "PASS：关键字段无丢失")

    def test_card_text_covers_list_fields(self) -> None:
        text = card_text({"summary": "s", "facts": ["预算12万"], "todos": ["张三给报价"]})
        self.assertIn("预算12万", text)
        self.assertIn("张三给报价", text)


if __name__ == "__main__":
    unittest.main()
