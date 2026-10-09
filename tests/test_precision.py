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
    CARD_TEXT_FIELDS,
    QUALITATIVE_KEYWORDS,
    SENTENCE_SPLIT_RE,
    Report,
    card_text,
    extract_key_fields,
    normalize,
    verify_field_mutual_exclusion,
    verify_no_silent_loss,
    verify_qualitative_retention,
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


class QualitativeRetentionTest(unittest.TestCase):
    """定性内容判据：硬字段保住 ≠ 内容保住。

    这一条来自 v1.3 Phase 0 的实测缺陷：只保留含硬字段的句子时，硬关卡
    `recall = 1.0000`、只产生软告警，而全部决策与条件已经丢光——
    「正常的抽象」与「错误的信息丢失」在关卡上不可区分。
    """

    RAW = [
        {"id": "m1", "text": "决定先做读缓存，写路径这一轮不动。"},
        {"id": "m2", "text": "压测数据 2026-09-30 之前给出。"},
        {"id": "m3", "text": "好的，谢谢！"},
    ]

    def test_passes_when_every_qualitative_message_has_a_carrier(self) -> None:
        report = verify_qualitative_retention(
            self.RAW,
            [
                {"tier": "T2", "source_ids": ["m1"]},
                {"tier": "T3", "source_ids": ["m2"]},
                {"tier": "T8", "source_ids": ["m3"]},
            ],
        )
        self.assertTrue(report.ok, report.summary())

    def test_fails_when_decision_message_has_no_card(self) -> None:
        report = verify_qualitative_retention(
            self.RAW, [{"tier": "T3", "source_ids": ["m2"]}]
        )
        self.assertFalse(report.ok)
        self.assertIn("qualitative", report.missing)
        self.assertIn("m1", "".join(report.missing["qualitative"]))
        self.assertIn("decision", "".join(report.missing["qualitative"]))
        self.assertIn("FAIL", report.summary())

    def test_greeting_card_is_not_a_carrier(self) -> None:
        """寒暄/噪音卡按定义就是要丢的内容，不能充当决策的载体。"""
        report = verify_qualitative_retention(
            self.RAW,
            [
                {"tier": "T8", "source_ids": ["m1"]},
                {"tier": "T9", "source_ids": ["m1"]},
                {"tier": "T3", "source_ids": ["m2"]},
            ],
        )
        self.assertFalse(report.ok)
        self.assertIn("T8", "".join(report.missing["qualitative"]))

    def test_negation_and_constraint_are_covered(self) -> None:
        report = verify_qualitative_retention(
            [{"id": "x", "text": "必须兼容旧版，不得删除 `adapter.py`。"}],
            [],
        )
        self.assertFalse(report.ok)
        kinds = "".join(report.missing["qualitative"])
        self.assertIn("constraint", kinds)

    def test_plain_message_needs_no_carrier(self) -> None:
        """没有定性内容的句子不构成要求——判据不能把所有消息都变成硬约束。"""
        report = verify_qualitative_retention(
            [{"id": "n", "text": "本地 QPS 比线上高了将近四成。"}], []
        )
        self.assertTrue(report.ok, report.summary())

    def test_keywords_and_splitter_match_compress(self) -> None:
        """词表与句子切分在 `verify` 里各有一份副本（循环依赖），必须与 `compress` 同步。"""
        from moonshadow.compress import SENTENCE_SPLIT_RE as COMPRESS_SPLIT
        from moonshadow.compress import TIER_KEYWORDS

        tier_words = {tier: words for tier, words in TIER_KEYWORDS}
        self.assertEqual(
            set(QUALITATIVE_KEYWORDS["decision"]), set(tier_words["T2"]),
            "decision 词表必须与 compress 的 T2 判定词一致",
        )
        self.assertEqual(
            set(QUALITATIVE_KEYWORDS["constraint"]), set(tier_words["T0"]),
            "constraint 词表必须与 compress 的 T0 判定词一致",
        )
        self.assertEqual(SENTENCE_SPLIT_RE.pattern, COMPRESS_SPLIT.pattern)


class RawQuoteAndBacklinkTest(unittest.TestCase):
    """Q3：`raw_quote` 移出卡文本，`T0/T1` 的逐字可用性改由 `source_ids` 回链保证。

    这组测试钉住两件事：**卡文本不再计 `raw_quote`**（否则逐字复制等于有许可证），
    以及**回链的边界**（它只覆盖「卡引用到的原文」，追不回没被引用的消息）。
    """

    RAW = "预算 12 万，截止 2026-10-15。"

    def test_card_text_fields_exclude_raw_quote(self) -> None:
        self.assertNotIn("raw_quote", CARD_TEXT_FIELDS)

    def test_card_text_ignores_raw_quote_content(self) -> None:
        """卡上仍带 `raw_quote`（旧卡与规则基线）时，它既不计入卡文本、也不进压缩率。"""
        text = card_text(
            {"summary": "预算 12 万。", "raw_quote": "原文整段：预算 12 万，截止 2026-10-15。"}
        )
        self.assertIn("12 万", text)
        self.assertNotIn("2026-10-15", text)

    def test_hard_field_is_reachable_through_the_source_link(self) -> None:
        report = verify_no_silent_loss(
            self.RAW,
            [card(summary="细节见原文。", source_ids=["m1"])],
            sources={"m1": self.RAW},
        )
        self.assertTrue(report.ok, report.summary())

    def test_backlink_does_not_rescue_a_dropped_message(self) -> None:
        """回链只覆盖「卡引用到的原文」：没有卡引用它，字段就真的追不回来。"""
        report = verify_no_silent_loss(
            self.RAW,
            [card(summary="另一件事。", source_ids=["m2"])],
            sources={"m1": self.RAW, "m2": "另一件事。"},
        )
        self.assertFalse(report.ok)
        self.assertEqual(report.missing["amount"], ["12 万"])

    def test_without_sources_the_check_stays_card_only(self) -> None:
        """不传 `sources` 时退回旧行为——回链必须由调用方显式打开。"""
        report = verify_no_silent_loss(self.RAW, [card(summary="细节见原文。")])
        self.assertFalse(report.ok)


class FieldMutualExclusionTest(unittest.TestCase):
    """C6：卡内字段互斥。同一句话只能有一个载体字段；summary 不得逐字复述。

    这条判据是实测逼出来的：同一句话既进 `facts` 又进 `decisions`/`todos`/`constraints` 时，
    token 加权压缩率 `0.8421x`；同样的代码、同样的卡数，只把重复那一份去掉就是 `1.4985x`。
    """

    def test_accepts_a_mutually_exclusive_card(self) -> None:
        report = verify_field_mutual_exclusion(
            [
                {
                    "summary": "T3 组：3 条消息、1 条事实。",
                    "facts": ["压测数据 2026-09-30 之前给出。"],
                    "decisions": ["决定先做读缓存。"],
                    "todos": ["张三负责压测报告。"],
                    "constraints": ["不得删除 `adapter.py`。"],
                }
            ]
        )
        self.assertTrue(report.ok, report.summary())

    def test_rejects_same_sentence_in_two_fields(self) -> None:
        report = verify_field_mutual_exclusion(
            [{"summary": "概述", "facts": ["决定先做读缓存。"], "decisions": ["决定先做读缓存。"]}]
        )
        self.assertFalse(report.ok)
        self.assertIn("field_mutual_exclusion", report.missing)
        self.assertIn("facts 与 decisions", "".join(report.missing["field_mutual_exclusion"]))

    def test_rejects_duplicate_within_one_field(self) -> None:
        report = verify_field_mutual_exclusion(
            [{"summary": "概述", "facts": ["预算 12 万。", "预算 12 万。"]}]
        )
        self.assertFalse(report.ok)
        self.assertIn("写了两次", "".join(report.missing["field_mutual_exclusion"]))

    def test_rejects_summary_restating_a_field_sentence(self) -> None:
        """C6 的第二条：summary 不得逐字复述字段里的句子——否则它只堵了一半。"""
        sentence = "决定先做读缓存，写路径这一轮完全不动。"
        report = verify_field_mutual_exclusion(
            [{"summary": sentence, "decisions": [sentence]}]
        )
        self.assertFalse(report.ok)
        self.assertIn("summary 逐字复述", "".join(report.missing["field_mutual_exclusion"]))

    def test_summary_restatement_check_ignores_fragments(self) -> None:
        """**边界**（漏洞 #10）：判的是句子，**片段不算**。

        `summary` 里出现 `12万` 这种字段片段不算「复述整句」。规则基线上实测有
        `685` 处这类片段重叠——把它们算成违规会让判据变成误报机器。
        代价是片段级复述逃得出这条判据，那是 C6 的已知缺口，记在 `v1.3-spec.md` §6.2。
        """
        report = verify_field_mutual_exclusion(
            [{"summary": "本周要点：12万。", "facts": ["12万"]}]
        )
        self.assertTrue(report.ok, "片段不是句子，不该判违规")

    def test_summary_may_use_its_own_words(self) -> None:
        """不复述就不算违规——判据只拦「同一句话写两遍」，不拦真正的概述。"""
        report = verify_field_mutual_exclusion(
            [{"summary": "本轮定下读缓存方向。", "decisions": ["决定先做读缓存。"]}]
        )
        self.assertTrue(report.ok, report.summary())


if __name__ == "__main__":
    unittest.main()
