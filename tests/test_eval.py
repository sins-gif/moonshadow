"""Phase 2.5 验收：评测关卡必须能判「没通过」，否则它就不是关卡。

关键用例是那个故意丢字段的抽取器：如果连它都能 PASS，说明指标是装饰品。
"""

from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow.compress import (  # noqa: E402
    SENTENCE_SPLIT_RE,
    TIER_IMPORTANCE,
    BaselineExtractor,
    classify_tier,
)
from moonshadow.eval import (  # noqa: E402
    DEFAULT_MIN_RECALL,
    GoldCase,
    load_gold,
    run_case,
    run_eval,
)
from moonshadow.pack import estimate_tokens  # noqa: E402
from moonshadow.tiers import TIER_ORDER  # noqa: E402
from moonshadow.verify import extract_key_fields, normalize  # noqa: E402


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


class DropQualitativeExtractor:
    """只给含硬字段的消息出卡；纯定性内容的消息（决策/约束/否定）**一张卡都不出**。

    这是定性内容判据要拦的形状：消息连同它的 `source_id` 一起消失，追溯链断了。
    """

    name = "drop-qualitative"

    def __call__(self, messages):  # type: ignore[no-untyped-def]
        for message in messages:
            text = str(message.get("text", ""))
            if not extract_key_fields(text):
                continue  # 没有硬字段 → 整条丢掉（含决策、条件、否定）
            tier = classify_tier(text)
            yield {
                "tier": tier,
                "importance": TIER_IMPORTANCE[tier],
                "network": "experience",
                "summary": text[:120],
                "facts": [text[:120]],
                "source_ids": [str(message["id"])],
            }


class StubCardExtractor:
    """每条消息都出一条「只在硬字段上有内容」的卡：引用了 source_id，但没有承载定性内容。

    用它把定性判据的**边界**钉住：这条判据是**可追溯性**判据（有没有卡引用），
    不是语义判据（卡里有没有那句话）。上面的形状它拦得住，这个形状它拦不住——
    要拦这一种得靠模型模糊校验（第二级），硬关卡不假装能判语义。
    """

    name = "stub-card"

    def __call__(self, messages):  # type: ignore[no-untyped-def]
        for message in messages:
            text = str(message.get("text", ""))
            tier = classify_tier(text)
            kept = [
                sentence.strip()
                for sentence in SENTENCE_SPLIT_RE.split(text)
                if sentence.strip() and extract_key_fields(sentence)
            ]
            yield {
                "tier": tier,
                "importance": TIER_IMPORTANCE[tier],
                "network": "experience",
                "summary": "（摘要未写）",
                "facts": kept,
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
    def test_baseline_passes_the_field_gate(self) -> None:
        """基线在**字段保留**上过门（`1.0000` ≥ `0.98`），但它**违反 C6**。

        v1.3 加入 C6（卡内字段互斥）之后，规则基线不再是「合规模板」：
        它把同一句话同时写进 `facts` 与 `decisions`/`todos`/`constraints`，
        实测 `751` 条违规。所以这条测试现在只断言**字段保留那一侧**，
        并显式断言 C6 违规确实存在——两者都是事实，不该被一条 `assertTrue(report.ok)` 混在一起。
        """
        report = run_eval(extractor=BaselineExtractor())
        self.assertGreaterEqual(report.key_field_recall, DEFAULT_MIN_RECALL)
        violations = [r for r in report.results if r.field_violations]
        self.assertTrue(violations, "基线按定义违反 C6，这里应当有违规")
        self.assertFalse(report.ok, "违反 C6 的抽取器不该过门")

    def test_lossy_extractor_fails_the_gate(self) -> None:
        """故意不可用的抽取器必须被判失败。

        Q3 之后它**失败的位置变了**：全局硬字段扫描改走 `source_id` 回链（卡引用了消息
        就算追得回），所以拦住它的是**金标准集的 `expected_key_fields`**（卡内文本判据）
        与定性内容判据，而不是 `lost`。这条测试跟着改，是为了不让它假装还在测原来那件事。
        """
        report = run_eval(extractor=LossyExtractor())
        self.assertFalse(report.ok)
        self.assertIn("FAIL", report.summary())
        failed = [r for r in report.results if not r.ok]
        self.assertTrue(failed)
        self.assertTrue(any(r.missing_expected for r in failed), "必须指明缺了哪些期望字段")

    def test_global_sweep_is_now_backlink_based(self) -> None:
        """**边界**（Q3 的代价，实测钉住）：卡引用了消息却没写字段时，全局扫描判通过。

        `key_field_recall` 因此恒为 `1.0`（只要原文还在 store 里）。这不是 bug，
        是「原文可用性由 store 保证、不再由卡保证」的直接后果；要恢复这条扫描的分辨力，
        只能改原文保留策略（原文过期后回链自然失效），不能靠改判据。
        """
        report = run_eval(extractor=LossyExtractor())
        self.assertEqual(report.key_field_recall, 1.0)
        self.assertTrue(all(not r.lost for r in report.results))

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
        """压缩率 < 1 必须明确告警：PASS 不等于压缩有效。

        不断言 `report.ok`：基线违反 C6，整体判定本来就是 FAIL——告警与判定是两件事。
        """
        report = run_eval(extractor=BaselineExtractor())
        self.assertLess(report.weighted_compression, 1.0)
        self.assertIn("膨胀", report.summary())

    def test_dropped_qualitative_message_is_caught(self) -> None:
        """把纯定性内容的消息整条丢掉的抽取器**必须失败**。

        这是定性内容判据的存在理由：这个形状的硬字段保留率是 `1.0000`，
        在加这条判据之前它能拿到 ``ok=True``。
        """
        report = run_eval(extractor=DropQualitativeExtractor())
        self.assertFalse(report.ok, "丢掉纯定性消息的抽取器必须被拦住")
        caught = [r for r in report.results if r.uncarried]
        self.assertTrue(caught, "必须报告哪些原文的定性内容没有载体")
        self.assertIn("定性内容无载体", caught[0].summary())

    def test_stub_card_passes_this_gate_by_design(self) -> None:
        """**边界**：出了卡、引用了 source_id 但没有承载定性内容的形状，这条判据拦不住。

        判据是可追溯性判据：它要求「有卡能追溯到这条原文」，不要求「卡里有那句话」——
        措辞可以不同是有意的（抽象是允许的）。把它当成语义校验会高估这道关卡的强度，
        所以用一条测试把边界钉住，而不是让读者以为它拦得住一切。
        """
        report = run_eval(extractor=StubCardExtractor())
        self.assertFalse(
            any(r.uncarried for r in report.results),
            "该形状按设计不会被可追溯性判据拦下；若这里变红，说明判据被加强了，请更新本说明",
        )

    def test_case_exception_is_contained(self) -> None:
        def broken(messages):  # type: ignore[no-untyped-def]
            raise RuntimeError("抽取器炸了")
            yield  # pragma: no cover

        case = GoldCase(id="unit-broken", messages=["有实质内容的一句话。"])
        result = run_case(case, broken)  # type: ignore[arg-type]
        self.assertFalse(result.ok)
        self.assertIsNotNone(result.error)
        self.assertIn("RuntimeError", str(result.error))


class TierReadingTest(unittest.TestCase):
    """分层读数必须能回答「膨胀出在哪一层」，所以它自己也要被测。

    两条口径（逐用例均值 / token 加权）与分层归集都是**报告语义**：
    算错不会让关卡变红，只会让人把结论读反，所以只能靠断言钉住。
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_eval(extractor=BaselineExtractor())

    def test_tier_reading_covers_all_ten_tiers(self) -> None:
        stats = self.report.tier_stats()
        self.assertEqual([stat.tier for stat in stats], list(TIER_ORDER))
        table = "\n".join(self.report.tier_table())
        for stat in stats:
            self.assertIn(stat.tier, table)
            if stat.cards == 0:
                # 「不可观测」必须与「压缩得很差」区分开
                self.assertEqual(stat.compression, 0.0)

    def test_tier_card_tokens_add_up_to_the_case_totals(self) -> None:
        """分层卡 token 之和必须等于逐用例卡 token 之和，否则表里会漏掉一段。"""
        per_case = sum(r.card_tokens for r in self.report.results if r.cards)
        per_tier = sum(stat.card_tokens for stat in self.report.tier_stats())
        self.assertEqual(per_tier, per_case)

    def test_weighted_and_mean_compression_are_both_reported(self) -> None:
        """两个口径都必须出现在报告里——只报一个就是把结论交给读者去猜。"""
        summary = self.report.summary()
        self.assertIn("平均压缩率", summary)
        self.assertIn("总体压缩率", summary)
        raw = sum(r.raw_tokens for r in self.report.results)
        cards = sum(r.card_tokens for r in self.report.results)
        self.assertAlmostEqual(self.report.weighted_compression, raw / cards, places=9)

    def test_per_case_raw_tokens_are_counted_once(self) -> None:
        """逐用例的 ``原文=Ntok`` 必须是**每条消息各计一次**。

        分层表允许同一段原文在多个层级各计一次（那段原文确实支撑了多张卡），
        但那个重复计数**绝不能进入逐用例或总体压缩率**，否则报告里的压缩率会被污染。
        因此两者的大小关系是单向的：分层原文之和不小于用例原文。
        """
        cases = {case.id: case for case in load_gold()}
        for result in self.report.results:
            case = cases[result.case_id]
            expected = sum(estimate_tokens(text) for text in case.messages)
            with self.subTest(case=result.case_id):
                self.assertEqual(result.raw_tokens, expected)
                tier_raw = sum(stat.raw_tokens for stat in result.tier_stats.values())
                self.assertGreaterEqual(
                    tier_raw, result.raw_tokens,
                    "分层原文之和不小于用例原文；若小于，说明有原文没被任何卡引用",
                )

    def test_gold_set_actually_exercises_multiple_tiers(self) -> None:
        """金标准集若只覆盖一两个层级，分层读数就是装饰品。"""
        covered = {stat.tier for stat in self.report.tier_stats() if stat.cards}
        self.assertGreaterEqual(len(covered), 5, f"只覆盖了 {sorted(covered)}")


class RedundancyTest(unittest.TestCase):
    """「跨消息冗余」必须是可机检的结构，不能是口头声明。

    理由直接决定 v1.3 的核心目标能不能成立：压缩率的定义是「原文 token / 卡 token」，
    原文里没有可被抽象的冗余时，任何抽取器都做不到 > 1x。若这些长用例的「冗余」
    只写在 `note` 里，压缩率会好看而不可信。

    判据用 `verify.HARD_PATTERNS`（日期/金额/URL/代码）——**与精度关卡同一套定义**，
    不另立一套。「单号」不在这套定义里（`INC-2043` 这种要靠反引号包成代码才会被认到），
    所以本判据不认它：认了就等于偷偷放宽精度关卡的口径。
    """

    #: 「长会话」的门槛（token）。旧 3 条 28–45 token 是短用例，不受此约束。
    LONG_CASE_TOKENS = 300

    #: 逐字重复句的字符占比上限。模板句不是跨消息冗余，是灌水：任何做字符串去重的
    #: 实现都能删掉它，压缩率会好看，但那是「去重」不是「记忆压缩」。
    #: 实测踩过：一个 agent 把「这段我记进纪要了，细节等下一轮再对一遍。」复制进
    #: 几乎每条消息，占该用例 28.8% 的字符。
    MAX_VERBATIM_SHARE = 0.05
    #: 判定为「模板句」的门槛：长度 ≥12 字，且在 ≥3 条不同消息里逐字出现。
    MIN_SENTENCE_CHARS = 12
    REPEAT_MESSAGES = 3

    def test_long_cases_carry_cross_message_redundancy(self) -> None:
        checked = 0
        for case in load_gold():
            raw = sum(estimate_tokens(text) for text in case.messages)
            if raw < self.LONG_CASE_TOKENS:
                continue
            checked += 1
            seen: dict[str, set[int]] = {}
            for index, text in enumerate(case.messages, 1):
                for values in extract_key_fields(text).values():
                    for value in values:
                        seen.setdefault(normalize(value), set()).add(index)
            repeated = {
                value: sorted(where) for value, where in seen.items() if len(where) >= 2
            }
            self.assertTrue(
                repeated,
                f"{case.id}（{raw} token）没有任何关键字段在两条以上消息里重复出现——"
                "只出现一次的字段不构成冗余，这条用例对「压缩」没有证据价值",
            )
        self.assertGreater(checked, 0, "没有任何长用例被检查，门槛或数据集有问题")

    def test_long_cases_do_not_reuse_template_sentences(self) -> None:
        """长用例不得靠「逐字复制的模板句」凑长度。

        这条与上一条是一对：上一条要求**有**冗余，这一条禁止**假**冗余。
        同一句填充话复制进每条消息，会让压缩率读数好看而不可信——
        任何字符串去重都能删掉它，那证明的不是记忆压缩能力。
        真正的冗余是**信息**的重复：同一个日期/金额/决策换措辞在别的轮次里重申。
        """
        for case in load_gold():
            raw = sum(estimate_tokens(text) for text in case.messages)
            if raw < self.LONG_CASE_TOKENS:
                continue
            total = sum(len(text) for text in case.messages)
            where: dict[str, set[int]] = {}
            for index, text in enumerate(case.messages, 1):
                for sentence in SENTENCE_SPLIT_RE.split(text):
                    sentence = sentence.strip()
                    if len(sentence) >= self.MIN_SENTENCE_CHARS:
                        where.setdefault(sentence, set()).add(index)
            repeated = {
                sentence: sorted(seen)
                for sentence, seen in where.items()
                if len(seen) >= self.REPEAT_MESSAGES
            }
            share = sum(len(s) * len(w) for s, w in repeated.items()) / max(total, 1)
            with self.subTest(case=case.id):
                self.assertLessEqual(
                    share, self.MAX_VERBATIM_SHARE,
                    f"{case.id}：逐字重复句占 {share:.1%} 的字符（上限 "
                    f"{self.MAX_VERBATIM_SHARE:.0%}）→ {dict(list(repeated.items())[:2])}",
                )


if __name__ == "__main__":
    unittest.main()
