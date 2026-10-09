"""Phase 2 验收：抽取管道必须幂等、可续跑、且不把寒暄送进模型。

覆盖：
- 规则快路径（T8/T9）与关键词层级判定
- 游标推进与「重跑即重放」
- 抽取器只收到非快路径消息
- 摘要压缩不得截掉关键字段
- 运行记录（模型名 + 提示词指纹）
"""

from __future__ import annotations

import pathlib
import shutil
import sys
import json
import unittest
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow import FixedClock, Store  # noqa: E402
from moonshadow.extractor import parse_llm_cards  # noqa: E402
from moonshadow.compress import (  # noqa: E402
    BaselineExtractor,
    build_prompt,
    build_summary,
    classify_fast_path,
    classify_tier,
    compile_session,
    prompt_hash,
)

MESSAGES = [
    "决定采用方案B。",
    "……",
    "嗯，好的，谢谢！",
    "张三负责在 2026-09-30 前给出报价。",
]


class FakeExtractor:
    """假抽取器：记录收到的消息，产出确定性的卡草稿。测试不依赖任何网络调用。"""

    name = "fake-extractor"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, messages):  # type: ignore[no-untyped-def]
        self.calls.append([str(m["id"]) for m in messages])
        for message in messages:
            yield {
                "tier": "T2",
                "importance": 8,
                "network": "experience",
                "summary": f"摘要：{str(message['text'])[:20]}",
                "facts": [],
                "source_ids": [str(message["id"])],
            }


class FakePathTest(unittest.TestCase):
    def test_noise_and_greeting_are_fast_pathed(self) -> None:
        self.assertEqual(classify_fast_path("……"), "T9")
        self.assertEqual(classify_fast_path(""), "T9")
        self.assertEqual(classify_fast_path("   "), "T9")
        self.assertEqual(classify_fast_path("嗯，好的，谢谢！"), "T8")
        self.assertEqual(classify_fast_path("收到"), "T8")

    def test_substantive_text_is_not_fast_pathed(self) -> None:
        self.assertIsNone(classify_fast_path("决定采用方案B。"))
        self.assertIsNone(classify_fast_path("好的，但是预算要改成 15 万。"))
        self.assertIsNone(classify_fast_path("好的方案"))

    def test_tier_keywords_priority(self) -> None:
        self.assertEqual(classify_tier("必须兼容旧版 API。"), "T0")
        self.assertEqual(classify_tier("以后都用 4 空格缩进。"), "T1")
        self.assertEqual(classify_tier("决定采用方案B。"), "T2")
        self.assertEqual(classify_tier("张三负责在 2026-09-30 前给出报价。"), "T3")
        self.assertEqual(classify_tier("服务器是 8 核 32G。"), "T5")
        self.assertEqual(classify_tier("接口当前延迟是 120ms。"), "T5")
        self.assertEqual(classify_tier("正在梳理中间推导。"), "T6")
        # 时间词与待办词同时出现时，待办优先：排期属于 T3 而不是 T5
        self.assertEqual(classify_tier("会议安排在明天下午。"), "T3")

    def test_summary_never_truncates_key_fields(self) -> None:
        text = "这是一段很长的说明。" * 20 + "预算 12 万，截止 2026-10-15。"
        summary = build_summary(text, limit=50)
        self.assertIn("12 万", summary)
        self.assertIn("2026-10-15", summary)

    def test_prompt_contains_ids_and_hash_is_stable(self) -> None:
        prompt = build_prompt([{"id": "msg_1", "created_at": "2026-10-01T09:00:00+00:00", "text": "你好"}])
        self.assertIn("msg_1", prompt)
        self.assertIn("你好", prompt)
        self.assertEqual(prompt_hash(prompt), prompt_hash(prompt))
        self.assertRegex(prompt_hash(prompt), r"^[0-9a-f]{16}$")


class CompileSessionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = ROOT / ".tmp" / f"{type(self).__name__}.{self._testMethodName}"
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.clock = FixedClock(datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc))
        self.store = Store(self.tmp, clock=self.clock)
        self.ids = [self.store.append_message("s1", text) for text in MESSAGES]

    def tearDown(self) -> None:
        self.store.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_compile_writes_cards_and_advances_cursor(self) -> None:
        extractor = FakeExtractor()
        report = compile_session(self.store, "s1", extractor, clock=self.clock)

        self.assertEqual(report.processed, 4)
        self.assertEqual(report.fast_path, 2)          # 「……」+「嗯，好的，谢谢！」
        self.assertEqual(report.extracted_batches, 1)  # 其余两条合成一批
        self.assertEqual(report.inserted, 4)
        self.assertEqual(self.store.pending_messages("s1"), [])
        self.assertEqual(report.cursor, (self.ids[-1], 3))

        tiers = sorted(card["tier"] for card in self.store.cards())
        self.assertEqual(tiers, ["T2", "T2", "T8", "T9"])

    def test_extractor_never_sees_fast_path_messages(self) -> None:
        extractor = FakeExtractor()
        compile_session(self.store, "s1", extractor, clock=self.clock)
        self.assertEqual(len(extractor.calls), 1)
        self.assertEqual(extractor.calls[0], [self.ids[0], self.ids[3]])

    def test_rerun_same_span_is_pure_replay(self) -> None:
        """重跑同一段原文：卡数不变、状态不变，只增加重放计数。"""
        compile_session(self.store, "s1", FakeExtractor(), clock=self.clock)
        before = sorted(card["id"] for card in self.store.cards())

        self.store.set_cursor("s1", last_source_id=self.ids[0], last_day="2026-10-01", last_line=-1)
        report = compile_session(self.store, "s1", FakeExtractor(), clock=self.clock)

        self.assertEqual(report.inserted, 0)
        self.assertEqual(report.replayed, 4)
        self.assertEqual(sorted(card["id"] for card in self.store.cards()), before)

    def test_batching_splits_runs(self) -> None:
        report = compile_session(self.store, "s1", FakeExtractor(), clock=self.clock, max_messages=2)
        self.assertEqual(report.processed, 4)
        self.assertEqual(len(self.store.runs("s1")), 2)

    def test_baseline_extractor_is_usable_as_extractor(self) -> None:
        report = compile_session(self.store, "s1", BaselineExtractor(), clock=self.clock)
        self.assertEqual(report.inserted, 4)
        run = self.store.runs("s1")[0]
        self.assertEqual(run["model"], "baseline-rules")
        self.assertRegex(run["prompt_hash"], r"^[0-9a-f]{16}$")

    def test_nothing_pending_is_a_noop(self) -> None:
        compile_session(self.store, "s1", FakeExtractor(), clock=self.clock)
        report = compile_session(self.store, "s1", FakeExtractor(), clock=self.clock)
        self.assertEqual(report.processed, 0)
        self.assertEqual(report.inserted, 0)
        self.assertIsNone(report.cursor)


class PromptAndLabelTest(unittest.TestCase):
    """prompt 模板与 source_id 标签映射（都是接口的一部分）。"""

    def test_prompt_renders_and_brace_example_is_json(self) -> None:
        """PROMPT_TEMPLATE 走 .format()：模板里的字面大括号必须写 {{ }}。
        实测踩过：写成单个大括号 → KeyError → 35 个测试失败。本测试同时验证
        渲染不抛错、且提示词里的 claims 示例是**合法 JSON**。"""
        prompt = build_prompt(
            [{"id": "m1", "text": "示例", "created_at": "2026-10-01T09:00:00+00:00"}]
        )
        self.assertIn("claims 必须是对象数组", prompt)
        marker = 'claims": ['
        start = prompt.index(marker) + len(marker) - 1
        end = prompt.index("]", start) + 1
        parsed = json.loads(prompt[start:end])
        self.assertEqual(parsed[0]["kind"], "decision")

    def test_label_to_id_maps_prompt_labels_to_store_ids(self) -> None:
        """LLM 的 `source_ids` 是提示词局部标签，管线必须映射；映射不到的原样保留。"""
        cards = parse_llm_cards(
            '[{"tier": "T3", "summary": "s", "source_ids": ["m1", "m9"]}]',
            label_to_id={"m1": "src_a"},
        )
        self.assertEqual(cards[0].source_ids, ["src_a", "m9"])


if __name__ == "__main__":
    unittest.main()
