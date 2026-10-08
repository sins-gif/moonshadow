"""Phase 1 验收：给定 source_id 必须逐字节取回原文，压缩后依然成立。

覆盖点：
- 追加/回取往返（含中文、表情、缩进，验证 UTF-8 字节区间不错位）
- sha256 篡改检测
- 冷存分帧后仍能精确回取（只解压所属帧）
- 记忆卡写入幂等（内容寻址 + dedup_key 唯一索引）
- schema 强制校验
- 压缩游标（触发机制不重复、不遗漏）
"""

from __future__ import annotations

import pathlib
import shutil
import sys
import unittest
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow import FixedClock, Store, make_card  # noqa: E402
from moonshadow.chunks import ChunkError  # noqa: E402
from moonshadow.schema import SchemaError, validate_card  # noqa: E402
from moonshadow.store import RawIntegrityError  # noqa: E402

DAY = "2026-01-01"
MESSAGES = [
    "预算 12 万，截止 2026-10-15。",
    "  缩进、换行以外的空白与 emoji 🚀 都要原样保留  ",
    "用户说：必须兼容旧版 API，除非客户端显式升级。",
    '{"code": "print(\'hi\')", "negation": "不要删除"}',
    "同意。",
]


class SourceRoundTripTest(unittest.TestCase):
    def setUp(self) -> None:
        # 测试目录放在工作区内：系统临时目录在受限沙箱下可能不可写
        self.tmp = ROOT / ".tmp" / f"{type(self).__name__}.{self._testMethodName}"
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.root = self.tmp
        self.clock = FixedClock(datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc))
        # frame_lines=2 让 5 条消息跨 3 帧，逼出「按帧定位」的真实路径
        self.store = Store(self.root, clock=self.clock, frame_lines=2)
        self.ids = [self.store.append_message("s1", text) for text in MESSAGES]

    def tearDown(self) -> None:
        self.store.close()
        # 清理失败（Windows 文件锁 / 沙箱权限）不应影响测试结论
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_round_trip_is_byte_exact(self) -> None:
        for source_id, expected in zip(self.ids, MESSAGES):
            self.assertEqual(self.store.get_raw(source_id), expected)

    def test_ids_are_unique_and_addressable(self) -> None:
        self.assertEqual(len(set(self.ids)), len(self.ids))
        for index, source_id in enumerate(self.ids):
            row = self.store.conn.execute(
                "SELECT day, line_no, byte_len FROM source WHERE id = ?", (source_id,)
            ).fetchone()
            self.assertEqual(row["day"], DAY)
            self.assertEqual(row["line_no"], index)
            self.assertGreater(row["byte_len"], 0)

    def test_tampering_is_detected(self) -> None:
        raw_file = self.root / "raw" / f"{DAY}.jsonl"
        blob = raw_file.read_bytes()
        tampered = blob.replace("预算 12 万".encode("utf-8"), "预算 13 万".encode("utf-8"))
        self.assertNotEqual(blob, tampered, "等长替换才能证明校验来自 sha256 而非长度")
        raw_file.write_bytes(tampered)
        with self.assertRaises(RawIntegrityError):
            self.store.get_raw(self.ids[0])

    def test_archived_round_trip_still_exact(self) -> None:
        index = self.store.archive_day(DAY)
        self.assertEqual(index.total_lines, len(MESSAGES))
        self.assertEqual(len(index.frames), 3)  # 2 + 2 + 1
        self.assertEqual(index.codec, "zlib")

        archived = self.store.conn.execute(
            "SELECT COUNT(*) AS n FROM source WHERE day = ? AND is_archived = 1", (DAY,)
        ).fetchone()
        self.assertEqual(archived["n"], len(MESSAGES))

        for source_id, expected in zip(self.ids, MESSAGES):
            self.assertEqual(self.store.get_raw(source_id), expected)

    def test_missing_container_index_raises(self) -> None:
        self.store.archive_day(DAY)
        container = self.store._container_for(DAY)
        self.assertIsNotNone(container)
        pathlib.Path(str(container) + ".idx.json").unlink()
        with self.assertRaises(ChunkError):
            self.store.get_raw(self.ids[0])


class CardWriteTest(unittest.TestCase):
    def setUp(self) -> None:
        # 测试目录放在工作区内：系统临时目录在受限沙箱下可能不可写
        self.tmp = ROOT / ".tmp" / f"{type(self).__name__}.{self._testMethodName}"
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.root = self.tmp
        self.clock = FixedClock(datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc))
        self.store = Store(self.root, clock=self.clock)
        self.store_id = self.store.append_message("s1", "决定采用方案B，预算12万，截止2026-10-15。")

    def tearDown(self) -> None:
        self.store.close()
        # 清理失败（Windows 文件锁 / 沙箱权限）不应影响测试结论
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _card(self, **overrides: object) -> dict:
        payload = {
            "clock": self.clock,
            "session_id": "s1",
            "tier": "T2",
            "importance": 8,
            "network": "experience",
            "source_ids": [self.store_id],
            "summary": "决定采用方案B，预算12万，截止2026-10-15。",
            "entities": ["项目A", "张三"],
            "facts": ["预算12万", "截止2026-10-15"],
            "decisions": ["采用方案B"],
            "todos": ["张三9月30日前给报价"],
            "constraints": ["必须兼容旧版API"],
            "claims": [
                {"kind": "decision", "subject_entity": "项目A", "predicate": "方案", "object": "方案B"},
                {"kind": "todo", "subject_entity": "张三", "predicate": "报价", "object": "报价", "due_at": "2026-09-30"},
            ],
        }
        payload.update(overrides)
        return make_card(**payload)  # type: ignore[arg-type]

    def test_card_is_content_addressed_and_idempotent(self) -> None:
        first = self._card()
        second = self._card()
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(first["dedup_key"], second["dedup_key"])
        self.assertRegex(first["id"], r"^mem_[0-9]{8}_[a-z2-7]{10}$")

        self.assertTrue(self.store.put_card(first))
        self.assertFalse(self.store.put_card(second), "重复压缩必须是无副作用的重放")
        self.assertEqual(self.store.stats()["T2"], 1)

        stored = self.store.get_card(first["id"])
        assert stored is not None
        self.assertEqual(stored["tier"], "T2")
        self.assertEqual(stored["half_life_days"], 180.0)
        self.assertEqual(stored["facts"], ["预算12万", "截止2026-10-15"])
        self.assertEqual(len(stored["claims"]), 2)

    def test_source_ids_survive_the_round_trip(self) -> None:
        """回链是核心承诺：卡存进库后必须还能查到它来自哪条原文。"""
        card = self._card()
        self.store.put_card(card)
        stored = self.store.get_card(card["id"])
        assert stored is not None
        self.assertEqual(stored["source_ids"], [self.store_id])
        self.assertEqual(self.store.sources_of_card(card["id"]), [self.store_id])
        self.assertEqual(self.store.cards_citing(self.store_id), [card["id"]])

    def test_same_span_can_yield_cards_of_different_tiers(self) -> None:
        """同一段原文合法地抽出多张卡（决策 / 待办），不许互相覆盖。"""
        decision = self._card(tier="T2", summary="决定采用方案B。")
        todo = self._card(tier="T3", summary="张三需在9月30日前给报价。")
        self.assertNotEqual(decision["id"], todo["id"])
        self.assertTrue(self.store.put_card(decision))
        self.assertTrue(self.store.put_card(todo))
        self.assertEqual(self.store.stats()["T2"], 1)
        self.assertEqual(self.store.stats()["T3"], 1)

    def test_summary_change_supersedes_previous_card(self) -> None:
        """换了提示词导致摘要变化：新卡取代旧卡，旧卡留痕而非删除。"""
        first = self._card(summary="决定采用方案B，预算12万，截止2026-10-15。")
        revised = self._card(summary="决定采用方案C，预算15万，截止2026-11-01。")
        self.assertNotEqual(first["id"], revised["id"])

        self.assertTrue(self.store.put_card(first))
        result = self.store.put_card(revised)
        self.assertTrue(result.inserted)
        self.assertEqual(result.superseded, (first["id"],))

        old = self.store.get_card(first["id"])
        new = self.store.get_card(revised["id"])
        assert old is not None and new is not None
        self.assertEqual(old["status"], "superseded")
        self.assertEqual(old["superseded_by"], revised["id"])
        self.assertEqual(new["supersedes"], [first["id"]])
        self.assertEqual(new["status"], "active")

    def test_schema_rejects_bad_cards(self) -> None:
        with self.assertRaises(SchemaError) as ctx:
            self._card(importance=11)
        self.assertIn("importance", str(ctx.exception))

        with self.assertRaises(SchemaError):
            validate_card({**self._card(), "id": "mem_1"})

        with self.assertRaises(SchemaError):
            validate_card({**self._card(), "unexpected": 1})

        with self.assertRaises(SchemaError):
            validate_card({**self._card(), "source_ids": []})

    def test_empty_source_ids_rejected_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            self._card(source_ids=[])


class CursorTest(unittest.TestCase):
    def setUp(self) -> None:
        # 测试目录放在工作区内：系统临时目录在受限沙箱下可能不可写
        self.tmp = ROOT / ".tmp" / f"{type(self).__name__}.{self._testMethodName}"
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.root = self.tmp
        self.clock = FixedClock(datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc))
        self.store = Store(self.root, clock=self.clock)

    def tearDown(self) -> None:
        self.store.close()
        # 清理失败（Windows 文件锁 / 沙箱权限）不应影响测试结论
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_pending_messages_respect_cursor(self) -> None:
        ids = [self.store.append_message("s1", f"消息 {i}") for i in range(3)]
        self.assertEqual(len(self.store.pending_messages("s1")), 3)

        self.store.set_cursor("s1", last_source_id=ids[-1], last_day=DAY, last_line=2)
        self.assertEqual(self.store.pending_messages("s1"), [])

        self.store.append_message("s1", "新消息")
        pending = self.store.pending_messages("s1")
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["line_no"], 3)

    def test_note_access_counts_only_real_injections(self) -> None:
        source = self.store.append_message("s1", "决定采用方案B")
        card = make_card(
            clock=self.clock,
            session_id="s1",
            tier="T2",
            importance=8,
            network="experience",
            source_ids=[source],
            summary="决定采用方案B",
        )
        self.store.put_card(card)

        self.assertEqual(self.store.note_access(card["id"], "s1"), 1)
        self.assertEqual(self.store.note_access(card["id"], "s2", render="lite"), 2)
        hits, sessions = self.store.access_stats(card["id"], since=datetime(2026, 1, 1, tzinfo=timezone.utc))
        self.assertEqual((hits, sessions), (2, 2))


if __name__ == "__main__":
    unittest.main()
