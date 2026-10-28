"""召回编排器（`recall.py`）的三条验收：正常召回 / 定性硬门全挡 / 预算逼降级。

为什么是这三条：实现里最容易被写错的就是这三处——

1. **两段硬门的顺序**：`θ` 是 `q` 的阈值，定量段只能在打分之后；而卡龄窗口与 `T8/T9`
   的载体禁令与分数无关，必须在打分之前。顺序写反了行为会「看起来正常」但口径全错。
2. **`T8/T9` 的显式回溯例外**：寒暄/噪音按定义可丢弃，只有 `explicit_recall` 才放行；
   而卡龄窗口对 `T4–T7` 是硬约束，显式回溯也不该放宽它（那条例外只给 `T8/T9`）。
3. **`full → lite` 降级**：装不下先降级、再丢弃；「降级」与「丢弃」是两个不同的计数，
   混在一起就看不见预算真正压掉了什么。
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from datetime import timedelta

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow import FixedClock, Store, make_card, recall  # noqa: E402
from moonshadow.clock import parse_ts, to_utc  # noqa: E402
from moonshadow.pack import estimate_tokens  # noqa: E402
from moonshadow.verify import card_text  # noqa: E402

NOW = to_utc(parse_ts("2026-10-01T09:00:00+00:00"))

#: 只有 `cards` 一个分区拿全部预算：把「配额」这个变量从降级测试里固定住，
#: 否则 `cards` 只有 40% 配额，小预算下会先撞配额、根本走不到 full→lite。
CARDS_ONLY = {"recent_raw": 0.0, "cards": 1.0, "todo": 0.0, "conflict": 0.0}


class RecallTest(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.store = Store(pathlib.Path(self._dir.name), clock=FixedClock(NOW))

    def tearDown(self) -> None:
        self.store.close()
        self._dir.cleanup()

    def _card(
        self,
        *,
        tier: str,
        summary: str,
        facts: tuple[str, ...] = (),
        importance: int = 8,
        age_days: int = 0,
        session: str = "s1",
    ) -> dict:
        """写一张卡（`observed_at` 由 clock 决定，故 `age_days` 用旧 clock 造）。"""
        moment = NOW - timedelta(days=age_days)
        source_id = self.store.append_message(session, f"原文：{summary}", ts=moment)
        card = make_card(
            clock=FixedClock(moment),
            session_id=session,
            tier=tier,
            importance=importance,
            network="world",
            source_ids=[source_id],
            summary=summary,
            facts=list(facts),
        )
        self.store.put_card(card)
        return card

    def _cos(self, score: float = 0.9) -> dict[str, float]:
        return {str(card["id"]): score for card in self.store.cards()}

    # ------------------------------------------------------------------ 1 正常召回

    def test_normal_recall_orders_by_score_and_reports_budget(self) -> None:
        self._card(tier="T3", summary="预算 12 万，截止 2026-10-15", facts=("预算 12 万",))
        self._card(tier="T5", summary="一般偏好：先做读缓存", age_days=1)

        out = recall(self.store, now=NOW, budget_tokens=5_000, query_cos=self._cos())

        # 返回形状是契约的一部分
        self.assertEqual(
            set(out), {"items", "remaining_budget", "stats", "blocks", "pack_result"}
        )
        stats = out["stats"]
        self.assertEqual(stats.cards, 2)
        self.assertEqual(stats.qualitative_blocked, 0)
        self.assertEqual(stats.quantitative_blocked, 0)
        self.assertEqual(stats.gated, 2)
        self.assertEqual(len(out["items"]), 2)
        self.assertEqual(stats.packed, 2)
        self.assertEqual(stats.degraded, 0)
        self.assertEqual(stats.dropped, 0)
        self.assertFalse(stats.over_budget)
        self.assertEqual(out["remaining_budget"], 5_000 - stats.tokens_used)

        scores = [item.score for item in out["items"]]
        self.assertEqual(scores, sorted(scores, reverse=True), "装箱顺序必须按分降序")
        self.assertTrue(out["blocks"], "装了东西就该有 blocks 可拼进上下文")

    # ------------------------------------------------------------ 2 定性硬门全挡

    def test_qualitative_gate_blocks_everything_without_explicit_recall(self) -> None:
        self._card(tier="T8", summary="寒暄：早", importance=1)
        self._card(tier="T9", summary="噪音：嗯", importance=1)
        self._card(tier="T5", summary="一条 40 天前的日常事实", age_days=40)

        out = recall(self.store, now=NOW, budget_tokens=5_000, query_cos=self._cos(0.99))

        self.assertEqual(out["items"], [])
        self.assertEqual(out["stats"].cards, 3)
        self.assertEqual(out["stats"].qualitative_blocked, 3)
        self.assertEqual(out["stats"].gated, 0)
        self.assertEqual(out["blocks"], [])

        # 显式回溯只对 T8/T9 放行；40 天前的 T5 仍被卡龄窗口挡住——
        # 那条例外**不**放宽卡龄窗口，这两件事必须分开测。
        again = recall(
            self.store,
            now=NOW,
            budget_tokens=5_000,
            query_cos=self._cos(0.99),
            explicit_recall=True,
        )
        self.assertEqual(again["stats"].qualitative_blocked, 1)
        self.assertEqual({item.tier for item in again["items"]}, {"T8", "T9"})

    # -------------------------------------------------------------- 3 预算逼降级

    def test_budget_forces_full_to_lite_downgrade(self) -> None:
        long_facts = tuple("这是一条很长的原始陈述，用来把 full 渲染撑大。" * 4 for _ in range(3))
        card = self._card(tier="T3", summary="短摘要：方案 B", facts=long_facts)
        card_id = str(card["id"])

        full_tokens = estimate_tokens(card_text(self.store.get_card(card_id)))
        lite_tokens = estimate_tokens("短摘要：方案 B")
        budget = lite_tokens + 1
        self.assertLess(budget, full_tokens, "这个预算必须装不下 full，否则测不到降级")

        out = recall(
            self.store,
            now=NOW,
            budget_tokens=budget,
            query_cos={card_id: 0.9},
            quotas=CARDS_ONLY,
        )

        from moonshadow.pack import Placement  # 局部导入：只为了断言类型

        placements = out["pack_result"].placements
        self.assertEqual(len(placements), 1)
        self.assertIsInstance(placements[0], Placement)
        self.assertEqual(placements[0].render, "lite")
        self.assertEqual(out["stats"].degraded, 1)
        self.assertEqual(out["stats"].dropped, 0, "能降级就不该丢弃")
        self.assertLessEqual(out["stats"].tokens_used, budget)
        self.assertEqual(out["stats"].remaining_budget, budget - out["stats"].tokens_used)


if __name__ == "__main__":
    unittest.main()
