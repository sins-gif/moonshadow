"""Phase 2.5 验收：打分公式与硬门必须是确定性的、可复现的。

覆盖 v1.0 修正过的三处：
- importance/10 归一化（原式 priority/5 会让分数越界）
- 半衰期语义（0.5 ** (delta/half_life)，而非 exp(-delta/half_life)）
- 硬门阈值与层级升降的夹逼规则
"""

from __future__ import annotations

import pathlib
import sys
import unittest
from datetime import datetime, timedelta, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow import scoring  # noqa: E402
from moonshadow.scoring import (  # noqa: E402
    THETA,
    THETA_HIGH,
    WEIGHTS,
    decay,
    match_score,
    next_tier,
    passes_gate,
    score_card,
    select,
    sim_norm,
)

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def make_stub(
    *,
    card_id: str = "mem_20260601_aaaaaaaaaa",
    tier: str = "T4",
    importance: int = 5,
    days_old: float = 0.0,
    entities: tuple[str, ...] = ("项目A",),
) -> dict:
    observed = NOW - timedelta(days=days_old)
    return {
        "id": card_id,
        "tier": tier,
        "importance": importance,
        "entities": list(entities),
        "observed_at": observed.isoformat(timespec="seconds"),
        "half_life_days": scoring.default_half_life(tier),
        "status": "active",
    }


class FormulaTest(unittest.TestCase):
    def test_weights_are_normalized(self) -> None:
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0, places=12)

    def test_decay_is_a_real_half_life(self) -> None:
        self.assertAlmostEqual(decay(180.0, 180.0), 0.5, places=12)
        self.assertAlmostEqual(decay(360.0, 180.0), 0.25, places=12)
        self.assertAlmostEqual(decay(0.0, 180.0), 1.0, places=12)

    def test_zero_half_life_never_decays(self) -> None:
        self.assertEqual(decay(10_000.0, 0.0), 1.0)

    def test_sim_norm_maps_cosine_range(self) -> None:
        self.assertEqual(sim_norm(1.0), 1.0)
        self.assertEqual(sim_norm(-1.0), 0.0)
        self.assertAlmostEqual(sim_norm(0.0), 0.5, places=12)
        with self.assertRaises(ValueError):
            sim_norm(1.5)

    def test_match_score_halves(self) -> None:
        self.assertEqual(match_score(["张三"], ["张三"], True), 1.0)
        self.assertEqual(match_score(["张三"], ["李四"], False), 0.0)
        self.assertEqual(match_score([], [], False), 0.0)
        self.assertEqual(match_score(["a", "b"], ["b"], False), 0.25)  # Jaccard 1/2，折半

    def test_score_is_bounded_and_saturates_at_one(self) -> None:
        card = make_stub(tier="T1", importance=10, days_old=0.0)
        top = score_card(
            card, now=NOW, query_cos=1.0, query_entities=["项目A"], task_hit=True
        )
        self.assertAlmostEqual(top, 1.0, places=12)

        # 最坏情形：相似度为 0、实体不匹配、时间几乎衰减殆尽、重要度最低
        worst = make_stub(tier="T4", importance=1, days_old=10_000.0)
        bottom = score_card(
            worst, now=NOW, query_cos=-1.0, query_entities=[], task_hit=False
        )
        self.assertGreaterEqual(bottom, 0.0)
        # 只剩 importance 一项：0.20 * (1 / 10)
        self.assertAlmostEqual(bottom, 0.02, places=9)

    def test_importance_uses_ten_point_scale(self) -> None:
        low = make_stub(importance=1, days_old=0.0)
        high = make_stub(importance=10, days_old=0.0)
        kwargs = {"now": NOW, "query_cos": 0.0, "query_entities": [], "task_hit": False}
        gap = score_card(high, **kwargs) - score_card(low, **kwargs)
        self.assertAlmostEqual(gap, 0.20 * 0.9, places=12)

    def test_bad_weights_rejected(self) -> None:
        with self.assertRaises(ValueError):
            score_card(make_stub(), now=NOW, weights={"sim": 0.9, "decay": 0.9, "importance": 0.9, "match": 0.9})


class GateTest(unittest.TestCase):
    def test_t0_t1_are_always_on(self) -> None:
        for tier in ("T0", "T1"):
            self.assertTrue(
                passes_gate(tier, delta_days=99_999.0, score=0.0, explicit_recall=False)
            )

    def test_t2_t3_need_score_or_open_task(self) -> None:
        self.assertTrue(passes_gate("T3", delta_days=10.0, score=THETA + 0.01))
        self.assertFalse(passes_gate("T3", delta_days=10.0, score=THETA - 0.01))
        self.assertTrue(
            passes_gate("T3", delta_days=10.0, score=0.0, has_open_task=True)
        )

    def test_t4_t5_have_a_thirty_day_window(self) -> None:
        self.assertTrue(passes_gate("T4", delta_days=30.0, score=0.99))
        self.assertFalse(passes_gate("T4", delta_days=30.01, score=0.99))

    def test_t6_t7_have_a_seven_day_window_and_higher_theta(self) -> None:
        self.assertTrue(passes_gate("T6", delta_days=7.0, score=THETA_HIGH + 0.01))
        self.assertFalse(passes_gate("T6", delta_days=7.0, score=THETA_HIGH - 0.01))
        self.assertFalse(passes_gate("T6", delta_days=7.01, score=0.99))

    def test_t8_t9_require_explicit_recall(self) -> None:
        self.assertFalse(passes_gate("T8", delta_days=0.0, score=1.0))
        self.assertTrue(passes_gate("T9", delta_days=0.0, score=0.0, explicit_recall=True))

    def test_unknown_tier_raises(self) -> None:
        with self.assertRaises(ValueError):
            passes_gate("T99", delta_days=0.0, score=1.0)


class SelectTest(unittest.TestCase):
    def test_select_filters_by_gate_and_sorts_by_score(self) -> None:
        fresh_t4 = make_stub(card_id="mem_20260601_bbbbbbbbbb", tier="T4", importance=9, days_old=1.0)
        stale_t4 = make_stub(card_id="mem_20260601_cccccccccc", tier="T4", importance=9, days_old=31.0)
        noise_t8 = make_stub(card_id="mem_20260601_dddddddddd", tier="T8", importance=1, days_old=0.1)
        core_t0 = make_stub(card_id="mem_20260601_eeeeeeeeee", tier="T0", importance=10, days_old=500.0)

        rows = select(
            [fresh_t4, stale_t4, noise_t8, core_t0],
            now=NOW,
            query_entities=["项目A"],
            query_cos={fresh_t4["id"]: 0.9, stale_t4["id"]: 0.9, noise_t8["id"]: 0.0},
        )
        ids = [row["card"]["id"] for row in rows]
        self.assertIn(fresh_t4["id"], ids)
        self.assertIn(core_t0["id"], ids)
        self.assertNotIn(stale_t4["id"], ids)   # 超出 30 天窗口
        self.assertNotIn(noise_t8["id"], ids)   # 未声明回溯
        self.assertEqual(ids[0], fresh_t4["id"])  # 分数降序

    def test_explicit_recall_lets_noise_through(self) -> None:
        noise = make_stub(card_id="mem_20260601_ffffffffff", tier="T8", importance=1, days_old=0.1)
        rows = select([noise], now=NOW, explicit_recall=True)
        self.assertEqual([row["card"]["id"] for row in rows], [noise["id"]])


class TierLifecycleTest(unittest.TestCase):
    def test_promotion_needs_access_and_sessions(self) -> None:
        self.assertEqual(
            next_tier("T3", access_count_30d=5, sessions_30d=3, idle_days=0.0, half_life_days=90.0),
            "T2",
        )
        self.assertEqual(
            next_tier("T3", access_count_30d=5, sessions_30d=2, idle_days=0.0, half_life_days=90.0),
            "T3",
        )

    def test_promotion_never_enters_t0_t1(self) -> None:
        self.assertEqual(
            next_tier("T2", access_count_30d=99, sessions_30d=99, idle_days=0.0, half_life_days=180.0),
            "T2",
        )

    def test_t0_t1_are_immune_to_rules(self) -> None:
        for tier in ("T0", "T1"):
            self.assertEqual(
                next_tier(tier, access_count_30d=0, sessions_30d=0, idle_days=99_999.0, half_life_days=0.0),
                tier,
            )

    def test_demotion_needs_two_half_lives_of_idleness(self) -> None:
        self.assertEqual(
            next_tier("T4", access_count_30d=0, sessions_30d=0, idle_days=61.0, half_life_days=30.0),
            "T5",
        )
        self.assertEqual(
            next_tier("T4", access_count_30d=0, sessions_30d=0, idle_days=59.0, half_life_days=30.0),
            "T4",
        )

    def test_open_task_blocks_demotion(self) -> None:
        self.assertEqual(
            next_tier(
                "T3", access_count_30d=0, sessions_30d=0, idle_days=9_999.0,
                half_life_days=90.0, has_open_task=True,
            ),
            "T3",
        )

    def test_demotion_floor_is_t8(self) -> None:
        self.assertEqual(
            next_tier("T8", access_count_30d=0, sessions_30d=0, idle_days=99.0, half_life_days=1.0),
            "T8",
        )


if __name__ == "__main__":
    unittest.main()
