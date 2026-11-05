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
    ALLOWED_STEPS,
    THETA,
    THETA_HIGH,
    WEIGHTS,
    decay,
    match_score,
    next_step,
    next_tier,
    passes_gate,
    score_card,
    select,
    sim_norm,
    step_votes,
    stepped_tier,
)
from moonshadow.tiers import TIER_ORDER  # noqa: E402

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

    def test_t0_is_immutable_but_t1_has_no_upper_level(self) -> None:
        """`T0` 任何证据都不动；`T1` 只是「没有上级」，不代表它不能降。

        旧版这条写的是「T0/T1 永不参与规则升降」——**那是旧政策**。新政策（按用户的
        「楼房」设计）只有 `T0` 固定；`T1` 若永不下降，常驻区就只增不减，会把预算吃光。
        """
        for evidence in (EVIDENCE_UP, EVIDENCE_NONE, EVIDENCE_DOWN):
            self.assertEqual(
                next_tier(
                    "T0",
                    access_count_30d=evidence["access_count_30d"],
                    sessions_30d=evidence["sessions_30d"],
                    span_days=evidence["span_days"],
                    idle_days=evidence["idle_days"],
                    half_life_days=evidence["half_life_days"],
                ),
                "T0",
            )
        # T1 没有上级：强升级证据也停在 T1
        self.assertEqual(
            next_tier("T1", span_days=45.0, access_count_30d=99, sessions_30d=99,
                      idle_days=0.0, half_life_days=365.0),
            "T1",
        )
        # 但闲置够久可以降（半衰期 365 天 → 需闲置 >730 天）
        self.assertEqual(
            next_tier("T1", access_count_30d=0, sessions_30d=0, idle_days=800.0, half_life_days=365.0),
            "T2",
        )
        self.assertEqual(
            next_tier("T1", access_count_30d=0, sessions_30d=0, idle_days=700.0, half_life_days=365.0),
            "T1",
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

    def test_demotion_floor_is_t8_and_t9_never_falls(self) -> None:
        """两条边界一起钉：`T8` 不许降成 `T9`；`T9` 不许「降」成 `T8`（旧 bug）。"""
        self.assertEqual(
            next_tier("T8", access_count_30d=0, sessions_30d=0, idle_days=99.0, half_life_days=1.0),
            "T8",
        )
        self.assertEqual(
            next_tier("T9", access_count_30d=0, sessions_30d=0, idle_days=999.0, half_life_days=1.0 / 24),
            "T9",
        )
        # 旧实现在这一格返回 T8（方向反了）——这条断言就是那个 bug 的墓志铭
        self.assertNotEqual(
            next_tier("T9", access_count_30d=0, sessions_30d=0, idle_days=999.0, half_life_days=1.0 / 24),
            "T8",
        )

    def test_next_tier_delegates_to_step_and_landing(self) -> None:
        """`next_tier` 必须只是 `next_step` + `stepped_tier` 的组合，不许有第二套逻辑。"""
        for tier in TIER_ORDER:
            for evidence in (EVIDENCE_UP, EVIDENCE_NONE, EVIDENCE_DOWN):
                step = next_step(tier, **evidence)
                expected = tier if step == 0 else stepped_tier(tier, step)
                self.assertEqual(
                    next_tier(
                        tier,
                        access_count_30d=evidence["access_count_30d"],
                        sessions_30d=evidence["sessions_30d"],
                        span_days=evidence["span_days"],
                        idle_days=evidence["idle_days"],
                        half_life_days=evidence["half_life_days"],
                    ),
                    expected,
                    f"{tier} 的 next_tier 与内核不一致",
                )


#: 三值楼梯的三组证据。每一票都要能指出出处，所以三组证据是**手工构造**的：
#:   UP   = 访问够、跨会话够、跨度够、最近还在用 → 4 票 ≥ +3 → 升
#:   NONE = 只满足「最近还在用」 → 1 票，落在死区 `-1..+2` → 不动
#:   DOWN = 30 天零访问且闲置超 2×半衰期 → -2 票 → 降
EVIDENCE_UP = dict(access_count_30d=12, sessions_30d=5, span_days=45.0, idle_days=0.0, half_life_days=90.0)
EVIDENCE_NONE = dict(access_count_30d=1, sessions_30d=1, span_days=2.0, idle_days=1.0, half_life_days=90.0)
EVIDENCE_DOWN = dict(access_count_30d=0, sessions_30d=0, span_days=1.0, idle_days=999.0, half_life_days=7.0)

#: 10×3 真值表：`ALLOWED_STEPS` 是「法律」，这张表是「判例」——两者必须一致。
EXPECTED_TRUTH_TABLE = {
    "T0": (0, 0, 0),      # 身份/硬约束：任何证据都不动
    "T1": (0, 0, -1),     # 没有上级，但闲置够久可以下去
    "T2": (0, 0, -1),     # 现在上不去 T1（那道「钥匙」还没落地）
    "T3": (1, 0, -1),
    "T4": (1, 0, -1),
    "T5": (1, 0, -1),
    "T6": (1, 0, -1),
    "T7": (1, 0, -1),
    "T8": (1, 0, 0),      # 降级下限是 T8：不许自动降成 T9（tiers.py:42-43）
    "T9": (1, 0, 0),      # 已在最底，不许再降（旧实现会错成 T8）
}


class TierStepTest(unittest.TestCase):
    """三值楼梯（`next_step`）：10×3 真值表 + 三个边界。

    它不替换 `next_tier`，是纯新增的方向判定内核；所以这一组测试只证明
    「规则本身是自洽的」，不改变任何现有行为。
    """

    def test_truth_table_covers_all_ten_tiers(self) -> None:
        self.assertEqual(sorted(EXPECTED_TRUTH_TABLE), sorted(TIER_ORDER))
        for tier in TIER_ORDER:
            up, neutral, down = EXPECTED_TRUTH_TABLE[tier]
            self.assertEqual(next_step(tier, **EVIDENCE_UP), up, f"{tier} 在 UP 证据下应得 {up}")
            self.assertEqual(next_step(tier, **EVIDENCE_NONE), neutral, f"{tier} 在 NONE 证据下应得 {neutral}")
            self.assertEqual(next_step(tier, **EVIDENCE_DOWN), down, f"{tier} 在 DOWN 证据下应得 {down}")

    def test_every_allowed_step_is_reachable_and_nothing_else(self) -> None:
        """法律与判例一致：每层能走的方向，恰好是它被允许的方向。"""
        for tier in TIER_ORDER:
            reached = {
                next_step(tier, **EVIDENCE_UP),
                next_step(tier, **EVIDENCE_NONE),
                next_step(tier, **EVIDENCE_DOWN),
            }
            self.assertEqual(reached, set(ALLOWED_STEPS[tier]), f"{tier} 的可达步进与 ALLOWED_STEPS 不一致")

    def test_votes_are_integers_with_explicit_provenance(self) -> None:
        self.assertEqual(step_votes(**EVIDENCE_UP), 4)
        self.assertEqual(step_votes(**EVIDENCE_NONE), 1)
        self.assertEqual(step_votes(**EVIDENCE_DOWN), -2)
        # 少一票（缺跨度）仍是 3 票，刚好到升级门限——所以「差一项」还升得动
        self.assertEqual(step_votes(**{**EVIDENCE_UP, "span_days": 0.0}), 3)
        # 少两票才掉进死区（+2 < +3）
        self.assertEqual(
            step_votes(**{**EVIDENCE_UP, "span_days": 0.0, "idle_days": 999.0, "half_life_days": 1.0}),
            2,
        )

    def test_dead_band_prevents_oscillation(self) -> None:
        """票数在 +2 / -1 之间来回时，层级必须纹丝不动。"""
        for votes_evidence in (
            dict(access_count_30d=5, sessions_30d=3, span_days=0.0, idle_days=0.0, half_life_days=90.0),   # +3 → 升
            dict(access_count_30d=5, sessions_30d=3, span_days=0.0, idle_days=500.0, half_life_days=90.0),  # +2 → 不动
            dict(access_count_30d=0, sessions_30d=0, span_days=45.0, idle_days=500.0, half_life_days=90.0),  # -1 → 不动
        ):
            for tier in ("T5", "T6", "T7"):
                expected = 1 if step_votes(**votes_evidence) >= 3 else 0
                self.assertEqual(next_step(tier, **votes_evidence), expected)

    def test_t0_is_immutable_under_any_evidence(self) -> None:
        for evidence in (EVIDENCE_UP, EVIDENCE_NONE, EVIDENCE_DOWN):
            self.assertEqual(next_step("T0", **evidence), 0)
        self.assertEqual(
            next_step("T0", access_count_30d=9_999, sessions_30d=9_999, span_days=9_999.0,
                      idle_days=9_999.0, half_life_days=0.0),
            0,
        )

    def test_t9_never_falls(self) -> None:
        """越界归零而不是夹逼：这是「T9 反向降级」那个 bug 的根治方式。"""
        for evidence in (EVIDENCE_UP, EVIDENCE_NONE, EVIDENCE_DOWN):
            self.assertIn(next_step("T9", **evidence), (0, 1))

    def test_open_task_blocks_demotion_only(self) -> None:
        self.assertEqual(next_step("T3", **EVIDENCE_DOWN), -1)
        self.assertEqual(next_step("T3", has_open_task=True, **EVIDENCE_DOWN), 0)
        self.assertEqual(next_step("T3", has_open_task=True, **EVIDENCE_UP), 1)   # 升不受影响

    def test_unknown_tier_raises(self) -> None:
        with self.assertRaises(ValueError):
            next_step("T99", **EVIDENCE_NONE)

    def test_zero_half_life_gets_no_idle_votes(self) -> None:
        """半衰期为 0 表示不衰减：既不该白拿「最近还在用」那一票，也不该被闲置惩罚。"""
        self.assertEqual(
            step_votes(access_count_30d=0, sessions_30d=0, span_days=0.0, idle_days=0.0, half_life_days=0.0),
            0,
        )
        self.assertEqual(
            step_votes(access_count_30d=0, sessions_30d=0, span_days=0.0, idle_days=9_999.0, half_life_days=0.0),
            0,
        )


#: **逐层落点表**：每一层允许的每个步进，落到哪一层。
#:
#: 为什么要逐层写死而不是「索引减一」了事：旧实现的 bug 正藏在「T2..T8 一刀切」的夹逼里——
#: `T9` 下降被夹成 `T8`（方向反了，还不报错）。这张表让 T0–T9 **每一层的落点都被显式钉住**，
#: 任何一层想悄悄变成邻层之外的层级，测试就会红。
EXPECTED_LANDING: dict[str, dict[int, str]] = {
    "T0": {0: "T0"},                                  # 不升不降
    "T1": {0: "T1", -1: "T2"},                        # 没有上级；降 → 紧邻 T2
    "T2": {0: "T2", -1: "T3"},                        # 升 T1 的门未开；降 → 紧邻 T3
    "T3": {1: "T2", 0: "T3", -1: "T4"},
    "T4": {1: "T3", 0: "T4", -1: "T5"},
    "T5": {1: "T4", 0: "T5", -1: "T6"},
    "T6": {1: "T5", 0: "T6", -1: "T7"},
    "T7": {1: "T6", 0: "T7", -1: "T8"},
    "T8": {1: "T7", 0: "T8"},                         # 降级下限：不能再降成 T9
    "T9": {1: "T8", 0: "T9"},                         # 已在最底：不许再降
}


class TierLandingTest(unittest.TestCase):
    """逐层落点：`stepped_tier` 必须只走到**紧邻**层级，越界必须报错。"""

    def test_every_tier_lands_exactly_where_the_table_says(self) -> None:
        self.assertEqual(sorted(EXPECTED_LANDING), sorted(TIER_ORDER))
        for tier in TIER_ORDER:
            self.assertEqual(
                set(EXPECTED_LANDING[tier]), set(ALLOWED_STEPS[tier]),
                f"{tier} 的落点表与 ALLOWED_STEPS 不一致",
            )
            for step, expected in EXPECTED_LANDING[tier].items():
                self.assertEqual(stepped_tier(tier, step), expected, f"{tier} 走 {step:+d} 应落到 {expected}")

    def test_every_move_is_adjacent_never_skips(self) -> None:
        """一步只能跨一层：不许 T4 一步跳到 T6，也不许 T9「降」成 T8。"""
        for tier in TIER_ORDER:
            for step in ALLOWED_STEPS[tier]:
                landing = stepped_tier(tier, step)
                gap = abs(TIER_ORDER.index(landing) - TIER_ORDER.index(tier))
                self.assertEqual(gap, abs(step), f"{tier} 走 {step:+d} 跨了 {gap} 层，不是紧邻")

    def test_up_then_down_is_identity(self) -> None:
        """上楼再下楼必须回到原层——这条能抓出任何一侧的方向写反。"""
        for tier in TIER_ORDER:
            if 1 in ALLOWED_STEPS[tier]:
                up = stepped_tier(tier, 1)
                if -1 in ALLOWED_STEPS[up]:
                    self.assertEqual(stepped_tier(up, -1), tier, f"{tier} 上楼再下楼没回到原层")

    def test_disallowed_step_raises_instead_of_clamping(self) -> None:
        """旧实现在这里静默夹逼；新实现必须抛错（这就是 T9 bug 的根因）。"""
        with self.assertRaises(ValueError):
            stepped_tier("T0", -1)      # T0 不许降
        with self.assertRaises(ValueError):
            stepped_tier("T0", 1)       # T0 不许升
        with self.assertRaises(ValueError):
            stepped_tier("T9", -1)      # T9 不许降（旧实现返回 T8）
        with self.assertRaises(ValueError):
            stepped_tier("T1", 1)       # T1 没有上级
        with self.assertRaises(ValueError):
            stepped_tier("T2", 1)       # T2 的门还没开
        with self.assertRaises(ValueError):
            stepped_tier("T5", 2)       # 没有「一次两层」这种东西

    def test_t9_would_have_to_go_up_not_down(self) -> None:
        """回归钉子：T9 唯一的非零步进是 +1（去 T8），不可能是 -1。"""
        self.assertEqual(ALLOWED_STEPS["T9"], frozenset({0, 1}))
        self.assertNotIn(-1, ALLOWED_STEPS["T9"])
        self.assertEqual(stepped_tier("T9", 1), "T8")

    def test_t8_is_the_demotion_floor(self) -> None:
        """`tiers.py:42-43` 的政策：降级下限是 T8，绝不自动降成可丢弃的 T9。

        这一条是我第一版**写错过**的地方：`T8` 曾允许 `-1`，于是「寒暄」会被规则降成
        「噪音」。逐层落点表跑出来才发现——所以这里留一条钉子。
        """
        self.assertNotIn(-1, ALLOWED_STEPS["T8"])
        self.assertEqual(ALLOWED_STEPS["T8"], frozenset({0, 1}))
        self.assertEqual(stepped_tier("T8", 1), "T7")
        with self.assertRaises(ValueError):
            stepped_tier("T8", -1)

    def test_t4_to_t7_staircase_is_symmetric(self) -> None:
        """T4–T7 逐层的上下对称性（降级下限之上那一段）。"""
        for index in range(3, 8):          # T3..T7
            tier = TIER_ORDER[index]
            self.assertEqual(stepped_tier(tier, 1), TIER_ORDER[index - 1], f"{tier} 上楼落点错")
            self.assertEqual(stepped_tier(tier, -1), TIER_ORDER[index + 1], f"{tier} 下楼落点错")
            self.assertEqual(stepped_tier(tier, 0), tier, f"{tier} 原地不该动")


if __name__ == "__main__":
    unittest.main()
