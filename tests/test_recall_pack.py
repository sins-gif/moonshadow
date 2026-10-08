"""Phase 3 验收：预算装箱必须确定性、可解释、且绝不静默裁掉 T0/T1。

确定性测试的意义：同一批候选 + 同一预算，两次运行必须给出逐条相同的上下文。
否则「为什么这次没带上某条约束」在生产里无法复盘。
"""

from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow.pack import (  # noqa: E402
    QUOTAS,
    SECTIONS,
    Dropped,
    Item,
    Placement,
    estimate_tokens,
    pack,
)


def tok(text: str) -> int:
    """测试用估算器：1 字符 = 1 token，便于手算配额。"""
    return len(text)


def item(
    ident: str,
    section: str,
    *,
    full: int = 10,
    lite: int = 0,
    score: float = 1.0,
    tier: str = "T4",
    reserved: bool = False,
) -> Item:
    return Item(
        id=ident,
        section=section,
        tier=tier,
        score=score,
        text_full=ident.ljust(full, "x") if full else ident,
        text_lite=ident.ljust(lite, "y") if lite else "",
        reserved=reserved,
    )


class PackTest(unittest.TestCase):
    def test_quotas_are_normalized(self) -> None:
        self.assertAlmostEqual(sum(QUOTAS.values()), 1.0, places=12)

    def test_unknown_section_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pack([item("a", "nope")], 100, estimator=tok)

    def test_bad_quotas_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pack([], 100, estimator=tok, quotas={"cards": 0.5})

    def test_reserved_survives_over_budget(self) -> None:
        """T0/T1 无条件进入：宁可超预算，也不静默丢身份与硬约束。"""
        core = item("core", "long_term", full=300, tier="T0")
        result = pack([core, item("card", "cards", full=10)], 100, estimator=tok)
        self.assertTrue(result.over_budget)
        self.assertIn("core", [p.item.id for p in result.placements])
        # 常驻已超支 => 剩余预算为 0，其余分区整批不进（宁可少说，不可乱说）
        self.assertEqual(result.tokens_used, 300)
        self.assertEqual(
            [(d.item.id, d.reason) for d in result.dropped], [("card", "quota_exhausted")]
        )

    def test_lite_render_is_used_before_dropping(self) -> None:
        candidate = item("c1", "cards", full=60, lite=20)
        result = pack([candidate], 100, estimator=tok)  # cards 配额 = 40
        self.assertEqual(
            [(p.item.id, p.render) for p in result.placements], [("c1", "lite")]
        )
        self.assertEqual(result.dropped, [])

    def test_item_without_lite_is_dropped_with_reason(self) -> None:
        candidate = item("c1", "cards", full=60)
        result = pack([candidate], 100, estimator=tok)
        self.assertEqual(result.placements, [])
        self.assertEqual([(d.item.id, d.reason) for d in result.dropped], [("c1", "no_space")])

    def test_leftover_quota_rolls_into_next_section(self) -> None:
        """cards 装不下的大条目，其配额应滚给 todo，而不是白白浪费。"""
        result = pack(
            [item("big", "cards", full=500), item("task", "todo", full=500)],
            1000,
            estimator=tok,
        )
        placed = [p.item.id for p in result.placements]
        self.assertEqual(placed, ["task"])
        self.assertEqual([(d.item.id, d.reason) for d in result.dropped], [("big", "no_space")])
        self.assertLessEqual(result.tokens_used, 1000)

    def test_within_section_higher_score_wins(self) -> None:
        items = [
            item("low", "cards", full=30, score=0.2),
            item("high", "cards", full=30, score=0.9),
        ]
        result = pack(items, 100, estimator=tok)  # cards 配额 = 40，只装得下一条
        self.assertEqual([p.item.id for p in result.placements], ["high"])

    def test_pack_is_deterministic_and_tie_sorted_by_id(self) -> None:
        items = [
            item("bbb", "cards", full=20, score=0.5),
            item("aaa", "cards", full=20, score=0.5),
            item("ccc", "todo", full=20, score=0.5),
        ]
        first = pack(items, 200, estimator=tok)
        second = pack(list(reversed(items)), 200, estimator=tok)
        self.assertEqual(
            [p.item.id for p in first.placements], [p.item.id for p in second.placements]
        )
        self.assertEqual([p.item.id for p in first.placements][:2], ["aaa", "bbb"])

    def test_blocks_group_by_section_in_canonical_order(self) -> None:
        result = pack(
            [
                item("core", "long_term", full=5, tier="T0"),
                item("task", "todo", full=5),
                item("card", "cards", full=5),
            ],
            200,
            estimator=tok,
        )
        names = [name for name, _ in result.blocks()]
        self.assertEqual(names, ["long_term", "cards", "todo"])
        self.assertEqual(set(names) <= set(SECTIONS), True)

    def test_degraded_report(self) -> None:
        result = pack([item("c1", "cards", full=60, lite=20)], 100, estimator=tok)
        self.assertEqual([p.item.id for p in result.degraded()], ["c1"])

    def test_estimate_tokens_counts_cjk_per_char(self) -> None:
        self.assertEqual(estimate_tokens("预算十二万"), 5)
        self.assertEqual(estimate_tokens("abcdefgh"), 2)
        self.assertEqual(estimate_tokens(""), 0)

    def test_budget_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            pack([], 0, estimator=tok)


if __name__ == "__main__":
    unittest.main()
