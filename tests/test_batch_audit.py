"""批次验收规则（`docs/v1.2-roadmap.md` §7.1 的 R1 / R2）自身的测试。

规则是**硬性**的（不满足即拒绝合入），所以判定逻辑必须被测试——
否则「拒绝合入」可能只是一个永远通过的装饰。
"""

from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from moonshadow.rank_eval import load_holdout_cases, load_rank_cases  # noqa: E402
from run_batch_audit import (  # noqa: E402
    BOUNDARY_GAP,
    MIN_BOUNDARY,
    audit,
    adjacent_min_gap,
    evaluate,
)


def pick(prefixes: list[str]):
    """批次与切分无关：同一批用例可能被分到任一侧，因此两侧都查。"""
    cases = load_rank_cases() + load_holdout_cases()
    return [case for case in cases if case.id[:2] in prefixes]


def pick_dev(prefixes: list[str]):
    return [case for case in load_rank_cases() if case.id[:2] in prefixes]


class BatchAuditTest(unittest.TestCase):
    """判定量本身。"""

    def test_first_batch_is_rejected_after_the_resplit_and_the_new_floor(self) -> None:
        """第一批（12–21）的实测：**R1 不通过、R2 通过**，且开发集侧两条都不通过。

        这是**事后**判定，记录它有两个用处：
        1. 证明判定量可算、可复现；
        2. 证明 R1 真的会失败——规则若从不失败，就只是装饰。

        读数变化的原因要写清：`20` 的相邻分差随采纳下限变化（`0.0497` → `0.0517` → `0.0533`），
        现已**越过** `0.05` 门槛，因此它不算边界用例；而提供约束的 `17`/`21` 被随机分到了留出集，
        开发集侧 7 条**全部**无约束（产率 0）。**分差是 `f` 的函数**——换参数会改变
        一条用例算不算「边界用例」。
        """
        batch = pick([str(i).zfill(2) for i in range(12, 22)])
        verdict = evaluate(batch)

        self.assertEqual(len(verdict.rows), 10)
        self.assertEqual(
            verdict.boundary,
            ["14-recent-draft-vs-older-fact", "17-explicit-recall-admits-t8"],
        )
        self.assertLess(len(verdict.boundary), MIN_BOUNDARY)
        self.assertEqual(
            verdict.constrained,
            ["17-explicit-recall-admits-t8", "21-similar-relevance-newer-wins"],
        )
        self.assertAlmostEqual(verdict.yield_rate, 0.2, places=9)
        self.assertFalse(verdict.passed)
        self.assertTrue(any("R1" in failure for failure in verdict.failures))
        self.assertFalse(any("R2" in failure for failure in verdict.failures))

    def test_dev_side_of_that_batch_constrains_nothing(self) -> None:
        """开发集侧 7 条全部无约束——「整批产率 0.2」不能读成「开发集也变好了」。"""
        dev_batch = pick_dev([str(i).zfill(2) for i in range(12, 22)])
        self.assertEqual(len(dev_batch), 7)
        verdict = evaluate(dev_batch)
        self.assertEqual(verdict.constrained, [])
        self.assertEqual(verdict.yield_rate, 0.0)
        self.assertTrue(any("R2" in failure for failure in verdict.failures))

    def test_boundary_rule_does_not_imply_constraint(self) -> None:
        """R1 只是预测指标：分差小的用例仍可能对 `f` 毫无约束。

        `14` 的相邻分差 `0.0089` 在门槛内，但可行区间宽度仍是 `1.000`。
        这条断言防止把 R1 当成「有约束」的替代品。
        """
        batch = pick([str(i).zfill(2) for i in range(12, 22)])
        verdict = evaluate(batch)
        by_id = {row.case_id: row for row in verdict.rows}

        row = by_id["14-recent-draft-vs-older-fact"]
        self.assertTrue(row.is_boundary)
        self.assertFalse(row.is_constraining, "分差小，但对 f 无约束")
        self.assertTrue(by_id["17-explicit-recall-admits-t8"].is_constraining)

    def test_rule_1_is_falsifiable(self) -> None:
        """一批分数相差很大的用例必须被判 R1 不通过。"""
        batch = pick(["04", "09", "11"])
        verdict = evaluate(batch)
        self.assertLess(len(verdict.boundary), MIN_BOUNDARY)
        self.assertFalse(verdict.passed)
        self.assertTrue(any("R1" in failure for failure in verdict.failures))

    def test_rule_2_is_falsifiable(self) -> None:
        """一批对 `f` 没有约束的用例必须被判 R2 不通过（产率为 0）。"""
        batch = pick(["04", "09", "11"])
        verdict = evaluate(batch)
        self.assertEqual(verdict.constrained, [])
        self.assertEqual(verdict.yield_rate, 0.0)
        self.assertTrue(any("R2" in failure for failure in verdict.failures))

    def test_single_candidate_case_has_no_adjacent_pair(self) -> None:
        """只有一张正例的用例不存在相邻对，因此不可能是边界用例。"""
        (case,) = pick(["15"])
        self.assertEqual(len(case.candidates), 1)
        self.assertIsNone(adjacent_min_gap(case))

    def test_threshold_constant_matches_the_documented_rule(self) -> None:
        """规则里的阈值必须与文档一致，改动时两处一起改。"""
        self.assertEqual(BOUNDARY_GAP, 0.05)
        self.assertEqual(MIN_BOUNDARY, 3)
        roadmap = (ROOT / "docs" / "v1.2-roadmap.md").read_text(encoding="utf-8")
        self.assertIn("至少 3 条候选分差 ≤ 0.05 的边界用例", roadmap)
        self.assertIn("约束产率为 0，则拒绝合入", roadmap)

    def test_cli_rejects_unknown_batches(self) -> None:
        """未知前缀必须被拒绝——静默通过会让规则失效。"""
        self.assertEqual(audit(["99-does-not-exist"]), 1)

    def test_bare_invocation_judges_the_post_freeze_batch(self) -> None:
        """不带参数 = 判定「冻结后新增的开发用例」这一批，而不是报错。

        判定「这一批新用例能不能合入」时手输 id 容易漏带/多带；而「哪些是冻结之后
        加进来的」有确切答案（切分清单里的 `dev` 快照），所以默认行为由它定义。
        当前冻结后新增的是 `22`（T9 噪声，无约束）与 `23`/`24`/`25`（三条边界用例）：
        R1 应按 3 条边界用例通过，R2 产率 0.750。
        """
        from run_batch_audit import default_batch, evaluate

        added, note = default_batch(load_rank_cases())
        self.assertIn("冻结后新增", note)
        self.assertEqual(
            [case.id for case in added],
            ["22-t9-noise-blocked-by-tier", "23-t6-tight-race", "24-t7-borderline",
             "25-t8-explicit-recall"],
        )
        verdict = evaluate(added)
        self.assertEqual(len(verdict.boundary), 3)
        self.assertAlmostEqual(verdict.yield_rate, 0.75, places=9)
        self.assertTrue(verdict.passed, verdict.failures)
        self.assertEqual(audit([]), 0, "裸调用在最坏情况下也应由规则判定，而非报用法")


if __name__ == "__main__":
    unittest.main()
