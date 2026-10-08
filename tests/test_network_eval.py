"""网络赋值评测的守卫。

v1.2 提案要用 `network` 门控时间下限 `f`，因此这个准确率是那条路线的前置指标。
本文件把实测结果与结构性事实钉住：**若哪天基线能正确输出 `observation`，
说明赋值规则被改进了，`f` 的网络门控才有讨论基础。**
"""

from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow.compress import NETWORKS, baseline_network, classify_tier  # noqa: E402
from moonshadow.network_eval import (  # noqa: E402
    classify,
    evaluate_network,
    format_confusion,
    load_network_cases,
)


class NetworkCaseSetTest(unittest.TestCase):
    def test_cases_cover_all_four_networks(self) -> None:
        cases = load_network_cases()
        self.assertGreaterEqual(len(cases), 20)
        covered = {case.expected for case in cases}
        self.assertEqual(covered, set(NETWORKS), "四个网络都必须有标注用例")
        for case in cases:
            self.assertIn(case.expected, NETWORKS, case.id)
            self.assertTrue(case.text, case.id)

    def test_boundary_cases_are_marked_and_explained(self) -> None:
        for case in load_network_cases():
            if case.boundary:
                self.assertTrue(case.note, f"{case.id} 标为边界用例就必须写明理由")


class BaselineNetworkTest(unittest.TestCase):
    def test_baseline_never_predicts_observation(self) -> None:
        """结构性事实：规则里没有 `observation` 分支。

        v1.2 提案给 `observation` 配的是最激进的遗忘参数（f 0.20–0.45），
        而该类别在基线上不可达——这是那条路线的前置问题。
        """
        cases = load_network_cases()
        predicted = {classify(case.text) for case in cases}
        self.assertNotIn("observation", predicted)
        for text in ("嗯，好的，谢谢！", "刚看到线上延迟有点抖动。", "……"):
            self.assertNotEqual(baseline_network(text, classify_tier(text)), "observation")

    def test_tier_leaks_into_network(self) -> None:
        """`network` 不是独立标签：同一条经验描述因为一个否定词被判成 T0，network 随之变成 world。

        提案把 tier 与 network 当作两个独立维度（tier→h，network→f），
        但在当前实现下两者耦合。要么先解耦，要么承认这个耦合。
        """
        text = "重构时不要删除 legacy_adapter.py，它是兼容层的入口。"
        self.assertEqual(classify_tier(text), "T0")           # 因「不要」被判成硬约束
        self.assertEqual(baseline_network(text, "T0"), "world")
        self.assertEqual(baseline_network(text, "T2"), "experience")
        self.assertNotEqual(baseline_network(text, "T0"), baseline_network(text, "T2"))

    def test_experience_is_the_dumping_ground(self) -> None:
        report = evaluate_network()
        self.assertLess(report.precision("experience"), 0.5, "experience 吸收了多数误判")
        self.assertGreater(report.recall("experience"), 0.5)


class NetworkRuleDiagnosisTest(unittest.TestCase):
    """诊断实验：分离「schema 难」与「分类器弱」。见 docs/v1.2-candidates.md §3.3。"""

    @classmethod
    def setUpClass(cls) -> None:
        from moonshadow.network_eval import rule_variants

        # 注意：不把规则函数直接存成类属性——普通函数作为类属性会变成绑定方法，
        # 调用时会被多传一个 self。存字典，用下标取，拿到的是原始函数。
        cls.engines = rule_variants()
        cls.v1_key = "v1 基线（3 分支，依赖 tier）"
        cls.v2_key = "v2（4 类词汇，不依赖 tier）"

    def v1(self, text: str) -> str:
        return self.engines[self.v1_key](text)

    def v2(self, text: str) -> str:
        return self.engines[self.v2_key](text)

    def test_splits_are_disjoint_and_balanced(self) -> None:
        dev = load_network_cases(split="dev")
        held = load_network_cases(split="holdout")
        self.assertEqual(len(dev), 24)
        self.assertEqual(len(held), 24)
        self.assertFalse({c.id for c in dev} & {c.id for c in held})

    def test_v2_beats_v1_substantially(self) -> None:
        for split in ("dev", "holdout"):
            cases = load_network_cases(split=split)
            self.assertGreater(
                evaluate_network(cases, self.engines[self.v2_key]).attainable_accuracy,
                evaluate_network(cases, self.engines[self.v1_key]).attainable_accuracy + 0.3,
            )

    def test_v2_reaches_the_schema_ceiling_on_holdout(self) -> None:
        """核心结论：有答案的用例全部答对 ⇒ 瓶颈是 schema，不是分类器。

        若该断言失败（v2 的可达准确率掉下来），说明 §3.3 的结论需要重估。
        """
        report = evaluate_network(load_network_cases(split="holdout"), self.engines[self.v2_key])
        self.assertAlmostEqual(report.attainable_accuracy, 1.0, places=9, msg=report.summary())
        self.assertEqual(report.unattainable_count, 1)

    def test_v2_has_no_train_test_gap(self) -> None:
        """dev 与 holdout 的严格准确率相等——说明规则是泛化的，不是拟合开发集。"""
        dev = evaluate_network(load_network_cases(split="dev"), self.engines[self.v2_key])
        held = evaluate_network(load_network_cases(split="holdout"), self.engines[self.v2_key])
        self.assertAlmostEqual(dev.accuracy, held.accuracy, places=9)

    def test_v2_fixes_the_tier_leak(self) -> None:
        """v1 因「不要」把 N09 判成 world；v2 不依赖 tier，应判成 experience。"""
        case = next(c for c in load_network_cases(split="dev") if c.id == "N09")
        self.assertEqual(self.v1(case.text), "world")
        self.assertEqual(self.v2(case.text), "experience")

    def test_unattainable_cases_are_the_schema_gap(self) -> None:
        """无正确类别的用例都是「待办/承诺」——四张网络装不下。"""
        unattainable = [c for c in load_network_cases() if c.unattainable]
        self.assertEqual(len(unattainable), 2)
        for case in unattainable:
            self.assertTrue(case.note, f"{case.id} 必须写明为何没有正确类别")

    def test_ceiling_matches_the_unattainable_share(self) -> None:
        report = evaluate_network(load_network_cases(split="dev"), self.engines[self.v2_key])
        self.assertAlmostEqual(report.ceiling, 23 / 24, places=9)
        self.assertLessEqual(report.attainable_accuracy, 1.0)

    def test_comparison_reports_a_verdict(self) -> None:
        from moonshadow.network_eval import compare_network_rules

        text = compare_network_rules()
        self.assertIn("schema 上限", text)
        self.assertIn("判定", text)


class NetworkReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = evaluate_network()

    def test_accuracy_is_measured_and_low(self) -> None:
        """基线在开发集上的严格准确率（历史记录值 0.375）。

        注意必须限定 split="dev"——不限定会加载全部 48 条，数值不同。
        """
        report = evaluate_network(load_network_cases(split="dev"))
        self.assertAlmostEqual(report.accuracy, 0.375, places=3)

    def test_two_classes_are_unreachable(self) -> None:
        self.assertEqual(sorted(self.report.unreachable), ["observation", "world"])

    def test_confusion_matrix_is_complete(self) -> None:
        table = self.report.confusion()
        self.assertEqual(set(table), set(NETWORKS))
        for expected in NETWORKS:
            self.assertEqual(set(table[expected]), set(NETWORKS))
            self.assertEqual(
                sum(table[expected].values()),
                sum(1 for o in self.report.outcomes if o.expected == expected),
            )

    def test_report_renders_matrix_and_summary(self) -> None:
        text = format_confusion(self.report) + "\n" + self.report.summary()
        for network in NETWORKS:
            self.assertIn(network, text)
        self.assertIn("从未被预测出的类别", text)


if __name__ == "__main__":
    unittest.main()
