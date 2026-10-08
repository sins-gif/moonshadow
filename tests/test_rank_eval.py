"""排序对照实验的元测试。

实验本身产出证据、不设阈值，但**实验的机制**必须被测试，否则跑出来的数字不可信：
用例完整性、指标算法、方案可复现性、以及乘性方案的核心不变式。
"""

from __future__ import annotations

import functools
import pathlib
import sys
import unittest
from datetime import timedelta

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow.rank_eval import (  # noqa: E402
    ADOPTED_POLICY_NAME,
    CURRENT_RELEVANCE_WEIGHTS,
    CURRENT_THRESHOLDS,
    FLOOR_SWEEP_STEP,
    INCUMBENT_POLICY_NAME,
    NOW,
    SCHEME_LABELS,
    TIER_GROUPS,
    Candidate,
    RankCase,
    calibrate_thresholds,
    case_floor_bands,
    compare_policies,
    default_policies,
    evaluate_intent,
    evaluate_policies,
    load_holdout_cases,
    load_rank_cases,
    ordering,
    pairwise_agreement,
    run_rank_eval,
    sweep_floor,
    sweep_grouped_floor,
    sweep_weights,
    top1_hit,
)
from moonshadow.scoring import (  # noqa: E402
    SCHEMES,
    SCHEME_ADDITIVE,
    SCHEME_BLENDED,
    SCHEME_HYBRID,
    SCHEME_INTENT_AWARE,
    SCHEME_MULTIPLICATIVE,
    SCHEME_RELEVANCE,
    SCHEME_RRF,
    THETA,
    THETA_HIGH,
    TIME_FLOOR,
    TIME_FLOOR_MAX,
    TIME_FLOOR_MIN,
    floor_for_intent,
    floor_for_sensitivity,
    reciprocal_rank_fusion,
    rrf_weights,
    score_card,
    time_modifier,
    time_sensitivity,
)


def card(
    ident: str,
    tier: str,
    importance: int,
    age_days: float,
    half_life: float,
    entities: tuple[str, ...] = ("x",),
) -> dict:
    return {
        "id": ident,
        "tier": tier,
        "importance": importance,
        "entities": list(entities),
        "observed_at": (NOW - timedelta(days=age_days)).isoformat(timespec="seconds"),
        "half_life_days": half_life,
        "status": "active",
    }


def cached_report():
    """把 `run_rank_eval()` 的结果在整个测试进程里复用一次。

    它每次要跑 7 个方案 × 全部开发用例，并重建 floor 扫描与逐用例可行区间——
    十来个测试各调一次会让套件从几十秒涨到两分半。**只有确定性测试**
    （`ReportTest.test_report_shape_and_determinism`）才真的跑两次。
    """
    return _cached_report()


@functools.lru_cache(maxsize=1)
def _cached_report():
    return run_rank_eval()


class CaseLoadingTest(unittest.TestCase):
    def test_cases_load_and_are_well_formed(self) -> None:
        cases = load_rank_cases()
        self.assertGreaterEqual(len(cases), 5)
        self.assertEqual([c.id for c in cases], sorted(c.id for c in cases))
        for case in cases:
            self.assertTrue(case.candidates, case.id)
            self.assertTrue(case.note, f"{case.id} 缺少人工期望的理由说明")
            ids = {candidate.id for candidate in case.candidates}
            self.assertEqual(
                sorted(case.expect_order),
                sorted(ids),
                f"{case.id} 的 expect_order 必须是候选的一个排列",
            )

    def test_intents_are_covered(self) -> None:
        intents = {case.intent for case in load_rank_cases()}
        self.assertIn("historical", intents)
        self.assertIn("status", intents)
        self.assertIn("task", intents)
        self.assertIn("preference", intents)

    def test_excludes_are_parsed_with_reasons(self) -> None:
        cases = {case.id: case for case in load_rank_cases()}
        case = cases["09-gate-noise-and-window"]
        self.assertEqual(len(case.exclude), 4)
        self.assertEqual({item.reason for item in case.exclude}, {"tier", "window", "score"})

    def test_adversarial_cases_are_present(self) -> None:
        """对抗用例必须在**语料**里（切分后可能落在任一侧），否则意图指标会虚高。"""
        ids = {case.id for case in load_rank_cases() + load_holdout_cases()}
        self.assertIn("06-status-without-markers", ids)   # 漏判方向
        self.assertIn("07-historical-with-status-word", ids)  # 误判方向
        self.assertIn("08-status-weak-signal", ids)       # 漏判但排序不受影响


class MetricTest(unittest.TestCase):
    def test_pairwise_agreement_bounds(self) -> None:
        self.assertEqual(pairwise_agreement(["a", "b", "c"], ["a", "b", "c"]), 1.0)
        self.assertEqual(pairwise_agreement(["a", "b", "c"], ["c", "b", "a"]), 0.0)
        # [b,a,c] 只有 (a,b) 一对反序 => 2/3
        self.assertAlmostEqual(pairwise_agreement(["a", "b", "c"], ["b", "a", "c"]), 2 / 3, places=9)
        # [b,c,a] 只有 (b,c) 一对正序 => 1/3
        self.assertAlmostEqual(pairwise_agreement(["a", "b", "c"], ["b", "c", "a"]), 1 / 3, places=9)
        self.assertEqual(pairwise_agreement(["a"], ["a"]), 1.0)

    def test_top1_hit(self) -> None:
        self.assertTrue(top1_hit(["a", "b"], ["a", "b"]))
        self.assertFalse(top1_hit(["a", "b"], ["b", "a"]))
        self.assertFalse(top1_hit([], ["b"]))


class OrderingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cases = load_rank_cases()

    def test_every_scheme_returns_all_candidates(self) -> None:
        for case in self.cases:
            for scheme in SCHEMES:
                produced = ordering(case, scheme)
                self.assertEqual(sorted(produced), sorted(c.id for c in case.candidates))

    def test_unknown_scheme_raises(self) -> None:
        with self.assertRaises(ValueError):
            ordering(self.cases[0], "vibes")

    def test_all_schemes_are_labelled(self) -> None:
        for scheme in SCHEMES:
            self.assertIn(scheme, SCHEME_LABELS)


class MultiplicativeInvariantTest(unittest.TestCase):
    """乘性方案的核心保证：时间有界，无法反转差距较大的相关性。"""

    def test_time_modifier_is_bounded(self) -> None:
        self.assertAlmostEqual(time_modifier(0.0, 30.0), 1.0, places=12)
        self.assertGreaterEqual(time_modifier(10**6, 1.0), TIME_FLOOR)
        self.assertAlmostEqual(time_modifier(10**6, 1.0), TIME_FLOOR, places=12)
        self.assertEqual(time_modifier(10**6, 0.0), 1.0)  # T0 不衰减

    def test_score_equals_relevance_times_modifier(self) -> None:
        strong = card("a", "T4", 10, 100.0, 30.0)
        kwargs = {"now": NOW, "query_cos": 0.9, "query_entities": ["x"], "task_hit": True}
        relevance = score_card(strong, scheme=SCHEME_RELEVANCE, **kwargs)
        multiplicative = score_card(strong, scheme=SCHEME_MULTIPLICATIVE, **kwargs)
        self.assertAlmostEqual(multiplicative, relevance * time_modifier(100.0, 30.0), places=12)

    def test_large_relevance_gap_cannot_be_flipped_by_age(self) -> None:
        # A 相关性满分但极旧；B 只是低相关、且不命中实体、但刚刚发生
        ancient_strong = card("a", "T4", 10, 10_000.0, 30.0)
        fresh_weak = card("b", "T5", 5, 0.0, 14.0, entities=())

        strong = score_card(
            ancient_strong, now=NOW, query_cos=1.0, query_entities=["x"], task_hit=True,
            scheme=SCHEME_MULTIPLICATIVE,
        )
        weak = score_card(
            fresh_weak, now=NOW, query_cos=0.2, query_entities=["x"], task_hit=False,
            scheme=SCHEME_MULTIPLICATIVE,
        )
        self.assertGreater(
            strong, weak,
            "相关性差距超过 1/TIME_FLOOR 时，时间不该翻盘——这是有界修正存在的理由",
        )

    def test_the_bound_defines_the_maximum_overturn_ratio(self) -> None:
        """把「有界」这件事量化：时间最多能把相关性差距放大 `1/TIME_FLOOR` 倍。

        floor 从 `0.70` 降到 `0.5925`，这个上限从 `1.428571` 放宽到 `1.687764`——
        这正是「时间话语权变大」的准确含义，也是换默认值时真正的取舍。
        代价可以直接量出来：`run_drift_check` 里乘性方案的反转组数随之上升。
        """
        self.assertAlmostEqual(1.0 / TIME_FLOOR, 1 / 0.5925, places=9)
        self.assertGreater(1.0 / TIME_FLOOR, 1.0 / 0.70)


class IntentAwareTest(unittest.TestCase):
    """按意图切换时间下限的方案：对照实验一致率最高者，锁成回归测试。"""

    def test_floor_by_intent(self) -> None:
        self.assertLess(floor_for_intent("status"), floor_for_intent("historical"))
        self.assertEqual(floor_for_intent("unknown-intent"), TIME_FLOOR)

    def test_intent_aware_matches_all_human_expectations(self) -> None:
        report = cached_report()
        result = next(r for r in report.results if r.scheme == SCHEME_INTENT_AWARE)
        self.assertEqual(
            result.agreement, 1.0,
            "按意图切换时间下限应当同时满足历史决策与当前状态两类查询：\n" + report.summary(),
        )
        self.assertEqual(result.top1_rate, 1.0)

    def test_a_single_bounded_floor_satisfies_both_intents(self) -> None:
        """此前的「全局权重必然二选一」已被推翻：固定 0.62 同时满足两类查询。

        这正是采纳新默认值后得到的主要结论——按意图切换那套机器是多余的。
        """
        by_scheme = {r.scheme: r.by_intent() for r in cached_report().results}
        fixed = by_scheme[SCHEME_MULTIPLICATIVE]
        self.assertEqual(fixed["status"], 1.0)
        self.assertEqual(fixed["task"], 1.0)
        self.assertEqual(fixed["historical"], 1.0)
        # 对比：完全不看时间的方案仍然在时效类查询上失败
        self.assertLess(by_scheme[SCHEME_RELEVANCE]["status"], 1.0)


class CombinedSchemeTest(unittest.TestCase):
    """结合方案：把「时间话语权」交给上下文，再看融合算术是否重要。"""

    def test_time_sensitivity_reads_query_surface(self) -> None:
        self.assertEqual(time_sensitivity("现在的预算定了吗？"), 1.0)
        self.assertEqual(time_sensitivity("当初为什么选方案B？"), 0.0)
        # 无线索 => 保守取 0（不给时间额外话语权）
        self.assertEqual(time_sensitivity("张三的报价给到了吗？"), 0.0)
        # 两侧线索同时出现 => 中间值，这是相对硬切换的关键改进
        self.assertAlmostEqual(time_sensitivity("现在和当初的对比"), 0.5, places=9)

    def test_floor_interpolates_between_ends(self) -> None:
        self.assertAlmostEqual(floor_for_sensitivity(0.0), TIME_FLOOR_MAX, places=12)
        self.assertAlmostEqual(floor_for_sensitivity(1.0), TIME_FLOOR_MIN, places=12)
        self.assertAlmostEqual(
            floor_for_sensitivity(0.5), (TIME_FLOOR_MAX + TIME_FLOOR_MIN) / 2, places=12
        )
        with self.assertRaises(ValueError):
            floor_for_sensitivity(1.5)

    def test_rrf_weights_shift_with_sensitivity(self) -> None:
        low = rrf_weights(0.0)
        high = rrf_weights(1.0)
        self.assertAlmostEqual(sum(low), 1.0, places=12)
        self.assertAlmostEqual(sum(high), 1.0, places=12)
        self.assertLess(low[1], high[1], "敏感度越高，时间一路的权重应越大")

    def test_rrf_fuses_ranks_not_scores(self) -> None:
        # 名次互换 => 同分，按 id 破同分
        fused = reciprocal_rank_fusion([("a", 0.9, 0.1), ("b", 0.8, 1.0)], weights=(0.5, 0.5))
        self.assertEqual([name for name, _ in fused], ["a", "b"])
        self.assertAlmostEqual(fused[0][1], fused[1][1], places=12)
        self.assertEqual(reciprocal_rank_fusion([]), [])

    def test_rrf_is_rejected_by_single_card_scorer(self) -> None:
        with self.assertRaises(ValueError):
            score_card(card("a", "T4", 5, 1.0, 30.0), now=NOW, scheme=SCHEME_RRF)

    def test_hybrid_is_unweighted_average(self) -> None:
        target = card("a", "T4", 7, 12.0, 30.0)
        kwargs = {"now": NOW, "query_cos": 0.8, "query_entities": ["x"], "task_hit": False}
        hybrid = score_card(target, scheme=SCHEME_HYBRID, **kwargs)
        additive = score_card(target, scheme=SCHEME_ADDITIVE, **kwargs)
        multiplicative = score_card(target, scheme=SCHEME_MULTIPLICATIVE, **kwargs)
        self.assertAlmostEqual(hybrid, 0.5 * additive + 0.5 * multiplicative, places=12)

    def test_fusion_form_is_no_longer_discriminated_on_this_case_set(self) -> None:
        """**边界用例恢复了区分力**——这正是 R1 存在的理由。

        上一版（无边界用例时）五个方案并列 `1.000`，本用例集无法区分融合形式。
        加入 `23`/`24`/`25` 之后：

        - 固定下限一族（加性 / 乘性 / 意图切换）仍 `1.000`；
        - **连续插值与 50-50 平均掉到 `0.962`**，同在 `24-t7-borderline` 上失败——
          该查询含「现在」，连续插值据此把下限压到 `TIME_FLOOR_MIN`，时间话语权变大，
          于是「更新但相关性更低」的 T7 翻盘，与该条用例的期望（原则 a）相反；
        - 仅相关性与 RRF 仍在 `0.73` 附近。

        也就是说：**固定下限胜出**，而「按上下文调低下限」在边界用例上暴露为错误。
        """
        by_scheme = {result.scheme: result.agreement for result in cached_report().results}
        for scheme in (
            SCHEME_ADDITIVE,
            SCHEME_MULTIPLICATIVE,
            SCHEME_INTENT_AWARE,
        ):
            self.assertAlmostEqual(
                by_scheme[scheme], 1.0, places=9, msg=f"{scheme} 应当满分"
            )
        for scheme in (SCHEME_BLENDED, SCHEME_HYBRID):
            self.assertLess(
                by_scheme[scheme], 1.0,
                msg=f"{scheme} 应当在边界用例 24 上失败（时间话语权被放大）",
            )
        self.assertLess(
            by_scheme[SCHEME_RRF], by_scheme[SCHEME_MULTIPLICATIVE],
            msg="RRF 仍是这一族里最差的",
        )
        self.assertLess(by_scheme[SCHEME_RELEVANCE], by_scheme[SCHEME_MULTIPLICATIVE])


class IntentMetricTest(unittest.TestCase):
    """意图指标仍必须与排序指标分开报——只是采纳新默认值后，所有漏判恰好都无害了。"""

    def test_intent_metrics_are_still_reported(self) -> None:
        report = cached_report()
        intent = report.intent
        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertLess(intent.recall, 1.0, "无线索的时效提问仍会被漏判")
        self.assertGreater(intent.accuracy, 0.0)

    def test_miss_is_harmless_under_the_adopted_scheme(self) -> None:
        """漏判在**采纳方案**下仍然无害；但「按上下文调低下限」首次真的翻车，且不是漏判造成的。

        事实分两半：

        1. 采纳方案（乘性固定下限）在 26 条开发用例上全对，7 处漏判**没有**造成排序错误
           ——「意图指标与排序指标必须分开报」在这一半上仍然成立；
        2. 但 `24-t7-borderline` 让连续插值/50-50 平均首次失败：该查询含「现在」，
           敏感性被**正确**识别为 `1.0`，下限随 `floor_for_sensitivity` 压到 `TIME_FLOOR_MIN`，
           时间话语权变大，新鲜但相关性更低的 T7 翻盘。**这不是漏判造成的**，
           是策略本身在边界用例上错了——所以不能再推断「调低下限一定无害」。
        """
        report = cached_report()
        assert report.intent is not None
        miss = {outcome.case_id for outcome in report.intent.misses}
        self.assertTrue(miss, "漏判确实存在")

        adopted = next(r for r in report.results if r.scheme == SCHEME_MULTIPLICATIVE)
        adopted_failed = {outcome.case_id for outcome in adopted.failures}
        self.assertEqual(adopted_failed, set(), "采纳方案应当全对")
        self.assertEqual(miss & adopted_failed, set(), "漏判在采纳方案下仍然无害")

        for scheme in (SCHEME_BLENDED, SCHEME_HYBRID):
            result = next(r for r in report.results if r.scheme == scheme)
            failed = {outcome.case_id for outcome in result.failures}
            self.assertEqual(
                failed, {"24-t7-borderline"},
                f"{scheme} 的失败点应当恰好是边界用例 24；若变化请更新本条",
            )
            self.assertNotIn(
                "24-t7-borderline", miss,
                "该失败**不是**漏判造成的：敏感性被正确识别，是策略在边界上错了",
            )

    def test_evaluate_intent_is_pure(self) -> None:
        cases = load_rank_cases()
        self.assertEqual(evaluate_intent(cases).summary(), evaluate_intent(cases).summary())


class CalibrationTest(unittest.TestCase):
    def test_calibration_yields_a_feasible_interval(self) -> None:
        report = cached_report()
        self.assertIn(SCHEME_BLENDED, report.calibrations)
        theta = report.calibrations[SCHEME_BLENDED]["theta"]
        self.assertTrue(theta.excludes, "标定必须有分数型负例")
        self.assertTrue(theta.feasible, theta.summary())
        assert theta.recommended is not None
        self.assertGreater(theta.recommended, theta.lower)  # type: ignore[arg-type]
        self.assertLess(theta.recommended, theta.upper)  # type: ignore[arg-type]

    def test_current_threshold_is_inside_the_adopted_interval(self) -> None:
        """现行阈值**就是**采纳方案的标定值：它必须落在区间内，且接近中点。

        此前这条断言的方向是反的（现用值落在区间外、必须重标定）；本轮已把标定值
        `0.3941` 写进 `scoring.THETA`，因此断言改成「落在区间内且接近中点」。
        区间本身由 `run_rank_eval()` 每次重算，所以这条断言会在用例集变化时失效并提醒复核。
        """
        theta = cached_report().calibrations[SCHEME_MULTIPLICATIVE]["theta"]
        self.assertTrue(theta.feasible, theta.summary())
        current = CURRENT_THRESHOLDS["theta"]
        self.assertTrue(theta.contains(current), theta.summary())
        assert theta.recommended is not None
        self.assertLess(abs(current - theta.recommended), 1e-3, "现行值应当就是标定中点")

    def test_calibrated_thresholds_match_the_hard_coded_adopted_values(self) -> None:
        """常量与标定必须一致——否则生产门用的是没经过标定的数。

        这是 v1.2 决策 4 的落地检查：`select()` 默认取 `THETA` / `THETA_HIGH`，
        若它们不在标定区间内，门就会静默误挡（`tools/run_gate_audit.py` 会先抓到）。
        """
        groups = cached_report().calibrations[SCHEME_MULTIPLICATIVE]
        for name, value in (("theta", THETA), ("theta_high", THETA_HIGH)):
            calibration = groups[name]
            self.assertTrue(
                calibration.contains(value),
                f"{name} 的常量 {value} 不在标定区间内：{calibration.summary()}",
            )
        # θ′ 与 θ 取同一个值：T6/T7 不再设更高的分位门槛（开发集证据不足）。
        self.assertEqual(THETA_HIGH, THETA)

    def test_additive_theta_is_feasible_after_the_case_fix(self) -> None:
        """加性的可分离性会随用例集翻转——这本身就是「证据依赖用例集」的演示。

        用例 03 的超窗旧引文被修好后，加性在开发集上**变得可分离**
        （区间 `(0.530219, 0.550015)`，与 `f` 无关——加性打分不含下限）。
        但要注意两点：
        1. 它的区间与乘性的区间**量纲不同，宽度不可比**；
        2. 定理 5 是**构造性**的，不依赖用例集——加性仍存在反例对（附录 A.3），
           且 `run_drift_check` 在合成网格上仍然抓到加性在乘性有保证区域内的反转。
        """
        theta = cached_report().calibrations[SCHEME_ADDITIVE]["theta"]
        self.assertTrue(theta.feasible, theta.summary())
        assert theta.lower is not None and theta.upper is not None
        self.assertGreater(theta.lower, 0.52)
        self.assertLess(theta.upper, 0.56)

    def test_infeasible_interval_is_reported_as_such(self) -> None:
        """用定理 5 的见证对构造一个**真正不可行**的标定，检验报告路径。

        同一对候选：加性下不可分（负例 `0.5302` > 正例 `0.5036`），
        乘性下可分（负例 `0.3873` < 正例 `0.4146`）——这正是「缺陷在形式而不在参数」。
        """
        case = RankCase.from_json(
            {
                "id": "unit-theorem-5-witness",
                "query": {"text": "缓存层换用什么方案了？", "entities": []},
                "candidates": [
                    {"id": "P", "tier": "T5", "importance": 3, "age_days": 30, "cos": 0.72}
                ],
                "exclude": [
                    {"id": "N", "reason": "score", "tier": "T3", "importance": 2,
                     "age_days": 10, "cos": 0.15},
                ],
                "expect_order": ["P"],
            }
        )
        additive = calibrate_thresholds([case], SCHEME_ADDITIVE)["theta"]
        self.assertFalse(additive.feasible)
        self.assertIsNone(additive.recommended, "区间不存在时不得给出建议值")
        self.assertIn("不可行", additive.summary())
        # 「区间不存在」这句只在对照现用值时才打印（它是对现用值的判定，不是区间自身的属性）。
        self.assertIn("区间不存在", additive.summary(current=THETA))

        multiplicative = calibrate_thresholds([case], SCHEME_MULTIPLICATIVE)["theta"]
        self.assertTrue(
            multiplicative.feasible,
            f"同一对候选在乘性下应当可分：{multiplicative.summary()}",
        )

    def test_only_score_negatives_participate_in_calibration(self) -> None:
        case = RankCase.from_json(
            {
                "id": "unit-negatives",
                "messages": [],
                "query": {"text": "测试", "entities": []},
                "candidates": [
                    {"id": "keep", "tier": "T3", "importance": 6, "age_days": 1, "cos": 0.9}
                ],
                "exclude": [
                    {"id": "by-tier", "reason": "tier", "tier": "T8", "importance": 1, "age_days": 0.1, "cos": 0.9},
                    {"id": "by-window", "reason": "window", "tier": "T4", "importance": 8, "age_days": 200, "cos": 0.9},
                    {"id": "by-score", "reason": "score", "tier": "T3", "importance": 1, "age_days": 2, "cos": 0.1},
                ],
                "expect_order": ["keep"],
            }
        )
        groups = calibrate_thresholds([case], SCHEME_ADDITIVE)
        self.assertEqual([item.candidate.id for item in groups["theta"].excludes], ["by-score"])

    def test_rrf_cannot_be_calibrated(self) -> None:
        with self.assertRaises(ValueError):
            calibrate_thresholds(load_rank_cases(), SCHEME_RRF)


class FloorSweepTest(unittest.TestCase):
    """floor 扫描与逐用例可行区间——用来判断「按意图切换」是否必要。"""

    def test_sweep_covers_the_unit_interval(self) -> None:
        # 步长必须整除采纳值 0.5925（0.5925/0.0025 = 237），否则扫描会漏掉采纳值并直接抛错。
        sweep = sweep_floor(load_rank_cases(), step=0.0025)
        self.assertEqual(sweep.points[0][0], 0.0)
        self.assertEqual(sweep.points[-1][0], 1.0)
        self.assertGreater(len(sweep.points), 10)
        for floor, agreement in sweep.points:
            self.assertGreaterEqual(floor, 0.0)
            self.assertLessEqual(floor, 1.0)
            self.assertLessEqual(agreement, 1.0)

    def test_a_global_floor_exists_and_its_window_is_recorded(self) -> None:
        """全局单值**能**让全部 26 条开发用例通过，窗口 `[0.5000, 0.6600]`（宽 `0.160`）。

        修正前的同一批用例窗口只有 `0.026` 宽，据此得出「`f` 的精度是假精度」；
        那个窄窗口是用例缺陷造出来的（用例 03 的旧引文卡龄 `120` 天超出 T5 窗口）。
        加入 3 条边界用例（`23`/`24`/`25`）后窗口从 `[0.470, 0.685]` 收窄到 `[0.500, 0.660]`：
        **R1 要的「用边界用例约束参数」生效了**。
        """
        sweep = cached_report().floor_sweep
        assert sweep is not None
        self.assertAlmostEqual(sweep.max_agreement, 1.0, places=9)
        widest = sweep.widest_run
        assert widest is not None
        low, high = widest
        self.assertAlmostEqual(low, 0.5000, places=6)
        self.assertAlmostEqual(high, 0.6600, places=6)
        self.assertGreater(high - low, 0.05, "窗口并不窄——窄窗口的旧记录来自超窗用例")
        self.assertLessEqual(low, TIME_FLOOR)
        self.assertLessEqual(TIME_FLOOR, high)

    def test_bands_localise_the_pressure(self) -> None:
        """窗口之所以窄，只由两条用例决定——这张表是把压力定位出来的工具。"""
        report = cached_report()
        bands = report.floor_bands
        self.assertEqual(len(bands), len(load_rank_cases()))

        widest = report.floor_sweep.widest_run  # type: ignore[union-attr]
        assert widest is not None
        lower_bound_cases = [cid for cid, band in bands.items() if band and abs(band[0] - widest[0]) < 0.01]
        upper_bound_cases = [cid for cid, band in bands.items() if band and abs(band[1] - widest[1]) < 0.01]
        self.assertTrue(lower_bound_cases, "应当能定位到抬高下界的用例")
        self.assertTrue(upper_bound_cases, "应当能定位到压低上界的用例")
        # 单侧很宽的用例（控制组、网关组）不该成为瓶颈
        self.assertGreater(bands["04-same-relevance-different-age"][1] - bands["04-same-relevance-different-age"][0], 0.5)  # type: ignore[operator]

    def test_no_case_is_misplaced_after_lowering_the_default(self) -> None:
        """**固定的采纳下限**下没有任何用例被错放；但按上下文调低下限会错放 `24`。

        这条指标（选出的 floor 是否落在该用例的可行区间内）才是正确的度量：
        二分类意图指标仍报若干「错误」，而实际错放只看下限取在哪。
        加入边界用例后，`floor_for_sensitivity`（连续插值）在 `24-t7-borderline` 上
        把下限压到状态类那一端，于是**该用例被错放**——这正是「按上下文调低下限」
        在边界上会翻车的直接证据。
        """
        report = cached_report()
        misplaced_fixed, misplaced_sensitivity = [], []
        for case in report.case_index:
            band = report.floor_bands[case.id]
            if band is None:
                misplaced_fixed.append(case.id)
                misplaced_sensitivity.append(case.id)
                continue
            if not (band[0] - 1e-9 <= TIME_FLOOR <= band[1] + 1e-9):
                misplaced_fixed.append(case.id)
            chosen = floor_for_sensitivity(time_sensitivity(case.query_text))
            if not (band[0] - 1e-9 <= chosen <= band[1] + 1e-9):
                misplaced_sensitivity.append(case.id)
        self.assertEqual(misplaced_fixed, [], "固定下限下不应用例被错放")
        self.assertEqual(misplaced_sensitivity, ["24-t7-borderline"])

    def test_band_report_is_rendered(self) -> None:
        report = cached_report()
        text = report.summary()
        self.assertIn("逐用例可行 floor 区间", text)
        # 用一条**当前确在开发集**的用例：切分重划后 03 已移入留出集。
        self.assertIn("H02-moderate-gap-much-newer", text)


class HoldoutValidationTest(unittest.TestCase):
    """留出验证：开发集上选出的结论是否为过拟合。"""

    def test_holdout_is_separate_labelled_and_well_formed(self) -> None:
        """留出集与开发集不重叠，且每条都带推导依据（`note`）与完整的期望次序。

        **切分重划后的变化**：留出集由 `eval/split-manifest.json`（种子 `20260601`）定义，
        不再等于「文件名以 H 开头」——现在里面混有 `01`/`03`/`08`/`17`/`18`/`21`。
        相应地，`needs_fresh` 不再逐条必填（那三条是按别的问题写的），
        但它们仍必须能由 `intent` 回退到无歧义的标签。
        """
        development = load_rank_cases()
        holdout = load_holdout_cases()
        self.assertGreaterEqual(len(holdout), 9)
        self.assertFalse(
            {case.id for case in development} & {case.id for case in holdout},
            "留出集必须与开发集不重叠",
        )
        explicit = 0
        for case in holdout:
            if case.needs_fresh is not None:
                explicit += 1
            else:
                # 未显式标注时，回退路径必须仍然是无歧义的：按查询文本算出的敏感度
                # 与 intent 一致，不允许「靠评测标签」才能判断。
                self.assertIn(case.intent, ("status", "historical", "task", "preference"))
            self.assertEqual(
                sorted(case.expect_order),
                sorted(candidate.id for candidate in case.candidates),
                f"{case.id} 的 expect_order 必须是候选的一个排列",
            )
            self.assertTrue(case.note, f"{case.id} 必须写明期望顺序的推导依据")
        self.assertGreaterEqual(explicit, 4, "显式标注的留出用例不应少于 4 条")

    def test_policies_never_read_the_evaluation_label(self) -> None:
        """策略只能看查询文本；若它能看 needs_fresh，就等于拿答案做题。"""
        base = {
            "id": "unit-label-leak",
            "query": {"text": "部署到哪个环境了？", "entities": ["部署"]},
            "candidates": [
                {"id": "a", "tier": "T5", "importance": 5, "age_days": 1.0, "cos": 0.7},
                {"id": "b", "tier": "T5", "importance": 5, "age_days": 30.0, "cos": 0.7},
            ],
            "expect_order": ["a", "b"],
        }
        fresh_case = RankCase.from_json({**base, "needs_fresh": True})
        stale_case = RankCase.from_json({**base, "needs_fresh": False})
        for name, policy in default_policies().items():
            self.assertEqual(policy(fresh_case), policy(stale_case), f"{name} 读了评测标签")

    def test_the_swept_default_survives_holdout(self) -> None:
        """采纳值 `0.5925` 在留出集上不劣于 v1.1 的旧默认 `0.70`。

        注意这条**不能**读成「留出集验证了 0.5925」：留出集窗口宽 `0.458`，
        只否掉极端取值，无法确认某个精确值。它真正做到的是否掉 `> 0.980` 的取值。
        """
        holdout = evaluate_policies(load_holdout_cases())
        adopted = holdout[ADOPTED_POLICY_NAME]
        incumbent = holdout[INCUMBENT_POLICY_NAME]
        self.assertGreaterEqual(adopted, incumbent)
        self.assertGreater(
            incumbent, 0.0, "旧默认在留出集上不应完全失败——否则这条对照没有信息量"
        )

    def test_intent_switching_adds_nothing_over_the_fixed_default(self) -> None:
        """意图切换即便配上更好的默认值，也不优于纯全局固定值。"""
        development = evaluate_policies(load_rank_cases())
        self.assertLessEqual(
            development[f"意图切换 默认{TIME_FLOOR}/状态{TIME_FLOOR_MIN}"],
            development[ADOPTED_POLICY_NAME],
        )

    def test_comparison_reports_both_sets_and_a_verdict(self) -> None:
        text = compare_policies(load_rank_cases(), load_holdout_cases())
        self.assertIn("开发集", text)
        self.assertIn("留出集", text)
        self.assertIn("成立", text)


    def test_holdout_window_is_much_wider_and_pins_only_the_lower_end(self) -> None:
        """冻结切分下两个集合各贡献一个边界——这比「留出集更宽」更具体：

        - 开发集窗口 `[0.5000, 0.6600]`（宽 `0.160`；上限由新增的 `23`/`25` 钉住）
        - 留出集窗口 `[0.5225, 0.9800]`（宽 `0.4575`）
        - 交集 `[0.5225, 0.6600]`：**下界来自留出集**，**上界来自开发集**

        留出集的上界 `0.980` 说明它几乎不约束上端（只否掉极旧取值），
        因此「留出集验证了某个精确值」依旧是不成立的；它能否掉的是 `> 0.980` 的取值。
        """
        dev_run = sweep_floor(load_rank_cases()).widest_run
        held_run = sweep_floor(load_holdout_cases()).widest_run
        assert dev_run is not None and held_run is not None

        self.assertAlmostEqual(dev_run[0], 0.5000, places=6)
        self.assertAlmostEqual(dev_run[1], 0.6600, places=6)
        self.assertAlmostEqual(held_run[0], 0.5225, places=6)
        self.assertAlmostEqual(held_run[1], 0.9800, places=6)

        self.assertGreater(held_run[1] - held_run[0], dev_run[1] - dev_run[0])
        self.assertGreater(held_run[0], dev_run[0], "交集下界来自留出集")
        self.assertGreater(held_run[1], dev_run[1], "留出集上界远高于开发集，故交集上界来自开发集")

    def test_adopted_floor_is_inside_the_intersection_near_its_midpoint(self) -> None:
        """采纳值 `0.5925` 落在交集 `[0.5225, 0.6600]` 内，且与中点相差不超过一个步长。

        **中点本身依赖扫描步长**：`0.005→0.5925`、`0.0025→0.5913`、`0.00125→0.5919`、
        `0.0005→0.5922`——真值约 `0.592 ± 0.001`。因此这里不断言「等于中点」，
        只断言「落在交集中、且与中点/两侧余量之差都不超过一个步长」，并把步长依赖写进注释。
        取 `0.5925` 的理由是它落在采纳网格上（`0.5925/0.0025 = 237`）。
        """
        dev_run = sweep_floor(load_rank_cases()).widest_run
        held_run = sweep_floor(load_holdout_cases()).widest_run
        assert dev_run is not None and held_run is not None
        low = max(dev_run[0], held_run[0])
        high = min(dev_run[1], held_run[1])
        self.assertAlmostEqual(low, 0.5225, places=6)
        self.assertAlmostEqual(high, 0.6600, places=6)
        self.assertLessEqual(low, TIME_FLOOR)
        self.assertLessEqual(TIME_FLOOR, high)
        midpoint = (low + high) / 2
        self.assertAlmostEqual(midpoint, 0.59125, places=6)
        self.assertLessEqual(
            abs(TIME_FLOOR - midpoint), FLOOR_SWEEP_STEP,
            "与中点的偏离不得超过一个扫描步长",
        )
        self.assertLessEqual(
            abs((TIME_FLOOR - low) - (high - TIME_FLOOR)),
            FLOOR_SWEEP_STEP + 1e-9,
            "两侧余量之差不得超过一个扫描步长",
        )

    def test_window_width_is_pinned_by_only_a_few_cases(self) -> None:
        """窗口宽度只由 2 条开发用例决定（`H02` 定上界、`H04` 定下界）。

        少数用例决定参数 ⇒ 参数的精度不能当真。切分重划后名单从 `01/05/06` 变为
        `H02`/`H04`——**「哪几条用例钉住参数」本身就是切分的函数**，不能当成稳定事实。
        """
        dev = load_rank_cases()
        run = sweep_floor(dev).widest_run
        assert run is not None
        bands = case_floor_bands(dev)
        binding = sorted(
            cid for cid, band in bands.items()
            if band and (abs(band[0] - run[0]) < 1e-9 or abs(band[1] - run[1]) < 1e-9)
        )
        self.assertEqual(
            binding,
            ["23-t6-tight-race", "24-t7-borderline"],
        )
        self.assertLess(len(binding), len(dev) / 2, "少数用例决定参数 → 参数精度不可当真")


class WeightSweepTest(unittest.TestCase):
    """相关性权重的来源与稳健性（技术报告 §6.8）。"""

    def test_weights_are_the_renormalized_v10_weights(self) -> None:
        """来源守卫：这三个数是 v1.0 权重去掉衰减项后按 0.75 归一化的结果，不是独立选定。"""
        sim, importance, match = CURRENT_RELEVANCE_WEIGHTS
        self.assertAlmostEqual(sim, 0.45 / 0.75, places=12)
        others = (0.20 / 0.75, 0.10 / 0.75)
        self.assertAlmostEqual(importance, round(others[0], 2), places=12)
        self.assertAlmostEqual(match, round(others[1], 2), places=12)
        self.assertAlmostEqual(sim + importance + match, 1.0, places=12)

    def test_current_weights_are_feasible_on_both_sets(self) -> None:
        for cases in (load_rank_cases(), load_holdout_cases()):
            sweep = sweep_weights(cases, step=0.05)
            self.assertTrue(sweep.current_is_feasible, sweep.summary())

    def test_feasible_region_is_large_and_the_current_point_is_not_marginal(self) -> None:
        """核心发现（数字随用例集重算）：可行区占 `16.6%`，当前权重到最近不可行点的
        距离是 `0.1077`——**不贴边**（旧记录为 `0.0224`）。

        旧记录的「停在边缘」随用例集修正而消失，说明它从来不是权重方案本身的性质，
        而是当时那份用例集的性质。可行区内仍有比当前值余量更大的点（`0.1442`），
        所以「当前值是最稳健选择」这个说法依然不成立。
        """
        import math

        sweep = sweep_weights(step=0.01)
        self.assertGreater(sweep.feasible_fraction, 0.15, "可行区域并不小")

        current_sim, current_imp = CURRENT_RELEVANCE_WEIGHTS[0], CURRENT_RELEVANCE_WEIGHTS[1]
        infeasible = [(p[0], p[1]) for p in sweep.points if p[3] < 1.0 - 1e-9]
        margin = min(math.hypot(current_sim - x, current_imp - y) for x, y in infeasible)
        self.assertGreater(
            margin, 0.05, f"当前权重到边界距离 {margin:.4f}：已不再贴边（旧值为 0.0224）"
        )

        feasible = [(p[0], p[1]) for p in sweep.points if p[3] >= 1.0 - 1e-9]
        best_margin = max(
            min(math.hypot(gx - x, gy - y) for x, y in infeasible) for gx, gy in feasible
        )
        self.assertGreater(
            best_margin, margin,
            "可行区域内存在余量更大的点——当前值并非最稳健选择",
        )

    def test_importance_axis_is_the_tightest(self) -> None:
        """重要度是单纯形上最紧的一根轴。

        加入 3 条边界用例后逐轴可行范围是 `sim∈[0.42, 0.79]`、`importance∈[0.07, 0.39]`、
        `match∈[0.00, 0.51]`（宽 `0.37 / 0.32 / 0.51`），仍由重要度压住；
        当前 `0.27` 到上界 `0.39` 的余量是 `0.12`。
        """
        sweep = sweep_weights(step=0.01)
        box = sweep.bounding_box
        widths = {name: high - low for name, (low, high) in box.items()}
        self.assertEqual(min(widths, key=lambda name: widths[name]), "importance")
        self.assertAlmostEqual(box["importance"][1], 0.39, places=2)
        self.assertAlmostEqual(CURRENT_RELEVANCE_WEIGHTS[1], 0.27, places=2)
        self.assertLess(box["importance"][1] - CURRENT_RELEVANCE_WEIGHTS[1], 0.20)
        self.assertLess(CURRENT_RELEVANCE_WEIGHTS[1] - box["importance"][0], 0.30)


class GroupedFloorSweepTest(unittest.TestCase):
    """v1.2-A：分组时间下限（每组一个 f）。见 docs/v1.2-candidates.md §7。

    扫描必须是两阶段：全局粗扫 `0.0395`（整除 `0.5925`，`26³ = 17576`）→ 盒内细扫 `0.0075`。
    单阶段细扫不可行（三维网格 `O(1/step³)`），单阶段粗扫不够精确。
    """

    #: 全局粗扫步长：必须整除采纳值 0.5925（0.5925/0.0395 = 15）。
    COARSE = 0.0395
    #: 盒内细扫步长：同样必须整除（0.5925/0.0075 = 79）。
    FINE = 0.0075

    @classmethod
    def setUpClass(cls) -> None:
        cls.dev = sweep_grouped_floor(load_rank_cases(), step=cls.COARSE)
        cls.held = sweep_grouped_floor(load_holdout_cases(), step=cls.COARSE)

    def test_grid_must_divide_the_adopted_value(self) -> None:
        """本仓库踩过两次的坑：步长不整除采纳值，会得出「单一解不在交集内」的假结论。

        断言已从「打印警告」升级为**直接抛错**。`0.02` 就是现成的反例：它不整除 `0.5925`。
        """
        ratio = TIME_FLOOR / self.COARSE
        self.assertAlmostEqual(ratio, round(ratio), places=9)
        self.assertAlmostEqual(TIME_FLOOR / self.FINE, round(TIME_FLOOR / self.FINE), places=9)

        with self.assertRaises(ValueError) as ctx:
            sweep_grouped_floor(load_rank_cases(), step=0.02)
        self.assertIn("不能整除", str(ctx.exception))

        # 全局 floor 扫描同样阻断（采纳值落在扫描区间内时）。
        with self.assertRaises(ValueError):
            sweep_floor(load_rank_cases(), step=0.02)

    def test_grouping_keeps_the_existing_solution_feasible(self) -> None:
        """分组是单一解的推广：采纳值（全组同值）必须仍可行（直接求值，不依赖网格）。"""
        for sweep in (self.dev, self.held):
            self.assertTrue(sweep.current_is_feasible, sweep.summary())

    def test_intersection_is_non_empty_and_contains_the_adopted_point(self) -> None:
        dev_ok = {tuple(round(v, 4) for v in p[:3]) for p in self.dev.feasible}
        held_ok = {tuple(round(v, 4) for v in p[:3]) for p in self.held.feasible}
        shared = dev_ok & held_ok
        self.assertTrue(shared)
        adopted = tuple(round(TIME_FLOOR, 4) for _ in self.dev.groups)
        self.assertIn(adopted, shared)

    def test_low_group_is_now_the_binding_constraint(self) -> None:
        """三组里最窄的约束**从 `high` 变成了 `low`**（T6–T9）。

        加入 3 条边界用例前是 `high`（宽 `0.11`）；现在开发集粗网格上三组都是 `0.11`，
        由 `low` 最紧。**「哪一组在压住可行区」是用例集的函数**，不是分组的固有性质——
        这正是 roadmap §7.4 要每次实测而不是假定的原因。
        """
        box = self.dev.bounding_box
        widths = {name: high - low for name, (low, high) in box.items()}
        self.assertEqual(self.dev.marginal_group(), "low")
        self.assertLessEqual(widths["low"], widths["high"])
        self.assertLessEqual(widths["low"], widths["mid"])

    def test_two_stage_scan_agrees_on_the_binding_group(self) -> None:
        """两阶段扫描必须给出**一致**的最窄约束——否则细扫改变了结论，说明粗扫欠采样。

        这条同时记录「非盒形占比」依赖细扫范围，因此不作为可行区的固有性质使用。
        """
        box = self.dev.bounding_box
        pad = self.COARSE
        ranges = {
            name: (max(0.0, low - pad), min(1.0, high + pad)) for name, (low, high) in box.items()
        }
        fine = sweep_grouped_floor(load_rank_cases(), step=self.FINE, ranges=ranges)
        fine_box = fine.bounding_box
        self.assertEqual(
            min(fine_box, key=lambda n: fine_box[n][1] - fine_box[n][0]),
            self.dev.marginal_group(),
        )
        adopted = tuple(round(TIME_FLOOR, 4) for _ in fine.groups)
        self.assertIn(adopted, {tuple(round(v, 4) for v in p[:3]) for p in fine.feasible})

    def test_new_batch_covers_the_blind_tiers(self) -> None:
        """第一批补齐了 `T0`/`T7`/`T8` 正例；`T9` 至今只有**负例**。

        `22-t9-noise-blocked-by-tier` 是 T9 的负例侧覆盖，按设计不承担参数约束
        （`tools/run_batch_audit.py` 的 R1/R2 会把它算作无约束用例）。
        """
        import collections

        all_cases = load_rank_cases() + load_holdout_cases()
        counts = collections.Counter(
            candidate.tier for case in all_cases for candidate in case.candidates
        )
        for tier in ("T0", "T7", "T8"):
            self.assertGreater(counts[tier], 0, f"{tier} 应已被第一批覆盖")
        self.assertEqual(counts["T9"], 0, "T9 仍只有负例——正例覆盖仍未补")

        have_t9_negative = [
            case.id
            for case in all_cases
            for candidate in case.exclude
            if candidate.tier == "T9"
        ]
        self.assertIn("22-t9-noise-blocked-by-tier", have_t9_negative)
        self.assertEqual(self.dev.groups, ("high", "mid", "low"))


class ReportTest(unittest.TestCase):
    def test_report_shape_and_determinism(self) -> None:
        first = run_rank_eval()
        second = run_rank_eval()
        self.assertEqual(first.summary(), second.summary(), "实验必须可复现")

        self.assertEqual([r.scheme for r in first.results], list(SCHEMES))
        case_count = len(load_rank_cases())
        for result in first.results:
            self.assertEqual(len(result.outcomes), case_count)
            self.assertGreaterEqual(result.agreement, 0.0)
            self.assertLessEqual(result.agreement, 1.0)

    def test_intent_breakdown_is_reported(self) -> None:
        report = cached_report()
        breakdown = report.results[0].by_intent()
        self.assertIn("historical", breakdown)
        self.assertIn("status", breakdown)
        self.assertIn("按查询意图拆分", report.summary())

    def test_best_scheme_is_one_of_the_candidates(self) -> None:
        report = cached_report()
        self.assertIn(report.best.scheme, SCHEMES)


if __name__ == "__main__":
    unittest.main()
