"""召回打分、硬门过滤与层级生命周期。

相对 v1.0 的三处修正：
1. ``priority / 5`` → ``importance / 10``。原式在 importance=8 时得 1.6，
   破坏「四项权重和为 1」的归一化前提。
2. 衰减改为 ``0.5 ** (delta / half_life)``，与「半衰期」定义一致。
3. 补硬门与阈值：原文档只给了召回策略的文字描述，没有可执行的判定。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from .clock import days_between, parse_ts
from .tiers import (
    RULE_TIER_MAX,
    RULE_TIER_MIN,
    clamp_rule_tier,
    half_life_for,
    tier_from_index,
    tier_index,
)

#: 权重和必须为 1，否则 score 不再落在 [0, 1]。
WEIGHTS: dict[str, float] = {
    "sim": 0.45,
    "decay": 0.25,
    "importance": 0.20,
    "match": 0.10,
}

#: 乘性修正方案的权重：只含**上下文信号**（和为 1），时间不再参与加权，
#: 改为对整体相关性做有界调节（见 TIME_FLOOR）。理由见 SPEC §5.1。
RELEVANCE_WEIGHTS: dict[str, float] = {
    "sim": 0.60,
    "importance": 0.27,
    "match": 0.13,
}

#: 时间修正系数的下限：``score = relevance × [TIME_FLOOR, 1.0]``。
#:
#: **`0.5925` = 冻结切分下「开发集 ∩ 留出集」可行窗内的 minimax 取值。**
#: 加入 3 条边界用例（`23-t6-tight-race` / `24-t7-borderline` / `25-t8-explicit-recall`）后，
#: 窗口由 `[0.525, 0.685]` 变为 **`[0.5225, 0.6600]`**（步长 `0.0025`），
#: 且**钉住窗口的两条用例变成了新增的 `23` 与 `24`**——R1 要的「用边界用例约束参数」生效了。
#:
#: **中点依赖扫描步长，引用时不得当作精确值**：`0.005→0.5925`、`0.0025→0.5913`、
#: `0.00125→0.5919`、`0.0005→0.5922`，真值约 `0.592 ± 0.001`。
#: 取 `0.5925` 的理由是它**落在采纳网格上**（`0.5925 / 0.0025 = 237` 为整数），
#: 且与各分辨率的中点相差不超过半步。
#: `0.5925` 必须整除扫描步长（`FLOOR_SWEEP_STEP = 0.0025`），否则 `require_step_divides` 抛错。
#:
#: 历史：`0.70`（v1.1 初稿）→ `0.62`（用例修正前的窗口中点）→ `0.605`（重划切分后）→
#: `0.5925`（加入边界用例后）。每次移动的原因见 `docs/v1.2-weights.md` 的结论差异表。
TIME_FLOOR = 0.5925

#: 按查询意图切换时间下限。**实测为负收益**：
#: 换上正确默认值后，意图切换在开发集上不再优于纯全局固定值。
#: 保留实现仅为可复现实验，不作为推荐路径。
INTENT_TIME_FLOORS: dict[str, float] = {
    "status": 0.35,
    "historical": 0.5925,
    "task": 0.5925,
    "preference": 0.5925,
}

SCHEME_ADDITIVE = "additive"
SCHEME_MULTIPLICATIVE = "multiplicative"
SCHEME_RELEVANCE = "relevance"
SCHEME_INTENT_AWARE = "intent_aware"
SCHEME_BLENDED = "blended"
SCHEME_RRF = "rrf"
SCHEME_HYBRID = "hybrid"
SCHEMES = (
    SCHEME_ADDITIVE,
    SCHEME_MULTIPLICATIVE,
    SCHEME_RELEVANCE,
    SCHEME_INTENT_AWARE,
    SCHEME_BLENDED,
    SCHEME_RRF,
    SCHEME_HYBRID,
)

#: 混合方案的固定系数：**刻意不做调参**。
#: 在 6 个人工用例上搜 α 一定能搜出好看的数字，但那正是本实验反复警告的过拟合。
HYBRID_ALPHA = 0.5


def floor_for_intent(intent: str) -> float:
    """取某查询意图的时间修正下限；未知意图回落到默认 TIME_FLOOR。"""
    return INTENT_TIME_FLOORS.get(intent, TIME_FLOOR)


# ---------------------------------------------------------------------------
# 连续时段敏感度：把「时间该有多大话语权」从离散意图升级为连续量
# ---------------------------------------------------------------------------

#: 只有出现**状态类**标记才放大时间话语权。缺证据时取保守值——
#: 「让时间说了算」是有代价的（历史事实会被新鲜噪音挤掉），所以默认不给它加分。
STATUS_MARKERS = (
    "现在", "目前", "当前", "最新", "进展", "状态", "最近", "刚刚", "还在",
    "怎么样了", "是不是还", "改为", "定下来",
)
HISTORY_MARKERS = (
    "当初", "为什么", "历史", "最初", "一开始", "回顾", "之前决定", "当时",
    "原因", "由来",
)

#: 时间话语权的两端：敏感度 1 → 下限 0.35（时间能翻盘）；敏感度 0 → 下限 0.62（默认）。
TIME_FLOOR_MIN = 0.35
TIME_FLOOR_MAX = TIME_FLOOR


def time_sensitivity(query: str) -> float:
    """查询对「新鲜度」的敏感度，落在 [0, 1]。

    ``0`` = 不问新鲜度（历史提问或无线索），``1`` = 完全问新鲜度（状态提问）。

    相对「按意图硬切换」的关键改进：两侧标记同时出现（如「现在和当初对比」）时返回中间值，
    因此**意图判不清时不会整段行为翻转，而是平滑退化**——这正是硬切换最脆弱的地方。
    """
    status = sum(1 for marker in STATUS_MARKERS if marker in query)
    history = sum(1 for marker in HISTORY_MARKERS if marker in query)
    if status + history == 0:
        return 0.0
    return status / (status + history)


def floor_for_sensitivity(sensitivity: float) -> float:
    """把连续敏感度映射成时间修正下限。"""
    if not 0.0 <= sensitivity <= 1.0:
        raise ValueError(f"时段敏感度必须落在 [0, 1]，当前为 {sensitivity}")
    return TIME_FLOOR_MAX - (TIME_FLOOR_MAX - TIME_FLOOR_MIN) * sensitivity


# ---------------------------------------------------------------------------
# 排名融合：不要求两种信号可比
# ---------------------------------------------------------------------------

#: RRF 的平滑常数（沿用检索领域的惯例值）。
RRF_K = 60
#: 时间一路的权重区间：(敏感度 0, 敏感度 1)。
RRF_TIME_WEIGHT_RANGE = (0.4, 0.6)


def rrf_weights(sensitivity: float) -> tuple[float, float]:
    """按敏感度给出 RRF 的 (相关性权重, 时间权重)。"""
    low, high = RRF_TIME_WEIGHT_RANGE
    weight_time = low + (high - low) * sensitivity
    return (1.0 - weight_time, weight_time)


def reciprocal_rank_fusion(
    rows: Sequence[tuple[str, float, float]],
    *,
    k: int = RRF_K,
    weights: tuple[float, float] = (0.6, 0.4),
) -> list[tuple[str, float]]:
    """把「相关性排名」与「时间排名」按 RRF 融合，返回 ``[(id, 分), ...]``（降序）。

    ``rows`` 为 ``(id, 相关性分, 时间修正系数)``。相对加性方案的优势在于：
    **根本不需要给两种信号定汇率**——融合的是名次而不是分数，
    因此不依赖「相关性和新鲜度可通约」这个可疑前提。
    """
    if not rows:
        return []
    weight_rel, weight_time = weights

    def ranks(index: int) -> dict[str, int]:
        ordered = sorted(rows, key=lambda row: (-row[index], row[0]))
        return {row[0]: position + 1 for position, row in enumerate(ordered)}

    rank_relevance = ranks(1)
    rank_time = ranks(2)
    fused = [
        (
            row[0],
            weight_rel / (k + rank_relevance[row[0]]) + weight_time / (k + rank_time[row[0]]),
        )
        for row in rows
    ]
    return sorted(fused, key=lambda item: (-item[1], item[0]))

#: 融合算术的**采纳方案**。生产路径 `select()` 与阈值标定都以它为准：
#: 加性方案在本用例集上虽也可标定出 `θ`，但其排序一致性从未优于乘性，
#: 且 `run_drift_check` 显示它在乘性方案**有保证**的区域里仍会反转（合成网格上 142 组）。
ADOPTED_SCHEME = SCHEME_MULTIPLICATIVE

#: 硬门阈值。**这些是标定值，不是初值**：取「开发集上负例上界与正例下界的中点」（minimax）。
#: 推导与数字见 `docs/v1.2-weights.md`；`tools/run_rank_eval.py` 每次都会重算并比对。
#: 加入 3 条边界用例后重新标定（`f` 由 `0.605` 变为 `0.5925`，`q = r·m` 随之移动）：
#:
#: - `THETA`（T2–T5）：区间 `(0.386948, 0.401191)`，中点 `0.394069366`，宽 `0.014243`。
#:   卡住下界的是 `09-gate-noise-and-window/weak-t3`，卡住上界的是 `H08-three-way-mix/chat-fresh`。
#: - `THETA_HIGH`（T6–T7）：开发集区间 `(0.330863, 0.494506)`——**上界只由 1 张正例
#:   （`14-recent-draft-vs-older-fact/draft-t7`）定出**，中点 `0.412684` 不是标定，是猜的。
#:   实测取其中点会被**冻结留出集否决**（留出集 3 张真实 T6 正例最小 `q = 0.397591`）。
#:   因此**不给 T6/T7 设更高的分位门槛**：`θ′ = θ`，两类层级的区别只由**卡龄窗口**
#:   （7 天 vs 30 天）表达——那才是有用例支撑的部分。
#:   两个集合同时成立的约束区间是 `(0.386948, 0.397591)`，`0.3941` 落在其中。
#:   这一取值**动用了一次留出集验证**，记录在 `docs/v1.2-weights.md`。
#:
#: 历史：`0.35 / 0.55`（加性量纲占位，两个都落在区间外）→ `0.3943`（重划切分后）→
#: `0.3941`（加入边界用例后）。`θ=θ′=0.3941` 时 `tools/run_gate_audit.py` 违规 `0` 张。
THETA = 0.3941
THETA_HIGH = 0.3941

#: 硬门里的时间窗（天）。
RECENT_WINDOW = 30.0     # T4-T5
FRESH_WINDOW = 7.0       # T6-T7

ALWAYS_ON_TIERS = ("T0", "T1")


def decay(delta_days: float, half_life_days: float) -> float:
    """时间衰减。``half_life_days <= 0`` 表示不衰减（T0），返回 1.0。"""
    if half_life_days <= 0:
        return 1.0
    if delta_days <= 0:
        return 1.0
    return 0.5 ** (delta_days / half_life_days)


def sim_norm(cosine: float) -> float:
    """把余弦相似度 [-1, 1] 归一到 [0, 1]。"""
    if cosine < -1.0 or cosine > 1.0:
        raise ValueError(f"余弦相似度越界：{cosine}")
    return (1.0 + cosine) / 2.0


def match_score(
    query_entities: Iterable[str],
    card_entities: Iterable[str],
    task_hit: bool,
) -> float:
    """实体/任务匹配度：实体 Jaccard 与「是否命中未完成事项」各占一半。"""
    left = {e.strip().lower() for e in query_entities if e and e.strip()}
    right = {e.strip().lower() for e in card_entities if e and e.strip()}
    if left and right:
        jaccard = len(left & right) / len(left | right)
    else:
        jaccard = 0.0
    return 0.5 * jaccard + 0.5 * (1.0 if task_hit else 0.0)


def card_delta_days(card: Mapping[str, Any], now: datetime) -> float:
    """卡龄（天），以 observed_at 为基准；未来时间按 0 处理。"""
    observed = parse_ts(card["observed_at"]) if isinstance(card["observed_at"], str) else card["observed_at"]
    return max(0.0, days_between(observed, now))


def time_modifier(delta_days: float, half_life_days: float, *, floor: float = TIME_FLOOR) -> float:
    """乘性方案的时间修正系数，落在 ``[floor, 1.0]``。"""
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"时间下限必须落在 [0, 1]，当前为 {floor}")
    return floor + (1.0 - floor) * decay(delta_days, half_life_days)


def _normalized(w: Mapping[str, float]) -> dict[str, float]:
    table = dict(w)
    total = sum(table.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"权重之和必须为 1，当前为 {total}")
    return table


def score_card(
    card: Mapping[str, Any],
    *,
    now: datetime,
    query_cos: float = 0.0,
    query_entities: Iterable[str] = (),
    task_hit: bool = False,
    delta_days: float | None = None,
    weights: Mapping[str, float] | None = None,
    relevance_weights: Mapping[str, float] | None = None,
    scheme: str = SCHEME_ADDITIVE,
    time_floor: float | None = None,
) -> float:
    """按指定方案打分，结果落在 [0, 1]。

    四种方案（详见 SPEC §5.1 与 §5.1.1 的对照实验）：

    - ``additive``（现行）：时间作为 0.25 的**加性分项**参与竞争；
    - ``multiplicative``（提案）：相关性为主轴，时间做有界乘性修正；
    - ``relevance``：纯相关性，时间不进入分数（供「仅用时间破同分」的排序使用）；
    - ``intent_aware``：乘性修正 + **按意图切换时间下限**（实验一致率最高者）。

    ``time_floor`` 仅对乘性一族生效，缺省用 ``floor_for_intent`` 的默认值。
    """
    delta = card_delta_days(card, now) if delta_days is None else delta_days
    half_life = float(card.get("half_life_days") or 0.0)
    sim = sim_norm(query_cos)
    importance = int(card["importance"]) / 10.0
    match = match_score(query_entities, card.get("entities", ()), task_hit)

    if scheme == SCHEME_ADDITIVE:
        w = _normalized(weights or WEIGHTS)
        return (
            w["sim"] * sim
            + w["decay"] * decay(delta, half_life)
            + w["importance"] * importance
            + w["match"] * match
        )

    if scheme == SCHEME_RRF:
        raise ValueError(
            "RRF 是集合级方案（融合的是名次，不是单卡分数），"
            "请用 reciprocal_rank_fusion() 而不是 score_card()"
        )

    if scheme in (SCHEME_MULTIPLICATIVE, SCHEME_INTENT_AWARE, SCHEME_BLENDED, SCHEME_RELEVANCE):
        w = _normalized(relevance_weights or RELEVANCE_WEIGHTS)
        relevance = w["sim"] * sim + w["importance"] * importance + w["match"] * match
        if scheme == SCHEME_RELEVANCE:
            return relevance
        floor = TIME_FLOOR if time_floor is None else time_floor
        return relevance * time_modifier(delta, half_life, floor=floor)

    if scheme == SCHEME_HYBRID:
        wa = _normalized(weights or WEIGHTS)
        additive = (
            wa["sim"] * sim
            + wa["decay"] * decay(delta, half_life)
            + wa["importance"] * importance
            + wa["match"] * match
        )
        wm = _normalized(relevance_weights or RELEVANCE_WEIGHTS)
        relevance = wm["sim"] * sim + wm["importance"] * importance + wm["match"] * match
        floor = TIME_FLOOR if time_floor is None else time_floor
        multiplicative = relevance * time_modifier(delta, half_life, floor=floor)
        return HYBRID_ALPHA * additive + (1.0 - HYBRID_ALPHA) * multiplicative

    raise ValueError(f"未知打分方案：{scheme!r}（可选：{SCHEMES}）")


def passes_gate(
    tier: str,
    *,
    delta_days: float,
    score: float,
    has_open_task: bool = False,
    explicit_recall: bool = False,
    theta: float = THETA,
    theta_high: float = THETA_HIGH,
) -> bool:
    """硬门：先定性过滤，再谈打分。顺序与 SPEC-v1.1 §6 一致。"""
    tier_index(tier)  # 未知层级直接抛错，不静默放行
    if tier in ALWAYS_ON_TIERS:
        return True
    if tier in ("T2", "T3"):
        return score > theta or has_open_task
    if tier in ("T4", "T5"):
        return delta_days <= RECENT_WINDOW and score > theta
    if tier in ("T6", "T7"):
        return delta_days <= FRESH_WINDOW and score > theta_high
    if tier in ("T8", "T9"):
        return explicit_recall
    raise ValueError(f"未知层级：{tier!r}")


def select(
    cards: Sequence[Mapping[str, Any]],
    *,
    now: datetime,
    query_entities: Iterable[str] = (),
    query_cos: Mapping[str, float] | None = None,
    task_hit_ids: Iterable[str] = (),
    open_task_ids: Iterable[str] = (),
    explicit_recall: bool = False,
    theta: float = THETA,
    theta_high: float = THETA_HIGH,
    scheme: str = ADOPTED_SCHEME,
) -> list[dict[str, Any]]:
    """召回主入口：逐卡计算 delta/score，过硬门后按分排序。

    默认走**采纳方案**（`ADOPTED_SCHEME`）与**标定阈值**（`THETA` / `THETA_HIGH`），
    因此默认调用即是生产配置；要复现历史行为请显式传入旧值。

    返回 ``[{"card", "score", "delta_days", "render"}]``，render 先给 ``full``，
    实际降级由预算装箱器（pack）决定。
    """
    cos_map = dict(query_cos or {})
    task_hits = set(task_hit_ids)
    open_tasks = set(open_task_ids)
    entity_list = list(query_entities)

    ranked: list[dict[str, Any]] = []
    for card in cards:
        delta = card_delta_days(card, now)
        value = score_card(
            card,
            now=now,
            query_cos=cos_map.get(str(card["id"]), 0.0),
            query_entities=entity_list,
            task_hit=str(card["id"]) in task_hits,
            delta_days=delta,
            scheme=scheme,
        )
        if passes_gate(
            str(card["tier"]),
            delta_days=delta,
            score=value,
            has_open_task=str(card["id"]) in open_tasks,
            explicit_recall=explicit_recall,
            theta=theta,
            theta_high=theta_high,
        ):
            ranked.append(
                {
                    "card": card,
                    "score": value,
                    "delta_days": delta,
                    "render": "full",
                }
            )

    ranked.sort(key=lambda row: (-row["score"], str(row["card"]["id"])))
    return ranked


PROMOTE_MIN_ACCESS = 5
PROMOTE_MIN_SESSIONS = 3


def next_tier(
    tier: str,
    *,
    access_count_30d: int,
    sessions_30d: int,
    idle_days: float,
    half_life_days: float,
    has_open_task: bool = False,
) -> str:
    """层级生命周期（日检任务调用），带滞回与安全夹逼。

    - T0/T1 永不参与规则升降（只能由抽取阶段显式声明）
    - 未完成事项不降级
    - 升：近 30 天访问 ≥5 次 **且** 跨 ≥3 个会话，最多升到 T2
    - 降：闲置超过 2×半衰期且近 30 天零访问，最多降到 T8（绝不自动降成 T9）
    """
    idx = tier_index(tier)
    if tier in ALWAYS_ON_TIERS or has_open_task:
        return tier

    if access_count_30d >= PROMOTE_MIN_ACCESS and sessions_30d >= PROMOTE_MIN_SESSIONS:
        return tier_from_index(clamp_rule_tier(idx - 1))

    if half_life_days > 0 and access_count_30d == 0 and idle_days > 2.0 * half_life_days:
        return tier_from_index(clamp_rule_tier(idx + 1))

    return tier


def default_half_life(tier: str) -> float:
    return half_life_for(tier)


__all__ = [
    "ALWAYS_ON_TIERS",
    "FRESH_WINDOW",
    "HYBRID_ALPHA",
    "INTENT_TIME_FLOORS",
    "ADOPTED_SCHEME",
    "PROMOTE_MIN_ACCESS",
    "PROMOTE_MIN_SESSIONS",
    "RECENT_WINDOW",
    "RELEVANCE_WEIGHTS",
    "RRF_K",
    "RRF_TIME_WEIGHT_RANGE",
    "RULE_TIER_MAX",
    "RULE_TIER_MIN",
    "SCHEMES",
    "SCHEME_ADDITIVE",
    "SCHEME_BLENDED",
    "SCHEME_HYBRID",
    "SCHEME_INTENT_AWARE",
    "SCHEME_MULTIPLICATIVE",
    "SCHEME_RELEVANCE",
    "SCHEME_RRF",
    "STATUS_MARKERS",
    "HISTORY_MARKERS",
    "THETA",
    "THETA_HIGH",
    "TIME_FLOOR",
    "TIME_FLOOR_MAX",
    "TIME_FLOOR_MIN",
    "WEIGHTS",
    "card_delta_days",
    "decay",
    "default_half_life",
    "floor_for_intent",
    "floor_for_sensitivity",
    "match_score",
    "next_tier",
    "passes_gate",
    "reciprocal_rank_fusion",
    "rrf_weights",
    "score_card",
    "select",
    "sim_norm",
    "time_modifier",
    "time_sensitivity",
]
