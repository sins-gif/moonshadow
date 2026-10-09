"""stub 抽取器探针：把「接口应该长什么样」跑成数据。

    python tools/run_extractor_stub.py

`src/moonshadow/extractor.py` 是 **STUB**（非生产路径）：它不接 LLM、不被
`BaselineExtractor` 调用、不接入 `select()`。本工具是它唯一的调用方。

它回答三个问题（对应 v1.3 Phase 1 的接口设计）：

1. **卡数**：`474 → ?`，离批内归并的下界（`241`）有多近；
2. **压缩率**：两层口径（与 `run_eval` 同源，并被交叉核对），是否进入 Phase 0 实测过的可达区；
3. **接口漏洞**：契约没规定、因此 stub 只能先猜的地方——把「猜」逐条打印出来。

退出码：契约检查（`T0/T1` 卡不含 `raw_quote`、硬字段无丢失、每条消息都被引用）
全部通过为 `0`，否则为 `1`。星号开头的接口诊段只报告、不判定。
"""

from __future__ import annotations

import pathlib
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow.clock import FixedClock, parse_ts  # noqa: E402
from moonshadow.compress import compile_session, classify_fast_path  # noqa: E402
from moonshadow.eval import (  # noqa: E402
    GoldCase,
    WORK_DIR,
    load_gold,
    run_eval,
)
from moonshadow.extractor import (  # noqa: E402
    BatchDiagnostics,
    Message,
    canonical_item,
    extract_batch,
    reset_diagnostics,
    last_batch_diagnostics,
)
from moonshadow.pack import estimate_tokens  # noqa: E402
from moonshadow.store import Store  # noqa: E402
from moonshadow.verify import CARD_TEXT_FIELDS, card_text_many, extract_key_fields, normalize  # noqa: E402


class StubExtractor:
    """把 `extractor.extract_batch` 适配成管线认识的 `Extractor`。

    适配器**必须自己编一个 `tier_hint`**：管线的 `compile_session` 不产出这个参数。
    这是接口漏洞 #1 的落点，探针里显式传空串（而不是编一个值），让「没人填」这件事可见。
    """

    def __init__(
        self,
        *,
        sentence_fields: bool = True,
        summary_style: str = "digest",
        allow_duplication: bool = False,
        name: str = "stub-rules",
    ) -> None:
        self.sentence_fields = sentence_fields
        self.summary_style = summary_style
        self.allow_duplication = allow_duplication
        self.name = name

    def __call__(self, messages: Sequence[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
        batch = [Message.from_mapping(message) for message in messages]
        for card in extract_batch(
            batch,
            tier_hint="",
            sentence_fields=self.sentence_fields,
            summary_style=self.summary_style,
            allow_duplication=self.allow_duplication,
        ):
            yield card.to_draft()


@dataclass
class CaseAudit:
    case_id: str
    messages: int = 0
    cards: int = 0
    raw_tokens: int = 0
    card_tokens: int = 0
    t0_t1_cards: int = 0
    raw_quote_cards: int = 0
    uncited_messages: list[str] = field(default_factory=list)
    lost_in_card: list[str] = field(default_factory=list)
    unreachable: list[str] = field(default_factory=list)
    fields_used: dict[str, int] = field(default_factory=dict)
    field_tokens: dict[str, int] = field(default_factory=dict)

    @property
    def compression(self) -> float:
        return self.raw_tokens / self.card_tokens if self.card_tokens else 0.0


def _iter_sentences(text: str) -> list[str]:
    from moonshadow.compress import SENTENCE_SPLIT_RE

    return [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]


def audit_case(case: GoldCase) -> tuple[CaseAudit, list[BatchDiagnostics]]:
    """跑一条用例的**完整流水线**（与 `eval.run_case` 同一条路径），并审计产出的卡。"""
    session = f"stub-{case.id}"
    work = WORK_DIR / f"stub-{case.id}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    clock = FixedClock(parse_ts(f"{case.day}T09:00:00+00:00"))
    store = Store(work, clock=clock)
    reset_diagnostics()
    try:
        for text in case.messages:
            store.append_message(session, text)
        compile_session(store, session, StubExtractor(), clock=clock)
        cards = store.cards(session_id=session)
        rows = store.messages_for_session(session)
        sources = {str(row["id"]): store.get_raw(str(row["id"])) for row in rows}

        audit = CaseAudit(case_id=case.id, messages=len(rows), cards=len(cards))
        audit.raw_tokens = sum(estimate_tokens(text) for text in case.messages)
        audit.card_tokens = sum(
            estimate_tokens(card_text_many([card])) for card in cards
        ) or 1

        cited: set[str] = set()
        for card in cards:
            cited.update(str(source_id) for source_id in card.get("source_ids", ()))
            if str(card["tier"]) in ("T0", "T1"):
                audit.t0_t1_cards += 1
                if card.get("raw_quote"):
                    audit.raw_quote_cards += 1
            text = card_text_many([card])
            for name in CARD_TEXT_FIELDS:
                value = card.get(name)
                if value:
                    audit.fields_used[name] = audit.fields_used.get(name, 0) + 1
                    if isinstance(value, str):
                        pieces = [value]
                    else:
                        pieces = [str(item) for item in value]
                    audit.field_tokens[name] = audit.field_tokens.get(name, 0) + sum(
                        estimate_tokens(piece) for piece in pieces
                    )
            for values in extract_key_fields(text).values():
                del values  # 只用于计数：见下面的硬字段全量比对
        audit.uncited_messages = sorted(
            str(row["id"]) for row in rows if str(row["id"]) not in cited
        )

        # 硬字段：卡内可查（契约）与回链可达（v1.3 C5）分别统计
        linked = normalize("".join(sources[sid] for sid in cited if sid in sources))
        haystack = normalize(card_text_many(cards))
        for name, values in extract_key_fields("\n".join(case.messages)).items():
            for value in values:
                key = normalize(value)
                if key not in haystack:
                    audit.lost_in_card.append(f"{name}:{value}")
                if key not in linked:
                    audit.unreachable.append(f"{name}:{value}")

        return audit, last_batch_diagnostics()
    finally:
        store.close()


def main() -> int:
    cases = load_gold()
    audits: list[CaseAudit] = []
    diagnostics: list[BatchDiagnostics] = []

    print("stub 抽取器探针（STUB：非生产路径，不接 LLM，不接入 select()）")
    print(f"金标准集：{len(cases)} 条")
    print()
    print(f"  {'用例':<26}{'消息':>5}{'卡数':>5}{'原文tok':>9}{'卡tok':>8}{'压缩率':>9}")
    for case in cases:
        audit, batch_diagnostics = audit_case(case)
        audits.append(audit)
        diagnostics.extend(batch_diagnostics)
        print(
            f"  {audit.case_id:<26}{audit.messages:>5}{audit.cards:>5}"
            f"{audit.raw_tokens:>9,}{audit.card_tokens:>8,}{audit.compression:>8.3f}x"
        )

    total_messages = sum(audit.messages for audit in audits)
    total_cards = sum(audit.cards for audit in audits)
    raw = sum(audit.raw_tokens for audit in audits)
    card = sum(audit.card_tokens for audit in audits)
    mean = sum(audit.compression for audit in audits if audit.compression) / len(audits)

    print()
    print("【1】卡数")
    print(f"  基线（一条消息一张卡）：{total_messages}  →  stub：{total_cards}")
    print(f"  批内归并下界（一个 (批, tier) 一张卡，见 Phase 0 实测）：241")
    print(f"  结构下界（一个 (用例, tier) 一张卡）：126")

    print()
    print("【2】压缩率（两层口径，与 run_eval 同源）")
    print(f"  逐用例平均：{mean:.4f}x    token 加权：{raw / card:.4f}x")
    print(f"  原文 {raw:,} tok → 卡 {card:,} tok")
    report = run_eval(cases=cases, extractor=StubExtractor())
    print(
        f"  交叉核对 run_eval：逐用例平均 {report.compression_ratio:.4f}x、"
        f"token 加权 {report.weighted_compression:.4f}x、"
        f"卡数 {sum(r.cards for r in report.results)}、PASS {report.ok}"
    )
    same = (
        abs(report.compression_ratio - mean) < 1e-9
        and abs(report.weighted_compression - raw / card) < 1e-9
    )
    print(f"  两侧一致：{same}")

    print()
    print("  归因阶梯（同一份代码、同一个卡数；A/B/C 三行 summary 口径相同，只有字段分配不同）：")
    print(
        f"      {'变体':<42}{'逐用例平均':>11}{'token 加权':>12}{'卡数':>6}{'过门':>6}{'C6违规':>8}"
    )
    for label, engine in (
        ("[现状] 遵守 C6+C8（四字段 + 片段摘要）", StubExtractor()),
        ("A 违反 C6（复述），其余同现状", StubExtractor(
            allow_duplication=True, name="stub-dup")),
        ("C 遵守 C6 只写 facts，其余同现状", StubExtractor(
            sentence_fields=False, name="stub-facts-only")),
        ("D 遵守 C6，但 summary 极简（违反 C8）", StubExtractor(
            summary_style="minimal", name="stub-minimal-summary")),
    ):
        variant = run_eval(cases=cases, extractor=engine)
        violations = sum(len(r.field_violations) for r in variant.results)
        print(
            f"      {label:<42}{variant.compression_ratio:>10.4f}x"
            f"{variant.weighted_compression:>11.4f}x"
            f"{sum(r.cards for r in variant.results):>6}"
            f"{'PASS' if variant.ok else 'FAIL':>6}{violations:>8}"
        )
    print("      现状→A = C6 的价值（同一句话不再写第二遍）；现状 vs D = C8 的代价（摘要必须承载内容）")
    print("      现状→C = 逐句字段承载整句原文的成本（只有抽象能去掉，规则版做不到）")

    print()
    print("  卡 token 的字段构成（全部 241 张卡，按字段归集）：")
    field_tokens: dict[str, int] = {}
    for audit in audits:
        for name, tokens in audit.field_tokens.items():
            field_tokens[name] = field_tokens.get(name, 0) + tokens
    total_card = sum(field_tokens.values()) or 1
    for name, tokens in sorted(field_tokens.items(), key=lambda item: -item[1]):
        print(f"      {name:<12}{tokens:>8,} tok   {tokens / total_card:>6.1%}")

    print()
    print("【3】契约检查（决定退出码）")
    raw_quote_cards = sum(audit.raw_quote_cards for audit in audits)
    uncited = [(audit.case_id, audit.uncited_messages) for audit in audits if audit.uncited_messages]
    lost = [(audit.case_id, audit.lost_in_card) for audit in audits if audit.lost_in_card]
    unreachable = [(audit.case_id, audit.unreachable) for audit in audits if audit.unreachable]
    t0_t1 = sum(audit.t0_t1_cards for audit in audits)
    print(f"  T0/T1 卡 {t0_t1} 张，其中带 raw_quote 的：{raw_quote_cards}")
    print(f"  没被任何卡引用的消息：{sum(len(a[1]) for a in uncited)} 条")
    for case_id, ids in uncited[:5]:
        print(f"      {case_id}: {ids[:6]}")
    print(f"  卡里查不到的硬字段（抽象损失，C5 契约要求为 0）：{sum(len(a[1]) for a in lost)} 个")
    for case_id, values in lost[:5]:
        print(f"      {case_id}: {values[:6]}")
    print(f"  连回链也追不到的硬字段：{sum(len(a[1]) for a in unreachable)} 个")
    for case_id, values in unreachable[:5]:
        print(f"      {case_id}: {values[:6]}")

    print()
    print("  卡上各字段的使用次数：")
    usage: dict[str, int] = {}
    for audit in audits:
        for name, count in audit.fields_used.items():
            usage[name] = usage.get(name, 0) + count
    for name in CARD_TEXT_FIELDS:
        print(f"      {name:<12}{usage.get(name, 0):>5}")

    print()
    print("【4】接口诊段（契约没规定、stub 只能先猜的地方；只报告不判定）")
    multi_tier = [item for item in diagnostics if len(item.tiers) > 1]
    print(f"  * 批次总数：{len(diagnostics)}")
    print(f"  * 一批跨多个 tier 的批次：{len(multi_tier)} / {len(diagnostics)}"
          f"（占比 {len(multi_tier) / max(len(diagnostics), 1):.1%}）"
          " → 「一批对应一个 tier」的签名假设不成立")
    print(f"  * 抽取器收到过 tier_hint 的批次：{sum(1 for i in diagnostics if i.tier_hint_used)}"
          "（管线不产出该参数，探针只能传空串）")
    no_item_batches = [i for i in diagnostics if i.items == 0]
    print(f"  * 一条硬字段都没有的批次：{len(no_item_batches)}"
          " → facts 会为空，卡只剩 summary，契约需规定这种卡是否可以不出")
    print(f"  * 批内信息项：{sum(i.items for i in diagnostics)} 个，"
          f"其中跨 ≥2 条消息重复的：{sum(i.items_multi_message for i in diagnostics)} 个"
          "（批内归并能吃掉的部分）")
    print(f"  * 不含硬字段的消息：{sum(i.messages_without_items for i in diagnostics)} 条"
          " → 它们靠 decisions/todos/constraints 或 source_ids 保留，契约未规定")

    fast_path = 0
    for case in cases:
        fast_path += sum(1 for text in case.messages if classify_fast_path(text))
    print(f"  * 走规则快路径、**从不进入抽取器**的消息：{fast_path} / {total_messages} 条"
          " → 「抽取器看到一批消息」实际看到的是被抽走快路径后的**不连续**一批")

    canonical: dict[str, set[str]] = {}
    for case in cases:
        for text in case.messages:
            for values in extract_key_fields(text).values():
                for value in values:
                    canonical.setdefault(canonical_item(value), set()).add(normalize(value))
    ambiguous = {key: forms for key, forms in canonical.items() if len(forms) > 1}
    print(f"  * 同一信息项存在多种写法（按数字+单位折算后相同、字面不同）：{len(ambiguous)} 组"
          " → 跨批注册表若按字面值建索引，这些会各留一处载体")
    for key, forms in list(ambiguous.items())[:5]:
        print(f"      {sorted(forms)[:4]}")

    contract_ok = raw_quote_cards == 0 and not lost and not uncited
    return 0 if contract_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
