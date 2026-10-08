"""Phase 2.5：评测关卡。

没有这一层，抽取器改好改坏只能靠感觉——而本系统的全部价值押在「精度不丢」上。
因此把它做成硬关卡：``python tools/run_eval.py`` 不达标就退出码非 0。

指标是**一对**而非一个：

- **关键字段保留率**（必须 ≥ 阈值）：日期、金额、URL、代码不能丢；
- **压缩率**（越大越好）：原文 token / 卡 token。

只盯其中一个都会走偏：基线抽取器保留率 1.0 却没有压缩价值；激进摘要压缩率很高却会丢字段。
两个数一起看，才知道一个抽取器相对基线是「更好」还是「只是更短」。

实测基线（`BaselineExtractor`）的压缩率约 **0.35x**，即它在膨胀 token——因为 SPEC 要求
T0/T1「保留原文」，基线把整段原文抄进了 `raw_quote`，加上 facts 又复述了一遍。
这是**符合规范的基线代价**，不是 bug：它标出了「不丢字段」的召回上限，
模型抽取器的任务是在保持该保留率的前提下把压缩率做上去。
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .clock import FixedClock, parse_ts
from .compress import BaselineExtractor, Extractor, compile_session
from .pack import estimate_tokens
from .store import Store
from .verify import card_text_many, extract_key_fields, normalize, verify_no_silent_loss

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_GOLD_DIR = ROOT / "eval" / "gold"
WORK_DIR = ROOT / ".tmp" / "eval"

#: 阈值来自 SPEC-v1.1 §9.2。未达标即视为关卡未通过。
DEFAULT_MIN_RECALL = 0.98


@dataclass
class GoldCase:
    """一条金标准：一段输入 + 必须活下来的东西。"""

    id: str
    messages: list[str]
    day: str = "2026-10-01"
    expected_key_fields: list[str] = field(default_factory=list)
    expected_tiers: list[str] = field(default_factory=list)
    note: str = ""

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "GoldCase":
        expected = payload.get("expected", {})
        return cls(
            id=str(payload["id"]),
            messages=[str(item) for item in payload["messages"]],
            day=str(payload.get("day", "2026-10-01")),
            expected_key_fields=[str(item) for item in expected.get("key_fields", [])],
            expected_tiers=[str(item) for item in expected.get("tiers_present", [])],
            note=str(payload.get("note", "")),
        )


def load_gold(directory: str | pathlib.Path | None = None) -> list[GoldCase]:
    """加载金标准集，按 id 排序保证报告可复现。"""
    target = pathlib.Path(directory) if directory is not None else DEFAULT_GOLD_DIR
    cases = []
    for path in sorted(target.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            cases.append(GoldCase.from_json(json.load(handle)))
    return sorted(cases, key=lambda case: case.id)


@dataclass
class CaseResult:
    case_id: str
    ok: bool
    recall: float
    total_fields: int
    lost: dict[str, list[str]] = field(default_factory=dict)
    missing_expected: list[str] = field(default_factory=list)
    missing_tiers: list[str] = field(default_factory=list)
    compression: float = 0.0
    cards: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def summary(self) -> str:
        verdict = "PASS" if self.ok else "FAIL"
        head = (
            f"{verdict}  {self.case_id:<28} 保留率={self.recall:.3f} "
            f"压缩率={self.compression:.2f}x 卡数={self.cards}"
        )
        details = []
        if self.lost:
            details.append(f"丢字段={self.lost}")
        if self.missing_expected:
            details.append(f"缺预期字段={self.missing_expected}")
        if self.missing_tiers:
            details.append(f"缺预期层级={self.missing_tiers}")
        if self.error:
            details.append(f"异常={self.error}")
        return head + (("\n      " + "；".join(details)) if details else "")


@dataclass
class EvalReport:
    results: list[CaseResult]
    min_recall: float = DEFAULT_MIN_RECALL

    @property
    def key_field_recall(self) -> float:
        total = sum(r.total_fields for r in self.results)
        if total == 0:
            return 1.0
        lost = sum(sum(len(v) for v in r.lost.values()) for r in self.results)
        return (total - lost) / total

    @property
    def compression_ratio(self) -> float:
        ratios = [r.compression for r in self.results if r.compression > 0]
        return sum(ratios) / len(ratios) if ratios else 0.0

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results) and self.key_field_recall >= self.min_recall

    def summary(self) -> str:
        lines = [result.summary() for result in self.results]
        verdict = "PASS" if self.ok else "FAIL"
        lines.append(
            f"{verdict}  合计：关键字段保留率={self.key_field_recall:.4f}"
            f"（阈值 {self.min_recall}），平均压缩率={self.compression_ratio:.2f}x，"
            f"用例 {sum(1 for r in self.results if r.ok)}/{len(self.results)} 通过"
        )
        if self.compression_ratio < 1.0:
            lines.append(
                "WARN  平均压缩率 < 1.0：抽取器在**膨胀** token。PASS 只说明没丢字段，"
                "不说明压缩有效——模型抽取器必须在保持保留率的同时把这一项做到 > 1。"
            )
        return "\n".join(lines)


def run_case(
    case: GoldCase,
    extractor: Extractor,
    *,
    session_id: str | None = None,
    workdir: str | pathlib.Path | None = None,
) -> CaseResult:
    """跑一条金标准：写入原文 → 压缩 → 校验关键字段与预期层级 → 算压缩率。"""
    session = session_id or f"gold-{case.id}"
    work = pathlib.Path(workdir) if workdir is not None else WORK_DIR / case.id
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    clock = FixedClock(parse_ts(f"{case.day}T09:00:00+00:00"))
    store = Store(work, clock=clock)
    try:
        for text in case.messages:
            store.append_message(session, text)
        compile_session(store, session, extractor, clock=clock)

        cards = store.cards(session_id=session)
        raw_text = "\n".join(case.messages)
        report = verify_no_silent_loss(raw_text, cards)

        extracted = extract_key_fields(raw_text)
        total_fields = sum(len(values) for values in extracted.values())
        lost_count = sum(len(values) for values in report.missing.values())
        recall = (total_fields - lost_count) / total_fields if total_fields else 1.0

        produced = normalize(card_text_many(cards))
        missing_expected = [
            value for value in case.expected_key_fields if normalize(value) not in produced
        ]
        tiers = {str(card["tier"]) for card in cards}
        missing_tiers = [tier for tier in case.expected_tiers if tier not in tiers]

        raw_tokens = sum(estimate_tokens(text) for text in case.messages)
        card_tokens = sum(estimate_tokens(card_text_many([card])) for card in cards) or 1
        return CaseResult(
            case_id=case.id,
            ok=report.ok and not missing_expected and not missing_tiers,
            recall=recall,
            total_fields=total_fields,
            lost=dict(report.missing),
            missing_expected=missing_expected,
            missing_tiers=missing_tiers,
            compression=raw_tokens / card_tokens,
            cards=len(cards),
            warnings=list(report.warnings),
        )
    except Exception as exc:  # 单条用例异常不应掩盖其他用例的结果
        return CaseResult(
            case_id=case.id, ok=False, recall=0.0, total_fields=0, error=f"{type(exc).__name__}: {exc}"
        )
    finally:
        store.close()


def run_eval(
    cases: Sequence[GoldCase] | None = None,
    extractor: Extractor | None = None,
    *,
    gold_dir: str | pathlib.Path | None = None,
    min_recall: float = DEFAULT_MIN_RECALL,
) -> EvalReport:
    """跑完整评测。默认用规则基线抽取器作为对照基准。"""
    selected = list(cases) if cases is not None else load_gold(gold_dir)
    engine = extractor or BaselineExtractor()
    return EvalReport(
        results=[run_case(case, engine) for case in selected],
        min_recall=min_recall,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：不达标返回非 0，可直接挂到 CI。"""
    args = list(sys.argv[1:] if argv is None else argv)
    gold_dir = args[0] if args else None
    report = run_eval(gold_dir=gold_dir)
    print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover - 需先安装包或设置 PYTHONPATH
    raise SystemExit(main())
