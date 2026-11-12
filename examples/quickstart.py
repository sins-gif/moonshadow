"""三分钟跑通最小闭环：存 → 查 → 审计挡得住。

为什么是这三步，而不是一个网页？老师要确认的只有两件事：
**它真的在跑**（一次真实召回能给出打分分解），**它真的会拒绝**（一条会失败的审计命令）。
前端展示不了这两件事——「跑起来了」看得见的是数字，「挡得住」看得见的是退出码。

三步都是零依赖、可复跑；本文件不碰任何已有库（用临时目录）。
退出码：三步都符合预期为 `0`，否则 `1`（可以直接接进 CI）。

用法：
    python examples/quickstart.py
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from moonshadow import FixedClock, Store  # noqa: E402
from moonshadow.compress import BaselineExtractor, compile_session  # noqa: E402
from moonshadow.recall import recall  # noqa: E402
from moonshadow.verify import card_text  # noqa: E402

from moon_cli import _lexical_sim  # noqa: E402  与 CLI 同源的词面相似度

BUDGET = 800
SESSION = "quickstart"
FACTS = [
    "客户说预算 12 万，截止 2026-10-15。",
    "以后都用 4 空格缩进，这件小事情别再问我一遍。",
    "重构时不要删除 legacy_adapter.py，迁移步骤见文档。",
]
RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  —— {detail}" if detail else ""))


def audit(args: list[str], expect: int) -> tuple[bool, str]:
    """跑一条审计命令，返回 (退出码是否符合预期, 一句话说明)。"""
    done = subprocess.run(
        [sys.executable, *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8"
    )
    ok = done.returncode == expect
    tail = [line for line in (done.stdout or "").splitlines() if line.strip()]
    return ok, f"exit {done.returncode}（期望 {expect}）｜{tail[-1][:60] if tail else '无输出'}"


def main() -> int:
    print("=" * 72)
    print("步骤 1／3　把三句话存成记忆卡（原文进 raw，卡进 SQLite）")
    print("=" * 72)
    with tempfile.TemporaryDirectory(prefix="quickstart-") as tmpdir:
        clock = FixedClock(datetime.now(timezone.utc))
        store = Store(pathlib.Path(tmpdir), clock=clock)
        try:
            for text in FACTS:
                store.append_message(SESSION, text)
            report = compile_session(store, SESSION, BaselineExtractor(), clock=clock)
            cards = store.cards(session_id=SESSION)
            tiers: dict[str, int] = {}
            for card in cards:
                tiers[str(card["tier"])] = tiers.get(str(card["tier"]), 0) + 1
            card_tokens = sum(_tokens(card_text(card)) for card in cards)
            raw_tokens = sum(_tokens(store.get_raw(str(c["source_ids"][0])) or "") for c in cards)
            print(f"  原文 {len(FACTS)} 条 = {raw_tokens} tok → 卡 {len(cards)} 张 = {card_tokens} tok")
            print(f"  分层：{tiers}（T0 常驻 / T5-T6 短期，由关键词规则判）")
            record("存得进去", len(cards) == len(FACTS), f"{len(cards)} 张卡")

            print()
            print("=" * 72)
            print(f"步骤 2／3　查两次：一次该命中，一次该被硬门挡掉（预算 {BUDGET} tok）")
            print("=" * 72)
            for query, must_hit in (("客户预算多少", True), ("明天天气怎么样", False)):
                cos = {str(c["id"]): _lexical_sim(query, card_text(c)) for c in cards}
                result = recall(store, now=clock.now(), budget_tokens=BUDGET,
                                query_cos=cos, session_id=SESSION)
                stats = result["stats"]
                placed = result["pack_result"].placements
                print(
                    f"  问「{query}」→ 卡 {stats.cards} 张，"
                    f"定性挡 {stats.qualitative_blocked}、定量挡 {stats.quantitative_blocked}，"
                    f"装入 {len(placed)} 张，用 {result['stats'].tokens_used}/{BUDGET} tok"
                )
                for placement in placed:
                    print(
                        f"      [{placement.item.tier}] q={placement.item.score:.4f}"
                        f" {placement.item.section:9} {_one_line(placement.text)}"
                    )
                hit = any(not p.item.reserved for p in placed)
                record(
                    f"「{query}」{'命中相关卡' if must_hit else '不硬塞无关卡'}",
                    hit is must_hit,
                    f"相关卡 {sum(1 for p in placed if not p.item.reserved)} 张",
                )
        finally:
            store.close()

    print()
    print("=" * 72)
    print("步骤 3／3　审计不是装饰：三条命令，其中一条**必须失败**")
    print("=" * 72)
    for args, expect, name in (
        (["tools/run_batch_audit.py"], 0, "新批（22–25）R1/R2：可以合入"),
        (["tools/run_batch_audit.py", "12", "13", "14", "15", "16", "19", "20"], 1,
         "旧批（12–21）R1/R2：必须被拒绝"),
        (["tools/audit_docs.py"], 0, "文档守卫：链接/定值/文件数与仓库一致"),
    ):
        ok, detail = audit(args, expect)
        record(name, ok, detail)

    print()
    failed = [name for name, ok, _ in RESULTS if not ok]
    if failed:
        print(f"结果：{len(RESULTS) - len(failed)}/{len(RESULTS)} 通过，未通过：{'、'.join(failed)}")
        return 1
    print(f"结果：{len(RESULTS)}/{len(RESULTS)} 全部符合预期 —— 闭环成立，且审计真的会拒绝。")
    return 0


def _tokens(text: str) -> int:
    from moonshadow.pack import estimate_tokens

    return estimate_tokens(text)


def _one_line(text: str, limit: int = 34) -> str:
    lines = [line for line in text.split("\n") if line.strip()]
    flat = " ".join(lines[0].split()) if lines else ""
    if len(flat) > limit:
        flat = f"{flat[:limit]}…"
    return f"{flat}（共 {len(lines)} 行）" if len(lines) > 1 else flat


if __name__ == "__main__":
    raise SystemExit(main())
