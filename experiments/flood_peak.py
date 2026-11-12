"""OQ4 造洪峰：把一天的原始消息复制 k 份，读 `PackResult.dropped` 的理由分布。

出处：[`docs/v1.2-future-llm-extractor.md`](../docs/v1.2-future-llm-extractor.md) §6.4 的 OQ4
（「密集交互下的长尾挤兑与逻辑时钟」）。那一节的结论是**先量再改**：在拿到洪峰读数之前，
三条候选解法（逻辑时钟 / 动态 `θ` / 密度惩罚 `D`）都不该进代码；若合成洪峰下挤兑不出现，
这条 OQ 降级为「已评估、未证实」。

## 口径（都写在这里，不留给读者猜）

- **数据**：`eval/gold` 里消息最多的一天（默认 `21-support-thread`，32 条）。用仓库内数据而不是
  某台机器的 `raw/*.jsonl`，否则这个实验换台机器就跑不了。
- **「复制」必须带标记**：卡 ID 与 `dedup_key` 都由内容派生，`compile_session` 保证「重复调用
  同一段输入」只增加 `replayed`——逐字相同的 k 份会被判成重放，k 份又变回 1 份，洪峰根本造不出来。
  所以第 j 份的每条消息都加 `（第 j 份）`：内容几乎相同、但确实是不同的卡。
- **两种时间口径**：
  - `same-day`：k 份全落在同一天——纯体量洪峰，同 `r` 时只能靠 `id` 决胜负；
  - `daily`：第 j 份落在 `day + j` 天——体量 + 新鲜度，这才是 OQ4 说的「同 `r` 时谁新谁赢」。
- **查询**：把当天的 32 条消息逐条当查询（词面相似度，与 `tools/moon_cli.py` 同源）。
- **读数**：
  1. `dropped` 的理由分布（`quota_exhausted` / `no_space`）与 `over_budget` 次数；
  2. 常驻区占比（`reserved_tokens / budget`）；
  3. **第 1 份（最旧）的卡还有多少被装进预算**——挤兑真正伤到的是旧信息还是新信息；
  4. 查询命中的是**同一条消息**的卡时，那份卡是否进了预算（`own_hit`）。

## 判据

合成洪峰下若 `no_space` / `quota_exhausted` 随 k 增长、且旧份保留率下降，挤兑成立；
若 `dropped` 始终为空且保留率不动，OQ4 降级。

用法：
    python experiments/flood_peak.py                 # k = 1,2,4,8,16，两种口径
    python experiments/flood_peak.py --k 1 2 4 --case 22-multi-day-notes
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from moonshadow import FixedClock, Store  # noqa: E402
from moonshadow.compress import BaselineExtractor, compile_session  # noqa: E402
from moonshadow.recall import recall  # noqa: E402
from moonshadow.verify import card_text  # noqa: E402

from moon_cli import _lexical_sim  # noqa: E402  词面相似度与 CLI 同源，不另写一套

GOLD = ROOT / "eval" / "gold"
BUDGET = 800
SESSION = "flood"
MARK = re.compile(r"（第 (\d+) 份）")


def load_case(case_id: str) -> dict:
    path = GOLD / f"{case_id}.json"
    if not path.exists():
        raise SystemExit(f"找不到用例：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def day_start(day: str) -> datetime:
    return datetime.fromisoformat(day).replace(tzinfo=timezone.utc)


def build(root: pathlib.Path, messages: list[str], k: int, mode: str, day: datetime) -> Store:
    """把同一段消息复制 k 份写进一个新库；第 j 份的时间按 mode 决定。"""
    shutil.rmtree(root, ignore_errors=True)
    for j in range(k):
        moment = day if mode == "same-day" else day + timedelta(days=j)
        clock = FixedClock(moment)
        store = Store(root, clock=clock)
        try:
            for text in messages:
                store.append_message(SESSION, f"{text}（第 {j + 1} 份）" if k > 1 else text)
            compile_session(store, SESSION, BaselineExtractor(), clock=clock)
        finally:
            store.close()
    store = Store(root, clock=FixedClock(day))
    return store


def provenance(store: Store, card: dict) -> tuple[int, int]:
    """这张卡来自第几份、第几条消息。取不到就返回 (0, -1)。"""
    for source_id in card.get("source_ids") or []:
        raw = store.get_raw(str(source_id)) or ""
        match = MARK.search(raw)
        copy_no = int(match.group(1)) if match else 1
        stripped = MARK.sub("", raw)
        return copy_no, stripped
    return 0, -1


def run_arm(case: dict, k: int, mode: str, budget: int = BUDGET, demote: bool = False) -> dict:
    messages: list[str] = case["messages"]
    day = day_start(case["day"])
    with tempfile.TemporaryDirectory(prefix="flood-") as tmpdir:
        store = build(pathlib.Path(tmpdir), messages, k, mode, day)
        try:
            if demote:
                # 隔离探针：把 T0/T1 降成 T5，让常驻区不再「无条件、不受配额」。
                # 这不是产品行为，只用来回答「扣掉常驻区这个混淆项之后，同 r 的新旧竞争还在不在」。
                # 直接写 SQL 是故意的：`set_tier` 还没实现（待办 B 类），而这里要的是最小侵入。
                store.conn.execute(
                    "UPDATE card SET tier = 'T5' WHERE tier IN ('T0', 'T1')"
                )
                store.conn.commit()
            cards = store.cards()
            origin = {}
            for card in cards:
                copy_no, text = provenance(store, card)
                origin[str(card["id"])] = (copy_no, text)

            last_moment = day if mode == "same-day" else day + timedelta(days=k - 1)
            reasons: Counter[str] = Counter()
            reserved_share = []
            over = 0
            keep_by_copy: Counter[int] = Counter()
            total_by_copy: Counter[int] = Counter()
            own_hit = Counter()
            own_total = Counter()
            own_any_hit = 0
            own_any_total = 0

            for query in messages:
                cos = {str(card["id"]): _lexical_sim(query, card_text(card)) for card in cards}
                result = recall(
                    store, now=last_moment, budget_tokens=budget,
                    query_cos=cos, session_id=SESSION,
                )
                pack = result["pack_result"]
                reasons.update(d.reason for d in pack.dropped)
                over += int(pack.over_budget)
                reserved_share.append(pack.reserved_tokens / budget)
                packed = {str(p.item.id) for p in pack.placements}
                for card_id, (copy_no, text) in origin.items():
                    if copy_no:
                        total_by_copy[copy_no] += 1
                        if card_id in packed:
                            keep_by_copy[copy_no] += 1
                own = [cid for cid, (_, text) in origin.items() if text == query]
                # 同一信息在 k 份里各有一张卡：只要**任一**副本进预算，这条信息就没丢。
                # 这是「信息保留率」；`own_hit_copy1` 才是 OQ4 关心的「旧份被挤掉」。
                if own:
                    own_any_total += 1
                    if any(cid in packed for cid in own):
                        own_any_hit += 1
                for cid in own:
                    copy_no = origin[cid][0]
                    own_total[copy_no] += 1
                    if cid in packed:
                        own_hit[copy_no] += 1

            n_queries = len(messages)
            return {
                "k": k,
                "mode": mode,
                "demote": demote,
                "budget": budget,
                "cards": len(cards),
                "long_term": sum(1 for c in cards if str(c["tier"]) in ("T0", "T1")),
                "reasons": dict(reasons),
                "over_budget": over,
                "reserved_share": sum(reserved_share) / n_queries,
                "keep_copy1": keep_by_copy[1] / total_by_copy[1] if total_by_copy[1] else 0.0,
                "keep_newest": (
                    keep_by_copy[k] / total_by_copy[k] if total_by_copy[k] else 0.0
                ),
                "own_hit_any": own_any_hit / own_any_total if own_any_total else 0.0,
                "own_hit_copy1": own_hit[1] / own_total[1] if own_total[1] else 0.0,
                "own_hit_newest": own_hit[k] / own_total[k] if own_total[k] else 0.0,
                "per_copy_cards": {copy_no: total for copy_no, total in sorted(total_by_copy.items())},
            }
        finally:
            store.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OQ4 造洪峰（先量再改）")
    parser.add_argument("--case", default="21-support-thread", help="用哪天当洪峰原料（eval/gold 的 id）")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8, 16], help="复制份数")
    parser.add_argument(
        "--budget", type=int, nargs="+", default=[800, 2000, 4000],
        help="注入预算（token）；要扫多个，因为 k=1 就已经装不下时，挤兑不是洪峰造成的",
    )
    parser.add_argument("--json", metavar="PATH", help="把逐格读数写成 JSON（默认不写文件）")
    parser.add_argument(
        "--demote-reserved", action="store_true",
        help="隔离探针：把 T0/T1 降成 T5，看扣掉常驻区之后是否还存在分区内挤出",
    )
    args = parser.parse_args(argv)

    case = load_case(args.case)
    print(f"数据：{case['id']}（{case['day']}，{len(case['messages'])} 条消息）")
    print(f"查询 = 当天全部消息（{len(case['messages'])} 条）；每种 (口径, 预算) 各跑一遍\n")
    print("术语：旧份保留 = 第 1 份（最旧）的卡被装进预算的比例；")
    print("      旧份命中 = 只有第 1 份那张卡算命中；信息保留 = 任一副本进预算即算命中\n")

    rows = []
    for budget in args.budget:
        for mode in ("same-day", "daily"):
            print(f"=== 预算 {budget} tok ／ 口径 {mode}")
            print(
                "  k   卡数  常驻  丢弃理由                超预算  常驻占比"
                "  旧份保留  新份保留  旧份命中  信息保留"
            )
            for k in args.k:
                row = run_arm(case, k, mode, budget, args.demote_reserved)
                rows.append(row)
                reasons = ",".join(f"{name}×{n}" for name, n in sorted(row["reasons"].items())) or "—"
                print(
                    f" {k:3d} {row['cards']:5d} {row['long_term']:5d}  {reasons:22}  {row['over_budget']:6d}"
                    f"  {row['reserved_share'] * 100:7.1f}%  {row['keep_copy1'] * 100:7.1f}%"
                    f"  {row['keep_newest'] * 100:7.1f}%  {row['own_hit_copy1'] * 100:7.1f}%"
                    f"  {row['own_hit_any'] * 100:7.1f}%"
                )
            print()

    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"逐格读数已写入 {args.json}\n")

    baseline = {(r["budget"], r["mode"], r["demote"]): r for r in rows if r["k"] == 1}
    print("== 判定")
    print("基线（k=1，没有洪峰）：")
    for (budget, mode, demote), row in sorted(baseline.items()):
        print(
            f"  预算 {budget:5d}／{mode:8}：卡 {row['cards']:2d}，丢弃 {sum(row['reasons'].values()):3d}"
            f"，旧份保留 {row['keep_copy1'] * 100:5.1f}%，信息保留 {row['own_hit_any'] * 100:5.1f}%"
            f"，超预算 {row['over_budget']} 次"
        )

    # 挤兑是否**由洪峰引入**，要看扣掉基线之后还剩什么。
    # 预算太小的时候基线的丢弃就已经饱和（一天 32 条卡 > 800 tok），那不算洪峰的账。
    introduced = []
    got_worse = []
    for row in rows:
        if row["k"] == 1:
            continue
        base = baseline[(row["budget"], row["mode"], row["demote"])]
        base_drop = sum(base["reasons"].values())
        if sum(row["reasons"].values()) > base_drop:
            introduced.append(row)
        if row["own_hit_any"] + 1e-9 < base["own_hit_any"]:
            got_worse.append((row, base))

    if introduced:
        worst = max(introduced, key=lambda r: sum(r["reasons"].values()))
        print(
            f"\n洪峰**加**了丢弃：{len(introduced)} 个非基座配置；最严重 k={worst['k']}、"
            f"预算 {worst['budget']}、{worst['mode']} → {worst['reasons']}"
            f"（基线同预算同口径 {sum(baseline[(worst['budget'], worst['mode'], worst['demote'])]['reasons'].values())} 条）"
        )
    else:
        print("\n洪峰**没有加**丢弃：每个配置的丢弃数与同预算 k=1 相同。")

    if got_worse:
        row, base = min(got_worse, key=lambda pair: pair[0]["own_hit_any"])
        print(
            f"信息保留率下降：{len(got_worse)} 个配置低于基线；最低 k={row['k']}、"
            f"预算 {row['budget']}、{row['mode']}：{row['own_hit_any'] * 100:.1f}%"
            f"（基线 {base['own_hit_any'] * 100:.1f}%）"
        )
    else:
        print("信息保留率没有下降：任一副本进预算的比例在所有配置下都不低于基线。")

    if introduced or got_worse:
        print(
            "\n结论：合成洪峰下挤兑**出现**，OQ4 维持「待解」——"
            "下一步才是量「谁被挤掉」与分区配额，三条候选解法仍不进代码。"
        )
    else:
        print("\n结论：合成洪峰下挤兑**不出现**。按 docs/v1.2-future-llm-extractor.md §6.4 的判据，"
              "OQ4 降级为「已评估、未证实」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
