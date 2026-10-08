"""端到端演示：原文冷存 → 短卡分层 → 召回打分 → 预算装箱 → 精确回取。

运行：
    python examples/demo.py

它同时是一份可执行文档：打印出来的每一步，都对应 SPEC-v1.1 里的一个设计决策。
"""

from __future__ import annotations

import pathlib
import shutil
import sys
from datetime import datetime, timedelta, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow import FixedClock, Store, make_card  # noqa: E402
from moonshadow.pack import Item, estimate_tokens, pack  # noqa: E402
from moonshadow.scoring import next_tier, select  # noqa: E402
from moonshadow.store import CardWriteResult  # noqa: E402
from moonshadow.verify import verify_no_silent_loss  # noqa: E402

WORK = ROOT / ".tmp" / "demo"


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    clock = FixedClock(datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc))
    store = Store(WORK, clock=clock, frame_lines=2)
    print(f"存储根目录：{WORK}")
    print(f"可用可选能力：{sorted(store.capabilities) or '（无，已降级为纯 SQL）'}")
    if store.warnings:
        for warning in store.warnings:
            print(f"  ! {warning}")

    rule("1. 原始层：追加原文（不可变），拿到可寻址的 source_id")
    transcripts = [
        "客户说预算 12 万，截止 2026-10-15，必须兼容旧版 API，除非客户端显式升级。",
        "决定采用方案B，理由是改造成本最低。",
        "张三负责在 9 月 30 日前给出报价。",
        "好的，收到，谢谢！",
    ]
    source_ids = [store.append_message("s1", text) for text in transcripts]
    for source_id, text in zip(source_ids, transcripts):
        print(f"  {source_id}  {text[:28]}…")

    rule("2. 记忆卡层：压缩成 T0-T9 短卡（id 由内容派生，重跑即重放）")
    cards = [
        make_card(
            clock=clock, session_id="s1", tier="T0", importance=10, network="world",
            source_ids=[source_ids[0]],
            summary="硬约束：预算上限 12 万；截止 2026-10-15；必须兼容旧版 API。",
            entities=["客户", "旧版API"], facts=["预算 12 万", "截止 2026-10-15"],
            constraints=["必须兼容旧版 API", "除非客户端显式升级，否则不得破坏兼容"],
            claims=[{"kind": "constraint", "subject_entity": "旧版API", "predicate": "兼容性",
                     "object": "必须兼容"}],
        ),
        make_card(
            clock=clock, session_id="s1", tier="T2", importance=8, network="experience",
            source_ids=[source_ids[0], source_ids[1]],
            summary="决定采用方案B，预算 12 万，截止 2026-10-15。",
            entities=["方案B"], decisions=["采用方案B"], keywords=["预算", "方案"],
            claims=[{"kind": "decision", "subject_entity": "项目", "predicate": "实施方案",
                     "object": "方案B"}],
        ),
        make_card(
            clock=clock, session_id="s1", tier="T3", importance=6, network="experience",
            source_ids=[source_ids[2]],
            summary="张三需在 2026-09-30 前给出报价。",
            entities=["张三"], todos=["张三9月30日前给报价"],
            claims=[{"kind": "todo", "subject_entity": "张三", "predicate": "给出报价",
                     "object": "报价", "due_at": "2026-09-30"}],
        ),
        make_card(
            clock=clock, session_id="s1", tier="T8", importance=1, network="observation",
            source_ids=[source_ids[3]], summary="寒暄确认。",
        ),
    ]
    for card in cards:
        result: CardWriteResult = store.put_card(card)
        print(f"  {card['tier']}  {card['id']}  写入={result.inserted}  取代={list(result.superseded)}")
    print(f"  重复写入同一张卡：inserted={store.put_card(cards[1]).inserted}（幂等重放）")

    rule("3. 冲突不覆盖：预算改了 => 新卡取代旧卡，旧卡留痕")
    revised = make_card(
        clock=clock, session_id="s1", tier="T2", importance=8, network="experience",
        source_ids=[source_ids[0], source_ids[1]],
        summary="决定采用方案C，预算 15 万，截止 2026-11-01。",
        entities=["方案C"], decisions=["采用方案C"],
    )
    print(f"  新卡 {revised['id']}")
    replaced: CardWriteResult = store.put_card(revised)
    print(f"  取代了 {list(replaced.superseded)}")
    old = store.get_card(cards[1]["id"])
    assert old is not None
    print(f"  旧卡状态={old['status']}  superseded_by={old['superseded_by']}  （历史未删除）")

    rule("4. 冷存分帧：归档后仍按 source_id 精确回取原文")
    day = clock.now().date().isoformat()
    index = store.archive_day(day)
    print(f"  容器 {pathlib.Path(index.container).name}：{len(index.frames)} 帧 / {index.total_lines} 行")
    print(f"  原文逐字回取：{store.get_raw(source_ids[0])}")
    print(f"  卡的来源可反查：{store.sources_of_card(cards[0]['id'])}")

    rule("5. 召回：硬门 + 归一化打分（T0 常驻，T8 需显式回溯）")
    now = clock.now()
    active = [c for c in store.cards() if c["status"] == "active"]
    ranked = select(
        active, now=now,
        query_entities=["旧版API", "方案C"],
        query_cos={cards[0]["id"]: 0.62, revised["id"]: 0.81, cards[2]["id"]: 0.44, cards[3]["id"]: 0.05},
        open_task_ids={cards[2]["id"]},
    )
    for row in ranked:
        print(f"  {row['card']['tier']}  score={row['score']:.3f}  {row['card']['id']}  {row['card']['summary'][:26]}…")
    print(f"  未进入召回的层级：T8（无显式回溯请求时不注入噪音）")

    rule("6. 预算装箱：不够就先降级渲染，再丢弃，且绝不裁掉 T0")
    items = [
        Item(id=cards[0]["id"], section="long_term", tier="T0", score=1.0,
             text_full=f"[约束] {cards[0]['summary']}", reserved=True),
        Item(id=revised["id"], section="cards", tier="T2", score=0.81,
             text_full=f"[决策] {revised['summary']} 理由：改造成本最低，但需评估兼容性与迁移风险。",
             text_lite=f"[决策] {revised['summary']}"),
        Item(id="mem_20261001_zzzzzzzzzz", section="cards", tier="T4", score=0.50,
             text_full="[结论] 讨论细节与中间推导，尚未收敛。" * 6,
             text_lite="[结论] 讨论细节与中间推导，尚未收敛。" * 3),
        Item(id=cards[2]["id"], section="todo", tier="T3", score=0.72,
             text_full=f"[待办] {cards[2]['summary']} 负责人：张三，逾期需升级。",
             text_lite=f"[待办] {cards[2]['summary']}"),
    ]
    packed = pack(items, 120)
    for section, text in packed.blocks():
        print(f"  [{section}] {text}")
    print(f"  用量 {packed.tokens_used}/{packed.budget}  tokens，超支={packed.over_budget}")
    print(f"  降级为 lite 的条目：{[p.item.id for p in packed.degraded()]}")
    print(f"  被丢弃的条目：{[(d.item.id, d.reason) for d in packed.dropped]}")

    rule("7. 访问计费与层级升降（只有真正注入上下文才计数）")
    for session in ("s1", "s2", "s3"):
        store.note_access(cards[2]["id"], session)
    store.note_access(cards[2]["id"], "s1")
    store.note_access(cards[2]["id"], "s2")
    hits, sessions = store.access_stats(cards[2]["id"], since=now - timedelta(days=30))
    promoted = next_tier(
        "T3", access_count_30d=hits, sessions_30d=sessions,
        idle_days=0.0, half_life_days=cards[2]["half_life_days"],
    )
    print(f"  近 30 天访问 {hits} 次 / 跨 {sessions} 个会话 => T3 升为 {promoted}")
    stale = next_tier(
        "T4", access_count_30d=0, sessions_30d=0, idle_days=61.0, half_life_days=30.0
    )
    print(f"  闲置 61 天且零访问的 T4 => {stale}（滞回：每周期最多降 1 级）")

    rule("8. 精度校验：关键字段丢失必须判 FAIL（确定性关卡，不依赖模型）")
    all_cards = [cards[0], cards[1], revised, cards[2], cards[3]]
    for source_id, text in zip(source_ids, transcripts):
        citing = [c for c in all_cards if source_id in c["source_ids"]]
        report = verify_no_silent_loss(text, citing)
        print(f"  {source_id}  引用卡 {len(citing)} 张  ->  {report.summary()}")

    rule("9. 反向验证：把「预算 12 万」从卡里删掉，校验必须报 FAIL")
    card_without_amount = dict(cards[0])
    card_without_amount["facts"] = ["截止 2026-10-15"]
    card_without_amount["summary"] = "硬约束：截止 2026-10-15；必须兼容旧版 API。"
    broken = verify_no_silent_loss(transcripts[0], [card_without_amount])
    print(f"  {broken.summary()}")

    store.close()
    print("\n完成。所有数据都在本地，删掉 .tmp/demo 即可完全回滚。")


if __name__ == "__main__":
    main()
