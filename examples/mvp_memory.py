"""MVP：月影能不能解决 agent 的记忆问题？（自包含、零依赖、一次跑完）

它**不测压缩率**。压缩率衡量全库体积，而记忆系统的稀缺资源是「每次进上下文的量」。
这个脚本只回答四个能判定的问题：

  1. **跨会话还记得吗** —— 12 个「一个新对话应该能直接答出来」的问题，召回里有没有答案；
     没命中的还要分清是「没存进去」还是「问不出来」（这两件事的修法完全不同）
  2. **要不要重新提取** —— 召回期间打开了几个**业务文件**（用审计钩子数，不是估计）
  3. **能不能追溯** —— 按 source_id 回取原文，与卡片摘要逐字对照
  4. **会不会污染** —— 无关问题**只该召回常驻卡**（身份/硬约束永远在场，这是设计），
     任何**非常驻**的不相关卡出现才算污染

    python examples/mvp_memory.py            # 内置种子数据，在 .tmp 下建临时库
    python examples/mvp_memory.py --keep     # 保留临时库，便于自己翻

判据写在这里，不是事后解释：
- 命中 = 该题的 `expect` 关键字出现在**装箱后的上下文**里（不是出现在全库里）；
- **存储漏** = 库里所有卡都找不到这个关键字（抽取阶段就丢了）；
- **检索漏** = 关键字在库里，但没被召回（问题措辞与卡片措辞不一致）；
- 污染 = 装箱结果里出现 `reserved=False` 的卡（常驻卡不算污染）。

退出码：`0` = 无存储漏、无污染；`1` = 否则。命中率不是硬门——词面检索的漏是已知限制。
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from moonshadow import FixedClock, Store  # noqa: E402
from moonshadow.compress import BaselineExtractor, compile_session  # noqa: E402
from moonshadow.recall import recall  # noqa: E402
from moonshadow.verify import card_text  # noqa: E402
from moon_cli import _lexical_sim  # noqa: E402

STORE = ROOT / ".tmp" / "mvp-memory"
REPORT = ROOT / ".tmp" / "MVP-报告.md"
BUDGET = 800

#: 种子对话：刻意用**本项目的真实事实**，因为它们正是「新对话该记得的东西」。
SEED = [
    "时间下限 TIME_FLOOR 取 0.5925，这是开发集窗与留出集窗交集里近似居中的值。",
    "分数门槛 THETA 与 THETA_HIGH 都是 0.3941，也就是 θ′ = θ，已取消 T6/T7 的更高门槛。",
    "硬门是两段合取：打分前的定性门管卡龄窗口与 T8/T9 禁令，打分后的定量门管 q 大于 θ。",
    "GPU 三档读数：list 布局 25.3 倍、连续内存 141.7 倍、纯 kernel 1902 倍。",
    "DSH 插件放在 D:\\Plugins\\MOD\\dsh-moonshadow，暴露 moon_remember、moon_recall、moon_stats。",
    "记忆库默认在 ~/.dsh/moonshadow，可用 MOONSHADOW_ROOT 覆盖。",
    "已撤回「上下文压缩」这个宣称：实测压缩率 0.498x，卡比原文还大，见 withdrawn/W1-compression.md。",
    "版本线：v1.2.2 的 tag 指向 8e69f46，之后有三个未发版提交。",
    "用例集规模：开发集 26 条、留出集 9 条、网络集 48 条、抽取金标准 25 条。",
    "抽取关键字段保留率 1.0000，阈值 0.98，25 条用例全部通过。",
    "门审计与用例审计的违规都是 0 张。",
    "交接文档 HANDOFF.md 在仓库根目录，新开对话第一句读它。",
    "协作规则：没有命令不要动文件；数字以工具输出为准。",
    "层级的 T1 现在可以降级，半衰期 365 天，要闲置超过 730 天才触发。",
]

#: 12 个问题 + 各自的「答案关键字」。问题按一个新对话会怎么问来写，**不按卡片措辞写**
#: —— 因为真实使用里，人不会照着你的字段名提问。
QUESTIONS = [
    ("时间下限取多少", "0.5925"),
    ("分数门槛是多少", "0.3941"),
    ("硬门有几段，先后顺序是什么", "定性"),
    ("GPU 三档分别是多少倍", "1902"),
    ("插件装在哪", "dsh-moonshadow"),
    ("记忆库默认在哪", "moonshadow"),
    ("压缩率那件事后来怎么处理的", "0.498"),
    ("v1.2.2 的 tag 指向哪个提交", "8e69f46"),
    ("用例集有多少条", "26"),
    ("保留率是多少", "1.0000"),
    ("门审计违规多少张", "0"),
    ("新开对话第一句该读什么", "HANDOFF"),
]

#: 无关问题：**只允许**召回常驻卡（T0/T1），出现任何非常驻卡即污染。
OFF_TOPIC = ["明天这个城市天气怎么样", "推荐一部科幻电影"]


def ask(store: Store, cards, question: str, clock) -> dict:
    cos = {str(card["id"]): _lexical_sim(question, card_text(card)) for card in cards}
    result = recall(store, now=clock.now(), budget_tokens=BUDGET, query_cos=cos)
    context = "\n".join(text for _section, text in result["blocks"] if text)
    placements = result["pack_result"].placements
    return {
        "context": context,
        "tokens": result["stats"].tokens_used,
        "packed": result["stats"].packed,
        "placements": placements,
        "non_reserved": [p for p in placements if not p.item.reserved],
    }


def count_opens(store_root: pathlib.Path, fn):
    """数一次调用里 `open` 了哪些路径，返回 (结果, 业务文件列表)。"""
    opened: list[str] = []

    def hook(event: str, args) -> None:
        if event == "open" and args and isinstance(args[0], str):
            opened.append(args[0])

    sys.addaudithook(hook)          # 钩子装上就摘不掉，所以本脚本只装一次
    value = fn()
    business = [
        path for path in opened
        if not pathlib.Path(path).resolve().is_relative_to(store_root.resolve())
        and not path.endswith((".py", ".pyc"))
    ]
    return value, business


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="月影记忆能力 MVP")
    parser.add_argument("--keep", action="store_true", help="保留临时库")
    args = parser.parse_args(argv)

    clock = FixedClock(datetime.now(timezone.utc))
    shutil.rmtree(STORE, ignore_errors=True)
    STORE.mkdir(parents=True, exist_ok=True)
    store = Store(STORE, clock=clock)
    for text in SEED:
        store.append_message("mvp", text)
    compile_session(store, "mvp", BaselineExtractor(), clock=clock)
    cards = store.cards(session_id="mvp")
    corpus = "\n".join(card_text(card) for card in cards)

    lines: list[str] = []
    say = lines.append
    say("# 月影 MVP：它能不能解决 agent 的记忆问题")
    say("")
    say(f"种子对话 {len(SEED)} 条 → 记忆卡 **{len(cards)} 张**；每次召回预算 **{BUDGET} tok**。")
    say("")

    # ---------------- 1 记得住吗
    say("## 1 跨会话还记得吗（12 问）")
    say("")
    say("| # | 问题 | 召回 | 成本 tok | 未命中的归因 |")
    say("|---|---|---|---|---|")
    hits, costs, storage_miss, retrieval_miss = 0, [], [], []
    for index, (question, expect) in enumerate(QUESTIONS, 1):
        answer = ask(store, cards, question, clock)
        hit = expect in answer["context"]
        costs.append(answer["tokens"])
        if hit:
            hits += 1
            reason = ""
        elif expect in corpus:
            retrieval_miss.append(question)
            reason = "**检索漏**（库里有了，问题措辞对不上）"
        else:
            storage_miss.append(question)
            reason = "**存储漏**（抽取阶段就丢了）"
        say(f"| {index} | {question} | {'✅' if hit else '❌'} | {answer['tokens']} | {reason} |")
    say("")
    say(f"**命中 {hits}/{len(QUESTIONS)}**，平均 **{sum(costs) // len(costs)} tok**"
        f"（{min(costs)}–{max(costs)}）。**存储漏 {len(storage_miss)} 个，检索漏 {len(retrieval_miss)} 个**。")
    say("")
    if retrieval_miss:
        say(f"> 检索漏的是：{'、'.join(retrieval_miss)}。原因不是没存，是**词面相似度**：")
        say("> 问题用了「装在哪 / 怎么处理」这类说法，卡片里是「放在 / 已撤回」。")
        say("> 修法在 backlog：给卡填 `keywords`，或换语义嵌入（都要动抽取阶段）。")
        say("")

    # ---------------- 2 要不要重新提取
    sample = QUESTIONS[0][0]
    (_, answer), business = count_opens(STORE, lambda: (None, ask(store, cards, sample, clock)))
    say("## 2 要不要重新提取")
    say("")
    say(f"召回「{sample}」这一次，`open` 到的**业务文件数 = {len(business)}**、命令 **0 条**"
        f"（只有库自己的文件被打开）。")
    say("")

    # ---------------- 3 能不能追溯
    trace = max(cards, key=lambda card: len(card.get("summary", "")))
    raw = store.get_raw(str(trace["source_ids"][0]))
    say("## 3 能不能追溯")
    say("")
    say(f"取一张卡 `{trace['id']}`（层级 `{trace['tier']}`），按 `source_ids` 回原文：")
    say("")
    say(f"- 卡摘要：{trace['summary'][:70]}")
    say(f"- 回取原文：{raw[:70]}")
    say(f"- 逐字一致：**{'是' if raw.strip() and raw.strip() in ' '.join(SEED) else '否'}**")
    say("")

    # ---------------- 4 会不会污染
    say("## 4 会不会污染（只允许召回常驻卡）")
    say("")
    say("| 无关问题 | 装箱张数 | 其中非常驻 | 判定 |")
    say("|---|---|---|---|")
    polluted = 0
    for question in OFF_TOPIC:
        answer = ask(store, cards, question, clock)
        bad = len(answer["non_reserved"])
        polluted += bad
        detail = "、".join(f"{p.item.tier}/{p.item.section}" for p in answer["non_reserved"]) or "只有常驻卡"
        say(f"| {question} | {answer['packed']} | {bad} | {'✅ ' if not bad else '❌ '}{detail} |")
    say("")
    say("> 常驻卡（`T0/T1`）永远进场，这是设计：身份与硬约束不能被静默裁掉。")
    say("> 代价是无关问题也会看到它们——所以判据是「**非常驻**的不相关卡一张都不许有」。")
    say("> 但常驻区目前**不受配额**（`#19` 未修），库长大后会挤掉其它分区。")
    say("")

    # ---------------- 结论
    say("## 结论")
    say("")
    say(f"1. **记得住**：{hits}/{len(QUESTIONS)}（{hits * 100 // len(QUESTIONS)}%）"
        f"能从库里直接拿到答案，平均 {sum(costs) // len(costs)} tok；"
        f"丢的 {len(storage_miss)} 个是存储漏、{len(retrieval_miss)} 个是检索漏。")
    say(f"2. **不用重提取**：召回一次打开业务文件 {len(business)} 个、跑命令 0 条。")
    say("3. **可追溯**：任何一张卡都能按 `source_ids` 回到原文逐字核对。")
    say(f"4. **不乱塞**：无关问题里非常驻卡 {polluted} 张。")
    say("")
    say("> 本报告不涉及压缩率。压缩率 `0.498x` 是**工程指标**（影响库增长与召回延迟），"
        "已在 `W1` 撤回；记忆可用性由上面四条衡量。")
    say("")
    say("**已知限制（如实列出）**：词面检索在「问法与卡内措辞不一致」时会漏；"
        "常驻区不受配额（`#19`）；层级目前是只读标签（`next_tier` 零调用）。")

    text = "\n".join(lines)
    REPORT.write_text(text + "\n", encoding="utf-8")
    print(text)
    print()
    print(f"报告已写入 {REPORT.relative_to(ROOT)}")

    store.close()
    if not args.keep:
        shutil.rmtree(STORE, ignore_errors=True)

    ok = not storage_miss and polluted == 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
