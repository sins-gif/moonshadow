"""给 DSH 插件用的 JSON CLI：把月影的能力暴露成「吃 JSON、吐 JSON」的三条命令。

    python tools/moon_cli.py remember '{"session":"s1","messages":[{"text":"预算 12 万，截止 2026-10-15"}]}'
    python tools/moon_cli.py recall   '{"query":"预算多少","budget_tokens":2000}'
    python tools/moon_cli.py stats    '{"session":"s1"}'

为什么要有这一层（而不是让 Node 插件直接调 Python API）：
- 插件是 JS，月影是 Python，中间必须有个边界；边界放在「JSON 进、JSON 出」最稳，
  也最容易被别的客户端（MCP、脚本、CI）复用。
- 真实逻辑一行都不复制：`remember` 调 `compile_session`，`recall` 调 `recall`，
  `stats` 只读库。这样插件永远跟库同源，不会出现「插件里另有一套实现」。

退出码：0 成功；1 参数/运行错误（错误信息也走 JSON，方便插件直接回给模型）。
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow import FixedClock, Store  # noqa: E402
from moonshadow.clock import parse_ts, to_utc  # noqa: E402
from moonshadow.compress import BaselineExtractor, compile_session  # noqa: E402
from moonshadow.pack import estimate_tokens  # noqa: E402
from moonshadow.recall import recall  # noqa: E402
from moonshadow.tiers import TIER_ORDER  # noqa: E402
from moonshadow.verify import card_text  # noqa: E402


def _root(payload: dict) -> pathlib.Path:
    """库的位置：默认 ~/.dsh/moonshadow（跟 DSH 的数据放一起，别塞进仓库）。"""
    raw = payload.get("root") or os.environ.get("MOONSHADOW_ROOT")
    if raw:
        path = pathlib.Path(raw).expanduser()
    else:
        home = os.environ.get("DSH_HOME") or (pathlib.Path.home() / ".dsh")
        path = pathlib.Path(home) / "moonshadow"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _now(payload: dict) -> datetime:
    stamp = payload.get("now")
    if stamp:
        return to_utc(parse_ts(stamp)) if isinstance(stamp, str) else stamp
    return datetime.now(timezone.utc)


def _bigrams(text: str) -> set[str]:
    flat = "".join(ch for ch in text if not ch.isspace())
    return {flat[i : i + 2] for i in range(len(flat) - 1)} or {flat}


def _lexical_sim(query: str, text: str) -> float:
    """词面相似度，映到余弦的取值范围 [-1, 1]。

    用的是「**查询二元组覆盖率**」= |A ∩ B| / |A|，不是 Jaccard：
    实测踩过——Jaccard 的分母含文档长度，卡片一长相似度就被摊薄到 0 附近，
    于是所有卡都被定量硬门挡掉（gated=0、context 为空）。而检索要回答的是
    「查询里的词在这张卡里出现了多少」，分母就该是查询。

    为什么不做真嵌入：插件不该偷偷联网、CLI 也不该引入模型。这里只保证
    「完全不相关的卡不被排上来」；**阈值 θ 是在语义余弦上标定的，词面模式下的
    通过/拦截不能当成质量结论**——这一点写进返回值的 note 里。
    """
    a, b = _bigrams(query), _bigrams(text)
    if not a or not b:
        return -1.0
    coverage = len(a & b) / len(a)
    # 平方根：抵消「覆盖率对小重叠的惩罚」。实测过两种口径——
    # 直接线性喂进去时，一张正确命中「预算」的卡 σ 只有 0.33，被 θ=0.3941 挡掉（gated=1）；
    # 而完全不相关的卡仍是 σ=0（照旧被挡）。所以映射只抬高小重叠，不改变「无关=0」。
    return 2.0 * (coverage ** 0.5) - 1.0


def cmd_remember(payload: dict) -> dict:
    session = str(payload.get("session") or "default")
    messages = payload.get("messages") or []
    if not messages:
        raise ValueError("remember 需要 messages（至少一条 {text}）")
    clock = FixedClock(_now(payload))
    store = Store(_root(payload), clock=clock)
    try:
        for item in messages:
            text = item.get("text") if isinstance(item, dict) else str(item)
            if not text:
                continue
            store.append_message(session, str(text))
        report = compile_session(store, session, BaselineExtractor(), clock=clock)
        cards = store.cards(session_id=session)
        raw_tokens = sum(
            estimate_tokens(store.get_raw(str(row["id"]))) for row in store.messages_for_session(session)
        )
        card_tokens = sum(estimate_tokens(card_text(card)) for card in cards) or 1
        per_tier = {tier: 0 for tier in TIER_ORDER}
        for card in cards:
            per_tier[str(card["tier"])] = per_tier.get(str(card["tier"]), 0) + 1
        return {
            "ok": True,
            "session": session,
            "appended": len(messages),
            "inserted": report.inserted,
            "replayed": report.replayed,
            "cards": len(cards),
            "raw_tokens": raw_tokens,
            "card_tokens": card_tokens,
            "compression": raw_tokens / card_tokens,
            "cards_per_tier": per_tier,
        }
    finally:
        store.close()


def cmd_recall(payload: dict) -> dict:
    query = str(payload.get("query") or "")
    budget = int(payload.get("budget_tokens") or 2000)
    session = payload.get("session")
    clock = FixedClock(_now(payload))
    store = Store(_root(payload), clock=clock)
    try:
        cards = store.cards(session_id=str(session) if session else None)
        cos = {str(card["id"]): _lexical_sim(query, card_text(card)) for card in cards} if query else None
        result = recall(
            store,
            now=clock.now(),
            budget_tokens=budget,
            query_cos=cos,
            session_id=str(session) if session else None,
            explicit_recall=bool(payload.get("explicit_recall")),
        )
        blocks = result["blocks"]
        context = "\n\n".join(text for _section, text in blocks if text)
        return {
            "ok": True,
            "query": query,
            "budget_tokens": budget,
            "used_tokens": result["stats"].tokens_used,
            "remaining_budget": result["remaining_budget"],
            "gated": result["stats"].gated,
            "packed": result["stats"].packed,
            "degraded": result["stats"].degraded,
            "dropped": result["stats"].dropped,
            "context": context,
            "note": (
                "相似度用的是「查询二元组覆盖率」（词面），不是语义嵌入；阈值 θ 是在语义余弦上标定的，"
                "所以词面模式下的通过/拦截只用于验证管道，不能当成质量结论。换真嵌入时只需替换 query_cos 的来源。"
            ),
        }
    finally:
        store.close()


def cmd_stats(payload: dict) -> dict:
    session = payload.get("session")
    clock = FixedClock(_now(payload))
    store = Store(_root(payload), clock=clock)
    try:
        cards = store.cards(session_id=str(session) if session else None)
        per_tier = {tier: 0 for tier in TIER_ORDER}
        tokens = 0
        for card in cards:
            per_tier[str(card["tier"])] = per_tier.get(str(card["tier"]), 0) + 1
            tokens += estimate_tokens(card_text(card))
        return {
            "ok": True,
            "root": str(_root(payload)),
            "session": session,
            "cards": len(cards),
            "card_tokens": tokens,
            "cards_per_tier": per_tier,
        }
    finally:
        store.close()


COMMANDS = {"remember": cmd_remember, "recall": cmd_recall, "stats": cmd_stats}


def main(argv: list[str] | None = None) -> int:
    # 输出固定走 UTF-8：本机管道默认是 cp936，`ensure_ascii=False` 的中文/「——」会在
    # 编码阶段抛 UnicodeEncodeError，而 Node 侧是按 UTF-8 解码的——两边必须同一编码。
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in COMMANDS:
        print(json.dumps({"ok": False, "error": f"用法：moon_cli.py {{{'|'.join(COMMANDS)}}} <json>"},
                         ensure_ascii=False))
        return 1
    command = args[0]
    # 参数来源有两路：argv 或 stdin。**argv 缺省时必须读 stdin**——上一版在这里默认成了
    # "{}"，于是管道喂进来的 JSON 被静默忽略，remember/recall 都拿到空参数（实测踩到）。
    #
    # stdin 必须按 **UTF-8 字节**读，不能 `sys.stdin.read()`：后者用系统编码（本机 cp936），
    # 中文 JSON 会被解成孤立代理字符，随后 print 时抛
    # `UnicodeEncodeError: '\udca2' ... surrogates not allowed`——即「插件写中文必崩」。
    if len(args) > 1:
        raw = args[1]
    else:
        buffer = getattr(sys.stdin, "buffer", None)
        if buffer is not None:
            raw = buffer.read().decode("utf-8")
        else:  # 已被替换成纯文本流（测试替身）时退回文本读
            raw = sys.stdin.read()
    raw = raw.strip()
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        print(json.dumps({"ok": False, "error": f"参数不是合法 JSON：{exc}"}, ensure_ascii=False))
        return 1
    try:
        print(json.dumps(COMMANDS[command](payload), ensure_ascii=False))
        return 0
    except Exception as exc:  # 错误也走 JSON：插件直接把 message 回给模型
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
