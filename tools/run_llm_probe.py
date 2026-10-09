"""Phase 2 最小实验探针：把金标准集里 3 条代表作直接喂给 LLM，看**形状**。

    python tools/run_llm_probe.py            # 需要本地 key（环境变量）
    python tools/run_llm_probe.py --mock     # 不联网：用内联假响应验证管道

设计原则（本轮只要形状，不要接口）：

- **不经任何后处理**：LLM 返回的 JSON 原样 dump 到 `.tmp/llm_probe/<case>.json`，
  再逐条重放成卡；这里**不**调用 `make_card`、**不**接入管线。
- **key 只从环境变量读**，绝不写进任何文件、绝不打印其值。
  支持 `DEEPSEEK_API_KEY` / `DASHSCOPE_API_KEY` / `MOONSHADOW_LLM_KEY`。
- 看三件事，全部**只读**分析，不改写 LLM 的输出：
  1. 同一句话是否被分到多个字段（C6 是否天然被遵守）；
  2. `summary` 是否自然覆盖要点（C8 是否天然被遵守）；
  3. 逐句字段是**复述整句**还是**抽象**——这正是参照线里那 `1.2x` 的来源。

退出码：`0` 成功；`2` 缺 key（打印需要的环境变量名与命令，不尝试联网）。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow.compress import PROMPT_TEMPLATE, classify_tier  # noqa: E402
from moonshadow.eval import load_gold  # noqa: E402
from moonshadow.verify import (  # noqa: E402
    SUMMARY_MENTION_CHARS,
    _shared_fragment,
    normalize,
)

OUT_DIR = ROOT / ".tmp" / "llm_probe"

#: 契约补丁：把 v1.3 的卡契约随提示词一起给模型，否则测的不是「LLM 天然怎么做」，
#: 而是「它在不知道规则时怎么做」。两条都要看，所以本轮把契约写进提示词。
CONTRACT_NOTE = """
额外硬约束（v1.3 卡契约，必须遵守）：
- 一张卡内，同一句话只能出现在 facts/decisions/todos/constraints 中的一个字段里（C6）。
- summary 是对整张卡的概述，不得逐字复述上述字段里的任何句子（C6）；
  并且必须覆盖 decisions 的全部要点、提及 todos/constraints 的每一项（C8）。
- 不要输出 raw_quote（C5）。T0/T1 的硬字段放进 facts 或 decisions。
只输出 JSON 数组，不要解释文字。
"""

#: 三个 provider 的连接参数；模型名可用 `MOONSHADOW_LLM_MODEL` 覆盖。
PROVIDERS = {
    "DEEPSEEK_API_KEY": ("https://api.deepseek.com/chat/completions", "deepseek-chat"),
    "DASHSCOPE_API_KEY": (
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "qwen-plus",
    ),
    "MOONSHADOW_LLM_KEY": ("http://127.0.0.1:8000/v1/chat/completions", "local-model"),
}


def pick_cases() -> list:
    """挑 3 条代表作：T0 密集 / T2 决策密集 / T5 混杂（按层级计数排序，规则确定）。"""
    cases = load_gold()
    scored = []
    for case in cases:
        tiers = [classify_tier(text) for text in case.messages]
        scored.append((case, tiers.count("T0"), tiers.count("T2"), len(set(tiers)), len(tiers)))
    by_t0 = max(scored, key=lambda item: (item[1], item[4]))
    by_t2 = max(scored, key=lambda item: (item[2], item[4]))
    by_mix = max(scored, key=lambda item: (item[3], item[4]))
    picked, seen = [], set()
    for entry in (by_t0, by_t2, by_mix):
        if entry[0].id not in seen:
            seen.add(entry[0].id)
            picked.append(entry[0])
    for case, *_ in scored:  # 不足 3 条时补齐
        if len(picked) >= 3:
            break
        if case.id not in seen:
            seen.add(case.id)
            picked.append(case)
    return picked[:3]


def build_prompt(case) -> str:
    lines = [f"- id=m{i} ts={case.day}T09:00:00+00:00: {text}"
             for i, text in enumerate(case.messages, 1)]
    return PROMPT_TEMPLATE.format(messages="\n".join(lines)) + CONTRACT_NOTE


def call_llm(prompt: str, *, dry_run: bool) -> tuple[str, str]:
    """返回 `(raw_text, how)`；缺 key 时抛 SystemExit(2) 并给出该设哪个变量。"""
    if dry_run:
        return MOCK_RESPONSE, "mock（非 LLM 输出，仅验证管道）"
    for env_name, (url, default_model) in PROVIDERS.items():
        key = os.environ.get(env_name)
        if not key:
            continue
        model = os.environ.get("MOONSHADOW_LLM_MODEL", default_model)
        payload = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",  # 只在内存里；不落盘、不打印
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"], f"{env_name} / {model}"
    raise SystemExit(
        "缺 key：请先 `setx DEEPSEEK_API_KEY <你的 key>`（或 DASHSCOPE_API_KEY）"
        " 并重开终端，再运行本命令。key 不要贴进对话。\n"
        "离线验证管道可用：python tools/run_llm_probe.py --mock"
    )


#: `--mock` 用的内联假响应：**故意**同时包含一处 C6 违规与一处 C8 违规，
#: 这样管道（dump → 三条分析）跑起来就能看见「分析确实会报问题」。
MOCK_RESPONSE = json.dumps(
    [
        {
            "tier": "T3",
            "importance": 6,
            "network": "experience",
            "summary": "T3",  # 故意违反 C8
            "facts": ["决定先做读缓存，写路径这一轮完全不动。"],
            "decisions": ["决定先做读缓存，写路径这一轮完全不动。"],  # 故意违反 C6
            "source_ids": ["m1"],
        }
    ],
    ensure_ascii=False,
    indent=2,
)


def parse_cards(raw: str) -> list[dict]:
    """从原始返回里抠出 JSON 数组；抠不出来就原样报错（不猜、不修补）。"""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text.split("\n", 1)[1] if "\n" in text else text
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < 0:
        raise ValueError("返回里找不到 JSON 数组（原始内容已 dump，未做任何修补）")
    return list(json.loads(text[start : end + 1]))


def analyse(cards: list[dict], messages: list[str]) -> dict[str, object]:
    """三条只读分析：C6 / C8 / 复述 vs 抽象。"""
    duplicated, uncovered, verbatim, entries = [], [], 0, 0
    joined = normalize("　".join(messages))
    for index, card in enumerate(cards, 1):
        seen: dict[str, str] = {}
        for field in ("facts", "decisions", "todos", "constraints"):
            for value in card.get(field) or ():
                key = normalize(str(value))
                entries += 1
                if key in seen:
                    duplicated.append(f"card{index}: {seen[key]} 与 {field} 重复 → {key[:30]}")
                else:
                    seen[key] = field
                # 复述判定：该条目在原文里逐字存在（≥12 字算整句复述）
                if len(key) >= 12 and key in joined:
                    verbatim += 1
        summary = normalize(str(card.get("summary") or ""))
        for field in ("decisions", "todos", "constraints"):
            for value in card.get(field) or ():
                key = normalize(str(value))
                if key and not _shared_fragment(summary, key, SUMMARY_MENTION_CHARS):
                    uncovered.append(f"card{index}: summary 未提及 {field} 的要点 → {key[:30]}")
    return {
        "cards": len(cards),
        "field_entries": entries,
        "C6_duplicates": duplicated,
        "C8_uncovered": uncovered,
        "verbatim_entries": verbatim,
        "abstracted_entries": entries - verbatim,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 最小实验：直接看 LLM 的卡形状")
    parser.add_argument("--mock", action="store_true", help="不联网，用内联假响应验证管道")
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for case in pick_cases():
        prompt = build_prompt(case)
        raw, how = call_llm(prompt, dry_run=args.mock)
        dump = OUT_DIR / f"{case.id}.raw.txt"
        dump.write_bytes(raw.encode("utf-8"))
        print(f"=== {case.id}（{len(case.messages)} 条消息，{how}）")
        print(f"    原始 dump: {dump.relative_to(ROOT)}")
        try:
            cards = parse_cards(raw)
        except Exception as exc:  # 原始内容已落盘，这里只报告
            print(f"    解析失败：{exc}")
            continue
        result = analyse(cards, case.messages)
        print(f"    卡数={result['cards']} 字段条目={result['field_entries']}")
        print(f"    C6 重复={len(result['C6_duplicates'])}  C8 未覆盖={len(result['C8_uncovered'])}")
        print(
            f"    逐句字段：复述整句 {result['verbatim_entries']} / "
            f"抽象 {result['abstracted_entries']}"
        )
        for line in (result["C6_duplicates"] + result["C8_uncovered"])[:4]:
            print(f"      · {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
