"""Phase 2 LLM 实验：**形状**（3 条代表作）与**全链路**（25 条 → §4.1 六条判据）。

    python tools/run_llm_probe.py                 # 形状：3 条代表作（一条用例一个请求）
    python tools/run_llm_probe.py --all           # 形状：全部 25 条
    python tools/run_llm_probe.py --pipeline      # 全链路：25 条走完整管线，读 §4.1 六条判据
    python tools/run_llm_probe.py --pipeline --replay   # 断点续跑：有 dump 就用 dump，缺的才联网
    python tools/run_llm_probe.py --pipeline --only 09-...  # 只跑一条（冒烟）
    python tools/run_llm_probe.py --mock          # 不联网：内联假响应验证管道

两种模式回答的是不同的问题，**不能互相替代**：

- **形状模式**：一条用例一次请求，**不经主管线**，只看 LLM 输出的卡长什么样
  （卡数、C6/C8、逐句字段是复述整句还是抽象）。它测的是「模型天然怎么做」。
- **全链路模式**：按管线的批边界（`≤10` 条 / `≤4096` tok）逐批请求，产物经 `make_card`
  落库，再用 `verify_*` 判据与 `run_case` 读数——**这才是 §4.1 六条判据的分母**。
  它测的是「把它接进协议之后会怎样」，其中包括批边界、`source_ids` 标签映射、
  快路径抢走的纯问候等**接口事实**。

设计原则：

- **key 只从环境变量读**，绝不写进任何文件、绝不打印其值。
  支持 `DEEPSEEK_API_KEY` / `DASHSCOPE_API_KEY` / `MOONSHADOW_LLM_KEY`。
- **判据同源**：C6/C8 不在这里另写一套，直接调用 `verify.verify_*`（与契约同一个函数）。
  上一轮 `0.66x` 作废的原因就是「同一件事在两处各算一遍」——探针里那条旧的片段级
  C8 判据已被删掉，改为调用契约判据。
- **原始返回一律 dump**（`.tmp/llm_probe/`）：读数可复算、可 `--replay`，不用再花钱。

退出码：形状模式 `2` = 缺 key；全链路模式 `0` = 实验跑完（**判据可以不过**——判据的结果是
实验结论，不是失败），`1` = 有用例因异常/解析失败没跑完（那才是工具的问题）。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow.compress import PROMPT_TEMPLATE, classify_tier  # noqa: E402
from moonshadow.eval import (  # noqa: E402
    WORK_DIR,
    EvalReport,
    GoldCase,
    load_gold,
    run_case,
)
from moonshadow.extractor import (  # noqa: E402
    BatchDiagnostics,
    Message,
    parse_llm_cards,
)
from moonshadow.pack import estimate_tokens  # noqa: E402
from moonshadow.store import Store  # noqa: E402
from moonshadow.verify import (  # noqa: E402
    CARD_TEXT_FIELDS,
    MUTUALLY_EXCLUSIVE_FIELDS,
    card_text_many,
    extract_key_fields,
    normalize,
    verify_field_mutual_exclusion,
    verify_summary_substance,
)

OUT_DIR = ROOT / ".tmp" / "llm_probe"

#: 契约补丁：把 v1.2.x Roadmap 的卡契约随提示词一起给模型，否则测的不是「LLM 天然怎么做」，
#: 而是「它在不知道规则时怎么做」。两条都要看，所以本轮把契约写进提示词。
CONTRACT_NOTE = """
额外硬约束（v1.2.x Roadmap 卡契约，必须遵守）：
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

#: 单次请求的输出上限。管线一批最多 `10` 条消息 / `4096` 原文 token，
#: 但模型要产出「一批卡」，默认上限会在长批次上截断成非法 JSON——截断的那批会整批丢失。
MAX_OUTPUT_TOKENS = 8000
TIMEOUT_SECONDS = 300
ATTEMPTS = 3
RETRY_SLEEP = 3.0


def pick_cases() -> list[GoldCase]:
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


def build_batch_prompt(messages: Sequence[Mapping[str, Any]]) -> str:
    """一批消息 → 提示词。

    `ts` 用消息自己的 `created_at`（规则版不用它；模型要靠它判「谁更晚」）。
    标签 `m1…mN` 是**提示词局部**的，回链时必须映射回 store id
    （`parse_llm_cards(label_to_id=…)`），否则 C1 会把每条消息都判成无载体。
    """
    lines = [
        f"- id=m{index} ts={message.get('created_at') or message.get('ts') or ''}: "
        f"{message.get('text', '')}"
        for index, message in enumerate(messages, 1)
    ]
    return PROMPT_TEMPLATE.format(messages="\n".join(lines)) + CONTRACT_NOTE


def build_prompt(case: GoldCase) -> str:
    """形状模式的提示词：**整条用例一个请求**（不分批）。"""
    return build_batch_prompt(
        [
            {"id": f"m{index}", "created_at": f"{case.day}T09:00:00+00:00", "text": text}
            for index, text in enumerate(case.messages, 1)
        ]
    )


def call_llm(prompt: str, *, dry_run: bool) -> tuple[str, str, dict[str, Any]]:
    """返回 `(raw_text, how, usage)`；缺 key 时抛 SystemExit(2) 并给出该设哪个变量。"""
    if dry_run:
        return MOCK_RESPONSE, "mock（非 LLM 输出，仅验证管道）", {}
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
                "max_tokens": MAX_OUTPUT_TOKENS,
            }
        ).encode("utf-8")
        last: Exception | None = None
        for attempt in range(1, ATTEMPTS + 1):
            request = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {key}",  # 只在内存里；不落盘、不打印
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                    body = json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = exc
                if attempt < ATTEMPTS:
                    time.sleep(RETRY_SLEEP * attempt)  # 网络抖动/限流：退避重试
                continue
            usage = body.get("usage") or {}
            return body["choices"][0]["message"]["content"], f"{env_name} / {model}", usage
        # 报错文本里只有 URL 与异常类型，**不含 key**（Authorization 头不进异常消息）
        raise RuntimeError(
            f"{env_name} 调用失败（尝试 {ATTEMPTS} 次）：{type(last).__name__}: {last}"
        )
    raise SystemExit(
        "缺 key：请先在当前终端设置 `$env:DEEPSEEK_API_KEY`（或 DASHSCOPE_API_KEY），"
        "再运行本命令。key 不要写进文件、不要提交。\n"
        "离线验证管道可用：python tools/run_llm_probe.py --mock"
    )


#: `--mock` 用的内联假响应：**故意**同时包含一处 C6 违规与一处 C8 违规，
#: 这样管道（dump → 判据）跑起来就能看见「判据确实会报问题」。
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


def count_verbatim(
    drafts: Sequence[Mapping[str, Any]], texts: Sequence[str]
) -> tuple[int, int]:
    """逐句字段里「逐字复述整句」的条数与总条数（≥12 字算整句复述）。

    **判据只有这一处**：形状模式与全链路模式都调它——上一轮 `0.66x` 作废的教训就是
    「同一件事在两处各算一遍」。返回 `(总条数, 复述条数)`。
    """
    joined = normalize("　".join(texts))
    entries = 0
    verbatim = 0
    for draft in drafts:
        for field_name in MUTUALLY_EXCLUSIVE_FIELDS:
            for value in draft.get(field_name) or ():
                key = normalize(str(value))
                entries += 1
                if len(key) >= 12 and key in joined:
                    verbatim += 1
    return entries, verbatim


def analyse_shape(raw: str, messages: Sequence[str]) -> dict[str, Any]:
    """形状模式的只读分析：C6 / C8 / 「复述整句 vs 抽象」。

    C6 与 C8 直接调用**契约自己的判据**（`verify.verify_*`）：探针里不再另写一套。
    最后一条是探针特有的读数——逐句字段承载的是整句原文（`1.2x` 的来源）还是抽象。
    """
    drafts = [card.to_draft() for card in parse_llm_cards(raw)]
    mutual = verify_field_mutual_exclusion(drafts)
    substance = verify_summary_substance(drafts)
    c6 = list(mutual.missing.get("field_mutual_exclusion", []))
    c8 = list(substance.missing.get("summary_substance", []))
    entries, verbatim = count_verbatim(drafts, messages)
    return {
        "cards": len(drafts),
        "field_entries": entries,
        "C6": c6,
        "C8": c8,
        "verbatim_entries": verbatim,
        "abstracted_entries": entries - verbatim,
    }


class LLMExtractor:
    """把 LLM 适配成管线认识的 `Extractor`。**一次调用 = 一个请求 = 一批。**

    三个接口事实由它自己承担（契约没规定，见 `docs/v1.2-future-llm-extractor.md` §6）：

    1. **管线不把 prompt 递给抽取器**（`compile_session` 自己也算了一份又扔掉），
       所以提示词在这里按同一模板重建——两处必须是同一份模板，否则测的不是管线。
    2. **`source_ids` 是提示词局部标签**（`m1…mN`）：必须映射回 store id。
    3. **批边界由管线决定**（`≤10` 条 / `≤4096` tok）：抽取器无权重划，只能接受。

    另外它把每批的原始返回按 `「用例.批号」.raw.txt` dump 下来，`--replay` 可原样重放。
    """

    def __init__(self, *, dry_run: bool = False, replay: bool = False, name: str = "llm") -> None:
        self.dry_run = dry_run
        self.replay = replay
        self.name = name  # 不在 `BASELINE_IDENTIFIERS` 里 → 不享受 C6.1 豁免
        self.case_id = "?"
        self.batch = 0
        self.calls = 0
        self.replayed = 0
        self.parse_failures: list[str] = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.diagnostics: list[BatchDiagnostics] = []
        #: 全部产出的草稿（判据只读它，不做任何改写）：字段级 breakdown 的归集口径。
        self.drafts: list[dict[str, Any]] = []
        #: 「复述 vs 抽象」的累计读数（判据见 `count_verbatim`，与形状模式同一处）。
        self.entries = 0
        self.verbatim = 0
        self.duplicates = 0

    def begin_case(self, case_id: str) -> None:
        self.case_id = case_id
        self.batch = 0

    def dump_path(self) -> pathlib.Path:
        """dump 路径。**mock 的 dump 带 `mock.` 前缀**：假响应绝不能被当成真实读数。"""
        prefix = "mock." if self.dry_run else ""
        return OUT_DIR / "pipeline" / f"{prefix}{self.case_id}.{self.batch:02d}.raw.txt"

    def __call__(self, messages: Sequence[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
        batch = list(messages)
        self.batch += 1
        dump = self.dump_path()
        if self.replay and dump.exists():
            raw = dump.read_text(encoding="utf-8")
            self.replayed += 1
        else:
            # `--replay` 是**断点续跑**而不是「纯离线」：缺的那一批才联网补。
            # 第一轮实测的用处：解析层修好后，41 批从 dump 复用、只有 14 条用例
            # 因「第一批就抛异常」而没请求过的批次需要补——复盘不重复花钱。
            raw, _how, usage = call_llm(build_batch_prompt(batch), dry_run=self.dry_run)
            dump.parent.mkdir(parents=True, exist_ok=True)
            dump.write_bytes(raw.encode("utf-8"))
            self.calls += 1
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)

        label_to_id = {f"m{index}": str(message["id"]) for index, message in enumerate(batch, 1)}
        container = BatchDiagnostics()
        try:
            cards = parse_llm_cards(raw, container, label_to_id=label_to_id)
        except Exception as exc:
            # 不猜、不修补：这一批的输出原样躺在 dump 里，损失由 C1/保留率如实暴露。
            self.parse_failures.append(
                f"{self.case_id} 第 {self.batch} 批：{type(exc).__name__}: {exc}"
            )
            self.diagnostics.append(container)
            return
        self.diagnostics.append(container)
        texts = [str(message.get("text", "")) for message in batch]
        for card in cards:
            draft = card.to_draft()
            self.drafts.append(draft)
            entries, verbatim = count_verbatim([draft], texts)
            self.entries += entries
            self.verbatim += verbatim
            yield draft


def hard_point_breakdown(cards: Sequence[Mapping[str, Any]]) -> dict[str, tuple[int, int]]:
    """§4.1 第 6 条的字段级 breakdown：`字段 → (硬要点总数, 未覆盖数)`。

    两个数都来自**同一批判据对象**（`run_case` 落库的卡）与**同一个判据函数**
    （`verify_summary_substance`）：分母数要点（`verify.extract_key_fields`，即
    `HARD_PATTERNS`），分子读判据自己打印的违规行。不在这里另写一套覆盖判定——
    上一轮的 `0.66x` 就是因为同一件事在两处各算了一遍而作废的。

    违规行必须能归类：格式一变就抛错，而不是静默给 0（静默 0 会让判据变成摆设）。
    """
    totals: dict[str, int] = {name: 0 for name in ("decisions", "todos", "constraints")}
    for card in cards:
        for field_name in totals:
            for value in card.get(field_name) or ():
                totals[field_name] += sum(
                    len(items) for items in extract_key_fields(str(value)).values()
                )
    missed: dict[str, int] = {name: 0 for name in totals}
    report = verify_summary_substance(cards)
    for line in report.missing.get("summary_substance", []):
        if "里的硬要点" in line:
            field_name = line.split("summary 未覆盖 ", 1)[1].split(" 里的硬要点", 1)[0]
            if field_name not in missed:
                raise AssertionError(f"C8 违规行里的字段名不认识：{line}")
            missed[field_name] += 1
        elif "summary 过短" in line or "summary 过长且未覆盖全部要点" in line:
            continue  # 长度条款：不是硬要点漏点
        else:
            raise AssertionError(f"C8 违规行格式未归类（判据改过？）：{line}")
    return {name: (totals[name], missed[name]) for name in totals}


def read_stored_cards(case_id: str) -> list[dict[str, Any]]:
    """重开 `run_case` 留下的库，读它判过的**落库卡**。

    判据必须在**同一批对象**上归集：拿产出草稿去算会与落库判据差出「内容去重」那一段
    （第一轮实测 172 张草稿 vs 121 张落库卡，硬要点漏点 41 vs 26——两处口径不一致，
    差的那 15 条不是判据松紧，是对象不同）。`run_case` 的工作目录是 `WORK_DIR/<用例 id>`。
    """
    store = Store(WORK_DIR / case_id)
    try:
        return list(store.cards(session_id=f"gold-{case_id}"))
    finally:
        store.close()


def unique_tier_tokens(case_id: str) -> dict[str, list[int]]:
    """按**唯一归属**口径统计每个 tier 的 `[原文 token, 卡 token]`。

    §4.1.1 把这条写进了判据：**一段原文只计入它第一个引用者的层级**。
    理由不是洁癖：`run_eval.tier_table` 的口径是「原文按 source_ids 归属」，同一段原文
    被 3 张卡引用就在 3 个层级各计一次，于是分层压缩率的分母随「引用次数」膨胀——
    拿它当判据会把「多引用」错读成「原文很长」。卡 token 天然只属于一张卡，无需去重。

    实现口径：按卡的落库顺序（= 插入顺序）claim，先到先得；`source_ids` 指向不存在的消息时
    直接跳过（不静默造数，也不把这部分原文摊给别的层级）。
    """
    store = Store(WORK_DIR / case_id)
    try:
        cards = list(store.cards(session_id=f"gold-{case_id}"))
        texts = {
            str(row["id"]): store.get_raw(str(row["id"]))
            for row in store.messages_for_session(f"gold-{case_id}")
        }
    finally:
        store.close()
    claimed: set[str] = set()
    stats: dict[str, list[int]] = {}
    for card in cards:
        tier = str(card["tier"])
        bucket = stats.setdefault(tier, [0, 0])
        bucket[1] += estimate_tokens(card_text_many([card]))
        for source_id in card.get("source_ids") or ():
            key = str(source_id)
            if key in claimed or key not in texts:
                continue
            claimed.add(key)
            bucket[0] += estimate_tokens(texts[key])
    return stats


def run_pipeline(cases: Sequence[GoldCase], *, dry_run: bool, replay: bool) -> int:
    """全链路：逐条用例走 `run_case`（与 `run_eval` 内部逐字相同的调用），再读 §4.1 六条判据。"""
    engine = LLMExtractor(dry_run=dry_run, replay=replay, name="llm-deepseek-chat")
    results = []
    stored: list[dict[str, Any]] = []
    tier_unique: dict[str, list[int]] = {}
    print("LLM 全链路实验（Phase 2 §4.1）：25 条金标准走完整管线")
    print(f"  {'用例':<28}{'消息':>5}{'卡数':>5}{'原文tok':>9}{'卡tok':>8}{'压缩率':>9}"
          f"{'保留率':>8}{'C6':>4}{'C8':>4}{'C1':>4}{'期望漏':>7}")
    for case in cases:
        engine.begin_case(case.id)
        result = run_case(case, engine)
        results.append(result)
        stored.extend(read_stored_cards(case.id))
        for tier, (raw, card) in unique_tier_tokens(case.id).items():
            bucket = tier_unique.setdefault(tier, [0, 0])
            bucket[0] += raw
            bucket[1] += card
        uncarried = len(result.uncarried)
        print(
            f"  {case.id:<28}{len(case.messages):>5}{result.cards:>5}"
            f"{result.raw_tokens:>9,}{result.card_tokens:>8,}{result.compression:>8.3f}x"
            f"{result.recall:>8.4f}{len(result.field_violations):>4}"
            f"{len(result.summary_violations):>4}{uncarried:>4}"
            f"{len(result.missing_expected):>7}"
            + (f"   ⚠ {result.error}" if result.error else "")
        )
        sys.stdout.flush()

    report = EvalReport(results=results)  # 与 run_eval 同一条装配路径（含默认 min_recall）
    c1 = sum(len(result.uncarried) for result in results)
    breakdown = hard_point_breakdown(stored)
    todos_total, todos_missed = breakdown["todos"]

    print()
    print(f"  合计：卡数 {sum(result.cards for result in results)}，"
          f"API 调用 {engine.calls} 次（重放 {engine.replayed} 批），"
          f"解析失败 {len(engine.parse_failures)} 批，用例异常 {sum(1 for r in results if r.error)} 条")
    if engine.prompt_tokens or engine.completion_tokens:
        print(f"  API 侧 token：入 {engine.prompt_tokens:,} / 出 {engine.completion_tokens:,}"
              "（provider 报的用量，与本项目 estimate_tokens 口径不同，只作交叉参考）")
    for line in engine.parse_failures[:5]:
        print(f"      ⚠ 解析失败：{line}")

    print()
    print("【Phase 2 六条判据（docs/v1.2-future-llm-extractor.md §4.1，口径见 §4.1.1/§4.1.2）】")
    # 第 5 条：分层压缩（唯一归属口径）。保留组 T0/T1 允许 1.0x–1.2x，压缩组 T3–T7 要 >= 2.0x。
    def ratio(tier: str) -> float | None:
        raw, card = tier_unique.get(tier, [0, 0])
        return raw / card if card else None

    keep_fail = []
    for tier in ("T0", "T1"):
        value = ratio(tier)
        if value is None:
            keep_fail.append(f"{tier} 无卡（不可观测）")
        elif not 1.0 <= value <= 1.2:
            keep_fail.append(f"{tier}={value:.3f}x 不在 1.0–1.2")
    compress_raw = sum(tier_unique.get(t, [0, 0])[0] for t in ("T3", "T4", "T5", "T6", "T7"))
    compress_card = sum(tier_unique.get(t, [0, 0])[1] for t in ("T3", "T4", "T5", "T6", "T7"))
    compress = compress_raw / compress_card if compress_card else 0.0
    checks = [
        ("1 关键字段保留率 >= 0.98", report.key_field_recall >= 0.98,
         f"{report.key_field_recall:.4f}"),
        ("2 C6 违规 = 0（卡内字段互斥）", report.field_violations == 0,
         f"{report.field_violations} 条"),
        ("3 C8 违规 = 0（summary 承载内容）", report.summary_violations == 0,
         f"{report.summary_violations} 条"),
        ("4 定性可追溯违规 = 0", c1 == 0, f"{c1} 条"),
        ("5a 保留组 T0/T1 在 1.0–1.2x", not keep_fail,
         "；".join(keep_fail) or "两者都在区间内"),
        ("5b 压缩组 T3–T7 >= 2.0x", compress >= 2.0,
         f"{compress:.4f}x（唯一归属口径；T6/T7 本轮无卡）"),
        ("6 todos 硬要点漏点率 = 0", todos_missed == 0,
         f"{todos_missed}/{todos_total} 条"
         + (f"（{todos_missed / todos_total:.1%}）" if todos_total else "")),
    ]
    for label, passed, reading in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {label:<32}{reading}")
    if todos_missed:
        print("  ⚠ 第 6 条是否决项：硬要点没保住，第 5 条的压缩读数**不计通过**（§4.1.1）")
    print(f"       旁证（非判据）：全局 token 加权 {report.weighted_compression:.4f}x"
          f"（逐用例平均 {report.compression_ratio:.4f}x）；decisions 硬要点 "
          f"{breakdown['decisions'][1]}/{breakdown['decisions'][0]} 未覆盖、constraints "
          f"{breakdown['constraints'][1]}/{breakdown['constraints'][0]} 未覆盖")
    # 违规样例：结论要能指到具体一条，而不是只给个数
    c6_lines = [line for result in results for line in result.field_violations]
    if c6_lines:
        print("  C6 违规样例（前 3 条）：")
        for line in c6_lines[:3]:
            print(f"      · {line}")
    c8_lines = [line for result in results for line in result.summary_violations]
    if c8_lines:
        print("  C8 违规样例（前 3 条）：")
        for line in c8_lines[:3]:
            print(f"      · {line}")
    print(f"  用例通过（含金标准期望字段与层级）：{sum(1 for r in results if r.ok)}/{len(results)}")
    total_c8 = sum(len(r.summary_violations) for r in results)
    judged = sum(missed for _total, missed in breakdown.values())
    length_only = total_c8 - judged
    print(f"  C8 归集核对（同一批落库卡）：store 判 {total_c8} 条 = 硬要点漏点 {judged} "
          f"+ 长度条款 {length_only}"
          + ("　✓ 两处一致" if length_only >= 0 else "　⚠ 长度条款为负，判据与归集不同源"))
    if length_only < 0:
        print("  ⚠ C8 与 C6 的判据对象不一致（落库卡 vs 产出草稿），本次读数不可用")

    print()
    print("  LLM 输出侧的接口读数：")
    unstructured = sum(item.unstructured_claims for item in engine.diagnostics)
    overlaps = sum(item.facts_decisions_overlap_candidates for item in engine.diagnostics)
    numeric_tier = sum(item.tier_as_number for item in engine.diagnostics)
    bad_confidence = sum(item.non_numeric_confidence for item in engine.diagnostics)
    fenced = sum(item.fenced_payload for item in engine.diagnostics)
    print(f"      `claims` 是字符串、被降级进 facts 的条数：{unstructured}")
    print(f"      `facts` 与 decisions/todos/constraints 的硬字段级重叠候选：{overlaps}"
          "（机械 C6 抓不到的上界）")
    print(f"      格式滑落（解析层归一/降级并计数）：裸数字 tier {numeric_tier} 张、"
          f"非数字 confidence {bad_confidence} 张、markdown 围栏 {fenced} 批")
    print(f"      产出草稿 {len(engine.drafts)} 张 / 落库卡 "
          f"{sum(result.cards for result in results)} 张（差值=内容去重）")
    abstracted = engine.entries - engine.verbatim
    print(f"      逐句字段 {engine.entries} 条：复述整句 {engine.verbatim} / 抽象 {abstracted}"
          + (f"（复述占 {engine.verbatim / engine.entries:.1%}）" if engine.entries else ""))

    print()
    print("  分层读数（与 run_eval 同一张表；**原文 token 按 source_ids 归属，会被多张卡重复计入**，"
          "不能直接当分层压缩率的分母）：")
    for line in report.tier_table():
        print("  " + line)

    # 分层压缩率：判据口径（唯一归属）。与上面 tier_table 的「原文按 source_ids 归属」不同，
    # 后者同一段原文被多张卡引用会被重复计入，只能当参考。
    print()
    print("  分层压缩率（**唯一归属口径**，§4.1.1 判据用这一份）：")
    print(f"      {'tier':<6}{'原文tok':>10}{'卡tok':>10}{'压缩率':>9}")
    for tier in sorted(tier_unique):
        raw, card = tier_unique[tier]
        shown = f"{raw / card:.3f}x" if card else "—"
        print(f"      {tier:<6}{raw:>10,}{card:>10,}{shown:>9}")
    for label, tiers in (
        ("保留组 T0–T2", ("T0", "T1", "T2")),
        ("压缩组 T3–T7", ("T3", "T4", "T5", "T6", "T7")),
        ("免设组 T8/T9", ("T8", "T9")),
    ):
        raw = sum(tier_unique.get(t, [0, 0])[0] for t in tiers)
        card = sum(tier_unique.get(t, [0, 0])[1] for t in tiers)
        print(f"      {label:<14}{raw:>10,}{card:>10,}"
              f"{(f'{raw / card:.3f}x' if card else '—'):>9}")
    attributed = sum(bucket[0] for bucket in tier_unique.values())
    total_raw = sum(result.raw_tokens for result in results)
    print(f"      · 分层合计覆盖原文 {attributed:,} tok / 全部 {total_raw:,} tok —— 差额 "
          f"{total_raw - attributed:,} tok 是**没有任何卡引用的原文**（未计入任何层级，"
          "属 C1/保留率的损失，不摊给别的层级）")

    # 卡 token 花在哪——「为什么在膨胀」的直接读数（分母 = 卡文本字段 token 之和）
    field_tokens: dict[str, int] = {}
    usage: dict[str, int] = {}
    for result in results:
        for name, value in result.field_tokens.items():
            field_tokens[name] = field_tokens.get(name, 0) + value
        for name, value in result.field_usage.items():
            usage[name] = usage.get(name, 0) + value
    total_field = sum(field_tokens.values()) or 1
    print()
    print(f"  卡 token 的字段构成（全部落库卡；字段 token 合计 {total_field:,}）：")
    for name, tokens in sorted(field_tokens.items(), key=lambda item: -item[1]):
        print(f"      {name:<12}{usage.get(name, 0):>4} 次{tokens:>8,} tok {tokens / total_field:>6.1%}")

    return 1 if any(result.error for result in results) or engine.parse_failures else 0


def run_shape(cases: Sequence[GoldCase], *, dry_run: bool) -> int:
    """形状模式：一条用例一个请求，只读分析，不进主管线。"""
    print(f"LLM 形状探针：{len(cases)} 条用例（每条一个请求，**不经主管线**）")
    for case in cases:
        raw, how, _usage = call_llm(build_prompt(case), dry_run=dry_run)
        prefix = "mock." if dry_run else ""
        dump = OUT_DIR / f"{prefix}{case.id}.raw.txt"
        dump.parent.mkdir(parents=True, exist_ok=True)
        dump.write_bytes(raw.encode("utf-8"))
        print(f"=== {case.id}（{len(case.messages)} 条消息，{how}）")
        print(f"    原始 dump: {dump.relative_to(ROOT)}")
        try:
            result = analyse_shape(raw, case.messages)
        except Exception as exc:  # 原始内容已落盘，这里只报告
            print(f"    解析失败：{type(exc).__name__}: {exc}")
            continue
        print(f"    卡数={result['cards']} 字段条目={result['field_entries']}")
        print(f"    C6 违规={len(result['C6'])}  C8 违规={len(result['C8'])}"
              "（判据来自 verify.verify_*，与契约同源）")
        print(
            f"    逐句字段：复述整句 {result['verbatim_entries']} / "
            f"抽象 {result['abstracted_entries']}"
        )
        for line in (result["C6"] + result["C8"])[:4]:
            print(f"      · {line}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 LLM 实验：形状与全链路")
    parser.add_argument("--mock", action="store_true", help="不联网，用内联假响应验证管道")
    parser.add_argument("--all", action="store_true", help="形状模式跑全部 25 条（默认 3 条）")
    parser.add_argument(
        "--pipeline", action="store_true", help="全链路模式：25 条走完整管线，读 §4.1 六条判据"
    )
    parser.add_argument(
        "--replay", action="store_true",
        help="断点续跑：已有 dump 直接复用，缺的批次才联网（复盘不重复花钱）",
    )
    parser.add_argument("--only", metavar="CASE_ID", help="只跑一条用例（冒烟）")
    args = parser.parse_args(argv)

    cases = load_gold()
    if args.only:
        picked = [case for case in cases if case.id == args.only]
        if not picked:
            print(f"没有这条用例：{args.only}")
            print("可用：", "、".join(case.id for case in cases))
            return 2
        cases = picked
    if args.pipeline:
        return run_pipeline(cases, dry_run=args.mock, replay=args.replay)
    if args.all or args.only:
        return run_shape(cases, dry_run=args.mock)
    return run_shape(pick_cases(), dry_run=args.mock)


if __name__ == "__main__":
    raise SystemExit(main())
