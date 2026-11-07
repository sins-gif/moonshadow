# 月影（Moonshadow）说明书

> **一句话**：把「对话里说过的东西」变成**可以按需取回的记忆卡**，让新对话不用重读文件、重跑命令，
> 就能答出上次定过什么。
>
> **先说它不是什么**，免得误会：
> 1. **不是向量库**——检索目前是词面匹配，不是语义嵌入（限制见 §7）；
> 2. **不是聊天记录备份**——原文不进上下文，只按 `source_id` 回取；
> 3. **不承诺省 token**——压缩率实测 `0.498x`，这条宣称已在 `v1.2.2` **撤回**（见 §9 与
>    [`withdrawn/W1-compression.md`](withdrawn/W1-compression.md)）。
>
> **它承诺的是**：跨会话记得住、不用重新提取、可追溯到原文、不硬塞无关内容。
> 这四条各自可判定，且能现场复算（`python examples/mvp_memory.py`）。

---

## 1 它解决的问题

Agent 的「记忆问题」不是「存不下」，而是这四件事：

| # | 问题 | 月影怎么答 | 实测（`examples/mvp_memory.py`） |
|---|---|---|---|
| 1 | 新对话还记得吗 | 从库里召回相关短卡，注入上下文 | **9/12** 问命中，平均 **86 tok** |
| 2 | 要不要重新提取 | 原文只在库里，召回不读业务文件 | 一次召回 **业务文件 0 个、命令 0 条** |
| 3 | 答错了能不能查 | 每张卡带 `source_ids`，可回原文 | **逐字一致** |
| 4 | 会不会硬塞无关内容 | 两段硬门 + 预算装箱 | 无关问题里**非常驻卡 0 张** |

> 第 1 条没到 12/12 的原因写在 §7：3 题是「问法与卡内措辞不一致」造成的**检索漏**，
> 不是没存进去（**存储漏 0**）。

## 2 五分钟上手

**依赖**：Python 3.10+，**零第三方依赖**。不需要装数据库、不需要联网。

```powershell
cd 'D:\代码库\moonshadow (月影)'

# ① 看它能不能记住（自包含，一次跑完，约 3 秒）
python examples/mvp_memory.py

# ② 看完整链路（原文冷存 → 短卡 → 召回 → 装箱 → 升降级）
python examples/demo.py

# ③ 全量自检（语法 + 测试 + 文档守卫）
python tools/check_syntax.py
python tools/audit_docs.py
```

`mvp_memory.py` 会把报告写到 `.tmp/MVP-报告.md`，退出码 `0` 表示「无存储漏、无污染」。

## 3 三个使用入口

### 3.1 Python API（最灵活）

```python
from datetime import datetime, timezone
from moonshadow import FixedClock, Store
from moonshadow.compress import BaselineExtractor, compile_session
from moonshadow.recall import recall

root = "./my-memory"
clock = FixedClock(datetime.now(timezone.utc))
store = Store(root, clock=clock)

# 写入：原文按天追加，字节不可变
store.append_message("s1", "时间下限 TIME_FLOOR 取 0.5925。")
store.append_message("s1", "GPU 三档：25.3 倍、141.7 倍、1902 倍。")

# 压成卡（幂等：重跑只会增加 replayed，不会重复落卡）
compile_session(store, "s1", BaselineExtractor(), clock=clock)

# 召回：需要一个「卡 id → 相似度」映射（这一步由调用方提供，见 §4）
cards = store.cards(session_id="s1")
query = "时间下限取多少"
cos = {str(card["id"]): my_similarity(query, card) for card in cards}
result = recall(store, now=clock.now(), budget_tokens=800, query_cos=cos)

context = "\n".join(text for _section, text in result["blocks"] if text)
```

**注意**：`recall()` 是**纯函数**、不写库、不记访问。它只做「门 → 打分 → 装箱」。

### 3.2 CLI（JSON 进 / JSON 出，给别的程序用）

```powershell
# 写入（消息按时间顺序；session 相同会累积）
'{"session":"s1","messages":[{"text":"预算 12 万，截止 2026-10-15。"}]}' | python tools/moon_cli.py remember

# 召回（不填 session 则全库检索）
'{"query":"预算多少","budget_tokens":4000}' | python tools/moon_cli.py recall

# 看库规模（只读）
'{}' | python tools/moon_cli.py stats
```

**库位置默认 `~/.dsh/moonshadow`**，可用环境变量 `MOONSHADOW_ROOT` 覆盖。
中文参数**走 stdin**（PowerShell 会吞掉 argv 里的双引号）。

### 3.3 DSH 插件（在 agent 里自动用）

| 项 | 说明 |
|---|---|
| 位置 | `D:\Plugins\MOD\dsh-moonshadow`（本地运行，不进 git） |
| 三个工具 | `moon_remember` / `moon_recall` / `moon_stats` |
| 自动采集 | 订阅 `session/event`：用户消息、助手消息、DSH 的压缩摘要 → 写库 |
| 自动注入 | 每来一个问题，召回一次（预算 **800 tok**），结果进下一轮上下文 |
| 开关 | `MOONSHADOW_AUTO=0` 退回「只有三个工具」 |

**第一轮通常没有注入块**：召回是异步的，第二轮起稳定出现，块头会写「按问题「…」召回 · N/M tok」。

## 4 它怎么工作（三分钟读懂）

```
写入：对话 → 原文（raw/<day>.jsonl，不可变）→ 抽取 → 记忆短卡（SQLite）
读取：问题 → 定性门（卡龄 + T8/T9 禁令）→ 打分 q = r · m → 定量门（q > θ）→ 装箱 → 注入
```

四件事值得记住：

1. **原文与卡分离**。原文按天追加、字节不可变；卡是摘要，带 `source_ids` 指回原文。
   所以「压掉了什么」永远查得回来。
2. **门是两段，不是一段**。定性段在打分**之前**（卡龄窗口、`T8/T9` 禁令），
   定量段在打分**之后**（`q > θ`）。`θ` 是分数的阈值，不可能排在打分前。
3. **时间是门 + 有界修正，不是权重**。`m = f + (1−f)·δ`，`f` 取 `0.5925`，
   最多把相关性打 `f` 折；`1/f = 1.687764` 是**次序可被反转的比值上界**。
4. **装箱有预算**。装不下先降级 `full → lite`，再丢弃并记理由（`no_space` / `quota_exhausted`）。

更细的证明与实验在 [`v1.2-spec.md`](v1.2-spec.md)（权威）与 [`v1.2-weights.md`](v1.2-weights.md)（证据）。

## 5 记住什么、忘掉什么

| 层级 | 含义 | 半衰期（天） | 卡龄窗口 |
|---|---|---|---|
| T0 | 身份 / 硬约束 / 安全 / 法律 | `0`（不衰减） | 无限制，**常驻** |
| T1 | 长期偏好 / 核心目标 / 关系 | `365` | 无限制，**常驻** |
| T2 | 项目决策 / 重要承诺 | `180` | 无限制 |
| T3 | 近期待办 / 截止 | `90` | 无限制 |
| T4 | 一般方案 / 讨论结论 | `30` | ≤ 30 天 |
| T5 | 日常事实 | `14` | ≤ 30 天 |
| T6 | 短期任务细节 | `7` | ≤ 7 天 |
| T7 | 中间推理 / 草稿 | `3` | ≤ 7 天 |
| T8 | 寒暄 / 确认 | `1` | 仅显式回溯 |
| T9 | 噪音 | `1/24` | 仅显式回溯 |

**两条必须知道的事实**：

- **规则只产出 8 层**：`T4` / `T7` 没有判定路径（关键词表只覆盖 `T0–T3`，兜底给 `T5`/`T6`）。
  实测 25 条金标准里这两层是 **0 张**。这不是「没有内容」，是**不可达**。
- **层级目前是只读标签**：`next_tier`（升降）有实现有测试，但**生产代码零调用**，
  而它依赖的 `note_access`（访问记账）只在示例里调过。所以一张卡的层级写入后不会变。

## 6 常用命令

| 想做的事 | 命令 |
|---|---|
| 记忆能力自检（12 问） | `python examples/mvp_memory.py` |
| 端到端演示 | `python examples/demo.py` |
| 全量测试 + 语法守卫 | `python tools/check_syntax.py` |
| 文档守卫 | `python tools/audit_docs.py` |
| 抽取评测关卡（保留率 + 压缩率） | `python tools/run_eval.py` |
| 负例是否真被门挡住 | `python tools/run_gate_audit.py` |
| 留出集是否被改动 | `python tools/run_split_integrity.py` |
| 召回规模压力测试 | `python tools/stress_recall.py --cards 1000,10000` |

## 7 会不会漏、会不会污染

| 现象 | 数据 | 原因与修法 |
|---|---|---|
| **漏答**（问了但没召回） | MVP 12 问漏 3 | **检索漏**：问「装在哪」，卡里是「放在」。修法在 backlog：给卡填 `keywords`，或换语义嵌入 |
| **召回常驻卡** | 无关问题也会看到 `T0/T1` | **不是 bug，是设计**：身份与硬约束不能被静默裁掉。判据因此是「非常驻的不相关卡 0 张」 |
| **常驻区挤爆预算** | 1k 卡占 `41%`、10k 卡 `448%`、30k 卡 `1322%` | 常驻区**不受配额**，是已知缺陷 `#19`，待修 |

## 8 已知边界（诚实清单）

| # | 边界 | 现状 |
|---|---|---|
| 1 | **压缩率不达标** | `0.498x`（卡比原文大）。宣称已撤回，见 `W1`；根因是抽取器只会「抄」不会「并」 |
| 2 | 常驻区不受配额 | 大库会挤掉其它分区（`#19`） |
| 3 | 层级不会演化 | `next_tier` 零调用、`note_access` 未接 |
| 4 | `T4`/`T7` 不可达 | 规则判不出；接 LLM 抽取器也补不上（模型不参与选层） |
| 5 | 检索是词面 | 措辞不一致会漏；阈值 `θ` 是在语义余弦上标定的，词面下的通过/拦截**不能当质量结论** |
| 6 | 冷存归档未接线 | `archive_day()` 只有示例与测试调用，真实库 `chunks/` 是 0 文件 |
| 7 | 单进程、单写者 | 没有并发设计；插件串行 `spawn` |
| 8 | 召回随规模变慢 | 1k 卡 `~91 ms`、10k `~2.7 s`、20k `~6 s`（20k 起需索引） |

## 9 常见问题

**Q：`moon_stats` 返回 0 张卡。**
库是空的——**没有任何东西被写进去过**。先 `moon_remember`，或开插件自动采集后聊几轮。

**Q：写中文报 `UnicodeEncodeError: '\udca2'`。**
`v1.2.2` 之前的老 bug（stdin 按系统编码解码）。**升级到 `v1.2.2` 或更新**即可；本版有守卫测试钉住。

**Q：报 `SchemaError: $.summary: 长度 4652 > maxLength 500`。**
同样是老 bug（关键字段回填没有上限）。`v1.2.2` 已修，上限 `500`。

**Q：库里有卡，但召回是空的。**
两种可能：① 相似度没过 `θ`（换种说法再问，或给卡补 `keywords`）；
② 该卡层级是 `T8/T9`，需要显式回溯（`explicit_recall`）才放行。

**Q：注入的第一轮没有记忆块。**
召回在事件里异步跑，**第二轮起**才稳定；块头会标明是按哪个问题召回的。

**Q：上下文还是被撑爆了。**
先看是不是常驻区（`T0/T1` 不受配额，见 §8 第 2 条）。预算本身由 `--budget-tokens` 控制，
默认注入 `800 tok`。

**Q：能不能只判门、不用库？**
可以。门与打分是**纯函数**，`from moonshadow.scoring import passes_gate` 直接调，
实测零文件 I/O（不建库、不读文件）。

## 10 术语表

| 词 | 含义 |
|---|---|
| **短卡** | 一条消息抽成的记忆卡：`tier` + `summary` + 字段 + `source_ids` |
| **两段门** | 定性（卡龄、`T8/T9`）+ 定量（`q > θ`），一前一后 |
| **`f`** | 时间下限 `0.5925`，把相关性最多打这个折扣 |
| **`θ`** | 分数门槛 `0.3941`，`q` 过它才准入 |
| **常驻区** | `T0/T1` 卡的归处，不受配额、总是进上下文 |
| **装箱** | 按 token 预算把卡拼进上下文；装不下先降级再丢弃 |
| **`W1`** | 撤回记录第 1 条：撤回「上下文压缩」这个定位 |
| **`P1`–`P5`** | 待解锁台账：已决定要做、被可判定前置卡住的事 |

## 11 文档地图

| 想知道 | 看 |
|---|---|
| 协议、定义、**定值参数表**（唯一权威） | [`v1.2-spec.md`](v1.2-spec.md) |
| 定理、证明、实验数据 | [`v1.2-weights.md`](v1.2-weights.md) |
| 交接总览、已失效结论、复跑索引 | [`v1.2-summary.md`](v1.2-summary.md) |
| 哪一版改了什么、哪条宣称被撤回 | [`CHANGELOG.md`](CHANGELOG.md)、[`withdrawn/`](withdrawn/README.md) |
| 哪些边界「以后能改、现在不行」 | [`pending/README.md`](pending/README.md) |
| 召回要多久、能撑到多少卡 | [`v1.2-perf-model.md`](v1.2-perf-model.md) |
| 重要但现在不该做的想法 | [`backlog.md`](backlog.md) |
| 新开对话该先读什么 | [`../HANDOFF.md`](../HANDOFF.md) |
