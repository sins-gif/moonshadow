# HANDOFF：把上下文交出去，不靠记忆

> **这份文件的用途**：新开一个对话时，第一句发 `读 HANDOFF.md`，就能拿到全部现状、判决与数字。
> 它替代「靠上下文记住」，也替代「把 46 万字聊天记录灌进记忆库」（后者实测会让新对话直接爆，见 §7）。
>
> **本文件不引入新结论**：协议与定值以 [`docs/v1.2-spec.md`](docs/v1.2-spec.md) 为唯一权威；
> 推理与实验数据以 [`docs/v1.2-weights.md`](docs/v1.2-weights.md) 为准；本文件只做交接索引。
> **所有数字以工具输出为准**，每条都标了产出它的命令。

---

## 1 一句话现状

协议、定值、评测集、GPU 加速、DSH 插件都已落地；`main` 是零依赖的 v1.2.0（250 项测试全绿），
`v1.3-gpu-acceleration` 分支上是 CuPy PoC（真实测到 1902× 的 kernel 级加速）。
**下一步的瓶颈只有一个：压缩率仍是 `0.498x`（卡比原文大）**——LLM 抽取器没接上之前，不做新小版本。

---

## 2 仓库与版本

| 项 | 值 |
|---|---|
| 仓库路径 | `D:\代码库\moonshadow (月影)`（**注意目录名带空格与中文**；`D:\代码库\moonshadow` 是旧路径残留，已废弃） |
| 当前分支 | `v1.3-gpu-acceleration`（HEAD `bac9649`） |
| `main` | `71a8b15`，零依赖，`python -m unittest discover -s tests` → `Ran 250 tests` + `OK` |
| 已打标签 | `v1.2.0`（`9e50357`）、`v1.2.1`（`5a829af`）、`v1.2.x-experimental`（`cfef6db`） |
| `__version__` | `1.2.0`（`src/moonshadow/__init__.py`） |
| 规模 | `docs/` 13 · `src/moonshadow/` 20 · `tools/` 20 · `tests/` 16 · `eval/` 62 · 根目录交给 `tests/test_docs.py` 的规模守卫算 |

---

## 3 定值（唯一权威在 `docs/v1.2-spec.md` 的参数表）

```
r = 0.60 σ + 0.27 (i/10) + 0.13 μ        σ = (1+s)/2
δ = 0.5 ^ (Δ / h)                        h = 半衰期（T0 为 0，不衰减）
m = f + (1 − f) δ                        f = TIME_FLOOR
q = r · m
```

| 参数 | 定值 | 备注 |
|---|---|---|
| `TIME_FLOOR`（`f`） | **`0.5925`** | 开发集窗与留出集窗交集 `[0.5225, 0.6600]` 内近似居中（下余 `0.0700`、上余 `0.0675`）；严格说中点依赖扫描步长，别写「精确中点」 |
| `THETA` | **`0.3941`** | 乘性量纲下两集合标定区间的建议值 |
| `THETA_HIGH` | **`0.3941`**（`θ′ = θ`） | 已取消 T6/T7 的独立更高门槛 |
| 采纳方案 | **`multiplicative`**（固定下限） | `scoring.select()` 默认即生产配置 |
| 反转上界 | `1/f = 1.687764` | 时间最多把相关性打 `f` 折 |
| `FLOOR_SWEEP_STEP` | `0.0025` | 必须整除 `0.5925`（`237` 步） |
| 半衰期（天） | T0 `0`、T1 `365`、T2 `180`、T3 `90`、T4 `30`、T5 `14`、T6 `7`、T7 `3`、T8 `1`、T9 `1/24` | `src/moonshadow/tiers.py` 是唯一权威 |
| 卡龄窗口 | T4–T5 ≤ 30 天、T6–T7 ≤ 7 天 | 定义 6 |

**门是两段合取，不要混成一个**：

1. `scoring.passes_qualitative_gate`（**打分之前**）：卡龄窗口 + T8/T9 禁令。
2. `scoring.passes_quantitative_gate`（**打分之后**）：`q > θ`。

`passes_gate` = 两者之合取；`recall.py` 用断言证明「两段合取 == `select()` 的结果」，且召回零副作用。

---

## 4 版本政策（已被纠正过一次，别改回去）

- **没有 v1.3**。原先规划的 v1.3 已并入 **v1.2.x Roadmap（实验性探索）**，
  文档是 `docs/v1.2-future-llm-extractor.md` 与 `docs/v1.2-future-llm-extractor-freeze.md`，
  由 `tools/audit_docs.py` 的 `EXPERIMENTAL` 名单强制标注「实验性」。
- **不开新小版本，直到 LLM 抽取器把真实压缩率做到 `> 1.0x`。** 这是硬条件，不是建议。

### 4.1 压缩率验收口径（改过一次，以现在这版为准）

全局 `2.0x` **在算术上不可能**，已证明：

| 分组 | 卡 token 占比 | 目标 |
|---|---|---|
| 保持组 `T0–T2` | `64.8%` | 不设压缩目标；`T0/T1` 允许 `1.0–1.2x` |
| 压缩组 `T3–T7` | `34.9%` | `≥ 2.0x` |
| 豁免组 `T8/T9` | `0.4%` | 不设目标 |

⇒ 全局最优约 **`≈1.41x`**。所以验收改成**按层级**给目标，再加一条**信息条目**指标：
同一事实的「载体数」≤ 1（原来「`todos` 硬扣分」升级为 **veto**，优先级 `6 > 5`）。

---

## 5 已验证的实测结果（都能复跑）

| 项 | 结果 | 命令 |
|---|---|---|
| 全量测试 | `Ran 250 tests` + `OK`，59 个 `.py` 全过 `ast.parse` | `python tools/check_syntax.py` |
| 文档守卫 | 退出 0 | `python tools/audit_docs.py` |
| 抽取评测 | 保留率 `1.0000`、压缩 `0.498x`、`25/25` 通过、基线 `C6 违规 705 条（预期）` | `python tools/run_eval.py` |
| 门审计 / 用例审计 | 违规 `0` 张 | `run_gate_audit.py` / `run_case_audit.py` |
| 切分完整性 | 留出集 9 条 SHA-256 与冻结值一致 | `run_split_integrity.py` |
| 批次验收 | R1 `3/4`、R2 `0.750` → 可以合入 | `run_batch_audit.py` |
| 端到端 | 退出 0（含 1 行**故意的** FAIL，是第 9 步负向验证） | `python examples/demo.py` |

### 5.1 GPU（`v1.3-gpu-acceleration` 分支，RTX 3050 Laptop 4GB，CUDA 12.9，驱动 577.03）

50,000 × 768 float64，15 组样本，**三档读数**：

| 档位 | 端到端 | 加速 | 抖动 |
|---|---|---|---|
| list 布局（默认，含 Python 装箱） | `10.63 s → 0.427 s` | 中位 **`25.3×`**，区间 `21.2–26.3×` | `20.0%` |
| 连续内存（`--contiguous`） | `10.82 s → 0.075 s` | 中位 **`141.7×`**，区间 `95.5–159.6×` | `45.2%`（双峰） |
| 纯 kernel | `5.4–5.6 ms` | **`1902–1995×`** | — |

两种布局都与 CPU 逐元素一致：50,000 个输出 `max_abs_diff = 5.551e-17`。
**时间去哪了**（`tools/transfer_breakdown.py`）：`list→ndarray` 序列化 `84.4%`、H2D `16.7%`、kernel `1.8%`、D2H `0.07%`。

**引用纪律**：说「1902×」必须同时说「纯 kernel」；说「25×」必须说「含序列化」。跨档位比较没有意义。

---

## 6 DSH 插件（月影作为 Agent 记忆）

| 项 | 位置 / 做法 |
|---|---|
| 插件包 | `D:\Plugins\MOD\dsh-moonshadow\`：`package.json`、`cordis.patch.yml`、`lib/index.js`、`README.md` |
| 桥接 CLI | 仓库内 `tools/moon_cli.py`，**JSON 进 / JSON 出**，三条命令 `remember` / `recall` / `stats` |
| 三个工具 | `moon_remember`、`moon_recall`、`moon_stats` |
| 记忆库位置 | `~/.dsh/moonshadow`（默认；`MOONSHADOW_ROOT` / `DSH_HOME` 可覆盖） |
| 插件装法 | 在 `~/.dsh/profiles/web/package.json` 的 `dependencies` 加 `link:`，**并且** `dsh.profile.bundles` 里登记；`patchReload: live` 只热更 patch 文件，不改 `package.json` |

**今天修掉的两条真 bug（都实测复现过）**：

1. `moon_cli.py` 原先用 `sys.stdin.read()` 读参数 → 走系统编码（cp936），中文 JSON 解出孤立代理字符，
   随后打印抛 `UnicodeEncodeError: '\udca2' ... surrogates not allowed`——**插件写中文必崩**。
   现在按 `sys.stdin.buffer` 读 UTF-8 字节，并把 `stdout/stderr` 强制 `reconfigure(encoding="utf-8")`。
2. `compress.build_summary` 的「关键字段回填」没有上限，长消息会把 summary 撑到 `4652` 字，
   撞穿 `schema.json` 的 `maxLength: 500` → `SchemaError` 让整次抽取中止。
   现在加了 `SUMMARY_MAX_CHARS = 500`：按出现顺序收关键字段，放不下的整条跳过，返回值一定过校验。

---

## 7 为什么「别把聊天记录全灌进记忆库」（实测，别再试一遍）

用本会话自己的 transcript 做过两个实验（临时库，未污染正式库）：

| 实验 | 规模 | 卡 | 卡 token | 一次 `recall(budget=2000)` |
|---|---|---|---|---|
| 全量导入（185 条用户原话 + 153 条最终答复 + 4 份交接摘要） | 462,603 字 → 1,922 块 | 1,922 张 | 436,156（原文 241,160，**压缩 `0.553x`，卡比原文大 1.8 倍**） | 装 475 张、`148,886` token，**超 74 倍** |
| 只导 4 份交接摘要 | 47,852 字 → 212 块 | 212 张 | 26,218 | 10 张，`2,131` token（**只剩这 10 张 T0**） |

原因在 `src/moonshadow/pack.py` 第 1 条规则：**常驻区（T0/T1）无条件进入，不受预算限制**
（这条规则本身是对的：身份与硬约束不该被静默裁掉）。但对话里「必须/不许」太多，
大量块被判成 T0，于是预算保证被冲穿。

**结论**：v1.2 的基线抽取器适合**少量、短、可检索**的事实条目，
不适合长篇原文。交接靠这份 `HANDOFF.md`，细节靠 `moon_recall`。

---

## 8 未完成 / 已知缺陷（按重要性）

| # | 项 | 现状 |
|---|---|---|
| 1 | **LLM 抽取器未接入** | 压缩率 `0.498x`（膨胀）。这是所有后续工作的前置条件 |
| 2 | 常驻区无上限 | 大量 T0 卡会让 `recall` 的预算失效（§7 实测 74 倍） |
| 3 | 10 个半衰期与 2 个卡龄窗口从未扫描 | 仍是占位值，全部实验按给定值计算 |
| 4 | `T9` 无正例 | 开发集正例 T9 为 0，只有负例侧证据 |
| 5 | `network` schema 缺口 | 每集合 1/24 条无正确类别，任何分类器都补不上 |
| 6 | 加性 vs 乘性仍只能靠定理与漂移实验区分 | 一致率表对这两者无区分力 |
| 7 | 留出集一次性已用掉一部分 | `f = 0.5925` 与 `θ′` 都动过它，下个参数决策需要**新建**留出集 |
| 8 | 一次召回里同一条事实会出现 2–3 次 | 同一张卡的不同字段落在不同分区（`cards` / `todo` / `long_term`），正是「载体数 ≤ `1`」要治的东西；实测一次 4,000 预算的召回里，「raw 不可变追加」那句重复 3 遍 |
| 9 | `build_summary` 回填在长消息上被「先出现的字段类型」吃光预算 | 日期能独吞 500 字；修法（按类型轮转）会让全仓库压缩率 `0.498x → 0.497x`，**故意不改**，记在 backlog 第 4 条 |
| 10 | **`T4`/`T7` 结构性不可达** | `classify_tier` 只有 `T0–T3` 关键词 + `T5/T6` 兜底；`extractor.py:203` 用同一函数分组，**模型不选层**。所以连 LLM 抽取器也补不上。见 §8.1 |
| 11 | **层级写入后永不改变** | `next_tier` 有实现有 8 条测试但**生产零调用**；`note_access` 只在 `examples/demo.py` 里调过 → `access_count` 恒为 `0`。层级是只读标签 |

想法池在 [`docs/backlog.md`](docs/backlog.md)（5 条：T5–T7 合并、跨批次信息条目台账、GPU 加速、
`build_summary` 回填轮转、`T4`/`T7` 判据缺口），每条都附了可判定的前置条件——**前置条件不满足就不做**。

### 8.1 分层实况（代码级，2026-11-05 逐条核过）

`TIER_ORDER` 写了 10 层，但**代码能赋到的只有 8 层，真实数据里只出现过 6 层**：

| 层 | 谁能赋给它 | 实测出现过 |
|---|---|---|
| T0 / T1 / T2 / T3 | `compress.classify_tier` 的四组关键词 | 是 |
| **T4 / T7** | **没有任何路径**（无关键词、无兜底分支） | **否**（25 条金标准、87 条事实，两张表都是 0） |
| T5 / T6 | 兜底：含数字或时间词 → `T5`，否则 → `T6` | 是（**占 72%**） |
| T8 / T9 | `classify_fast_path`（纯寒暄 / 噪音） | 是 |

三条硬事实：

1. **`T4`/`T7` 连 LLM 抽取器也补不上**——`extractor.py:203` 是 `tiers_of = [(message, classify_tier(message.text)) ...]`，模型只负责在同一层内合并成卡。
2. **层级写入后永不改变**——`next_tier` 那是死路径，`access_count` 永远是 0。
3. **`TIER_LABEL` 是死数据**（全仓库零引用）；`TIER_IMPORTANCE`/`TIER_KEYWORDS` 住 `compress.py`、`TIER_GROUPS` 住 `rank_eval.py`，与 `tiers.py` 自称「唯一权威」不一致。

`tier` 在运行时**真正只影响 4 件事**：半衰期（`cards.py:66`）、卡龄窗（`scoring.py:358-366`）、
阈值档 `θ`/`θ′`（`scoring.py:384-393`）、是否进常驻区（`recall.py:85`）。其余都是标签。

### 今天新加的两个守卫测试（别删）

- `tests/test_compress.py::test_summary_respects_schema_max_length`：
  钉住 `SUMMARY_MAX_CHARS == schema.json 的 maxLength`，并断言长消息的 summary 不越界。
- `tests/test_moon_cli.py`（3 条）：用真实子进程跑字节流，钉住「stdin 是 UTF-8、stdout 是 UTF-8」，
  以及「出错也必须回 JSON」。这三条存在的理由就是那个中文必崩的 bug 只在中文下复现。

---

## 9 协作规则（这些是用户明确要求过的，不是我的偏好）

1. **没有命令不要动文件。** 要改先说明改什么、为什么。
2. **不许 hedging**：「大概」「也许」「可能」一律删掉；不确定就写「不确定」，并说清边界。
3. **数字以工具输出为准。** 任何数字都要能指出是哪条命令产出的。
4. **同一件事只有一处说了算。** 定值只在 spec 的参数表；层级只在 `tiers.py`。
5. **自己错了就当场认、并改正**；被推翻的结论要**记录**（写进「已失效」表），不能删掉。
6. 回答**短、有结论、只给一个推荐**。不要重新讨论已经定过的事（比如「SQL 只是存卡的地方」）。

### Windows 环境的坑（踩过，会再踩）

- 只有 `powershell.exe` 5.1，**没有 `pwsh`**。
- PowerShell 会把原生参数里的双引号吃掉 → 传 JSON 走 **stdin** 或临时文件，不要走 argv。
- 控制台是 cp936：跑会打印 `✓`／中文脚本前先 `$env:PYTHONIOENCODING='utf-8'`，否则报 `UnicodeEncodeError`。
- `Set-Content` / `write_text` 默认写 CRLF → 仓库有 `test_whole_tree_is_lf` 会红，务必用字节写。
- 中文用户名会让部分 CUDA 工具链找不到头文件（`nvidia-cuda-*` + `TEMP` 指向 ASCII 路径可绕）。

---

## 10 复跑全部验证

```powershell
$env:PYTHONIOENCODING='utf-8'
cd 'D:\代码库\moonshadow (月影)'
python tools/check_syntax.py          # ast.parse 全树 + 全量测试（250 项）
python tools/audit_docs.py            # 文档守卫
python tools/run_eval.py              # 抽取评测关卡
python tools/run_gate_audit.py        # 负例是否真被门挡住
python tools/run_case_audit.py        # 用例卡龄窗口
python tools/run_split_integrity.py   # 留出集冻结哈希
python tools/run_batch_audit.py       # R1/R2 批次验收
python examples/demo.py               # 端到端
python tools/benchmark_gpu.py --n 50000 --dim 768 --runs 15   # GPU 三档
```

---

## 11 其它产物（都在 `.tmp/`，已被 gitignore）

| 文件 | 是什么 |
|---|---|
| `.tmp/Moonshadow-项目汇报-v7.pptx` | 31 页技术版汇报（页数偏多，可裁到 15–20 页） |
| `.tmp/Moonshadow-实用落地版.pptx` | 11 页落地版（给非技术听众） |
| `.tmp/Moonshadow-架构图.pptx` | 1 页架构图 |
| `.tmp/blog-final-numbers.md`、`blog-final-sections.md`、`resume-bullet.md` | 博客与简历素材（数字已按三档口径统一） |

**凭据**：之前用过一次 DeepSeek key，**需要用户自行轮换**；当前环境没有 key，
`.tmp/llm_probe/pipeline/` 里 58 份原始 dump 可用 `--replay` 复现，不需要联网。
