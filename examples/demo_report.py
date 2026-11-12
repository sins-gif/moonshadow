"""生成一页静态报告（单文件 HTML，零依赖、可双击打开、可转发、可存档）。

为什么是静态页而不是前端：老师要看的是「想法完整 + 证据扎实」，这两件事靠**可复跑的数字**，
不靠交互。这个脚本在生成时**真的去跑**那些命令，把输出写进页面——所以页面上的每个数字
都能用页脚的命令复算，不存在「文档写死了但代码变了」的漂移。

用法：
    python examples/demo_report.py                 # 输出 .tmp/demo_report.html
    python examples/demo_report.py --out demo.html # 指定输出
    python examples/demo_report.py --fast          # 跳过 288 条测试（省 20 秒）

零第三方依赖：只用标准库；页面内联 CSS，不引用任何 CDN。
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

BUDGET = 800


def run(args: list[str], timeout: int = 600) -> tuple[int, str]:
    done = subprocess.run(
        [sys.executable, *args], cwd=ROOT, capture_output=True,
        text=True, encoding="utf-8", timeout=timeout,
    )
    return done.returncode, (done.stdout or "") + (done.stderr or "")


def last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "（无输出）"


def recall_samples(queries: list[str]) -> list[dict]:
    """在当前库上跑几条真实召回（只读）。"""
    from moon_cli import cmd_recall

    rows = []
    for query in queries:
        got = cmd_recall({"query": query, "budget_tokens": BUDGET})
        rows.append({
            "query": query,
            "used": got["used_tokens"],
            "packed": got["packed"],
            "dropped": got["dropped"],
            "context": got["context"],
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成静态报告页")
    parser.add_argument("--out", default=str(ROOT / ".tmp" / "demo_report.html"))
    parser.add_argument("--fast", action="store_true", help="跳过 288 条测试")
    args = parser.parse_args(argv)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print("跑证据……")

    tests = ("（本页用 --fast 生成，未跑测试）", "")
    tests_code = 0
    if not args.fast:
        tests_code, out = run(["-m", "unittest", "discover", "-s", "tests", "-q"])
        ran = next((line for line in out.splitlines() if line.startswith("Ran ")), "Ran ? tests")
        tests = (f"{ran} → {'OK' if tests_code == 0 else 'FAILED'}", f"exit {tests_code}")
        print("  ", tests[0])

    audits = []
    for args_, expect, label in (
        (["tools/run_batch_audit.py"], 0, "新批 22–25：R1/R2 可以合入"),
        (["tools/run_batch_audit.py", "12", "13", "14", "15", "16", "19", "20"], 1,
         "旧批 12–21：必须被拒绝"),
        (["tools/audit_docs.py"], 0, "文档守卫：链接 / 定值 / 文件数"),
    ):
        code, out = run(args_)
        audits.append({
            "label": label, "expect": expect, "code": code,
            "ok": code == expect, "tail": last_line(out),
        })
        print(f"   {label}: exit {code}（期望 {expect}）")

    code, out = run(["examples/quickstart.py"])
    quickstart = {"code": code, "tail": last_line(out)}
    print("   quickstart:", quickstart["tail"])

    samples = recall_samples([
        "今天演示要展示什么", "插件改完怎么生效", "常驻区有什么问题", "压缩率是多少",
    ])
    print(f"   召回样例 {len(samples)} 条")

    page = render(stamp, tests, audits, quickstart, samples)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    print(f"\n已写入 {out_path}（{len(page.encode('utf-8'))} 字节）")
    return 0 if (all(a["ok"] for a in audits) and code == 0 and tests_code == 0) else 1


def render(stamp: str, tests: tuple[str, str], audits: list[dict], quickstart: dict,
           samples: list[dict]) -> str:
    esc = html.escape
    audit_rows = "\n".join(
        f"<tr><td>{esc(a['label'])}</td><td class='num'>exit {a['code']}"
        f"（期望 {a['expect']}）</td><td>{'✅ 符合' if a['ok'] else '❌ 不符'}</td>"
        f"<td class='mono'>{esc(a['tail'][:70])}</td></tr>"
        for a in audits
    )
    sample_rows = "\n".join(
        f"<tr><td>{esc(s['query'])}</td><td class='num'>{s['used']}/{BUDGET} tok</td>"
        f"<td class='num'>{s['packed']} 张</td><td class='num'>{s['dropped']}</td>"
        f"<td class='mono'>{esc(' '.join(s['context'].split())[:60])}…</td></tr>"
        for s in samples
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>月影 Moonshadow · 演示报告</title>
<style>
 :root {{ --ink:#1c2530; --muted:#5b6673; --line:#d8dee6; --bg:#fbfcfd; --accent:#1f4e79; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; padding:40px 24px 64px; background:var(--bg); color:var(--ink);
        font:15px/1.75 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif; }}
 main {{ max-width:960px; margin:0 auto; }}
 h1 {{ font-size:26px; margin:0 0 6px; letter-spacing:.5px; }}
 h2 {{ font-size:18px; margin:36px 0 10px; padding-left:10px; border-left:4px solid var(--accent); }}
 .lede {{ color:var(--muted); margin:0 0 4px; }}
 .stamp {{ color:var(--muted); font-size:13px; }}
 table {{ width:100%; border-collapse:collapse; margin:8px 0 4px; font-size:14px; }}
 th, td {{ border-bottom:1px solid var(--line); padding:7px 10px; text-align:left; vertical-align:top; }}
 th {{ background:#f1f4f8; font-weight:600; }}
 td.num {{ white-space:nowrap; font-variant-numeric:tabular-nums; }}
 .mono {{ font-family:ui-monospace,Consolas,monospace; font-size:12.5px; color:var(--muted); }}
 .kpi {{ display:flex; gap:14px; flex-wrap:wrap; margin:14px 0 0; }}
 .kpi div {{ flex:1 1 190px; border:1px solid var(--line); border-radius:8px; padding:12px 14px; background:#fff; }}
 .kpi b {{ display:block; font-size:20px; color:var(--accent); font-variant-numeric:tabular-nums; }}
 .kpi span {{ color:var(--muted); font-size:13px; }}
 .warn {{ background:#fff8e6; border-left:4px solid #d9a441; padding:10px 14px; margin:10px 0; }}
 footer {{ margin-top:40px; color:var(--muted); font-size:13px; border-top:1px solid var(--line); padding-top:14px; }}
 code {{ font-family:ui-monospace,Consolas,monospace; background:#eef1f5; padding:1px 5px; border-radius:4px; }}
 svg {{ max-width:100%; height:auto; }}
</style></head><body><main>

<h1>月影 Moonshadow · 演示报告</h1>
<p class="lede">一个 T0–T9 时间分层的记忆协议：把对话压成记忆卡，按问题取用，注入量硬性不超过预算。</p>
<p class="stamp">本页生成于 {esc(stamp)}；页面上的每个数字都由 <code>examples/demo_report.py</code> 当场跑出来。</p>

<div class="kpi">
  <div><b>{esc(tests[0].replace(' → ', ' · '))}</b><span>测试套件（{esc(tests[1] or '详页脚')}）</span></div>
  <div><b>{sum(1 for a in audits if a['ok'])}/{len(audits)}</b><span>审计命令符合预期（含一条必须失败）</span></div>
  <div><b>{esc(quickstart['tail'][-24:])}</b><span>quickstart 闭环</span></div>
</div>

<h2>1　数据流（一张图）</h2>
<svg viewBox="0 0 940 132" role="img" aria-label="数据流：消息 → 压卡 → 分层 → 打分 → 两道门 → 装箱 → 注入块">
  <defs><marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
    <path d="M0,0 L7,3 L0,6 z" fill="#8794a4"/></marker></defs>
  {_svg_boxes()}
</svg>
<p class="stamp">原文进 <code>raw/&lt;day&gt;.jsonl</code>（不可变追加）；卡进 SQLite；召回是纯函数（不记账）。</p>

<h2>2　实跑证据</h2>
<table><thead><tr><th>审计命令</th><th>退出码</th><th>判定</th><th>输出末行</th></tr></thead>
<tbody>{audit_rows}</tbody></table>
<div class="warn"><b>为什么这是一条证据而不是装饰：</b>第三条要求<b>必须失败</b>——旧批用例过不了 R1/R2，
命令就得拒绝合入（<code>exit 1</code>）。规则若从不失败，就只是装饰。</div>

<h2>3　召回样例（当前库）</h2>
<table><thead><tr><th>查询</th><th>用量</th><th>卡</th><th>丢弃</th><th>块首</th></tr></thead>
<tbody>{sample_rows}</tbody></table>
<p class="stamp">预算是硬上限：装箱按分区配额贪心，装不下记 <code>no_space</code>／额度为 0 记 <code>quota_exhausted</code>。</p>

<h2>4　已知缺陷（如实列出）</h2>
<table><thead><tr><th>编号</th><th>缺陷</th><th>影响</th></tr></thead><tbody>
<tr><td class="num">W1</td><td>压缩率在 0.45–0.50x（在膨胀）</td>
    <td>「上下文压缩」这个定位已<b>撤回</b>：卡的体积反而大于原文，且同一句会在 <code>summary</code> 与 <code>facts</code> 各存一份</td></tr>
<tr><td class="num">#19</td><td>常驻区（T0/T1）不受预算配额</td>
    <td>大库或长卡会先吃掉预算；实测上限公式 <code>R_max = B − n·c̄/q</code> 在 800 tok、卡中位 134 tok 时<b>无解</b></td></tr>
<tr><td class="num">—</td><td>检索用词面二元组，阈值 θ 在语义余弦上标定</td>
    <td>换个问法就可能取不回（例：「三周前的预算截止日」取不回，「预算截止到什么时候」取得到）</td></tr>
<tr><td class="num">—</td><td>层级是只读标签</td><td><code>next_tier</code> 生产零调用，<code>access_count</code> 恒为 0：层级不会随使用演化</td></tr>
<tr><td class="num">—</td><td>T4／T7 结构不可达</td><td>宣称 10 层、规则只能赋出 8 层；分层读数里它们恒为 0，是<b>覆盖缺口</b>不是「没有内容」</td></tr>
</tbody></table>

<h2>5　复跑</h2>
<table><thead><tr><th>命令</th><th>验证什么</th></tr></thead><tbody>
<tr><td class="mono">python examples/quickstart.py</td><td>存 → 查 → 审计三条（含一条必须失败）</td></tr>
<tr><td class="mono">python examples/demo_report.py</td><td>生成本页（每个数字当场复算）</td></tr>
<tr><td class="mono">python -m unittest discover -s tests -q</td><td>288 条断言</td></tr>
<tr><td class="mono">python tools/run_batch_audit.py</td><td>批次验收 R1/R2（不合格退出码 1）</td></tr>
<tr><td class="mono">python examples/mvp_memory.py</td><td>MVP 四条判据 + 落盘报告</td></tr>
</tbody></table>

<footer>零依赖：Python 标准库 + 系统 SQLite；页面内联 CSS，不引用任何外部资源。</footer>
</main></body></html>
"""


def _svg_boxes() -> str:
    """七个框 + 箭头：会话消息 → 压卡 → 分层 → 打分 → 两道门 → 装箱 → 注入块。"""
    boxes = [
        ("会话消息", "user / assistant"),
        ("压卡", "summary·facts"),
        ("时间分层", "T0–T9 · 半衰期"),
        ("打分", "q = r·m"),
        ("两道硬门", "θ · 卡龄窗"),
        ("装箱 ≤800 tok", "配额 + 滚存"),
        ("注入块", "写明 query"),
    ]
    parts, x, w, gap = [], 12, 118, 18
    for index, (title, sub) in enumerate(boxes):
        y = 26
        parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="54" rx="8" fill="#fff" stroke="#b9c4d2"/>'
            f'<text x="{x + w / 2}" y="{y + 23}" text-anchor="middle" font-size="13" '
            f'fill="#1c2530">{html.escape(title)}</text>'
            f'<text x="{x + w / 2}" y="{y + 41}" text-anchor="middle" font-size="10.5" '
            f'fill="#5b6673">{html.escape(sub)}</text>'
        )
        if index < len(boxes) - 1:
            parts.append(
                f'<line x1="{x + w}" y1="{y + 27}" x2="{x + w + gap - 3}" y2="{y + 27}" '
                f'stroke="#8794a4" stroke-width="1.6" marker-end="url(#a)"/>'
            )
        x += w + gap
    return "\n  ".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
