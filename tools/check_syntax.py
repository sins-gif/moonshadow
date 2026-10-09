"""语法守卫 + 全量测试：一道命令跑完「语法」与「行为」两层。

    python tools/check_syntax.py            # ast.parse 全部 *.py，再跑全量测试
    python tools/check_syntax.py --no-tests # 只查语法（改文档时想快点）

存在的理由：本项目踩过**脚本拼引号把测试文件写坏**的坑（SyntaxError 直到跑测试才发现）
——那是「用脚本生成代码，却没在写入后立即验证产物」的第三个实例。
进一步的理由：本轮又踩了「改动正确、周边记账未验证」——`ast.parse` 抓不到那种错，
只有跑测试能抓。所以两层合到一条命令里，避免只跑一层就以为验证过了。

退出码：语法或测试任一失败为 1。
"""

from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SKIP = ("__pycache__", ".git", ".venv", "node_modules", ".tmp", "build", "dist")


def run_tests() -> int:
    """第二层：跑全量测试。`ast.parse` 抓不到「改动正确但周边记账过期」那一类错。"""
    print("运行全量测试：python -m unittest discover -s tests")
    # 中文断言信息走管道时必须显式指定编码：Windows 上 `text=True` 默认按本地代码页（GBK）
    # 解码，子进程一旦输出中文断言信息就会在这里崩掉 `UnicodeDecodeError`——
    # 守卫自己崩掉比测试红更糟：它会让「跑过守卫」变成假信号（本轮实测踩到）。
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    for line in output.splitlines():
        if line.startswith(("Ran ", "OK", "FAILED")):
            print("  " + line)
    if completed.returncode != 0:
        # 记账漂移的报错就在断言信息里，只打印 `FAILED` 等于把守卫的信息量扔掉一半。
        print("  ✗ 测试未全绿（下面是输出末段；完整输出请直接跑 unittest）")
        for line in output.splitlines()[-25:]:
            print("    " + line)
        return 1
    print("  ✓ 语法与测试双绿")
    return 0


def main() -> int:
    bad: list[str] = []
    checked = 0
    for path in sorted(ROOT.rglob("*.py")):
        parts = path.relative_to(ROOT).parts
        if any(part in SKIP for part in parts):
            continue
        checked += 1
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            bad.append(
                "%s:%s:%s %s" % (path.relative_to(ROOT), exc.lineno, exc.offset, exc.msg)
            )
    print("语法守卫：检查 %d 个 .py 文件" % checked)
    if bad:
        print("  违规 %d 项：" % len(bad))
        for line in bad:
            print("    ✗", line)
        return 1
    print("  ✓ 全部文件可被 ast.parse")
    if "--no-tests" in sys.argv[1:]:
        return 0
    return run_tests()


if __name__ == "__main__":
    raise SystemExit(main())
