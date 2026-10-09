"""语法守卫：对仓库内所有 *.py 跑 ast.parse。

    python tools/check_syntax.py

存在的理由：本项目踩过**脚本拼引号把测试文件写坏**的坑（SyntaxError 直到跑测试才发现）
——那是「用脚本生成代码，却没在写入后立即验证产物」的第三个实例。
本命令 3 秒内拦住所有拼引号、漏括号、错缩进的同类错误，并挂进验证集。

退出码：全部解析通过为 0；任一文件语法错误为 1（打印文件与行列）。
"""

from __future__ import annotations

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SKIP = ("__pycache__", ".git", ".venv", "node_modules", ".tmp", "build", "dist")


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
