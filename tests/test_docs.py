"""文档守卫的测试：让「数字漂移」与「链接失效」在每次跑测试时被强制检查。

与 `tools/audit_docs.py` 检查同一件事：工具给人看，测试让它不可绕过。
"""

from __future__ import annotations

import collections
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from audit_docs import (  # noqa: E402
    DOCS,
    HISTORICAL_MARKERS,
    INDEX,
    ROOT_DOCS,
    SPEC,
    check,
    check_authority,
    check_citations_of_history,
    check_classification,
    check_historical_banners,
    check_index_completeness,
    check_inline_doc_paths,
    check_links,
    check_parameter_table,
    check_values,
    is_historical,
    markdown_files,
)


class DocGuardTest(unittest.TestCase):
    def test_no_violations(self) -> None:
        self.assertEqual(check(), [])

    def test_each_rule_is_reported_separately(self) -> None:
        """八条规则各自独立可跑——否则一条挂了会掩盖其他几条。"""
        for rule in (
            check_links,
            check_inline_doc_paths,
            check_classification,
            check_values,
            check_historical_banners,
            check_citations_of_history,
            check_parameter_table,
            check_authority,
            check_index_completeness,
        ):
            with self.subTest(rule=rule.__name__):
                self.assertEqual(rule(), [])

    def test_root_facades_are_inspected(self) -> None:
        """根目录门面文档（中文/英文 README）必须真的进内容检查，而不只是被「登记」。

        实测漏过的一类错：分类表里加了英文 README，但取文件全集的函数仍写死中文版，
        于是英文版不进链接/赋值式/历史引用检查，守卫却照样打印「全部可达」。
        「登记了」不等于「被检」——这条断言把两者钉在一起。
        """
        inspected = {path.resolve() for path in markdown_files()}
        for path in ROOT_DOCS:
            with self.subTest(path=path.name):
                self.assertIn(
                    path.resolve(), inspected,
                    f"{path.name} 在 ROOT_DOCS 里登记了，却没有进受检文件全集",
                )

    def test_historical_versions_are_retained_not_removed(self) -> None:
        """v1.0/v1.1 必须**留在 docs/ 里**——它们的价值是留下推理链。

        这条断言把「历史版本不删」变成契约：谁把它们移走或删掉，测试就会红。
        同时它们必须带状态标注（说明数据不作为依据），否则读者会误当现行。
        """
        for name in ("v1.1-spec.md", "v1.1-weights.md", "v1.1-summary.md", "v1.0-moonshadow.md"):
            path = DOCS / name
            self.assertTrue(path.exists(), f"{name} 是历史推理记录，不应被删除或移出 docs/")
            self.assertTrue(is_historical(path), f"{name} 应被识别为历史版本")
            text = path.read_text(encoding="utf-8")
            for marker in HISTORICAL_MARKERS:
                self.assertIn(marker, text, f"{name} 缺状态标注「{marker}」")

    def test_index_exists_and_lists_every_document(self) -> None:
        self.assertTrue(INDEX.exists(), "docs/README.md 是文档索引，必须有")
        text = INDEX.read_text(encoding="utf-8")
        for path in sorted(DOCS.rglob("*.md")):
            if path == INDEX:
                continue
            self.assertIn(path.name, text, f"{path.name} 未登记进索引")

    def test_spec_is_the_single_authority(self) -> None:
        """索引必须把 spec 标成唯一权威——否则「以谁为准」又会散掉。"""
        text = INDEX.read_text(encoding="utf-8")
        self.assertIn(SPEC.name, text)
        self.assertIn("唯一权威", text)

    def test_backlog_entries_carry_status_and_preconditions(self) -> None:
        """想法池里每条想法都要写清 状态 / 前置条件 / 来源。

        想法池的价值在于「以后能被捡起来」：只写结论的想法，几个月后没人知道它卡在哪、
        为什么当时不做。前置条件必须尽量指向**可判定的东西**（某条命令通过、某参数落地）。
        """
        backlog = DOCS / "backlog.md"
        self.assertTrue(backlog.exists(), "想法池 docs/backlog.md 必须存在")
        text = backlog.read_text(encoding="utf-8")
        self.assertIn("backlog.md", INDEX.read_text(encoding="utf-8"), "想法池必须登记进索引")

        sections = [s for s in text.split("\n## ") if s.strip()][1:]
        self.assertTrue(sections, "想法池里至少要有一条想法")
        for section in sections:
            title = section.split("\n", 1)[0].strip()
            for field in ("状态", "前置条件", "来源"):
                self.assertIn(field, section, f"想法「{title}」缺字段：{field}")

    def test_documented_file_count_matches_reality(self) -> None:
        """文档「规模」行的文件总数与目录分解必须与仓库现状一致。

        口径（与 `docs/v1.2-summary.md` 的规模行一致）：计入**非隐藏文件**，
        排除以 `.` 开头的路径与缓存/构建目录；`src/` 下只有 `moonshadow/` 一个包，
        故合并成一行计。

        加这条是因为实测漏过一次：新增英文 README 后总数与「根 4」都过期了，
        而没有任何检查会发现——文件数漂移比测试条数漂移更隐蔽（前者没有输出可对）。
        """
        skip = {
            "__pycache__", ".git", ".venv", "node_modules",
            ".mypy_cache", ".pytest_cache", ".ruff_cache", "build", "dist",
        }
        counts: collections.Counter[str] = collections.Counter()
        total = 0
        for path in ROOT.rglob("*"):
            parts = path.relative_to(ROOT).parts
            if any(part.startswith(".") or part in skip for part in parts):
                continue
            if not path.is_file():
                continue
            total += 1
            if len(parts) == 1:
                key = "(root)"
            elif parts[0] == "src":
                key = "/".join(parts[:2])
            else:
                key = parts[0]
            counts[key] += 1

        expected = (
            f"| 规模 | {total} 个文件（`docs/` {counts['docs']} · "
            f"`src/moonshadow/` {counts['src/moonshadow']} · `tools/` {counts['tools']} · "
            f"`tests/` {counts['tests']} · `eval/` {counts['eval']} · "
            f"`examples/` {counts['examples']} · `experiments/` {counts['experiments']} · "
            f"根 {counts['(root)']}）"
        )
        summary = (DOCS / "v1.2-summary.md").read_text(encoding="utf-8")
        self.assertIn(
            expected, summary,
            "docs/v1.2-summary.md 的「规模」行与仓库现状不一致：\n"
            f"  实际应为：{expected}\n"
            "（新增文件后请同步该行；口径见本测试的说明）",
        )

    def test_documented_test_count_matches_reality(self) -> None:
        """文档里写的 `Ran N tests` 必须等于真实条数——让这个数字自维护。

        写死条数必然漂移（本轮就从 174 一路变到 195）。与其每加一个测试就手工改文档，
        不如让文档里的数字**被测试钉住**：加了测试而没更新文档，这里就红。
        只检查现行文档；历史文档说的是当时的条数，属历史，不查。
        """
        import re

        actual = unittest.TestLoader().discover(str(ROOT / "tests")).countTestCases()
        for path in (ROOT / "README.md", DOCS / "v1.2-summary.md"):
            claimed = {int(n) for n in re.findall(r"Ran (\d+) tests", path.read_text(encoding="utf-8"))}
            self.assertTrue(claimed, f"{path.name} 未写明测试条数")
            self.assertEqual(
                claimed, {actual},
                f"{path.name} 声称的条数 {sorted(claimed)} 与实际 {actual} 不一致",
            )


if __name__ == "__main__":
    unittest.main()
