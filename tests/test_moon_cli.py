"""插件桥接验收：`tools/moon_cli.py` 必须是「UTF-8 进、UTF-8 出」。

为什么要单独钉住：这个 CLI 是 DSH 插件（Node）与月影（Python）之间唯一的边界。
它曾经用 `sys.stdin.read()` 读参数，于是走**系统编码**（本机 cp936），
中文 JSON 被解成孤立代理字符，打印时抛
`UnicodeEncodeError: '\udca2' ... surrogates not allowed`——即「插件写中文必崩」，
而且只在中文下崩、纯英文测试全绿。这里用**真实子进程**跑字节流，把编码钉死。
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLI = ROOT / "tools" / "moon_cli.py"


class MoonCliStdinTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = ROOT / ".tmp" / f"{type(self).__name__}.{self._testMethodName}"
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.tmp.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, command: str, payload: dict) -> dict:
        """把 payload 以 UTF-8 字节写进 stdin，按 UTF-8 解 stdout。"""
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        proc = subprocess.run(
            [sys.executable, str(CLI), command],
            input=body,
            capture_output=True,
            cwd=str(ROOT),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))
        return json.loads(proc.stdout.decode("utf-8"))

    def test_chinese_payload_survives_stdin(self) -> None:
        result = self._run(
            "remember",
            {
                "session": "cli-utf8",
                "root": str(self.tmp),
                "messages": [{"text": "用户：预算 12 万，截止 2026-10-15。"}],
            },
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["cards"], 1)
        # 中文一路活着穿过 stdin → SQLite → stdout，不是在某处被替换成问号
        stats = self._run("stats", {"session": "cli-utf8", "root": str(self.tmp)})
        self.assertEqual(stats["cards"], 1)
        recalled = self._run("recall", {"query": "预算多少", "root": str(self.tmp), "budget_tokens": 800})
        self.assertIn("12 万", recalled["context"])

    def test_payload_from_argv_is_the_same_path(self) -> None:
        """argv 与 stdin 两条入口必须等价：argv 走 str，stdin 走 UTF-8 字节。"""
        payload = json.dumps({"session": "cli-argv", "root": str(self.tmp)}, ensure_ascii=False)
        proc = subprocess.run(
            [sys.executable, str(CLI), "stats", payload],
            capture_output=True,
            cwd=str(ROOT),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))
        self.assertTrue(json.loads(proc.stdout.decode("utf-8"))["ok"])

    def test_bad_json_is_reported_as_json(self) -> None:
        """错误也必须是 JSON：插件把 message 原样回给模型，不能让它读栈回溯。"""
        proc = subprocess.run(
            [sys.executable, str(CLI), "stats"],
            input="不是 JSON".encode("utf-8"),
            capture_output=True,
            cwd=str(ROOT),
        )
        self.assertEqual(proc.returncode, 1)
        result = json.loads(proc.stdout.decode("utf-8"))
        self.assertFalse(result["ok"])
        self.assertIn("error", result)
