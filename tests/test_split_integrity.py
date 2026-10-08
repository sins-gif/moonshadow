"""留出集冻结的守卫（v1.2 决策 3）。

冻结是声明，校验才是约束。`eval/split-manifest.json` 记录了每个留出文件的 SHA-256；
**改一个字节即违规**。本测试与 `tools/run_split_integrity.py` 检查同一件事：
工具给人看，测试让它在每次跑测试时都被强制执行。
"""

from __future__ import annotations

import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from moonshadow.rank_eval import load_holdout_cases, load_rank_cases  # noqa: E402
from run_split_integrity import check  # noqa: E402

MANIFEST = ROOT / "eval" / "split-manifest.json"


class SplitIntegrityTest(unittest.TestCase):
    def test_manifest_exists_and_is_well_formed(self) -> None:
        self.assertTrue(MANIFEST.exists(), "留出集必须先冻结：eval/split-manifest.json")
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for key in ("seed", "dev", "holdout", "note"):
            self.assertIn(key, manifest)
        self.assertEqual(len(manifest["holdout"]), 9)
        for entry in manifest["holdout"]:
            self.assertEqual(len(entry["sha256"]), 64, entry)

    def test_holdout_files_match_the_frozen_hashes(self) -> None:
        violations, _added = check()
        self.assertEqual(violations, [], "留出集被改动过，或与冻结清单不一致")

    def test_holdout_and_dev_are_disjoint(self) -> None:
        dev = {case.id for case in load_rank_cases()}
        held = {case.id for case in load_holdout_cases()}
        self.assertFalse(dev & held)

    def test_dev_may_grow_but_must_not_lose_frozen_members(self) -> None:
        """冻结后新增开发用例是正常开发活动；但清单里的开发用例不许消失。

        消失意味着它被搬进了留出集——那就是污染，必须被挡住。
        """
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        dev = {case.id for case in load_rank_cases()}
        self.assertFalse(set(manifest["dev"]) - dev, "冻结清单里的开发用例不见了")
        added = dev - set(manifest["dev"])
        self.assertIn("22-t9-noise-blocked-by-tier", added, "新用例应被记为「冻结后新增」")


if __name__ == "__main__":
    unittest.main()
