"""重划开发集 / 留出集，并把留出集**冻结**（v1.2 决策 3）。

    python tools/resplit_cases.py --seed 20260601 --dry-run     # 先看会怎么分
    python tools/resplit_cases.py --seed 20260601 --apply       # 真正落盘并写清单

背景：步 0 修用例时改动了留出集里的 `H02`/`H03`/`H05`，而 `H02` 正是当时上界的钉住者。
留出集因此被开发过程污染，不能再当作留出集。这里把**全部用例合并后按固定随机种子重划**，
并把切分结果连同每个留出文件的 SHA-256 写进 `eval/split-manifest.json`。

**冻结的含义**：清单一旦写出，留出集文件不得再改。任何改动都会被
`tools/run_split_integrity.py` 与本仓库的测试判为违规。要改就必须显式重新冻结，
并说明理由——这一步刻意做成需要人工动作，避免「顺手改一下留出集」。

切分粒度是**用例文件**：`eval/rank/` = 开发集，`eval/holdout/` = 留出集。
文件名保持用例 id 不变（不重编号），目录决定归属——这样每条用例的来历可追溯。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RANK_DIR = ROOT / "eval" / "rank"
HOLDOUT_DIR = ROOT / "eval" / "holdout"
MANIFEST = ROOT / "eval" / "split-manifest.json"

#: 留出集占比。31 条用例下 0.30 给出 9–10 条；实际条数由 `holdout_size` 决定。
HOLDOUT_FRACTION = 0.30


def holdout_size(total: int) -> int:
    """留出集条数。四舍五入到最近的整数，且至少 1 条、最多不超过总数的一半。"""
    size = int(round(total * HOLDOUT_FRACTION))
    return max(1, min(size, total // 2))


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect() -> dict[str, pathlib.Path]:
    """合并两个目录，按用例 id 建索引；同一个 id 出现在两边直接报错。"""
    found: dict[str, pathlib.Path] = {}
    for directory in (RANK_DIR, HOLDOUT_DIR):
        for path in sorted(directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            case_id = str(payload["id"])
            if case_id in found:
                raise SystemExit(f"用例 id 重复：{case_id}（{found[case_id]} 与 {path}）")
            found[case_id] = path
    return found


def plan(seed: int, cases: dict[str, pathlib.Path]) -> tuple[list[str], list[str]]:
    """按 id 排序后随机抽样，得到 (开发集 id, 留出集 id)。排序保证与文件系统顺序无关。"""
    ids = sorted(cases)
    size = holdout_size(len(ids))
    rng = random.Random(seed)
    holdout = sorted(rng.sample(ids, size))
    dev = sorted(set(ids) - set(holdout))
    return dev, holdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="重划开发集 / 留出集并冻结留出集")
    parser.add_argument("--seed", type=int, required=True, help="固定随机种子（会写进清单）")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="只打印切分结果")
    group.add_argument("--apply", action="store_true", help="移动文件并写清单")
    args = parser.parse_args(argv)

    cases = collect()
    dev_ids, held_ids = plan(args.seed, cases)

    print(f"用例总数 {len(cases)} → 开发集 {len(dev_ids)} / 留出集 {len(held_ids)}")
    print(f"随机种子 = {args.seed}")
    print(f"留出集：{', '.join(held_ids)}")

    if args.dry_run:
        print("\n（--dry-run：未改动任何文件）")
        return 0

    for case_id in dev_ids:
        path = cases[case_id]
        if path.parent != RANK_DIR:
            shutil.move(str(path), str(RANK_DIR / path.name))
    for case_id in held_ids:
        path = cases[case_id]
        if path.parent != HOLDOUT_DIR:
            shutil.move(str(path), str(HOLDOUT_DIR / path.name))

    manifest = {
        "seed": args.seed,
        "fraction": HOLDOUT_FRACTION,
        "dev": dev_ids,
        "holdout": [
            {"id": case_id, "file": (HOLDOUT_DIR / cases[case_id].name).name,
             "sha256": sha256(HOLDOUT_DIR / cases[case_id].name)}
            for case_id in held_ids
        ],
        "note": (
            "留出集在本次冻结后不得再修改。改动会被 tools/run_split_integrity.py "
            "与 tests/test_split_integrity.py 判为违规。"
        ),
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n已写入 {MANIFEST.relative_to(ROOT)}（含每个留出文件的 SHA-256）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
