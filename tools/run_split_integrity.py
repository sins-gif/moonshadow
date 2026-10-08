"""校验留出集自冻结以来未被修改（v1.2 决策 3 的强制手段）。

    python tools/run_split_integrity.py

冻结是声明，校验才是约束。本工具检查四件事：

1. `eval/split-manifest.json` 存在且可解析；
2. 清单里每个留出文件的 SHA-256 与磁盘一致（**改一个字节即违规**）；
3. 留出目录里的文件集合与清单完全一致（多一个、少一个都算违规）；
4. 开发集与留出集的用例 id 不相交，且两边 id 集合与清单一致。

退出码：全部通过为 `0`，任何一项违规则为 `1`。
要合法地改留出集，必须用 `python tools/resplit_cases.py --seed <新种子> --apply` 重新冻结，
并说明理由——刻意做成需要人工动作。
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RANK_DIR = ROOT / "eval" / "rank"
HOLDOUT_DIR = ROOT / "eval" / "holdout"
MANIFEST = ROOT / "eval" / "split-manifest.json"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_in(directory: pathlib.Path) -> dict[str, pathlib.Path]:
    found: dict[str, pathlib.Path] = {}
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        found[str(payload["id"])] = path
    return found


def check() -> tuple[list[str], list[str]]:
    """返回 (违规列表, 冻结后新增的开发用例)。"""
    violations: list[str] = []
    if not MANIFEST.exists():
        return [f"缺少 {MANIFEST.relative_to(ROOT)}——留出集尚未冻结"], []

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    frozen = {entry["id"]: entry for entry in manifest.get("holdout", [])}
    dev_on_disk = ids_in(RANK_DIR)
    held_on_disk = ids_in(HOLDOUT_DIR)

    for case_id, entry in sorted(frozen.items()):
        path = HOLDOUT_DIR / entry["file"]
        if not path.exists():
            violations.append(f"清单里的留出用例 {case_id} 不在 {HOLDOUT_DIR.name}/")
            continue
        actual = sha256(path)
        if actual != entry["sha256"]:
            violations.append(
                f"留出用例 {case_id} 已被修改（SHA-256 {actual[:12]} ≠ 冻结值 "
                f"{entry['sha256'][:12]}）"
            )

    for case_id, path in sorted(held_on_disk.items()):
        if case_id not in frozen:
            violations.append(f"{path.name} 在留出目录里，但不在冻结清单内")

    overlap = sorted(set(dev_on_disk) & set(held_on_disk))
    for case_id in overlap:
        violations.append(f"用例 {case_id} 同时出现在开发集与留出集")

    expected_dev = set(manifest.get("dev", []))
    expected_held = set(frozen)
    # 开发集**允许增长**（新增用例是正常的开发活动），但不允许原有用例消失或搬去留出集。
    missing_dev = sorted(expected_dev - set(dev_on_disk))
    if missing_dev:
        violations.append(
            f"清单记录的开发用例不见了：{missing_dev}（若搬去了留出集，留出集已被污染）"
        )
    added_dev = sorted(set(dev_on_disk) - expected_dev)
    if set(held_on_disk) != expected_held:
        violations.append("留出集与清单不一致（见上）")

    return violations, added_dev


def main() -> int:
    violations, added_dev = check()
    print("留出集冻结校验")
    if violations:
        print(f"  违规 {len(violations)} 项：")
        for line in violations:
            print(f"    ✗ {line}")
        return 1
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    print(f"  ✓ 留出集 {len(manifest['holdout'])} 条，全部与冻结值一致")
    print(f"  ✓ 开发集 {len(manifest['dev']) + len(added_dev)} 条，与留出集无交集")
    if added_dev:
        print(f"  · 冻结后新增开发用例 {len(added_dev)} 条：{', '.join(added_dev)}")
    print(f"  ✓ 冻结种子 = {manifest['seed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
