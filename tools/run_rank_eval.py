"""排序方案对照实验入口。

    python tools/run_rank_eval.py [用例目录]

只输出证据（各方案与人工期望排序的一致率），不设通过阈值——
权重该取哪种，是人看完数据后的决定，不适合自动化。
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow.rank_eval import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
