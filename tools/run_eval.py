"""评测入口。

    python tools/run_eval.py [金标准目录]

退出码 0 = 关卡通过，1 = 有指标未达标。可直接接入 CI。
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow.eval import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
