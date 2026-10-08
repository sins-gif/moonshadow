"""参数漂移下的反转区域测量（薄封装）。

    python tools/run_drift_check.py

测量内容与判据见 `src/moonshadow/drift.py`。
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow.drift import format_drift_report, measure_drift  # noqa: E402

if __name__ == "__main__":
    print(format_drift_report(measure_drift()))
