"""Print a CUDA preflight report for deployment troubleshooting.

Run this from the project's Python environment before starting a GPU run.  It
checks the driver API separately from NVML, because ``nvidia-smi`` can work
while CUDA applications cannot initialize in a misconfigured container.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gpu import get_system_status  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(get_system_status(), indent=2, sort_keys=True))
