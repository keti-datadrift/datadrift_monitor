from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CMPT_SOURCE = PROJECT_ROOT / "src_cmpt"
sys.path.insert(0, str(CMPT_SOURCE))
