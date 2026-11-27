from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPLORE_SOURCE = PROJECT_ROOT / "src_explore"
sys.path.insert(0, str(EXPLORE_SOURCE))
