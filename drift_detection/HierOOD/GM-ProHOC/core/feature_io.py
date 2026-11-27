from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_artifact(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_artifact(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)
