from __future__ import annotations

import json
from pathlib import Path

import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_feature_file(path: str | Path, payload: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(payload, path)


def load_feature_file(path: str | Path) -> dict:
    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
