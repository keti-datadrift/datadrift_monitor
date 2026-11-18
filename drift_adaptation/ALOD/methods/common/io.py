"""Method-neutral JSON I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def to_jsonable(value: Any) -> Any:
    """Convert common scientific Python values into JSON-serializable data."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, 'tolist'):
        return to_jsonable(value.tolist())
    if hasattr(value, 'item'):
        return value.item()
    return value


def read_json(path: Path) -> Any:
    with Path(path).open('r', encoding='utf-8') as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any, indent: Optional[int] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(to_jsonable(payload), handle, indent=indent, ensure_ascii=False)
    return path
