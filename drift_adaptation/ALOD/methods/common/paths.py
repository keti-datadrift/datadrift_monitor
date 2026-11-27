"""Shared path containment helpers."""

from __future__ import annotations

from pathlib import Path


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        Path(path).relative_to(Path(parent))
        return True
    except ValueError:
        return False
