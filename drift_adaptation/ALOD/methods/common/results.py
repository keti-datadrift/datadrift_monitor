"""Shared acquisition result and diagnostics payload helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from methods.common.io import write_json


def acquisition_result(
    method: str,
    stage: Optional[str],
    round_index: int,
    budget: int,
    selected_image_ids: Iterable[Any],
    inputs: Optional[Mapping[str, Any]] = None,
    outputs: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build a consistent method-neutral acquisition result envelope."""

    selected = list(selected_image_ids)
    payload: Dict[str, Any] = {
        'method': method,
        'round_index': int(round_index),
        'budget': int(budget),
        'selected_image_ids': selected,
        'selected_count': len(selected),
        'inputs': dict(inputs or {}),
        'outputs': dict(outputs or {}),
        'metrics': dict(metrics or {}),
    }
    if stage is not None:
        payload['stage'] = stage
    payload.update(extra)
    return payload


def write_diagnostics(path: Path, diagnostics: Dict[str, Any]) -> Path:
    """Write a diagnostics dictionary as stable UTF-8 JSON."""

    return write_json(path, diagnostics, indent=2)
