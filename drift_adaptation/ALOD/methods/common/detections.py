"""Shared detector-result JSON loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from methods.common.io import read_json


DEFAULT_RECORD_KEYS = ('detections', 'annotations', 'results')


def load_detection_records(
    path: Path,
    record_keys: Sequence[str] = DEFAULT_RECORD_KEYS,
    required_keys: Optional[Iterable[str]] = None,
    schema_name: str = 'detection',
) -> List[Dict[str, Any]]:
    """Load list-like detector result records from common JSON layouts."""

    result_path = Path(path)
    data = read_json(result_path)
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = None
        for key in record_keys:
            if isinstance(data.get(key), list):
                records = data[key]
                break
        if records is None:
            raise ValueError('Unsupported %s JSON schema: %s' % (schema_name, result_path))
    else:
        raise ValueError('Unsupported %s JSON schema: %s' % (schema_name, result_path))

    normalized = []
    required = tuple(required_keys or ())
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                '%s record %d in %s must be an object'
                % (schema_name, index, result_path))
        missing = [key for key in required if key not in record]
        if missing:
            raise KeyError(
                '%s record %d in %s is missing keys: %s'
                % (schema_name, index, result_path, ', '.join(missing)))
        normalized.append(dict(record))
    return normalized
