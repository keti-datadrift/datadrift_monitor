from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from methods.common.image_identity import normalize_image_id
from methods.common.io import read_json


def load_uncertainty_records(path: Path) -> List[Dict[str, Any]]:
    payload = read_json(Path(path))
    records = payload.get('records')
    if not isinstance(records, list):
        raise ValueError('MIAL uncertainty artifact must contain a records list: %s' % path)
    normalized = []
    for record in records:
        if 'image_id' not in record:
            raise KeyError('MIAL uncertainty record is missing image_id')
        if 'score' not in record:
            raise KeyError('MIAL uncertainty record is missing score')
        item = dict(record)
        item['image_id'] = normalize_image_id(item['image_id'])
        item['score'] = float(item['score'])
        item.setdefault('components', {})
        item.setdefault('metadata', {})
        normalized.append(item)
    return normalized
