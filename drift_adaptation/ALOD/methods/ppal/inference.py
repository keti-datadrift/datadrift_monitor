"""PPAL detector artifact loading.

PPAL method code consumes artifacts produced by the local MMDetection backend:

* ``RetinaHeadUncertainty`` writes ``*.bbox.json`` records with
  ``cls_uncertainty`` for DCUS.
* ``RetinaQualityEMAHead`` stores ``bbox_head.class_quality`` in
  ``latest.pth`` for DCUS class reweighting.

The loaders here keep that backend boundary explicit so ``dcus.py`` and
``ccms.py`` can focus on acquisition logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from methods.common.detections import load_detection_records


UNCERTAINTY_RESULT_KEYS = ('image_id', 'category_id', 'bbox', 'score', 'cls_uncertainty')
CLASS_QUALITY_KEY = 'bbox_head.class_quality'


def _as_path(path: Any) -> Path:
    return path if isinstance(path, Path) else Path(path)


def class_quality_checkpoint_path(result_json: Any) -> Path:
    """Return the checkpoint path paired with a PPAL uncertainty result JSON."""

    return _as_path(result_json).parent / 'latest.pth'


def load_uncertainty_detections(path: Any) -> List[Dict[str, Any]]:
    """Load and validate PPAL uncertainty detections."""

    result_path = _as_path(path)
    return load_detection_records(
        result_path,
        required_keys=UNCERTAINTY_RESULT_KEYS,
        schema_name='PPAL uncertainty',
    )


def load_class_quality(checkpoint_path: Any) -> np.ndarray:
    """Load ``bbox_head.class_quality`` from a PPAL detector checkpoint."""

    ckpt_path = _as_path(checkpoint_path)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            'PPAL class quality loading requires PyTorch because %s is a '
            'MMDetection checkpoint.' % ckpt_path) from exc

    checkpoint = torch.load(str(ckpt_path), map_location='cpu')
    state_dict = checkpoint.get('state_dict')
    if not isinstance(state_dict, dict) or CLASS_QUALITY_KEY not in state_dict:
        raise KeyError('Checkpoint is missing state_dict[%r]: %s' % (CLASS_QUALITY_KEY, ckpt_path))

    value = state_dict[CLASS_QUALITY_KEY]
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    qualities = np.asarray(value, dtype=np.float64).reshape(-1)
    if qualities.size == 0:
        raise ValueError('PPAL class quality is empty: %s' % ckpt_path)
    if not np.all(np.isfinite(qualities)):
        raise ValueError('PPAL class quality contains non-finite values: %s' % ckpt_path)
    return qualities

