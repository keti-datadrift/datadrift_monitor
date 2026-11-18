"""COCO 2017 active-learning data preparation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from methods.common.io import read_json
from tools.common.dataset_pools import ensure_seeded_initial_splits


def _resolve_path(value: object, root: Path) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def _configured_path(
    cfg: Mapping[str, object],
    key: str,
    default: Path,
    root: Path,
) -> Path:
    value = cfg.get(key)
    return _resolve_path(value, root) if value is not None else default.resolve()


def _require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError('%s directory does not exist: %s' % (label, path))


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError('%s file does not exist: %s' % (label, path))


def _validate_coco_structure(payload: Any, path: Path) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError('COCO annotation root must be an object: %s' % path)
    for key in ('images', 'annotations', 'categories'):
        if key not in payload:
            raise ValueError('COCO annotation is missing %s: %s' % (key, path))
        if not isinstance(payload[key], list):
            raise ValueError('COCO annotation %s must be a list: %s' % (key, path))

    for index, image in enumerate(payload['images']):
        if not isinstance(image, dict) or 'id' not in image or 'file_name' not in image:
            raise ValueError(
                'COCO image entry %d must contain id and file_name: %s'
                % (index, path)
            )
    for index, annotation in enumerate(payload['annotations']):
        if not isinstance(annotation, dict):
            raise ValueError('COCO annotation entry %d is not an object: %s'
                             % (index, path))
        if 'image_id' not in annotation or 'category_id' not in annotation:
            raise ValueError(
                'COCO annotation entry %d must contain image_id and category_id: %s'
                % (index, path)
            )
    for index, category in enumerate(payload['categories']):
        if not isinstance(category, dict) or 'id' not in category:
            raise ValueError('COCO category entry %d must contain id: %s'
                             % (index, path))
    return payload


def ensure_coco_active_learning(
    coco_cfg: Mapping[str, object],
    root: Path,
    seeds: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    """Validate local COCO 2017 data and create deterministic seed pools."""

    data_root_value = coco_cfg.get(
        'data_root',
        coco_cfg.get('coco_root', 'data/coco'),
    )
    data_root = _resolve_path(data_root_value, root)
    train_image_dir = _configured_path(
        coco_cfg, 'train_image_dir', data_root / 'train2017', root)
    val_image_dir = _configured_path(
        coco_cfg, 'val_image_dir', data_root / 'val2017', root)
    train_annotations = _configured_path(
        coco_cfg,
        'train_annotations',
        data_root / 'annotations' / 'instances_train2017.json',
        root,
    )
    if coco_cfg.get('oracle_path') is not None:
        train_annotations = _resolve_path(coco_cfg['oracle_path'], root)
    val_annotations = _configured_path(
        coco_cfg,
        'val_annotations',
        data_root / 'annotations' / 'instances_val2017.json',
        root,
    )
    split_output_dir = _configured_path(
        coco_cfg,
        'split_output_dir',
        Path(root) / 'data' / 'active_learning' / 'coco',
        root,
    )

    _require_directory(train_image_dir, 'COCO train2017')
    _require_directory(val_image_dir, 'COCO val2017')
    _require_file(train_annotations, 'COCO train annotation')
    _require_file(val_annotations, 'COCO validation annotation')

    n_labeled = int(coco_cfg.get('n_labeled', 2365))
    n_images = (
        int(coco_cfg['n_images'])
        if coco_cfg.get('n_images') is not None else None
    )
    dataset_prefix = str(coco_cfg.get('dataset_prefix', 'coco'))
    oracle = _validate_coco_structure(
        read_json(train_annotations),
        train_annotations,
    )
    if n_images is not None and len(oracle['images']) != n_images:
        raise ValueError(
            'COCO train annotation must contain %d images, found %d: %s'
            % (n_images, len(oracle['images']), train_annotations)
        )

    split_paths = ensure_seeded_initial_splits(
        oracle,
        output_dir=split_output_dir,
        dataset_prefix=dataset_prefix,
        n_labeled=n_labeled,
        seeds=[0] if seeds is None else seeds,
    )
    pools_created = any(
        record.get('status') == 'created' for record in split_paths
    )

    return {
        'component': 'dataset',
        'type': 'coco2017',
        'status': 'ready',
        'action': 'created' if pools_created else 'kept',
        'oracle_path': str(train_annotations),
        'validation_path': str(val_annotations),
        'split_paths': split_paths,
    }
