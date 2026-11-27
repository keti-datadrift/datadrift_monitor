"""Automatic input preparation for the active-learning runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from tools.common.coco import ensure_coco_active_learning
from tools.common.pal_embeddings import ensure_pal_embeddings
from tools.common.paths import assert_not_code_refs
from tools.common.pretrained import ensure_pretrained
from tools.common.voc import ensure_voc_active_learning


def _resolve_path(value: object, root: Path) -> Path:
    path = Path(str(value))
    resolved = (Path(root) / path).resolve() if not path.is_absolute() else path.resolve()
    assert_not_code_refs(resolved, root)
    return resolved


def _ensure_dataset(
    cfg: Mapping[str, Any],
    root: Path,
    seeds: Optional[Sequence[int]] = None,
) -> Optional[Dict[str, object]]:
    dataset_prep = cfg.get('dataset_prep')
    if not isinstance(dataset_prep, dict):
        return None
    kind = str(dataset_prep.get('type', '')).lower()
    if kind == 'voc0712':
        return ensure_voc_active_learning(dataset_prep, root, seeds=seeds)
    if kind == 'coco2017':
        return ensure_coco_active_learning(dataset_prep, root, seeds=seeds)
    raise ValueError('Unsupported dataset preparation type: %s'
                     % dataset_prep.get('type'))


def _ensure_pretrained(cfg: Mapping[str, Any], root: Path) -> Optional[Dict[str, object]]:
    pretrained = cfg.get('pretrained')
    if not isinstance(pretrained, dict):
        return None
    return ensure_pretrained(pretrained, root)


def _ensure_pal_embeddings(cfg: Mapping[str, Any], root: Path) -> Optional[Dict[str, object]]:
    embedding_prep = cfg.get('pal_embedding_prep')
    if not isinstance(embedding_prep, dict):
        return None
    if str(cfg.get('pal_embedding_source', '')).lower() != 'external':
        return None
    if str(cfg.get('pal_mode', '')).lower() not in ('full', 'guide'):
        return None

    oracle_path = cfg.get('oracle_path')
    if not oracle_path:
        raise ValueError('PAL embedding preparation requires oracle_path')
    ann_paths = [_resolve_path(oracle_path, root)]
    image_root = _resolve_path(cfg.get('image_root', 'data/VOCdevkit'), root)
    for path in ann_paths + [image_root]:
        assert_not_code_refs(path, root)
    return ensure_pal_embeddings(embedding_prep, ann_paths, image_root, root)


def prepare_required_inputs(
    cfg: Mapping[str, Any],
    root: Path,
    seeds: Optional[Sequence[int]] = None,
) -> List[Dict[str, object]]:
    """Prepare all catalog-declared inputs needed before AL rounds start."""

    results: List[Dict[str, object]] = []
    dataset_result = _ensure_dataset(cfg, root, seeds=seeds)
    if dataset_result is not None:
        results.append(dataset_result)
    for step in (_ensure_pretrained, _ensure_pal_embeddings):
        result = step(cfg, root)
        if result is not None:
            results.append(result)
    return results
