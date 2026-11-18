"""PPAL acquisition orchestration using local method modules."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, Iterable, List

from methods.common.candidates import (
    build_candidate_artifact,
    write_candidate_artifact,
)
from methods.common.coco_pool import (
    write_candidate_pool_from_selection,
    write_next_round_pool_split,
)
from methods.common.results import acquisition_result


@dataclass(frozen=True)
class PPALStep:
    name: str
    description: str


PPAL_STEPS = (
    PPALStep('train', 'Train RetinaNet with the labeled pool.'),
    PPALStep('eval', 'Evaluate the current checkpoint on the configured validation data.'),
    PPALStep('uncertainty_inference', 'Run RetinaHeadUncertainty on unlabeled data.'),
    PPALStep('dcus_acquisition', 'Select the expanded uncertainty pool with DCUS.'),
    PPALStep('feature_inference', 'Export detector features for the uncertainty pool.'),
    PPALStep('diversity_acquisition', 'Select the final budget with CCMS.'),
)

SAMPLER_TYPES = {
    'DCUSSampler': 'dcus',
    'DCUS': 'dcus',
    'DiversitySampler': 'ccms',
    'CCMS': 'ccms',
}


def describe_steps() -> List[Dict[str, str]]:
    return [dict(name=step.name, description=step.description) for step in PPAL_STEPS]


def sampler_config_names() -> Iterable[str]:
    return ('uncertainty_sampler_config', 'diversity_sampler_config')


def local_ppal_available() -> bool:
    return find_spec('methods.ppal') is not None


def _resolve_sampler_config(sampler_config: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    resolved = dict(sampler_config)
    oracle_path = resolved.get('oracle_annotation_path')
    if oracle_path:
        oracle = Path(str(oracle_path))
        if not oracle.is_absolute():
            resolved['oracle_annotation_path'] = str((repo_root / oracle).resolve())
    return resolved


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError('%s does not exist: %s' % (description, path))


def _round_artifact_path(anchor: Path, filename: str) -> Path:
    return anchor.parent.parent / filename


def build_local_sampler(sampler_config: Dict[str, Any], repo_root: Path) -> Any:
    """Build a PPAL sampler from local method modules."""

    resolved = _resolve_sampler_config(sampler_config, repo_root)
    sampler_type = resolved.pop('type', None)
    if sampler_type not in SAMPLER_TYPES:
        raise KeyError('Unsupported PPAL sampler type: %s' % sampler_type)
    if SAMPLER_TYPES[sampler_type] == 'dcus':
        from methods.ppal.dcus import DCUSSampler

        return DCUSSampler(**resolved)
    if SAMPLER_TYPES[sampler_type] == 'ccms':
        from methods.ppal.ccms import DiversitySampler

        return DiversitySampler(**resolved)
    raise KeyError('Unsupported PPAL sampler type: %s' % sampler_type)


def run_uncertainty_acquisition(
    cfg: Dict[str, Any],
    repo_root: Path,
    round_index: int,
    result_json: Path,
    last_labeled_json: Path,
    out_candidate_json: Path,
) -> Dict[str, Any]:
    """Run PPAL DCUS acquisition with the local method implementation."""

    _require_file(result_json, 'PPAL uncertainty inference result')
    _require_file(last_labeled_json, 'PPAL previous labeled pool')
    _require_file(result_json.parent / 'latest.pth', 'PPAL checkpoint for class quality')
    out_candidate_json.parent.mkdir(parents=True, exist_ok=True)

    sampler = build_local_sampler(cfg['uncertainty_sampler_config'], repo_root)
    if hasattr(sampler, 'set_round'):
        sampler.set_round(round_index)
    result = sampler.al_round(
        str(result_json),
        str(last_labeled_json),
    )
    selected = result.get('selected_image_ids', [])
    candidate_ids, remainder_ids = write_candidate_pool_from_selection(
        sampler.oracle_json,
        last_labeled_json,
        selected,
        out_candidate_json,
        candidate_include_annotations=False,
    )
    metrics = dict(result.get('metrics', {}))
    metrics.update({
        'uncertainty_pool_count': len(candidate_ids),
        'remaining_unlabeled_count': len(remainder_ids),
    })
    out_candidate = str(out_candidate_json)
    candidates_json = _round_artifact_path(out_candidate_json, 'ppal_dcus_candidates.json')
    candidate_artifact = build_candidate_artifact(
        method='ppal',
        stage='dcus',
        round_index=round_index,
        budget=int(cfg['uncertainty_sampler_config'].get('n_sample_images', 0)),
        candidates=result.get('candidate_records', []),
        selected_image_ids=selected,
        candidate_pool_json=out_candidate,
    )
    write_candidate_artifact(candidates_json, candidate_artifact)
    return acquisition_result(
        method='ppal',
        stage='dcus',
        runner_stage='uncertainty',
        round_index=round_index,
        budget=int(cfg['uncertainty_sampler_config'].get('n_sample_images', 0)),
        selected_image_ids=selected,
        inputs={
            'labeled_pool_json': str(last_labeled_json),
            'uncertainty_result_json': str(result_json),
            'class_quality_checkpoint': str(result_json.parent / 'latest.pth'),
        },
        outputs={
            'candidate_pool_json': out_candidate,
            'candidates_json': str(candidates_json),
        },
        metrics=metrics,
    )


def run_diversity_acquisition(
    cfg: Dict[str, Any],
    repo_root: Path,
    round_index: int,
    result_json: Path,
    feature_npz: Path,
    last_labeled_json: Path,
    out_labeled_json: Path,
    out_unlabeled_json: Path,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run PPAL diversity acquisition with the local method implementation."""

    _require_file(result_json, 'PPAL uncertainty inference result')
    _require_file(feature_npz, 'PPAL diversity feature artifact')
    _require_file(last_labeled_json, 'PPAL previous labeled pool')
    out_labeled_json.parent.mkdir(parents=True, exist_ok=True)

    sampler_config = dict(cfg['diversity_sampler_config'])
    sampler_config['seed'] = seed
    sampler = build_local_sampler(sampler_config, repo_root)
    if hasattr(sampler, 'set_round'):
        sampler.set_round(round_index)
    result = sampler.al_round(
        str(feature_npz),
        str(last_labeled_json),
    )
    selected = result.get('selected_image_ids', [])
    labeled_ids, unlabeled_ids = write_next_round_pool_split(
        sampler.oracle_json,
        last_labeled_json,
        selected,
        out_labeled_json,
        out_unlabeled_json,
    )
    metrics = dict(result.get('metrics', {}))
    metrics.update({
        'new_labeled_count': len(labeled_ids),
        'new_unlabeled_count': len(unlabeled_ids),
    })
    out_labeled = str(out_labeled_json)
    out_unlabeled = str(out_unlabeled_json)
    candidate_pool_json = out_labeled_json.parent / 'uncertainty_pool.json'
    candidates_json = _round_artifact_path(out_labeled_json, 'ppal_ccms_candidates.json')
    candidate_artifact = build_candidate_artifact(
        method='ppal',
        stage='ccms',
        round_index=round_index,
        budget=int(cfg['diversity_sampler_config'].get('n_sample_images', 0)),
        candidates=result.get('candidate_records', []),
        selected_image_ids=selected,
        candidate_pool_json=str(candidate_pool_json),
    )
    write_candidate_artifact(candidates_json, candidate_artifact)
    return acquisition_result(
        method='ppal',
        stage='ccms',
        runner_stage='diversity',
        round_index=round_index,
        budget=int(cfg['diversity_sampler_config'].get('n_sample_images', 0)),
        selected_image_ids=selected,
        inputs={
            'labeled_pool_json': str(last_labeled_json),
            'uncertainty_result_json': str(result_json),
            'feature_npz': str(feature_npz),
        },
        outputs={
            'labeled_pool_json': out_labeled,
            'unlabeled_pool_json': out_unlabeled,
            'candidates_json': str(candidates_json),
        },
        metrics=metrics,
    )
