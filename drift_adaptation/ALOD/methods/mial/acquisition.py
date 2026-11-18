"""MIAL acquisition from MI-AOD instance-discrepancy uncertainty."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from methods.common.coco_pool import image_ids, read_coco_json

from .inference import load_uncertainty_records
from .scoring import rank_mial_candidates


def select_mial_images(
    unlabeled_pool: Mapping[str, Any],
    uncertainty_records: Any,
    budget: int,
) -> Dict[str, Any]:
    """Select images with highest MIAL image-level uncertainty."""

    unlabeled_ids = image_ids(dict(unlabeled_pool))
    ranking = rank_mial_candidates(uncertainty_records, unlabeled_ids)
    selected = ranking['ranked_image_ids'][:max(int(budget), 0)]
    diagnostics = {
        'mode': 'mial',
        'stage': 'instance_discrepancy',
        'budget': int(budget),
        'pool_counts': {
            'unlabeled_images': len(unlabeled_ids),
            'uncertainty_records': len(uncertainty_records),
        },
        'score_summary': ranking['score_summary'],
        'selected_count': len(selected),
        'selected_image_ids': selected,
    }
    return {
        'selected_image_ids': selected,
        'candidate_records': ranking['candidate_records'],
        'diagnostics': diagnostics,
        'mode': 'mial',
        'stage': 'instance_discrepancy',
    }


def sample_mial_from_files(
    unlabeled_pool_json: Path,
    uncertainty_json: Path,
    budget: int,
) -> Dict[str, Any]:
    """Run MIAL acquisition from a COCO unlabeled pool and uncertainty JSON."""

    unlabeled_pool = read_coco_json(Path(unlabeled_pool_json))
    uncertainty_records = load_uncertainty_records(Path(uncertainty_json))
    result = select_mial_images(
        unlabeled_pool=unlabeled_pool,
        uncertainty_records=uncertainty_records,
        budget=budget,
    )
    result['diagnostics']['inputs'] = {
        'unlabeled_pool_json': str(unlabeled_pool_json),
        'uncertainty_json': str(uncertainty_json),
    }
    return result
