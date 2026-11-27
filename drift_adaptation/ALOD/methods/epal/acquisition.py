"""EPAL acquisition using the selected error-entropy components."""

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from methods.common.coco_pool import image_ids, read_coco_json
from methods.epal.components import DEFAULT_COMPONENTS, normalize_components
from methods.epal.features import FEATURE_SCHEMA, FEATURE_SCHEMA_VERSION
from methods.epal.inference import filter_feature_records, load_feature_records
from methods.epal.labels import build_training_labels
from methods.epal.predictors import fit_predictors
from methods.epal.scoring import inverse_scale_weights, mean_eua_profile, score_unlabeled_records


def select_epal_images(
    labeled_pool: Mapping[str, Any],
    unlabeled_pool: Mapping[str, Any],
    labeled_features: List[Mapping[str, Any]],
    unlabeled_features: List[Mapping[str, Any]],
    budget: int,
    components: str = DEFAULT_COMPONENTS,
    foreground_iou_threshold: float = 0.5,
    background_iou_threshold: float = 0.1,
    weight_eps: float = 1e-6,
) -> Dict[str, Any]:
    components = normalize_components(components)
    if budget < 0:
        raise ValueError('EPAL budget must be nonnegative')
    if not 0 <= background_iou_threshold <= foreground_iou_threshold <= 1:
        raise ValueError('EPAL requires 0 <= background IoU <= foreground IoU <= 1')
    labeled_ids = image_ids(dict(labeled_pool))
    unlabeled_ids = image_ids(dict(unlabeled_pool))
    labeled_records = filter_feature_records(
        labeled_features, labeled_ids, artifact_name='EPAL labeled features', require_all=True)
    unlabeled_records = filter_feature_records(
        unlabeled_features, unlabeled_ids, artifact_name='EPAL unlabeled features', require_all=True)
    label_data = build_training_labels(
        labeled_records, labeled_pool,
        foreground_iou_threshold=foreground_iou_threshold,
        background_iou_threshold=background_iou_threshold)
    predictors = fit_predictors(label_data, components=components)
    means = mean_eua_profile(labeled_records, predictors)
    weights = inverse_scale_weights(means, weight_eps=weight_eps)
    scored = score_unlabeled_records(unlabeled_records, predictors, weights)
    selected = [item['image_id'] for item in scored[:budget]]
    candidates = [
        {
            'image_id': item['image_id'],
            'rank': rank,
            'score': item['score'],
            'source': 'epal',
            'components': dict(item['profile'], weighted_eua=item['score']),
            'weighted_profile': item['weighted_profile'],
            'metadata': {
                'candidate_rank': rank,
                'selected': True,
                'final_detection_count': item['final_detection_count'],
            },
        }
        for rank, item in enumerate(scored[:budget], start=1)
    ]
    diagnostics = {
        'mode': 'epal',
        'stage': 'eua',
        'epal_components': components,
        'feature_schema': FEATURE_SCHEMA,
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'budget': int(budget),
        'thresholds': {
            'foreground_iou': float(foreground_iou_threshold),
            'background_iou': float(background_iou_threshold),
        },
        'weight_eps': float(weight_eps),
        'pool_counts': {
            'labeled_images': len(labeled_ids),
            'unlabeled_images': len(unlabeled_ids),
            'labeled_feature_records': len(labeled_records),
            'unlabeled_feature_records': len(unlabeled_records),
        },
        'label_summary': label_data['summary'],
        'predictors': predictors.diagnostics,
        'mean_labeled_uncertainties': means,
        'scale_basis': 'labeled_predictive_uncertainties',
        'scale_weights': weights,
        'scored_image_count': len(scored),
        'candidate_count': len(candidates),
        'selected_count': len(selected),
        'selected_image_ids': selected,
    }
    return {
        'selected_image_ids': selected,
        'candidate_records': candidates,
        'diagnostics': diagnostics,
        'mode': 'epal',
        'stage': 'eua',
    }


def sample_epal_from_files(
    labeled_pool_json: Path,
    unlabeled_pool_json: Path,
    labeled_features_json: Path,
    unlabeled_features_json: Path,
    budget: int,
    components: str = DEFAULT_COMPONENTS,
    foreground_iou_threshold: float = 0.5,
    background_iou_threshold: float = 0.1,
    weight_eps: float = 1e-6,
    oracle_json: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fit on labeled data and score the pool; oracle is never used here."""
    del oracle_json
    result = select_epal_images(
        labeled_pool=read_coco_json(Path(labeled_pool_json)),
        unlabeled_pool=read_coco_json(Path(unlabeled_pool_json)),
        labeled_features=load_feature_records(Path(labeled_features_json)),
        unlabeled_features=load_feature_records(Path(unlabeled_features_json)),
        budget=budget, components=components,
        foreground_iou_threshold=foreground_iou_threshold,
        background_iou_threshold=background_iou_threshold,
        weight_eps=weight_eps)
    result['diagnostics']['inputs'] = {
        'labeled_pool_json': str(labeled_pool_json),
        'unlabeled_pool_json': str(unlabeled_pool_json),
        'labeled_features_json': str(labeled_features_json),
        'unlabeled_features_json': str(unlabeled_features_json),
    }
    return result
