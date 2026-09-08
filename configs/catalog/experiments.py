"""Supported active-learning experiment catalog."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .datasets import DatasetSpec, list_datasets, resolve_dataset
from .detectors import DetectorSpec, list_detectors, resolve_detector
from .methods import MethodSpec
from .methods import list_methods, normalize_method_alias, resolve_method_alias as _resolve_method_alias
from .methods import resolve_method_spec


DATASET_OWNED_CONFIG_KEYS = frozenset({
    'dataset_name',
    'oracle_path',
    'init_label_json',
    'init_unlabeled_json',
    'init_label_json_template',
    'init_unlabeled_json_template',
    'dataset_prep',
    'image_root',
})


def validate_no_dataset_owned_overrides(
    fragment: Dict[str, object],
    fragment_name: str,
) -> None:
    """Reject non-dataset catalog fragments that redefine dataset inputs."""

    overridden = sorted(DATASET_OWNED_CONFIG_KEYS.intersection(fragment))
    if overridden:
        raise ValueError(
            '%s cannot override dataset-owned config key(s): %s'
            % (fragment_name, ', '.join(overridden))
        )


@dataclass(frozen=True)
class ExperimentPreset:
    name: str
    method_key: str
    method: str
    detector: str
    dataset: str
    description: str
    aliases: Tuple[str, ...] = ()
    cfg_overrides: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CatalogSelection:
    preset: ExperimentPreset
    method: str
    method_alias: str
    cfg_overrides: Dict[str, object] = field(default_factory=dict)
    method_spec: Optional[MethodSpec] = None
    detector_spec: Optional[DetectorSpec] = None
    dataset_spec: Optional[DatasetSpec] = None


_PRESET_TOKENS = {
    'ppal': 'ppal',
    'pal_full': 'pal',
    'pal_lius': 'pal-lius',
    'epal': 'epal',
    'coreset': 'coreset',
    'mial': 'mial',
    'random': 'random',
    'entropy': 'entropy',
}


def _build_preset(method: MethodSpec, detector: DetectorSpec, dataset: DatasetSpec) -> ExperimentPreset:
    method_description = method.description.rstrip('.')
    return ExperimentPreset(
        name='%s-%s-%s' % (
            _PRESET_TOKENS[method.key],
            detector.name,
            dataset.name,
        ),
        method_key=method.key,
        method=method.method,
        detector=detector.name,
        dataset=dataset.name,
        description='%s on %s/%s.' % (method_description, detector.name, dataset.name.upper()),
        aliases=method.aliases,
        cfg_overrides=dict(method.cfg_overrides),
    )


def list_presets() -> List[ExperimentPreset]:
    presets = []
    for detector in list_detectors():
        for dataset in list_datasets():
            for method in list_methods():
                presets.append(_build_preset(method, detector, dataset))
    return presets


def resolve_method_alias(method: str) -> Optional[Tuple[str, Dict[str, object]]]:
    return _resolve_method_alias(method)


def _preset_names(preset: ExperimentPreset) -> Iterable[str]:
    yield preset.name
    for alias in preset.aliases:
        yield alias


def _find_preset_by_name(name: str) -> Optional[ExperimentPreset]:
    normalized = normalize_method_alias(name)
    for preset in list_presets():
        if normalized in {normalize_method_alias(value) for value in _preset_names(preset)}:
            return preset
    return None


def _find_preset_by_combo(method: str, detector: str, dataset: str) -> Optional[ExperimentPreset]:
    method_spec = resolve_method_spec(method)
    detector_spec = resolve_detector(detector)
    dataset_spec = resolve_dataset(dataset)
    if method_spec is None or detector_spec is None or dataset_spec is None:
        return None
    return _build_preset(method_spec, detector_spec, dataset_spec)


def _selection_from_preset(preset: ExperimentPreset, method_alias: Optional[str] = None) -> Optional[CatalogSelection]:
    method_spec = resolve_method_spec(method_alias or preset.method_key)
    detector_spec = resolve_detector(preset.detector)
    dataset_spec = resolve_dataset(preset.dataset)
    if method_spec is None or detector_spec is None or dataset_spec is None:
        return None
    return CatalogSelection(
        preset=preset,
        method=method_spec.method,
        method_alias=normalize_method_alias(method_alias or method_spec.default_alias()),
        cfg_overrides=dict(method_spec.cfg_overrides),
        method_spec=method_spec,
        detector_spec=detector_spec,
        dataset_spec=dataset_spec,
    )


def resolve_experiment(
    method: Optional[str] = None,
    detector: Optional[str] = None,
    dataset: Optional[str] = None,
    preset: Optional[str] = None,
) -> Optional[CatalogSelection]:
    """Resolve a user-facing selection to an experiment config and base method."""

    if preset:
        matched = _find_preset_by_name(preset)
        if matched is None:
            return None
        return _selection_from_preset(matched, method_alias=method)

    if not method or not detector or not dataset:
        return None

    matched = _find_preset_by_combo(method, detector, dataset)
    if matched is None:
        return None

    return _selection_from_preset(matched, method_alias=method)


def build_experiment_config(selection: CatalogSelection) -> Dict[str, object]:
    """Build runner config from catalog specs instead of experiment globals."""

    if selection.method_spec is None or selection.detector_spec is None or selection.dataset_spec is None:
        raise ValueError('Catalog selection is missing resolved specs.')

    gpus = 8
    cfg: Dict[str, object] = {
        'python_path': 'python',
        'port': 29500,
        'gpus': gpus,
    }
    dataset_cfg = selection.dataset_spec.to_config()
    detector_cfg = selection.detector_spec.to_config(
        selection.dataset_spec.name,
        selection.method_spec.key,
    )
    method_cfg = selection.method_spec.to_config(
        selection.dataset_spec,
        selection.detector_spec.name,
        gpus=gpus,
    )
    validate_no_dataset_owned_overrides(
        detector_cfg,
        'Detector catalog fragment %r' % selection.detector_spec.name,
    )
    validate_no_dataset_owned_overrides(
        method_cfg,
        'Method catalog fragment %r' % selection.method_spec.key,
    )
    validate_no_dataset_owned_overrides(
        selection.cfg_overrides,
        'Method catalog overrides %r' % selection.method_alias,
    )
    cfg.update(dataset_cfg)
    cfg.update(detector_cfg)
    cfg.update(method_cfg)
    cfg.update(selection.cfg_overrides)
    return cfg
