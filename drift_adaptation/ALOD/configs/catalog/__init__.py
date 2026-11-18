"""User-facing experiment catalog for ALOD."""

from .datasets import DatasetSpec, list_datasets, resolve_dataset
from .detectors import DetectorSpec, list_detectors, resolve_detector
from .experiments import CatalogSelection, ExperimentPreset, build_experiment_config
from .experiments import list_presets, resolve_experiment, resolve_method_alias
from .methods import MethodSpec, list_methods, resolve_method_spec

__all__ = [
    'CatalogSelection',
    'DatasetSpec',
    'DetectorSpec',
    'ExperimentPreset',
    'MethodSpec',
    'build_experiment_config',
    'list_datasets',
    'list_detectors',
    'list_methods',
    'list_presets',
    'resolve_dataset',
    'resolve_detector',
    'resolve_experiment',
    'resolve_method_alias',
    'resolve_method_spec',
]
