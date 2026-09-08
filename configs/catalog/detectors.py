"""Detector specifications exposed by the ALOD catalog."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .datasets import normalize_token


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    aliases: Tuple[str, ...]
    model_name: str
    configs_by_dataset: Dict[str, Dict[str, str]]
    pretrained: Dict[str, object] = field(default_factory=dict)

    def to_config(self, dataset_name: str, method_key: str) -> Dict[str, object]:
        dataset_configs = self.configs_by_dataset.get(dataset_name)
        if dataset_configs is None:
            raise ValueError(
                'Detector %s does not support dataset %s'
                % (self.name, dataset_name)
            )
        cfg = dict(dataset_configs)
        mial_train_config = cfg.pop('mial_train_config')
        if method_key == 'mial':
            cfg['train_config'] = mial_train_config
        cfg.update({
            'model_name': self.model_name,
            'pretrained': dict(self.pretrained),
        })
        return cfg


RETINANET = DetectorSpec(
    name='retinanet',
    aliases=('retinanet', 'retina-net', 'retina_net'),
    model_name='retinanet',
    configs_by_dataset={
        'voc': dict(
            train_config='configs/alod_mmdet/retinanet_voc_train_quality_ema_26e.py',
            uncertainty_infer_config='configs/alod_mmdet/retinanet_voc_infer_uncertainty.py',
            image_feature_infer_config='configs/alod_mmdet/retinanet_voc_infer_image_features.py',
            detection_feature_infer_config='configs/alod_mmdet/retinanet_voc_infer_detection_features.py',
            pal_infer_config='configs/alod_mmdet/retinanet_voc_infer_pal_detections.py',
            epal_infer_config='configs/alod_mmdet/retinanet_voc_infer_epal_detections.py',
            mial_train_config='configs/alod_mmdet/retinanet_voc_train_mial.py',
        ),
        'coco': dict(
            train_config='configs/alod_mmdet/retinanet_coco_train_quality_ema_26e.py',
            uncertainty_infer_config='configs/alod_mmdet/retinanet_coco_infer_uncertainty.py',
            image_feature_infer_config='configs/alod_mmdet/retinanet_coco_infer_image_features.py',
            detection_feature_infer_config='configs/alod_mmdet/retinanet_coco_infer_detection_features.py',
            pal_infer_config='configs/alod_mmdet/retinanet_coco_infer_pal_detections.py',
            epal_infer_config='configs/alod_mmdet/retinanet_coco_infer_epal_detections.py',
            mial_train_config='configs/alod_mmdet/retinanet_coco_train_mial.py',
        ),
    },
    pretrained=dict(
        type='resnet50',
        output_path='data/pretrain_models/resnet50-19c8e357.pth',
    ),
)


def list_detectors() -> List[DetectorSpec]:
    return [RETINANET]


def resolve_detector(name: str) -> Optional[DetectorSpec]:
    normalized = normalize_token(name)
    for detector in list_detectors():
        aliases = {normalize_token(value) for value in detector.aliases}
        if normalized == detector.name or normalized in aliases:
            return detector
    return None
