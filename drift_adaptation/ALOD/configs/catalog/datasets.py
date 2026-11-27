"""Dataset specifications exposed by the ALOD catalog."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    aliases: Tuple[str, ...]
    dataset_type: str
    oracle_path: str
    init_label_json: str
    init_unlabeled_json: str
    image_root: str
    test_ann_file: str
    test_img_prefix: str
    round_num: int
    budget: int
    initial_percent: int
    final_percent: int
    eval_metric: str
    summary_metrics: Tuple[str, ...]
    init_label_json_template: Optional[str] = None
    init_unlabeled_json_template: Optional[str] = None
    dataset_prep: Dict[str, object] = field(default_factory=dict)
    mmdet_common_cfg_options: Dict[str, object] = field(default_factory=dict)
    mmdet_eval_cfg_options: Dict[str, object] = field(default_factory=dict)
    skip_terminal_acquisition: bool = False
    max_consecutive_nonfinite_grad_norm: Optional[int] = None

    def __post_init__(self) -> None:
        label_template = self.init_label_json_template
        unlabeled_template = self.init_unlabeled_json_template
        if label_template is None or unlabeled_template is None:
            raise ValueError(
                'Catalog datasets must define labeled and unlabeled seed templates'
            )
        for template in (label_template, unlabeled_template):
            if template is not None and '{seed}' not in template:
                raise ValueError(
                    'Initial-pool template must contain {seed}: %s' % template
                )

    @property
    def protocol_slug(self) -> str:
        return '%drounds_%dpercent_to_%dpercent' % (
            self.round_num,
            self.initial_percent,
            self.final_percent,
        )

    def to_config(self) -> Dict[str, object]:
        common_options = {
            'data.train.img_prefix': self.image_root,
            'data.val.ann_file': self.test_ann_file,
            'data.val.img_prefix': self.test_img_prefix,
            'data.test.img_prefix': self.image_root,
        }
        common_options.update(self.mmdet_common_cfg_options)

        eval_options = {
            'data.test.ann_file': self.test_ann_file,
            'data.test.img_prefix': self.test_img_prefix,
        }
        eval_options.update(self.mmdet_eval_cfg_options)

        cfg = {
            'dataset_name': self.name,
            'oracle_path': self.oracle_path,
            'init_label_json': self.init_label_json,
            'init_unlabeled_json': self.init_unlabeled_json,
            'init_model': None,
            'image_root': self.image_root,
            'round_num': self.round_num,
            'budget': self.budget,
            'eval_metric': self.eval_metric,
            'summary_metrics': self.summary_metrics,
            'protocol_slug': self.protocol_slug,
            'mmdet_common_cfg_options': common_options,
            'mmdet_eval_cfg_options': eval_options,
        }
        if self.init_label_json_template:
            cfg['init_label_json_template'] = self.init_label_json_template
        if self.init_unlabeled_json_template:
            cfg['init_unlabeled_json_template'] = self.init_unlabeled_json_template
        if self.dataset_prep:
            cfg['dataset_prep'] = dict(self.dataset_prep)
        if self.skip_terminal_acquisition:
            cfg['skip_terminal_acquisition'] = True
        if self.max_consecutive_nonfinite_grad_norm is not None:
            cfg['max_consecutive_nonfinite_grad_norm'] = int(
                self.max_consecutive_nonfinite_grad_norm
            )
        return cfg


VOC = DatasetSpec(
    name='voc',
    aliases=('voc', 'pascal-voc', 'pascal_voc'),
    dataset_type='voc',
    oracle_path='data/VOC0712/annotations/trainval_0712.json',
    init_label_json='data/active_learning/voc/voc_827_labeled_seed_0.json',
    init_unlabeled_json='data/active_learning/voc/voc_827_unlabeled_seed_0.json',
    init_label_json_template='data/active_learning/voc/voc_827_labeled_seed_{seed}.json',
    init_unlabeled_json_template='data/active_learning/voc/voc_827_unlabeled_seed_{seed}.json',
    image_root='data/VOCdevkit/',
    test_ann_file='data/VOCdevkit/VOC2007/ImageSets/Main/test.txt',
    test_img_prefix='data/VOCdevkit/VOC2007/',
    round_num=7,
    budget=414,
    initial_percent=5,
    final_percent=20,
    eval_metric='mAP',
    summary_metrics=('mAP', 'AP50'),
    mmdet_eval_cfg_options=dict(fp16=None),
    skip_terminal_acquisition=True,
    max_consecutive_nonfinite_grad_norm=20,
    dataset_prep=dict(
        type='voc0712',
        vocdevkit='data/VOCdevkit',
        oracle_output='data/VOC0712/annotations/trainval_0712.json',
        split_output_dir='data/active_learning/voc',
        n_labeled=827,
        split='trainval',
        dataset_prefix='voc',
    ),
)


COCO = DatasetSpec(
    name='coco',
    aliases=('coco', 'ms-coco', 'mscoco'),
    dataset_type='coco',
    oracle_path='data/coco/annotations/instances_train2017.json',
    init_label_json='data/active_learning/coco/coco_2365_labeled_seed_0.json',
    init_unlabeled_json='data/active_learning/coco/coco_2365_unlabeled_seed_0.json',
    init_label_json_template='data/active_learning/coco/coco_2365_labeled_seed_{seed}.json',
    init_unlabeled_json_template='data/active_learning/coco/coco_2365_unlabeled_seed_{seed}.json',
    image_root='data/coco/train2017/',
    test_ann_file='data/coco/annotations/instances_val2017.json',
    test_img_prefix='data/coco/val2017/',
    round_num=5,
    budget=2365,
    initial_percent=2,
    final_percent=10,
    eval_metric='bbox',
    summary_metrics=('bbox_mAP', 'bbox_mAP_50', 'bbox_mAP_75'),
    dataset_prep=dict(
        type='coco2017',
        train_image_dir='data/coco/train2017',
        val_image_dir='data/coco/val2017',
        oracle_path='data/coco/annotations/instances_train2017.json',
        val_annotations='data/coco/annotations/instances_val2017.json',
        split_output_dir='data/active_learning/coco',
        n_labeled=2365,
        n_images=118287,
        dataset_prefix='coco',
    ),
)


def normalize_token(value: str) -> str:
    return value.strip().lower().replace('_', '-')


def list_datasets() -> List[DatasetSpec]:
    return [VOC, COCO]


def resolve_dataset(name: str) -> Optional[DatasetSpec]:
    normalized = normalize_token(name)
    for dataset in list_datasets():
        aliases = {normalize_token(value) for value in dataset.aliases}
        if normalized == dataset.name or normalized in aliases:
            return dataset
    return None
