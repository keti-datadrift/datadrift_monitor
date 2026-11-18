from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml

from meta_models.commands.meta_classifier.train import run_train as run_classifier_train
from meta_models.commands.meta_regressor.train import run_train as run_regressor_train

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'documents/appendix/generated'
DEFAULT_CLASSIFIER_CONFIG = PROJECT_ROOT / 'meta_models/configs/meta_classifier/train.yaml'
DEFAULT_REGRESSOR_CONFIG = PROJECT_ROOT / 'meta_models/configs/meta_regressor/train.yaml'
MERGE_KEYS = ['image_id', 'image_path', 'raw_pred_idx']
META_COLUMNS = {'image_id', 'image_path', 'pred_idx', 'raw_pred_idx', 'xmin', 'ymin', 'xmax', 'ymax', 'score', 'pred_class', 'max_iou', 'gt_iou', 'tp'}

@dataclass(frozen=True)
class DatasetConfig:
    detector_key: str
    detector_name: str
    key: str
    display_name: str
    summary_name: str
    gt_root: Path
    unto_o_root: Path
    grad_grid_root: Path
    target_class_cols: tuple[str, ...]
    target_score_cols: tuple[str, ...]
    target_loc_cols: tuple[str, ...]
    grad_mode: str
    grad_components: dict[str, tuple[str, tuple[str, ...]]]

@dataclass
class EvalResult:
    detector: str
    detector_key: str
    dataset_key: str
    dataset: str
    type: str
    component: str
    num_rows: int
    num_features: int
    tp_ratio: float
    auroc_mean: float
    auroc_std: float
    ap_mean: float
    ap_std: float
    fpr95_mean: float
    fpr95_std: float
    r2_mean: float
    r2_std: float
    classifier_run_dir: str
    regressor_run_dir: str

DATASET_INFO = {
    'coco': ('MS COCO', 'MS COCO train2017'),
    'voc': ('Pascal VOC', 'Pascal VOC 2012 train'),
}
DETECTOR_PROFILES: dict[str, dict[str, Any]] = {
    'yolov5': {
        'name': 'YOLOv5',
        'target_class_cols': ('cls_loss',),
        'target_score_cols': ('obj_loss',),
        'target_loc_cols': ('x_loss', 'y_loss', 'w_loss', 'h_loss', 'size_diff', 'circum_diff', 'size_circum_diff'),
        'grad_mode': 'grid_suffix',
        'grad_components': {
            'Class': ('Classification gradient only', ('t-null__term-cls__c-kl-pred',)),
            'Score': ('Detection-score gradient only', ('t-null__term-obj__o-bce-pred',)),
            'Localization': ('Localization gradient only', ('t-null__term-bbox__b-box_l1-pred',)),
        },
    },
    'fcos': {
        'name': 'FCOS',
        'target_class_cols': ('cls_loss',),
        'target_score_cols': ('cnt_loss',),
        'target_loc_cols': ('x_loss', 'y_loss', 'w_loss', 'h_loss', 'size_diff', 'circum_diff', 'size_circum_diff'),
        'grad_mode': 'grid_suffix',
        'grad_components': {
            'Class': ('Classification gradient only', ('t-null__term-cls__c-kl-pred',)),
            'Score': ('Detection-score gradient only', ('t-null__term-cnt__cnt-bce-pred',)),
            'Localization': ('Localization gradient only', ('t-null__term-bbox__b-l1-pred',)),
        },
    },
    'faster_rcnn': {
        'name': 'Faster R-CNN',
        'target_class_cols': ('roi_cls_loss',),
        'target_score_cols': ('rpn_obj_loss',),
        'target_loc_cols': (
            'rpn_x_loss', 'rpn_y_loss', 'rpn_w_loss', 'rpn_h_loss',
            'rpn_size_diff', 'rpn_circum_diff', 'rpn_size_circum_diff',
            'roi_x_loss', 'roi_y_loss', 'roi_w_loss', 'roi_h_loss',
            'roi_size_diff', 'roi_circum_diff', 'roi_size_circum_diff',
        ),
        'grad_mode': 'single_file_prefix',
        'grad_components': {
            'Class': ('Classification gradient only', ('roi_cls_loss_',)),
            'Score': ('Detection-score gradient only', ('rpn_obj_loss_',)),
            'Localization': ('Localization gradient only', ('rpn_bbox_loss_', 'roi_bbox_loss_')),
        },
    },
}
DETECTOR_DATASET_ROOTS = {
    ('yolov5', 'coco'): (
        PROJECT_ROOT / 'object_detectors/runs/yolov5/predict/coco/06-15-2026_18;54_gt',
        PROJECT_ROOT / 'object_detectors/runs/yolov5/predict/coco/06-16-2026_04;47_null_detect',
        PROJECT_ROOT / 'object_detectors/runs/yolov5/predict/coco/06-16-2026_06;46_layer_grad_grid',
    ),
    ('yolov5', 'voc'): (
        PROJECT_ROOT / 'object_detectors/runs/yolov5/predict/voc/06-14-2026_13;36_gt',
        PROJECT_ROOT / 'object_detectors/runs/yolov5/predict/voc/06-14-2026_15;54_null_detect',
        PROJECT_ROOT / 'object_detectors/runs/yolov5/predict/voc/06-14-2026_17;09_layer_grad_grid',
    ),
    ('fcos', 'coco'): (
        PROJECT_ROOT / 'object_detectors/runs/fcos/predict/coco/06-17-2026_02;48_gt',
        PROJECT_ROOT / 'object_detectors/runs/fcos/predict/coco/06-18-2026_08;17_null_detect',
        PROJECT_ROOT / 'object_detectors/runs/fcos/predict/coco/06-18-2026_14;22_fcos_layer_grad_grid_v2',
    ),
    ('fcos', 'voc'): (
        PROJECT_ROOT / 'object_detectors/runs/fcos/predict/voc/06-15-2026_01;09_gt',
        PROJECT_ROOT / 'object_detectors/runs/fcos/predict/voc/06-15-2026_15;08_null_detect',
        PROJECT_ROOT / 'object_detectors/runs/fcos/predict/voc/06-15-2026_15;25_fcos_layer_grad_grid',
    ),
    ('faster_rcnn', 'coco'): (
        PROJECT_ROOT / 'object_detectors/runs/faster_rcnn/predict/coco/06-25-2026_12;29_gt',
        PROJECT_ROOT / 'object_detectors/runs/faster_rcnn/predict/coco/06-26-2026_23;13_null_detect',
        PROJECT_ROOT / 'object_detectors/runs/faster_rcnn/predict/coco/06-29-2026_14;07_layer_grad_null_target_loss',
    ),
    ('faster_rcnn', 'voc'): (
        PROJECT_ROOT / 'object_detectors/runs/faster_rcnn/predict/voc/07-03-2026_02;12_gt',
        PROJECT_ROOT / 'object_detectors/runs/faster_rcnn/predict/voc/07-03-2026_02;23_null_detect',
        PROJECT_ROOT / 'object_detectors/runs/faster_rcnn/predict/voc/07-03-2026_03;01_layer_grad_null_target_loss',
    ),
}
UNTO_O_COMPONENTS = [
    ('Final-detection', 'Class', 'Final class'),
    ('Final-detection', 'Score', 'Final score'),
    ('Final-detection', 'Localization', 'Final localization'),
    ('Final-detection', 'Full', 'Final-detection only'),
    ('Target-relative', 'Class', 'Target class'),
    ('Target-relative', 'Score', 'Target score'),
    ('Target-relative', 'Localization', 'Target localization'),
    ('Target-relative', 'Full', 'Target-relative only'),
    ('Full UnTO-O', 'Full', 'Full UnTO-O'),
]
UNTO_G_COMPONENTS = [
    ('Class', 'Classification gradient only'),
    ('Score', 'Detection-score gradient only'),
    ('Localization', 'Localization gradient only'),
    ('Full', 'All components'),
]
GRAD_LAYER_ROWS = [('YOLOv5', 'All components', 'model.24.m.0, model.24.m.1, model.24.m.2'), ('FCOS', 'Localization', 'detector_model.rpn.head.bbox_pred'), ('FCOS', 'Classification', 'detector_model.rpn.head.cls_logits'), ('FCOS', 'Detection score', 'detector_model.rpn.head.centerness'), ('Faster R-CNN', 'RPN objectness', 'rpn.head.conv, rpn.head.cls_logits'), ('Faster R-CNN', 'RPN box', 'rpn.head.conv, rpn.head.bbox_pred'), ('Faster R-CNN', 'RoI class', 'roi_heads.box_head.fc7, roi_heads.box_predictor.cls_score'), ('Faster R-CNN', 'RoI box', 'roi_heads.box_head.fc7, roi_heads.box_predictor.bbox_pred')]

def resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path

def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace('\\', '/')
    except ValueError:
        return str(path.resolve())

def load_yaml(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f'YAML config must contain a mapping: {path}')
    return data

def train_config(base: dict[str, Any], input_roots: Path | list[Path], gt_root: Path, *, regressor: bool) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    roots = input_roots if isinstance(input_roots, list) else [input_roots]
    cfg.setdefault('dataset', {})['input_root'] = [rel(root) for root in roots]
    cfg['dataset']['gt_root'] = rel(gt_root)
    if regressor:
        cfg.setdefault('model', {})['type'] = 'xgb_regressor'
    return cfg

def header(csv_path: Path) -> list[str]:
    return list(pd.read_csv(csv_path, nrows=0).columns)

def native_features(cols: list[str], cue: str) -> list[str]:
    meta = set(META_COLUMNS)
    if cue == 'score':
        meta.discard('score')
    return [col for col in cols if col not in meta]

def prob_cols(cols: list[str]) -> list[str]:
    out = []
    for col in cols:
        m = re.fullmatch(r'prob_(\d+)', col)
        if m:
            out.append((int(m.group(1)), col))
    return [col for _idx, col in sorted(out)]

def unto_o_feature_sets(cols: list[str], dataset: DatasetConfig) -> dict[str, list[str]]:
    final_class = prob_cols(cols) + (['prob_sum'] if 'prob_sum' in cols else [])
    final_score = ['final_score'] if 'final_score' in cols else []
    final_loc = [c for c in ['size', 'circum', 'size_circum'] if c in cols]
    target_class = [c for c in dataset.target_class_cols if c in cols]
    target_score = [c for c in dataset.target_score_cols if c in cols]
    target_loc = [c for c in dataset.target_loc_cols if c in cols]
    return {
        'Full UnTO-O': native_features(cols, 'null_detect'),
        'Final-detection only': final_class + final_score + final_loc,
        'Final class': final_class,
        'Final score': final_score,
        'Final localization': final_loc,
        'Target-relative only': target_class + target_score + target_loc,
        'Target class': target_class,
        'Target score': target_score,
        'Target localization': target_loc,
    }

def slug(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.lower().replace('+', 'plus')).strip('_') or 'variant'

def feature_root_base(output_dir: Path, dataset: DatasetConfig) -> Path:
    return output_dir / 'feature_roots/object_detectors/runs' / dataset.detector_key / 'predict' / dataset.key

def materialize_root(source_root: Path, csv_name: str, cue: str, dataset: DatasetConfig, tail: str, output_dir: Path, feature_cols: list[str] | None, limit_rows: int | None) -> Path:
    source_csv = source_root / csv_name
    cols = header(source_csv)
    if feature_cols is None:
        usecols = cols
    else:
        missing = [col for col in feature_cols if col not in cols]
        if missing:
            raise ValueError(f'Missing feature columns in {source_csv}: {missing}')
        keep = set(MERGE_KEYS) | (set(cols) & META_COLUMNS) | set(feature_cols)
        usecols = [col for col in cols if col in keep]
    if not set(MERGE_KEYS).issubset(usecols):
        raise ValueError(f'{source_csv} does not contain required merge keys: {MERGE_KEYS}')
    target = feature_root_base(output_dir, dataset) / f'00-00-0000_00;00_{cue}_{tail}'
    target.mkdir(parents=True, exist_ok=True)
    pd.read_csv(source_csv, usecols=usecols, nrows=limit_rows).to_csv(target / csv_name, index=False)
    return target

def find_component_root(grid_root: Path, suffix: str) -> Path:
    matches = sorted(path for path in grid_root.iterdir() if path.is_dir() and path.name.endswith(suffix))
    if not matches:
        raise FileNotFoundError(f'No gradient component directory ending with {suffix!r} under {grid_root}')
    if len(matches) > 1:
        raise ValueError(f'Multiple gradient component directories ending with {suffix!r}: {[p.name for p in matches]}')
    if not (matches[0] / 'layer_grad.csv').is_file():
        raise FileNotFoundError(matches[0] / 'layer_grad.csv')
    return matches[0]

def maybe_limit_root(source_root: Path, csv_name: str, cue: str, dataset: DatasetConfig, tail: str, output_dir: Path, limit_rows: int | None) -> Path:
    if limit_rows is None:
        return source_root
    return materialize_root(source_root, csv_name, cue, dataset, tail, output_dir, None, limit_rows)

def summary_rows(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    eval_path = run_dir / 'results/evaluation_results.csv'
    meta_path = run_dir / 'metadata.json'
    df = pd.read_csv(eval_path)
    mean = df.loc[df['row_type'] == 'mean']
    std = df.loc[df['row_type'] == 'std']
    if mean.empty or std.empty:
        raise ValueError(f'Missing mean/std rows in {eval_path}')
    with meta_path.open('r', encoding='utf-8') as f:
        meta = json.load(f)
    return mean.iloc[0].to_dict(), std.iloc[0].to_dict(), meta

def has_train_outputs(run_dir: Path) -> bool:
    return (run_dir / 'results/evaluation_results.csv').is_file() and (run_dir / 'metadata.json').is_file()

def variant_run_root(output_dir: Path, dataset: DatasetConfig, run_group: str, run_name: str) -> Path:
    return output_dir / 'runs' / dataset.detector_key / dataset.key / slug(run_group) / slug(run_name)

def legacy_variant_run_root(output_dir: Path, dataset: DatasetConfig, run_group: str, run_name: str) -> Path | None:
    if dataset.detector_key != 'yolov5':
        return None
    return output_dir / 'runs' / dataset.key / slug(run_group) / slug(run_name)

def selected_variant_run_root(output_dir: Path, dataset: DatasetConfig, run_group: str, run_name: str) -> Path:
    run_root = variant_run_root(output_dir, dataset, run_group, run_name)
    if run_root.exists():
        return run_root
    legacy_root = legacy_variant_run_root(output_dir, dataset, run_group, run_name)
    if legacy_root is not None and legacy_root.exists():
        return legacy_root
    return run_root

def has_variant_outputs(output_dir: Path, dataset: DatasetConfig, run_group: str, run_name: str) -> bool:
    run_root = selected_variant_run_root(output_dir, dataset, run_group, run_name)
    return has_train_outputs(run_root / 'meta_classifier') and has_train_outputs(run_root / 'meta_regressor')

def prefixed_features(cols: list[str], prefixes: tuple[str, ...]) -> list[str]:
    meta = set(META_COLUMNS)
    return [col for col in cols if col not in meta and any(col.startswith(prefix) for prefix in prefixes)]

def gradient_component_root(dataset: DatasetConfig, component: str, output_dir: Path, limit_rows: int | None) -> Path:
    run_name, selectors = dataset.grad_components[component]
    if dataset.grad_mode == 'grid_suffix':
        suffix = selectors[0]
        root = find_component_root(dataset.grad_grid_root, suffix)
        return maybe_limit_root(root, 'layer_grad.csv', 'layer_grad', dataset, suffix, output_dir, limit_rows)
    if dataset.grad_mode == 'single_file_prefix':
        cols = header(dataset.grad_grid_root / 'layer_grad.csv')
        features = prefixed_features(cols, selectors)
        if not features:
            raise ValueError(f'No gradient columns for {dataset.detector_name} / {component}: prefixes={selectors}')
        return materialize_root(dataset.grad_grid_root, 'layer_grad.csv', 'layer_grad', dataset, slug(run_name), output_dir, features, limit_rows)
    raise ValueError(f'Unknown gradient mode: {dataset.grad_mode}')

def run_variant(dataset: DatasetConfig, table_type: str, component_name: str, run_name: str, run_group: str, input_roots: Path | list[Path], classifier_base: dict[str, Any], regressor_base: dict[str, Any], output_dir: Path) -> EvalResult:
    run_root = selected_variant_run_root(output_dir, dataset, run_group, run_name)
    cls_dir = run_root / 'meta_classifier'
    reg_dir = run_root / 'meta_regressor'
    cls_cfg = train_config(classifier_base, input_roots, dataset.gt_root, regressor=False)
    reg_cfg = train_config(regressor_base, input_roots, dataset.gt_root, regressor=True)
    if has_train_outputs(cls_dir):
        print(f'[{dataset.detector_name} / {dataset.display_name}][{run_group}] {run_name}: reuse classifier')
    else:
        print(f'[{dataset.detector_name} / {dataset.display_name}][{run_group}] {run_name}: classifier')
        run_classifier_train(cls_cfg, cls_dir)
    if has_train_outputs(reg_dir):
        print(f'[{dataset.detector_name} / {dataset.display_name}][{run_group}] {run_name}: reuse regressor')
    else:
        print(f'[{dataset.detector_name} / {dataset.display_name}][{run_group}] {run_name}: regressor')
        run_regressor_train(reg_cfg, reg_dir)
    cls_mean, cls_std, cls_meta = summary_rows(cls_dir)
    reg_mean, reg_std, _reg_meta = summary_rows(reg_dir)
    rows = int(cls_meta.get('num_rows', 0))
    tp = int(cls_meta.get('num_positive_tp', 0))
    return EvalResult(dataset.detector_name, dataset.detector_key, dataset.key, dataset.display_name, table_type, component_name, rows, int(cls_meta.get('feature_dimension', 0)), float(tp / rows) if rows else 0.0, float(cls_mean['auroc']), float(cls_std['auroc']), float(cls_mean['ap']), float(cls_std['ap']), float(cls_mean['fpr95']), float(cls_std['fpr95']), float(reg_mean['r2']), float(reg_std['r2']), rel(cls_dir), rel(reg_dir))

def evaluate_unto_o(dataset: DatasetConfig, classifier_base: dict[str, Any], regressor_base: dict[str, Any], output_dir: Path, limit_rows: int | None) -> list[EvalResult]:
    sets = unto_o_feature_sets(header(dataset.unto_o_root / 'null_detect.csv'), dataset)
    results = []
    run_group = 'UnTO-O ablation'
    for table_type, component_name, run_name in UNTO_O_COMPONENTS:
        features = sets[run_name]
        if not features:
            raise ValueError(f'Empty UnTO-O subset: {dataset.display_name} / {run_name}')
        if has_variant_outputs(output_dir, dataset, run_group, run_name):
            root = dataset.unto_o_root
        elif run_name == 'Full UnTO-O' and limit_rows is None:
            root = dataset.unto_o_root
        else:
            root = materialize_root(dataset.unto_o_root, 'null_detect.csv', 'null_detect', dataset, slug(run_name), output_dir, None if run_name == 'Full UnTO-O' else features, limit_rows)
        results.append(run_variant(dataset, table_type, component_name, run_name, run_group, root, classifier_base, regressor_base, output_dir))
    return results

def evaluate_unto_g(dataset: DatasetConfig, classifier_base: dict[str, Any], regressor_base: dict[str, Any], output_dir: Path, limit_rows: int | None) -> list[EvalResult]:
    run_group = 'UnTO-G component ablation'
    roots: dict[str, Path | list[Path]] = {}
    real_roots: dict[str, Path] = {}

    def real_root(component: str) -> Path:
        if component not in real_roots:
            real_roots[component] = gradient_component_root(dataset, component, output_dir, limit_rows)
        return real_roots[component]

    for component, (run_name, _selectors) in dataset.grad_components.items():
        roots[run_name] = dataset.grad_grid_root if has_variant_outputs(output_dir, dataset, run_group, run_name) else real_root(component)
    if has_variant_outputs(output_dir, dataset, run_group, 'All components'):
        roots['All components'] = dataset.grad_grid_root
    else:
        roots['All components'] = [
            real_root('Localization'),
            real_root('Class'),
            real_root('Score'),
        ]
    return [
        run_variant(dataset, '', table_name, run_name, run_group, roots[run_name], classifier_base, regressor_base, output_dir)
        for table_name, run_name in UNTO_G_COMPONENTS
    ]

def result_dict(result: EvalResult) -> dict[str, Any]:
    return result.__dict__.copy()

def esc(text: str) -> str:
    for old, new in {'&': r'\&', '%': r'\%', '_': r'\_', '#': r'\#'}.items():
        text = text.replace(old, new)
    return text

def write_gradient_layers_table(path: Path) -> None:
    lines = [r'\begin{table}[t]', r'\centering', r'\caption{Gradient extraction layers used for UnTO-G.}', r'\label{tab:app_gradient_layers}', r'\begin{tabular}{L{0.18\linewidth}L{0.28\linewidth}L{0.44\linewidth}}', r'\toprule', r'Detector & Component & Layers \\', r'\midrule']
    prev_detector = None
    for detector, component, layers in GRAD_LAYER_ROWS:
        if prev_detector is not None and detector != prev_detector:
            lines.append(r'\specialrule{\cmidrulewidth}{\aboverulesep}{\belowrulesep}')
        detector_cell = esc(detector) if detector != prev_detector else ''
        lines.append(f'{detector_cell} & {esc(component)} & {esc(layers)} ' + r'\\')
        prev_detector = detector
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def config_rows(classifier_path: Path, regressor_path: Path, classifier_cfg: dict[str, Any], regressor_cfg: dict[str, Any]) -> list[dict[str, str]]:
    cm, ce = classifier_cfg.get('model', {}), classifier_cfg.get('experiment', {})
    rm, rexp = copy.deepcopy(regressor_cfg.get('model', {})), regressor_cfg.get('experiment', {})
    rm['type'] = 'xgb_regressor'
    cr, rr = ce.get('repeat', {}), rexp.get('repeat', {})
    return [
        {'item': 'Classifier config', 'setting': rel(classifier_path)},
        {'item': 'Classifier', 'setting': f'{cm.get("type", "gb_classifier")} ({cm.get("device", "cpu")})'},
        {'item': 'Classifier process', 'setting': str(ce.get('process', 'kfold'))},
        {'item': 'Classifier repeats', 'setting': str(cr.get('repeats', '--'))},
        {'item': 'Classifier test ratio', 'setting': str(cr.get('split', '--'))},
        {'item': 'Classifier augmentation', 'setting': str(ce.get('augmentation', 'none'))},
        {'item': 'Regressor config', 'setting': rel(regressor_path)},
        {'item': 'Regressor', 'setting': f'{rm.get("type")} ({rm.get("device", "cpu")})'},
        {'item': 'Regressor process', 'setting': str(rexp.get('process', 'kfold'))},
        {'item': 'Regressor repeats', 'setting': str(rr.get('repeats', '--'))},
        {'item': 'Regressor test ratio', 'setting': str(rr.get('split', '--'))},
        {'item': 'Random seed', 'setting': f'classifier {cm.get("random_seed", 42)}, regressor {rm.get("random_seed", 42)}'},
        {'item': 'Search', 'setting': f'classifier {bool(cm.get("search", False))}, regressor {bool(rm.get("search", False))}'},
    ]

def write_hyperparameters(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [r'\begin{table}[t]', r'\centering', r'\caption{Meta-model and split settings used for appendix ablations.}', r'\label{tab:app_hyperparameters}', r'\begin{tabular}{L{0.28\linewidth}L{0.62\linewidth}}', r'\toprule', r'Item & Setting \\', r'\midrule']
    for row in rows:
        lines.append(f'{esc(row["item"])} & {esc(row["setting"])} ' + r'\\')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def make_config(detector_key: str, dataset_key: str) -> DatasetConfig:
    if detector_key not in DETECTOR_PROFILES:
        raise ValueError(f'Unknown detector: {detector_key}. Available: {sorted(DETECTOR_PROFILES)}')
    if dataset_key not in DATASET_INFO:
        raise ValueError(f'Unknown dataset: {dataset_key}. Available: {sorted(DATASET_INFO)}')
    try:
        gt_root, unto_o_root, grad_grid_root = DETECTOR_DATASET_ROOTS[(detector_key, dataset_key)]
    except KeyError as exc:
        raise ValueError(f'No default roots for detector/dataset: {detector_key}/{dataset_key}') from exc
    profile = DETECTOR_PROFILES[detector_key]
    display_name, summary_name = DATASET_INFO[dataset_key]
    return DatasetConfig(
        detector_key=detector_key,
        detector_name=str(profile['name']),
        key=dataset_key,
        display_name=display_name,
        summary_name=summary_name,
        gt_root=gt_root,
        unto_o_root=unto_o_root,
        grad_grid_root=grad_grid_root,
        target_class_cols=profile['target_class_cols'],
        target_score_cols=profile['target_score_cols'],
        target_loc_cols=profile['target_loc_cols'],
        grad_mode=str(profile['grad_mode']),
        grad_components=profile['grad_components'],
    )

def make_custom_config(args: argparse.Namespace) -> DatasetConfig:
    profile = DETECTOR_PROFILES['yolov5']
    return DatasetConfig(
        detector_key=args.custom_detector_key,
        detector_name=args.custom_detector_name,
        key='custom',
        display_name=args.custom_dataset_name,
        summary_name=args.custom_dataset_name,
        gt_root=resolve_path(args.gt_root),
        unto_o_root=resolve_path(args.unto_o_root),
        grad_grid_root=resolve_path(args.grad_grid_root),
        target_class_cols=profile['target_class_cols'],
        target_score_cols=profile['target_score_cols'],
        target_loc_cols=profile['target_loc_cols'],
        grad_mode=str(profile['grad_mode']),
        grad_components=profile['grad_components'],
    )

def parse_dataset_names(raw: str) -> list[str]:
    names = [name.strip().lower() for name in raw.split(',') if name.strip()]
    invalid = [name for name in names if name not in DATASET_INFO and name != 'custom']
    if not names or invalid:
        raise ValueError(f'Unknown dataset names: {invalid}. Available: {sorted(DATASET_INFO)} or custom')
    if 'custom' in names and len(names) > 1:
        raise ValueError('The custom dataset cannot be combined with named datasets.')
    return names

def parse_detector_names(raw: str) -> list[str]:
    names = [name.strip().lower() for name in raw.split(',') if name.strip()]
    invalid = [name for name in names if name not in DETECTOR_PROFILES]
    if not names or invalid:
        raise ValueError(f'Unknown detector names: {invalid}. Available: {sorted(DETECTOR_PROFILES)}')
    return names

def selected_configs(args: argparse.Namespace) -> list[DatasetConfig]:
    roots = [args.gt_root, args.unto_o_root, args.grad_grid_root]
    if any(roots):
        if not all(roots):
            raise ValueError('--gt-root, --unto-o-root, and --grad-grid-root must be provided together.')
        return [make_custom_config(args)]
    names = parse_dataset_names(args.datasets)
    if names == ['custom']:
        raise ValueError('--datasets custom requires custom roots.')
    detectors = parse_detector_names(args.detectors)
    return [make_config(detector, dataset) for detector in detectors for dataset in names]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build appendix ablation results using the main meta-model training code.')
    parser.add_argument('--datasets', default='coco,voc')
    parser.add_argument('--detectors', default='yolov5,fcos,faster_rcnn')
    parser.add_argument('--gt-root', default=None)
    parser.add_argument('--unto-o-root', default=None)
    parser.add_argument('--grad-grid-root', default=None)
    parser.add_argument('--custom-dataset-name', default='Custom')
    parser.add_argument('--custom-detector-key', default='custom')
    parser.add_argument('--custom-detector-name', default='Custom')
    parser.add_argument('--classifier-config', default=str(DEFAULT_CLASSIFIER_CONFIG))
    parser.add_argument('--regressor-config', default=str(DEFAULT_REGRESSOR_CONFIG))
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--limit-rows', type=int, default=None, help='Limit feature CSV rows for a smoke test only.')
    parser.add_argument('--skip-unto-o', action='store_true')
    parser.add_argument('--skip-unto-g', action='store_true')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = selected_configs(args)
    classifier_path = resolve_path(args.classifier_config)
    regressor_path = resolve_path(args.regressor_config)
    classifier_cfg = load_yaml(classifier_path)
    regressor_cfg = load_yaml(regressor_path)
    regressor_cfg.setdefault('model', {})['type'] = 'xgb_regressor'
    rows = config_rows(classifier_path, regressor_path, classifier_cfg, regressor_cfg)
    pd.DataFrame(rows).to_csv(output_dir / 'meta_model_hyperparameters.csv', index=False)
    write_hyperparameters(output_dir / 'hyperparameters_table.tex', rows)
    write_gradient_layers_table(output_dir / 'gradient_layers_table.tex')
    metadata: dict[str, Any] = {
        'selected_detectors': sorted({d.detector_key for d in datasets}),
        'selected_datasets': sorted({d.key for d in datasets}),
        'selected_configs': [f'{d.detector_key}/{d.key}' for d in datasets],
        'limit_rows': args.limit_rows,
        'classifier_config_path': rel(classifier_path),
        'regressor_config_path': rel(regressor_path),
        'regressor_model_override': 'xgb_regressor',
        'datasets': {},
    }
    dataset_rows: list[dict[str, Any]] = []
    unto_o_results: list[EvalResult] = []
    unto_g_results: list[EvalResult] = []
    for dataset in datasets:
        print(f'[Dataset] {dataset.detector_name} / {dataset.display_name}')
        if not args.skip_unto_o:
            unto_o_results.extend(evaluate_unto_o(dataset, classifier_cfg, regressor_cfg, output_dir, args.limit_rows))
        if not args.skip_unto_g:
            unto_g_results.extend(evaluate_unto_g(dataset, classifier_cfg, regressor_cfg, output_dir, args.limit_rows))
        first = next((r for r in unto_o_results + unto_g_results if r.detector_key == dataset.detector_key and r.dataset_key == dataset.key), None)
        if first:
            dataset_rows.append({
                'detector': dataset.detector_name,
                'detector_key': dataset.detector_key,
                'dataset_key': dataset.key,
                'dataset': dataset.summary_name,
                'display_name': dataset.display_name,
                'detections': first.num_rows,
                'tp_ratio': first.tp_ratio,
                'regression_target': 'best same-class IoU',
                'limit_rows': args.limit_rows or '',
            })
        metadata['datasets'][f'{dataset.detector_key}/{dataset.key}'] = {
            'detector': dataset.detector_name,
            'detector_key': dataset.detector_key,
            'dataset_key': dataset.key,
            'display_name': dataset.display_name,
            'gt_root': str(dataset.gt_root),
            'unto_o_root': str(dataset.unto_o_root),
            'grad_grid_root': str(dataset.grad_grid_root),
            'grad_mode': dataset.grad_mode,
        }
    pd.DataFrame(dataset_rows).to_csv(output_dir / 'dataset_summary.csv', index=False)
    (output_dir / 'dataset_summary_table.tex').unlink(missing_ok=True)
    if unto_o_results:
        pd.DataFrame([result_dict(r) for r in unto_o_results]).to_csv(output_dir / 'unto_o_ablation_results.csv', index=False)
    if unto_g_results:
        pd.DataFrame([result_dict(r) for r in unto_g_results]).to_csv(output_dir / 'unto_g_component_ablation_results.csv', index=False)
    metadata['unto_o_results'] = [result_dict(r) for r in unto_o_results]
    metadata['unto_g_results'] = [result_dict(r) for r in unto_g_results]
    (output_dir / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Appendix outputs written to: {output_dir}')

if __name__ == '__main__':
    main()
