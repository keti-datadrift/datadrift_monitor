# Config Cleanup

## Goal

Make `configs/` names and runner-facing keys describe their actual roles:

- `configs/alod_mmdet`: MMDetection train/inference configs.
- `configs/catalog`: ALOD runner catalog for method/detector/dataset presets.

## File Rename

Renamed MMDetection config files to clarify stage and output type:

- `retinanet_voc_train_26e.py` ->
  `retinanet_voc_train_quality_ema_26e.py`
- `retinanet_voc_uncertainty.py` ->
  `retinanet_voc_infer_uncertainty.py`
- `retinanet_voc_diversity.py` ->
  `retinanet_voc_infer_features.py`
- `retinanet_voc_pal.py` ->
  `retinanet_voc_infer_pal_detections.py`

`configs/catalog/detectors.py` now points to the renamed files.

## Runner Key Rename

Renamed catalog-to-runner config keys so they clearly refer to MMDetection
`--cfg-options`:

- `common_cfg_options` -> `mmdet_common_cfg_options`
- `eval_cfg_options` -> `mmdet_eval_cfg_options`
- `pal_cfg_options` -> `mmdet_pal_infer_cfg_options`

The old keys are not kept as fallbacks.

## Removed Generated And Placeholder Files

- Removed `configs/**/__pycache__/`.
- Removed unused MMDetection config placeholders:
  - `labeled_data = ''`
  - `unlabeled_data = ''`

## CLASSES Decision

`CLASSES` remains in the child inference configs. Those configs pass
`classes=CLASSES` to `ALCocoDataset` or `PALCocoDataset`; without a local
`CLASSES` definition, MMDetection config loading can fail or fall back to COCO
class ordering. Keeping the duplicated class tuple is safer until the dataset
class mapping is centralized explicitly.
