# VOC Per-Iteration Progress Update

## Objective

Update the terminal training `tqdm` display after every iteration instead of
every 50 iterations for the current VOC experiments.

## Change

Changed `log_config.interval` from `50` to `1` in
`configs/alod_mmdet/retinanet_voc_base.py`. The public active-learning runner
advances its training progress bar from the MMDetection iteration records, so
this makes the bar and its loss/ETA postfix refresh on every training
iteration. All VOC train configs inherit this value, including the standard
QualityEMA path and the separate MIAL path.

COCO configuration, training math, checkpoint intervals, evaluation intervals,
and inference behavior were not changed. Per-iteration records will increase
TensorBoard console/scalar write frequency, as requested.

## Validation

- Loaded the resolved VOC QualityEMA and MIAL configs and confirmed
  `log_config.interval == 1`.
- Confirmed the runner's training parser advances from each emitted
  `Iter [current/total]` record.
- Compiled the modified Python config in memory.
- Ran `git diff --check`.

No training or disposable experiment was run. The first real run will provide
the runtime confirmation that the terminal visibly refreshes every iteration.
