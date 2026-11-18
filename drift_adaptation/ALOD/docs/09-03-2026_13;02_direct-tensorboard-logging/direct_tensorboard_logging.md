# Direct TensorBoard Logging

## Objective

Remove the one-time TensorBoard export path now that historical runs have been
converted, and make future experiments write TensorBoard events directly.

## Affected behavior and files

- `tools/run_active_learning.py` now writes validation, labeled-pool,
  acquisition-count, round-duration, and aggregate mean/std scalars as each run
  completes its rounds.
- `configs/alod_mmdet/retinanet_voc_base.py` and
  `configs/alod_mmdet/retinanet_coco_base.py` enable MMCV's
  `TensorboardLoggerHook` alongside the text logger, so training loss and
  learning-rate scalars are emitted during training.
- Removed `tools/export_tensorboard.py`,
  `tools/common/tensorboard_export.py`, and the exporter-only
  `tools/common/metrics_scanner.py`.
- `requirements/runtime.txt` retains TensorBoard as a required runtime
  dependency and constrains OpenCV below 4.12 for compatibility with the
  repository's NumPy 1.23.5 pin. The removed Streamlit dashboard and parallel
  seed worker no longer have dedicated dependencies.
- `README.md` now documents direct logging only. Existing historical event files
  remain usable with `tensorboard --logdir work_dirs`.

## Validation

- Compiled the modified runner and both base configs in memory with Python 3.9:
  passed without syntax errors.
- Ran `python -B tools/run_active_learning.py --help` and `--list-presets`:
  both completed successfully.
- Imported `torch.utils.tensorboard.SummaryWriter` through the runner's
  dependency check: passed.
- Loaded a preserved historical event directory with TensorBoard's
  `EventAccumulator`: nine scalar tags were readable in the sampled directory.
- Confirmed that all 215 historical `events.out.tfevents.*` files remain and
  that all 11 obsolete `tensorboard_export.json` manifests were removed.
- Confirmed the three exporter/scanner files are absent, searched current
  runtime code and documentation for stale exporter/dashboard/parallel-worker
  references, and ran `git diff --check`: passed.
- MMCV hook registry validation was attempted but could not run because MMCV is
  not installed in the current shell. Config syntax and hook names were checked;
  an actual GPU training run was not executed.

## Compatibility and follow-up

- The manual export command is intentionally removed and is no longer a
  supported interface.
- Historical TensorBoard event files are preserved. Exporter manifests are not
  required by TensorBoard and were removed separately.
- Multiple seeds remain supported through `--seeds`, but execute sequentially.
