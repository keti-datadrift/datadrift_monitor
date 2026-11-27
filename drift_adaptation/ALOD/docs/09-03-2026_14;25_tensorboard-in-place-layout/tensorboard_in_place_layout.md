# TensorBoard In-Place Result Layout

## Objective

Keep the established timestamp/seed/round result hierarchy and replace plain
text logs with TensorBoard events without creating a parallel
`tensorboard/seed_*` directory tree.

## Affected Behavior and Files

- `tools/run_active_learning.py` now writes aggregate events in the timestamped
  run directory, active-learning events in the corresponding `seed_*`
  directory, and command output plus training scalars in the existing
  `round_XX/logs/` directory.
- Command output is stored in TensorBoard's Text data and training `lr`, loss,
  and gradient-norm values are stored as scalars. Command and round metadata now
  use `tensorboard_dir` instead of the removed `log_path` field.
- `tools/internal/train_detector.py` and
  `tools/internal/train_mial_detector.py` no longer create timestamped `.log`
  files. Temporary MMDetection `.log.json` files are removed by the runner after
  their console output has been captured in the event file.
- The VOC and COCO MMDetection base configs retain `TextLoggerHook` for console
  progress but no longer create a separate `tf_logs` directory through
  `TensorboardLoggerHook`.
- `requirements/runtime.txt` pins `setuptools==59.5.0` for the PyTorch 1.10
  `distutils.version` compatibility requirement and retains TensorBoard 2.10.1.
- `README.md` documents the in-place event layout.

## Existing Result Migration

- Moved 215 existing event files out of 11 parallel `tensorboard/` trees and
  verified all moved files against their original SHA-256 digests.
- Converted 1,059 `.log` and `.log.json` files from 179 round directories into
  16,329 TensorBoard Text chunks under their existing `round_XX/logs/`
  directories, then removed the source text logs.
- Updated 1,417 historical `log_path` metadata fields to `tensorboard_dir`.
- Removed the 11 empty parallel `tensorboard/` directory trees. The resulting
  `work_dirs` contains 394 event files, no `.log`/`.log.json` files, and no
  directory named `tensorboard`.

## Validation

- `git diff --check` completed without whitespace errors; Git only reported the
  repository's existing LF-to-CRLF checkout warnings.
- Parsed all five modified Python/config files with `ast.parse`: successful.
- Imported `_summary_writer_type()` with the `alod` Conda environment using
  `python -B`: returned `SummaryWriter`, including with the previously failing
  PyTorch 1.10 environment.
- Loaded a migrated round and seed directory with TensorBoard's
  `EventAccumulator`: training scalar tags, active-learning scalar tags, and all
  sampled legacy text tags were present.
- Counted the final result tree: zero parallel `tensorboard` directories, zero
  raw log files, and 394 TensorBoard event files.

## Compatibility and Follow-Up

Historical command-plan and round-summary consumers must read
`tensorboard_dir`; `log_path` no longer exists because there is no corresponding
plain log file. Existing checkpoints, evaluation result JSON, annotations,
numeric summary values, and other experiment artifacts were not changed.
