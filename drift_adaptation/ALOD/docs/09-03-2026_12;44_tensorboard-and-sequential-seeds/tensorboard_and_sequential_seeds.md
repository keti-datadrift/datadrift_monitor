# TensorBoard and Sequential Seed Execution

## Objective

Replace the Streamlit metrics viewer with TensorBoard, migrate existing
experiment metrics into TensorBoard event files, and remove concurrent seed
execution on a shared GPU.

## Behavior Changes

- `tools/export_tensorboard.py` converts every discovered run below `work_dirs`
  into TensorBoard events.
- Completed active-learning runs automatically export their validation,
  acquisition, runtime, aggregate, and training metrics.
- TensorBoard events are rebuilt under each timestamped run's `tensorboard/`
  directory from the original JSON and text logs.
- `--seeds` remains available for grouped reporting, but seed pipelines always
  execute sequentially.
- `--seed-workers` and `--seed-cpu-cores` are removed. The associated thread
  pool, process-affinity code, and `psutil` dependency are also removed.
- Existing summary fields related to seed workers and CPU affinity remain with
  fixed sequential values for file-format compatibility.

## Affected Files

- `tools/run_active_learning.py`
- `tools/export_tensorboard.py`
- `tools/common/tensorboard_export.py`
- `tools/common/metrics_scanner.py`
- `requirements/runtime.txt`
- `README.md`
- `AGENTS.md`

The former Streamlit launcher, application, parsing helpers, and dashboard
requirements file were removed.

## Existing Result Migration

The migration command is:

```powershell
python -B tools/export_tensorboard.py
```

The resulting experiments can be viewed with:

```powershell
tensorboard --logdir work_dirs
```

## Validation

- Converted 11 historical runs containing 29 seed runs and 179 completed or
  partial rounds.
- Generated 215 event files containing 1,056,000 scalar points (about 51.7 MB)
  without modifying the source experiment logs.
- Loaded representative event files with TensorBoard's `EventAccumulator` and
  verified validation, aggregate mean/std, and train loss/lr tags and steps.
- Started TensorBoard 2.10.1 against `work_dirs` and received HTTP 200 from the
  local server.
- Checked the changed Python sources, both public CLI help paths, rejection of
  the removed `--seed-workers` option, and `git diff --check`.
- No training, evaluation, smoke, or synthetic experiment was run.

## Compatibility

TensorBoard 2.10.1 is pinned to remain compatible with the project's Python 3.8,
PyTorch 1.10, and NumPy 1.23 runtime. Historical summary fields for removed
parallel execution settings remain present with fixed sequential values.
