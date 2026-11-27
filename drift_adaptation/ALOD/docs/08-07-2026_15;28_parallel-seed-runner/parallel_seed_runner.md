# Parallel Seed Runner

## Goal

`tools/run_active_learning.py` now supports running multiple seed pipelines
concurrently from one command. The active learning protocol, detector settings,
dataset split, batch size, learning rate, and method logic are unchanged.

The parallel unit is a full seed pipeline:

```text
seed_0: round_1 -> ... -> round_7
seed_1: round_1 -> ... -> round_7
seed_2: round_1 -> ... -> round_7
```

## Command

Sequential three-seed run:

```powershell
python -B tools/run_active_learning.py --method pal --detector retinanet --dataset voc --gpus 1 --seeds 0 1 2
```

Parallel three-seed run on the visible GPU:

```powershell
python -B tools/run_active_learning.py --method pal --detector retinanet --dataset voc --gpus 1 --seeds 0 1 2 --seed-workers 3
```

The same option works for other methods:

```powershell
python -B tools/run_active_learning.py --method ecpal:eca-only --detector retinanet --dataset voc --gpus 1 --seeds 0 1 2 --seed-workers 3
python -B tools/run_active_learning.py --method ppal --detector retinanet --dataset voc --gpus 1 --seeds 0 1 2 --seed-workers 3
```

## Output Layout

The output layout is unchanged. A single timestamped run directory contains one
subdirectory per seed plus the aggregate summary:

```text
work_dirs/<experiment_name>/<MM-DD-YYYY_HH;mm>/
  seed_0/
  seed_1/
  seed_2/
  aggregate_summary.json
```

Each seed writes only inside its own `seed_*` directory. The main process writes
`aggregate_summary.json` after all seed workers finish.

## Console Output

With `--seed-workers 1`, the runner keeps the existing train/eval/inference
progress bars.

With `--seed-workers > 1`, the runner uses compact seed and round status lines.
Detailed stdout/stderr remains in the existing log files:

```text
seed_*/round_XX/logs/train.log
seed_*/round_XX/logs/eval.log
seed_*/round_XX/logs/*_inference.log
```

This avoids multiple concurrent tqdm bars overwriting each other.

## Limits

- Parallel seed execution currently supports `--gpus 1` only.
- `--verbose` cannot be combined with `--seed-workers > 1`.
- GPU memory is not partitioned. Seed processes share the visible GPU.
- If GPU memory is insufficient, reduce `--seed-workers` to 2 or 1.

## Implementation Notes

- `--seed-workers` defaults to 1, preserving the previous sequential behavior.
- The runner deep-copies the resolved experiment config for each seed worker.
- Input preparation still runs once before seed workers start.
- Failed seed summaries are collected before aggregate summary writing.
