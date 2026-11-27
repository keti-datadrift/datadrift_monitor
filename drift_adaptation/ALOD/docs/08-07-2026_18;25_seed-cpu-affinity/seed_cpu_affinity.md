# Seed CPU Affinity

## Purpose

ALOD can run multiple seed pipelines concurrently with `--seed-workers`. This
shares one visible GPU across seed processes. `--seed-cpu-cores` adds an optional
CPU placement limit so each concurrently running seed pipeline is restricted to
a fixed logical CPU core block.

## Command

```powershell
python -B tools/run_active_learning.py --method ecpal:eca-only --detector retinanet --dataset voc --gpus 1 --seeds 0 1 2 --seed-workers 3 --seed-cpu-cores 4
```

This creates three concurrent seed worker slots and assigns four logical CPU
cores to each slot.

Example:

```text
slot 0 -> cores 0-3
slot 1 -> cores 4-7
slot 2 -> cores 8-11
```

If there are more seeds than workers, later seeds reuse a slot only after an
earlier seed in that slot has finished. This avoids simultaneous seed pipelines
sharing the same CPU affinity block.

## Behavior

- Default behavior is unchanged. If `--seed-cpu-cores` is omitted, no affinity is
  applied.
- Affinity is applied to subprocesses launched by the runner, including
  training, evaluation, and inference commands.
- PyTorch DataLoader workers are child processes of the training process, so
  they normally inherit the training subprocess affinity.
- Acquisition code that runs inside the runner process is not separately pinned.
  It is lightweight compared with training and inference.
- CPU affinity limits CPU placement only. GPU memory, GPU compute, disk I/O, and
  CUDA scheduling are still shared by concurrent seed pipelines.

## Saved Metadata

Each seed `run_summary.json` records:

```json
{
  "seed_cpu_cores": 4,
  "cpu_affinity": [0, 1, 2, 3]
}
```

The timestamp-level `aggregate_summary.json` records:

```json
{
  "seed_workers": 3,
  "seed_cpu_cores": 4,
  "cpu_affinity_enabled": true,
  "seed_runs": [
    {
      "seed": 0,
      "cpu_affinity": [0, 1, 2, 3]
    }
  ]
}
```

## Dependency

The option uses `psutil` and requires:

```text
psutil==5.9.8
```

The dependency is imported lazily, so normal runs without `--seed-cpu-cores` do
not require affinity setup.
