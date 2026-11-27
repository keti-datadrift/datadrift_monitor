# SACIL-v0

SACIL-v0 tests the core hypothesis that old geometry should be preserved
strongly in stable regions and relaxed only around old visual neighborhoods
that conflict with incoming classes.

The current implementation supports the representative CIFAR-100 protocol:

- class order: PODNet/AFC order 1
- sessions: `B50 + 10x5`
- memory: 20 exemplars per class, herding
- backbones: standard CIFAR ResNet-32 and AFC Rebuffi-style ResNet-32
- classifiers: single-proxy cosine and AFC 10-proxy cosine

## Implemented methods

| Method | Configuration |
|---|---|
| Replay-CE | `configs/cifar100/replay_ce_b50_inc5.yaml` |
| Global-HAP | `configs/cifar100/global_hap_b50_inc5.yaml` |
| Flat-LRHAP | `configs/cifar100/flat_lrhap_b50_inc5.yaml` |
| SACIL-v0 | `configs/cifar100/sacil_v0_b50_inc5.yaml` |

All four methods share the same dataset pipeline, class order, memory,
backbone, classifier, and evaluator.

The primary strong-substrate comparison is:

| Method | Configuration |
|---|---|
| AFC | `configs/cifar100/afc_b50_inc5.yaml` |
| AFC + Global-HAP | `configs/cifar100/afc_global_hap_b50_inc5.yaml` |
| AFC + Flat-LRHAP | `configs/cifar100/afc_flat_lrhap_b50_inc5.yaml` |
| AFC + SACIL | `configs/cifar100/afc_sacil_v0_b50_inc5.yaml` |

These configurations reproduce the main AFC training ingredients: multi-proxy
NCA, adaptive feature-map distillation, KMeans proxy imprinting, iCaRL
herding, and balanced classifier fine-tuning.

## Validation

```powershell
& 'C:\Users\user\anaconda3\envs\hoyoung\python.exe' -m pytest -q
```

Validate a configuration and local dataset without training:

```powershell
& 'C:\Users\user\anaconda3\envs\hoyoung\python.exe' `
  scripts\train_cifar100.py `
  configs\cifar100\sacil_v0_b50_inc5.yaml `
  --dry-run
```

Run the two-session end-to-end smoke experiment:

```powershell
& 'C:\Users\user\anaconda3\envs\hoyoung\python.exe' `
  scripts\train_cifar100.py `
  configs\cifar100\smoke_sacil_v0.yaml `
  --max-sessions 2
```

## Full CIFAR-100 run

Run the strong AFC-substrate SACIL experiment:

```powershell
& 'C:\Users\user\anaconda3\envs\hoyoung\python.exe' `
  scripts\train_cifar100.py `
  configs\cifar100\afc_sacil_v0_b50_inc5.yaml
```

The earlier Replay-CE substrate can still be run with:

```powershell
& 'C:\Users\user\anaconda3\envs\hoyoung\python.exe' `
  scripts\train_cifar100.py `
  configs\cifar100\sacil_v0_b50_inc5.yaml
```

Resume from a completed session:

```powershell
& 'C:\Users\user\anaconda3\envs\hoyoung\python.exe' `
  scripts\train_cifar100.py `
  configs\cifar100\sacil_v0_b50_inc5.yaml `
  --resume outputs\<run>\seed_1\checkpoints\session_00.pt
```

## Session artifacts

Each run records:

- resolved configuration and Git commit
- per-session JSONL and CSV metrics
- average/final accuracy and average forgetting
- old/new accuracy and harmonic mean
- conflict/stable anchor-affinity drift
- conflict weights and most-relaxed classes
- deterministic taxonomy JSON
- model, memory indices, prototype bank, hierarchy, and anchor bank checkpoint

The implementation plan and mathematical contract are in
`mds/imp_plan/sacil_v0_core_hypothesis_implementation_plan.md`.

## Unified Table-1 experiments

Paper-facing controlled comparisons use one in-repo engine for every method.
PyCIL and author repositories are read-only algorithm references; their
entrypoints are never imported or executed by the unified runner.

```powershell
python scripts\train_table1.py configs\table1\cifar100\icarl_nme_b50_inc5_resnet32.yaml --device cuda:0 --seed 1
python scripts\train_table1.py configs\table1\cifar100\sacil_nme_b50_inc5_resnet32.yaml --device cuda:0 --seed 1
```

The shared contract fixes CIFAR-100 B50-Inc5 order, ResNet-32, augmentation,
20 exemplars per class, and AIA. Evaluation uses NME except for CREATE's core
native reconstruction-error classifier. Method-native losses and optimization
recipes live under `src/sacil`. Commands and implementation boundaries are documented in
`mds/experiments/table1_powershell_commands.md` and
`mds/experiments/table1_experiment_design.md`.
