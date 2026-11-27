# ALOD runner and method refactor

Date: 2026-07-23

## Summary

This pass reorganized the local ALOD code so users can select a method,
detector, and dataset without editing copied reference code or installing the
repository as a Python package.

## Changes

- Added `configs/catalog/` as the small user-facing registry for currently
  supported combinations.
- Updated `tools/run_active_learning.py` so the positional experiment config is
  optional when `--method`, `--detector`, and `--dataset` resolve through the
  catalog.
- Preserved the existing explicit config command style.
- Moved PAL acquisition orchestration into `methods/pal/acquisition.py`; kept
  `methods/pal/sampler.py` as an import compatibility shim.
- Ported PPAL acquisition code into `methods/ppal/`:
  - `dcus.py` for difficulty-calibrated uncertainty selection.
  - `ccms.py` for the diversity/CCMS selection step.
  - `common/base.py` for the shared sampler base.
  - `pipeline.py` for runner-facing PPAL acquisition execution.
- Replaced shell-based PPAL logging in the local method code with direct
  `print`, avoiding Windows `>` and `>>` shell parsing errors.
- Replaced the stale PPAL upstream README with ALOD local-source usage.

## Validation

- `python -B -m py_compile` for the runner, catalog, PAL, and PPAL method files.
- `python -B tools/run_active_learning.py --list-presets`.
- `python -B tools/run_active_learning.py --method pal --detector retinanet --dataset voc --rounds 1 --gpus 1 --dry-run`.
- `python -B tools/run_active_learning.py --method pal:lius --detector retinanet --dataset voc --rounds 1 --gpus 1 --dry-run`.
- `python -B tools/run_active_learning.py --preset ppal-retinanet-voc-smoke --acquisition-only --round-index 1 --ppal-stage diversity --gpus 1 --dry-run`.
- `python -B -m unittest tests.test_pal_lius tests.test_pal_guide tests.test_pal_embeddings tests.test_pal_vit_embeddings`.

No long training, inference, or full smoke execution was run in this pass.
