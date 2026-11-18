# ALOD cleanup refactor

Date: 2026-07-23

## Summary

This pass converted the prior intermediate port into a cleaner ALOD structure.
The repository now keeps only ALOD-owned config files and local method code for
the implemented RetinaNet/VOC target.

## Changes

- Reduced `configs/` to:
  - `configs/catalog/`
  - `configs/alod_mmdet/`
- Removed the copied MMDetection config zoo and the old experiment config files.
- Moved catalog defaults into dataclass specs for datasets, detectors, methods,
  and presets.
- Kept explicit positional `--config` support only as an expert override for
  user-provided files.
- Removed `methods/ppal/adapter.py` and `methods/ppal/pipeline.py`.
- Kept PPAL method source in:
  - `methods/ppal/base.py`
  - `methods/ppal/dcus.py`
  - `methods/ppal/ccms.py`
  - `methods/ppal/acquisition.py`
  - `methods/ppal/logging.py`
- Removed obsolete `mmdet/ppal/sampler`, `mmdet/ppal/builder.py`, and
  `mmdet/ppal/utils`.
- Kept `mmdet/ppal/datasets` and `mmdet/ppal/models` because train/test
  entrypoints still need those registry registrations.
- Removed generated `__pycache__` directories outside `code_refs/`.

## Validation

- Static stale-reference scans with `rg`.
- `py_compile` for runner, catalog, methods, tests, and minimal configs.
- Catalog and PAL unit tests.
- Dry-run planning for PAL and PPAL catalog commands.

No full training, inference, or execute smoke pipeline was run.
