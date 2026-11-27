# PPAL Artifact and Logging Refactor

## Scope

This change keeps `code_refs/` untouched and removes local compatibility shims
from the runtime method tree.

## Method Layout

- Removed `methods/pal/sampler.py`; PAL runtime imports now rely on
  `methods.pal.acquisition` directly.
- Removed the PPAL-only `methods/ppal/logging.py` helper after moving PPAL
  acquisition output to structured diagnostics.
- Added `methods/ppal/inference.py` as the explicit artifact boundary for PPAL:
  - `RetinaHeadUncertainty` output: `*.bbox.json` with `cls_uncertainty`
  - class quality checkpoint: `latest.pth` with `bbox_head.class_quality`
  - `RetinaHeadFeat` output: `image_dis.npy`

## PPAL Responsibilities

- `methods/ppal/dcus.py` now focuses on DCUS scoring and selection.
- `methods/ppal/ccms.py` now focuses on CCMS diversity selection.
- `methods/ppal/acquisition.py` wires paths, invokes the local samplers, and
  returns stage diagnostics.
- `methods/ppal/base.py` writes COCO-style pool JSONs and returns pool counts
  without printing verbose status text.

## Diagnostics

PAL and PPAL now use the same runner-level stdout shape:

```text
ALOD acquisition: method=<method> stage=<stage> round=<n> selected=<count>
wrote diagnostics: <path>
wrote labeled pool: <path>
wrote unlabeled pool: <path>
```

Detailed acquisition data is written as JSON. PPAL writes
`round_##/ppal_diagnostics.json`; PAL keeps its existing diagnostics filename
while adding common metadata such as method, stage, round, inputs, and outputs.

## Verification

- `python -B -m py_compile ...` passed for changed runtime files.
- `python -B tools/run_active_learning.py --method ppal --detector retinanet --dataset voc --rounds 1 --gpus 1 --dry-run` passed.
- `python -B tools/run_active_learning.py --method pal --detector retinanet --dataset voc --rounds 1 --gpus 1 --dry-run` passed.
- `python -B tools/run_active_learning.py --preset ppal-retinanet-voc-smoke --acquisition-only --round-index 1 --ppal-stage diversity --gpus 1 --dry-run` passed.

Tests were not added or edited in this change, per request.
