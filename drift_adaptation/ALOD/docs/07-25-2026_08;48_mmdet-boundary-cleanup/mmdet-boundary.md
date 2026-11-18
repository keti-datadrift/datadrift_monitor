# MMDetection Boundary

## Goal

`mmdet/` is the local detector backend for ALOD. PAL, PPAL, and baseline active
learning methods should stay independent from MMDetection internals and consume
detector outputs as saved artifacts: COCO annotations, detection JSON files,
feature caches, checkpoints, and diagnostics.

This keeps paper method code in `methods/` readable and prevents acquisition
logic from depending on one detector framework.

## Layer Responsibilities

### `mmdet/`

The local MMDetection backend. It contains the copied detector framework plus
ALOD registry extensions under `mmdet/alod/`.

`mmdet/alod/` is for MMDetection dataset/model registrations only. These modules
may import MMDetection internals and should be imported by MMDetection
entrypoints for registry side effects.

### `tools/train.py` and `tools/test.py`

The only ALOD command entrypoints that directly import `mmdet`. They run
training, evaluation, and inference through MMDetection APIs.

### `tools/run_active_learning.py`

The active learning runner. It should orchestrate commands, paths, plans,
rounds, progress display, and method acquisition. It should call train/test as
subprocesses and read/write artifacts, not import `mmdet` directly.

### `tools/common/`

Runner and preparation helpers. This layer prepares source data, pretrained
weights, embedding caches, and paths. It should not import `mmdet`.

### `methods/`

Paper method logic and method-runtime common helpers. This layer should not
import `mmdet`; it should read detector artifacts produced by train/test.

## Allowed Imports

- `tools/train.py -> mmdet`
- `tools/test.py -> mmdet`
- `mmdet/** -> mmdet/**`
- `configs/alod_mmdet/**` may reference MMDetection config objects/classes as
  part of the detector backend configuration.

## Forbidden Imports

- `methods/** -> mmdet`
- `tools/common/** -> mmdet`
- `tools/run_active_learning.py -> mmdet`
- `configs/catalog/** -> mmdet`
- `mmdet/alod/** -> methods/**`
- `mmdet/alod/** -> tools/**`
- `mmdet/alod/** -> configs/**`

## Validation

The boundary is enforced by:

```powershell
python -m unittest tests.test_mmdet_boundaries
```

Useful manual checks:

```powershell
rg "from mmdet|import mmdet" methods tools/common tools/run_active_learning.py configs/catalog
rg "from (methods|tools|configs)|import (methods|tools|configs)" mmdet/alod
```
