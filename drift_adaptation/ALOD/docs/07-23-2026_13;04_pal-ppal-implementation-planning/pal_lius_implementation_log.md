# PAL LIUS Implementation Log

## Implemented

- Added `methods/common/matching.py` for greedy class-wise detection-to-ground-truth TP/FP matching with IoU threshold control.
- Added `methods/pal/inference.py` to parse PAL detection JSON records and expose the LIUS feature tuple:
  - `pre_nms_count`
  - detection confidence
- Added `methods/pal/lius.py` with dependency-light NumPy class-wise logistic models and binary entropy scoring.
- Added `methods/pal/sampler.py` for LIUS-only PAL image selection:
  - matches labeled detections to labeled ground truth;
  - trains class-wise logistic TP/FP models;
  - scores unlabeled detections with LIUS;
  - allocates class budgets with the PAL rarity formula and deterministic largest-remainder rounding;
  - selects unique image ids and fills short budgets deterministically.
- Added `configs/experiments/pal_lius_retinanet_voc.py` as the first PAL experiment config.
- Added `--method pal` support to `tools/run_active_learning.py`.
- Added `tests/test_pal_lius.py` synthetic tests for TP/FP matching and LIUS-only image selection.
- Added detector-side PAL LIUS export:
  - `mmdet/ppal/models/retinanet_al/retinanet_pal_head.py`
  - `mmdet/ppal/datasets/pal_coco.py`
  - `configs/voc_active_learning/al_inference/retinanet_pal_lius.py`
- Updated `tools/run_active_learning.py` so `--method pal` plans labeled inference, unlabeled inference, then LIUS acquisition.
- Added `configs/experiments/pal_lius_retinanet_voc_smoke.py` for a tiny user-run smoke pipeline.

## Validation

Ran Python compile checks for the new PAL files and updated runner.

Ran:

```powershell
& 'C:\Users\Yeseongjin\anaconda3\envs\alod\python.exe' -B -m unittest tests.test_pal_lius
```

Result: passed, 2 tests.

Ran:

```powershell
& 'C:\Users\Yeseongjin\anaconda3\envs\alod\python.exe' -B tools\run_active_learning.py configs\experiments\pal_lius_retinanet_voc.py --method pal --rounds 1
```

Result: passed as a dry-run. It produced train/eval/acquisition command plans and launched no training, inference, or acquisition subprocesses.

No project code, tests, training, or smoke commands were executed during the detector-export integration pass because execution is user-controlled.

User-run PAL-LIUS smoke execution on 2026-07-23 completed end-to-end after
adding smoke-only PAL inference threshold overrides. The diagnostics file
reported:

```json
{
  "matched_detection_count": 400,
  "scored_detection_count": 400
}
```

The run selected image ids `5693` and `10219`.

## Current Limitations

- This is PAL-LIUS-only. GUIDE components CWIE, RCDI, and RCSP are not implemented yet.
- Detector-side PAL inference is now wired for RetinaNet and writes round-local files:
  - `pal_labeled_detections.bbox.json`
  - `pal_unlabeled_detections.bbox.json`
- The logistic model is a local NumPy implementation because PAL has no released code. This is reproducible and dependency-light, but it should be compared against any future official PAL implementation if one appears.
- Full PAL reproduction still requires adding GUIDE, embedding cache, and final score combination.
