# VOC Data Preparation Log

## Why This Was Added

The PPAL dry-run output showed:

```text
missing data\active_learning\voc\voc_827_labeled_1.json
missing data\active_learning\voc\voc_827_unlabeled_1.json
```

That is not a runner failure. It means the command stayed in dry-run mode and the initial active-learning pool JSON files have not been generated yet.

## Implemented

- Added `datasets/prepare_voc_active_learning.py`.
- The script converts standard VOC2007+VOC2012 `trainval` XML annotations under `data/VOCdevkit/` into:
  - `data/VOC0712/annotations/trainval_0712.json`
  - `data/active_learning/voc/voc_827_labeled_1.json`
  - `data/active_learning/voc/voc_827_unlabeled_1.json`
- Added a clearer runner hint when initial pool JSONs are missing.
- Updated PPAL/PAL-LIUS experiment configs so COCO-style `file_name` values like `VOC2007/JPEGImages/000001.jpg` resolve against `data/VOCdevkit/`.

## Command

```powershell
python -B datasets/prepare_voc_active_learning.py --vocdevkit data/VOCdevkit --n-labeled 827 --n-diff 1 --seed 0
```

After that, rerun:

```powershell
python -B tools/run_active_learning.py configs/experiments/ppal_retinanet_voc.py --method ppal --rounds 1 --gpus 1
```

## Notes

- The script does not download VOC. It expects `data/VOCdevkit/VOC2007` and `data/VOCdevkit/VOC2012` to already exist.
- The generated split is deterministic for the chosen seed. PPAL did not include the exact published split files in `code_refs/`, so this is a reproducible local split rather than a byte-identical published split.
- The experiment config keeps PPAL's original `gpus=8`, but `tools/run_active_learning.py` now supports `--gpus`, `--port`, and `--python-path` overrides for local runs.
