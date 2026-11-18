# VOC Inference Batch 16 Adjustment

## Objective

Reduce the current VOC evaluation and acquisition inference batch from 32 to
16 because batch 32 exceeded the practical memory limit.

## Changes

- Set `data.val.samples_per_gpu` to 16.
- Set `data.test.samples_per_gpu` to 16.
- Updated the current runtime settings in `README.md`.

All VOC inference configs inherit the test value from
`configs/alod_mmdet/retinanet_voc_base.py`. QualityEMA training remains batch
16 with LR 0.032, warm-up 32, EMA momentum 0.8514577710948755, and dynamic
AMP. Workers remain 4 and evaluation/acquisition stays FP32. The separate MIAL
training batch and all COCO settings are unchanged.

## Validation

- Loaded the resolved QualityEMA config and all five VOC inference configs,
  confirming training batch 16 and inference batch 16.
- Confirmed the MIAL evaluation path resolves to inference batch 16 while its
  training batch remains 1.
- Compiled the modified config in memory.
- Ran `git diff --check`.

No GPU experiment or disposable validation run was started.
