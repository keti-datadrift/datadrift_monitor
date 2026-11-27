# VOC PoC Batch-16 / Inference-32 Update

## Objective

Increase the current single-GPU VOC PoC throughput settings from training
batch 8 and inference batch 16 to training batch 16 and inference batch 32.

## Changes

- Set the common VOC training batch to 16.
- Set VOC validation, evaluation, and acquisition inference batch to 32.
- Scale the QualityEMA learning rate from 0.016 to 0.032, preserving the
  original batch-1 linear scaling rule (`0.002 * 16`).
- Set warm-up to 32 optimizer iterations (`ceil(500 / 16)`).
- Set Quality EMA base momentum to `0.8514577710948755` (`0.99 ** 16`) to
  preserve approximately the previous decay per image exposure.
- Keep workers at 4, persistent workers enabled, dynamic AMP for training,
  FP32 for evaluation/acquisition, 26 epochs, LR step 20, seven rounds, and
  per-iteration progress logging.

Affected files:

- `configs/alod_mmdet/retinanet_voc_base.py`
- `configs/alod_mmdet/retinanet_voc_train_quality_ema_26e.py`
- `README.md`

The separate MIAL training config remains batch 1, workers 2, LR 0.002, and
FP32. It inherits the new batch-32 test pipeline for evaluation. COCO settings
were not changed. No automatic batch fallback was added.

## Validation

- Loaded the resolved VOC QualityEMA config and checked all new batch, LR,
  warm-up, EMA, precision, worker, epoch, LR-step, and logging values.
- Loaded every VOC inference config and checked batch 32, workers 4, and FP32.
- Checked that MIAL training and COCO configuration values remain unchanged.
- Compiled both modified Python configs in memory.
- Ran `git diff --check`.

No real or disposable GPU experiment was run. RTX 3090 memory fit and numerical
stability must be established by the next real experiment; an OOM remains
visible rather than triggering a silent fallback.
