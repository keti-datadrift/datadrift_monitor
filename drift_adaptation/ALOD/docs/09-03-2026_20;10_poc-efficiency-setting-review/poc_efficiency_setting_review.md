# PoC Efficiency Setting Review

## Objective

Identify experiment settings that can reduce single-GPU active-learning
turnaround time during proof-of-concept work, while keeping the existing
PAL-compatible settings available as a separate reference protocol.

No runtime setting or experiment output was changed in this review.

## Inspected Paths

- `configs/alod_mmdet/retinanet_voc_base.py`
- `configs/alod_mmdet/retinanet_coco_base.py`
- `configs/alod_mmdet/retinanet_voc_train_quality_ema_26e.py`
- `configs/alod_mmdet/retinanet_coco_train_quality_ema_26e.py`
- `mmdet/alod/models/retinanet/al_quality_ema_head.py`
- `mmdet/alod/models/retinanet/feature_export_utils.py`
- `mmdet/alod/models/retinanet/ecpal_head.py`
- `mmdet/alod/models/retinanet/al_retinanet.py`
- `mmdet/apis/train.py`
- `mmdet/datasets/builder.py`
- `tools/internal/infer_detector.py`
- `tools/run_active_learning.py`
- completed ECPAL ECA-only summaries under
  `work_dirs/retinanet_voc_ecpal_eca_only_7rounds_5percent_to_20percent/08-05-2026_16;35/`

## Measured Baseline

The completed ECPAL ECA-only run gives a representative sequential runtime for
the current VOC protocol. Values below were recomputed from all 21
`round_summary.json` files (3 seeds x 7 rounds).

| Stage | Mean per seed | Share of runtime |
|---|---:|---:|
| Training | 554.3 min | 87.4% |
| Evaluation | 19.4 min | 3.1% |
| Labeled inference | 8.8 min | 1.4% |
| Unlabeled inference | 50.3 min | 7.9% |
| Acquisition computation | 1.4 min | 0.2% |
| Total | 10.57 h | 100% |

This means inference batching is a low-risk improvement, but training changes
are required for a large end-to-end speedup.

The current seven-round runner also executes acquisition after round 7. The
round-7 model is already trained and evaluated on the 20% labeled pool, so that
last acquisition only creates an unused 22.5% pool. In the measured ECA-only
run, round-7 post-evaluation inference and acquisition cost 8.6 minutes per
seed. Skipping it does not change round-1 through round-7 mAP. It should be kept
only when round-7 selections themselves are an analysis target.

## Current Constraints Relevant to Batch Size

- All train and test paths currently default to `samples_per_gpu=1`.
- `infer_detector.py` already replaces `ImageToTensor` when test batch size is
  greater than one. Standard evaluation, ECPAL result formatting, and the
  feature-export base iterate over every image in `img_metas`, so the source
  structure supports batched inference. Numerical and artifact-equivalence
  checks are still required before changing the default.
- The ResNet backbone uses `norm_eval=True`; increasing the physical training
  batch therefore does not change running BatchNorm statistics. It still
  changes gradient noise and optimizer behavior.
- Training is epoch-based for 26 epochs. Raising the batch from 1 to `B` keeps
  the number of images processed but reduces optimizer updates to roughly
  `1/B`.
- `RetinaQualityEMAHead` updates class quality once per optimizer step. A batch
  change therefore changes the EMA time scale as well as SGD. To approximately
  preserve decay per image exposure, use `new_momentum = 0.99 ** B` (0.9801 for
  B=2 and 0.9606 for B=4), subject to a bridge experiment.
- Warm-up is fixed at 500 iterations. To preserve roughly the same number of
  images seen during warm-up, use `500/B` iterations when batch size becomes
  `B`.
- The runner always requests deterministic cuDNN behavior, which disables
  cuDNN benchmark mode. This improves repeatability but can reduce throughput.
- The installed RTX 3090 / PyTorch 1.10 environment already reports TF32
  enabled for both CUDA matrix multiplication and cuDNN convolution. There is
  no additional TF32 switch to gain under the current environment.
- Checkpoints are already written only at epoch 26 and intermediate in-training
  evaluation is disabled. Checkpoint/evaluation-hook frequency is not a current
  training bottleneck.

## Recommended Priority

### 1. Safe or nearly safe protocol changes

1. Do not execute acquisition after the terminal round.
2. Batch all evaluation and acquisition inference. Benchmark powers of two and
   select by images/second and peak memory. Good starting candidates for the
   24 GB RTX 3090 are VOC batch 8 and COCO batch 4, but these are candidates,
   not validated limits.
3. Use one seed for ordinary PoC work. Run three seeds only for shortlisted
   comparisons.
4. Use staged rounds: rounds 1-3 for code/mechanism screening, then all seven
   rounds for promising methods. Current ECA timing indicates the first three
   rounds cost about 3.0 hours per seed versus 10.57 hours for all seven, but
   late-round method differences can be missed by the short screen.
5. Keep evaluation at every round for actual method comparisons because the
   active-learning curve is evidence. Intermediate evaluation may be disabled
   only for debugging, not for a result intended to compare methods.

### 2. Main training-throughput changes

1. Enable mixed-precision training with dynamic loss scaling. The local
   MMDetection/MMCV path already supports `fp16` through
   `Fp16OptimizerHook`. Keep acquisition inference and final evaluation in FP32
   initially so small FP16 score changes do not alter EUA rankings or reported
   metrics.
2. Benchmark physical training batches 2 and 4 for VOC and batch 2 for COCO.
   A practical first target is VOC batch 4 with AMP; COCO's larger 1333x800
   inputs make batch 2 the safer first target.
3. When using batch `B`, start the bridge experiment with:
   - VOC learning rate `0.002 * B`;
   - COCO learning rate `0.01 * B`;
   - warm-up iterations `round(500/B)`;
   - quality EMA momentum `0.99 ** B`;
   - the existing 26 epochs and epoch-20 LR step.
4. Try `workers_per_gpu` values 2, 4, and 8 and enable
   `persistent_workers=True`. Windows worker spawning and dataset size mean
   that more workers are not automatically faster. The local dataloader
   currently hard-codes `pin_memory=False`; changing it is a lower-priority
   benchmark because MMCV's custom containers may limit the benefit.

Batch 4 does not imply a fourfold training speedup: each epoch still processes
the same images and approximately the same convolution work. The gain comes
from better GPU utilization, fewer per-step overheads, and AMP. It must be
measured on the actual detector.

### 3. Faster but method-changing PoC options

Use these only in a clearly named fast profile shared by every compared method:

- reduce training from 26 epochs to the repository's coherent 12-epoch base
  schedule (`step=[8, 11]`);
- train round 1 normally, then warm-start later rounds from the previous
  checkpoint for a small number of epochs;
- reduce VOC input from 1000x600 to 800x480 and COCO from 1333x800 to about
  1067x640, reducing nominal pixel area to about 64%;
- rank only a fixed, shared random subset of the unlabeled pool.

These changes can alter convergence, small-object accuracy, uncertainty
calibration, and the active-learning trajectory. Warm-starting in particular
changes the protocol from independent retraining to continual fine-tuning.

### 4. Lower-priority micro-optimizations

- Expose `--fuse-conv-bn` for inference and benchmark it.
- Optionally disable deterministic cuDNN and enable `cudnn_benchmark` only if a
  measured speed gain justifies the loss of exact repeatability. Active-learning
  ranking can amplify small numerical differences over later rounds.
- Do not use gradient accumulation as a speed optimization; it preserves or
  increases compute and is mainly a memory workaround.
- Do not reduce ECPAL `nms_pre`, score threshold, `max_per_img`, or support
  thresholds as generic performance tuning. Those values define the EUA/ECA
  signal and would change the method itself.

## Proposed Protocol Separation

Keep the current behavior available as a `paper` profile and add a separate
`poc` profile instead of overwriting reference configs.

Suggested `poc` profile after calibration:

| Setting | VOC | COCO |
|---|---:|---:|
| Seeds during development | 1 | 1 |
| Train batch candidate | 4 | 2 |
| Inference batch candidate | 8 | 4 |
| Precision | AMP train, FP32 infer/eval | AMP train, FP32 infer/eval |
| Epochs, first conservative version | 26 | 26 |
| Workers candidate | 4 persistent | 4 persistent |
| Deterministic | keep initially | keep initially |
| Terminal acquisition | skip | skip |

Use three rounds only as a quick screen. A result used to decide the final
method should use all seven rounds with the same PoC profile for every method.

## Required Bridge Checks Before Adoption

1. Inference: compare batch 1, 4, and 8 on the same checkpoint and pool. Record
   throughput, peak GPU memory, image-ID coverage/order, detection counts,
   maximum score/box differences, EUA values, and selected-budget overlap.
2. Training: on the same seed-0 round-1 pool, compare current batch-1 FP32 with
   batch-2 and batch-4 AMP. Record epoch time, peak memory, final mAP, loss
   curves, and class-quality EMA state.
3. Choose one fixed PoC profile from those results; do not vary the profile
   between methods.
4. Run one complete seven-round EUA-only bridge under that profile. Existing
   paper-profile results and new PoC-profile results must remain separate.

The bridge is important because the locally observed EUA-only advantage over
PAL full is only about 0.0023 mAP, which is small enough that a training-protocol
change could be of comparable size.

## Validation Performed

- Statically inspected the train, inference, dataloader, custom RetinaNet head,
  feature-export, and round-plan code paths listed above.
- Recomputed stage totals from the 21 completed ECPAL ECA-only round summaries.
- Queried the installed `alod` environment for PyTorch/CUDA/GPU and backend
  flags: PyTorch 1.10.0+cu113, RTX 3090, TF32 enabled, fresh-process cuDNN
  benchmark and deterministic flags disabled. The runner later explicitly
  enables deterministic mode for training.
- No training, inference benchmark, smoke run, or experiment-output mutation
  was performed.

## Compatibility and Follow-up

This document is an analysis and recommendation only. No existing result can be
relabelled as a PoC-profile result, and no existing aggregate should combine
paper and PoC profiles. Implementing the recommendation requires explicit
profile metadata in run summaries and separate output directories so the two
protocols cannot be confused.
