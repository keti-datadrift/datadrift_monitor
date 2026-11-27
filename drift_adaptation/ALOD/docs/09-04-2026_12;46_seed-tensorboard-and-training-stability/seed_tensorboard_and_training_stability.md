# Seed TensorBoard Consolidation and VOC Training Stability

## Objective

Make each seed appear as exactly one TensorBoard Run, migrate the completed
`09-03-2026_21;27` run to that layout without losing its recorded values, and
diagnose the anomalous round-7 evaluation before changing the VOC training
protocol.

## Inspected Run

`work_dirs/current_work/retinanet_voc_ecpal_eua_only_7rounds_5percent_to_20percent/09-03-2026_21;27`

## Measured Evidence

- Round 6 reported mAP `0.6709525585` and AP50 `0.671`.
- Round 7 reported mAP `6.8507228e-7` and AP50 `0.0`.
- The round-7 evaluation command used `round_07/latest.pth`, the training
  config, the VOC2007 test list, its VOC2007 image directory, and FP32
  evaluation. `latest.pth` is byte-identical to `epoch_26.pth`.
- Round 7 trained from the expected round-6 labeled pool: 3,311 unique images
  and 15,362 annotations. The terminal round copied that same pool forward as
  intended.
- Round-7 loss first exceeded 10 at iteration 160 and reached `118203.77`.
  `grad_norm` was non-finite in 5,187 of 5,408 logged iterations.
- The checkpoint contained no NaN or infinity parameters, but
  `bbox_head.class_quality` mean fell from `0.5331837` in round 6 to
  `0.00011093` in round 7.
- Rounds 1, 2, and 3 also diverged under the same setting. Their mAP values
  were `0.0001984`, `0.0003507`, and `0.0029127`; their logged non-finite
  gradient counts were 966, 1,869, and 2,519. Rounds 4 through 6 were stable.
- The earlier batch-1 EUA-only run `08-07-2026_18;32` produced a monotonic
  seed-0 curve from mAP `0.4462` to `0.7116`.

## Interpretation

The round-7 number is a real measurement of a failed model, not a metric parser,
checkpoint selection, dataset split, or TensorBoard display error. The current
run used the unvalidated linear batch scaling from LR `0.002` at batch 1 to LR
`0.032` at batch 16. The repeated large finite losses and sustained non-finite
gradient norms show optimizer instability. The collapsed quality buffer is a
symptom rather than the cause because that buffer is updated under `no_grad`
and does not feed the training loss.

The entire run is invalid for method comparison because the failed early-round
models also determined later acquisition pools. Retraining only round 7 would
not reconstruct a valid active-learning trajectory.

## Implemented Changes

- Kept VOC training batch 16, dynamic AMP, 32 warm-up iterations, 26 epochs,
  and QualityEMA momentum unchanged.
- Reduced the VOC QualityEMA LR from `0.032` to `0.008`. This uses a
  conservative square-root batch scaling from the stable batch-1 LR and is
  close to the standard LR used by RetinaNet at total batch 16.
- Added a VOC-only runner guard that terminates a training subprocess and marks
  the round/run failed after 20 consecutive `grad_norm=inf` or `grad_norm=nan`
  iteration records. Transient dynamic-loss-scale adjustment remains allowed.
  COCO behavior is unchanged.
- Changed all command and training TensorBoard records to the existing seed
  directory. Round-specific tags use `round_XX/...`; seed-wide validation,
  pool, acquisition, and runtime metrics retain the round number as their step.
- A seed now owns one `SummaryWriter` for its entire run. No event is written to
  `round_XX/logs` or the timestamp root. Cross-seed mean/std data remains in
  `aggregate_summary.json` rather than creating an extra TensorBoard Run.
- Added `result_validity.json` to the inspected run and added an equivalent
  TensorBoard text diagnostic. Existing metric JSON files and checkpoints were
  not rewritten.

## Existing Event Migration

- Rewrote 108,050 scalar events and 165,761 text events into one seed-level
  event file, prefixing round-specific tags with `round_01` through `round_07`.
- Added one `diagnostics/result_validity` text event.
- Preserved all 34 original event files under
  `work_dirs/backup_09-03-2026/_tensorboard_migration_backup/` with
  `legacy_` filenames so TensorBoard will not discover them automatically.
- Removed the now-empty `round_XX/logs` directories and the timestamp-root
  aggregate event from the current run. No checkpoint, annotation, evaluation,
  acquisition, or summary artifact was deleted.

## Validation

- Compiled the three modified Python source/config files in memory.
- Loaded the catalog and confirmed the VOC guard resolves to 20 while COCO has
  no guard.
- Loaded the resolved VOC training config and checked batch 16, LR `0.008`,
  dynamic AMP, warm-up 32, epoch 26, and EMA momentum.
- Compared source and migrated TensorBoard event counts before replacement.
- Reloaded the installed event stream and checked the required seed and
  round-prefixed tags.
- Confirmed TensorBoard discovery sees one event directory under
  `work_dirs/current_work`.
- Ran `git diff --check`.

No GPU training or evaluation was run. The corrected LR must be confirmed by
the next real seven-round experiment; the new guard prevents a sustained
non-finite-gradient failure from being recorded as successful.
