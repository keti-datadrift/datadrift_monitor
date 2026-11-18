# EPAL CECP+LECP round 7 interruption

## Objective and scope

Determine whether the existing run stopped before round 7 completed and provide
an appropriate recovery command. Diagnosis only: no training, evaluation,
checkpoint edits or runner changes were executed.

Source seed directory:
`work_dirs/current_work/retinanet_voc_epal_cecp_lecp_7rounds_5percent_to_20percent/09-10-2026_00;47/seed_0`.

## Observed evidence

- Rounds 1-6 have completed round summaries and final epoch-26 checkpoints.
  Round 6 completed acquisition of 414 images; its next labeled pool has 3,311
  images. Native round-6 mAP is 0.7010747194290161.
- Round 7 has a dumped configuration, annotation files and
  `20260910_043719.log.json`, but no `.pth`, evaluation JSON or round summary.
- The train log contains 208 iterations for each epoch 1-13. Its last record
  is epoch 13, cumulative iteration 2704, loss 0.56668, grad_norm 2.75144.
  Configured training is 26 epochs (5408 iterations at this loader length).
- `run_summary.json` still says `running` and lists only rounds 1-6.
  The live Python process query shows TensorBoard only, not detector training.
- The saved checkpoint configuration is interval=26, by_epoch=True,
  max_keep_ckpts=1. No intermediate round-7 checkpoint exists.
- No nonfinite loss/grad_norm entries were found in the saved round-7 JSON log.
  The termination cause cannot be established from these files alone.

## Recovery interpretation

Exact optimizer/model-state resume from epoch 13 is impossible with the
available files. Reuse the completed round-6 acquisition pool and restart only
round 7 for all 26 epochs with its dumped configuration, seed 0, deterministic
mode and one GPU; then run FP32 native VOC evaluation on its new latest.pth.
The dumped config already includes the exact 3,311-image training-pool path.

Do not resume from round_06/latest.pth: that is a different AL round and its
epoch counter has already reached 26. The recorded round-7 training command
does not initialize from that checkpoint; switching to it changes the protocol.

The top-level runner has no existing-run resume option and always creates a
new timestamp directory. `--start-round 7 --rounds 1` alone therefore does not
resume this run. Direct train/eval entry points can finish the detector work,
but do not update the AL runner's round/run/aggregate summaries or seed-level
TensorBoard records. This limitation must be explicit in the user handoff.

## Commands to provide

From an activated alod environment at the repository root, assign `$epalRound7`
to the source seed directory plus `/round_07` and `$epalRound7Config` to its
`retinanet_voc_train_quality_ema_26e.py`.

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
python -B tools/internal/train_detector.py "$epalRound7Config" --work-dir "$epalRound7" --launcher none --seed 0 --deterministic --gpus 1
if ($LASTEXITCODE -ne 0) { throw 'Round 7 training failed; evaluation was not started.' }
python -B tools/internal/infer_detector.py "$epalRound7Config" "$epalRound7/latest.pth" --work-dir "$epalRound7" --launcher none --eval mAP --cfg-options fp16=None data.test.ann_file=data/VOCdevkit/VOC2007/ImageSets/Main/test.txt data.test.img_prefix=data/VOCdevkit/VOC2007/
```

## Validation and affected files

Read the saved plan/configuration, round/run summaries, full round-7 JSON log,
checkpoint inventory, current Python process list, runner CLI/output selection,
train planning and internal training argument handling. Confirmed the recovery
commands against the saved train/eval arguments; they have not been executed.
Only this diagnostic record is added; original outputs and runtime code are
unchanged. Whitespace checks are performed after writing this record.
