# Corrected EPAL round-7 recovery command

## Objective

Diagnose the failure of the preceding recovery command and provide a verified
configuration-loading alternative. No detector training/evaluation or runtime
code changes were requested or executed in this work item.

Source seed directory:
`work_dirs/current_work/retinanet_voc_epal_cecp_lecp_7rounds_5percent_to_20percent/09-10-2026_00;47/seed_0`.

## Evidence and cause

The previous record, `docs/09-10-2026_16;53_epal-round7-interruption/diagnosis.md`,
recommended loading the dumped round-7 `.py` config. That recommendation was
unsafe with this MMCV serializer and Windows paths and is superseded here.

- MMCV's `_format_basic_types` places string values inside ordinary single
  quotes without escaping backslashes. The dumped config contained Windows
  absolute paths as such literals.
- On Python reloading, path fragments become escapes. Examination of the
  original config text preserved in the first training-log JSON record found
  control characters in string literals at lines 115 and 217: character codes
  0 (NUL), 7 (bell), 10 (newline) and 13 (carriage return).
- The subsequent pretty-text formatting therefore fails, matching the user's
  null-byte/YAPF traceback. The final IndexError is a secondary exception in
  YAPF's error reporting, not the underlying cause.
- MMCV `Config.dump` opens the output with mode `w` before evaluating
  `self.pretty_text`. Since input and output were the same round-7 file, the
  failed command truncated it. Filesystem inspection found size 0 bytes,
  modified 2026-09-10 16:57:10.
- Training failed before model initialization. The original training log
  still contains the complete original configuration text. Existing round
  1-6 checkpoints and the 3,311-image round-7 input pool remain available.

## Corrected command design

Use `configs/alod_mmdet/retinanet_voc_train_quality_ema_26e.py` as the input,
not the dumped/now-empty round config. Pass repository-relative forward-slash
paths for work_dir and the existing round-6 acquired labeled pool.

Required train overrides:

- `data.train.ann_file=<seed>/round_06/annotations/new_labeled.json`
- `data.train.img_prefix=data/VOCdevkit/`
- `data.test.img_prefix=data/VOCdevkit/`

The base configuration already has the original VOC2007 validation paths.
Preserve seed=0, deterministic mode, gpus=1, batch=16, LR=.008, AMP training,
26 epochs and the existing checkpoint interval. Evaluate the resulting
round-7 latest.pth with the source config, VOC2007 test paths and `fp16=None`.

Successful training startup will regenerate the round-7 dumped config from
the valid source; no manual deletion is required. This is still a restart of
round 7 from its beginning, not an epoch-13 state resume. Direct commands still
do not finalize the parent runner's summaries or seed-level TensorBoard.

## Validation and changes

Read source configs, MMCV serialization/dump implementation, compatibility
wrapper and train startup ordering. The real proposed config was loaded,
merged and formatted in memory, parsed with AST, then loaded from its formatted
text. Full dictionary round-trip equality passed. After normalizing only
path syntax in the original logged config, comparison against the proposed
configuration found **zero differing top-level fields**. The training pool
exists and contains **3,311 images**.

This validation used real config inputs, no synthetic/smoke experiment, and
wrote no files in the run. Only this documentation record was added; the empty
run config was not manually restored or edited. No dependencies changed.
Training and evaluation have not been executed. A whitespace check follows
record creation.
