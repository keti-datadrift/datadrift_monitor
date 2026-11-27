# Round Completeness Audit

## Objective

Verify whether completed seven-round experiments lost round directories or
TensorBoard scalar steps during the in-place log migration.

## Source Results

The audit read `run_summary.json`, `round_XX/round_summary.json`,
`aggregate_summary.json`, round-7 artifacts, and TensorBoard events under these
timestamped runs:

- `work_dirs/retinanet_voc_coreset_7rounds_5percent_to_20percent/07-31-2026_23;51`
- `work_dirs/retinanet_voc_ecpal_eca_full_7rounds_5percent_to_20percent/08-02-2026_19;46`
- `work_dirs/retinanet_voc_ecpal_eca_only_7rounds_5percent_to_20percent/08-05-2026_16;35`
- `work_dirs/retinanet_voc_ecpal_eua_only_7rounds_5percent_to_20percent/08-07-2026_18;32`
- `work_dirs/retinanet_voc_pal_7rounds_5percent_to_20percent/07-28-2026_17;23`
- `work_dirs/retinanet_voc_pal_lius_7rounds_5percent_to_20percent/08-04-2026_09;05`
- `work_dirs/retinanet_voc_ppal_7rounds_5percent_to_20percent/07-30-2026_12;34`

The incomplete and legacy runs were also inspected:

- `work_dirs/retinanet_voc_ecpal_eca_only_7rounds_5percent_to_20percent/08-07-2026_16;20`
- `work_dirs/retinanet_coco_pal_5rounds_2percent_to_10percent/08-11-2026_20;22`
- The PAL and PPAL `07-24-2026_12;00` legacy directories.

## Measured Evidence

- All seven completed runs have `seed_0`, `seed_1`, and `seed_2`.
- Every seed in those runs contains `round_00` through `round_07`.
- Every `round_01` through `round_07` summary has status `done`.
- Every completed run's aggregate summary contains round indices 1 through 7.
- Every seed-level TensorBoard scalar set contains steps 1 through 7, with
  seven points per active-learning scalar tag.
- Every completed `round_07` contains a checkpoint, evaluation JSON,
  annotations, and TensorBoard event data.
- The ECPAL ECA-only `08-07-2026_16;20` run records failure in round 2 and only
  has rounds 0 through 2.
- The COCO PAL `08-11-2026_20;22` run records failure in round 4 and only has
  rounds 0 through 4.
- The two `07-24-2026_12;00` legacy directories still contain round directories
  1 through 7 and one migrated text event per round, but have no run summaries,
  checkpoints, or evaluation JSON files.

## Interpretation

No completed seven-round experiment lost rounds or scalar steps during the log
migration. A reduced-looking list is most likely caused by TensorBoard showing
the two genuinely failed runs or by selecting per-round training runs instead
of the seed-level active-learning/aggregate runs. The legacy `07-24` directories
cannot be classified as completed experiments from their stored artifacts even
though their round directory names extend through round 7.

## Validation

- Parsed all relevant JSON summaries and checked their declared statuses and
  round indices.
- Loaded every completed seed/run event directory with TensorBoard's
  `EventAccumulator` and enumerated scalar steps.
- Checked required round-7 artifact types directly on disk.

No experiment files were modified during this audit.
