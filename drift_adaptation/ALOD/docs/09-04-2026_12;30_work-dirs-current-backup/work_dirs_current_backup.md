# Current and Archived Experiment Directories

## Objective

Separate the latest experiment and all future runs from experiments produced
before the September 3, 2026 PoC setting changes.

## Data Migration

The following completed run was moved to `work_dirs/current_work` while
preserving its experiment/timestamp relative path:

- `retinanet_voc_ecpal_eua_only_7rounds_5percent_to_20percent/09-03-2026_21;27`

Nine older run directories were moved to `work_dirs/backup_09-03-2026`:

- `retinanet_voc_coreset_7rounds_5percent_to_20percent/07-31-2026_23;51`
- `retinanet_voc_ecpal_eca_full_7rounds_5percent_to_20percent/08-02-2026_19;46`
- `retinanet_voc_ecpal_eca_only_7rounds_5percent_to_20percent/08-05-2026_16;35`
- `retinanet_voc_ecpal_eua_only_7rounds_5percent_to_20percent/08-07-2026_18;32`
- `retinanet_voc_pal_7rounds_5percent_to_20percent/07-24-2026_12;00`
- `retinanet_voc_pal_7rounds_5percent_to_20percent/07-28-2026_17;23`
- `retinanet_voc_pal_lius_7rounds_5percent_to_20percent/08-04-2026_09;05`
- `retinanet_voc_ppal_7rounds_5percent_to_20percent/07-24-2026_12;00`
- `retinanet_voc_ppal_7rounds_5percent_to_20percent/07-30-2026_12;34`

The 10 runs contained 2,447 files totaling approximately 121.19 GiB before
the move. The same run-relative directory structure was retained. The empty
original experiment directories were removed after the moves. No run content
was deleted. `work_dirs/pal_embeddings` was left in place because it is a
shared prepared-input cache rather than an experiment result.

## Runtime Changes

- `configs/catalog/methods.py` now places every catalog experiment under
  `work_dirs/current_work/<experiment_name>`.
- `tools/run_active_learning.py` now uses the same root for custom configs that
  do not explicitly define `output_dir`.
- An explicit `output_dir` in a custom config remains authoritative for
  backward compatibility.
- `README.md` documents the current/archive layout and points the normal
  TensorBoard command at `work_dirs/current_work`.

## Compatibility

JSON summaries and command plans inside migrated runs retain the absolute paths
recorded when those experiments actually executed. They were not rewritten,
which preserves provenance but means those stored path strings refer to the old
location. Checkpoints, TensorBoard events, metrics, and other files are present
at their new filesystem locations.

## Validation

- Verified the latest run had `status: done` and an aggregate summary before
  moving it.
- Verified exactly one run was classified as current and exactly nine as
  backup, with no destination collisions.
- Compared run-relative paths, file counts, and byte totals after the move.
- Resolved catalog output paths for every supported method/dataset combination
  and confirmed they start with `work_dirs/current_work`.
- Checked the fallback output path for a custom config without `output_dir`.
- Compiled the modified Python files in memory.
- Ran `git diff --check`.

No training, evaluation, smoke workflow, or synthetic run was started.
