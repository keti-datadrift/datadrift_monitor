# Output Artifact Cleanup

## Goal

ALOD keeps files that are required for later rounds, reproducibility, or
method analysis. Files that duplicate another saved artifact or can be
reconstructed from oracle annotations and candidate ids should not be written
by default.

## Kept Artifacts

- `active_learning_plan.json`: exact runner plan and subprocess arguments.
- `run_summary.json`: compact run-level summary, prepared inputs, and paths to
  per-round summaries.
- `round_XX/round_summary.json`: per-round step status, logs, timings, and
  acquisition outputs.
- `round_XX/logs/*.log`: raw subprocess stdout/stderr for debugging.
- `round_XX/annotations/new_labeled.json` and `new_unlabeled.json`: next-round
  pools.
- PAL detection JSONs: raw inputs needed to recompute LIUS/GUIDE acquisition.
- PPAL `unlabeled_inference_result.bbox.json`, `latest.pth`,
  `annotations/uncertainty_pool.json`, and `image_dis.npy`: raw inputs needed
  by DCUS and CCMS.
- `*_diagnostics.json`: method-level acquisition diagnostics and metrics.
- `*_candidates.json`: compact candidate ranking artifacts with selected flags.

## Removed Default Artifacts

- `preparation_summary.json`: duplicated by `run_summary.preparation`.
- `run_summary.rounds_detail`: duplicated by `round_XX/round_summary.json`.
- PAL candidate COCO split files:
  - `pal_candidate_pool.json`
  - `pal_candidate_remainder.json`
  - `pal_lius_candidate_pool.json`
  - `pal_lius_candidate_remainder.json`
- PPAL `uncertainty_remainder.json`: reconstructable from the oracle, previous
  labeled pool, and `uncertainty_pool.json`.

## Candidate Artifact Policy

Candidate artifacts store ranking-oriented analysis fields only:

- common fields: `image_id`, `category_id`, `rank`, `score`, `source`,
  `selected`, `selection_rank`
- score decomposition under `components`
- non-score selection metadata under `metadata`

Inputs, outputs, and aggregate metrics stay in diagnostics and round summaries
instead of being repeated inside candidate artifacts.
