# Candidate Pool Artifacts

Note: this earlier candidate-pool design was superseded by
`docs/07-25-2026_19;00_output-artifact-cleanup/output-artifact-cleanup.md`.
Current runner output keeps compact `*_candidates.json` files but no longer
writes PAL candidate COCO split files or PPAL uncertainty remainder files by
default.

## Rule

ALOD stores candidate pools with a method-neutral artifact schema, while
candidate scoring and selection remain inside each method implementation.

Common code may normalize candidate records, mark selected image ids, write
candidate JSON artifacts, and build COCO-style candidate pools. It must not
compute LIUS, GUIDE, DCUS, or CCMS scores.

## Common API

`methods/common/candidates.py` provides:

- `candidate_image_ids(...)`
- `unique_candidate_image_ids(...)`
- `normalize_candidate_record(...)`
- `rank_candidate_records(...)`
- `build_candidate_artifact(...)`
- `write_candidate_artifact(...)`

`methods/common/coco_pool.py` continues to own COCO pool writing:

- `write_candidate_pool_from_selection(...)`
- `write_next_round_pool_split(...)`

## PPAL Artifacts

PPAL DCUS writes:

- `round_XX/ppal_dcus_candidates.json`
- `round_XX/annotations/uncertainty_pool.json`
- `round_XX/annotations/uncertainty_remainder.json`

DCUS candidate records are produced from the ranked weighted uncertainty scores.
The uncertainty pool is the detector input for the following CCMS diversity
inference step.

PPAL CCMS writes:

- `round_XX/ppal_ccms_candidates.json`
- `round_XX/annotations/new_labeled.json`
- `round_XX/annotations/new_unlabeled.json`

CCMS candidate records correspond to the image ids in `image_dis.npy`. Since
CCMS is centroid/diversity based, records store distance index and centroid
selection metadata instead of inventing a scalar score.

## PAL Artifacts

PAL GUIDE writes:

- `round_XX/pal_candidates.json`
- `round_XX/annotations/pal_candidate_pool.json`
- `round_XX/annotations/pal_candidate_remainder.json`

PAL LIUS-only writes:

- `round_XX/pal_lius_candidates.json`
- `round_XX/annotations/pal_lius_candidate_pool.json`
- `round_XX/annotations/pal_lius_candidate_remainder.json`

PAL candidate records come from LIUS or GUIDE scoring outputs already produced
inside `methods/pal/acquisition.py` and `methods/pal/guide.py`.

## Analysis Entry Points

For per-round candidate analysis, inspect:

- `*_candidates.json` for score/component/rank/selection metadata.
- `annotations/*candidate_pool.json` for COCO-style image membership.
- `annotations/new_labeled.json` for final images promoted to the next round.
