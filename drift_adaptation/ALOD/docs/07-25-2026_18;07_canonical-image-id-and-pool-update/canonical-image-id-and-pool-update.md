# Canonical Image Id And Pool Update

Note: output artifact retention was later tightened in
`docs/07-25-2026_19;00_output-artifact-cleanup/output-artifact-cleanup.md`.
The canonical image-id rule still applies, but PPAL no longer writes
`uncertainty_remainder.json` by default.

## Canonical Image Id Rule

ALOD uses COCO annotation `images[].id` as the canonical image id for every
method-runtime artifact. Method code should not derive image ids from file
names, paths, or dataset-specific naming conventions.

## Detector Artifact Boundary

`RetinaHeadFeat` now reads `img_meta["image_id"]` and writes that value into
`image_dis.npy`. The image id is injected by `AddImageIdToMeta` and collected
through the MMDetection test pipeline.

This means PPAL CCMS consumes a diversity cache that already contains canonical
ALOD image ids. It validates those ids against the oracle pool, but it no
longer maps filename-stem ids back to oracle ids.

Existing `image_dis.npy` files created before this change may still contain
filename-derived ids. They should not be reused after this change; rerun the
PPAL diversity inference step or start from a fresh work directory.

## Pool Update Rule

Method modules compute selected image ids. They do not write labeled or
unlabeled pool JSON files directly.

`methods.common.coco_pool` owns the pool update behavior:

- `write_candidate_pool_from_selection(...)` writes the PPAL DCUS uncertainty
  candidate pool and the remaining unlabeled pool.
- `write_next_round_pool_split(...)` writes the next round's labeled and
  unlabeled pools for final acquisitions.

For PPAL, DCUS writes:

- `uncertainty_pool.json`
- `uncertainty_remainder.json`

CCMS writes:

- `new_labeled.json`
- `new_unlabeled.json`

## Method Boundaries

`methods/ppal/dcus.py` owns DCUS scoring and candidate selection.
`methods/ppal/ccms.py` owns CCMS diversity selection.

Neither module writes pool JSON files. Pool construction stays in
`methods/common/coco_pool.py` and is called by `methods/ppal/acquisition.py`.

## Validation

Targeted validation for this refactor:

- `python -m compileall -q methods tools configs mmdet/alod`
- `python -B tools/run_active_learning.py --help`
- Inline pool writer fixture checks for candidate and next-round splits.
