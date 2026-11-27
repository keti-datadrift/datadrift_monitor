# Common API Cleanup

## Goal

`methods/common` contains pure method-runtime helpers shared by active-learning
methods. It should not become a broad utility barrel or keep unused helpers for
possible future use.

## Import Rule

Call sites should import from concrete common modules:

- `methods.common.coco_pool`
- `methods.common.selection`
- `methods.common.candidates`
- `methods.common.results`
- `methods.common.io`

`methods.common.__init__` intentionally does not re-export helper functions.
This keeps dependencies explicit and makes helper ownership easier to find.

## Cleanup Performed

- Moved `write_diagnostics()` into `methods.common.results` so acquisition
  result envelopes and diagnostics writing live together.
- Removed `methods.common.diagnostics`.
- Removed unused candidate helpers:
  - `normalize_candidate_record()`
  - `rank_candidate_records()`
- Removed unused acquisition console helper:
  - `print_acquisition_summary()`
- Removed unused image-id helper:
  - `canonical_image_ids()`
- Reduced `methods.common.__init__` to package documentation only.

## Kept Boundaries

- `coco_pool.py`: COCO-style annotation pool operations.
- `selection.py`: deterministic image-id ranking, sampling, and refill.
- `detections.py`: detection JSON loading.
- `matching.py`: method-neutral bbox/ground-truth matching.
- `candidates.py`: compact candidate artifact construction and writing.
- `image_identity.py`: scalar image-id normalization and validation.
- `paths.py`: path containment checks shared by method/tool code.
- `vectors.py`: vector math shared by embedding builders.
