# Core-set and PPAL Feature Refactor

## Goal

Add the Core-set active learning method to ALOD while making detector feature
export method-neutral. The runner should keep using the PAL paper protocol for
RetinaNet/VOC: 827 initial labeled images, 7 rounds, and 414 images per round.

## Reference Boundary

`code_refs/Core-set` is a classification-oriented reference implementation. ALOD
does not import or copy its training code or Gurobi solver. The implemented part
is the paper's method-level idea: greedy k-center selection in feature space.

PPAL's CCMS distance is preserved as PPAL-specific method logic. The detector
backend no longer produces `image_dis.npy`; it exports generic feature artifacts,
and `methods/ppal` computes the PPAL distance matrix from those artifacts.

## Runtime Ownership

- `mmdet/alod`: detector/backend inference code only. It exports feature
  artifacts and does not run acquisition algorithms.
- `methods/common`: method-neutral runtime utilities such as feature artifact
  loading, vector distances, greedy k-center, candidate artifacts, and COCO pool
  updates.
- `methods/ppal`: PPAL DCUS, PPAL detection-feature distance, and CCMS.
- `methods/coreset`: Core-set acquisition orchestration only.
- `tools/run_active_learning.py`: runner orchestration only.

## Feature Artifact Schema

Feature inference writes compressed `.npz` files with:

```text
image_ids       [N]
image_features  [N, D]
det_labels      [N, K]
det_scores      [N, K]
det_features    [N, K, D_det]
det_valid       [N, K]
metadata_json   scalar JSON string
```

Core-set uses `image_ids` and `image_features`.

PPAL CCMS uses `image_ids`, `det_labels`, `det_scores`, `det_features`, and
`det_valid` to compute the same score-weighted same-class detection feature
distance that the old backend-side `RetinaHeadFeat` path computed.

## Round Flow

Core-set:

```text
train -> eval -> labeled feature inference -> unlabeled feature inference
      -> k-center acquisition -> common pool update
```

PPAL:

```text
train -> eval -> uncertainty inference -> DCUS candidate pool
      -> candidate feature inference -> CCMS acquisition -> common pool update
```

## Output Files

Core-set writes round-local outputs:

```text
coreset_labeled_features.npz
coreset_unlabeled_features.npz
coreset_candidates.json
coreset_diagnostics.json
annotations/new_labeled.json
annotations/new_unlabeled.json
```

PPAL writes:

```text
ppal_candidate_features.npz
ppal_dcus_candidates.json
ppal_ccms_candidates.json
ppal_diagnostics.json
annotations/uncertainty_pool.json
annotations/new_labeled.json
annotations/new_unlabeled.json
```

The PPAL distance matrix is computed during acquisition and not persisted by
default. Diagnostics record the feature artifact path and distance matrix shape.
Core-set diagnostics store compact distance summaries only. Per-image selection
details are stored in `coreset_candidates.json` through common candidate fields
such as `selected`, `selection_rank`, and `metadata.kcenter_rank`.

## User-Facing Invocation

```powershell
python tools/run_active_learning.py --method coreset --detector retinanet --dataset voc --gpus 1
```

Useful aliases are `coreset`, `core-set`, `core_set`, `kcenter`, and `k-center`.
