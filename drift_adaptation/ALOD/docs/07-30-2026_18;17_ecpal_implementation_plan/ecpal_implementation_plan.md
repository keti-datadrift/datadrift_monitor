# ECPAL Implementation Plan

## Goal

Implement `ECPAL` (Error-Count Prediction-Based Active Learning for Object Detection) as a first-class ALOD method for RetinaNet + VOC, while preserving the current ALOD architecture:

- method-specific algorithm logic stays in `methods/ecpal/`
- method-neutral runtime helpers stay in `methods/common/`
- runner/preparation helpers stay in `tools/common/`
- detector artifact export stays in `mmdet/alod/`
- no import from `code_refs/`
- no extra tests unless explicitly requested later

ECPAL should run through the same user-facing runner:

```bash
python tools/run_active_learning.py --method ecpal --detector retinanet --dataset voc --gpus 1
```

## Algorithm Summary

One active-learning round follows this flow:

1. Train RetinaNet on the current labeled pool.
2. Evaluate the current checkpoint on VOC test.
3. Run ECPAL inference on the labeled pool.
4. Run ECPAL inference on the unlabeled pool.
5. Use labeled detections and GT to train four lightweight predictors:
   - FDP: Foreground Detection Predictor
   - CECP: Classification-Error Count Predictor
   - LECP: Localization-Error Count Predictor
   - MOCP: Missed-Object Count Predictor
6. Predict image-level error-count profiles for the unlabeled pool.
7. Select candidate images by weighted Error-Count Amount (ECA).
8. Select the final budget from candidates by Error-Count Diversity (ECD).
9. Update labeled/unlabeled pools with existing common pool helpers.

## Required Detector Feature Artifact

ECPAL should follow PAL's compact artifact style. PAL does not save all raw pre-NMS detections; it uses them inside the detector head and stores only the computed `pre_nms_count` with each final detection. ECPAL should do the same: use raw pre-NMS detections during inference, compute the required support/residual features immediately, and save only compact per-image/per-final-detection features.

Add ECPAL-specific image-level feature artifacts:

```text
ecpal_labeled_features.json
ecpal_unlabeled_features.json
```

Each image record should be compact and explicit:

```json
{
  "image_id": 123,
  "final_detections": [
    {
      "bbox": [x, y, w, h],
      "category_id": 7,
      "score": 0.91,
      "p_max": 0.91,
      "A_cls": 0.67,
      "n_sup": 3,
      "mu_iou": 0.74
    }
  ],
  "miss_features": {
    "R_amt": 12.4,
    "R_prob": 0.36
  }
}
```

Raw detections are not persisted. They are used only inside `RetinaHeadECPAL` to compute:

- final-detection features: `p_max`, `A_cls`, `n_sup`, `mu_iou`
- image-wise missed-object features: `R_amt`, `R_prob`

For RetinaNet + VOC, use the same raw detection scope as PAL's pre-NMS count computation unless a later experiment explicitly changes it:

- `score_thr=0.3`
- `nms_pre=1000`
- `max_per_img=200`
- final NMS IoU threshold `0.5`
- support/pre-NMS matching IoU threshold `0.5`

This keeps ECPAL directly comparable to PAL and avoids writing large raw-detection JSON files.

## New Method Files

Create:

```text
methods/ecpal/
  __init__.py
  inference.py
  features.py
  labels.py
  predictors.py
  scoring.py
  diversity.py
  acquisition.py
```

File roles:

- `inference.py`
  - load and validate ECPAL image-level feature JSON
  - normalize image ids
  - expose grouped image records to the sampler

- `features.py`
  - define pure feature computation helpers used by detector-export validation or lightweight offline checks
  - mirror the detector-side feature definitions:
    - `p_max`
    - `A_cls`
    - `n_sup`
    - `mu_iou`
  - compute MOCP residual features:
    - `R_amt`
    - `R_prob`

- `labels.py`
  - generate TIDE-style labels from labeled final detections and GT:
    - `y_fg`
    - `y_cls`
    - `y_loc`
    - `N_miss`
  - default thresholds:
    - foreground IoU `t_f=0.5`
    - background IoU `t_b=0.1`

- `predictors.py`
  - orchestrate FDP/CECP/LECP/MOCP fitting and prediction
  - use common predictor models, not PAL-specific imports

- `scoring.py`
  - compute expected counts:
    - `N_cls_hat`
    - `N_loc_hat`
    - `N_miss_hat`
  - compute inverse-scale weights
  - compute weighted ECA
  - compute weighted error-count composition

- `diversity.py`
  - compute JS distance / ECD
  - run farthest-first selection
  - first selected image is max ECA
  - tie-break by `ECA desc`, then stable image-id order

- `acquisition.py`
  - provide `sample_ecpal_from_files(...)`
  - coordinate loading, predictor training, scoring, candidate selection, final selection
  - return:
    - `selected_image_ids`
    - `candidate_records`
    - compact diagnostics

## Common Code Changes

Reuse existing common helpers directly:

```text
methods/common/coco_pool.py      # pool reads and next-round split
methods/common/candidates.py     # candidate artifact format/write
methods/common/results.py        # diagnostics envelope
methods/common/detections.py     # JSON loading pattern
methods/common/image_identity.py # canonical ids
methods/common/selection.py      # deterministic ranking/tie-break/refill
methods/common/matching.py       # bbox IoU helpers
methods/common/io.py             # JSON I/O
```

Add or extend common code only when it is method-neutral:

```text
methods/common/predictors.py
```

Replace PAL's current NumPy-only `BinaryLogisticModel` with a common scikit-learn-backed predictor wrapper in `methods/common/predictors.py`, then make PAL import that common wrapper. Add a scikit-learn-backed Poisson count wrapper for ECPAL MOCP.

The change must preserve PAL's current functional behavior:

- same input features and targets
- same probability semantics
- same constant-probability fallback when samples are too few or labels are single-class
- no method-to-method import from ECPAL to PAL

Add `scikit-learn` to the minimal runtime requirements because both PAL and ECPAL will use the common predictor wrappers.

Do not import from `methods/pal` inside `methods/ecpal`. Method-to-method coupling should be avoided.

Extend common helpers if useful:

- `methods/common/matching.py`
  - IoU matrix or best-overlap helpers for image-level detection/GT matching

- `methods/common/vectors.py`
  - probability normalization
  - JS divergence/distance if later useful outside ECPAL

- `methods/common/selection.py`
  - generic farthest-first selection only if it can be cleanly method-neutral

Keep ECPAL-specific ECA/ECD naming and diagnostics in `methods/ecpal/`.

## MMDetection Export Changes

Add:

```text
mmdet/alod/models/retinanet/ecpal_head.py
mmdet/alod/datasets/ecpal_feature_coco.py
configs/alod_mmdet/retinanet_voc_infer_ecpal_detections.py
```

Responsibilities:

- `RetinaHeadECPAL`
  - use PAL-aligned raw pre-NMS candidates internally
  - compute support-set features for each final detection
  - compute image-wise residual features for MOCP
  - return only compact feature records plus final bbox/category/score
  - perform no ECPAL scoring

- `ECPALFeatureCocoDataset`
  - write the image-level ECPAL feature artifact
  - keep JSON schema stable and compact

- `ALRetinaNet`
  - minimally support the ECPAL head result structure
  - avoid embedding method scoring inside detector code

Registration updates:

- add the ECPAL dataset to `mmdet/alod/datasets/__init__.py`
- add the ECPAL head to `mmdet/alod/models/retinanet/__init__.py`

## Catalog Changes

Update `configs/catalog/methods.py`:

- add aliases:
  - `ecpal`
  - `ecpal/full`
- add method spec:
  - method: `ecpal`
  - output: `retinanet_voc_ecpal_7rounds_5percent_to_20percent`
  - rounds: `7`
  - budget: `414`
- add default method config:
  - `ecpal_candidate_expand_ratio=2`
  - `ecpal_foreground_iou_threshold=0.5`
  - `ecpal_background_iou_threshold=0.1`
  - `ecpal_eps=1e-12`
  - `ecpal_weight_eps=1e-6`
  - `ecpal_diagnostics_file='ecpal_diagnostics.json'`
  - `ecpal_candidates_file='ecpal_candidates.json'`
  - `ecpal_labeled_features='ecpal_labeled_features.json'`
  - `ecpal_unlabeled_features='ecpal_unlabeled_features.json'`

Update `configs/catalog/detectors.py`:

- add `ecpal_infer_config` to `DetectorSpec`
- set RetinaNet value:
  - `configs/alod_mmdet/retinanet_voc_infer_ecpal_detections.py`

Update `configs/catalog/experiments.py`:

- add preset name:
  - `ecpal-retinanet-voc`

## Runner Changes

Update `tools/run_active_learning.py`:

- import:
  - `from methods.ecpal.acquisition import sample_ecpal_from_files`

- add helpers:
  - `_ecpal_feature_json(cfg, output_dir, round_index, pool_name)`
  - `_ecpal_infer_plan(cfg, output_dir, round_index, input_paths, pool_name)`

- update `build_round_plan(...)`:

```text
train
eval
ecpal labeled inference
ecpal unlabeled inference
ecpal acquisition
```

- update progress labels and progress totals:
  - labeled inference total = labeled pool image count
  - unlabeled inference total = unlabeled pool image count

- update `_execute_lightweight_acquisition(...)`:
  - call `sample_ecpal_from_files(...)`
  - write `ecpal_candidates.json` with `methods.common.candidates`
  - write `ecpal_diagnostics.json` with `methods.common.results`
  - update pools with `write_next_round_pool_split(...)`

The runner should not know ECPAL internals beyond file paths and method config values.

## Saved Outputs

Per round, ECPAL should save:

```text
round_XX/
  ecpal_labeled_features.json
  ecpal_unlabeled_features.json
  ecpal_candidates.json
  ecpal_diagnostics.json
  annotations/
    new_labeled.json
    new_unlabeled.json
  logs/
    ecpal_labeled_inference.log
    ecpal_unlabeled_inference.log
```

Candidate artifact should follow the common schema:

- `method='ecpal'`
- `stage='ecd'`
- `score=ECA`
- `components`:
  - `n_cls_hat`
  - `n_loc_hat`
  - `n_miss_hat`
  - `weighted_eca`
  - `pi_cls`
  - `pi_loc`
  - `pi_miss`
- `metadata`:
  - `candidate_rank`
  - `selection_rank`
  - optional nearest selected distance for ECD analysis

Diagnostics should be compact:

- predictor sample counts
- positive rates for FDP/CECP/LECP
- mean labeled true error counts
- scale weights
- candidate count
- selected count
- fallback counts for constant predictors

Do not save large intermediate matrices unless explicitly needed.

## Predictor Fallback Rules

Small labeled pools can make binary/Poisson fitting unstable. Use explicit deterministic fallback, not silent broad exception handling. The predictor implementation should use scikit-learn for the normal path and preserve the current NumPy PAL logistic behavior at the API/semantic level.

- Binary logistic:
  - use scikit-learn `LogisticRegression` for PAL LIUS, ECPAL FDP, ECPAL CECP, and ECPAL LECP
  - if samples are too few or labels are single-class, return constant smoothed probability
  - keep the same feature normalization/probability meaning as PAL's current NumPy implementation
  - keep PAL's LIUS score behavior functionally unchanged

- Poisson count:
  - use scikit-learn `PoissonRegressor` for ECPAL MOCP
  - if samples are too few or targets are degenerate, return constant mean count
  - clamp predicted lambda to a finite range

Record fallback usage in diagnostics.

## Implementation Order

1. Add `scikit-learn` to runtime requirements.
2. Add common scikit-learn-backed predictor wrappers.
3. Make PAL use `methods.common.predictors.BinaryLogisticModel` without changing PAL LIUS behavior.
4. Add ECPAL detector artifact export:
   - `RetinaHeadECPAL`
   - `ECPALFeatureCocoDataset`
   - `retinanet_voc_infer_ecpal_detections.py`
5. Add `methods/ecpal/inference.py` and schema validation.
6. Add ECPAL feature and label generation.
7. Add ECPAL predictors and scoring.
8. Add ECPAL candidate and diversity selection.
9. Add `methods/ecpal/acquisition.py`.
10. Add catalog entries.
11. Add runner planning/execution branch.
12. Run syntax/import validation only; leave full experiment execution to the user.

## Subagent Work Split For Implementation

Use subagents only for disjoint, bounded scopes:

- Artifact Agent
  - write compact `mmdet/alod` feature-export code and ECPAL inference config
  - write scope: `mmdet/alod/**`, `configs/alod_mmdet/retinanet_voc_infer_ecpal_detections.py`

- Common Boundary Agent
  - replace PAL's NumPy logistic implementation with common scikit-learn-backed wrappers
  - add ECPAL Poisson predictor wrapper
  - update minimal runtime requirements
  - write scope: `methods/common/predictors.py`, `methods/pal/lius.py`, `requirements*.txt`

- Method Logic Agent
  - implement `methods/ecpal/features.py`, `labels.py`, `predictors.py`, `scoring.py`, `diversity.py`, `inference.py`
  - write scope: `methods/ecpal/**`

- Runner Integration Agent
  - add catalog and runner integration
  - write scope: `configs/catalog/**`, `tools/run_active_learning.py`

- Review Agent
  - read-only pass checking:
    - method/common boundaries
    - no `methods/ecpal` import from `methods/pal`
    - no `code_refs` imports
    - no large unnecessary outputs
    - runner output remains user-friendly

## Open Decisions

Before coding, decide these defaults explicitly:

1. Raw detection scope:
   - use PAL's pre-NMS count scope for the initial ECPAL implementation
   - do not save raw detections

2. Exact-box exclusion:
   - document defines strict equality
   - implementation should probably use `box_equal_tol=1e-6` because JSON/tensor conversions can introduce tiny coordinate differences

3. Candidate ratio:
   - use `delta=2`, aligned with PAL's class-candidate multiplier style rather than PPAL's DCUS pool expansion

4. ECPAL final diversity:
   - recommended: JS distance over weighted error-count composition
   - first selected image is max ECA

5. Dependency policy:
   - add `scikit-learn`
   - use scikit-learn for PAL and ECPAL regressors/classifiers
   - keep functional behavior equivalent to the current NumPy PAL logistic implementation

## Validation Plan

Do not run full AL experiments during implementation validation.

Use targeted checks:

```bash
python -m py_compile methods/common/predictors.py
python -m py_compile methods/ecpal/*.py
python -m py_compile tools/run_active_learning.py
python tools/run_active_learning.py --method ecpal --detector retinanet --dataset voc --rounds 1 --gpus 1 --verbose
```

The final command should only be used if it does not launch heavy training unexpectedly after runner behavior is confirmed. Full `--method ecpal` execution should remain a user-run experiment.
