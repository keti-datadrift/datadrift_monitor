# EPAL Implementation Plan

## Objective and status

Plan the replacement of ECPAL with EPAL (Error Prediction-based Active
Learning), using EUA acquisition exclusively, without FDP or diversity
reranking. This document incorporates the user's corrected name, EPAL.
Runtime implementation has not started; only this plan is added.

Design source: `.chatgpt/error_predictor_feature_design.md` (read as UTF-8).

## Public interface

- Use one method entry, `--method epal`, with no ECA/EUA or only/full mode.
- Add `--epal-components` with exactly these initial choices:
  `cecp`, `lecp`, `mocp`, `cecp+lecp`, `cecp+lecp+mocp`.
- Default to `cecp+lecp+mocp`. Normalize case and component order, reject
  duplicates, unknown names, and unsupported combinations before preparation.
- Store the resolved selection as `epal_components`; CLI overrides explicit
  config, which overrides the default. Reject this option for other methods.
- Use a canonical component suffix in generated output directories, including
  the default, e.g. `retinanet_voc_epal_cecp_lecp_7rounds_5percent_to_20percent`.
  Preserve the timestamp/seed/round hierarchy under `work_dirs/current_work`.
- Resolve output naming after components are resolved. For custom config
  output roots, append a canonical EPAL component subdirectory so variants
  remain separated. Store components explicitly in plans and summaries.
- Old ECPAL names/modes/config keys produce an actionable migration error;
  they must not silently select a new algorithm.

## Feature extraction contract

Keep the existing pre-NMS candidate population, thresholding, NMS, coordinate
system, maximum detections, and support assignment. Candidates remain the
existing class-expanded entries; do not introduce anchor deduplication.
Support candidates are assigned to the maximum-IoU final detection, require
IoU >= 0.5, and exclude same-coordinate boxes at the existing tolerance.

For final detection i, use its exported native `score` (the fifth column of
`det_bboxes`) as s_i. Do not recompute the maximum of its class vector.

For pre-NMS candidate k, use its actual candidate score after existing score
factors as s_k. This can differ from the maximum class probability. Preserve
the unfactored class probability vector separately for classification support.

| Predictor | Input 1 | Input 2 |
|---|---|---|
| CECP | s_i | E_cls = sum over supports k of p[k, predicted_class_i] |
| LECP | s_i | E_loc = sum over supports k of IoU(box_k, box_i) |
| MOCP | R_amt = sum_k (1 - rho_k) | R_conf = sum_k (1 - rho_k) * s_k |

Here rho_k is maximum IoU with any final detection. Classification support
uses the final detection's class column even when the candidate's own label
differs. It is not a same-label count or a maximum probability.

Use sums directly, without log transforms, average/sum switches, or separate
support-count predictor inputs. Keep existing feature standardization.

Boundary behavior: no supports gives E_cls = E_loc = 0; no final detections
gives rho_k = 0; no raw candidates gives R_amt = R_conf = 0. Maintain batched
image identity/order, tensor device/dtype, and perform reductions on GPU before
the existing compact export boundary.

## Artifact schema

Introduce an explicit EPAL feature schema version, shared by exporter and
reader. Each image records ID, final detections (`bbox`, `category_id`,
`score`, `E_cls`, `E_loc`), and miss features (`R_amt`, `R_conf`).

Remove p_max, A_cls, n_sup, mu_iou and R_prob from required runtime features.
Remove score/p_max substitution and missing-feature-to-zero fallback. Validate
version, required fields, finite values, shapes and image coverage.

Old compact ECPAL artifacts lack the class-probability support sums and native
candidate score evidence needed to reconstruct the new features exactly.
Reject them with a re-inference message; do not auto-convert or reuse them as
EPAL artifacts. Existing detector weights remain usable if architecture keys
match; the renamed inference head adds no learned parameters. Fresh feature
inference and predictor fitting are required.

## Predictor fitting and labels

Remove the FDP model, its four-dimensional feature vector, fitting, inference,
diagnostics, and q_fg multiplication. Fit and execute only selected predictors;
inactive predictors are explicitly disabled, not failed or constant models.
All variants can share the same complete feature schema.

Retain logistic CECP/LECP and Poisson MOCP with the existing regularization,
standardization and explicit small-sample/single-class fallback diagnostics.
Keep PAL's shared predictor wrappers compatible.

The feature-design document does not redefine error targets. Preserve current
GT matching, foreground training eligibility (IoU >= 0.1), classification and
localization labels, and missed-object target. Represent foreground eligibility
as a label mask, not an FDP target/model. CECP/LECP still learn conditional
error probabilities on eligible labeled detections. Without FDP, inference
applies their entropy to all retained detections. This extrapolation to
background detections is an explicit limitation, not a newly learned
unconditional probability. Revising labels or adding a new foreground gate
would be a separate experimental change.

## EUA and component ablations

U_cls = sum_i H_Bernoulli(q_cls_i)

U_loc = sum_i H_Bernoulli(q_loc_i)

U_miss = H_Poisson(lambda_miss)

For selected components S, compute each labeled-pool mean mu_k, then:

`a_k = 1 / (mu_k + weight_eps)`

`w_k = |S| * a_k / sum_{j in S} a_j`

`score = sum_{k in S} w_k * U_k`

Single-component weight is exactly 1. Disabled components do not participate
in fitting, entropy evaluation, weight normalization or score. Preserve
finite-value checks and existing stable image-ID tie-breaking. Select top
budget directly, with pool-size capping and unique IDs.

Remove ECA score branches, mode dispatch, diversity implementation, candidate
expansion, and diversity-only composition/selection metadata. Preserve useful
ranked candidate and selected-ID diagnostics under EPAL names. Do not introduce
density normalization or reliability-based weights in this change.

## Files and implementation order

1. Update catalog method/preset definitions and component resolution in
   `configs/catalog/{methods,experiments,detectors}.py` and
   `tools/run_active_learning.py`.
2. Rename `methods/ecpal/` to `methods/epal/`, remove its diversity module,
   update package exports, then implement features, reader, labels, predictors,
   EUA scoring and acquisition with the contracts above.
3. Rename and update `mmdet/alod/models/retinanet/ecpal_head.py`,
   `mmdet/alod/datasets/ecpal_feature_coco.py`, associated package imports,
   `mmdet/alod/models/utils.py`, and `al_retinanet.py`. Use `RetinaHeadEPAL`,
   `EPALFeatureCocoDataset`, and epal-prefixed result keys.
4. Rename VOC and COCO ECPAL inference configs and fix references/registrations.
   COCO changes are limited to integration with the shared renamed exporter;
   do not tune its training/evaluation protocol.
5. Integrate runner stages, artifact filenames, output naming, console labels,
   config-path validation, plans, run/round/aggregate summaries and TensorBoard.
   Validate components before data preparation or training starts.
6. Update README examples and shared dependency error text. Existing NumPy,
   SciPy and scikit-learn dependencies suffice; retain dependencies used by PAL
   and other methods. Preserve past docs and experiment outputs as history.

## Diagnostics

Persist method name, feature schema, active components, exact feature order,
label thresholds, fitting sample counts, positive rates, fallback reasons,
scaler statistics, fitted coefficients/intercepts (or constant model value),
labeled uncertainty means, weights, scores and selected IDs. Reflect relevant
round-level summaries in the existing seed TensorBoard stream. Record disabled
predictors explicitly; do not display their absent metrics as measured zeros.
These records support later predictor audits without introducing a new
evaluation workflow in this implementation.

## Validation and limits

Completed for planning: read the design document, current feature head/exporter,
feature reader, label/predictor/scoring/acquisition modules, catalogs, runner,
README and dependency declarations; searched runtime ECPAL references and EPAL
name collisions. EPAL has no existing runtime name collision.

During implementation: inspect rename/reference completeness, parse changed
Python without bytecode artifacts, check actual CLI help/preset output, validate
resolved configs/imports where the installed environment permits, and run Git
diff checks. Inspect five-component choices, native-score provenance, support
class indexing, empty inputs, selected-only fitting/weighting, and artifact
version checks. Follow repository policy: no new test suite, synthetic/smoke
experiment or temporary run outputs. A real GPU training/acquisition trajectory
is not part of this planning task and numerical predictor gains remain unproven.

Keep VOC batch 16/16, LR 0.008, AMP/FP32 split, seed policy, seven evaluated
pools, terminal acquisition skip and seed-level TensorBoard layout intact.
The implementation is a method/feature protocol change; old and new results
must be identified separately even when training settings match.
