# EUA Method and Error-Predictor Audit

## Objective and Research Decision

Document the exact current EUA-only method and test whether its four auxiliary
predictors generalize to the unlabeled acquisition pool. The earlier implicit-
diversity claim is dropped. This analysis focuses on what EUA actually measures,
which components work, and which assumptions need revision.

## Source Run and Evaluation Design

Primary run:

`work_dirs/current_work/retinanet_voc_ecpal_eua_only_7rounds_5percent_to_20percent/09-04-2026_13;08`

The run uses seed 0, RetinaNet/VOC, six acquisitions of 414 images, training
batch 16, inference batch 16, AMP, and LR 0.008. Round 7 has no acquisition, so
predictor evaluation covers acquisition rounds 1 through 6.

For each round, this audit:

1. loaded that round's exact labeled and unlabeled ECPAL feature artifacts;
2. rebuilt labels and refit the four predictors with the production code;
3. used the VOC oracle only after fitting to label the complete unlabeled pool;
4. evaluated predictor ranking, calibration, and image-level count accuracy;
5. rebuilt the EUA scores and checked the resulting top-414 IDs against the
   saved acquisition result.

All six reconstructed top-414 lists exactly match the saved lists. Oracle
annotations were used only for retrospective evaluation and did not affect the
original acquisition.

## Current EUA Method

### Detector feature extraction

ECPAL inference uses final detections produced with score threshold 0.3, NMS
IoU 0.5, and at most 200 detections per image. For each final detection `d`, it
saves:

- `p_max`: maximum class probability;
- `A_cls`: fraction of supporting pre-NMS candidates that predict the same
  class;
- `n_sup`: number of supporting pre-NMS candidates;
- `mu_iou`: mean IoU of those supporting candidates with the final box.

A pre-NMS candidate supports the final box to which it has maximum overlap
when that overlap is at least 0.5. The final box itself is excluded with a
coordinate tolerance of `1e-6`.

For each image, the missed-object features are:

```text
rho_j  = maximum IoU of pre-NMS candidate j with any final detection
R_amt  = sum_j (1 - rho_j)
R_prob = sum_j (1 - rho_j) * score_j / (R_amt + eps)
```

`R_amt` measures the amount of pre-NMS evidence not covered by final
detections. `R_prob` is its confidence-weighted average score.

### Retrospective error labels

For labeled images, every final detection is compared with ground truth:

```text
y_fg  = 1[max IoU with any GT >= 0.1]
y_cls = 1[max IoU with a different-class GT > max IoU with a same-class GT]
y_loc = 1[max IoU with any GT < 0.5], conditional on y_fg = 1
```

An image's classification and localization counts are the sums of `y_cls` and
`y_loc` over foreground detections. A GT is considered explained when it is the
best match of any detection with IoU at least 0.1; remaining GTs form
`N_miss`.

These are TIDE-style proxy categories, not disjoint standard TIDE errors. There
is no one-to-one detection/GT assignment, classification and localization
labels can overlap, and a wrong-class low-IoU detection can prevent a GT from
being counted as missed. This label definition is especially important when
interpreting MOCP.

### Four predictors

The predictors are refit from scratch in every acquisition round:

| Predictor | Model | Inputs | Target/output |
|---|---|---|---|
| FDP | logistic regression | `p_max, A_cls, n_sup, mu_iou` | `q_fg = P(y_fg=1)` |
| CECP | logistic regression | `p_max, A_cls` | `q_cls = P(y_cls=1 | y_fg=1)` |
| LECP | logistic regression | `n_sup, mu_iou` | `q_loc = P(y_loc=1 | y_fg=1)` |
| MOCP | Poisson regression | `R_amt, R_prob` | `lambda_miss = E[N_miss]` |

Features are standardized on the current labeled set. Logistic models use
LBFGS with L2 regularization `1e-4`; MOCP uses a Poisson regressor with the same
regularization strength. No predictor required a constant fallback in the six
analyzed rounds.

The fitted standardized coefficients are structurally stable:

- CECP: `p_max` coefficient from -0.525 to -0.395 and `A_cls` from -2.138 to
  -1.650. Low confidence and, especially, low class agreement imply higher
  classification-error probability.
- LECP: `n_sup` from -5.140 to -2.512 and `mu_iou` from +0.115 to +0.375. Low
  support count dominates the localization prediction; the much smaller
  positive `mu_iou` term may reflect collinearity rather than a causal effect.
- FDP: `n_sup` is the dominant positive coefficient, from +3.149 to +6.366.
- MOCP: `R_amt` is positive, from +0.370 to +0.516, while `R_prob` remains near
  zero, from -0.104 to +0.041. MOCP effectively relies on residual amount.

### ECA counts and EUA score

The same predictors first imply expected error counts:

```text
N_cls_hat  = sum_d q_fg(d) * q_cls(d)
N_loc_hat  = sum_d q_fg(d) * q_loc(d)
N_miss_hat = lambda_miss
```

EUA replaces error probability/count with distribution entropy:

```text
U_cls  = sum_d q_fg(d) * H_Bernoulli(q_cls(d))
U_loc  = sum_d q_fg(d) * H_Bernoulli(q_loc(d))
U_miss = H_Poisson(lambda_miss)
EUA     = w_cls * U_cls + w_loc * U_loc + w_miss * U_miss
```

For component `k`, `w_k` is inverse to its mean `U_k` on the labeled pool and
the three weights are normalized to sum to 3. This makes the average weighted
component contributions equal on the current labeled set. EUA-only selects the
top 414 images by this scalar score without reranking.

This is predictive-distribution entropy, not parameter or epistemic uncertainty
of the auxiliary models. There is no ensemble, bootstrap variance, posterior,
or confidence interval. FDP entropy is not included; `q_fg` only gates the
classification and localization entropies. Poisson entropy is monotonic in
`lambda_miss`, so `U_miss` is largely a transformed predicted miss amount.

## Out-of-Pool Binary Predictor Performance

Brier skill is measured against a constant predictor set to the labeled-set
prevalence: `1 - model Brier / baseline Brier`. Higher is better.

| Predictor | Pool target prevalence | ROC-AUC range | AP range | Brier-skill range | Assessment |
|---|---:|---:|---:|---:|---|
| FDP | 0.934-0.963 | 0.709-0.750 | 0.974-0.984 | 0.015-0.046 | Weak incremental value over the dominant-positive baseline |
| CECP | 0.242-0.400 | 0.840-0.889 | 0.641-0.720 | 0.342-0.412 | Strong and useful |
| LECP | 0.178-0.190 | 0.645-0.777 | 0.248-0.362 | 0.039-0.090 | Weak early, moderate late |

FDP's high AP is caused mainly by its 93-96% positive rate. It predicts a mean
foreground probability of 0.980-0.993, systematically above the pool's true
0.934-0.963 rate. Its average Brier skill is only 0.034.

CECP is the strongest predictor. Its mean pool ROC-AUC is 0.868 and mean Brier
skill is 0.384. It generalizes with a consistent calibration shift: after the
first round, its mean predicted error probability is about 0.035-0.043 below
the actual pool error rate.

LECP improves from ROC-AUC 0.645 in round 1 to 0.777 in round 6, but its mean
Brier skill is only 0.060. Round 1 is badly undercalibrated: predicted mean
0.081 versus true rate 0.186. Later mean rates are closer.

Training-set AUC is higher than pool AUC for every predictor. FDP falls from
0.878-0.929 in-sample to 0.709-0.750 on the pool; CECP falls from 0.896-0.960
to 0.840-0.889; LECP falls from 0.755-0.929 to 0.645-0.777. The labeled pool is
an adaptively selected distribution, and the current predictors have no
held-out calibration step to handle this shift.

## Image-Level Error-Count Accuracy

The density baseline predicts classification/localization counts as the number
of final detections multiplied by the labeled-set error prevalence. MOCP is
compared with the constant labeled-set mean miss count.

| Component | Pool R2 range | Spearman range | Mean model MAE | Mean baseline MAE | Assessment |
|---|---:|---:|---:|---:|---|
| Classification | 0.594-0.800 | 0.744-0.834 | 0.391 | 0.761 | Strong beyond image density |
| Localization | 0.131-0.270 | 0.441-0.509 | 0.557 | 0.548 | No average improvement over density baseline |
| Miss | -0.057-0.008 | 0.073-0.337 | 0.937 | 1.018 | Small MAE gain but almost no per-image explanatory power |

Classification-count prediction is genuinely useful and not explained only by
having more detections. Localization count has modest correlation, but the
feature-based model's average MAE is slightly worse than the simple density
baseline. MOCP has approximately zero R2 in every round. Its mean prediction
underestimates pool misses from 0.580 versus 1.217 in round 1; this gap narrows
to 0.597 versus 0.617 by round 6 because misses become rarer, not because
image-level ranking becomes strong.

## What the EUA Score Selects

### Component composition

| Acquisition round | Classification share | Localization share | Miss share |
|---:|---:|---:|---:|
| 1 | 38.5% | 43.9% | 17.6% |
| 2 | 34.7% | 44.3% | 21.0% |
| 3 | 34.6% | 44.8% | 20.7% |
| 4 | 32.5% | 42.6% | 24.9% |
| 5 | 32.2% | 47.6% | 20.2% |
| 6 | 32.7% | 47.9% | 19.4% |

Localization uncertainty is the largest selected-score component. Miss
uncertainty still contributes 18-25% even though MOCP has almost no pool-level
count accuracy.

Component uncertainty has the following Spearman correlation with its own
absolute count-prediction error:

- `U_cls`: 0.487-0.710;
- `U_loc`: 0.695-0.793;
- `U_miss`: 0.188-0.372.

LECP is not a strong expected-count model, but its entropy aggregate is useful
for finding images on which its localization-count prediction is inaccurate.
The miss component is weak both as a count predictor and as an uncertainty
indicator.

### Density dependence and residual signal

| Round | rho(EUA, detection count) | rho(EUA, actual total error) | Partial rho(error | detection count) | rho(EUA, absolute predictor residual) | Partial rho(residual | detection count) | Top-414 actual-error lift |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.950 | 0.662 | 0.223 | 0.581 | 0.186 | 3.99x |
| 2 | 0.938 | 0.678 | 0.259 | 0.603 | 0.216 | 3.31x |
| 3 | 0.924 | 0.658 | 0.275 | 0.614 | 0.292 | 3.09x |
| 4 | 0.937 | 0.693 | 0.287 | 0.579 | 0.207 | 2.84x |
| 5 | 0.912 | 0.667 | 0.283 | 0.684 | 0.350 | 3.13x |
| 6 | 0.905 | 0.657 | 0.309 | 0.698 | 0.396 | 2.98x |

Top-414 absolute prediction-residual lift ranges from 2.04x to 3.26x, but top-
414 detection-count lift is also 2.66x to 4.39x. EUA overlaps the simple top-
detection-count set by 219-348 of 414 images. Therefore, current EUA is mainly
an exposure-weighted score: summing entropy over detections strongly favors
images containing many detections.

The non-density signal is real but smaller. After rank-wise control for
detection count, partial correlation with actual total error remains 0.22-0.31
and partial correlation with prediction residual remains 0.19-0.40. EUA's
top-414 has slightly more actual error than top detection-count selection in
every round, by approximately 1.7-7.1%.

Compared on each frozen EUA checkpoint, EUA top-414 usually contains more
actual error and larger predictor residual than counterfactual ECA top-414.
This occurs despite EUA optimizing entropy rather than expected error count.
It is consistent with EUA identifying error-rich difficult images, but density
is a substantial common cause.

## Overall Assessment

The answer to whether the error predictors work is mixed:

- **CECP works well.** It has strong pool ranking, meaningful Brier improvement,
  and strong image-level classification-count accuracy beyond a density-only
  baseline.
- **LECP is partially useful.** Its binary ranking becomes moderate and its
  entropy strongly tracks localization-count residual, but expected
  localization count is no better than a detection-density baseline on average.
- **MOCP does not predict image-level misses well.** It has approximately zero
  R2 and weak ranking. Its current two features and loose miss target are not
  sufficient.
- **FDP contributes little discrimination.** Almost every retained detection
  meets the IoU-0.1 foreground proxy, so the model mostly outputs a value near
  one and is systematically overconfident on the unlabeled pool.

Current EUA should therefore not be described as the uncertainty of four
well-performing error predictors. A more accurate description is:

> EUA sums classification- and localization-error probability entropies over
> final detections, gates them with a nearly saturated foreground predictor,
> and adds a Poisson-entropy term based on a weak missed-object count model.
> It finds dense, error-rich images and retains a smaller error-uncertainty
> signal beyond detection density.

This explains both its strengths and its unstable active-learning curve. The
classification path is strong, localization entropy captures hard cases, and
the selected batches contain many actual errors. At the same time, score
magnitude is dominated by detection count, miss uncertainty is unreliable, and
the adaptively selected labeled set produces calibration shift on the remaining
pool.

## Recommended EUA Research Sequence

1. **Run component ablations offline first.** Re-rank every saved pool with
   `cls-only`, `loc-only`, `miss-only`, `cls+loc`, and current `cls+loc+miss`.
   Measure actual-error lift, absolute-residual lift, overlap, density, and
   class balance. This requires no detector retraining.
2. **Test no-miss EUA first in downstream training.** MOCP has the clearest
   negative evidence while still receiving 18-25% of selected-score mass.
3. **Separate uncertainty from exposure.** Compare current sum aggregation with
   `sum/sqrt(N_det)`, a capped top-K detection sum, and mean entropy plus a
   separate controlled density term. This determines whether dense-image bias
   is useful or merely confounding.
4. **Use reliability-aware component weights.** The current inverse-mean
   weights equalize scale without considering predictor quality. Estimate
   weights from held-out Brier skill/count skill so an unreliable MOCP cannot
   receive a large contribution solely because its raw scale is small.
5. **Improve labels and features before adding model complexity.** Use a
   one-to-one GT/detection assignment and disjoint error definitions. LECP needs
   richer regression/box-consistency features. MOCP needs spatial and FPN-level
   residual/objectness features beyond `R_amt` and `R_prob`.
6. **If epistemic predictor uncertainty is the intended novelty, measure it
   directly.** Bootstrap or fold ensembles can provide variance/disagreement.
   Current Bernoulli/Poisson entropy is predictive label/count entropy, not
   uncertainty about fitted predictor parameters.
7. **Only retrain multiple seeds after choosing a corrected score.** The present
   seed-0 EUA result is a useful PoC, but round-level method differences are too
   noisy for a final claim.

The highest-priority next implementation target is an offline component and
density-normalization audit using the existing six round artifacts. It can
identify which reduced EUA formula deserves the next full GPU trajectory.

No runtime code, configuration, or experiment output was changed during this
analysis.

## Validation Performed

- Read the detector feature exporter, label construction, predictor wrappers,
  acquisition orchestration, score implementation, and current inference
  configuration.
- Refit production FDP, CECP, LECP, and MOCP models for all six EUA-only
  acquisition rounds.
- Evaluated every predictor against post-hoc oracle labels on each complete
  unlabeled pool.
- Compared in-sample and pool metrics, probability calibration, image-count
  regression, and density baselines.
- Reconstructed ECA/EUA profiles, adaptive weights, scores, and candidate
  ordering; all saved top-414 selections matched exactly.
- Measured score/error, score/residual, score/density, component-residual, and
  density-controlled rank correlations.
- Ran `git diff --check` after adding this analysis record.

No model was trained or reevaluated.
