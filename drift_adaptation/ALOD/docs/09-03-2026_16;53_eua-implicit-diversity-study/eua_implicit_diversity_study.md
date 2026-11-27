# EUA Implicit-Diversity Study Design

## Objective

Support a bounded empirical claim that ECPAL EUA-only can select informative
and sufficiently diverse image batches without using an explicit diversity
stage in the proposed method.

Recommended claim wording:

> Although EUA optimizes a pointwise predictive-error uncertainty score, its
> top-ranked batches exhibit implicit diversity across visual, semantic, and
> error-type spaces. Explicit diversity re-ranking provides no consistent
> additional coverage or downstream AP gain under the evaluated protocol.

This must remain dataset/detector/protocol-specific unless it is reproduced on
multiple benchmarks and detectors. A scalar EUA score does not mathematically
guarantee batch diversity.

## Existing Evidence and Available Artifacts

The completed legacy VOC experiments already show:

- EUA-only final mAP: 0.7109.
- PAL full final mAP: 0.7086.
- PPAL final mAP: 0.7071.
- ECA-only final mAP: 0.7064.
- ECA-full final mAP: 0.7047.

This is preliminary evidence that an explicit diversity method is not required
to obtain the best local performance. It does not isolate whether EUA itself
produces diverse batches because the methods follow different pools after the
first acquisition.

The current result tree contains enough data for a no-training retrospective
study:

- `work_dirs/pal_embeddings/voc_google_vit_embeddings.npy` contains a
  768-dimensional embedding for all 16,551 VOC train-pool images.
- Every EUA-only seed/round contains `ecpal_unlabeled_features.json`, EUA
  diagnostics, candidate records, and the selected pool transition.
- Oracle VOC annotations can be used strictly for post-hoc semantic analysis;
  they must not be used by the acquisition method.

## Two Claims That Must Be Separated

### Claim A: EUA batches are implicitly diverse

This is a set-property claim. It needs representation-, semantic-, and
error-profile diversity measurements on selected batches.

### Claim B: Explicit diversity is unnecessary for performance

This is a causal performance claim. Diversity metrics alone cannot establish
it. EUA-only must be compared with an explicit diversity re-ranking under the
same checkpoint, unlabeled pool, budget, initialization, and training protocol.

## Experiment 1: Shared-State Offline Diversity Audit

Use saved EUA-only checkpoints/pools at every round and seed. From each exact
unlabeled state, create fixed-size `K=414` query sets:

1. EUA top-K: the proposed method.
2. ECA top-K: pointwise error-amount comparator.
3. Random K: repeat at least 100 times to form a null distribution.
4. ViT k-center/Core-set from the whole pool: explicit visual-coverage
   comparator.
5. EUA top-2K followed by an explicit diversity re-ranking: uncertainty-matched
   comparator.

All methods must operate on the same frozen state. Existing selections from
different active-learning trajectories may be reported descriptively, but only
round 1 is directly matched across those trajectories.

### Visual/representation metrics

Compute these in both the frozen ViT space and, if practical, the detector
backbone feature space:

- Mean pairwise cosine distance: higher indicates greater dispersion.
- Mean nearest-neighbor distance within the batch: higher indicates less local
  redundancy.
- Pool-to-query mean and 95th-percentile nearest distance: lower indicates
  better coverage of the unlabeled pool.
- K-center radius: lower indicates better worst-case coverage, but report it
  with mean/95th-percentile coverage because a single outlier can dominate the
  maximum.
- Cluster occupancy and normalized cluster entropy for multiple cluster counts,
  such as 20, 50, and 100. Conclusions must not depend on one chosen K.

Do not use mean pairwise distance alone. It can reward isolated outliers while
failing to represent the pool.

### Semantic metrics, post-hoc only

- Number of covered object classes.
- Object-class entropy and effective number of classes.
- Jensen-Shannon distance between the selected and unlabeled-pool class
  distributions; lower means more representative, while entropy measures
  spread.
- Coverage of object-count, object-scale, and class-co-occurrence strata.
- Near-duplicate rate based on the full nearest-neighbor similarity
  distribution rather than an uncalibrated single threshold.

### EUA-native error-profile metrics

For each image, normalize `[U_cls, U_loc, U_miss]` into an error-uncertainty
composition and report:

- pairwise Jensen-Shannon distance;
- entropy of the dominant error-type distribution;
- coverage across predicted classes and error types;
- concentration of the top clusters/classes in the batch.

Because these metrics use the score's own components, they must be accompanied
by the independent ViT/detector-feature metrics to avoid a circular argument.

### Analysis

- Report each metric per round and seed, not only a pooled average.
- Place EUA against the random null distribution as a percentile or standardized
  effect.
- Plot the trade-off between mean EUA score and coverage for all query sets.
- Use paired round/seed differences and bootstrap confidence intervals.
- Include image grids only as qualitative support, not as primary evidence.

## Experiment 2: Paired Downstream-Gain Ablation

A full EUA-full trajectory is the cleanest comparison, but a cheaper paired
branch experiment can test the mechanism without adopting diversity as the
method.

At representative early, middle, and late EUA-only states, for example rounds
1, 4, and 6, and for every seed:

1. Keep the saved detector, labeled pool, and unlabeled pool fixed.
2. Branch A uses the existing EUA top-K set.
3. Branch B takes the EUA top-2K candidates and applies one explicit diversity
   re-ranking, preferably the native error-profile JS rule or ViT k-center.
4. Train the next-round detector with identical initialization, schedule, and
   seed.
5. Compare next-round mAP/AP50 gain and the diversity metrics from Experiment 1.

The existing Branch-A next-round checkpoints may be reused if all settings are
identical, so only the diversity branches require additional training. This is
nine additional branch trainings for three rounds and three seeds rather than a
new 21-training seven-round trajectory.

Predefine a practically meaningful non-inferiority margin before examining the
branch results and report paired confidence intervals. Three seeds are a
minimum; five are preferable for a strong statistical claim.

## Experiment 3: Current-Protocol Main Comparison

The old completed runs share one initial labeled pool across all seeds. Current
seed-specific runs are a different protocol. For a defensible new main table,
rerun at least:

- PAL full as an established uncertainty-plus-diversity baseline;
- EUA-only as the proposed method;
- the explicit EUA diversity comparator used in the ablation.

For a complete ECPAL two-by-two analysis, also run ECA-only and ECA-full. If only
VOC is evaluated, constrain the claim to RetinaNet/VOC. A broader claim requires
COCO replication and ideally another detector.

## Decision Rules

The strong claim is supported if:

1. EUA top-K has consistently low redundancy and competitive pool coverage in
   independent feature spaces.
2. The result is stable across rounds, seeds, and reasonable cluster counts.
3. EUA retains substantially higher informativeness than pure diversity
   selection.
4. Explicit diversity re-ranking yields no consistent or practically meaningful
   next-round AP improvement.

If EUA has high AP but materially lower diversity than random/k-center, use the
narrower claim that explicit diversity was unnecessary for performance, not the
claim that EUA naturally produced diverse images. If EUA is diverse only in its
own three-component error profile, call it error-type diversity rather than
general image diversity.

## Literature Context

- Core-set formulates batch selection as feature-space coverage and motivates
  pool-to-selected nearest-distance and k-center radius measurements:
  <https://arxiv.org/abs/1708.00489>
- BADGE explicitly evaluates uncertainty and batch diversity in one embedding
  space, including Gram-determinant diversity, and motivates reporting the
  informativeness/diversity trade-off:
  <https://arxiv.org/abs/1906.03671>
- PPAL is the closest object-detection comparator: it uses uncertainty filtering
  followed by CCMS and k-means++ diversity selection:
  <https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Plug_and_Play_Active_Learning_for_Object_Detection_CVPR_2024_paper.html>

## Validation Performed

This design was checked against the current experiment inventory and artifact
schemas. The ViT cache shape and EUA round artifacts were inspected directly.
No diversity metrics, training branches, or experiment outputs were generated
or modified in this work item.
