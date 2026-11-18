# Random and ECA-only Fast-Protocol Result Analysis

## Objective

Analyze the two newly completed VOC runs, Random and ECPAL ECA-only, under the
current fast seed-0 PoC protocol. Compare them with the matching EUA-only,
EUA-full, and PAL runs and reassess the current ECPAL and implicit-diversity
hypotheses.

## Source Runs

- Random:
  `work_dirs/current_work/retinanet_voc_random_7rounds_5percent_to_20percent/09-05-2026_19;11`
- ECA-only:
  `work_dirs/current_work/retinanet_voc_ecpal_eca_only_7rounds_5percent_to_20percent/09-07-2026_15;22`
- EUA-only comparator:
  `work_dirs/current_work/retinanet_voc_ecpal_eua_only_7rounds_5percent_to_20percent/09-04-2026_13;08`
- EUA-full comparator:
  `work_dirs/current_work/retinanet_voc_ecpal_eua_full_7rounds_5percent_to_20percent/09-05-2026_13;34`
- PAL comparator:
  `work_dirs/current_work/retinanet_voc_pal_7rounds_5percent_to_20percent/09-04-2026_19;37`

All five runs use seed 0, seven rounds, a 414-image acquisition budget,
training batch 16, inference batch 16 where inference is required, AMP, and LR
0.008. Their run summaries contain identical initial labeled and unlabeled
image-ID hashes. A direct diff of the copied Random and ECA-only training
configs found only the expected annotation-path and work-directory changes.

## Measured Evaluation Results

| Round | Labeled images | Random | ECA-only | EUA-only | EUA-full | PAL |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 827 | 0.343160 | 0.343160 | 0.343160 | 0.343160 | 0.343160 |
| 2 | 1,241 | 0.428089 | 0.376202 | 0.493091 | 0.479134 | 0.546944 |
| 3 | 1,655 | 0.591605 | 0.563356 | 0.581770 | 0.583117 | 0.617543 |
| 4 | 2,069 | 0.630857 | 0.600668 | 0.547513 | 0.620589 | 0.644175 |
| 5 | 2,483 | 0.649495 | 0.673580 | 0.672222 | 0.673148 | 0.675272 |
| 6 | 2,897 | 0.687789 | 0.697140 | 0.692949 | 0.676078 | 0.696474 |
| 7 | 3,311 | 0.693590 | 0.703707 | 0.710058 | 0.710689 | 0.703466 |

| Curve summary | Random | ECA-only | EUA-only | EUA-full | PAL |
|---|---:|---:|---:|---:|---:|
| Equal-weight seven-round mean | 0.574941 | 0.565402 | 0.577252 | 0.583702 | 0.603862 |
| Mean over active rounds 2-7 | 0.613571 | 0.602442 | 0.616267 | 0.623793 | 0.647312 |
| Trapezoidal curve mean | 0.584368 | 0.572397 | 0.585693 | 0.593165 | 0.617287 |
| Final mAP | 0.693590 | 0.703707 | 0.710058 | 0.710689 | 0.703466 |

### ECA-only versus Random

ECA-only minus Random is 0.000000, -0.051887, -0.028249, -0.030188,
+0.024085, +0.009352, and +0.010117 across rounds 1 through 7. ECA-only is
worse in rounds 2 through 4 and better in rounds 5 through 7. It gains 0.010117
at the final budget but loses 0.009539 on the equal-weight curve mean and
0.011971 on the trapezoidal mean.

### EUA versus ECA and Random

EUA-only exceeds ECA-only by 0.006350 at round 7 and by 0.011850 on the
equal-weight seven-round mean. The difference is not consistent at every
round: EUA-only is much higher in round 2, while its isolated round-4
regression makes it lower there.

EUA-only exceeds Random by 0.016468 at round 7, but its equal-weight mean gain
is only 0.002311 and its trapezoidal mean gain is 0.001324. EUA-full exceeds
Random by 0.017099 at round 7 and 0.008762 on the equal-weight mean. PAL exceeds
Random in every active round and has the largest curve-mean gain, +0.028921.

## Measured Numerical Stability

### Random

| Round | Iterations | Non-finite grad norms | Maximum loss | Final logged loss |
|---:|---:|---:|---:|---:|
| 1 | 1,378 | 0 | 2.1330 | 0.3981 |
| 2 | 2,054 | 0 | 2.1355 | 0.4317 |
| 3 | 2,730 | 0 | 2.2194 | 0.2507 |
| 4 | 3,380 | 0 | 2.1239 | 0.3799 |
| 5 | 4,056 | 0 | 2.1421 | 0.2247 |
| 6 | 4,732 | 0 | 2.1403 | 0.1858 |
| 7 | 5,408 | 0 | 2.1810 | 0.1659 |

### ECA-only

| Round | Iterations | Non-finite grad norms | Maximum loss | Final logged loss |
|---:|---:|---:|---:|---:|
| 1 | 1,378 | 0 | 2.1330 | 0.3981 |
| 2 | 2,028 | 0 | 2.0948 | 0.3576 |
| 3 | 2,704 | 0 | 2.0642 | 0.3805 |
| 4 | 3,380 | 0 | 2.0425 | 0.3866 |
| 5 | 4,082 | 0 | 2.0979 | 0.2924 |
| 6 | 4,732 | 0 | 2.0759 | 0.2623 |
| 7 | 5,408 | 0 | 2.0239 | 0.1967 |

Both runs are numerically stable. ECA-only's round-2 mAP of 0.376202 is a real
low-performing checkpoint rather than numerical divergence: its losses and
gradient norms are finite, and subsequent rounds recover normally. All ECA
predictors report no fallback in any acquisition round.

## Measured Selection Characteristics

### Object density and overlap

| Acquisition after round | Random annotations/image | ECA annotations/image | EUA annotations/image | ECA-Random overlap | ECA-EUA overlap |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.03 | 8.06 | 9.39 | 17 / 414 | 317 / 414 |
| 2 | 2.73 | 6.57 | 7.15 | 7 / 414 | 118 / 414 |
| 3 | 2.85 | 5.48 | 5.79 | 7 / 414 | 50 / 414 |
| 4 | 3.02 | 5.14 | 5.14 | 5 / 414 | 42 / 414 |
| 5 | 2.83 | 4.82 | 5.18 | 7 / 414 | 35 / 414 |
| 6 | 2.68 | 4.54 | 4.34 | 6 / 414 | 43 / 414 |

ECA and EUA strongly prioritize object-dense images. Their round-1 selections
overlap by 317 images because they score the same frozen detector and use
related error predictors. Their later paths diverge as the different selected
pools produce different checkpoints. Random has little overlap with either
error-driven method.

### Post-hoc semantic diversity

Normalized ground-truth object-class entropy is measured only for analysis;
oracle labels were not used by the acquisition methods.

| Method | Round-1 normalized class entropy | Six-acquisition mean |
|---|---:|---:|
| Random | 0.861857 | 0.847326 |
| ECA-only | 0.701947 | 0.779540 |
| EUA-only | 0.643697 | 0.766670 |
| EUA-full | 0.683051 | 0.763289 |
| PAL | 0.859357 | 0.862081 |

At the shared round-1 pool, Random is also more representative of the pool's
class distribution. Its selected-to-pool class-distribution JS distance is
0.0858, compared with 0.2517 for ECA-only and 0.2572 for EUA-only.

### Post-hoc visual diversity

The cached normalized Google ViT embeddings were used to compute within-batch
cosine distances.

| Method | Round-1 mean pairwise distance | Round-1 mean nearest-neighbor distance | Six-acquisition pairwise mean | Six-acquisition nearest-neighbor mean |
|---|---:|---:|---:|---:|
| Random | 0.615879 | 0.342984 | 0.613523 | 0.344732 |
| ECA-only | 0.600438 | 0.292123 | 0.610630 | 0.327965 |
| EUA-only | 0.603202 | 0.299418 | 0.610390 | 0.322533 |
| EUA-full | 0.605759 | 0.308113 | 0.611908 | 0.326934 |
| PAL | 0.597068 | 0.291401 | 0.611249 | 0.330640 |

Round 1 is the clean paired comparison because every method starts from the
same pool and checkpoint. In that round, EUA-only has lower visual dispersion
and higher local redundancy than the observed Random batch. Later-round means
are descriptive because each method operates on a different remaining pool.

## Measured Runtime

| Stage | Random | ECA-only |
|---|---:|---:|
| Training | 12,062.184 s | 12,146.870 s |
| Evaluation | 1,261.552 s | 1,286.485 s |
| Acquisition inference | 0.000 s | 3,136.406 s |
| Acquisition computation | 4.104 s | 82.063 s |
| Recorded total | 13,327.840 s | 16,651.824 s |

Random took approximately 3 h 42 min 8 s. ECA-only took approximately 4 h 37
min 32 s. Random is about 55 min 24 s faster because it requires no acquisition
inference.

## Interpretation

The completed runs are valid single-seed PoC results. ECA-only eventually
beats Random and finishes almost identically to PAL, but its poor rounds 2
through 4 make it less sample-efficient over the full curve. EUA improves the
final result over ECA by about 0.64 mAP percentage points, which is consistent
with the working hypothesis that predictor uncertainty is more useful than
predicted error amount. The comparison is nevertheless noisy and non-monotonic
across rounds, so one seed is insufficient for a firm method-level claim.

Random is more competitive than the final-round table alone suggests. PAL is
the only evaluated method that beats Random at every active round and clearly
wins on curve-level efficiency. The ECPAL variants obtain the highest final
values but do not consistently beat Random during the trajectory. Therefore,
the current evidence supports a bounded late-budget benefit for EUA, not broad
dominance across labeling budgets.

The observed diversity evidence does not support the proposed strong claim
that EUA naturally selects diverse images. At the only directly matched
selection state, EUA-only is less diverse than Random in both semantic class
distribution and ViT nearest-neighbor dispersion, while selecting over three
times as many object annotations per image. The data instead support a narrower
description: EUA selects dense, predicted-error-uncertain images, and adding
the current error-profile diversity reranker does not improve final mAP.

A proper natural-diversity conclusion still requires multiple Random draws and
pool-coverage metrics. The single observed Random batch is not a null
distribution, and within-batch dispersion alone does not measure how well a
query batch represents the remaining pool.

No runtime code or configuration was changed during this analysis.

## Validation Performed

- Confirmed `done` status, all seven round summaries, and terminal acquisition
  skips for both new runs.
- Verified identical initial-pool membership hashes and compared copied
  training configs.
- Parsed all five matching aggregate summaries and recomputed curve means and
  per-round differences.
- Parsed every Random and ECA-only training loss and gradient-norm scalar from
  their seed-level TensorBoard event streams.
- Verified ECA predictor sample counts and fallback status in all six
  diagnostics files.
- Compared selected-image identities and post-hoc oracle annotation statistics.
- Computed within-batch cosine distances from the cached normalized ViT
  embeddings.
- Summed recorded stage durations.
- Ran `git diff --check` after adding this analysis record.

No model was retrained or reevaluated.
