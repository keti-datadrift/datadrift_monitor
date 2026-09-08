# EUA-full Fast-Protocol Result Analysis

## Objective

Analyze the completed VOC EUA-full run under the current fast PoC protocol and
determine whether explicit EUA-profile diversity reranking improves upon
EUA-only. Compare both ECPAL variants with PAL under the same protocol.

## Source Runs

- EUA-full:
  `work_dirs/current_work/retinanet_voc_ecpal_eua_full_7rounds_5percent_to_20percent/09-05-2026_13;34`
- EUA-only:
  `work_dirs/current_work/retinanet_voc_ecpal_eua_only_7rounds_5percent_to_20percent/09-04-2026_13;08`
- PAL:
  `work_dirs/current_work/retinanet_voc_pal_7rounds_5percent_to_20percent/09-04-2026_19;37`

All three runs use seed 0, seven rounds, a 414-image acquisition budget,
training batch 16, inference batch 16, AMP, and LR 0.008. They use the same
827-image initial labeled pool. Terminal-round acquisition is skipped.

## Measured Evaluation Results

| Round | Labeled images | EUA-full mAP | EUA-only mAP | Full - only | PAL mAP | Full - PAL |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 827 | 0.343160 | 0.343160 | +0.000000 | 0.343160 | +0.000000 |
| 2 | 1,241 | 0.479134 | 0.493091 | -0.013958 | 0.546944 | -0.067811 |
| 3 | 1,655 | 0.583117 | 0.581770 | +0.001346 | 0.617543 | -0.034426 |
| 4 | 2,069 | 0.620589 | 0.547513 | +0.073076 | 0.644175 | -0.023585 |
| 5 | 2,483 | 0.673148 | 0.672222 | +0.000925 | 0.675272 | -0.002124 |
| 6 | 2,897 | 0.676078 | 0.692949 | -0.016870 | 0.696474 | -0.020396 |
| 7 | 3,311 | 0.710689 | 0.710058 | +0.000631 | 0.703466 | +0.007223 |

EUA-full improves monotonically across all seven rounds. Its round-to-round
increments are +0.135974, +0.103983, +0.037473, +0.052559, +0.002930, and
+0.034611. Round 6 is a plateau, not a regression.

| Curve summary | EUA-full | EUA-only | PAL |
|---|---:|---:|---:|
| Equal-weight seven-round mean | 0.583702 | 0.577252 | 0.603862 |
| Trapezoidal curve mean | 0.593165 | 0.585693 | 0.617287 |
| Final mAP | 0.710689 | 0.710058 | 0.703466 |

EUA-full exceeds EUA-only by 0.006450 on the equal-weight mean and 0.007473 on
the trapezoidal mean. This apparent curve gain is dominated by EUA-only's
round-4 regression: excluding round 4, the mean per-round Full-minus-only
difference is -0.004654. At the final budget, the difference is only +0.000631.

PAL remains the strongest method over the full acquisition curve. EUA-full has
the highest final value, exceeding PAL by 0.007223, but it is below PAL in
rounds 2 through 6.

## Measured Training Stability

| Round | Optimizer iterations | Non-finite grad norms | Maximum loss | Final logged loss |
|---:|---:|---:|---:|---:|
| 1 | 1,378 | 0 | 2.1330 | 0.3981 |
| 2 | 2,054 | 0 | 2.1095 | 0.3891 |
| 3 | 2,704 | 0 | 2.0433 | 0.4460 |
| 4 | 3,406 | 0 | 2.0147 | 0.2802 |
| 5 | 4,056 | 0 | 2.0529 | 0.1897 |
| 6 | 4,732 | 0 | 2.0528 | 0.2325 |
| 7 | 5,382 | 0 | 2.0257 | 0.2303 |

No non-finite gradient event was logged. The finite losses and monotonic mAP
curve provide no evidence of numerical divergence.

## Measured Effect of Diversity Reranking

EUA-full ranks all unlabeled images by EUA, keeps the top 828 candidates, and
selects 414 of them by farthest-first Jensen-Shannon distance over the
three-component `(pi_cls, pi_loc, pi_miss)` error profile. The following table
compares its selected set with the plain top-414 EUA candidates produced by the
same EUA-full checkpoint in each round. This isolates the reranker from later
model-trajectory differences.

| Acquisition after round | Selected from top 414 | Mean selected EUA rank | Pairwise JS increase | Nearest-neighbor JS increase | Mean EUA score reduction |
|---:|---:|---:|---:|---:|---:|
| 1 | 214 / 414 | 399.55 | +26.6% | +66.5% | -12.1% |
| 2 | 221 / 414 | 392.56 | +25.4% | +56.3% | -10.8% |
| 3 | 217 / 414 | 403.48 | +22.7% | +55.1% | -9.3% |
| 4 | 212 / 414 | 405.18 | +24.1% | +58.0% | -9.8% |
| 5 | 217 / 414 | 387.25 | +29.4% | +58.6% | -7.6% |
| 6 | 222 / 414 | 387.57 | +27.5% | +56.2% | -7.5% |

The reranker is not a no-op. It consistently replaces nearly half of the plain
top-414 EUA set, materially increases error-profile diversity, and accepts a
lower mean EUA score in exchange.

Actual EUA-full versus EUA-only selected-image overlap is:

| Acquisition after round | Same images | Overlap |
|---:|---:|---:|
| 1 | 214 / 414 | 51.7% |
| 2 | 111 / 414 | 26.8% |
| 3 | 70 / 414 | 16.9% |
| 4 | 51 / 414 | 12.3% |
| 5 | 53 / 414 | 12.8% |
| 6 | 44 / 414 | 10.6% |

Only round 1 is a direct selection comparison from an identical model. Later
overlaps also include the compounding effect of different labeled pools and
checkpoints.

Selected ground-truth annotation density for EUA-full is 8.08, 6.87, 6.18,
5.65, 4.97, and 4.80 annotations per image over acquisition rounds 1 through
6. EUA-only produces 9.39, 7.15, 5.79, 5.14, 5.18, and 4.34. Diversity
reranking reduces the extreme object-density bias in the first two rounds, but
the direction is not consistent in later rounds.

## Measured Runtime

| Stage | EUA-full | EUA-only | Full - only |
|---|---:|---:|---:|
| Training | 12,091.818 s | 12,135.553 s | -43.735 s |
| Evaluation | 1,258.458 s | 1,276.413 s | -17.955 s |
| Acquisition inference | 3,130.902 s | 3,137.873 s | -6.971 s |
| Acquisition computation | 158.692 s | 162.314 s | -3.622 s |
| Recorded total | 16,639.870 s | 16,712.153 s | -72.283 s |

EUA-full took approximately 4 h 37 min 20 s. Its runtime is effectively the
same as EUA-only; the measured 72-second difference is ordinary run-to-run
variation rather than evidence that the more complex method is faster.

## Interpretation

The EUA-full experiment is a valid, numerically stable single-seed PoC result.
Explicit EUA-profile diversity clearly changes the selected set and increases
the exact diversity quantity that its reranker optimizes. Despite that strong
intervention, it provides no practically meaningful final-mAP improvement over
EUA-only: +0.000631 mAP. The full-curve advantage comes almost entirely from
avoiding EUA-only's isolated round-4 regression, while EUA-full is worse in
rounds 2 and 6.

For the current research direction, the evidence favors retaining EUA-only:
the additional reranking complexity did not improve final utility under this
protocol. However, this experiment alone does not prove that EUA-only images
are naturally diverse. It shows that adding this specific three-dimensional
error-profile diversity objective is not useful for final mAP in seed 0.
Supporting a broader natural-image-diversity claim still requires direct
diversity measurements using an independent representation, such as visual
embedding dispersion, semantic/category coverage, and redundancy or nearest-
neighbor similarity, with PAL and random sampling as references.

One seed is insufficient to claim statistical equivalence or to determine
whether EUA-only's round-4 regression is systematic. The +0.000631 final
difference is too small to treat as a method-level win.

No runtime code or configuration was changed during this analysis.

## Validation Performed

- Confirmed `done` status, seven round summaries, terminal acquisition skip,
  and final metrics from the EUA-full run and aggregate summaries.
- Parsed all round metrics for EUA-full, EUA-only, and PAL.
- Parsed every EUA-full training loss and gradient-norm scalar from the
  seed-level TensorBoard event stream.
- Compared selected image IDs and annotation density across all six
  acquisition rounds.
- Recomputed pairwise and nearest-neighbor Jensen-Shannon distances from each
  EUA-full candidate file using the repository's diversity implementation.
- Summed recorded training, evaluation, inference, and acquisition durations.
- Ran `git diff --check` after adding this analysis record.

No model was retrained or reevaluated.
