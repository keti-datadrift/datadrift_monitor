# EUA-only Batch-16 / LR-0.008 Result Analysis

## Objective

Determine whether the completed batch-16, LR-0.008 VOC EUA-only experiment
diverged in an intermediate round, assess the validity of its active-learning
curve, and compare it with the earlier stable batch-1 EUA-only result.

## Source Runs

- Analyzed run:
  `work_dirs/current_work/retinanet_voc_ecpal_eua_only_7rounds_5percent_to_20percent/09-04-2026_13;08`
- Batch-1 reference:
  `work_dirs/backup_09-03-2026/retinanet_voc_ecpal_eua_only_7rounds_5percent_to_20percent/08-07-2026_18;32`

The analyzed run completed all seven rounds with seed 0, training batch 16,
inference batch 16, AMP, LR 0.008, 32 warm-up iterations, and 26 epochs per
round.

## Evaluation Results

| Round | Labeled images | New mAP | Change from prior round | Batch-1 seed-0 mAP | Difference |
|---:|---:|---:|---:|---:|---:|
| 1 | 827 | 0.343160 | - | 0.446198 | -0.103038 |
| 2 | 1,241 | 0.493091 | +0.149931 | 0.598499 | -0.105408 |
| 3 | 1,655 | 0.581770 | +0.088679 | 0.643367 | -0.061597 |
| 4 | 2,069 | 0.547513 | -0.034257 | 0.668760 | -0.121246 |
| 5 | 2,483 | 0.672222 | +0.124709 | 0.684781 | -0.012559 |
| 6 | 2,897 | 0.692949 | +0.020726 | 0.704586 | -0.011637 |
| 7 | 3,311 | 0.710058 | +0.017109 | 0.711598 | -0.001540 |

The equal-weight seven-round mean is 0.577252 versus 0.636827 for the batch-1
reference, a difference of -0.059575. The final-round difference is only
-0.001540, so the main accuracy cost is concentrated in early and middle
rounds.

## Numerical-Stability Evidence

| Round | Iterations | Non-finite grad norms | Longest consecutive run | Maximum loss | Epoch-26 mean loss |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,378 | 0 | 0 | 2.1330 | 0.4413 |
| 2 | 2,028 | 0 | 0 | 2.0577 | 0.3782 |
| 3 | 2,704 | 0 | 0 | 2.0537 | 0.3386 |
| 4 | 3,406 | 1 | 1 | 1.9804 | 0.3839 |
| 5 | 4,056 | 0 | 0 | 2.0391 | 0.2705 |
| 6 | 4,732 | 0 | 0 | 2.0141 | 0.2463 |
| 7 | 5,408 | 0 | 0 | 2.0543 | 0.2483 |

The only non-finite gradient norm occurred at round-4 iteration 1,147. Loss
was 0.9928 on that iteration; the adjacent iterations had finite gradient
norms and losses of 0.9724 and 0.9024. Dynamic loss scaling therefore recovered
immediately, and the 20-consecutive-event divergence guard correctly did not
trigger. Round-3, round-4, and round-5 checkpoints contain no non-finite model
parameters. Round-4 `class_quality` remains finite with mean 0.363587, rather
than collapsing toward zero as it did in the invalid LR-0.032 run.

Every round reached its lowest mean training loss at epoch 26. Round 4 did end
with a higher mean loss than round 3 despite using more labeled images, which
is consistent with a harder acquired pool or an optimization outlier but not
with runaway numerical divergence.

## Round-4 Error Pattern

The mAP decrease is broad rather than a parser or single-class error:

- 15 of 20 class AP values decreased from round 3 to round 4.
- The largest decreases were bird -0.134, cow -0.108, cat -0.092, bicycle
  -0.067, bus -0.061, and sheep -0.060.
- Mean recall increased from 0.8986 to 0.9108 while detections increased from
  444,103 to 494,073 (+11.3%). Mean AP decreased from 0.5818 to 0.5474.
- Round 5 reduced detections to 358,628 and recovered mAP to 0.672222.

This measured pattern indicates transiently poorer precision or confidence
ranking in the round-4 checkpoint, not a failed evaluator. The evidence does
not identify a unique cause from one seed.

## Acquisition-Trajectory Difference

The new and batch-1 runs share the same initial seed-0 pool, but their selected
batches diverge quickly:

| Acquisition after round | Same selected images | Overlap | Cumulative labeled-pool Jaccard |
|---:|---:|---:|---:|
| 1 | 209 / 414 | 50.5% | 0.7165 |
| 2 | 108 / 414 | 26.1% | 0.6330 |
| 3 | 49 / 414 | 11.8% | 0.6026 |
| 4 | 46 / 414 | 11.1% | 0.5775 |
| 5 | 38 / 414 | 9.2% | 0.5676 |
| 6 | 40 / 414 | 9.7% | 0.5603 |

Consequently, the performance difference is not a pure detector-training
comparison after round 1; changed detector scores alter EUA selections and all
later labeled pools.

## Runtime

| Stage | Batch-1 seed 0 | New seed 0 | Speedup |
|---|---:|---:|---:|
| Training | 23.39 h | 3.37 h | 6.94x |
| Evaluation | 44.6 min | 21.3 min | 2.10x |
| Acquisition inference | 134.4 min | 52.3 min | 2.57x |
| Acquisition computation | 6.3 min | 2.7 min | 2.35x |
| Recorded total | 26.48 h | 4.64 h | 5.70x |

The total comparison also includes protocol-era differences such as skipping
terminal acquisition in the new run, so the training-stage speedup is the
cleaner throughput measurement.

## Interpretation and Validity

The analyzed run did not numerically diverge. Round 4 is a real transient
performance regression under a stable training trace. The run remains valid as
a single-seed PoC result under the new fast protocol, but the round-4 cause
cannot be separated into acquired-pool difficulty, finite-sample optimizer
variation, and model confidence calibration using one seed.

The new protocol trades substantially faster iteration for weaker early-round
accuracy. Its final mAP nearly matches the older batch-1 run, but it must not be
mixed directly with old method results. Every baseline used to compare EUA
should be rerun with the same batch-16/LR-0.008 protocol. A second seed is the
minimum useful follow-up if deciding whether the round-4 dip is systematic;
repeating seed 0 is not independent evidence.

No configuration or runtime code was changed in this analysis.

## Validation Performed

- Parsed all seven evaluation JSON files and both aggregate summaries.
- Parsed every training scalar and console record from the seed-level
  TensorBoard stream.
- Counted non-finite gradients and longest consecutive non-finite runs.
- Inspected training loss by epoch and checkpoint tensor finiteness.
- Parsed the VOC per-class evaluation tables for rounds 3 through 5.
- Compared stage durations and selected-image identities with the batch-1
  reference run.

No model was retrained or reevaluated.
