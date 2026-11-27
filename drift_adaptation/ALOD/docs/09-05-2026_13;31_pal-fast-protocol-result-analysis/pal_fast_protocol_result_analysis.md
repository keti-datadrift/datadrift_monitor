# PAL Fast-Protocol Result Analysis

## Objective

Analyze the completed VOC PAL run under the current fast PoC protocol, compare
it with the matching EUA-only run, and choose the next experiment that most
directly tests whether EUA already provides sufficient implicit diversity.

## Source Runs

- PAL:
  `work_dirs/current_work/retinanet_voc_pal_7rounds_5percent_to_20percent/09-04-2026_19;37`
- Matching EUA-only:
  `work_dirs/current_work/retinanet_voc_ecpal_eua_only_7rounds_5percent_to_20percent/09-04-2026_13;08`
- Historical PAL seed-0 reference:
  `work_dirs/backup_09-03-2026/retinanet_voc_pal_7rounds_5percent_to_20percent/07-28-2026_17;23`

The PAL and matching EUA-only runs both use seed 0, seven rounds, a 414-image
acquisition budget, training batch 16, inference batch 16, AMP, and LR 0.008.
The initial labeled set contains 827 images and each non-terminal round adds
414 images. Terminal-round acquisition is skipped.

## Measured Evaluation Results

| Round | Labeled images | PAL mAP | EUA-only mAP | PAL - EUA-only |
|---:|---:|---:|---:|---:|
| 1 | 827 | 0.343160 | 0.343160 | +0.000000 |
| 2 | 1,241 | 0.546944 | 0.493091 | +0.053853 |
| 3 | 1,655 | 0.617543 | 0.581770 | +0.035772 |
| 4 | 2,069 | 0.644175 | 0.547513 | +0.096661 |
| 5 | 2,483 | 0.675272 | 0.672222 | +0.003050 |
| 6 | 2,897 | 0.696474 | 0.692949 | +0.003525 |
| 7 | 3,311 | 0.703466 | 0.710058 | -0.006591 |

PAL improves monotonically in all seven rounds. Its equal-weight seven-round
mean is 0.603862, compared with 0.577252 for EUA-only, a difference of
+0.026610. Its trapezoidal curve mean is 0.617287, compared with 0.585693 for
EUA-only. PAL is therefore stronger through most of the acquisition trajectory,
while EUA-only is higher by 0.006591 at the final budget.

## Measured Training Stability

| Round | Optimizer iterations | Non-finite grad norms | Maximum loss | Final logged loss |
|---:|---:|---:|---:|---:|
| 1 | 1,378 | 0 | 2.1330 | 0.3981 |
| 2 | 2,028 | 0 | 2.1383 | 0.3489 |
| 3 | 2,730 | 0 | 2.1304 | 0.2609 |
| 4 | 3,380 | 0 | 2.0945 | 0.2620 |
| 5 | 4,056 | 0 | 2.0915 | 0.2167 |
| 6 | 4,732 | 0 | 2.0853 | 0.2335 |
| 7 | 5,408 | 0 | 2.0443 | 0.2589 |

No non-finite gradient events were logged. Together with the monotonic mAP
curve, this provides no evidence of the intermediate-round divergence seen in
the earlier invalid LR-0.032 experiment.

## Measured Acquisition Difference

| Acquisition after round | Same selected images | Overlap | PAL selected annotations/image | EUA-only selected annotations/image |
|---:|---:|---:|---:|---:|
| 1 | 93 / 414 | 22.5% | 4.59 | 9.39 |
| 2 | 38 / 414 | 9.2% | 4.39 | 7.15 |
| 3 | 30 / 414 | 7.2% | 4.04 | 5.79 |
| 4 | 28 / 414 | 6.8% | 4.59 | 5.14 |
| 5 | 23 / 414 | 5.6% | 4.67 | 5.18 |
| 6 | 25 / 414 | 6.0% | 4.41 | 4.34 |

The two methods select substantially different image batches. EUA-only selects
far more object-dense images in the early rounds; this difference narrows by
round 6. Annotation count is not itself a diversity metric, but it confirms
that the methods create materially different labeled-pool trajectories.

## Measured Runtime

| Stage | PAL | EUA-only | PAL difference |
|---|---:|---:|---:|
| Training | 12,061.952 s | 12,135.553 s | -73.601 s |
| Evaluation | 1,244.215 s | 1,276.413 s | -32.198 s |
| Acquisition inference | 3,080.806 s | 3,137.873 s | -57.067 s |
| Acquisition computation | 34.573 s | 162.314 s | -127.741 s |
| Recorded total | 16,421.546 s | 16,712.153 s | -290.607 s |

PAL finished in approximately 4 h 33 min 42 s, about 4 min 51 s faster than
EUA-only. Training and inference costs are similar; most method-specific time
saving comes from acquisition computation.

## Interpretation

The PAL result is a valid, stable single-seed PoC result under the current fast
protocol. PAL has the better early-to-middle active-learning curve and avoids
EUA-only's round-4 regression. EUA-only nevertheless obtains the higher final
mAP by about 0.66 percentage points. One seed is insufficient to establish
that either final-round difference is systematic.

This result does not yet support the claim that EUA alone makes explicit
diversity unnecessary. The cleanest next comparison is EUA-full: it holds the
EUA uncertainty signal fixed and adds the implemented EUA-profile diversity
reranking. Its full seven-round trajectory will distinguish the following
possibilities:

- EUA-full exceeds EUA-only: explicit diversity adds value.
- EUA-full approximately matches EUA-only: EUA ranking may already supply
  enough diversity for this protocol.
- EUA-full underperforms EUA-only: diversity reranking disrupts useful EUA
  ordering.

The next command is:

```powershell
python -B tools/run_active_learning.py --method ecpal:eua-full --detector retinanet --dataset voc --gpus 1
```

No runtime code or configuration was changed as part of this analysis.

## Compatibility Note

The historical PAL seed-0 run used the older experiment protocol. Its mAP was
0.446198, 0.599725, 0.638669, 0.663163, 0.679415, 0.688455, and 0.707821.
The current fast run is substantially weaker early but finishes within
-0.004355 mAP at round 7. Old-protocol results must not be mixed into direct
method comparisons with the current runs.

## Validation Performed

- Confirmed completion and final metrics from the PAL aggregate and run
  summaries.
- Parsed all seven PAL and EUA-only round metrics.
- Inspected seed-level TensorBoard scalars for iteration counts, losses, and
  non-finite gradient events.
- Compared selected image identities and annotation counts for all six
  acquisition rounds.
- Summed recorded stage durations for both matching runs.
- Confirmed `ecpal:eua-full` is a registered method alias in the experiment
  catalog and active-learning runner help.

No model was retrained or reevaluated.
