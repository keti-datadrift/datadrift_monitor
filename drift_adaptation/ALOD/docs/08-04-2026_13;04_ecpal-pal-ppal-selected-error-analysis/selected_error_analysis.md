# ECPAL, PAL, PPAL Selected Error Analysis

## Purpose

This note summarizes the completed RetinaNet + VOC active learning runs for
PAL, PPAL, and ECPAL, then compares the actual GT/error composition of the
images selected by each method.

The main question is:

> Why does ECPAL outperform PAL/PPAL around rounds 4-6, but fall slightly behind
> in the final round?

## Analyzed Runs

The analysis uses the completed 3-seed runs below.

| Method | Run directory | Seeds | Rounds | Budget |
|---|---|---:|---:|---:|
| PAL | `work_dirs/retinanet_voc_pal_7rounds_5percent_to_20percent/07-28-2026_17;23` | 0, 1, 2 | 7 | 414 |
| PPAL | `work_dirs/retinanet_voc_ppal_7rounds_5percent_to_20percent/07-30-2026_12;34` | 0, 1, 2 | 7 | 414 |
| ECPAL | `work_dirs/retinanet_voc_ecpal_7rounds_5percent_to_20percent/08-02-2026_19;46` | 0, 1, 2 | 7 | 414 |

The stale ECPAL directory
`work_dirs/retinanet_voc_ecpal_7rounds_5percent_to_20percent/07-31-2026_22;44`
was excluded because `seed_0/run_summary.json` had `status=running` and no
completed round summaries.

## Performance Summary

ECPAL completed all 7 rounds for all 3 seeds.

| Round | Train labeled images | Labeled after acquisition | ECPAL mAP mean | ECPAL mAP std | ECPAL AP50 mean | ECPAL AP50 std |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 827 | 1241 | 0.4677 | 0.0154 | 0.4677 | 0.0156 |
| 2 | 1241 | 1655 | 0.5914 | 0.0019 | 0.5913 | 0.0017 |
| 3 | 1655 | 2069 | 0.6394 | 0.0048 | 0.6393 | 0.0049 |
| 4 | 2069 | 2483 | 0.6690 | 0.0015 | 0.6690 | 0.0014 |
| 5 | 2483 | 2897 | 0.6849 | 0.0037 | 0.6850 | 0.0037 |
| 6 | 2897 | 3311 | 0.6988 | 0.0015 | 0.6990 | 0.0016 |
| 7 | 3311 | 3725 | 0.7047 | 0.0008 | 0.7050 | 0.0008 |

Final ECPAL mAP by seed:

| Seed | Final mAP | Final AP50 |
|---:|---:|---:|
| 0 | 0.7056 | 0.7060 |
| 1 | 0.7049 | 0.7050 |
| 2 | 0.7037 | 0.7040 |

PAL, PPAL, and ECPAL mAP by round:

| Round | PAL | PPAL | ECPAL |
|---:|---:|---:|---:|
| 1 | 0.4677 | 0.4677 | 0.4677 |
| 2 | 0.5929 | 0.5939 | 0.5914 |
| 3 | 0.6402 | 0.6365 | 0.6394 |
| 4 | 0.6659 | 0.6658 | 0.6690 |
| 5 | 0.6843 | 0.6841 | 0.6849 |
| 6 | 0.6947 | 0.6966 | 0.6988 |
| 7 | 0.7086 | 0.7071 | 0.7047 |

ECPAL is strongest in rounds 4-6, but the final round is lower than PAL by
0.0038 mAP and lower than PPAL by 0.0023 mAP.

## Round Gain Interpretation

The runner order is:

1. Train with the current labeled pool.
2. Evaluate the trained detector.
3. Run acquisition to build the next labeled pool.

Therefore, the round 7 mAP is not the result of the images selected at round 7.
It is the result of training with the labeled pool produced by acquisition at
round 6.

The round 6 to round 7 gain is:

| Method | Round 6 mAP | Round 7 mAP | Gain |
|---|---:|---:|---:|
| PAL | 0.6947 | 0.7086 | +0.0138 |
| PPAL | 0.6966 | 0.7071 | +0.0105 |
| ECPAL | 0.6988 | 0.7047 | +0.0059 |

This means ECPAL's round 6 selected images produced less next-round training
gain than the images selected by PAL and PPAL, even though ECPAL was ahead at
round 6.

## Selected Error Recalculation Protocol

The selected-image analysis uses a single method-agnostic protocol.

Selected images:

```text
selected_ids(round_i) = round_i/new_labeled.json - previous_labeled_pool
```

GT source:

```text
data/VOC0712/annotations/trainval_0712.json
```

Detector prediction source:

| Method | Prediction artifact |
|---|---|
| PAL | `pal_unlabeled_detections.bbox.json` |
| PPAL | `diversity_inference_result.bbox.json` |
| ECPAL | `ecpal_unlabeled_features.json` / `final_detections` |

For PPAL, `diversity_inference_result.bbox.json` was used instead of the full
`unlabeled_inference_result.bbox.json` because the full file is very large and
the final selected images are contained in the diversity candidate pool.

The same score threshold and matching rules were used for all methods:

| Item | Value |
|---|---:|
| Detection score threshold | 0.3 |
| Background/foreground IoU threshold | 0.1 |
| Localization IoU threshold | 0.5 |

The error definitions follow the ECPAL TIDE-style label definition, not full
TIDE:

| Error | Definition |
|---|---|
| Foreground detection | `max(u_same, u_diff) >= 0.1` |
| Classification error | foreground detection with `u_diff > u_same` |
| Localization error | foreground detection with `max(u_same, u_diff) < 0.5` |
| Missed object | GT object not explained by any foreground detection |

This is sufficient for fair method-to-method comparison because the same
definition is applied to PAL, PPAL, and ECPAL.

## Selected GT and Error Count

The table below reports 3-seed averages over the 414 newly selected images at
each round.

| Round | PAL GT/img | PAL err/img | PPAL GT/img | PPAL err/img | ECPAL GT/img | ECPAL err/img |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.246 | 6.383 | 6.610 | 6.598 | 8.916 | 10.282 |
| 2 | 4.850 | 4.923 | 6.285 | 5.304 | 6.407 | 7.031 |
| 3 | 4.676 | 4.667 | 5.774 | 4.891 | 5.251 | 5.952 |
| 4 | 4.427 | 4.304 | 5.031 | 4.233 | 4.796 | 5.362 |
| 5 | 4.519 | 4.229 | 4.858 | 3.934 | 4.235 | 4.984 |
| 6 | 4.191 | 4.009 | 4.444 | 3.473 | 4.065 | 4.531 |
| 7 | 4.042 | 3.840 | 4.266 | 3.345 | 3.884 | 4.229 |

ECPAL selects the highest actual error count per image in every round. This
confirms that the error-count prediction objective is behaving as intended:
ECPAL is selecting images where the current detector makes many mistakes.

However, higher selected error count does not automatically translate into
higher next-round mAP gain.

## Error Type Breakdown

The table below reports actual error counts per selected image.

| Round | PAL cls | PAL loc | PAL miss | PPAL cls | PPAL loc | PPAL miss | ECPAL cls | ECPAL loc | ECPAL miss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.919 | 1.398 | 2.066 | 2.420 | 1.271 | 2.907 | 3.978 | 3.191 | 3.113 |
| 2 | 2.019 | 1.331 | 1.573 | 1.647 | 1.280 | 2.377 | 2.787 | 2.315 | 1.928 |
| 3 | 1.938 | 1.308 | 1.421 | 1.581 | 1.279 | 2.031 | 2.514 | 1.911 | 1.527 |
| 4 | 1.786 | 1.277 | 1.241 | 1.457 | 1.176 | 1.600 | 2.202 | 1.730 | 1.429 |
| 5 | 1.665 | 1.197 | 1.367 | 1.307 | 1.079 | 1.548 | 2.244 | 1.637 | 1.103 |
| 6 | 1.626 | 1.188 | 1.195 | 1.185 | 0.913 | 1.375 | 1.889 | 1.467 | 1.175 |
| 7 | 1.501 | 1.191 | 1.148 | 1.165 | 0.853 | 1.327 | 1.824 | 1.337 | 1.068 |

ECPAL consistently selects more classification and localization error than PAL
or PPAL. PPAL tends to select more missed-object-heavy images than PAL in most
rounds. ECPAL is strongest on detector-facing foreground confusion, especially
classification and localization error.

## ECPAL Candidate and Diversity Behavior

For full ECPAL, each round builds `2 * budget = 828` candidates and then selects
414 images using Jensen-Shannon diversity over predicted error composition.

| Round | Selected from top-414 ECA | Selected mean rank | Selected score mean | cls_hat | loc_hat | miss_hat |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 228.7 | 387.4 | 5.593 | 3.966 | 1.581 | 1.469 |
| 2 | 276.3 | 332.1 | 6.563 | 2.751 | 2.454 | 1.444 |
| 3 | 230.0 | 404.6 | 5.825 | 2.379 | 2.351 | 0.958 |
| 4 | 177.0 | 437.9 | 5.385 | 2.036 | 2.143 | 1.013 |
| 5 | 210.0 | 388.7 | 5.403 | 2.053 | 2.086 | 1.064 |
| 6 | 212.0 | 398.1 | 4.898 | 1.821 | 1.872 | 0.994 |
| 7 | 238.0 | 375.2 | 4.606 | 1.699 | 1.752 | 0.934 |

ECPAL does not simply select the top 414 images by ECA score. Roughly half of
the selected images can come from outside the top 414 because of diversity.

This appears helpful in the middle rounds, where exploration can still find
useful new samples. In later rounds, it may become too exploratory and displace
high-value ECA samples that would have produced better next-round mAP gain.

## Predictor Diagnostics

The ECPAL predictor fallback count was zero for every seed and round. This means
the scikit-learn predictors trained normally and the result is not explained by
predictor fallback behavior.

The selected actual error density decreases over rounds:

| Round | ECPAL GT/img | ECPAL cls/img | ECPAL loc/img | ECPAL miss/img |
|---:|---:|---:|---:|---:|
| 4 | 4.796 | 2.202 | 1.730 | 1.429 |
| 5 | 4.235 | 2.244 | 1.637 | 1.103 |
| 6 | 4.065 | 1.889 | 1.467 | 1.175 |
| 7 | 3.884 | 1.824 | 1.337 | 1.068 |

As the active learning process progresses, the remaining unlabeled pool has
fewer high-error samples. Candidate score differences likely become smaller,
and ranking noise or diversity tradeoffs become more important.

## Runtime Notes

ECPAL runtime was dominated by training, not acquisition.

Average time by round:

| Round | Train min | Eval min | Labeled inference min | Unlabeled inference min | Acquisition min |
|---:|---:|---:|---:|---:|---:|
| 1 | 34.16 | 2.75 | 0.64 | 7.67 | 0.22 |
| 2 | 49.07 | 2.70 | 0.84 | 7.48 | 0.22 |
| 3 | 64.01 | 2.82 | 1.04 | 7.28 | 0.21 |
| 4 | 78.89 | 2.79 | 1.24 | 7.08 | 0.20 |
| 5 | 93.82 | 2.82 | 1.44 | 6.91 | 0.20 |
| 6 | 109.05 | 2.68 | 1.65 | 6.73 | 0.19 |
| 7 | 123.62 | 2.68 | 1.84 | 6.51 | 0.18 |

Sequential runtime was about 10.5 hours per seed and 31.6 hours for 3 seeds.

The optimized JS diversity implementation is no longer a practical bottleneck.

## Interpretation

The evidence supports two separate conclusions.

First, ECPAL's core acquisition objective works. It selects images with more
actual detector errors than PAL or PPAL, especially classification and
localization errors. This is exactly what error-count prediction is supposed to
capture.

Second, the final mAP gap suggests that the number of selected errors is not
the only factor controlling active learning performance. The late-round samples
selected by ECPAL contained more measured errors, but those errors converted
into less next-round mAP gain than PAL/PPAL.

The most plausible explanation is:

1. In early and middle rounds, selecting error-rich images is highly beneficial.
2. In later rounds, many easy/high-impact error-rich samples have already been
   selected.
3. ECPAL still finds images with many errors, but some may be redundant,
   outlier-like, weakly AP-relevant, or too heavily chosen for diversity.
4. PAL/PPAL may select fewer total errors, but their selected samples in the
   late rounds may contain errors that produce more effective decision-boundary
   correction.

In short:

> ECPAL is good at finding many detector errors, but final AL performance also
> depends on whether those errors are high-value training signals.

## Recommended Follow-up Experiments

The next experiments should isolate candidate quality from diversity quality.

1. Compare `pal:lius` and `ecpal:eca`.
   - Both should select exactly `budget` images from the uncertainty/error score.
   - This tests whether ECPAL's candidate scoring is better than PAL LIUS
     without the diversity stage.

2. Compare full ECPAL against `ecpal:eca`.
   - If `ecpal:eca` closes or reverses the final-round gap, JS diversity is the
     likely source of the late-round loss.
   - If full ECPAL remains better, the diversity stage is still useful and the
     issue is likely score composition or late-round weighting.

3. Analyze round 6 selected images by class and scene duplication.
   - Round 6 acquisition is the direct source of round 7 model performance.
   - This is where ECPAL's final gain is much lower than PAL/PPAL.

4. Consider a late-round schedule for diversity.
   - Keep ECA candidate scoring unchanged.
   - Reduce JS diversity strength in later rounds, or use a larger fraction of
     strict top-ECA images.

5. Consider adding an uncertainty/confusion term to ECA.
   - Expected error count should remain the main score.
   - A small entropy/confusion bonus may prioritize decision-boundary-correcting
     samples over redundant high-error images.

