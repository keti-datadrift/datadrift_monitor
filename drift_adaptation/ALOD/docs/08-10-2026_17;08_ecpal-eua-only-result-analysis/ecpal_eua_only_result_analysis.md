# ECPAL EUA-only Result Analysis

## Analysis Target

This document summarizes the server run copied into:

```text
work_dirs/retinanet_voc_ecpal_eua_only_7rounds_5percent_to_20percent/08-07-2026_18;32
```

The run is a completed 3-seed experiment:

| Item | Value |
|---|---|
| Method argument | `ecpal:eua-only` |
| Detector | `retinanet` |
| Dataset | `voc` |
| Rounds | 7 |
| Initial labeled pool | 827 images, 5% |
| Acquisition budget | 414 images per round, 2.5% |
| Final labeled ratio | 20% |
| Seeds | 0, 1, 2 |
| Seed workers | 3 |
| CPU affinity | seed 0: `0-3`, seed 1: `4-7`, seed 2: `8-11` |

All three seeds finished successfully.

## Method Meaning

ECPAL EUA-only uses the same ECPAL predictors as ECA:

- FDP: foreground detection probability
- CECP: conditional classification-error probability
- LECP: conditional localization-error probability
- MOCP: missed-object count predictor

However, EUA does not directly select images with the largest expected error
count. Instead, it selects images with the largest predictive error uncertainty:

```text
U_cls  = sum q_fg * H(q_cls_given_fg)
U_loc  = sum q_fg * H(q_loc_given_fg)
U_miss = H(Poisson(n_miss_hat))
EUA    = w_cls * U_cls + w_loc * U_loc + w_miss * U_miss
```

Because this is the `only` branch:

- candidate expand ratio is `1.0`;
- no diversity step is applied;
- the top `budget=414` images by EUA score are used directly for the next
  labeled pool.

## Main Result

ECPAL EUA-only is the best completed 3-seed result currently available in the
local `work_dirs` set.

| Method | Final mAP mean | std |
|---|---:|---:|
| ECPAL EUA-only | 0.7109 | 0.0009 |
| PAL full | 0.7086 | 0.0015 |
| PAL LIUS | 0.7071 | 0.0012 |
| PPAL | 0.7071 | 0.0035 |
| ECPAL ECA-only | 0.7064 | 0.0014 |
| ECPAL ECA-full | 0.7047 | 0.0008 |
| Core-set | 0.6715 | 0.0037 |

The improvement over the closest baselines is small but consistent:

| Comparison | Final mAP difference |
|---|---:|
| EUA-only - PAL full | +0.0023 |
| EUA-only - PAL LIUS | +0.0038 |
| EUA-only - PPAL | +0.0038 |
| EUA-only - ECA-only | +0.0045 |
| EUA-only - ECA-full | +0.0062 |

## Round-by-Round Performance

| Round | Training data | mAP mean | mAP std | AP50 mean | Gain |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.0% | 0.4677 | 0.0154 | 0.4677 | - |
| 2 | 7.5% | 0.5973 | 0.0019 | 0.5973 | +0.1296 |
| 3 | 10.0% | 0.6439 | 0.0024 | 0.6437 | +0.0466 |
| 4 | 12.5% | 0.6705 | 0.0013 | 0.6707 | +0.0266 |
| 5 | 15.0% | 0.6886 | 0.0028 | 0.6887 | +0.0181 |
| 6 | 17.5% | 0.7046 | 0.0009 | 0.7047 | +0.0161 |
| 7 | 20.0% | 0.7109 | 0.0009 | 0.7110 | +0.0063 |

Round 1 is identical to other methods because all methods train on the same
initial 5% labeled pool. Acquisition quality starts affecting performance from
Round 2 onward.

EUA-only is stronger than the main completed baselines from Round 2 onward:

| Round | EUA - ECA-only | EUA - PAL full | EUA - PPAL |
|---:|---:|---:|---:|
| 1 | +0.0000 | +0.0000 | +0.0000 |
| 2 | +0.0031 | +0.0043 | +0.0034 |
| 3 | +0.0058 | +0.0037 | +0.0074 |
| 4 | +0.0074 | +0.0046 | +0.0046 |
| 5 | +0.0014 | +0.0043 | +0.0044 |
| 6 | +0.0080 | +0.0099 | +0.0080 |
| 7 | +0.0045 | +0.0023 | +0.0038 |

The largest relative advantage appears at Round 6.

## Seed-Level Results

| Seed | Final mAP | AP50 | CPU affinity |
|---:|---:|---:|---|
| 0 | 0.7116 | 0.712 | `0,1,2,3` |
| 1 | 0.7096 | 0.710 | `4,5,6,7` |
| 2 | 0.7115 | 0.711 | `8,9,10,11` |

The final standard deviation is `0.0009`, so the final result is stable across
seeds.

## Selected Image Error Counts

The table below recomputes actual selected-image error counts using the same
ECPAL label definition as the predictors. Each row is the 3-seed mean over the
newly selected 414 images at that round.

| Round | GT/img | det/img | cls/img | loc/img | miss/img | total err/img |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.868 | 13.491 | 4.903 | 3.638 | 3.453 | 11.994 |
| 2 | 7.482 | 10.064 | 2.511 | 2.602 | 2.258 | 7.371 |
| 3 | 6.174 | 8.692 | 2.416 | 2.242 | 1.754 | 6.412 |
| 4 | 5.204 | 7.661 | 2.303 | 1.969 | 1.476 | 5.747 |
| 5 | 4.393 | 6.895 | 2.158 | 1.671 | 1.167 | 4.996 |
| 6 | 4.114 | 6.423 | 1.936 | 1.473 | 1.080 | 4.489 |
| 7 | 3.862 | 6.057 | 1.867 | 1.400 | 0.975 | 4.242 |

Compared with ECA-only:

| Round | ECA-only total err/img | EUA-only total err/img |
|---:|---:|---:|
| 1 | 11.769 | 11.994 |
| 2 | 5.696 | 7.371 |
| 3 | 6.099 | 6.412 |
| 4 | 5.696 | 5.747 |
| 5 | 4.940 | 4.996 |
| 6 | 4.993 | 4.489 |
| 7 | 4.301 | 4.242 |

EUA-only selects more actual error than ECA-only through Round 5, then slightly
less in Rounds 6 and 7. Despite that, EUA-only has higher mAP at the end. This
supports the interpretation that acquisition quality is not determined only by
raw error count. Selecting uncertain error cases can be more useful than
selecting the largest number of predicted errors.

## EUA Candidate Score Characteristics

The selected candidates' average score profile is:

| Round | score mean | top score | min score | u_cls | u_loc | u_miss | pi_cls | pi_loc | pi_miss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10.025 | 26.628 | 7.381 | 4.153 | 4.493 | 1.299 | 0.348 | 0.520 | 0.132 |
| 2 | 7.757 | 18.441 | 6.425 | 2.467 | 4.193 | 1.288 | 0.314 | 0.504 | 0.182 |
| 3 | 7.075 | 12.596 | 6.012 | 2.265 | 3.586 | 1.269 | 0.315 | 0.498 | 0.187 |
| 4 | 6.761 | 11.861 | 5.771 | 2.161 | 3.239 | 1.311 | 0.311 | 0.495 | 0.194 |
| 5 | 6.380 | 10.985 | 5.479 | 2.023 | 2.904 | 1.307 | 0.316 | 0.495 | 0.189 |
| 6 | 5.953 | 10.069 | 5.105 | 1.852 | 2.672 | 1.272 | 0.316 | 0.494 | 0.190 |
| 7 | 5.807 | 9.707 | 4.972 | 1.823 | 2.501 | 1.316 | 0.316 | 0.488 | 0.196 |

The selected profile is consistently dominated by localization uncertainty:

```text
pi_loc ~= 0.49-0.52
pi_cls ~= 0.31-0.35
pi_miss ~= 0.13-0.20
```

This is important because PAL's strength was also interpreted as selecting
images near detector decision boundaries, especially around foreground
classification/localization ambiguity. EUA appears to capture a related signal,
but through ECPAL's error predictors rather than PAL's LIUS score.

## Predictor Diagnostics

All seed/round predictor fits completed without fallback.

| Round | labeled imgs | unlabeled imgs | det examples | fg examples | CECP pos | LECP pos |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 827 | 15724 | 3175 | 3164 | 0.325 | 0.079 |
| 2 | 1241 | 15310 | 7692 | 7631 | 0.166 | 0.125 |
| 3 | 1655 | 14896 | 10879 | 10794 | 0.136 | 0.126 |
| 4 | 2069 | 14482 | 13453 | 13353 | 0.130 | 0.123 |
| 5 | 2483 | 14068 | 15329 | 15222 | 0.116 | 0.115 |
| 6 | 2897 | 13654 | 17171 | 17038 | 0.110 | 0.115 |
| 7 | 3311 | 13240 | 18795 | 18652 | 0.107 | 0.113 |

The sample count grows normally as the labeled pool expands. The CECP positive
rate decreases over rounds, which is expected as the detector improves and hard
classification errors become less dense in the labeled pool. LECP remains around
`0.11-0.13` after Round 2.

EUA scale weights are computed from labeled predictive uncertainties:

| Round | w_cls | w_loc | w_miss |
|---:|---:|---:|---:|
| 1 | 0.845 | 1.165 | 0.990 |
| 2 | 0.990 | 0.939 | 1.071 |
| 3 | 0.987 | 0.988 | 1.025 |
| 4 | 0.978 | 1.037 | 0.986 |
| 5 | 0.997 | 1.093 | 0.910 |
| 6 | 1.018 | 1.105 | 0.877 |
| 7 | 1.008 | 1.138 | 0.854 |

The weights remain well-behaved and do not show instability.

## ECA-only vs EUA-only Selection Overlap

EUA-only and ECA-only use the same ECPAL predictors but different score
definitions. Their selected-image overlap quickly drops after Round 1:

| Round | Overlap / 414 |
|---:|---:|
| 1 | 344.3 |
| 2 | 91.7 |
| 3 | 57.7 |
| 4 | 41.7 |
| 5 | 29.3 |
| 6 | 32.7 |
| 7 | 32.7 |

Round 1 starts from the same detector, so the high overlap there means ECA and
EUA are related signals. From Round 2 onward, each method changes its own
labeled pool, causing acquisition trajectories to diverge.

## Runtime

This run used seed-level parallel execution:

```text
--seed-workers 3 --seed-cpu-cores 4
```

The measured wall-clock time was about `26.48 hours`.

| Seed | Sum of step durations | Clock time |
|---:|---:|---:|
| 0 | 26.48 h | 26.48 h |
| 1 | 26.23 h | 26.23 h |
| 2 | 26.48 h | 26.48 h |

Round-level mean duration:

| Round | total h | train h | eval h | labeled infer h | unlabeled infer h | acquisition h |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.810 | 1.362 | 0.104 | 0.021 | 0.310 | 0.013 |
| 2 | 2.451 | 1.999 | 0.112 | 0.029 | 0.296 | 0.014 |
| 3 | 3.106 | 2.657 | 0.106 | 0.038 | 0.294 | 0.012 |
| 4 | 3.771 | 3.325 | 0.106 | 0.046 | 0.284 | 0.011 |
| 5 | 4.451 | 4.000 | 0.106 | 0.052 | 0.280 | 0.013 |
| 6 | 5.106 | 4.656 | 0.106 | 0.058 | 0.275 | 0.011 |
| 7 | 5.700 | 5.311 | 0.104 | 0.057 | 0.217 | 0.011 |

Training dominates runtime. EUA acquisition itself is not a bottleneck.

## Interpretation

The main finding is:

```text
EUA-only > ECA-only > ECA-full
```

This is important for ECPAL development.

ECA-only directly targets images with a large expected error count. It is good
at finding error-rich samples, but previous results showed that maximizing error
count alone does not guarantee the best mAP.

EUA-only instead targets images where the error predictors are uncertain about
whether detections are classification/localization/missed-object errors. This
appears to select samples that better correct the current detector's decision
boundary. The selected actual error count is still high, especially through
Round 5, but the score is not simply "more errors is better."

The result supports the following working hypothesis:

```text
Good AL samples are not just images with many objects or many detector errors.
They are images containing object/detection cases that can strongly correct the
current detector's uncertain classification, localization, and recall behavior.
```

## Next Experiment

The most direct next experiment is:

```powershell
python -B tools/run_active_learning.py --method ecpal:eua-full --detector retinanet --dataset voc --gpus 1 --seeds 0 1 2 --seed-workers 3 --seed-cpu-cores 4
```

This will test whether adding diversity on top of the EUA candidate pool helps
or hurts. Because ECA-full underperformed ECA-only, the key question is whether
EUA's uncertainty profile makes the diversity step more useful than it was for
ECA.

Expected interpretations:

| Result | Interpretation |
|---|---|
| EUA-full > EUA-only | EUA profile diversity adds useful coverage beyond high-EUA samples. |
| EUA-full ~= EUA-only | EUA ranking already provides enough diversity. |
| EUA-full < EUA-only | Diversity again disrupts the most useful top-ranked samples; keep EUA-only or weaken diversity. |

