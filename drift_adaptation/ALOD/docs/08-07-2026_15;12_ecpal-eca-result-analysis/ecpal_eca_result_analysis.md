# ECPAL ECA-only Result Analysis

## 분석 대상

이 문서는 서버에서 실행 후 로컬로 옮긴 ECPAL ECA-only 결과를 정리한다.

분석 대상 run:

```text
work_dirs/retinanet_voc_ecpal_eca_7rounds_5percent_to_20percent/08-05-2026_16;35
```

실험 설정:

| Item | Value |
|---|---:|
| Method argument | `ecpal:eca` |
| Method stage | ECA-only |
| Detector | RetinaNet |
| Dataset | PASCAL VOC |
| Seeds | 0, 1, 2 |
| Rounds | 7 |
| Initial labeled images | 827 |
| Acquisition budget | 414 images/round |
| Candidate expand ratio | 1 |

`seed_0`, `seed_1`, `seed_2` 모두 `status=done`이며 7 rounds를 완료했다.

## ECA-only 의미

`ecpal:eca`는 full ECPAL에서 diversity selection을 제거한 ablation이다.

ECA-only는 다음 순서로 동작한다.

1. 현재 labeled pool과 unlabeled pool에 대해 ECPAL feature inference를 수행한다.
2. Labeled pool feature와 GT로 ECPAL predictor를 학습한다.
3. Unlabeled image별 expected error count를 예측한다.
4. Weighted ECA score로 unlabeled images를 정렬한다.
5. Top `budget=414` images를 바로 선택한다.

Full ECPAL과의 차이:

| Method | Candidate pool | Final selection |
|---|---:|---|
| `ecpal` | `2 * budget = 828` | Jensen-Shannon diversity / farthest-first |
| `ecpal:eca` | `1 * budget = 414` | Top ECA score |

따라서 이 실험의 목적은 ECPAL의 핵심 uncertainty score인 ECA가 PAL LIUS, PPAL, full ECPAL과 비교해 어느 정도 acquisition signal인지 확인하는 것이다.

## Round 의미

ALOD runner의 round 순서는 다음과 같다.

1. 현재 labeled pool로 detector를 학습한다.
2. VOC test set에서 detector를 평가한다.
3. Acquisition을 수행해 다음 round의 labeled pool을 만든다.

따라서 `round_01` mAP는 initial 5%인 827장으로 학습한 baseline detector의 성능이다. `round_01`에서 선택한 414장은 `round_02` 학습부터 반영된다.

| Round | Train labeled images | Labeled after acquisition | Train data percent | After-acquisition percent |
|---:|---:|---:|---:|---:|
| 1 | 827 | 1241 | 5.00 | 7.50 |
| 2 | 1241 | 1655 | 7.50 | 10.00 |
| 3 | 1655 | 2069 | 10.00 | 12.50 |
| 4 | 2069 | 2483 | 12.50 | 15.00 |
| 5 | 2483 | 2897 | 15.00 | 17.50 |
| 6 | 2897 | 3311 | 17.50 | 20.00 |
| 7 | 3311 | 3725 | 20.00 | 22.51 |

`round_07` acquisition 결과로 3725장 pool이 저장되지만, 이 pool은 평가되지 않는다. 최종 reported mAP는 3311장, 약 20%로 학습한 결과다.

## 성능 요약

3-seed 평균 기준 ECPAL ECA-only 성능은 다음과 같다.

| Round | Train % | mAP mean | mAP std | AP50 mean | AP50 std | Gain |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.0 | 0.4677 | 0.0154 | 0.4677 | 0.0156 | - |
| 2 | 7.5 | 0.5941 | 0.0099 | 0.5940 | 0.0102 | +0.1265 |
| 3 | 10.0 | 0.6381 | 0.0133 | 0.6380 | 0.0134 | +0.0439 |
| 4 | 12.5 | 0.6630 | 0.0072 | 0.6630 | 0.0071 | +0.0249 |
| 5 | 15.0 | 0.6871 | 0.0071 | 0.6870 | 0.0073 | +0.0241 |
| 6 | 17.5 | 0.6966 | 0.0029 | 0.6970 | 0.0029 | +0.0095 |
| 7 | 20.0 | 0.7064 | 0.0014 | 0.7063 | 0.0012 | +0.0097 |

Seed별 mAP:

| Seed | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 | Round 6 | Round 7 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.4462 | 0.6035 | 0.6192 | 0.6570 | 0.6773 | 0.6926 | 0.7084 |
| 1 | 0.4751 | 0.5805 | 0.6470 | 0.6732 | 0.6900 | 0.6978 | 0.7051 |
| 2 | 0.4817 | 0.5984 | 0.6479 | 0.6588 | 0.6941 | 0.6995 | 0.7057 |

최종 성능:

| Seed | Final mAP | Final AP50 |
|---:|---:|---:|
| 0 | 0.7084 | 0.708 |
| 1 | 0.7051 | 0.705 |
| 2 | 0.7057 | 0.706 |

Round 2-3에서 seed variance가 상대적으로 크다. Round 2 mAP std는 0.0099, round 3 mAP std는 0.0133이다. 후반에는 variance가 줄어든다.

## 다른 method와의 비교

동일 RetinaNet + VOC protocol의 완료된 3-seed run들과 비교하면 다음과 같다.

| Round | PAL full | PAL LIUS | PPAL | ECPAL full | ECPAL ECA | Core-set |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.4677 | 0.4677 | 0.4677 | 0.4677 | 0.4677 | 0.4677 |
| 2 | 0.5929 | 0.5914 | 0.5939 | 0.5914 | 0.5941 | 0.5547 |
| 3 | 0.6402 | 0.6384 | 0.6365 | 0.6394 | 0.6381 | 0.6059 |
| 4 | 0.6659 | 0.6592 | 0.6658 | 0.6690 | 0.6630 | 0.6217 |
| 5 | 0.6843 | 0.6810 | 0.6841 | 0.6849 | 0.6871 | 0.6439 |
| 6 | 0.6947 | 0.6920 | 0.6966 | 0.6988 | 0.6966 | 0.6608 |
| 7 | 0.7086 | 0.7071 | 0.7071 | 0.7047 | 0.7064 | 0.6715 |

Final ranking:

| Rank | Method | Final mAP | Final mAP std |
|---:|---|---:|---:|
| 1 | PAL full | 0.7086 | 0.0015 |
| 2 | PAL LIUS | 0.7071 | 0.0012 |
| 3 | PPAL | 0.7071 | 0.0035 |
| 4 | ECPAL ECA | 0.7064 | 0.0014 |
| 5 | ECPAL full | 0.7047 | 0.0008 |
| 6 | Core-set | 0.6715 | 0.0037 |

ECPAL ECA-only는 full ECPAL보다 최종 mAP가 0.0016 높다. 따라서 현재 full ECPAL의 JS diversity step은 최종 round 기준으로는 이득이 불명확하거나 약간 손해일 수 있다.

다만 round 4와 round 6에서는 full ECPAL이 ECA-only보다 높다. 따라서 diversity가 항상 나쁘다고 단정할 수는 없다. 중간 round에서는 도움이 될 수 있고, 후반에는 과한 exploratory selection이 될 수 있다는 해석이 더 적절하다.

## ECPAL ECA와 다른 method의 차이

ECPAL ECA-only와 주요 method의 mAP 차이는 다음과 같다.

| Round | ECA - PAL LIUS | ECA - ECPAL full | ECA - PAL full | ECA - PPAL |
|---:|---:|---:|---:|---:|
| 1 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| 2 | +0.0027 | +0.0027 | +0.0012 | +0.0002 |
| 3 | -0.0004 | -0.0014 | -0.0021 | +0.0016 |
| 4 | +0.0039 | -0.0060 | -0.0029 | -0.0028 |
| 5 | +0.0062 | +0.0022 | +0.0029 | +0.0030 |
| 6 | +0.0047 | -0.0022 | +0.0019 | +0.0001 |
| 7 | -0.0007 | +0.0016 | -0.0022 | -0.0007 |

차이는 대부분 0.001-0.006 mAP 수준이다. 따라서 final mAP만으로 강한 우열을 주장하기보다는, selected sample의 성격과 round별 gain을 함께 봐야 한다.

## Selected image error 재계산

ECA-only가 선택한 414장의 실제 GT/error 구성을 기존 분석과 같은 coarse TIDE-like 기준으로 재계산했다.

사용한 기준:

| Item | Value |
|---|---:|
| Feature artifact | `ecpal_unlabeled_features.json` |
| Detection source | `final_detections` |
| Detection score threshold | 0.3 |
| Background/foreground IoU threshold | 0.1 |
| Localization IoU threshold | 0.5 |
| GT source | `data/VOC0712/annotations/trainval_0712.json` |

Error 정의:

| Error | Definition |
|---|---|
| Foreground detection | `max(u_same, u_diff) >= 0.1` |
| Classification error | foreground detection with `u_diff > u_same` |
| Localization error | foreground detection with `max(u_same, u_diff) < 0.5` |
| Missed object | GT object not explained by any foreground detection |

3-seed 평균 기준 selected image당 count:

| Round | GT/img | Detection/img | Cls error/img | Loc error/img | Miss/img | Total error/img |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.129 | 12.955 | 5.256 | 3.274 | 3.239 | 11.769 |
| 2 | 5.249 | 6.796 | 2.097 | 1.750 | 1.849 | 5.696 |
| 3 | 5.395 | 7.781 | 2.543 | 1.900 | 1.656 | 6.099 |
| 4 | 4.852 | 7.166 | 2.395 | 1.840 | 1.461 | 5.696 |
| 5 | 4.119 | 6.177 | 2.061 | 1.601 | 1.278 | 4.940 |
| 6 | 4.109 | 6.490 | 2.208 | 1.631 | 1.154 | 4.993 |
| 7 | 3.786 | 5.869 | 1.917 | 1.363 | 1.021 | 4.301 |

ECPAL ECA-only는 실제 error count가 매우 높은 이미지를 고른다. 특히 round 1에서 selected image당 total error가 11.769로, full ECPAL의 10.282보다도 높다.

## Selected error count 비교

Total error/img 기준으로 비교하면 다음과 같다.

| Round | PAL full | PPAL | ECPAL full | PAL LIUS | ECPAL ECA |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.383 | 6.598 | 10.282 | 4.890 | 11.769 |
| 2 | 4.923 | 5.304 | 7.031 | 3.787 | 5.696 |
| 3 | 4.667 | 4.891 | 5.952 | 3.950 | 6.099 |
| 4 | 4.304 | 4.233 | 5.362 | 3.636 | 5.696 |
| 5 | 4.229 | 3.934 | 4.984 | 3.429 | 4.940 |
| 6 | 4.009 | 3.473 | 4.531 | 3.608 | 4.993 |
| 7 | 3.840 | 3.345 | 4.229 | 3.332 | 4.301 |

ECPAL ECA-only는 대부분 round에서 가장 높은 actual error count를 선택한다. 이는 ECPAL의 핵심 가설, 즉 detector error count를 직접 예측해 error-rich image를 고른다는 부분이 잘 작동한다는 증거다.

하지만 final mAP는 최고가 아니다. 따라서 selected image의 error count 양만으로 acquisition quality가 완전히 결정되지는 않는다.

## ECA score와 predicted error profile

ECA 후보 414장의 평균 score와 predicted profile은 다음과 같다.

| Round | ECA score | n_cls_hat | n_loc_hat | n_miss_hat | pi_cls | pi_loc | pi_miss | det count |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.568 | 5.164 | 1.873 | 1.471 | 0.329 | 0.486 | 0.185 | 12.955 |
| 2 | 7.394 | 2.234 | 1.926 | 3.310 | 0.283 | 0.350 | 0.368 | 6.796 |
| 3 | 6.337 | 2.513 | 2.351 | 1.330 | 0.390 | 0.427 | 0.184 | 7.781 |
| 4 | 6.004 | 2.277 | 2.267 | 1.252 | 0.413 | 0.419 | 0.168 | 7.166 |
| 5 | 6.080 | 2.028 | 1.917 | 2.342 | 0.351 | 0.356 | 0.293 | 6.177 |
| 6 | 5.636 | 2.151 | 2.078 | 1.222 | 0.418 | 0.401 | 0.181 | 6.490 |
| 7 | 5.152 | 1.913 | 1.840 | 1.230 | 0.421 | 0.399 | 0.180 | 5.869 |

초반에는 cls/loc/miss가 모두 크게 잡힌다. 후반으로 갈수록 ECA score가 감소한다. 이는 active learning이 진행되면서 남은 unlabeled pool의 high-error sample density가 줄어드는 자연스러운 패턴으로 볼 수 있다.

## Predictor diagnostics

모든 seed/round에서 predictor fallback은 없었다. 즉 scikit-learn predictor가 정상적으로 학습되었고, ECA-only 성능 패턴은 fallback 때문이 아니다.

3-seed 평균 predictor training data summary:

| Round | Labeled imgs | Detection examples | FG examples | Mean cls | Mean loc | Mean miss | w_cls | w_loc | w_miss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 827 | 3175.3 | 3164.3 | 1.247 | 0.303 | 0.548 | 0.408 | 1.670 | 0.923 |
| 2 | 1241 | 7243.0 | 7192.3 | 0.978 | 0.693 | 0.904 | 0.859 | 1.212 | 0.929 |
| 3 | 1655 | 9493.3 | 9410.0 | 0.825 | 0.702 | 0.921 | 0.976 | 1.148 | 0.877 |
| 4 | 2069 | 11599.0 | 11492.7 | 0.691 | 0.679 | 0.942 | 1.091 | 1.108 | 0.801 |
| 5 | 2483 | 13828.7 | 13702.0 | 0.707 | 0.667 | 0.936 | 1.065 | 1.127 | 0.808 |
| 6 | 2897 | 15624.7 | 15467.0 | 0.658 | 0.662 | 0.912 | 1.104 | 1.096 | 0.800 |
| 7 | 3311 | 17096.7 | 16937.0 | 0.583 | 0.593 | 0.895 | 1.139 | 1.119 | 0.743 |

Labeled pool이 커질수록 detection examples도 증가한다. Positive rate와 mean true count도 안정적으로 변한다. 이 결과는 ECA-only가 충분한 predictor training samples를 사용하고 있음을 보여준다.

## Overlap 분석

ECPAL ECA-only와 full ECPAL의 selected image overlap:

| Round | Overlap / 414 | Jaccard mean |
|---:|---:|---:|
| 1 | 228.7 | 0.3816 |
| 2 | 59.7 | 0.0794 |
| 3 | 45.0 | 0.0575 |
| 4 | 34.0 | 0.0428 |
| 5 | 30.0 | 0.0377 |
| 6 | 30.3 | 0.0380 |
| 7 | 23.3 | 0.0290 |

Round 1은 같은 initial detector에서 비교한 것이므로 의미가 크다. Full ECPAL은 ECA top-414 중 평균 약 229장만 유지하고, 나머지는 JS diversity 때문에 바뀐다. 이후 round는 학습 pool 자체가 달라지므로 overlap이 낮아지는 것이 정상이다.

ECPAL ECA-only와 다른 method의 selected overlap:

| Round | vs PAL LIUS | vs PAL full | vs PPAL |
|---:|---:|---:|---:|
| 1 | 52.3 | 82.7 | 82.3 |
| 2 | 16.0 | 27.7 | 45.7 |
| 3 | 23.7 | 29.7 | 39.0 |
| 4 | 20.7 | 26.3 | 32.7 |
| 5 | 15.0 | 19.3 | 27.3 |
| 6 | 18.7 | 26.3 | 21.7 |
| 7 | 22.3 | 21.7 | 15.7 |

ECA-only는 PAL/PPAL과 매우 다른 이미지를 선택한다. 그럼에도 mAP가 비슷하다는 점은 서로 다른 acquisition path가 비슷한 최종 성능에 도달할 수 있음을 보여준다.

## Runtime

3-seed 평균 기준 round별 duration:

| Round | Train sec | Eval sec | Labeled inference sec | Unlabeled inference sec | Acquisition sec | Total sec |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2055.0 | 165.8 | 38.5 | 465.2 | 13.1 | 2737.6 |
| 2 | 2959.3 | 163.4 | 50.9 | 453.2 | 12.7 | 3639.5 |
| 3 | 3855.2 | 162.5 | 62.7 | 444.2 | 12.7 | 4537.3 |
| 4 | 4758.0 | 162.5 | 75.3 | 432.0 | 12.3 | 5440.1 |
| 5 | 5651.2 | 171.1 | 87.4 | 417.9 | 12.0 | 6339.6 |
| 6 | 6548.8 | 169.3 | 99.4 | 407.7 | 11.8 | 7236.9 |
| 7 | 7433.2 | 171.0 | 111.5 | 395.4 | 11.2 | 8122.3 |

Mean duration 기준 한 seed의 전체 runtime은 약 10.57시간이고, 3 seeds sequential 실행 기준 약 31.71시간이다.

Acquisition 자체는 round당 약 12초로 작다. 대부분의 시간은 detector training과 ECPAL feature inference가 차지한다.

## 해석

ECPAL ECA-only 결과는 ECPAL 개발 방향을 판단하는 데 중요하다.

### 1. ECA score 자체는 성공적이다

ECA-only는 대부분 round에서 가장 높은 selected actual error count를 보인다. Round 1의 total error/img는 11.769로 full ECPAL, PAL, PPAL보다 높다. 따라서 ECA score가 실제 detector error가 많은 이미지를 찾는다는 핵심 목적은 달성하고 있다.

### 2. 하지만 error count 최대화가 mAP 최적은 아니다

Final mAP는 PAL full, PAL LIUS, PPAL보다 약간 낮다. 이는 다음을 의미한다.

```text
High actual error count != highest next-round mAP gain
```

Detector가 많이 틀리는 이미지는 중요하지만, 모든 error가 같은 학습 가치를 갖는 것은 아니다. 지나치게 어려운 이미지, noisy image, 이미 detector가 충분히 혼란스러워하지만 학습 signal이 분산되는 이미지가 섞일 수 있다.

### 3. Full ECPAL diversity는 최종 round 기준으로 이득이 불명확하다

Full ECPAL은 round 4와 round 6에서는 ECA-only보다 높지만, final round에서는 ECA-only보다 낮다. 따라서 현재 JS diversity는 중간 round에서는 도움이 될 수 있으나, 후반에는 high-value ECA samples를 지나치게 대체했을 가능성이 있다.

### 4. PAL의 LIUS/GUIDE와 ECA는 다른 sample을 고른다

ECA-only와 PAL LIUS의 overlap은 매우 낮다. Round 1에서도 평균 52.3장만 겹친다. ECA는 error count가 큰 이미지를 직접 고르고, PAL LIUS는 TP/FP decision boundary 근처 detection을 고른다. 두 signal은 상당히 다르다.

### 5. 다음 개선 방향은 ECA 보존형 diversity다

ECA를 버릴 이유는 없다. 오히려 ECA score 자체는 강하다. 개선은 diversity를 더 조심스럽게 섞는 방향이 맞다.

가능한 방향:

- `candidate_expand_ratio=1.25` 또는 `1.5`로 줄이기
- top ECA 일부를 반드시 보존하는 constrained diversity
- round가 뒤로 갈수록 diversity weight를 줄이는 late-round-aware schedule
- ECA score와 diversity score를 hard replacement가 아니라 weighted ranking으로 결합
- selected error type composition이 한쪽으로 과도하게 쏠릴 때만 diversity를 적용

## 결론

ECPAL ECA-only는 정상적으로 실행되었고, 결과는 다음처럼 정리할 수 있다.

1. ECA score는 실제 error-rich image를 잘 고른다.
2. ECA-only는 full ECPAL보다 최종 mAP가 높다.
3. ECA-only는 PAL LIUS/PPAL과 거의 동급이지만 최종 평균은 아주 조금 낮다.
4. Error count 양만으로 acquisition quality가 완전히 결정되지는 않는다.
5. 향후 ECPAL 개선은 ECA top ranking을 최대한 보존하면서 diversity를 약하게, 또는 round-aware하게 섞는 방향이 적절하다.

## 확인한 파일

```text
work_dirs/retinanet_voc_ecpal_eca_7rounds_5percent_to_20percent/08-05-2026_16;35/aggregate_summary.json
work_dirs/retinanet_voc_ecpal_eca_7rounds_5percent_to_20percent/08-05-2026_16;35/seed_*/run_summary.json
work_dirs/retinanet_voc_ecpal_eca_7rounds_5percent_to_20percent/08-05-2026_16;35/seed_*/round_*/round_summary.json
work_dirs/retinanet_voc_ecpal_eca_7rounds_5percent_to_20percent/08-05-2026_16;35/seed_*/round_*/eval_*.json
work_dirs/retinanet_voc_ecpal_eca_7rounds_5percent_to_20percent/08-05-2026_16;35/seed_*/round_*/ecpal_eca_diagnostics.json
work_dirs/retinanet_voc_ecpal_eca_7rounds_5percent_to_20percent/08-05-2026_16;35/seed_*/round_*/ecpal_eca_candidates.json
work_dirs/retinanet_voc_ecpal_eca_7rounds_5percent_to_20percent/08-05-2026_16;35/seed_*/round_*/ecpal_unlabeled_features.json
```
