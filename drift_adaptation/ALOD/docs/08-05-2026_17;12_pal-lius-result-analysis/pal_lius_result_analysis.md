# PAL LIUS-only Result Analysis

## 분석 대상

이 문서는 서버에서 실행 후 로컬로 옮긴 PAL LIUS-only 결과를 정리한다.

분석 대상 run:

```text
work_dirs/retinanet_voc_pal_lius_7rounds_5percent_to_20percent/08-04-2026_09;05
```

실험 설정:

| Item | Value |
|---|---:|
| Method argument | `pal:lius` |
| Detector | RetinaNet |
| Dataset | PASCAL VOC |
| Seeds | 0, 1, 2 |
| Rounds | 7 |
| Initial labeled images | 827 |
| Acquisition budget | 414 images/round |
| PAL mode | LIUS-only |
| Candidate multiplier | 1 |

`seed_0`, `seed_1`, `seed_2` 모두 `status=done`이며 7 rounds를 완료했다.

## Round 의미

ALOD runner의 round 순서는 다음과 같다.

1. 현재 labeled pool로 detector를 학습한다.
2. VOC test set에서 detector를 평가한다.
3. PAL LIUS acquisition을 수행해 다음 round의 labeled pool을 만든다.

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

3-seed 평균 기준 PAL LIUS-only 성능은 다음과 같다.

| Round | mAP mean | mAP std | AP50 mean | AP50 std | Gain from previous round |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.4677 | 0.0154 | 0.4677 | 0.0156 | - |
| 2 | 0.5914 | 0.0021 | 0.5917 | 0.0019 | +0.1238 |
| 3 | 0.6384 | 0.0003 | 0.6383 | 0.0005 | +0.0470 |
| 4 | 0.6592 | 0.0051 | 0.6593 | 0.0052 | +0.0207 |
| 5 | 0.6810 | 0.0008 | 0.6810 | 0.0008 | +0.0218 |
| 6 | 0.6920 | 0.0031 | 0.6920 | 0.0033 | +0.0110 |
| 7 | 0.7071 | 0.0012 | 0.7070 | 0.0014 | +0.0151 |

Seed별 mAP:

| Seed | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 | Round 6 | Round 7 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.4462 | 0.5885 | 0.6388 | 0.6628 | 0.6819 | 0.6920 | 0.7055 |
| 1 | 0.4751 | 0.5933 | 0.6380 | 0.6519 | 0.6800 | 0.6958 | 0.7076 |
| 2 | 0.4817 | 0.5925 | 0.6384 | 0.6627 | 0.6810 | 0.6881 | 0.7082 |

최종 성능은:

| Seed | Final mAP | Final AP50 |
|---:|---:|---:|
| 0 | 0.7055 | 0.705 |
| 1 | 0.7076 | 0.708 |
| 2 | 0.7082 | 0.708 |

## 다른 method와의 비교

동일 RetinaNet + VOC protocol의 완료된 3-seed run들과 비교하면 다음과 같다.

| Round | PAL full | PAL LIUS | PPAL | ECPAL | Core-set |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.4677 | 0.4677 | 0.4677 | 0.4677 | 0.4677 |
| 2 | 0.5929 | 0.5914 | 0.5939 | 0.5914 | 0.5547 |
| 3 | 0.6402 | 0.6384 | 0.6365 | 0.6394 | 0.6059 |
| 4 | 0.6659 | 0.6592 | 0.6658 | 0.6690 | 0.6217 |
| 5 | 0.6843 | 0.6810 | 0.6841 | 0.6849 | 0.6439 |
| 6 | 0.6947 | 0.6920 | 0.6966 | 0.6988 | 0.6608 |
| 7 | 0.7086 | 0.7071 | 0.7071 | 0.7047 | 0.6715 |

PAL full과 PAL LIUS-only의 차이:

| Round | PAL full - PAL LIUS |
|---:|---:|
| 1 | +0.0000 |
| 2 | +0.0015 |
| 3 | +0.0018 |
| 4 | +0.0067 |
| 5 | +0.0033 |
| 6 | +0.0028 |
| 7 | +0.0015 |

해석:

- LIUS-only는 full PAL보다 모든 non-initial round에서 조금 낮다.
- 차이는 전반적으로 작고, 최종 round 차이는 0.0015 mAP다.
- 가장 큰 차이는 round 4의 0.0067 mAP다.
- 따라서 PAL 성능의 큰 부분은 LIUS가 담당하고, GUIDE는 특히 중간 round에서 추가 이득을 주는 보조 signal로 해석된다.
- LIUS-only는 최종 mAP 기준 PPAL과 거의 동일하고 Core-set보다 크게 높다.

## LIUS-only acquisition 동작 확인

PAL LIUS-only는 다음 과정을 수행한다.

1. labeled detections에서 class-wise logistic model을 학습한다.
2. unlabeled detections에 대해 TP/FP decision boundary 근접도를 LIUS score로 계산한다.
3. PAL의 class-wise budget allocation을 적용한다.
4. 각 class에서 `1 * class_budget`만큼 후보 image를 만든다.
5. GUIDE 없이 LIUS score만으로 최종 414장을 선택한다.

실제 diagnostics에서도 모든 round의 `candidate_multiplier`가 1이고, 후보 수와 selected 수가 모두 414로 확인됐다.

| Round | Candidate count | Selected count | LIUS score mean | Matched detections | Scored detections |
|---:|---:|---:|---:|---:|---:|
| 1 | 414 | 414 | 0.6803 | 3175.3 | 45392.7 |
| 2 | 414 | 414 | 0.6891 | 4889.3 | 46027.3 |
| 3 | 414 | 414 | 0.6884 | 6544.0 | 45698.7 |
| 4 | 414 | 414 | 0.6909 | 8348.0 | 43790.3 |
| 5 | 414 | 414 | 0.6912 | 9874.7 | 41259.3 |
| 6 | 414 | 414 | 0.6863 | 11683.3 | 40086.0 |
| 7 | 414 | 414 | 0.6910 | 13291.7 | 37915.0 |

LIUS score의 최대값은 거의 `0.693147`이다. 이는 binary entropy의 최대값 `ln(2)`에 해당한다. 즉 선택된 image들은 PAL의 class-wise TP/FP logistic classifier 기준 decision boundary 근처 detection을 많이 포함한다.

Class budget은 전체적으로 class당 약 20-22장 수준으로 분산된다. Round 1 기준 평균 class budget은 아래와 같다.

| Class id | Budget |
|---:|---:|
| 0 | 21.0 |
| 1 | 21.0 |
| 2 | 21.0 |
| 3 | 21.0 |
| 4 | 21.0 |
| 5 | 21.3 |
| 6 | 20.0 |
| 7 | 21.0 |
| 8 | 20.0 |
| 9 | 21.0 |
| 10 | 22.0 |
| 11 | 21.0 |
| 12 | 21.0 |
| 13 | 21.0 |
| 14 | 15.0 |
| 15 | 21.0 |
| 16 | 21.7 |
| 17 | 21.0 |
| 18 | 21.0 |
| 19 | 21.0 |

## Selected image error 재계산

선택 이미지의 실제 GT/error 구성을 기존 PAL/PPAL/ECPAL 분석과 같은 coarse TIDE-like 기준으로 재계산했다.

사용한 기준:

| Item | Value |
|---|---:|
| Detection artifact | `pal_unlabeled_detections.bbox.json` |
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
| 1 | 4.291 | 5.378 | 2.129 | 1.102 | 1.659 | 4.890 |
| 2 | 4.171 | 4.911 | 1.387 | 0.986 | 1.415 | 3.787 |
| 3 | 4.190 | 5.184 | 1.485 | 1.097 | 1.368 | 3.950 |
| 4 | 3.865 | 5.020 | 1.397 | 1.140 | 1.099 | 3.636 |
| 5 | 3.908 | 4.717 | 1.216 | 0.986 | 1.228 | 3.429 |
| 6 | 3.957 | 5.043 | 1.302 | 1.171 | 1.134 | 3.608 |
| 7 | 3.805 | 4.937 | 1.221 | 1.075 | 1.036 | 3.332 |

기존 full PAL/ECPAL 분석과 비교하면 LIUS-only의 selected actual error density는 낮은 편이다. 예를 들어 round 1 기준:

| Method | Total error/img |
|---|---:|
| PAL full | 6.383 |
| ECPAL | 10.282 |
| PAL LIUS-only | 4.890 |

이는 LIUS-only가 실제 error count가 가장 많은 이미지를 직접 고르는 방식은 아니라는 뜻이다. LIUS는 class-wise logistic TP/FP decision boundary 근처 detection을 고르며, 이 신호가 곧 actual error count 최대화와 완전히 같지는 않다.

## Full PAL과 LIUS-only의 selected overlap

Full PAL과 LIUS-only가 같은 round/seed에서 선택한 image overlap은 낮다.

| Round | Mean overlap out of 414 | Jaccard mean |
|---:|---:|---:|
| 1 | 95.0 | 0.1296 |
| 2 | 24.3 | 0.0303 |
| 3 | 24.7 | 0.0307 |
| 4 | 25.0 | 0.0311 |
| 5 | 22.0 | 0.0273 |
| 6 | 26.3 | 0.0329 |
| 7 | 20.7 | 0.0256 |

Round 1에서도 overlap이 약 95장뿐이다. 이는 full PAL이 `2 * class_budget` 후보를 만든 뒤 GUIDE score로 재정렬하는 반면, LIUS-only는 `1 * class_budget` 후보를 바로 최종 selection으로 사용하기 때문이다. Round 2 이후에는 각 method의 학습 pool이 달라지므로 overlap이 더 낮아지는 것이 자연스럽다.

## Runtime

3-seed 평균 기준 round별 duration:

| Round | Train sec | Eval sec | Labeled inference sec | Unlabeled inference sec | Acquisition sec | Total sec |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2049.2 | 164.2 | 38.1 | 451.0 | 4.7 | 2707.2 |
| 2 | 2943.2 | 166.9 | 49.4 | 438.2 | 4.8 | 3602.4 |
| 3 | 3833.8 | 170.4 | 61.3 | 432.1 | 4.9 | 4502.4 |
| 4 | 4733.7 | 175.9 | 73.1 | 419.7 | 4.8 | 5407.2 |
| 5 | 5619.4 | 170.8 | 85.0 | 410.8 | 4.7 | 6290.7 |
| 6 | 6512.2 | 162.1 | 96.8 | 399.7 | 4.7 | 7175.5 |
| 7 | 7392.1 | 161.8 | 108.8 | 387.0 | 4.6 | 8054.3 |

Mean duration 기준 한 seed의 전체 runtime은 약 10.48시간이고, 3 seeds sequential 실행 기준 약 31.45시간이다.

Acquisition 자체는 round당 약 5초로 매우 작고, 전체 시간은 대부분 detector training과 unlabeled inference가 차지한다.

## 결론

PAL LIUS-only는 정상적으로 실행되었고, 직접적인 uncertainty-only ablation으로 해석할 수 있다.

주요 결론:

- LIUS-only는 full PAL보다 약간 낮지만, 차이는 작다.
- 최종 mAP는 PPAL과 거의 동일하고 Core-set보다 크게 높다.
- PAL full의 GUIDE는 성능의 주된 원천이라기보다, LIUS 후보를 중간 round에서 더 유리하게 다듬는 보조 signal로 보인다.
- LIUS-only는 actual error count를 직접 최대화하지는 않는다. 대신 class-wise TP/FP logistic boundary 근처 detection을 포함한 이미지를 고른다.
- Full PAL과 LIUS-only의 selected overlap이 낮기 때문에, GUIDE는 단순한 tie-breaker가 아니라 최종 선택 구성을 크게 바꾸는 역할을 한다.
- 그럼에도 full PAL 대비 최종 성능 차이가 작다는 점은 LIUS signal 자체가 상당히 강하다는 것을 보여준다.

## 확인한 파일

```text
work_dirs/retinanet_voc_pal_lius_7rounds_5percent_to_20percent/08-04-2026_09;05/aggregate_summary.json
work_dirs/retinanet_voc_pal_lius_7rounds_5percent_to_20percent/08-04-2026_09;05/seed_*/run_summary.json
work_dirs/retinanet_voc_pal_lius_7rounds_5percent_to_20percent/08-04-2026_09;05/seed_*/round_*/round_summary.json
work_dirs/retinanet_voc_pal_lius_7rounds_5percent_to_20percent/08-04-2026_09;05/seed_*/round_*/eval_*.json
work_dirs/retinanet_voc_pal_lius_7rounds_5percent_to_20percent/08-04-2026_09;05/seed_*/round_*/pal_lius_diagnostics.json
work_dirs/retinanet_voc_pal_lius_7rounds_5percent_to_20percent/08-04-2026_09;05/seed_*/round_*/pal_lius_candidates.json
work_dirs/retinanet_voc_pal_lius_7rounds_5percent_to_20percent/08-04-2026_09;05/seed_*/round_*/pal_unlabeled_detections.bbox.json
```
