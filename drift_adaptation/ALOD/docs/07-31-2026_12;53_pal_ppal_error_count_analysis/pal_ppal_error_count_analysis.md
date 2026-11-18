# PAL/PPAL Error Count Analysis

이 문서는 `work_dirs/retinanet_voc_pal_7rounds_5percent_to_20percent`와
`work_dirs/retinanet_voc_ppal_7rounds_5percent_to_20percent`에 저장된 PAL/PPAL 실행 결과를
기반으로, 각 round에서 새로 acquisition 된 이미지들의 detector error count를 비교한 대화 내용을
정리한 것이다.

## 분석 대상

이 분석의 각 row는 **해당 round에서 새로 acquisition 된 414장**에 대한 결과이다.

누적 labeled pool 전체도 아니고, VOC test set evaluation 결과도 아니다. 구체적으로 round `r`의 분석
대상 이미지는 다음과 같이 정의했다.

```text
selected_images_r = round_r/annotations/new_labeled.json - previous_labeled.json
```

`previous_labeled.json`은 round에 따라 다음과 같다.

| Round | previous_labeled.json |
|---:|---|
| 1 | `round_00/annotations/labeled.json` |
| 2 이상 | `round_{r-1}/annotations/new_labeled.json` |

즉 round 1의 row는 "초기 827장으로 학습한 detector가 unlabeled pool을 예측했고, 그중 PAL/PPAL이
새로 고른 414장"에 대한 error count이다.

## 사용한 Artifact

| Method | 분석 이미지 범위 | Detection source |
|---|---|---|
| PAL | `round_r/annotations/new_labeled.json - previous_labeled.json` | `round_r/pal_unlabeled_detections.bbox.json` |
| PPAL | `round_r/annotations/new_labeled.json - previous_labeled.json` | `round_r/diversity_inference_result.bbox.json` |

GT는 unlabeled annotation JSON 안에 없기 때문에 `data/VOC0712/annotations/trainval_0712.json`의 oracle
annotation으로 매칭했다.

## 계산 기준

저장된 detection 중 `score >= 0.3`인 detection만 사용했다.

IoU threshold는 다음과 같이 사용했다.

| Name | Value | Meaning |
|---|---:|---|
| `t_b` | 0.1 | background/foreground 구분용 IoU threshold |
| `t_f` | 0.5 | localization error 구분용 IoU threshold |

각 detection에 대해 다음 값을 계산했다.

| Symbol | Definition |
|---|---|
| `u_same` | 같은 class GT와의 최대 IoU |
| `u_diff` | 다른 class GT와의 최대 IoU |
| `u_max` | 모든 GT와의 최대 IoU |

Error type은 다음처럼 집계했다.

| Count | Unit | Definition |
|---|---|---|
| `GT` | GT object | 분석 대상 이미지에 포함된 oracle GT object 수 |
| `Det` | detection | `score >= 0.3` detection 수 |
| `BG` | detection | `u_max < t_b` |
| `Correct-like` | detection | `u_max >= t_b`, classification/localization error가 모두 아님 |
| `Cls` | detection | foreground detection 중 `u_diff > u_same` |
| `Loc` | detection | foreground detection 중 `u_max < t_f` |
| `Cls+Loc` | detection | `Cls`와 `Loc`에 동시에 해당 |
| `Miss` | GT object | 어떤 foreground detection으로도 설명되지 않은 GT object 수 |

`Cls+Loc`은 `Cls`와 `Loc`의 교집합이므로 `Cls`, `Loc` count에도 포함된다.

이 계산은 **coarse TIDE-like count**이다. 공식 TIDE의 duplicate error 분리는 별도로 구현하지 않았다.

## Round별 Error Count

| Method | Round | GT | Det | BG | Correct-like | Cls | Loc | Cls+Loc | Miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PAL | 1 | 1778 | 2601 | 174 | 1145 | 981 | 520 | 219 | 530 |
| PAL | 2 | 1614 | 2481 | 202 | 1119 | 780 | 569 | 189 | 381 |
| PAL | 3 | 1531 | 2545 | 273 | 1179 | 759 | 512 | 178 | 289 |
| PAL | 4 | 1602 | 2575 | 266 | 1149 | 803 | 598 | 241 | 346 |
| PAL | 5 | 1592 | 2442 | 239 | 1151 | 692 | 542 | 182 | 333 |
| PAL | 6 | 1416 | 2456 | 317 | 1112 | 657 | 568 | 198 | 225 |
| PAL | 7 | 1293 | 2179 | 237 | 1020 | 630 | 464 | 172 | 226 |
| PPAL | 1 | 2266 | 2570 | 259 | 1104 | 895 | 509 | 197 | 866 |
| PPAL | 2 | 2151 | 2782 | 317 | 1353 | 731 | 630 | 249 | 553 |
| PPAL | 3 | 1887 | 2462 | 343 | 1212 | 596 | 550 | 239 | 452 |
| PPAL | 4 | 1704 | 2295 | 298 | 1127 | 590 | 482 | 202 | 397 |
| PPAL | 5 | 1702 | 2258 | 279 | 1184 | 538 | 430 | 173 | 350 |
| PPAL | 6 | 1488 | 1928 | 218 | 998 | 471 | 395 | 154 | 331 |
| PPAL | 7 | 1438 | 1897 | 220 | 1001 | 480 | 341 | 145 | 296 |

## 7 Rounds 합계

| Method | Images | GT | Det | BG | Correct-like | Cls | Loc | Cls+Loc | Miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PAL | 2898 | 10826 | 17279 | 1708 | 7875 | 5302 | 3773 | 1379 | 2330 |
| PPAL | 2898 | 12636 | 16192 | 1934 | 7979 | 4301 | 3337 | 1359 | 3245 |

## 비율 비교

단순 count보다 비율로 보면 차이가 더 분명하다.

| Method | GT/image | Det/image | Det/GT | FG Det/GT | Miss/GT | Error Det/FG Det |
|---|---:|---:|---:|---:|---:|---:|
| PAL | 3.74 | 5.96 | 1.60 | 1.44 | 21.5% | 49.4% |
| PPAL | 4.36 | 5.59 | 1.28 | 1.13 | 25.7% | 44.0% |

여기서 `Det/GT`, `FG Det/GT`, `Miss/GT`가 핵심이다.

PAL selected images는 PPAL보다 GT 수가 적지만 detector가 더 많은 box를 낸다. 즉 detector가 이미 object
근처에 많이 반응하고 있는 이미지를 더 많이 고른 것이다. 그래서 missed object는 적다. 대신 그 detection들이
class나 localization에서 불안정하므로 `Cls`, `Loc` count가 크게 나온다.

PPAL selected images는 GT가 더 많지만, GT 수에 비해 detection 수가 적다. 즉 object는 더 많지만 detector가
아직 충분히 box를 내지 못한 이미지가 더 많이 섞인 것이다. 그래서 missed object가 더 많다. PPAL의 `Cls`,
`Loc` count가 낮아 보이는 것은 반드시 더 좋은 현상이 아니라, detection 자체가 덜 나왔기 때문에
detection-level error로 잡힐 기회가 적어진 결과일 수 있다.

## mAP/AP50 결과

현재 VOC 설정에서는 저장된 metric의 `mAP`가 사실상 AP50 기준 값으로 기록되어 있다.

| Round | PAL mAP | PPAL mAP |
|---:|---:|---:|
| 1 | 0.4913 | 0.4818 |
| 2 | 0.5910 | 0.6006 |
| 3 | 0.6457 | 0.6455 |
| 4 | 0.6683 | 0.6655 |
| 5 | 0.6811 | 0.6815 |
| 6 | 0.7029 | 0.6890 |
| 7 | 0.7045 | 0.7012 |

단일 seed 기준으로 PAL이 최종 round에서 약간 높다. 차이는 작기 때문에 강한 결론보다는 경향 분석으로 보는
것이 맞다.

## 알고리즘 차이에 따른 해석

### PAL

PAL은 LIUS와 GUIDE로 구성된다.

LIUS는 labeled pool에서 detection과 GT를 매칭해 class별 TP probability model을 학습한다. 그 다음
unlabeled detection마다 "이 detection이 TP일 확률이 얼마나 불확실한가"를 entropy로 점수화한다. 따라서
PAL은 기본적으로 detector가 낸 detection 중 맞는지 애매한 것을 찾는다.

GUIDE는 여기에 class imbalance, class-wise image entropy, predicted category diversity, representation
diversity를 더한다. 즉 단순히 uncertainty가 큰 detection만 고르는 것이 아니라 여러 class와 다양한
appearance를 포함하도록 보정한다.

이 구조 때문에 PAL은 detector가 object를 아예 못 본 이미지보다, box/class 예측은 하고 있지만 class가
헷갈리거나 localization이 불안정한 이미지를 더 잘 끌어온다.

결과적으로 PAL selected images는 다음 성격을 가진다.

- detector가 이미 box를 많이 내는 이미지
- missed object가 상대적으로 적은 이미지
- box/class가 애매한 foreground detection이 많은 이미지
- 현재 detector의 classification head와 box regression head를 직접 교정하기 쉬운 sample

### PPAL

PPAL은 DCUS와 CCMS로 구성된다.

DCUS는 unlabeled detection의 `cls_uncertainty * class_weight`를 image 단위로 합산해 uncertainty candidate
pool을 만든다. 이 방식은 detection이 존재하는 object의 uncertainty를 반영한다. 반대로 detector가 object를
아예 놓친 경우에는 해당 object가 uncertainty score를 직접 만들 수 없다.

CCMS는 DCUS candidate pool 안에서 detector feature distance 기반 diversity를 보고 최종 414장을 고른다.
즉 최종 selection은 "가장 error가 큰 이미지" 자체보다 feature space에서 다양한 이미지를 고르는 성격이 강하다.

이 구조 때문에 PPAL selected images는 다음 성격을 가진다.

- object가 더 많이 포함된 이미지
- feature diversity가 강하게 반영된 이미지
- detector가 아직 box를 충분히 내지 못한 object가 더 많이 남아 있는 이미지
- missed object가 상대적으로 많은 이미지

## 핵심 해석

이번 결과에서 PAL의 `Cls`, `Loc` count가 크다는 것은 "PAL이 더 나쁜 이미지를 골랐다"는 뜻이 아니다.
`Cls`, `Loc`은 detection 단위 count이기 때문에 detector가 box를 많이 낼수록 커질 수 있다.

반대로 `Miss`는 GT object 단위 count이다. detector가 아예 box를 내지 못하면 `Miss`가 증가한다.

따라서 이번 결과는 다음처럼 해석하는 것이 가장 자연스럽다.

PAL은 detector가 이미 object에 반응하고 있지만 decision boundary가 틀리는 이미지를 더 많이 골랐다. 이
이미지들은 labeling 후 classification/localization 교정에 바로 기여하기 쉽다.

PPAL은 더 많은 object와 diversity를 가진 이미지를 골랐지만, 그중 상당수는 current detector가 아직 충분히
box를 내지 못한 object였다. 그래서 selected set의 GT 수는 더 많지만 missed-object 비율도 더 높았다.

결국 PAL이 최종 AP에서 약간 높게 나온 이유는 "cls/loc error가 많아서"가 아니라, **missed object를 줄이면서
detector가 이미 반응하고 있는 어려운 foreground detection을 더 많이 labeling한 것**이 학습 효율에 유리하게
작용했을 가능성이 크다.

## 현재 Artifact로 계산할 수 없는 것

현재 저장된 artifact만으로는 다음 분석은 바로 계산할 수 없다.

- VOC test set 기준 error count
- PPAL labeled pool 전체 기준 error count
- 공식 TIDE와 동일한 duplicate/classification/localization/background/missed decomposition

이유는 다음과 같다.

- `eval_*.json`에는 mAP/AP50 metric만 저장되어 있고 VOC test prediction JSON은 저장되어 있지 않다.
- PPAL은 labeled inference prediction artifact가 별도로 저장되어 있지 않다.
- 현재 계산은 duplicate error를 별도로 분리하지 않는 coarse TIDE-like 기준이다.

## ECPAL 설계에 주는 시사점

PAL/PPAL error count 분석에서 가장 중요한 관찰은 다음이다.

좋은 active learning sample은 단순히 **GT object가 많은 이미지**가 아니라, **현재 detector의 decision
boundary를 많이 교정할 수 있는 GT/detection이 많은 이미지**에 가깝다.

PPAL은 selected images에 GT object가 더 많았다. 하지만 detector가 그 object들을 충분히 detection으로
만들지 못해 missed-object 비율도 더 높았다. 즉 annotation하면 새 object 정보는 많이 들어오지만, current
detector가 어떤 식으로 틀리고 있는지 직접 교정하는 signal은 상대적으로 약한 sample이 더 많이 포함됐다고
볼 수 있다.

PAL은 GT 수는 더 적었지만 detector가 이미 box를 많이 냈고, 그 box들이 class/localization에서 헷갈리는
경우가 많았다. 이런 이미지는 labeling 후 "이 object는 이 class가 맞다", "box는 이렇게 조정해야 한다"는
학습 signal이 바로 생긴다. Active learning 관점에서는 이런 sample이 더 효율적일 수 있다.

따라서 좋은 acquisition은 다음 두 축을 함께 봐야 한다.

| Axis | Meaning | Expected benefit |
|---|---|---|
| Confusing foreground | detector가 object 근처까지는 왔지만 class/box/confidence 판단이 불안정한 경우 | classification/localization decision boundary 교정 |
| Uncovered object/region | detector가 아직 잘 못 보는 object, scene, rare class | recall 보완, missed object 감소 |

PAL은 첫 번째 축을 더 잘 잡았고, PPAL은 diversity 때문에 두 번째 축과 GT-rich 이미지 쪽으로 더 기운 것으로
해석할 수 있다.

## ECPAL의 현재 Score

현재 ECPAL은 image의 error 개수를 직접 예측해 acquisition score로 사용한다.

Classification/localization error는 hard count regressor가 아니라 logistic probability의 expected count로
계산한다.

```text
n_cls_hat  = sum q_fg * q_cls_given_fg
n_loc_hat  = sum q_fg * q_loc_given_fg
n_miss_hat = Poisson predicted count
```

여기서 각 값의 의미는 다음과 같다.

| Quantity | Meaning |
|---|---|
| `q_fg` | detection이 foreground-related일 확률 |
| `q_cls_given_fg` | foreground-related detection이 classification error일 조건부 확률 |
| `q_loc_given_fg` | foreground-related detection이 localization error일 조건부 확률 |
| `n_miss_hat` | image-level missed-object 예상 개수 |

따라서 현재 ECPAL score는 이미 "이 이미지에 학습할 error signal이 얼마나 많이 있을 것인가"를 직접 보는
형태이다. 이 방향 자체는 유지하는 것이 좋다.

## Entropy를 단독 Score로 쓰면 생기는 문제

Regressor/predictor의 예측값 자체 대신 entropy를 보는 아이디어는 타당하다. Entropy가 높다는 것은 해당
detection이 어떤 error인지 predictor가 확신하지 못한다는 뜻이기 때문이다. 하지만 entropy만으로 acquisition을
하면 문제가 생긴다.

예를 들어 classification error probability가 다음과 같다고 하자.

```text
q_cls_given_fg = 0.99
```

이 detection은 classification error일 가능성이 매우 높다. Label을 추가하면 detector를 교정하는 데 유용할 수
있다. 하지만 Bernoulli entropy는 낮다. Entropy-only acquisition은 이런 거의 확실한 error를 낮게 평가한다.

반대로 다음과 같은 경우를 생각할 수 있다.

```text
q_cls_given_fg = 0.50
```

이 값의 entropy는 최대이다. 하지만 이것이 detector가 진짜 헷갈리는 것인지, 아니면 ECPAL의 error predictor가
아직 해당 feature region을 잘 모르는 것인지는 구분하기 어렵다. 즉 entropy는 **detector confusion**일 수도
있고, **surrogate error predictor uncertainty**일 수도 있다.

따라서 entropy-only acquisition은 추천하지 않는다.

## 권장 방향: Expected Error Count + Error Uncertainty

ECPAL의 핵심인 expected error count는 유지하고, entropy는 secondary signal로 추가하는 것이 더 적절하다.

목표는 다음 형태가 된다.

```text
good_sample = high expected error count
              + high uncertainty among foreground-related errors
              + enough diversity across error profiles
```

즉 최종 score는 다음 세 성격의 균형을 잡는 것이 좋다.

| Component | Role |
|---|---|
| Expected error count | 학습에 들어올 error signal의 양 |
| Error uncertainty | detector/predictor가 헷갈리는 foreground error의 정도 |
| Error-profile diversity | 같은 error profile만 반복 선택하는 것을 방지 |

가장 보수적인 score 형태는 다음이다.

```text
ECA = weighted expected count
ECU = weighted error uncertainty
score = ECA * (1 + lambda * normalized_ECU)
```

여기서 uncertainty는 penalty가 아니라 boost로 쓰는 것이 기본 제안이다. `ECA`가 낮은 이미지는 uncertainty가
높더라도 score가 크게 올라가지 않고, `ECA`가 높은 이미지 중 더 헷갈리는 sample이 우선된다.

Additive 형태도 가능하다.

```text
score = ECA + lambda * normalized_ECU
```

하지만 이 경우 `ECA`가 낮고 entropy만 높은 이미지가 상위로 올라올 수 있으므로, 초기 구현에서는
multiplicative boost가 더 안전하다.

`lambda`는 작게 시작하는 것이 좋다.

```text
lambda = 0.1 ~ 0.3
```

이렇게 하면 ECPAL의 핵심인 error-count prediction은 유지하면서, PAL이 잘 잡았던 confusing foreground 성격을
일부 흡수할 수 있다.

## ECPAL Error Uncertainty 계산안

Detection-level binary predictors에 대해서는 Bernoulli entropy를 사용할 수 있다.

```text
H(p) = -p log(p) - (1 - p) log(1 - p)
```

Image-level uncertainty는 다음처럼 계산할 수 있다.

```text
U_fg  = sum H(q_fg)
U_cls = sum q_fg * H(q_cls_given_fg)
U_loc = sum q_fg * H(q_loc_given_fg)
```

`U_cls`, `U_loc`에 `q_fg`를 곱하는 이유는 background일 가능성이 높은 detection의 class/localization
uncertainty가 과도하게 반영되는 것을 막기 위해서이다. 우리가 강조하고 싶은 것은 foreground-related detection
중 class/localization 판단이 헷갈리는 경우이다.

Weighted uncertainty는 기존 ECPAL의 error-type weight를 재사용할 수 있다.

```text
ECU = w_cls * U_cls + w_loc * U_loc
```

`U_fg`는 별도 항으로 둘지, `U_cls/U_loc` 계산을 위한 gate로만 둘지 선택할 수 있다. 현재 PAL/PPAL 분석의
시사점은 foreground object 주변의 class/box confusion이 중요하다는 것이므로, 초기 구현에서는 `U_cls`,
`U_loc`를 우선하고 `U_fg`는 diagnostics로 저장하는 편이 안전하다.

## Missed-object Uncertainty는 다르게 봐야 함

Missed-object predictor인 `MOCP`는 Poisson count predictor이다. Binary Bernoulli entropy가 바로 적용되는
`FDP`, `CECP`, `LECP`와 다르다.

Poisson entropy를 사용할 수는 있지만, Poisson entropy는 대체로 predicted count가 커질수록 함께 커지는 경향이
있다. 그러면 `n_miss_hat`과 역할이 많이 겹친다. 진짜 model uncertainty를 보려면 ensemble, bootstrap,
MC dropout 같은 별도 불확실성 추정이 필요하다.

따라서 현재 구조에서는 missed-object 쪽은 다음처럼 두는 것이 좋다.

```text
miss contribution = w_miss * n_miss_hat
```

즉 `miss`는 expected count 중심으로 유지하고, entropy boost는 우선 `cls`, `loc` foreground confusion에만
적용한다.

## 정리

ECPAL을 PAL/PPAL 분석 결과에 맞게 개선하려면 다음 방향이 적절하다.

1. Expected error count score는 유지한다.
2. Entropy-only acquisition은 사용하지 않는다.
3. `CECP`, `LECP`의 Bernoulli entropy를 이용해 foreground classification/localization confusion을 보조 항으로 추가한다.
4. `score = ECA * (1 + lambda * normalized_ECU)`를 우선 검토한다.
5. `lambda`는 작게 시작한다.
6. Missed-object는 당장은 `n_miss_hat` expected count 중심으로 둔다.
7. ECD/farthest-first diversity는 error composition diversity를 유지하는 역할로 계속 사용한다.

이 방향은 ECPAL이 단순히 "error가 많을 것 같은 이미지"만 고르는 것을 넘어서, **현재 detector가 object 근처까지는
왔지만 class/localization 판단을 헷갈리는 이미지**를 더 잘 고르도록 만드는 개선이다.
