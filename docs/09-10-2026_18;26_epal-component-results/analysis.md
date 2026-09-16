# EPAL CECP / LECP / CECP+LECP 결과 분석

## 목적과 범위

2026-09-10 종료된 EPAL component ablation 3개를 기존 PAL 및 Random과 비교한다.
실제 평가 JSON, 저장 설정, 학습 로그, TensorBoard event, acquisition diagnostics,
선택 이미지 ID를 읽었다. 새로운 학습, 추론, predictor 재학습은 실행하지 않았다.
이번 변경은 이 분석 문서 추가뿐이며 원본 실험 결과와 코드는 수정하지 않았다.

## 원본 실험

아래 경로는 모두 `work_dirs/current_work/` 기준이며 분석 seed는 `seed_0`이다.

| 구분 | run 경로 |
| --- | --- |
| CECP | `retinanet_voc_epal_cecp_7rounds_5percent_to_20percent/09-08-2026_20;22/seed_0` |
| LECP | `retinanet_voc_epal_lecp_7rounds_5percent_to_20percent/09-09-2026_17;33/seed_0` |
| CECP+LECP | `retinanet_voc_epal_cecp_lecp_7rounds_5percent_to_20percent/09-10-2026_00;47/seed_0` |
| PAL | `retinanet_voc_pal_7rounds_5percent_to_20percent/09-04-2026_19;37/seed_0` |
| Random | `retinanet_voc_random_7rounds_5percent_to_20percent/09-05-2026_19;11/seed_0` |

주요 자료는 각 round의 `eval_*.json`, 저장된 학습 config,
`active_learning_plan.json`, `round_summary.json`, `run_summary.json`,
`epal_diagnostics.json`, `epal_candidates.json`, annotations 및 seed event 파일이다.
중단/복구 경위는 기존 `docs/09-10-2026_16;53_epal-round7-interruption/` 및
`docs/09-10-2026_16;57_epal-recovery-config-paths/` 기록을 함께 확인했다.

## 비교 조건 및 완료 여부

- RetinaNet / VOC, seed 0, train batch 16, inference batch 16, LR 0.008,
  26 epochs, LR step 20, warm-up 32 iterations, 학습 AMP, gradient clip 35.
- 저장 설정의 model, optimizer, optimizer hook, LR schedule, runner, fp16,
  checkpoint 설정과 데이터 pipeline/batch/worker 설정을 5 methods × 7 rounds에서
  비교했다. Pool별 annotation 경로를 제외한 비교 대상 설정은 동일했다.
  Windows 경로 문자열은 읽기 전용 AST 비교 과정에서만 slash로 정규화했다.
- 실행 계획의 모든 평가 명령은 VOC2007 test split과 `fp16=None`을 사용한다.
- 초기 labeled 827장과 실제 image ID 집합이 5개 방법에서 같고, round 1 평가값도
  완전히 같다. 이후 round당 414장씩 6회 추가하여 마지막 학습 pool은 3,311장이다.
- EPAL 18개 acquisition 모두 선택 수가 414이며, 이전 labeled pool과 중복 없이
  저장된 다음 labeled pool의 증가분과 정확히 일치한다.
- 35개 round 모두 native 평가 JSON과 비어 있지 않은 `epoch_26.pth`가 있다.

### CECP+LECP round 7: 실제 완료, 상위 기록은 미완료

이전 `round_07/20260910_043719.log.json`은 epoch 13 / iteration 2,704에서 끝난다.
복구 학습 `round_07/20260910_170003.log.json`은 epoch 26 / iteration 5,408까지
완료됐다. 마지막 loss는 0.24318, grad norm은 1.70497이다.
새 `epoch_26.pth` 및 `latest.pth`는 각각 293,055,453 bytes이며,
`round_07/eval_20260910_174501.json`의 `metric.mAP`는 0.7142027020454407이다.

반면 `run_summary.json`은 아직 `running`이고, round 7의 `round_summary.json`과
완료 aggregate summary가 없다. seed TensorBoard의 round 7 학습 trace도 이전 실행의
iteration 2,704에서 끝난다. 따라서 상위 summary/TensorBoard만 보면 완료 결과를
놓칠 수 있다. 아래 표는 새 native 평가 JSON을 직접 읽어 작성했다.
이 작업에서는 해당 기록들을 자동 보정하거나 덮어쓰지 않았다.

## Round별 AP50

단위는 %. VOC의 `metric.mAP × 100`을 사용했으며 COCO AP@[.5:.95]가 아니다.
동일 JSON의 `AP50` 필드는 이미 소수 셋째 자리로 반올림되어 있어 집계에는 쓰지 않았다.
Labeled 비율은 명목상 비율이다.

| Round | Labeled | CECP | LECP | CECP+LECP | PAL | Random |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 5% | 34.32 | 34.32 | 34.32 | 34.32 | 34.32 |
| 2 | 7.5% | 55.28 | 29.78 | 50.20 | 54.69 | 42.81 |
| 3 | 10% | 57.62 | 53.78 | 59.35 | 61.75 | 59.16 |
| 4 | 12.5% | 63.80 | 58.21 | 63.33 | 64.42 | 63.09 |
| 5 | 15% | 68.74 | 68.22 | 68.18 | 67.53 | 64.95 |
| 6 | 17.5% | 69.03 | 69.15 | 70.11 | 69.65 | 68.78 |
| 7 | 20% | 71.72 | 70.89 | 71.42 | 70.35 | 69.36 |

| 요약 지표 | CECP | LECP | CECP+LECP | PAL | Random |
| --- | ---: | ---: | ---: | ---: | ---: |
| 최종 AP50 | 71.72 | 70.89 | 71.42 | 70.35 | 69.36 |
| 최종 PAL 대비 차이 (pp) | +1.37 | +0.55 | +1.07 | 0.00 | -0.99 |
| 최종 Random 대비 차이 (pp) | +2.36 | +1.54 | +2.06 | +0.99 | 0.00 |
| Round 1–7 산술평균 | 60.07 | 54.91 | 59.56 | 60.39 | 57.49 |
| Round 5–7 산술평균 | 69.83 | 69.42 | 69.90 | 69.17 | 67.70 |
| 정규화 사다리꼴 면적 | 61.25 | 55.29 | 60.67 | 61.73 | 58.44 |

정규화 면적은 동일한 labeled 증가 간격에서
`(0.5*A1 + A2 + A3 + A4 + A5 + A6 + 0.5*A7) / 6`이다.
이 평균/면적은 단일 learning curve의 기술 통계이지 반복 실험 평균이나 유의성 검정이 아니다.

## 해석

### 1. CECP 단독의 최종 결과가 가장 좋다

CECP는 최종 71.7195로 PAL보다 1.3729 pp, Random보다 2.3605 pp 높다.
CECP+LECP보다 0.2993 pp, LECP보다 0.8248 pp 높다.
다만 round 3에서는 PAL 및 Random보다 낮고, 전체 curve 평균과 면적은 PAL이 높다.
따라서 "최종 예산에서 개선"은 관측되지만 "모든 예산에서 PAL보다 효율적"이라고
주장할 수는 없다.

### 2. CECP+LECP의 일관된 결합 이득은 아직 없다

조합은 CECP보다 round 3에서 1.7255 pp, round 6에서 1.0816 pp 높지만,
round 2/4/5/7에서는 각각 5.0740/0.4747/0.5592/0.2993 pp 낮다.
마지막 세 round 평균은 조합이 0.0744 pp 높을 뿐이다.
현 seed에서 조합의 일부 round 개선은 있지만, 상보성 또는 안정적인 시너지의 증거로
보기에는 부족하다. 최종 0.30 pp 차이만으로 LECP를 무용하다고 단정하는 것도 이르다.

### 3. LECP의 문제는 초반 성능 저하이며, 지속적 수치 발산으로 확인되지는 않았다

Round 1 → 2 AP50는 34.3160 → 29.7803으로 4.5357 pp 감소했다.
Round 2는 CECP보다 25.4982 pp, PAL보다 24.9141 pp, Random보다 13.0286 pp 낮다.
그러나 round 5부터 68.22 → 69.15 → 70.89로 회복하여 최종 PAL/Random을 넘는다.
초반 실패는 round 1 acquisition으로 선택한 pool을 학습한 결과이며,
round 2 acquisition의 결과와 혼동하면 안 된다.

LECP round 2의 TensorBoard scalar와 console text를 함께 읽은 결과:

- 학습은 전체 2,054 iterations까지 완료됐다.
- Loss 평균은 iteration 1–100에서 1.8895, 101–500에서 1.7058,
  501–1,000에서 1.1278, 1,001–1,500에서 0.8337,
  1,501–2,000에서 0.5995, 2,001–2,054에서 0.5366으로 감소했다.
- 최대 loss는 2.0903, 마지막 loss는 0.5357이며, nonfinite loss는 확인되지 않았다.
- Iteration 346에 `grad_norm=inf`가 한 번 있다. 당시 loss는 1.9377이고 이후 학습은
  계속 진행됐다. 이 한 건만으로 성능 하락의 원인을 AMP overflow라고 확정할 수 없다.
- 마지막 100 iterations 평균 loss는 CECP 0.3372, LECP 0.5308, 조합 0.3917이다.
  서로 다른 pool의 training loss이므로 일반화 성능의 공정한 직접 비교 지표는 아니지만,
  LECP의 낮은 AP가 지속적인 loss 폭발과 동반된 패턴은 아니다.

전체 EPAL 로그에서도 드문 nonfinite gradient가 있었다. CECP는 round 4/7에서 각 1회,
LECP는 round 2/3/4에서 각 1회, 조합은 round 4 및 복구 round 7에서 각 1회이다.
복구 round 7의 해당 행은 epoch 21 / iteration 4,174, loss 0.44034이다.
확인한 로그에서 nonfinite loss나 지속적인 폭발 패턴은 없었다.
이는 수치 안정성이 완벽하다는 뜻도, 해당 overflow가 무해함을 입증했다는 뜻도 아니다.

현재 증거만으로 선택 pool의 구성, 제한된 학습에서의 최적화 차이, 확률적 변동 중
무엇이 LECP의 초반 저하를 주도했는지 분리할 수 없다. 낮은 round 2를 임의로 제외하지 않았다.

## Acquisition 및 component 동작 확인

`methods/epal/components.py`, `predictors.py`, `scoring.py`, `acquisition.py`를
기록된 diagnostics와 대조했다.

- CECP 입력 feature는 `[score, E_cls]`, LECP 입력 feature는 `[score, E_loc]`이다.
  `score`는 detection score이며, 이번 세 실험은 MOCP 및 FDP를 사용하지 않는다.
- 이미지 점수는 활성 predictor의 오류 확률에 대한 Bernoulli entropy를 detection별로
  합산한 EUA이다. 예측 오류 확률 그 자체나 실제 오류 개수를 합하는 것이 아니다.
- Predictor fitting은 foreground-eligible labeled detection을 사용하지만, EUA scoring은
  모든 final detection에 적용된다. 별도의 FDP gating이나 diversity selection은 없다.
- 18개 acquisition의 활성 predictor는 모두 실제 fitting됐고 fallback 기록은 없다.
  의도한 component만 활성화됐으며 단일 component weight는 1이다.
- 첫 round에서 두 predictor의 fitting 표본은 동일한 3,859개이고, CECP positive 비율은
  46.72%, LECP는 7.41%이다. 이는 학습 target 비율이지 predictor 정확도가 아니다.

조합의 가중치는 labeled pool의 평균 uncertainty 역수로 계산하고 합을 2로 정규화한다.
따라서 CECP와 LECP를 단순 고정 1:1로 더한 실험이 아니다.

| Acquisition round | CECP weight | LECP weight | 선택 414장 점수의 CECP 기여율 | LECP 기여율 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.6824 | 1.3176 | 46.64% | 53.36% |
| 2 | 0.8704 | 1.1296 | 47.01% | 52.99% |
| 3 | 0.9044 | 1.0956 | 46.99% | 53.01% |
| 4 | 0.9079 | 1.0921 | 47.67% | 52.33% |
| 5 | 0.9397 | 1.0603 | 47.46% | 52.54% |
| 6 | 0.9203 | 1.0797 | 47.63% | 52.37% |

기여율은 selected candidates의 weighted component score 총합을 전체 score 총합으로
나눈 값이다. 실제 오류 감소량이나 인과적 중요도를 의미하지 않는다.
LECP가 점수에서 무시되거나 한쪽만 100% 지배한 상황은 아니다.

첫 acquisition의 414장 중 선택 교집합은 CECP–LECP 319장(77.05%),
CECP–조합 361장(87.20%), LECP–조합 372장(89.86%)이다.
동일 초기 상태의 선택이 상당히 겹치면서 일부 달라진다. 이후 round는 학습 pool과
모델 자체가 달라지므로 같은 round의 선택 overlap만으로 predictor 상보성을 주장할 수 없다.
이 수치는 이미지 다양성을 입증하는 지표로 사용하지 않는다.

## 결론과 한계

측정 결과는 "seed 0의 최종 20% 예산에서 세 EPAL 구성이 PAL/Random보다 높고,
CECP 단독이 최종 최고"이다. 반면 전체 예산 구간의 평균 성능은 PAL이 최고이고,
조합의 일관된 개선이나 LECP 초반 안정성은 확보되지 않았다.

단일 seed이므로 통계적 우월성이나 일반적인 method 순위를 확정하지 않는다.
또한 이번 비교는 downstream AP50와 실행/선택 기록 분석이다. CECP/LECP의
held-out AUROC, calibration, 실제 classification/localization 오류 감소량을 측정한
실험은 아니며, AP50만으로 LECP가 localization error를 얼마나 줄였는지 알 수 없다.
성공적인 fitting과 정상적인 선택 수를 predictor의 높은 예측력으로 해석하면 안 된다.

## 검증 및 변경 내역

- `python -B -` 읽기 전용 분석으로 35개 평가 JSON, config AST signature,
  초기/선택 pool ID 및 18개 acquisition diagnostics를 비교했다.
- 같은 환경에서 `LegacyEventFileLoader`로 세 EPAL seed의 기존 event를 읽어
  학습 scalar와 console text를 분석하고, 복구 round 7의 native log JSON을 별도로 읽었다.
  Scalar parser는 nonfinite 값이 누락될 수 있어 console text도 확인했다.
- 실행 계획의 평가 split/FP32 override 일치와 복구 round 7의 마지막 epoch/iteration,
  checkpoint 크기 및 새 평가값을 확인했다.
- 문서 추가 후 `git diff --check` 및 새 문서의 trailing whitespace 확인을 수행했다.
- 코드, 설정, checkpoint, summary 및 TensorBoard event는 변경하지 않았다.
  새로운 GPU 작업, 재평가, test/smoke workflow는 실행하지 않았다.
