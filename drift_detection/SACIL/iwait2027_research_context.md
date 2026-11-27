# IWAIT 2027 연구 Context 정리

> 작성 기준일: 2026-09-06  
> 목적: 기존 PowerPoint를 수정할 AI와 연구 보조 AI가 현재 연구의 사실, 가설, 한계, 미완료 항목을 혼동하지 않도록 하는 기술 문서  
> 현재 paper-facing 표현: **Class Prototype Transformation**  
> 저장소 내부 구현명: **CMPT** (`cmpt`, `affine_ridge`)  
> 주의: 논문 제목과 최종 방법 acronym은 아직 확정하지 않는다.

## 증거 태그와 읽는 법

- `[Verified]`: 현재 저장소의 코드, config 또는 결과 JSON에서 직접 확인했다.
- `[Literature]`: 공개된 원 논문 또는 공식 proceedings에 근거한다.
- `[Hypothesis]`: 실험 결과에 대한 연구자의 해석이거나 아직 일반화되지 않은 설명이다.
- `[Planned]`: 필요하지만 아직 수행하지 않은 실험이다.
- `[Unknown]`: 저장소와 확인한 문헌만으로 확정할 수 없거나 연구자의 결정이 필요하다.

`accuracy`는 별도 언급이 없으면 백분율(%)이고, 두 accuracy의 차이는 percentage point(%p)이다. 주 결과의 모든 NME–Affine 차이는 **동일 checkpoint, 동일 exemplar, 동일 query**에서 prototype estimator만 바꾼 paired comparison이다. 핵심 근거 파일은 `mds/results/cmpt_nme_vs_affine_results_current.md`, 원본 JSON은 `outputs/cmpt/**/evaluation/**.json`이다.

---

## 1. 연구를 한 문단으로 요약

[Verified] 이 연구는 exemplar-based class-incremental learning(CIL)에서 각 old class를 20개 exemplar의 현재 feature 평균으로 나타내는 NME classifier가 전체 class distribution의 현재 평균을 충분히 대표하지 못하는 문제를 다룬다. [Verified] CIFAR-100의 7개 learner에서 old class의 현재 full-training-data mean을 사용하는 oracle은 표준 20-exemplar NME보다 평균 AIA가 `+1.838%p` 높아, prototype estimation에 실제 성능 headroom이 있음을 보였다. [Verified] 제안 방법은 class가 처음 등장했을 때 전체 current-session training data로 계산한 고품질 prototype을 저장하고, 다음 session마다 동일한 old exemplar를 이전 모델과 현재 모델에 통과시킨 paired feature로 하나의 ridge-regularized affine map `(A_t, b_t)`을 closed form으로 추정한 뒤 모든 old-class prototype을 현재 feature space로 운반한다. [Verified] CIFAR-100/ResNet-32와 ImageNet-100/ResNet-18의 7개 learner, 총 14개 seed-1 trajectory에서 Affine CMPT는 NME 대비 AIA를 모두 높였으며, 평균 증가는 각각 `+0.457%p`, `+0.360%p`였다. 가장 큰 개별 증가는 CIFAR-100 LUCIR-natural의 AIA `+1.291%p`, final accuracy `+1.38%p`이다. [Hypothesis] 논문의 가장 방어 가능한 contribution은 새로운 affine 수학 자체가 아니라 **exemplar-based NME의 full-data prototype gap을 다수 learner에서 진단하고, 저장된 old exemplars를 consecutive feature spaces 사이의 직접 paired landmarks로 재해석하여 introduction-time full-data prototype을 갱신하는 learner-independent post-hoc estimator를 제시한 것**이다. [Literature] 다만 LDC(ECCV 2024) 등 exemplar-free prototype drift compensation 선행연구가 이미 old/current feature mapping과 prototype update를 제안했으므로, “최초의 prototype transport” 또는 “affine mapping 자체가 novelty”라고 주장하면 안 된다.

---

## 2. 연구 배경과 문제 설정

### 2.1 Class-Incremental Learning의 정확한 정의

- `[Literature]` CIL에서는 서로 겹치지 않는 class 집합이 session별로 순차 도착하며, test 때 task/session ID를 제공하지 않는다. 모델은 현재까지 본 모든 class를 하나의 label space에서 구분해야 한다. iCaRL은 이를 “지금까지 관측된 모든 class에 대한 multi-class classifier를 언제나 제공하는” 설정으로 정의한다.
- `[Verified]` 본 실험에서 session `t`의 seen-class 집합은

  $$
  \mathcal C_{\le t}=\bigcup_{k=0}^{t}\mathcal C_k,
  \qquad
  \mathcal C_j\cap\mathcal C_k=\varnothing\;(j\ne k)
  $$

  이며, 평가 시 classifier의 후보는 항상 `\mathcal C_{\le t}` 전체다.
- `[Verified]` 본 설정은 domain-OOD detection이 아니다. 새 class는 현재 모델 관점에서 label-space novelty이지만, CIFAR-100 또는 ImageNet-100이라는 동일 benchmark 안의 class다.

### 2.2 Exemplar-based CIL

- `[Literature]` Exemplar-based 또는 rehearsal-based CIL은 과거 class의 입력 이미지 일부를 제한된 memory에 저장하고, 이후 session의 학습 또는 분류에 다시 사용한다.
- `[Verified]` 본 실험은 class당 20개 exemplar를 유지한다. S0에는 `50×20=1,000`개, S10에는 `100×20=2,000`개가 저장된다. selection은 모든 primary learner에서 iCaRL-style greedy herding으로 통일했다.
- `[Verified]` incremental session `t>0`의 training에서 접근 가능한 데이터는 현재 새 class의 전체 training split과 old class의 저장 exemplar다. old class의 나머지 training image는 사용할 수 없다.
- `[Verified]` CMPT는 exemplar memory를 늘리지 않지만, class당 한 개의 introduction-time full-data prototype vector를 auxiliary state로 추가한다.

### 2.3 Base N–Inc M

- `[Literature]` `Base N–Inc M`은 첫 session에서 `N`개 class를 학습하고, 이후 incremental session마다 `M`개 class를 추가한다는 뜻이다.
- `[Verified]` 본 연구의 `B50-Inc5`는 S0에서 50개를 학습하고 S1–S10에서 매번 5개씩 추가한다. 따라서 incremental session은 10개이고, base를 포함한 평가 point는 총 11개다.
- `[Verified]` 표기 충돌을 피하기 위해 이 문서에서는 base class 수를 `N_base=50`, increment class 수를 `M=5`, 마지막 session index를 `T=10`으로 쓴다. Affine fitting에 쓰는 paired support 수에는 `n_t`를 사용한다.

### 2.4 모델의 학습·평가 시점

- `[Verified]` S0 학습 후 `f_0`를 평가하고, 각 incremental session `t`의 학습이 끝날 때마다 `f_t`를 all-seen test set에서 평가한다.
- `[Verified]` transition `t-1→t`의 CMPT fitting에는 frozen `f_{t-1}`와 학습이 끝난 `f_t`만 필요하다. `f_0,…,f_{t-2}`를 동시에 유지할 필요는 없다.
- `[Verified]` 최종 inference에는 현재 feature extractor `f_t`와 현재 prototype bank만 사용한다. 이전 model은 다음 affine map을 맞추는 session 종료 시점에만 일시적으로 필요하다.
- `[Verified]` 현재 실험 구현은 이미 저장된 S0–S10 checkpoint를 순서대로 다시 불러오는 offline evaluator다. 실제 online 배포에서는 class introduction 시 full-data prototype을 즉시 저장하고, session 종료 직후 이전/현재 모델로 affine update를 수행하면 같은 정보 제약을 만족한다.

### 2.5 Catastrophic forgetting의 구체적 원인

- `[Literature]` 새 class 중심의 gradient update는 old class에 유용한 parameter와 representation을 바꾸고, 제한된 old memory와 많은 new data의 불균형은 learned classification head를 new class 쪽으로 편향시킬 수 있다.
- `[Verified]` 본 연구가 직접 다루는 하위 문제는 representation drift 이후의 **prototype estimation error**다. 현재 모델에서 20개 exemplar를 다시 forward하면 좌표계는 최신이지만, 그 평균이 현재 full class mean과 달라질 수 있다.
- `[Hypothesis]` Prototype displacement는 nearest-prototype decision boundary를 이동시켜 일부 query가 잘못된 class prototype에 더 가까워지게 한다.
- `[Verified]` 이것은 forgetting의 유일한 원인이 아니다. CIFAR-100 pooled 분석에서 prototype cosine error와 oracle accuracy headroom은 강한 양의 상관을 보였지만, LUCIR와 PODNet의 learner 내부 상관은 음수였다. Training-induced representation damage, class overlap, classifier bias 등은 CMPT가 직접 해결하지 않는다.

---

## 3. 용어와 기호 정의

### 3.1 용어

| 용어 | 정확한 정의 | 본 연구에서의 사용 |
|---|---|---|
| feature extractor / backbone | 이미지 `x`를 `d`차원 representation으로 변환하는 신경망 `f_t` | CIFAR-100은 ResNet-32 계열, ImageNet-100은 ResNet-18 계열 |
| classifier | feature 또는 이미지에서 최종 class label을 결정하는 전체 decision rule | learned head일 수도 있고 nearest-prototype rule일 수도 있음 |
| classification head | backbone 뒤의 학습 가능한 FC, cosine, multi-proxy 모듈 | learner 학습에는 사용될 수 있으나 primary CMPT paired evaluation에서는 NME로 교체 |
| parametric classifier | 학습 가능한 weight를 가진 classifier | linear FC, cosine head, multi-proxy head |
| non-parametric classifier | 별도 gradient로 학습되는 head parameter 없이 저장 sample/statistic으로 결정하는 classifier | exemplar mean이나 transported prototype을 이용한 nearest-prototype classifier |
| nearest-prototype classifier (NPC) | query feature와 class별 representative vector의 거리를 비교해 가장 가까운 class를 고르는 큰 범주의 classifier | NME와 CMPT classifier를 모두 포함하는 umbrella term. `NPC` acronym은 분야 전체에서 유일하게 표준화된 명칭이라고 단정하지 않는다. |
| nearest class mean (NCM) | class mean을 prototype으로 사용하는 nearest-prototype classifier | full-data mean 또는 저장된 class prototype을 쓰는 문헌에서 넓게 사용 |
| NME | **Nearest-Mean-of-Exemplars**. 저장 exemplar의 normalized feature 평균을 class prototype으로 만들고 가장 가까운 mean을 선택하는 iCaRL식 classifier | NPC의 특수한 경우이며, 단순히 prototype 계산만이 아니라 그 prototype을 이용한 decision rule까지 포함 |
| exemplar | old class에서 memory에 보존한 실제 training image | class당 20개 |
| exemplar feature | exemplar `x_i`를 특정 session의 extractor에 통과시킨 normalized vector | `v_{t,i}` |
| exemplar-based prototype / memory mean | 현재 `f_t`에서 한 class의 저장 exemplar feature를 평균하고 다시 normalize한 vector | baseline NME prototype `\mu^{mem}_{t,c}` |
| full-data prototype / full-data mean | 한 class의 **training split 전체**를 현재 `f_t`에 통과시켜 얻은 normalized-feature mean | class introduction 시에는 사용 가능; old class의 현재 값은 oracle만 가능 |
| class mean | 모호한 표현 | 논문에서는 반드시 `memory mean`, `introduction full-data mean`, `current full-data oracle mean` 중 하나로 수식어를 붙인다. |
| full training data | test/validation image가 아니라 해당 class의 training split 전체 | CIFAR-100 class당 500장; ImageNet-100은 class별 1,071–1,300장 |
| old class | session `t`보다 먼저 등장한 class `c∈C_{<t}` | CMPT로 prototype을 교체하는 대상 |
| new/current class | session `t`에 처음 등장한 class `c∈C_t` | 현재 session에서는 baseline NME mean을 그대로 사용하고 full-data mean은 미래 transport용으로 저장 |
| seen class | S0부터 현재 session까지 등장한 모든 class `C_{≤t}` | evaluation 후보 전체 |
| post-hoc | base learner의 training objective와 gradient를 바꾸지 않고, 학습된 checkpoint에서 classifier/prototype을 갱신하는 것 | CMPT의 현재 scope |

### 3.2 기호

| 기호 | shape | 정의 | 코드 대응 | 계산/지속 여부 |
|---|---:|---|---|---|
| `t∈{0,…,T}` | scalar | session index, 본 실험은 `T=10` | `session_id` | protocol에 고정 |
| `f_t` | `X→R^d` | session `t` 학습 후 feature extractor | `current_model`/checkpoint model | 현재 `f_t` 지속, 직전 `f_{t-1}`은 fitting 중 일시 사용 |
| `x_i` | image | paired landmark로 쓰는 동일 old exemplar | previous checkpoint의 `memory_indices`가 가리키는 image | exemplar memory에 지속 |
| `v_{t-1,i}` | `R^d` | `normalize(f_{t-1}(x_i))` | `paired_support.old_fit_features` | session transition 중 일시적 |
| `v_{t,i}` | `R^d` | `normalize(f_t(x_i))` | `paired_support.current_fit_features` | session transition 중 일시적 |
| `\mu^{mem}_{t,c}` | `R^d` | `f_t`에서 20 exemplar로 계산한 NME mean | checkpoint/evaluator `class_means` | session별 계산·저장 |
| `\mu^{full}_{t,c}` | `R^d` | `f_t`에서 class `c`의 full training data로 계산한 mean | `full_mean_oracle` diagnostics | old class에 대해 실제 CIL에서는 접근 불가; oracle |
| `\mu^{intro}_c` | `R^d` | class가 등장한 `s(c)`에서의 `\mu^{full}_{s(c),c}` | `_full_introduction_prototypes` output | 등장 session에 계산해 class당 하나 저장 |
| `\tilde\mu_{t,c}` | `R^d` | session `t`까지 affine transport된 prototype | evaluator의 `transported` bank | CMPT prototype bank에 지속 |
| `A_t` | `R^{d×d}` | `t-1` coordinates를 `t` coordinates로 보내는 session-specific linear part | `mapping[:-1]`의 transpose가 column convention `A_t` | transition마다 계산; bank update 후 폐기 가능 |
| `b_t` | `R^d` | affine translation vector. **matrix가 아니다.** | `mapping[-1]` | transition마다 계산; bank update 후 폐기 가능 |
| `\lambda_{aff}` | scalar | `A_t`에 대한 ridge penalty coefficient | config `cmpt.affine_ridge` / argument `ridge` | 현재 모든 primary run에서 `0.01` |
| `n_t` | scalar | affine fitting support row 수 | `paired_support.fit_support_count` | original+flip 사용 시 S1 `2,000`, S10 `3,800` |
| `N_base` | scalar | base class 수 | protocol `base_classes` | 50 |
| `M` | scalar | session당 새 class 수 | protocol `increment` | 5 |
| `Acc_t` | scalar (%) | session `t`의 all-seen top-1 accuracy | result record `accuracy_percent` | 각 session 평가 시 계산 |
| `AIA` | scalar (%) | `(T+1)^{-1}\sum_{t=0}^T Acc_t`; 본 문서는 S0 포함 | summary `*_aia_percent` | 전체 trajectory 집계 |
| `Acc_T` / Last | scalar (%) | 마지막 session S10 accuracy | summary `*_final_percent` | 전체 trajectory 집계 |
| `\mathcal A_t` | — | 과거 SACIL 문서의 anchor/signature 표기이며 현재 Affine CMPT에는 대응되는 quantity가 없음 | 없음 | `[Verified]` 본 논문 notation에서 제거 권장 |

### 3.3 NPC, NCM, NME, CMPT의 포함 관계

```text
Nearest-prototype classifier (가장 큰 범주)
├─ NCM: true/stored class mean을 prototype으로 사용
│  └─ NME: 저장 exemplar의 산술평균을 mean으로 사용
└─ transported-prototype classifier
   └─ 현재 Affine CMPT: introduction full-data mean을 affine하게 갱신
```

- `[Literature]` iCaRL의 정확한 명칭은 nearest-mean-of-exemplars classification이다.
- `[Verified]` Affine CMPT의 최종 decision rule도 nearest-prototype이지만, old-class prototype이 현재 exemplar mean이 아니므로 엄밀히는 표준 NME가 아니다.
- `[Verified]` 주 표에서 `NME`는 baseline estimator/classifier, `Affine`은 같은 cosine NPC에 CMPT prototype을 넣은 제안 estimator다.

---

## 4. 기존 NME 방식

### 4.1 동작 과정

1. `[Verified]` Session `t` 학습 후 현재 extractor `f_t`를 evaluation mode로 둔다.
2. `[Verified]` 각 seen class `c`의 저장 exemplar 20개를 `f_t`에 다시 통과시킨다.
3. `[Verified]` 각 exemplar의 feature와 horizontal-flip feature를 각각 L2 normalize한 뒤 평균한다.
4. `[Verified]` class별 평균을 다시 L2 normalize해 `\mu^{mem}_{t,c}`를 만든다.
5. `[Verified]` test query는 현재 `f_t`의 normalized feature로 표현한다. 현재 primary CMPT config에서는 query flip TTA를 쓰지 않는다.
6. `[Verified]` cosine similarity가 가장 큰 class를 예측한다.

$$
\mu^{mem}_{t,c}
=\operatorname{norm}\left[
\frac{1}{|\mathcal M_c|}
\sum_{x\in\mathcal M_c}\frac{1}{2}
\left(
\operatorname{norm}(f_t(x))+
\operatorname{norm}(f_t(\operatorname{flip}(x)))
\right)
\right],
$$

$$
\hat y(x)=\arg\max_{c\in\mathcal C_{\le t}}
\operatorname{norm}(f_t(x))^\top\mu^{mem}_{t,c}.
$$

코드 근거: `src_cmpt/sacil/engine/evaluator.py::compute_nme_class_means`, `evaluate_nme`.

### 4.2 어느 feature space의 mean인가

- `[Verified]` NME prototype은 매 session **현재 모델 `f_t`의 feature space**에서 다시 계산한다. 최초 모델의 feature를 고정해 쓰지 않는다.
- `[Verified]` 따라서 저장 prototype의 좌표가 낡는 문제는 줄지만, 20개 sample만으로 전체 분포를 근사하는 sampling/coverage error가 남는다.

### 4.3 20-exemplar mean이 full-data mean과 달라지는 이유

- `[Literature]` Herding은 모든 조합을 exhaustive search하지 않는다. 현재 mean에 가장 가까워지도록 exemplar를 하나씩 추가하는 greedy approximation이다.
- `[Hypothesis]` 20개 sample은 multi-modal class distribution, background, pose, texture 등 전체 within-class variation을 모두 담기 어렵다.
- `[Verified]` 더 중요한 점은 exemplar가 class introduction 당시 feature space에서 herding으로 선택되지만, 이후 `f_t`가 바뀐다는 것이다. 선택 당시의 mean-approximation 성질은 nonlinear representation update 아래에서 보존된다는 보장이 없다.
- `[Verified]` CIFAR-100 7-learner 평균 old-class memory/full prototype cosine distance는 S1 `0.01437`에서 S10 `0.05624`로 전반적으로 커졌다.

### 4.4 Classification error와의 연결

- `[Hypothesis]` Cosine NPC에서 prototype이 이동하면 query와 competing prototype 사이의 margin이 바뀌며, true class prototype의 오차가 충분히 크면 decision boundary를 넘어 misclassification이 발생한다.
- `[Verified]` 70개 incremental learner-session을 합친 분석에서 prototype cosine error와 old-only oracle accuracy headroom의 Pearson `r=0.724 (p=1.46×10^-12)`, Spearman `ρ=0.722 (p=1.77×10^-12)`였다.
- `[Verified]` 그러나 이 관계는 learner별로 동일하지 않다. 따라서 “prototype error가 catastrophic forgetting의 유일한 원인”이 아니라 “측정 가능한 원인 중 하나”라고만 써야 한다.

---

## 5. Oracle Experiment

### 5.1 Oracle이 사용하는 정보

- `[Verified]` Old-only oracle은 session `t`의 current model `f_t`로 모든 old class의 **전체 training images**를 다시 forward해 `\mu^{full}_{t,c}`를 계산한다.
- `[Verified]` 현재-session class는 baseline과 동일한 20-exemplar NME mean을 사용한다.
- `[Verified]` All-seen oracle은 old와 current class 모두 full-training-data mean을 사용한다.
- `[Verified]` Test image와 test label은 prototype 계산에 사용하지 않는다. 그러나 old class의 full training images에 재접근하므로 실제 CIL 방법으로는 허용되지 않는다.

### 5.2 Baseline과 oracle의 차이

| 구분 | Old class prototype | Current-session prototype | CIL에서 사용 가능? |
|---|---|---|---|
| 20-exemplar NME | 현재 모델의 저장 20개 mean | 현재 모델의 저장 20개 mean | 가능 |
| Old-only oracle | 현재 모델의 old full-training-data mean | 현재 모델의 저장 20개 mean | 불가능 |
| All-seen oracle | 현재 모델의 full-training-data mean | 현재 모델의 full-training-data mean | 불가능 |

### 5.3 CIFAR-100 원본 수치

`Old-only oracle headroom = Oracle AIA - NME AIA`이고, `Recovery = (Affine AIA-NME AIA)/(Oracle AIA-NME AIA)`다.

| Learner | NME AIA | Affine AIA | Old-only oracle AIA | Oracle headroom | Affine recovery | NME final | Oracle final | Final headroom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| iCaRL | 52.422 | 52.851 | 54.326 | +1.904 | 22.5% | 44.08 | 46.62 | +2.54 |
| LUCIR-natural | 55.742 | 57.033 | 58.464 | +2.722 | 47.4% | 43.09 | 45.88 | +2.79 |
| FGP-ICL | 56.878 | 57.195 | 59.927 | +3.049 | 10.4% | 45.62 | 50.81 | +5.19 |
| PODNet | 59.614 | 59.824 | 60.064 | +0.449 | 46.6% | 47.98 | 48.18 | +0.20 |
| AFC | 63.586 | 63.609 | 64.454 | +0.868 | 2.7% | 54.74 | 56.21 | +1.47 |
| CSCCT | 57.244 | 57.629 | 59.472 | +2.228 | 17.3% | 44.08 | 47.73 | +3.65 |
| CaSpeR-IL+iCaRL | 54.855 | 55.396 | 56.503 | +1.648 | 32.8% | 44.83 | 47.25 | +2.42 |
| **7-learner mean** | **57.191** | **57.648** | **59.030** | **+1.838** | **24.8%** | **46.346** | **48.954** | **+2.609** |

- `[Verified]` All-seen oracle 평균 AIA는 `59.038%`로 old-only oracle `59.030%`와 `0.008%p`만 다르다. 관찰된 headroom은 거의 전적으로 old-class prototype에서 발생한다.
- `[Verified]` 70개 incremental learner-session 모두에서 old-only oracle이 20-exemplar NME보다 높았다.
- `[Verified]` ImageNet-100 full-data oracle은 수행하지 않았다.

### 5.4 이 실험이 입증하는 것과 입증하지 못하는 것

- `[Verified]` 입증: 현재 CIFAR-100 seed-1 checkpoint들에서 20-exemplar mean과 current full-data mean 사이에 분류 성능으로 이어지는 gap이 있다.
- `[Verified]` 입증: Affine CMPT가 그 평균 AIA gap의 `0.457/1.838=24.8%`, 즉 약 1/4을 회복한다. “약 30%”는 느슨한 반올림이며 정확한 문서 수치는 24.8%다.
- `[Verified]` Final accuracy 기준 평균 회복률은 약 `0.561/2.609=21.5%`다.
- `[Verified]` 미입증: full-data prototype이 Bayes-optimal classifier라는 주장, prototype error가 forgetting의 전부라는 주장, 다른 seed·dataset에서도 같은 oracle gap이 나온다는 주장.
- `[Literature]` LDC도 exemplar-free CIL에서 stale prototype, oracle prototype, corrected prototype을 비교한다. 따라서 “full-data oracle로 prototype drift headroom을 최초 발견했다”는 주장은 불가능하다.

원본 근거: `mds/results/cmpt_full_training_mean_oracle_cifar100.md`, `outputs/cmpt/full_mean_oracle/cifar100/<learner>/seed_1/results.json`.

---

## 6. 제안 방법

### 6.1 발표용 설명

`[Verified]` 새 class를 처음 배울 때에는 그 class의 모든 현재 training image를 사용할 수 있으므로, 이때 계산한 충분한 표본의 prototype을 하나 저장한다. 이후에는 보존된 old exemplar가 이전 모델과 새 모델의 feature space에서 어떻게 함께 이동했는지를 이용해 session별 affine transformation을 구하고, 저장된 old prototype을 현재 space로 갱신한다. `[Hypothesis]` 이렇게 하면 매번 20개 exemplar의 현재 평균을 prototype으로 다시 만드는 것보다 전체 class mean에 가까운 representative를 유지할 수 있다.

### 6.2 입력, 출력, 저장 정보

| 구분 | 내용 | CIL-valid 여부 |
|---|---|---|
| 입력 checkpoint | consecutive models `f_{t-1}`, `f_t` | 가능: 보통 distillation에도 직전 model을 보존 |
| paired support | 동일한 old exemplar image를 두 model에 통과시킨 feature pair | 가능: 이미 허용된 exemplar memory 사용 |
| introduction prototype | class가 처음 등장한 session의 full training data mean | 가능: 그 session의 new-class data는 모두 접근 가능 |
| 출력 | 현재 feature space의 old-class transported prototype bank | 가능 |
| 추가 영구 저장 | class당 `d`차원 float vector 한 개 | 가능하다고 가정하나, memory accounting에 명시해야 함 |
| 사용하지 않는 정보 | old class의 비저장 training image, test image/label | 사용하지 않음 |

- `[Verified]` CIFAR-100 `d=64`이면 100-class bank는 float32 기준 `100×64×4=25,600 bytes`(25 KiB)다.
- `[Verified]` ImageNet-100 `d=512`이면 `204,800 bytes`(200 KiB)다.
- `[Verified]` exemplar 이미지 memory는 baseline과 동일하며 class당 20개다.
- `[Unknown]` 최종 논문의 memory budget이 auxiliary float statistic까지 포함해 동일 byte budget으로 통제할지는 연구자가 확정해야 한다.

코드 근거: `src_cmpt/sacil/cmpt/evaluator.py::CMPTCheckpointEvaluator`, `_full_introduction_prototypes`, `_paired_support_features`, `build_old_class_cmpt_means`.

### 6.3 Session별 전체 과정

#### S0 또는 class가 처음 등장하는 session

`[Verified]` class `c`가 session `s(c)`에 처음 등장하면, 그 session에서 접근 가능한 class `c`의 전체 training data로 다음 값을 계산한다.

$$
\mu^{intro}_c
=
\operatorname{norm}\left(
\frac{1}{|\mathcal D_c|}
\sum_{x\in\mathcal D_c}
\operatorname{norm}(f_{s(c)}(x))
\right).
$$

현재 구현은 original image와 horizontal flip을 모두 prototype 계산에 넣는다. S0의 50개 class와 이후 매 session의 5개 new class에 대해 이 값을 계산하여 transported bank에 append한다. `[Verified]` S0에서는 affine map이 없으므로 NME와 CMPT accuracy가 동일하다.

#### Transition `t-1→t`

1. `[Verified]` 이전 checkpoint의 exemplar memory에서 동일한 old image `x_i`를 가져온다.
2. `[Verified]` `x_i`를 frozen previous extractor와 current extractor에 각각 통과시켜

   $$
   u_i=v_{t-1,i}=\operatorname{norm}(f_{t-1}(x_i)),
   \qquad
   y_i=v_{t,i}=\operatorname{norm}(f_t(x_i))
   $$

   를 얻는다. 동일 image이므로 feature 차이는 model update가 만든 representation change를 직접 관찰한다.
3. `[Verified]` original/flip 각각을 paired regression row로 사용한다. S1에는 `50×20×2=2,000` rows, S10에는 `95×20×2=3,800` rows가 있다.
4. `[Verified]` 모든 old-class pair를 합쳐 **하나의 session-global map**을 구한다.

   $$
   (A_t^*,b_t^*)=
   \arg\min_{A,b}
   \sum_{i=1}^{n_t}
   \|A u_i+b-y_i\|_2^2
   +\lambda_{aff}\|A\|_F^2.
   $$

5. `[Verified]` 기존 bank의 모든 old-class prototype에 같은 map을 적용하고 L2 normalize한다.

   $$
   \tilde\mu_{t,c}
   =\operatorname{norm}(A_t^*\tilde\mu_{t-1,c}+b_t^*),
   \qquad c\in\mathcal C_{<t}.
   $$

6. `[Verified]` current new classes에는 이 map을 적용하지 않는다. 새 class의 `\mu_c^{intro}`를 현재 space에서 직접 계산해 append한다.
7. `[Verified]` 이 갱신은 session마다 반복되므로 old prototype은 여러 affine map을 순차적으로 거친다. 따라서 fitting error가 누적될 가능성이 있다.

### 6.4 Closed-form solution과 실제 구현

열벡터 convention에서 `U_c=[u_1-\bar u,\ldots,u_n-\bar u]`, `Y_c=[y_1-\bar y,\ldots,y_n-\bar y]`라 두면

$$
A_t^*=Y_cU_c^\top(U_cU_c^\top+\lambda_{aff}I)^{-1},
\qquad
b_t^*=\bar y-A_t^*\bar u.
$$

- `[Verified]` 실제 함수 `src_cmpt/sacil/methods/prototype_transport.py::affine_ridge_transport`는 평균을 명시적으로 빼지 않고 row-vector augmented design `X=[U,\mathbf 1]`을 만든다.
- `[Verified]` 코드는 `W=(X^TX+R)^{-1}X^TY`를 `torch.linalg.solve`로 계산하며, `R`의 마지막 intercept 대각 원소를 0으로 만들어 `A`만 regularize하고 `b`는 regularize하지 않는다.
- `[Verified]` 따라서 명시적 centering 구현은 아니지만, unregularized intercept가 있는 ridge regression이므로 위 centered 해와 수학적으로 동등하다. 코드의 upper block은 표기한 `A_t^T`, 마지막 row는 `b_t^T`에 대응한다.
- `[Verified]` 역행렬을 실제로 만들지 않고 linear solve를 쓰는 것은 같은 해를 더 안정적으로 구하는 구현 방식이다.

### 6.5 Inference

`[Verified]` query `x`에 대해 current model만 사용한다.

$$
z=\operatorname{norm}(f_t(x)),
\qquad
\hat y=arg\max_{c\in\mathcal C_{\le t}}z^T p_{t,c},
$$

여기서 `p_{t,c}=\tilde\mu_{t,c}` for old class, `p_{t,c}=\mu^{mem}_{t,c}` for current-session class다. 현재 primary config는 prototype/support flip은 사용하고 query flip TTA는 사용하지 않는다.

### 6.6 Baseline과 제안 방법 pseudocode

```text
Baseline NME at session t
  for every seen class c:
      p[c] = normalized mean of 20 stored exemplar features under f_t
  for each query x:
      predict argmax_c cosine(f_t(x), p[c])

Affine prototype transformation at session t
  if t == 0:
      bank[c] = full-data mean under f_0 for every base class c
  else:
      U = features of the same old exemplars under f_(t-1)
      Y = features of those exemplars under f_t
      (A_t, b_t) = ridge_affine_fit(U, Y)
      for c in old classes:
          bank[c] = normalize(A_t @ bank[c] + b_t)
      for c in current new classes:
          bank[c] = full-data mean under f_t

  evaluation_prototypes = baseline NME means
  replace only their old-class rows with bank[old classes]
  predict argmax_c cosine(f_t(x), evaluation_prototypes[c])
```

`[Verified]` 현재 evaluator가 checkpoint에서 introduction prototype을 재구성하는 것은 분석 편의를 위한 offline 실행 형식이다. 논문 알고리즘에는 해당 class가 도착한 시점에 계산·저장하는 online 형식으로 써야 한다.

---

## 7. Affine Transformation을 사용하는 근거

### 7.1 결론부터: 보장이 아니라 검증 가능한 modeling assumption

**발표용 설명:** `[Hypothesis]` model update가 feature space를 완전히 임의적으로 바꾸기보다, 많은 old samples에 공통된 회전·축별 크기 변화·shear·평행이동 성분을 포함한다고 보고 그 공통 변화를 가장 단순하게 표현할 수 있는 affine map을 사용한다. `[Verified]` 이는 정리로 보장된 사실이 아니라, paired exemplars로 직접 fitting하고 downstream accuracy로 검증한 근사 가정이다.

**기술적 설명:**

- `[Literature]` SDC(CVPR 2020), LDC(ECCV 2024), ADC(CVPR 2024) 등은 old representation/prototype을 current space로 보정하는 문제를 다룬다. LDC는 current-task data의 old/new feature pair로 forward projector를 학습하고 old prototype을 update한다.
- `[Literature]` DGASA(PLOS ONE 2026)는 exemplar-free adapter space에서 new-class prototype pair로 regularized global linear mapping을 추정한다.
- `[Verified]` 본 연구는 이 계열의 아이디어를 참고하지만, retained **old exemplars의 동일-image correspondence**를 사용하고 affine intercept를 포함한 closed-form map을 learner-independent post-hoc classifier에 사용한다.
- `[Hypothesis]` Deep representation drift가 정확히 affine이라는 이론적 보장은 없다. 논문에는 “we approximate” 또는 “we model the shared component of drift”라고 써야 한다.

### 7.2 Transformation family 비교 상태

| Family | 표현력 | 현재 검증 상태 | 해석 |
|---|---|---|---|
| global translation `z+b` | 모든 point가 같은 방향·크기로 이동 | `[Planned]` 순수 global `b`-only 미수행 | 가장 단순하지만 anisotropic drift를 놓침 |
| diagonal scaling + translation | dimension별 scale과 shift | `[Planned]` 미수행 | full affine보다 parameter가 적음 |
| orthogonal/rigid `Rz+b` | 거리·각도를 보존하는 rotation/reflection + shift | `[Verified]` CIFAR topology learner 3개에서 비교 | LUCIR의 과거 50:50 run에서는 affine보다 좋았지만 FGP/CaSpeR에서는 affine이 좋음 |
| full affine `Az+b` | rotation, anisotropic scaling, shear, translation | `[Verified]` primary 14 trajectory | 모든 trajectory의 AIA를 NME보다 개선했으나 gain 크기는 learner별로 다름 |
| nonlinear/quadratic | location-dependent drift 표현 | `[Verified]` moment-aware 탐색 수행 | NME보다 좋았지만 affine을 일관되게 넘지 못함 |

- `[Verified]` rigid-vs-affine CIFAR 결과는 LUCIR `59.427 vs 58.771`, FGP `56.925 vs 57.195`, CaSpeR `54.803 vs 55.396` AIA였다. 단, 이 LUCIR는 primary natural-loader가 아니라 과거 50:50 loader checkpoint이므로 주 표와 직접 결합하면 안 된다.
- `[Verified]` full affine을 primary로 택한 실험적 이유는 두 데이터셋×7 learner에서 동일 estimator로 사용할 수 있고 14/14 AIA가 양수였기 때문이다. 이는 “full affine이 모든 alternative보다 최적”임을 입증하지 않는다.

### 7.3 Ridge regularization이 필요한 이유

- `[Literature]` Ridge regression은 correlated predictor와 ill-conditioned normal equation에서 coefficient variance를 줄이는 고전적 regularization이다.
- `[Verified]` normalized deep features의 dimensions와 original/flip rows는 강하게 상관될 수 있다. `U^TU`가 singular 또는 ill-conditioned하면 unregularized full affine의 해가 불안정해질 수 있다.
- `[Verified]` 현재 `\lambda_{aff}=0.01`은 `A`에만 적용되고 `b`에는 적용되지 않는다.
- `[Verified]` S1부터 row 수는 CIFAR `2,000 > d+1=65`, ImageNet `2,000 > d+1=513`이므로 단순 row-count 기준으로 underdetermined는 아니다. 그러나 original/flip correlation, within-class redundancy와 effective rank 때문에 ridge는 여전히 필요하다.
- `[Unknown]` condition number, effective rank, train/held-out fitting residual을 주 결과 전 learner/session에서 체계적으로 보고한 표는 없다.
- `[Planned]` `\lambda=0`, `0.001`, `0.01`, `0.1` 등의 sensitivity가 없으므로 `0.01`의 최적성은 주장할 수 없다.

### 7.4 Affine model이 놓치는 것

- `[Hypothesis]` class-dependent, region-dependent, piecewise 또는 highly nonlinear drift는 하나의 global `A_t,b_t`로 설명할 수 없다.
- `[Verified]` feature-neighbor 5-class local affine은 global affine보다 7-learner 평균 AIA가 `-0.564%p` 낮았고 70 transition 중 1개에서만 global을 이겼다. 단순 visual proximity grouping은 해결책이 아니었다.
- `[Verified]` global affine에 class-specific memory translation residual을 더한 combined variant도 global affine보다 평균 `-0.422%p` 낮았다.
- `[Verified]` PCA/degree-2 moment-aware variants는 일부 learner에서 추가 개선됐지만 전체적으로 affine보다 일관되게 우수하지 않았다.
- `[Hypothesis]` 이는 nonlinear drift가 없다는 뜻이 아니라, 제한된 exemplar로 더 복잡한 map을 안정적으로 추정하지 못했을 가능성도 포함한다.

### 7.5 표현상 주의

- `linear transform`은 `Az`이며 원점을 보존한다. `affine transform`은 `Az+b`이며 translation을 포함한다. 두 표현을 같은 뜻으로 쓰면 안 된다.
- `A`는 transformation **matrix**, `b`는 translation **vector**다.
- `transport`는 prototype을 한 representation space에서 다음 space로 옮기는 동작을 설명하는 일반적 표현으로 사용할 수 있으나, 고유한 수학적 보장을 뜻하지 않는다.
- 권장 문장: `[Hypothesis]` “We use a ridge-regularized affine map as a computationally tractable approximation to the shared component of inter-session representation drift.”
- 피할 문장: “Feature drift is affine,” “the closed form exactly recovers the current full-data prototype,” “full affine is theoretically optimal.”

### 7.6 관련 문헌

1. `[Literature]` S.-A. Rebuffi et al., “iCaRL: Incremental Classifier and Representation Learning,” CVPR, 2017. Official page: <https://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html>
2. `[Literature]` L. Yu et al., “Semantic Drift Compensation for Class-Incremental Learning,” CVPR, 2020. Official page: <https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_Semantic_Drift_Compensation_for_Class-Incremental_Learning_CVPR_2020_paper.html>
3. `[Literature]` A. Gomez-Villa et al., “Exemplar-free Continual Representation Learning via Learnable Drift Compensation,” ECCV, 2024. Official page: <https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1192_ECCV_2024_paper.php>
4. `[Literature]` D. Goswami et al., “Resurrecting Old Classes with New Data for Exemplar-Free Continual Learning,” CVPR, 2024. Official page: <https://openaccess.thecvf.com/content/CVPR2024/html/Goswami_Resurrecting_Old_Classes_with_New_Data_for_Exemplar-Free_Continual_Learning_CVPR_2024_paper.html>
5. `[Literature]` “Dual geometric alignment via subspace adaptation for class-incremental learning,” PLOS ONE, 2026 (DGASA). Official article: <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0348270>
6. `[Literature]` A. E. Hoerl and R. W. Kennard, “Ridge Regression: Biased Estimation for Nonorthogonal Problems,” *Technometrics*, vol. 12, no. 1, pp. 55–67, 1970. DOI: <https://doi.org/10.1080/00401706.1970.10488634>

---

## 8. 기존 연구와의 차이

### 8.1 Exemplar-free drift compensation과의 비교

| Method | Setting / available correspondence | Prototype update | 본 연구와의 핵심 차이 |
|---|---|---|---|
| SDC (CVPR 2020) | exemplar-free; current-task samples로 local semantic drift 추정 | old prototype 보정 | 본 연구는 retained old samples의 실제 old/new pair로 global affine을 fitting |
| LDC (ECCV 2024) | exemplar-free; current new data를 frozen old/new model에 통과 | learned forward linear projector로 old prototype 재귀 update | 본 연구는 old exemplar correspondence, closed-form ridge affine, introduction full-data prototype을 사용 |
| ADC (CVPR 2024) | exemplar-free; new image를 old prototype 쪽으로 adversarial perturb | synthetic old-oriented samples로 drift 추정 | 본 연구는 실제 old exemplar를 이용해 synthesis 없이 fitting |
| DGASA (PLOS ONE 2026) | exemplar-free, pretrained ViT adapters; new-class old/new subspace prototype pair | global regularized linear mapping | 본 연구는 CNN rehearsal learners의 old exemplar sample pairs와 affine intercept를 사용 |
| **본 연구** | exemplar-based; class당 20 old images + class introduction 시점 full data | session-global `A_t,b_t`로 stored full-data prototype update | 7개 heterogeneous rehearsal learner에 training 변경 없이 동일 post-hoc estimator 적용 |

### 8.2 공통점과 달라지는 점

- `[Literature]` 공통점: representation space가 바뀌면 과거 prototype이 stale해진다는 관점, old/current space 사이 correspondence로 prototype을 update한다는 큰 틀은 기존 exemplar-free 연구와 공유한다.
- `[Verified]` 정보 차이: exemplar-free 방법은 old image를 저장할 수 없어 current new samples를 proxy correspondence로 쓴다. 본 연구는 실제 old-class exemplar `x_i`가 있으므로 동일 old image가 어떻게 이동했는지 직접 관찰한다.
- `[Verified]` 목표 차이: 현재 NME도 current exemplar features로 prototype을 다시 만들 수 있다. 본 연구의 목표는 단순히 stale coordinate를 최신화하는 것만이 아니라, class introduction 당시의 full-data mean에 담긴 broader class information을 유지하면서 current space로 갱신하는 것이다.
- `[Verified]` 추가 정보: baseline NME보다 class당 prototype vector 한 개를 더 저장한다. 별도의 trainable projector, generator, test-time optimization은 없다.

### 8.3 방어 가능한 contribution과 novelty 위험

**현재 결과로 방어 가능한 contribution**

1. `[Verified]` 7개 exemplar-based learner에서 current 20-exemplar NME와 current full-training-data prototype 사이의 accuracy headroom을 동일 checkpoint로 정량화했다.
2. `[Verified]` retained old exemplars를 consecutive feature spaces의 direct sample-level landmarks로 활용해, introduction-time full-data prototype을 갱신하는 closed-form ridge-affine post-hoc estimator를 구현했다.
3. `[Verified]` CIFAR-100/ResNet-32와 ImageNet-100/ResNet-18의 총 14 paired trajectories에서 AIA positive gain을 관찰했다.

**주장하면 안 되는 내용**

- `[Literature]` “최초로 prototype drift를 보정했다,” “최초로 linear/affine mapping으로 old prototype을 이동했다”는 선행연구 때문에 방어하기 어렵다.
- `[Unknown]` `b`를 포함한 것만으로 novelty라고 할 수 있는지는 문헌 전체를 exhaustively audit하지 않았고, 일반 linear layer는 bias를 포함할 수 있으므로 주장하지 않는다.
- `[Hypothesis]` novelty는 `affine` 한 단어보다 **exemplar-based NME에서 full-data information preservation이라는 문제 정의 + actual old-pair fitting + broad plug-in evaluation**의 결합으로 서술하는 편이 안전하다.
- `[Planned]` LDC/SDC식 estimator를 동일 checkpoint에 직접 adapter한 비교가 없으므로, 기존 drift compensation보다 우월하다는 주장은 아직 할 수 없다.

---

## 9. 실험 설정

### 9.1 공통 protocol

| 항목 | CIFAR-100 | ImageNet-100 |
|---|---|---|
| `[Verified]` CIL protocol | B50-Inc5, S0–S10 | B50-Inc5, S0–S10 |
| `[Verified]` class 수 | 100 | 100-class subset |
| `[Verified]` train/test | 50,000 / 10,000; class당 500 / 100 | train 129,395; test 5,000, class당 test 50 |
| `[Verified]` train class 크기 | class당 500 | class당 1,071–1,300 |
| `[Verified]` backbone | CIFAR ResNet-32, feature `d=64` | ImageNet ResNet-18, feature `d=512` |
| `[Verified]` memory | class당 20, final 2,000 images | class당 20, final 2,000 images |
| `[Verified]` exemplar selection | iCaRL-style greedy herding | iCaRL-style greedy herding |
| `[Verified]` training loader | current full data + memory의 natural shuffle | current full data + memory의 natural shuffle |
| `[Verified]` run seed | 1 | 1 |
| `[Verified]` CMPT ridge | `0.01` | `0.01` |

- `[Verified]` CIFAR class order는 PODNet/AFC에서 사용한 published order이며 원래 iCaRL order와 연결된다. 파일: `experiment_configs/class_orders/cifar100_b50_t10_afc_order1.json`.
- `[Verified]` ImageNet-100 class order는 AFC/PODNet/R-DFCIL 계열 order이며 NumPy seed 1993 permutation을 저장한 것이다. 파일: `experiment_configs/class_orders/imagenet100_b50_inc5_afc_order1.json`.
- `[Verified]` run seed 1은 initialization/data-loader randomness이고 class-order seed와 다른 개념이다.
- `[Verified]` validation split은 두지 않았고 `\lambda_{aff}=0.01`은 모든 primary run에 고정했다.
- `[Unknown]` `0.01`이 어떤 독립 validation protocol로 선택됐는지는 저장소에 문서화되어 있지 않다. 따라서 “validation으로 최적화했다”고 쓰면 안 된다.

### 9.2 Data augmentation

| Dataset | Train augmentation | Evaluation transform |
|---|---|---|
| CIFAR-100 | `RandomCrop(32,padding=4)`, random horizontal flip, brightness color jitter `63/255`, tensor conversion, dataset normalization | tensor conversion + normalization |
| ImageNet-100 | `RandomResizedCrop(224)`, random horizontal flip, ImageNet normalization | resize 256, center crop 224, normalization |

`[Verified]` CIFAR 근거는 `src_cmpt/sacil/data/cifar100.py`, ImageNet 근거는 `src_cmpt/sacil/data/imagenet100.py` 및 `configs/cmpt/imagenet100_b50_inc5/_training_common.yaml`이다. CMPT prototype/support는 horizontal flip features를 함께 쓰지만 test query에는 flip TTA를 쓰지 않는다.

### 9.3 비교한 7개 CIL training methods

| Learner | Publication | 현재 구현의 주요 training objective | Primary NPC 평가와의 관계 |
|---|---|---|---|
| iCaRL | CVPR 2017 | classification + old-output distillation, rehearsal | native inference가 NME |
| LUCIR | CVPR 2019 | cosine classifier, less-forget feature constraint, margin ranking | native cosine head와 별도로 NME/Affine 평가 |
| FGP-ICL | ACCV 2020 | classification + feature-graph preservation | native learned head와 별도로 평가 |
| PODNet | ECCV 2020 | NCA/multi-proxy classification + spatial/flat POD | native cosine/multi-proxy head와 별도로 평가 |
| AFC | CVPR 2022 | NCA/multi-proxy + adaptive feature consolidation | native multi-proxy head와 별도로 평가 |
| CSCCT | ECCV 2022 | classification/distillation + CSC/CT terms | native split-cosine head와 별도로 평가 |
| CaSpeR-IL+iCaRL | PRL 2024 | iCaRL-style loss + spectral topology regularizer | 현재 결합에서는 native inference가 NME |

- `[Verified]` CMPT는 이 learner들의 training loss를 바꾸지 않는다. 학습 완료 checkpoint에 baseline NME와 proposed prototype estimator를 paired evaluation했다.
- `[Verified]` ImageNet-100에서 LUCIR, PODNet, AFC, CSCCT는 native head 결과도 저장되어 있다. 따라서 “CMPT가 각 learner의 native classifier를 모두 이겼다”는 주장은 사실이 아니다.

### 9.4 Optimizer와 method별 training recipe

`[Verified]` 모든 주 learner는 SGD momentum `0.9`를 사용하며 Nesterov는 사용하지 않는다. 나머지는 learner-native/adapted recipe라 완전히 통일되어 있지 않다.

#### CIFAR-100

| Learner | Batch | Base / Inc epochs | LR | WD | Schedule / 추가 단계 |
|---|---:|---:|---:|---:|---|
| iCaRL | 128 | 200 / 170 | 0.1 | `5e-4` / `2e-4` | multi-step base 60/120/170, inc 80/120, ×0.1 |
| LUCIR-natural | 128 | 200 / 80 | 0.1 | `5e-4` | multi-step base 60/120/170, inc 40/70, ×0.1 |
| FGP-ICL | 128 | 200 / 80 | 2.0 | `1e-5` | base 60/120/170, inc 50/64, ×0.2 |
| PODNet | 128 | 160 / 160 | 0.1 | `5e-4` | cosine; 20-epoch balanced finetune, LR .005 |
| AFC | 128 | 160 / 160 | 0.1 | `5e-4` | cosine; 20-epoch balanced finetune, LR .05 |
| CSCCT | 128 | 160 / 160 | 0.1 | `5e-4` | multi-step 80/120; fusion LR `1e-8` |
| CaSpeR-IL+iCaRL | 128 | 200 / 80 | 0.3 | optimizer WD 0; explicit regularizer `1e-5` | base 60/120/170, inc 50/64, ×0.2 |

#### ImageNet-100

| Learner | Batch | Base / Inc epochs | LR | WD | Schedule / 추가 단계 |
|---|---:|---:|---:|---:|---|
| iCaRL | 64 | 60 / 60 | 1.0 | `1e-5` | milestones 20/30/40/50, ×0.2 |
| LUCIR | 128 | 90 / 90 | 0.1 | `1e-4` | milestones 30/60, ×0.1 |
| FGP-ICL | 128 | 60 / 60 | 2.0 | `1e-5` | milestones 20/30/40/50, ×0.2 |
| PODNet | 64 | 90 / 90 | 0.05 | `5e-4` | cosine; finetune 20 epochs, LR .01 |
| AFC | 128 | 90 / 90 | 0.1 | `1e-4` | cosine; finetune 20 epochs, LR .02 |
| CSCCT | 128 | 160 / 160 | 0.1 | `5e-4` | cosine; fusion milestones 53/106 |
| CaSpeR-IL+iCaRL | 128 | 60 / 60 | 0.3 | optimizer WD 0; explicit `1e-5` | milestones 20/30/40/50, ×0.2 |

- `[Verified]` config 근거: `configs/cmpt/common_recipe/train_*.yaml`, `configs/cmpt/imagenet100_b50_inc5/train_*.yaml`, 상속되는 공통 config와 각 output의 `resolved_config.json`.
- `[Verified]` ImageNet AFC 주 결과는 저자 physical batch 128 trajectory다. 8 GiB GPU에서 OOM이 발생해 48 GiB Linux GPU에서 완료했다.
- `[Literature]` method별 원 논문 recipe를 가능한 범위에서 반영했지만 dataset/protocol/memory/class order는 paired comparison을 위해 공통화했다.
- `[Unknown]` 모든 세부 구현이 원 저자 repository와 bitwise parity라는 보장은 없다. 이 문서는 현재 repository가 실제 실행한 recipe를 보고한다.
- `[Literature]` CaSpeR-IL은 author-provided ImageNet-100 B50-Inc5 recipe가 없어 현재 ImageNet 설정은 controlled adaptation이다.

### 9.5 Classifier와 metric

- `[Verified]` 주 baseline classifier는 normalized features와 memory means의 cosine nearest-prototype rule인 NME다.
- `[Verified]` 제안 classifier는 동일 query feature/similarity/test set에서 old rows만 transported prototypes로 교체한다.
- `[Verified]` `Acc_t`는 session `t`에서 S0–St seen classes 전체에 대한 top-1 accuracy다.
- `[Verified]` AIA는

  $$
  \mathrm{AIA}=\frac{1}{T+1}\sum_{t=0}^{T}\mathrm{Acc}_t
  $$

  로 S0를 포함한 11개 accuracy 평균이다. Last는 `Acc_T`다.
- `[Verified]` 결과는 **seed 1 한 번**이며 mean±standard deviation, confidence interval 또는 statistical significance test가 없다.
- `[Verified]` 발표자료의 “3.37%p average / 5.20%p last oracle gap”은 FGP-ICL all-seen oracle에서 **S1–S10만 평균한 incremental AIA** `+3.374%p`와 final `+5.20%p`다. 본 문서의 primary AIA(S0 포함) gap은 `+3.082%p`; old-only oracle gap은 `+3.049%p`다.

### 9.6 계산량과 memory overhead

- `[Verified]` 영구 overhead는 prototype bank `O(Cd)`이며 100 classes에서 CIFAR 25 KiB, ImageNet 200 KiB다.
- `[Verified]` session fitting의 map/design solution은 대략 `O(n_td^2+d^3)`이고 prototype update는 `O(Cd^2)`다. paired features를 만들기 위해 old memory를 previous/current model로 한 번씩 forward한다.
- `[Verified]` S10 paired old/current feature tensor 자체는 float32 기준 CIFAR 약 1.86 MiB, ImageNet 약 14.84 MiB이고 일시적이다. affine parameter block은 CIFAR 약 16.25 KiB, ImageNet 약 1.0 MiB다.
- `[Planned]` wall-clock, peak GPU/CPU memory, FLOPs를 baseline NME와 같은 hardware에서 직접 profiling한 결과는 없다.

---

## 10. 전체 결과

### 10.1 Primary paired results

| Dataset | Training method | Baseline NME AIA | Proposed Affine AIA | Δ AIA | NME Last | Affine Last | Δ Last |
|---|---|---:|---:|---:|---:|---:|---:|
| CIFAR-100 | iCaRL | 52.422 | 52.851 | +0.429 | 44.08 | 44.24 | +0.16 |
| CIFAR-100 | LUCIR-natural | 55.742 | 57.033 | **+1.291** | 43.09 | 44.47 | **+1.38** |
| CIFAR-100 | FGP-ICL | 56.878 | 57.195 | +0.317 | 45.62 | 46.52 | +0.90 |
| CIFAR-100 | PODNet | 59.614 | 59.824 | +0.209 | 47.98 | 48.30 | +0.32 |
| CIFAR-100 | AFC | 63.586 | 63.609 | +0.024 | 54.74 | 54.70 | **-0.04** |
| CIFAR-100 | CSCCT | 57.244 | 57.629 | +0.385 | 44.08 | 44.69 | +0.61 |
| CIFAR-100 | CaSpeR-IL+iCaRL | 54.855 | 55.396 | +0.541 | 44.83 | 45.43 | +0.60 |
| ImageNet-100 | iCaRL | 52.732 | 53.625 | +0.893 | 43.12 | 43.96 | +0.84 |
| ImageNet-100 | LUCIR | 66.275 | 66.721 | +0.447 | 54.86 | 56.00 | +1.14 |
| ImageNet-100 | FGP-ICL | 65.303 | 65.512 | +0.208 | 54.56 | 55.08 | +0.52 |
| ImageNet-100 | PODNet | 70.021 | 70.102 | +0.082 | 59.38 | 59.46 | +0.08 |
| ImageNet-100 | AFC | 74.125 | 74.447 | +0.322 | 66.50 | 66.54 | +0.04 |
| ImageNet-100 | CSCCT | 61.105 | 61.339 | +0.234 | 49.50 | 49.56 | +0.06 |
| ImageNet-100 | CaSpeR-IL+iCaRL | 64.742 | 65.077 | +0.335 | 55.16 | 56.22 | +1.06 |

- `[Verified]` CIFAR 평균: NME `57.191`, Affine `57.648`, `ΔAIA=+0.457%p`; final 평균 `46.346→46.907`, `+0.561%p`.
- `[Verified]` ImageNet 평균: NME `64.900`, Affine `65.260`, `ΔAIA=+0.360%p`; final 평균 `54.726→55.260`, `+0.534%p`.
- `[Verified]` 14개 전체 macro mean: `ΔAIA=+0.408%p`, `ΔLast=+0.548%p`.
- `[Verified]` AIA는 14/14에서 양수다. Last는 13/14에서 양수이고 CIFAR AFC가 `-0.04%p`다.
- `[Verified]` “최대 AIA `+1.29%p`, 최대 Last `+1.38%p`”는 두 값 모두 CIFAR LUCIR-natural을 가리키며 원본과 일치한다. 전체 평균이나 multi-seed 결과처럼 표현하면 안 된다.

### 10.2 CIFAR-100 session trajectory: `NME→Affine`

| Learner | S0 | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| iCaRL | 76.24→76.24 | 56.64→57.18 | 55.97→56.73 | 52.58→53.68 | 51.83→52.01 | 50.68→51.80 | 49.44→49.50 | 47.08→47.26 | 46.80→47.20 | 45.31→45.52 | 44.08→44.24 |
| LUCIR | 76.24→76.24 | 68.87→69.62 | 64.30→66.12 | 59.69→61.69 | 56.50→58.09 | 53.52→55.28 | 51.34→52.94 | 47.76→48.82 | 46.99→47.91 | 44.85→46.19 | 43.09→44.47 |
| FGP | 74.28→74.28 | 66.91→66.80 | 64.42→64.50 | 60.77→60.88 | 58.03→58.26 | 55.05→55.47 | 53.36→53.58 | 50.86→51.13 | 49.20→49.80 | 47.16→47.94 | 45.62→46.52 |
| PODNet | 76.76→76.76 | 71.36→71.44 | 68.05→68.28 | 64.05→64.22 | 59.94→60.23 | 58.09→58.07 | 55.58→55.74 | 52.76→53.28 | 51.44→51.63 | 49.74→50.12 | 47.98→48.30 |
| AFC | 76.66→76.66 | 72.04→71.78 | 69.23→69.17 | 65.26→65.11 | 63.50→63.84 | 62.56→62.63 | 61.34→61.11 | 59.44→59.75 | 58.07→58.23 | 56.61→56.72 | 54.74→54.70 |
| CSCCT | 76.10→76.10 | 73.51→73.80 | 68.08→68.05 | 61.31→61.63 | 57.63→57.94 | 53.96→54.61 | 51.01→51.63 | 49.01→49.49 | 48.82→49.40 | 46.17→46.57 | 44.08→44.69 |
| CaSpeR | 70.60→70.60 | 64.27→64.51 | 61.33→61.87 | 57.60→57.94 | 54.74→55.36 | 52.61→53.59 | 51.69→52.68 | 49.64→49.96 | 48.87→49.51 | 47.22→47.92 | 44.83→45.43 |

### 10.3 ImageNet-100 session trajectory: `NME→Affine`

| Learner | S0 | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| iCaRL | 71.00→71.00 | 62.98→63.53 | 56.47→57.97 | 53.97→54.77 | 51.91→53.51 | 52.45→53.17 | 49.45→50.40 | 48.07→49.36 | 45.91→46.93 | 44.72→45.26 | 43.12→43.96 |
| LUCIR | 82.28→82.28 | 77.93→78.07 | 73.40→73.30 | 69.54→69.54 | 67.09→67.63 | 65.65→66.21 | 63.05→63.50 | 60.31→60.96 | 58.27→59.04 | 56.65→57.39 | 54.86→56.00 |
| FGP | 82.08→82.08 | 76.55→76.36 | 71.27→71.67 | 67.54→67.82 | 65.71→66.14 | 64.77→64.88 | 62.95→63.10 | 59.98→60.05 | 56.91→57.18 | 56.02→56.27 | 54.56→55.08 |
| PODNet | 84.48→84.48 | 81.13→81.02 | 76.67→76.83 | 73.02→73.29 | 70.86→71.17 | 69.39→69.55 | 67.13→66.98 | 64.45→64.64 | 62.67→62.49 | 61.07→61.22 | 59.38→59.46 |
| AFC | 85.28→85.28 | 82.62→82.44 | 78.50→78.87 | 75.14→75.85 | 74.31→74.60 | 74.59→75.12 | 72.88→73.05 | 69.95→70.21 | 68.58→69.20 | 67.03→67.77 | 66.50→66.54 |
| CSCCT | 84.12→84.12 | 82.80→82.87 | 69.37→69.43 | 60.55→61.02 | 58.11→58.54 | 57.79→58.27 | 56.05→56.08 | 52.92→53.25 | 51.93→52.27 | 49.01→49.33 | 49.50→49.56 |
| CaSpeR | 77.32→77.32 | 73.75→73.89 | 69.47→69.50 | 67.32→67.42 | 65.31→65.86 | 64.85→65.33 | 63.60→63.78 | 60.24→60.75 | 58.11→58.69 | 57.03→57.09 | 55.16→56.22 |

### 10.4 Native classifier를 함께 볼 때

- `[Verified]` ImageNet LUCIR native cosine head: AIA `67.529`, final `57.62`; Affine NPC `66.721/56.00`.
- `[Verified]` ImageNet FGP-ICL native learned head: `55.406/44.58`; NME `65.303/54.56`, Affine NPC `65.512/55.08`.
- `[Verified]` ImageNet PODNet native head: `70.330/59.90`; Affine NPC `70.102/59.46`.
- `[Verified]` ImageNet AFC native multi-proxy head: `75.285/67.44`; Affine NPC `74.447/66.54`.
- `[Verified]` ImageNet CSCCT native head: `56.279/45.68`; Affine NPC `61.339/49.56`.
- `[Verified]` iCaRL과 현재 CaSpeR+iCaRL의 native inference는 NME다.
- `[Unknown]` CIFAR에서 모든 learner의 native-head 결과를 같은 형식으로 정리한 완전한 주 표는 없다.

Primary source: `mds/results/cmpt_nme_vs_affine_results_current.md`와 그 문서에 열거된 14개 evaluation JSON.

---

## 11. 추가 분석 및 Ablation

### 11.1 수행 여부 요약

| 분석 | 상태 | 핵심 결과 |
|---|---|---|
| NME vs global affine | `[Verified]` 완료, 2 datasets×7 learners | AIA 14/14 positive; macro `+0.408%p` |
| full-data oracle | `[Verified]` CIFAR 7 learners 완료 | mean headroom `+1.838%p`, affine recovery `24.8%` |
| rigid vs affine | `[Verified]` CIFAR 3 topology learners | learner-dependent; affine가 항상 우월하지 않음 |
| pure global translation `b` only | `[Planned]` 미수행 | 결과 없음 |
| diagonal affine | `[Planned]` 미수행 | 결과 없음 |
| affine에서 `b` 제거 | `[Planned]` 미수행 | 결과 없음 |
| ridge 제거 / `λ_aff` sweep | `[Planned]` 미수행 | `0.01`만 primary 사용 |
| exemplar 수 변화 | `[Planned]` 미수행 | 20/class만 사용 |
| class-wise translation compensation | `[Verified]` CIFAR 완료 | 평균 NME 대비 `+0.028%p`, global affine보다 낮음 |
| global affine + class residual | `[Verified]` CIFAR 완료 | 평균 NME 대비 `+0.035%p`, global affine보다 `-0.422%p` |
| nearest-class K=5 local affine | `[Verified]` CIFAR 완료 | global보다 평균 `-0.564%p` |
| feature proximity vs drift correlation | `[Verified]` CIFAR 완료 | raw drift에는 관계, global residual에는 거의 없음 |
| herding trajectory extrapolation | `[Verified]` CIFAR 완료 | 평균 NME 대비 `-0.168%p` |
| second/moment-aware nonlinear correction | `[Verified]` CIFAR+ImageNet 완료 | NME는 14/14 개선, affine을 일관되게 넘지 못함 |
| mean/moment-weighted affine fitting | `[Verified]` CIFAR 완료 | uniform affine보다 모두 낮음 |
| adaptive alpha blend | `[Verified]` exploratory 완료 | 추가 복잡성을 정당화할 일관된 gain 없음; 폐기 결정 |
| class별 accuracy 종합 | `[Planned]` 미수행 | 일부 diagnostic 외 paper-ready 표 없음 |
| prototype prediction error | `[Verified]` oracle cosine-distance 분석 일부 완료 | full-data gap과 pooled 상관 확인; 두 dataset 전부는 아님 |
| compute/memory profiling | `[Planned]` 미수행 | 이론적 크기만 계산 |

### 11.2 Class residual ablation

| CIFAR learner | NME | Global affine | Class translation only | Global affine + class residual |
|---|---:|---:|---:|---:|
| iCaRL | 52.422 | 52.851 | 52.473 | 52.476 |
| LUCIR | 55.742 | 57.033 | 55.698 | 55.707 |
| FGP | 56.878 | 57.195 | 56.876 | 56.886 |
| PODNet | 59.614 | 59.824 | 59.649 | 59.635 |
| AFC | 63.586 | 63.609 | 63.642 | 63.656 |
| CSCCT | 57.244 | 57.629 | 57.245 | 57.264 |
| CaSpeR | 54.855 | 55.396 | 54.954 | 54.958 |

`[Verified]` class-specific translation과 global map이 상보적일 것이라는 가설은 이 단순 residual 결합으로 지지되지 않았다. 근거: `mds/results/cmpt_global_class_residual_ablation.md`.

### 11.3 Local grouping과 drift structure

- `[Verified]` feature proximity와 raw class drift similarity는 session 평균 Spearman `ρ≈+0.358`이었고 70개 중 64개가 유의했다.
- `[Verified]` 그러나 global affine이 raw drift energy의 평균 `88.4%`를 설명한 뒤 residual drift의 proximity correlation은 평균 `ρ≈-0.101`, cross-fit `-0.078`이었다.
- `[Verified]` K=5 neighbor affine의 평균 AIA는 `57.084`, global affine은 `57.648`이었다. Local은 70 transitions 중 한 번만 global을 이겼다.
- `[Hypothesis]` visual neighbor별로 별도 map을 주면 사용할 correspondence 수가 줄어 regression variance가 커지고, global map 이후 남은 residual은 visual hierarchy와 정렬되지 않는 것으로 보인다.

근거: `mds/results/cmpt_drift_grouping_correlation_cifar100.md`, `mds/results/cmpt_neighbor_affine_k5_ablation.md`.

### 11.4 Moment-aware와 weighted affine 탐색

- `[Verified]` moment-aware prototype transport는 NME보다 14/14 AIA가 높았지만 global affine 대비 CIFAR LUCIR, FGP, PODNet, CaSpeR와 ImageNet LUCIR, FGP, CaSpeR에서 낮았다.
- `[Verified]` mean-calibrated, second-moment-calibrated, 둘 모두를 쓴 weighted affine의 7-learner CIFAR 평균 AIA는 각각 `57.539`, `57.575`, `57.577`로 uniform affine `57.648`보다 낮았다.
- `[Verified]` full-space degree-2, PCA-rank 변화와 hyperparameter grid는 exploratory 기록에 있으나, independent validation이 아니라 test comparison을 포함하므로 primary result로 올리면 안 된다.
- `[Hypothesis]` 더 복잡한 nonlinear estimator가 표현력은 높지만 20 exemplar/class 환경에서 variance와 hyperparameter sensitivity를 증가시켰을 가능성이 있다.

근거: `mds/results/moment_aware_prototype_transport_cifar100_imagenet100.md`, `mds/results/cmpt_moment_calibrated_weighted_affine_cifar100.md`.

### 11.5 Oracle gap recovery 정의

$$
\mathrm{Recovery}_m=
\frac{\mathrm{AIA}^{affine}_m-\mathrm{AIA}^{NME}_m}
{\mathrm{AIA}^{oracle}_m-\mathrm{AIA}^{NME}_m}\times100.
$$

- `[Verified]` CIFAR method별 recovery는 iCaRL 22.5%, LUCIR 47.4%, FGP 10.4%, PODNet 46.6%, AFC 2.7%, CSCCT 17.3%, CaSpeR 32.8%다.
- `[Verified]` 보고한 24.8%는 method별 ratio의 단순 평균이 아니라 **7-learner mean gain `0.457`을 mean oracle gap `1.838`로 나눈 aggregate ratio**다.
- `[Verified]` 현재 7개 ratio는 음수나 100% 초과가 없다. 향후 그런 경우가 생기면 clamp하지 말고 원값을 보고해야 한다.
- `[Verified]` ImageNet oracle이 없어 two-dataset recovery ratio는 계산할 수 없다.

---

## 12. 연구의 한계

1. **Oracle 회복률이 제한적이다.** `[Verified]` CIFAR 7-learner mean에서 oracle headroom `1.838%p` 중 Affine CMPT가 회복한 것은 `0.457%p`, 즉 `24.8%`다. Final 기준은 `21.5%`다. “약 30% 회복”보다 “약 25% 회복”이 정확하다.
2. **모든 metric/session에서 개선하지 않는다.** `[Verified]` AIA는 14/14 양수지만 CIFAR AFC final은 `-0.04%p`다. Session별로도 FGP S1, AFC S1–S3/S6/S10, PODNet 일부 session처럼 음수 변화가 존재한다.
3. **Affine assumption은 보장되지 않는다.** `[Hypothesis]` 하나의 global map은 class-dependent nonlinear drift를 놓칠 수 있다. 반대로 더 복잡한 local/quadratic estimator는 제한된 paired sample에서 variance가 커질 수 있다.
4. **순차 update error가 누적될 수 있다.** `[Verified]` `\tilde\mu_{t,c}`는 매 session previous bank를 다시 transform한다. `[Verified]` canonical S0 reference에서 current session으로 직접 보내는 exploratory variant는 iCaRL/FGP에서 sequential affine보다 나빴지만, 이것이 accumulation이 없다는 증거는 아니다.
5. **단일 seed다.** `[Verified]` 모든 primary 수치는 seed 1 한 번이며 표준편차와 significance test가 없다. 현재 효과 크기, 특히 `+0.024%p`나 `+0.082%p`는 random variation보다 크다고 입증되지 않았다.
6. **Dataset/protocol 범위가 좁다.** `[Verified]` CIFAR-100과 ImageNet-100의 B50-Inc5, 한 class order만 평가했다. B0, 다른 increments, larger ImageNet subset, long-tailed stream, 다른 memory size는 미수행이다.
7. **20 exemplars에 고정됐다.** `[Verified]` memory-size sensitivity가 없으므로 적은 memory에서 더 유리하거나 많은 memory에서 필요 없어진다는 추세를 주장할 수 없다.
8. **Native head 우월성과 별개다.** `[Verified]` ImageNet LUCIR/PODNet/AFC에서는 native learned head가 Affine NPC보다 높다. 현재 결과는 prototype classifier를 사용할 때 estimator를 개선한다는 증거이지 learner의 최고 가능한 classifier를 항상 개선한다는 증거가 아니다.
9. **Method recipe가 완전 통일되지 않았다.** `[Verified]` dataset, class order, memory, loader는 맞췄지만 epoch/LR/batch는 method별 recipe다. 따라서 learner 간 absolute ranking은 CMPT paired gain보다 덜 중요한 해석 대상이다.
10. **추가 statistic을 저장한다.** `[Verified]` class당 introduction prototype 하나를 저장한다. 크기는 작지만 “동일 memory”라고 쓸 때 image count만 같은지 total bytes까지 같은지 명시해야 한다.
11. **현재 실행은 offline checkpoint evaluator다.** `[Verified]` 정보 사용은 online CIL-compatible하게 구성할 수 있으나, 실제 streaming trainer에 통합하여 session arrival 때 state를 저장하는 end-to-end demonstration은 별도로 제시되지 않았다.
12. **Novelty overlap이 존재한다.** `[Literature]` LDC, SDC, ADC, DGASA가 prototype/semantic drift와 cross-space mapping을 이미 다룬다. 본 연구는 해당 분야를 인용하고 old-exemplar correspondence와 exemplar-based NME gap에 초점을 맞춰야 한다.
13. **Oracle은 upper bound일 뿐 목표 prototype의 최적성을 보장하지 않는다.** `[Verified]` full-data mean도 multimodal class나 cosine classifier에 최적이 아닐 수 있다.
14. **직접 overhead benchmark가 없다.** `[Planned]` 이론적 memory/complexity는 계산했지만 wall-clock과 peak memory 측정은 필요하다.

### 현재 결과로 주장하면 안 되는 문장

- “CMPT solves catastrophic forgetting.”
- “Feature drift is affine.”
- “CMPT outperforms all native CIL classifiers.”
- “The method recovers 30% of the oracle gap on both datasets.”
- “The results are statistically significant.”
- “This is the first prototype transformation/transport method in CIL.”
- “The method adds no memory.”

---

## 13. 교수님 피드백에 대한 기술적 답변

### 13.1 CIL에는 OOD 데이터만 들어가는가?

**발표용 답:** 아니요. CIL stream의 새 class는 이전 label set에 없다는 의미에서 novel class이지만, OOD detector가 골라낸 sample만 들어온다고 정의되지 않는다.

**기술적 답:** `[Verified]` 현재 코드에는 OOD detection, reject option, unknown buffer가 없다. 미리 정한 CIFAR-100/ImageNet-100 class order에 따라 labeled new-class training data가 session별로 제공된다. `[Hypothesis]` “실세계에서 unknown/OOD sample이 모여 새 class가 된다”는 것은 동기용 conceptual connection으로는 가능하지만 본 실험이 검증한 pipeline이라고 말하면 안 된다.

### 13.2 Base N–Inc M에서 N과 M은 무엇인가?

`[Literature]` `N`은 base session class 수, `M`은 각 incremental session에 추가되는 class 수다. `[Verified]` B50-Inc5는 S0 50개, S1–S10에 5개씩이므로 총 `1+(100-50)/5=11` sessions/평가 points다. 이 문서에서는 affine pair 수 `n_t`와 혼동하지 않도록 `N_base`, `M`으로 표기한다.

### 13.3 `f_1,…,f_T`는 모두 사용되는가?

`[Verified]` 모두 해당 session 종료 후 all-seen test set에서 평가된다. CMPT transition `t`에서는 `f_{t-1}`, `f_t`가 paired features 계산에 사용되며, inference에는 `f_t`만 사용한다. 과거 model 전체를 동시에 보존하지 않는다.

### 13.4 각 session에서 사용 가능한 데이터와 memory budget은?

- `[Verified]` S0: base 50-class full training data.
- `[Verified]` `t>0`: 현재 5 new-class full training data + old class당 20 exemplar.
- `[Verified]` test: 평가에만 사용.
- `[Verified]` memory는 total fixed budget이 아니라 **class당 고정 20개**이므로 class 수가 증가하면 총 memory가 1,000에서 2,000장으로 증가한다.

### 13.5 Exemplar-based와 exemplar-free CIL은 왜 나뉘는가?

**발표용 답:** 과거 원본 이미지를 보관할 수 있는지에 따라 사용할 수 있는 정보와 해결 난도가 달라지기 때문이다.

**기술적 답:** `[Literature]` exemplar-based는 제한된 old inputs의 저장/replay를 허용한다. Exemplar-free는 privacy, storage, policy 또는 benchmark constraint 때문에 old raw inputs를 저장하지 않으며 model parameters, class statistics/prototypes 등의 허용 범위는 논문 protocol에 따라 다르다. `[Verified]` 본 연구는 old exemplar image 20개/class를 사용하므로 exemplar-based CIL이다. `[Literature]` 기존 exemplar-free prototype prediction은 old image가 없어 new-class data를 drift proxy로 쓰지만, 본 연구는 old exemplar의 direct correspondence를 쓴다.

### 13.6 Training strategy와 classification strategy

**발표용 답:** Training strategy는 model representation이 old knowledge를 덜 잊도록 학습을 바꾸고, classification strategy는 학습된 feature space에서 label을 결정하는 방식을 바꾼다. 둘은 함께 사용할 수 있다.

**기술적 답:**

- `[Literature]` replay는 old samples를 gradient에 다시 포함하고, KD는 previous model output/feature를 target으로 두어 representation/decision의 급격한 변화를 제한한다.
- `[Literature]` classification strategy는 learned head, bias correction, nearest-prototype 등 test-time decision rule을 정한다.
- `[Hypothesis]` 이 이분법은 발표를 위한 유용한 축이지 CIL 문헌 전체를 배타적으로 양분하는 공식 taxonomy는 아니다. 예를 들어 prototype-based loss는 training과 classification 양쪽에 걸친다.
- `[Verified]` 본 연구는 7개 training learners의 checkpoint를 그대로 두고 같은 NME/Affine classification comparison을 적용했다.
- `[Verified]` CMPT는 feature extractor 자체의 forgetting을 되돌리지 않는다. 주어진 current feature space에서 old-class representative의 estimation error를 줄이려 한다.

### 13.7 Parametric classifier와 non-parametric classifier

- `[Literature]` Parametric classifier는 FC/cosine/proxy weight처럼 gradient로 학습되는 persistent decision parameters를 갖는다.
- `[Literature]` Non-parametric classifier는 별도 learned classification head 대신 stored samples/statistics와 거리 규칙으로 예측한다. kNN은 sample neighbors, NCM/NME는 class representatives를 사용한다.
- `[Literature]` nearest-prototype는 family이고, NME는 exemplar mean을 prototype으로 쓰는 구체적 rule이다. kNN은 일반적으로 nearest-prototype의 하위가 아니라 별도 instance-based neighbor family로 보는 편이 명확하다.
- `[Hypothesis]` learned head가 new-class imbalance로 bias될 때 NME가 더 robust할 수 있지만, “head가 없어서 본질적으로 forgetting에 강하다”는 보편 명제는 성립하지 않는다.
- `[Verified]` 현재 결과에서도 native LUCIR/PODNet/AFC head가 Affine NPC보다 높으므로, headless robustness를 직접 입증하지 않는다. 해당 문장은 논문에서 사용하지 않는 편이 안전하다.

### 13.8 NPC, NCM, NME, exemplar mean, full-data mean의 관계

- `[Literature]` NPC: prototype과의 거리로 분류하는 family.
- `[Literature]` NCM: class mean을 prototype으로 쓰는 NPC.
- `[Literature]` NME: 저장 exemplar의 mean을 쓰는 iCaRL식 NCM/NPC.
- `[Verified]` Exemplar mean: current `f_t`에서 저장 20개로 추정한 mean.
- `[Verified]` Full-data mean: current `f_t`에서 해당 class training split 전체로 계산한 mean.
- `[Verified]` “NME algorithm is used to calculate a prototype”은 불완전하다. 권장 표현은 “We compute each NME prototype as the normalized mean of stored exemplar features and classify by cosine nearest-prototype matching.”이다.
- `[Verified]` CMPT도 NPC지만 old-class prototype이 exemplar mean이 아니므로 NME라고 부르지 않는다.

### 13.9 왜 exemplar mean이 full-data mean과 달라지는가?

`[Verified]` 두 요인이 함께 작용한다. 첫째, 20개가 전체 within-class distribution을 덜 포괄하는 finite-sample/coverage error다. 둘째, exemplar는 과거 feature space에서 herding되었고 representation이 변하므로 당시의 mean approximation 성질이 current `f_t`에서 유지되지 않는다. “모델이 exemplar만 학습해서 mean이 달라진다”는 설명은 부정확하다. 학습 data imbalance는 drift의 원인이 될 수 있지만, 측정된 mean gap은 selection coverage와 drift의 결합이다.

### 13.10 Prototype error가 classification error로 이어지는 근거

`[Verified]` CIFAR pooled correlation은 prototype cosine distance와 oracle accuracy gap 사이 Pearson `0.724`, Spearman `0.722`이고, old-only oracle은 70/70 incremental points에서 NME보다 높았다. `[Hypothesis]` 이는 prototype error가 decision boundary error에 기여한다는 근거지만 인과의 유일 경로나 모든 learner 내부에서 단조 관계임을 증명하지 않는다.

### 13.11 Feature-space transformation을 affine으로 모델링하는 근거

`[Literature]` cross-session prototype/semantic alignment를 linear/projector 방식으로 근사한 선행연구가 있다. `[Verified]` 본 구현은 실제 old exemplar pairs가 제공하는 overdetermined regression으로 `A,b`를 맞추고 ridge로 conditioning을 안정화한다. `[Hypothesis]` 선택 이유는 exact theory가 아니라 common drift component를 표현하면서 closed-form으로 추정 가능한 bias–variance compromise다. Full affine의 절대적 우월성은 입증되지 않았다.

### 13.12 Oracle 관련 교수님 수치 질문

- `[Verified]` `+3.37%p average`, `+5.20%p last`는 CIFAR FGP-ICL **all-seen oracle**과 20-exemplar NME의 차이다.
- `[Verified]` 여기서 `+3.374%p`는 S0를 제외한 S1–S10 평균이다. 본 논문 주 AIA 정의(S0–S10)에서는 `+3.082%p`다.
- `[Verified]` old-only oracle의 S0–S10 gap은 `+3.049%p`, final gap은 `+5.19%p`다.
- `[Verified]` 7-learner 평균 oracle gap은 AIA `+1.838%p`, final `+2.609%p`다.
- 권장 문장: “On CIFAR-100, replacing only old-class memory means with current full-training-data means yields a mean AIA headroom of 1.84 percentage points across seven learners.”
- 피할 문장: “Full-data prototype prediction reduces forgetting by 3.37% on average.” 이는 특정 FGP curve, 다른 AIA convention, oracle access를 숨긴다.

---

## 14. 논문에서 사용해야 할 핵심 문장

아래 문장은 과장을 줄인 English paper-ready 초안이며 최종 문장 교정은 별도로 필요하다.

### 14.1 Problem statement

1. `[Verified]` “Nearest-mean-of-exemplars classification represents each previously learned class by the mean feature of a small rehearsal memory.”
2. `[Hypothesis]` “Although these means are recomputed in the current representation space, a small set of exemplars may no longer approximate the current full-data class mean after repeated representation updates.”
3. `[Verified]` “Across seven CIFAR-100 learners, current full-data old-class means provide an average AIA headroom of 1.84 percentage points over 20-exemplar NME.”

### 14.2 Motivation

1. `[Hypothesis]` “The full class data are available when a class is introduced but become inaccessible in later sessions; preserving their mean therefore captures information that cannot be reconstructed from memory alone.”
2. `[Hypothesis]` “Stored old exemplars provide paired landmarks that directly reveal how old-class features move between consecutive models.”
3. `[Hypothesis]` “This suggests updating an introduction-time full-data prototype rather than re-estimating it solely from the current memory mean.”

### 14.3 Method summary

1. `[Verified]` “For each class, we store its normalized full-data prototype at the session of introduction.”
2. `[Verified]` “After each incremental update, the same retained old exemplars are embedded by the previous and current feature extractors to form paired correspondences.”
3. `[Verified]` “We fit a session-wise ridge-regularized affine map in closed form and apply it to all stored old-class prototypes.”
4. `[Verified]` “New-class prototypes are initialized directly from their currently available full training data.”
5. `[Verified]` “Inference uses cosine nearest-prototype classification without modifying the base learner's training objective.”

### 14.4 Contributions

1. `[Verified]` “We quantify the gap between memory-based and current full-data prototypes across seven exemplar-based CIL learners and identify a consistent classification headroom on CIFAR-100.”
2. `[Verified]` “We introduce a learner-independent post-hoc estimator that uses retained old exemplars as direct cross-session correspondences to transform introduction-time full-data prototypes.”
3. `[Verified]` “In single-seed paired evaluations on CIFAR-100 and ImageNet-100, the estimator improves NME AIA for all 14 learner–dataset trajectories, with mean gains of 0.457 and 0.360 percentage points, respectively.”

### 14.5 Result summary

1. `[Verified]` “The largest observed gain is 1.291 percentage points in AIA and 1.38 percentage points in final accuracy on CIFAR-100 LUCIR.”
2. `[Verified]` “On CIFAR-100, the proposed estimator recovers 24.8% of the average old-class full-data oracle AIA gap.”
3. `[Verified]` “The gains are estimator-specific: the proposed NPC does not exceed every learner's native parametric head.”

### 14.6 Limitation

1. `[Verified]` “Our current results use one class order and one random seed, and therefore do not establish statistical significance.”
2. `[Hypothesis]` “A single affine map captures only shared drift and may miss class-dependent nonlinear changes, while richer maps can overfit the limited exemplar correspondences.”
3. `[Verified]` “Sequential transformation may accumulate estimation error, and the method recovers only about one quarter of the measured CIFAR-100 oracle AIA gap.”

---

## 15. 사실·가설·향후 계획 구분

| Tag | 본 연구에서 해당하는 핵심 내용 |
|---|---|
| `[Verified]` | 20-exemplar NME와 full-data oracle gap; 코드의 augmented ridge-affine solve; introduction prototype과 paired old exemplars; 14개 primary trajectories; exact AIA/final gains; seed 1; memory/recipe/config |
| `[Literature]` | CIL/NME 정의; rehearsal/KD 개념; SDC/LDC/ADC/DGASA의 prototype/semantic drift compensation; ridge regression의 역할 |
| `[Hypothesis]` | small-memory coverage error가 boundary error의 한 원인이라는 해석; global affine이 shared drift를 근사한다는 modeling assumption; actual old pairs가 new-data proxy보다 적절할 수 있다는 해석 |
| `[Planned]` | multi-seed; ImageNet oracle; `λ_aff`/memory-size/b-removal/diagonal ablation; direct LDC/SDC adapter comparison; runtime profiling; online trainer integration |
| `[Unknown]` | `λ_aff=0.01`의 독립 validation 근거; 최종 acronym/title; auxiliary vectors를 포함한 exact byte-budget policy; IWAIT 제출판에서 native-head 결과를 main/appendix 어디에 둘지 |

### 핵심 용어 권장안

| 현재 혼용 가능 표현 | 논문/PPT 권장 표현 | 피할 표현 |
|---|---|---|
| NME algorithm / prototype mean | nearest-mean-of-exemplars (NME) classifier; NME prototype | “NME is only a prototype calculation” |
| NPC | nearest-prototype classifier (처음에 full spelling) | acronym만 정의 없이 반복 |
| full mean / ideal prototype | current full-training-data prototype; oracle일 때 oracle 명시 | ideal/optimal prototype |
| transformed/transported prototype | affine-transformed prototype | recovered ground-truth prototype |
| feature drift | inter-session representation drift | affine drift as a fact |
| initial prototype | introduction-time full-data prototype | base prototype: incremental new classes와 혼동 |
| A, b | transformation matrix `A`, translation vector `b` | translation matrix `b` |

---

## PPT 수정 우선순위

저장소에는 `.ppt/.pptx` 원본이 없으므로 아래 “Current problem”은 제공된 교수님 질문과 연구 문서에서 확인된 개념 혼동을 기준으로 작성했다.

| Priority | Slide topic | Current problem | Required correction | Evidence |
|---:|---|---|---|---|
| 1 | 문제 정의 | CIL stream을 OOD detector 출력처럼 설명할 위험 | 현재 benchmark는 labeled class stream이고 OOD 연결은 conceptual임을 명시 | training configs/data manager에 OOD module 없음 |
| 2 | NME 용어 | NME를 prototype 계산법으로만 표현 | “exemplar mean을 쓰는 nearest-prototype classifier”로 정의 | iCaRL; `compute_nme_class_means`, `evaluate_nme` |
| 3 | Prototype limitation | “20개라서 나쁘다”를 단정 | selection-time coverage error + representation drift로 mean approximation이 변할 수 있다는 논리로 수정 | oracle/cosine-distance 분석 |
| 4 | Oracle graph | 3.37/5.20을 전체 평균처럼 보이게 할 가능성 | FGP all-seen, S1–S10 incremental average임을 caption에 명시; 주 평균은 1.838/2.609 | FGP figure JSON, oracle result JSON |
| 5 | Method diagram | `f_0,…,f_T`, data availability, 저장 시점이 불명확 | introduction full data → stored prototype; same exemplars through `f_{t-1},f_t`; global `A_t,b_t` → old bank update 표시 | CMPT evaluator flow |
| 6 | Closed form | centering 식과 augmented code가 달라 보일 수 있음 | 두 식이 unregularized intercept 아래 동등함과 row/column convention을 함께 표기 | `affine_ridge_transport` |
| 7 | Affine rationale | affine drift가 이론적 사실처럼 보일 위험 | tractable modeling assumption + paired fitting + ridge라고 명시 | section 7, ablations |
| 8 | Results | maximum gain만 강조 | 14-row paired table, mean gain, AFC negative last, single seed를 함께 표시 | primary result MD/JSON |
| 9 | Baselines | learner 간 absolute 성능을 같은 recipe처럼 비교 | common protocol이지만 method-specific hyperparameters임을 footnote | train configs/resolved configs |
| 10 | Native classifiers | 모든 learner의 기본 classifier가 NME인 것처럼 표현 | iCaRL/CaSpeR는 NME, 나머지는 native learned head가 있음을 분리 | result doc native-head section |
| 11 | Contribution | first/solve/zero-memory 과장 위험 | diagnostic + actual-old-pair post-hoc estimator + broad paired evaluation으로 한정 | literature comparison + memory accounting |
| 12 | Limitations | 단일 seed·24.8% recovery 누락 | single seed/no std, affine assumption, accumulation, native-head gap 명시 | result JSON/section 12 |

---

## 권장 발표 스토리라인

| Slide | 전달할 한 문장 | 반드시 포함할 기술 내용 | 권장 용어 | 피할 과장 | Diagram / graph |
|---:|---|---|---|---|---|
| 1. CIL 설정 | 모델은 session마다 새 class를 배우고 task ID 없이 모든 seen class를 구분해야 한다. | B50-Inc5; S0 50, S1–S10 each 5; all-seen evaluation | class-incremental learning, seen classes | OOD-only stream | class timeline |
| 2. Catastrophic forgetting | 새 class 학습은 old representation과 decision rule을 바꿔 old accuracy를 낮춘다. | new full data vs limited old memory; representation/head bias | representation drift, forgetting | 단일 원인 설명 | feature clusters before/after |
| 3. 두 완화 축 | 학습을 안정화하는 방법과 학습 후 분류 rule을 개선하는 방법은 결합 가능하다. | replay/KD vs learned head/NPC; taxonomy가 편의적임 | training strategy, classification strategy | 공식적으로 완전한 양분 | two-branch conceptual diagram |
| 4. NPC와 NME | NME는 20 exemplar mean을 prototype으로 쓰는 nearest-prototype classifier다. | normalization, cosine argmax, current `f_t`에서 재계산 | NME classifier, memory mean | NME=mean calculation only | query와 class prototypes |
| 5. NME prototype 한계 | 과거에 잘 고른 20개도 representation이 바뀐 뒤 current full class mean을 보장하지 않는다. | herding at introduction; coverage + drift; not “only trained on exemplars” | memory prototype, current full-data prototype | 20개는 항상 나쁨 | 2D exemplar/full mean displacement |
| 6. Oracle motivation | 실제 full-data mean을 쓰면 남아 있는 classification headroom을 측정할 수 있다. | old-only oracle; inaccessible old data; mean AIA gap 1.838 | oracle upper bound/headroom | oracle is usable method | NME vs oracle session graph |
| 7. Proposed transformation | class introduction 때 저장한 full-data prototype을 actual old exemplar drift로 갱신한다. | `μ_intro`; paired same images; `A_t,b_t`; old-only replacement | affine-transformed prototype | exact recovery | previous/current feature spaces with arrows |
| 8. Algorithm | 한 session에 global affine 하나를 closed form으로 맞추고 old bank에 반복 적용한다. | objective, ridge only on A, new classes direct initialize, accumulation | session-global ridge-affine map | class-specific map | 6-step pseudocode/flowchart |
| 9. Experiment setting | 7 learners×2 datasets에서 checkpoint를 고정하고 prototype estimator만 paired 비교했다. | B50-Inc5, R32/R18, 20/class, seed1, `λ=.01`, natural loader | paired evaluation | fully unified training recipe | compact setup table |
| 10. Results | AIA는 14/14 증가했지만 평균 gain은 0.36–0.46%p이고 native head 우월성과는 별개다. | full 14-row or summary; max1.291; AFC final−.04; no std | AIA, final accuracy, %p | overwhelming/SOTA gain | dataset-wise paired dot/bar plot |
| 11. Limitations/next | Affine은 oracle gap의 약 25%만 회복하며 multi-seed와 core ablation이 남았다. | single seed, ImageNet oracle absent, λ/memory/b ablations, accumulated error | measured limitation, planned validation | solves forgetting | oracle-gap decomposition bar |

---

## 연구자에게 확인할 질문

1. **주장 범위:** 최종 논문의 중심 claim을 “NME를 native로 쓰는 learner의 개선”으로 제한할 것인가, 아니면 learned native head가 있는 learner에도 alternative NPC로 적용 가능하다는 plug-in claim으로 둘 것인가?
2. **추가 seed:** CIFAR-100과 ImageNet-100의 7개 모두를 최소 3 seeds로 재실행할 것인가? 시간이 부족하면 어떤 3개 learner를 필수로 고를 것인가?
3. **`λ_aff` 선택:** `0.01`을 고정한 사전 근거가 있는가? 없다면 training/test label을 보지 않는 validation 또는 CIFAR 고정→ImageNet transfer protocol로 sensitivity를 수행할 것인가?
4. **필수 ablation:** 제출 전에 `b` 제거(linear-only), global translation-only, diagonal affine, `λ=0`을 수행할 것인가?
5. **Memory accounting:** 20 exemplar image budget 외의 class prototype float vectors를 허용하는 protocol로 명시할 것인가? 동일 total-byte budget control도 할 것인가?
6. **Oracle convention:** PPT의 `3.37/5.20` FGP-specific incremental metric을 유지할 것인가, 아니면 7-learner S0-included mean `1.838/2.609`로 교체할 것인가?
7. **Method name:** 저장소의 `CMPT`와 paper-facing `Affine Prototype Transformation` 중 어떤 명칭을 최종 acronym으로 사용할 것인가? 이 문서는 제목을 확정하지 않는다.
8. **Online integration:** offline evaluator 결과만 제출할 것인가, 아니면 learner training loop에 introduction prototype 저장과 session-end update를 통합한 online demonstration을 추가할 것인가?
9. **Native head reporting:** LUCIR/PODNet/AFC/CSCCT native head 결과를 main table, appendix, 또는 별도 reference column 중 어디에 둘 것인가?
10. **ImageNet oracle:** CIFAR에서만 보고한 oracle headroom/recovery를 ImageNet-100에도 계산할 것인가?
11. **Prior-method comparison:** LDC/SDC/DGASA의 projector를 exemplar-pair 조건에 맞춰 직접 adapter한 baseline을 추가할 것인가, 아니면 setting 차이를 related work에서만 논의할 것인가?
12. **CaSpeR 표기:** 현재 구현을 “CaSpeR-IL+iCaRL”로 일관되게 표기할 것인가? ImageNet recipe가 author-provided가 아닌 controlled adaptation임을 어디에 명시할 것인가?
13. **Class order 확대:** 한 published order만 유지할 것인가, 여러 class orders를 seed 실험에 포함할 것인가?
14. **발표의 OOD 동기:** OOD와 CIL의 연결을 완전히 제거할 것인가, “잠재적 upstream scenario” 한 문장으로만 남길 것인가?
15. **IWAIT 제출 범위:** negative exploratory methods(local, residual, moment-aware)를 appendix ablation으로 공개할 것인가, 아니면 global affine 선택 근거만 간략히 제시할 것인가?

---

## 핵심 근거 파일 색인

| 목적 | 근거 |
|---|---|
| Primary 14-row result | `mds/results/cmpt_nme_vs_affine_results_current.md` |
| CIFAR full-data oracle | `mds/results/cmpt_full_training_mean_oracle_cifar100.md` |
| Main CMPT evaluator | `src_cmpt/sacil/cmpt/evaluator.py` |
| Affine closed-form implementation | `src_cmpt/sacil/methods/prototype_transport.py::affine_ridge_transport` |
| NME implementation | `src_cmpt/sacil/engine/evaluator.py::compute_nme_class_means`, `evaluate_nme` |
| CMPT primary config | `configs/cmpt/_cmpt_affine_common.yaml` |
| CIFAR protocol/config | `configs/table1/cifar100/_common_b50_inc5.yaml`, `configs/cmpt/common_recipe/train_*.yaml` |
| ImageNet protocol/config | `configs/cmpt/imagenet100_b50_inc5/_training_common.yaml`, `train_*.yaml` |
| Class orders | `experiment_configs/class_orders/cifar100_b50_t10_afc_order1.json`, `imagenet100_b50_inc5_afc_order1.json` |
| Dataset transforms | `src_cmpt/sacil/data/cifar100.py`, `src_cmpt/sacil/data/imagenet100.py` |
| Rigid/affine analysis | `mds/results/cmpt_geometry_rigid_vs_affine_seed1.md` |
| Class residual analysis | `mds/results/cmpt_global_class_residual_ablation.md` |
| Grouping analysis | `mds/results/cmpt_drift_grouping_correlation_cifar100.md`, `cmpt_neighbor_affine_k5_ablation.md` |
| Moment analyses | `mds/results/moment_aware_prototype_transport_cifar100_imagenet100.md`, `cmpt_moment_calibrated_weighted_affine_cifar100.md` |
| FGP oracle plot metadata | `mds/results/figures/cifar100_fgp_icl_full_vs_affine_vs_20_session_accuracy.json` |

`[Verified]` 이번 문서 작성 중 기존 source/config/result는 수정하지 않았고 새 학습 실험도 실행하지 않았다. 생성한 산출물은 이 Markdown 문서 하나다.
