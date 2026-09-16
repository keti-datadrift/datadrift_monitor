# AL for OD 아이디어 구체화

**User:** Anonymous  
**Created:** 7/20/2026 15:47:30  
**Updated:** 9/8/2026 20:41:42  
**Exported:** 9/8/2026 20:52:52  
**Link:** [https://chatgpt.com/g/g-p-67ced699603c8191b57089e3c489b87c/c/6a5dc46e-7ab4-83ee-8cd2-7da61f020bc8](https://chatgpt.com/g/g-p-67ced699603c8191b57089e3c489b87c/c/6a5dc46e-7ab4-83ee-8cd2-7da61f020bc8)  

## Prompt:
2026. 9. 8. 오후 8:26:23

조금 근본적인 질문으로 돌아갈게. 이 Error Prediction-based Active Learning이 충분한 novelty가 있다고 생각해?

## Response:
2026. 9. 8. 오후 8:26:27

> **웹 검색 중**
>
> [CVPR 2019 Open Access Repository](https://openaccess.thecvf.com/content_CVPR_2019/html/Yoo_Learning_Loss_for_Active_Learning_CVPR_2019_paper.html)
> [Learning Loss for Active Learning](https://arxiv.org/abs/1905.03677)
> [Learning Loss for Active Learning | ML Anthology](https://mlanthology.org/cvpr/2019/yoo2019cvpr-learning/)
> [CVPR 2019 开源访问库 - 计算机视觉基金会开放获取](https://openaccess.thecvf.com.cn/content_CVPR_2019/html/Yoo_Learning_Loss_for_Active_Learning_CVPR_2019_paper.html)
> [GitHub - superannotateai/active_learning · GitHub](https://github.com/superannotateai/active_learning)
>
> **신규성 재평가**
>
> PAL과 Learning Loss를 고려하면, 오류 예측과 불확실성 기반 샘플링만으로는 신규성이 약해 보여요. 오류 유형 분해, 탐지기 출력 기반 이식성, 누락 객체 모델링이 실제로 차별적이고 강하게 검증되어야 합니다.
>
> **신규성을 재평가함**
>
> 
>
> **웹 검색 중**
>
> [Portable Active Learning for Object Detection](https://arxiv.org/abs/2605.10349)
> [Portable Active Learning for Object Detection](https://openaccess.thecvf.com/content/CVPR2026/papers/Sharma_Portable_Active_Learning_for_Object_Detection_CVPR_2026_paper.pdf)
> [[Literature Review] Portable Active Learning for Object Detection](https://www.themoonlight.io/en/review/portable-active-learning-for-object-detection)
> [Portable Active Learning for Object Detection | alphaXiv](https://www.alphaxiv.org/abs/2605.10349v1)
> [PAL](https://www.catalyzex.com/s/PAL)
> [CVPR2026 Object Detection Papers · 99 Notes](https://en.papernotes.org/CVPR2026/object_detection/)
> [Portable Active Learning for Object Detection — Scifaro](https://scifaro.com/en/abs/portable-active-learning-for-object-detection-2605.10349)
> [[2605.10349] Portable Active Learning for Object Detection](https://rt.http3.lol/index.php?q=aHR0cHM6Ly9hcnhpdi5vcmcvYWJzLzI2MDUuMTAzNDk)
> [Portable Active Learning for Object Detection | Request PDF](https://www.researchgate.net/publication/404753739_Portable_Active_Learning_for_Object_Detection)
> [Portable Active Learning for Object Detection](https://chatpaper.com/chatpaper/paper/278287)
>
> **구분해서 평가하기**
>
> 아이디어 수준에서는 PAL과 Learning Loss 때문에 신규성이 약합니다. 논문 수준의 기여는 분류·위치·누락 오류의 명시적 분해, 탐지기 출력 기반 특징, 정답성 엔트로피 결합에 달려 있으며, 각 요소의 실효성을 엄격히 입증해야 합니다.
>
> **웹사이트 5개 검색함**
>
> 
>
> 44s 동안 처리함

현재 상태를 기준으로 보면, **“Error Prediction-based Active Learning”이라는 큰 아이디어 자체만으로는 novelty가 충분하다고 보기 어렵다.** 다만 네가 지금까지 만든 구조 안에는 **논문 novelty로 발전시킬 수 있는 부분은 분명히 있다.** 핵심은 무엇을 contribution으로 잡느냐야.

가장 큰 이유는 이미 두 계열의 선행연구가 있기 때문이야. `Learning Loss for Active Learning`은 아예 별도 모듈로 unlabeled sample의 target loss를 예측하고, 모델이 잘못할 가능성이 높은 sample을 선택하는 아이디어를 제시했다. Object detection에도 실험했다. 따라서 **“모델의 오류/실패를 예측해서 AL에 사용한다”**는 일반적인 발상은 새롭지 않다. ([Open Access CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Yoo_Learning_Loss_for_Active_Learning_CVPR_2019_paper.html?utm_source=chatgpt.com))

더 직접적인 문제는 PAL이야. PAL은 object detection에서 labeled detections를 TP/FP로 나누고, `confidence + pre-NMS box count`를 입력으로 logistic classifier를 학습한 뒤, TP/FP 확률의 entropy를 instance uncertainty로 사용한다. 즉 **“detector output으로 correctness/error boundary를 학습하고, 그 boundary의 entropy를 AL acquisition에 사용한다”**는 구조가 이미 아주 가까운 형태로 존재한다. ([arXiv](https://arxiv.org/abs/2605.10349?utm_source=chatgpt.com))

그래서 네 방법을 단순히 이렇게 설명하면 위험해.

$$
\boxed{
\text{Predict detection errors}
\rightarrow
\text{compute entropy}
\rightarrow
\text{select uncertain images}
}
$$

이 정도는 PAL과 너무 가깝다.

특히 지금 CECP가 가장 잘 동작하고, MOCP와 FDP가 약하고 LECP도 제한적이라고 했잖아. 만약 최종적으로 CECP만 남는다면 더 문제가 커진다.

예를 들어 네 CECP가

$$
[s_i,E_i^{cls}]
$$

두 feature를 사용하고,

$$
q_i^{cls}
=
P(\text{classification correct}\mid s_i,E_i^{cls})
$$

를 logistic regression으로 예측한 뒤,

$$
H_{\rm Bern}(q_i^{cls})
$$

를 쓰는 구조라면 PAL의

$$
[\text{confidence},\text{pre-NMS support count}]
\rightarrow
P(\mathrm{TP})
\rightarrow
H(P(\mathrm{TP}))
$$

와 conceptual structure가 상당히 유사해진다. PAL은 실제로 simple logistic classifier가 TP/FP decision boundary를 잘 포착한다고 주장하고 있고, 그 entropy를 LIUS로 사용한다. ([Moonlight](https://www.themoonlight.io/en/review/portable-active-learning-for-object-detection?utm_source=chatgpt.com))

그래서 **CECP 하나만 잘 되는 상태에서는 novelty가 약하다**고 보는 게 맞아.

---

## 그런데 네 방법에서 살릴 수 있는 novelty는 분명히 있음

내가 보기에는 핵심을 **Error Prediction**이 아니라 **Error-Type-Aware Error Prediction**으로 옮겨야 해.

PAL은 본질적으로 detection이

$$
\text{TP인가 FP인가}
$$

를 본다.

반면 네가 궁극적으로 만들고 싶은 것은 detection/model failure를

$$
\boxed{
\text{Classification}
\quad
\text{Localization}
\quad
\text{Missed Object}
}
$$

로 분해해서 각각 다른 predictor를 학습하는 것이잖아.

이 차이는 꽤 중요하다.

Object detection에서 한 detection이 FP라는 사실만 알아서는

- class를 잘못 예측한 건지,
- 위치가 부정확한 건지,
- background hallucination인지

알 수 없고, 아예 output되지 않은 missed object는 detection-level TP/FP predictor로는 다룰 수도 없다.

그래서 네 방법이 성공적으로 완성된다면 다음 구조가 된다.

$$
\boxed{
\text{Generic detection correctness}
}
$$

가 아니라

$$
\boxed{
\text{Structured prediction of detector failure modes}
}
$$

를 AL에 활용하는 거야.

이건 PAL과 분명히 다른 이야기가 될 수 있다.

---

# 내가 novelty를 평가하면 현재는 이렇게 봄

| 아이디어 | Novelty 평가 |
|---|---|
| Error predictor를 AL에 사용 | 약함 |
| Logistic regression으로 error probability 추정 | 약함 |
| Error probability entropy로 sampling | PAL 때문에 매우 약함 |
| Pre-NMS/raw detection support 활용 | PAL과 부분적으로 겹침 |
| Classification correctness를 별도 예측 | 약~중간 |
| Localization correctness를 별도로 예측 | **중간 이상 가능** |
| Missed-object count를 별도로 예측 | **상당히 차별화 가능** |
| Cls/Loc/Miss uncertainty를 하나의 error profile로 구성 | **좋은 novelty 후보** |
| Detector-output-only로 세 failure modes를 통합 | **좋은 novelty 후보** |

즉 **개별 부품보다 decomposition과 통합 구조가 novelty**가 되어야 해.

---

# 특히 LECP가 중요해지는 이유

처음에는 CECP가 잘 되니까 CECP가 핵심처럼 보이지만, novelty 관점에서는 반대야.

CECP는 PAL과 가장 많이 겹친다.

반면 LECP가 정말 잘 작동한다면

$$
[s_i,E_i^{loc}]
\rightarrow
P(\text{localization correct})
$$

라는 별도의 localization decision boundary를 학습하고,

$$
H_{\rm Bern}
(P(\text{localization correct}))
$$

를 acquisition에 사용하는 구조가 된다.

이건

> “Detection uncertainty를 하나의 TP/FP uncertainty로 표현하지 않고, classification과 localization failure boundary를 별도로 학습한다.”

라는 명확한 contribution이 된다.

PPAL도 classification/localization difficulty를 uncertainty sampling에 함께 고려하지만, detector가 실제로 어떤 error type을 낼지를 별도 correctness predictor로 학습하는 구조와는 다르다. PPAL은 classification/localization difficulty를 이용해 instance uncertainty를 calibration하는 two-stage AL 방식이다. ([Open Access CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Plug_and_Play_Active_Learning_for_Object_Detection_CVPR_2024_paper.html?utm_source=chatgpt.com))

그래서 지금 개발 우선순위에서도 **LECP를 CECP 수준에 가깝게 끌어올리는 게 novelty 측면에서 매우 중요하다.**

---

# MOCP는 성공하면 가장 독특한 component일 수 있음

Missed object는 final prediction이 없기 때문에 일반적인 detection uncertainty로 잡기가 어렵다.

PAL의 TP/FP classifier도 기본적으로 **존재하는 predicted instance**를 대상으로 한다. ([arXiv](https://arxiv.org/abs/2605.10349?utm_source=chatgpt.com))

반면 네가 raw detections의 residual structure로

$$
N_{\rm miss}
$$

를 예측하고,

$$
N_{\rm miss}
\sim
\operatorname{Poisson}(\lambda_x)
$$

의 uncertainty를 AL에 사용한다면 이건 꽤 다른 문제를 해결한다.

즉

$$
\boxed{
\text{uncertainty of existing detections}
}
$$

뿐 아니라

$$
\boxed{
\text{uncertainty of objects absent from final detections}
}
$$

까지 acquisition signal에 포함하는 거야.

이 부분이 제대로 동작하면 논문의 novelty가 상당히 좋아질 수 있어.

문제는 지금 MOCP 성능이 거의 없다는 거고, 그래서 **현재 상태에서는 contribution이라고 주장하기 어렵다.**

---

# 따라서 이 연구의 운명은 사실 predictor 개선 결과에 달려 있음

나는 다음 세 상황을 구분할 것 같아.

### Case A. CECP만 잘 됨

$$
\text{CECP strong},\quad
\text{LECP weak},\quad
\text{MOCP weak}
$$

이면 논문 novelty는 약해.

PAL의 LIUS 변형으로 보일 위험이 크다.

아무리 EUA-only가 PAL보다 mAP가 조금 높아도 리뷰어가

> “This is essentially PAL-style logistic uncertainty with modified features.”

라고 볼 수 있다.

---

### Case B. CECP + LECP가 잘 됨

$$
\text{CECP strong},\quad
\text{LECP strong}
$$

이면 이야기가 꽤 좋아진다.

$$
\boxed{
\text{classification error boundary}
\neq
\text{localization error boundary}
}
$$

를 명시적으로 학습하고 두 uncertainty를 결합하는 것이 contribution이 된다.

그리고 2D decision-boundary figure를 각각 보여주면 시각적으로도 매우 설득력 있어.

---

### Case C. CECP + LECP + MOCP가 모두 의미 있게 됨

이게 가장 이상적이다.

$$
\boxed{
\text{Classification}
+
\text{Localization}
+
\text{Missed Object}
}
$$

라는 object detector의 서로 다른 failure modes를 예측하고 각각의 predictive uncertainty를 AL에 사용한다.

이 상태면 나는 **독립된 방법으로서 충분한 novelty가 생길 가능성이 높다**고 본다.

특히 missed object까지 들어가면 기존 predicted-instance 중심 AL과 차이가 명확해진다.

---

# 또 하나 중요한 건 EUA 자체가 단순 entropy sum이면 조금 약함

Predictor decomposition이 성공하더라도

$$
EUA
=
U_{cls}+U_{loc}+U_{miss}
$$

만 하면 acquisition 부분 자체의 novelty는 크지 않아.

하지만 반드시 acquisition 식까지 복잡하게 만들 필요는 없어.

오히려

> “새로움은 uncertainty aggregation 공식이 아니라 **어떤 random variables의 uncertainty를 추정하느냐**에 있다.”

라고 가는 게 낫다.

즉

$$
H(q_{cls}),\quad
H(q_{loc}),\quad
H_{\rm Pois}(\lambda_{miss})
$$

라는 서로 다른 error distributions를 정의한 게 핵심.

EUA는 그걸 image-wise acquisition score로 단순히 합치는 역할만 해도 된다.

---

# 현재 네 방법에서 가장 위험한 부분

지금 분석에서 EUA와 detection count의 Spearman이 $0.9+$라는 건 novelty보다도 **mechanism validity**에 문제가 될 수 있어.

만약 최종 방법이 사실상

$$
\text{많은 detections}
\rightarrow
\text{많은 entropy terms}
\rightarrow
\text{high EUA}
$$

라면 리뷰어 입장에서는

> “Does the proposed error uncertainty actually matter, or is this just object-density sampling?”

이라는 질문이 바로 나온다.

그래서 predictor를 개선하는 것과 별개로 나중에는 반드시

$$
\boxed{
\text{error uncertainty signal}
\neq
\text{detection density signal}
}
$$

임을 보여줘야 해.

이건 성능뿐 아니라 method identity에도 중요해.

---

# 결론

내 판단은 이거야.

$$
\boxed{
\text{“Error Prediction-based AL” 자체는 novelty가 부족하다.}
}
$$

이미 Learning Loss가 failure/loss prediction을 AL에 사용했고, PAL은 object detection에서 TP/FP logistic prediction의 entropy를 직접 acquisition에 사용한다. ([Open Access CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Yoo_Learning_Loss_for_Active_Learning_CVPR_2019_paper.html?utm_source=chatgpt.com))

하지만 네 연구가

$$
\boxed{
\text{Classification correctness}
+
\text{Localization correctness}
+
\text{Missed-object count}
}
$$

를 **각각 명시적으로 예측하고**, 그 failure-mode-specific uncertainty를 이용해 데이터를 선택하는 방식으로 완성된다면 이야기가 달라진다.

그때의 핵심 novelty는

> **“Error prediction을 AL에 썼다”**

가 아니라

> **“Object detection의 heterogeneous failure modes를 각각 모델링하고, 그 failure-specific predictive uncertainty를 active acquisition에 사용한다.”**

가 돼야 해.

그래서 지금은 **CECP를 더 꾸미는 것보다 LECP와 MOCP를 실제로 의미 있게 만드는 것이 논문의 novelty를 확보하는 데 훨씬 중요하다.** 만약 LECP/MOCP 개선이 실패한다면, 그때는 연구 방향 자체를 한 번 다시 고민하는 게 맞고, 둘이 살아난다면 충분히 독립적인 논문 방향으로 발전할 여지가 있다고 본다.

---

**Sources:**

- [CVPR 2019 Open Access Repository](https://openaccess.thecvf.com/content_CVPR_2019/html/Yoo_Learning_Loss_for_Active_Learning_CVPR_2019_paper.html?utm_source=chatgpt.com)
- [Portable Active Learning for Object Detection](https://arxiv.org/abs/2605.10349?utm_source=chatgpt.com)
- [[Literature Review] Portable Active Learning for Object Detection](https://www.themoonlight.io/en/review/portable-active-learning-for-object-detection?utm_source=chatgpt.com)

## Prompt:
2026. 9. 8. 오후 8:38:13

그럼 PAL의 문제점을 분석하는 게 필요하겠네? 실제로 TP/FP만 고려하기 때문에 error-type별 개선을 보면 특정 error-type은 개선이 되지 않는다든지 하는 그런 문제를 찾는 것? 어떻게 생각해? 어떤 분석을 해야 할까?

## Response:
2026. 9. 8. 오후 8:39:17

> **구체 분석 설계하기**
>
> LIUS 단독으로 불확실성 효과를 분리하고, PAL 전체는 종합 성능으로 평가하겠습니다. 선택 이미지의 TIDE 오류 구성, 오류 유형별 포함률, 재학습 후 오류 감소량과 AP 개선량을 비교해 실제 오류 회복 여부를 검증하겠습니다.
>
> **구체 분석을 설계함**
>
> 
>
> **가설과 평가 설계**
>
> 저는 먼저 TIDE 오류를 분류한 뒤 LIUS 점수와의 관계를 검증하겠습니다. 특히 누락 오류, 배경 오류, 위치·분류 오류가 균등하게 선택되는지와 재학습 후 유형별 감소량을 비교하겠습니다. LIUS 단독과 PAL 전체를 분리해 GUIDE의 영향을 통제하겠습니다.
>
> **가설과 평가 설계를 수립함**
>
> 
>
> **PAL 성능 저하와 TP/FP 경계의 선택 편향 및 LIUS 불확실성 분석**
>
> 
>
> 1m 47s 동안 처리함

맞아. **PAL의 TP/FP 이진화가 실제로 어떤 failure mode를 놓치는지 분석하는 게 지금 제일 중요해 보여.** 다만 처음부터 “PAL은 localization/miss를 못 고친다”라고 결론을 정해놓기보다는,

$$
\boxed{
\text{TP/FP uncertainty가 서로 다른 detection failure modes를 균등하게 포착하는가?}
}
$$

를 검증하는 형태가 좋아.

PAL의 LIUS는 detection confidence와 pre-NMS box count로 class-specific logistic classifier를 학습해 TP/FP를 구분하고, 그 binary posterior의 entropy를 uncertainty로 사용한다. 즉 classification, localization, background 등 **FP가 발생한 원인은 구분하지 않는다.** PAL 논문 자체도 LIUS를 TP/FP decision boundary에 대한 uncertainty로 정의한다. `[CVPR]-[2026]-[Portable Active Learning for Object Detection].pdf`

내가 한다면 아래 순서로 분석해.

---

# 1. 가장 먼저: PAL uncertainty가 어떤 error type에 반응하는가

이게 핵심 분석이야.

PAL의 각 final detection $i$에 대해 LIUS를

$$
U_i^{\mathrm{PAL}}
=
H_{\mathrm{Bern}}
\left(
q_i^{\mathrm{TP}}
\right)
$$

라고 하자.

그 detection을 GT를 이용해 사후적으로 TIDE-type error로 분류한다.

$$
e_i
\in
\{
\mathrm{Correct},
\mathrm{Cls},
\mathrm{Loc},
\mathrm{Both},
\mathrm{Dup},
\mathrm{Bkg}
\}.
$$

여기서는 **ECPAL에서 만든 custom label 말고 standard TIDE-style diagnosis를 쓰는 게 좋다.** PAL의 한계를 분석하는 외부 diagnostic이기 때문이야.

그 다음 error type별 PAL entropy distribution을 본다.

예:

| Error type | Mean LIUS | Median LIUS | Top-10% LIUS 포함률 |
|---|---:|---:|---:|
| Correct | | | |
| Classification | | | |
| Localization | | | |
| Cls+Loc | | | |
| Duplicate | | | |
| Background | | | |

그리고 violin plot이나 box plot 하나.

x축:

$$
\mathrm{Error\ Type}
$$

y축:

$$
U_i^{\mathrm{PAL}}
$$

이걸 보면 예를 들어

$$
U_{\mathrm{Cls}}\gg U_{\mathrm{Loc}}
$$

같은 경향이 있는지 바로 알 수 있어.

### 여기서 찾고 싶은 현상

예를 들어 classification error는 PAL decision boundary 근처에 많이 있지만,

$$
\text{Localization errors}
\rightarrow
\text{low LIUS}
$$

라면 굉장히 좋은 motivation이 생겨.

> PAL models whether a detection is TP or FP, but the resulting uncertainty is not equally sensitive to different causes of false positives.

이게 CECP/LECP 분리의 직접적인 근거가 된다.

---

# 2. PAL의 TP/FP probability 자체가 error type별로 얼마나 informative한가

Entropy만 보지 말고 PAL classifier output도 봐야 해.

PAL은

$$
q_i^{TP}
=
P(\mathrm{TP}\mid s_i,n_i^{pre})
$$

를 예측하잖아.

따라서

$$
q_i^{FP}=1-q_i^{TP}
$$

가 각 error type을 얼마나 잘 찾아내는지 각각 측정할 수 있어.

예를 들어:

$$
\mathrm{AUROC}_{cls}
=
\operatorname{AUROC}
(q_i^{FP},
\mathbb 1[e_i=\mathrm{Cls}])
$$

$$
\mathrm{AUROC}_{loc}
=
\operatorname{AUROC}
(q_i^{FP},
\mathbb 1[e_i=\mathrm{Loc}])
$$

등.

표는:

| Target error | PAL $q^{FP}$ AUROC |
|---|---:|
| Classification | |
| Localization | |
| Cls+Loc | |
| Background | |

만약

$$
\mathrm{AUROC}_{cls}=0.85,
\qquad
\mathrm{AUROC}_{loc}=0.60
$$

같이 나오면 메시지가 아주 명확해.

**TP/FP predictor 자체가 모든 FP 원인을 동일하게 모델링하지 못한다.**

이건 네가 현재 CECP/LECP predictor quality를 따로 분석한 것과 정확히 대응할 수도 있어.

---

# 3. PAL high-uncertainty region 안에 실제로 어떤 error들이 있는가

PAL의 논문 그림처럼 TP/FP decision boundary를 그리되, 점을 TP/FP가 아니라 **error type별로 표시**하면 아주 좋은 그림이 될 수 있어.

PAL은 실제로 confidence와 pre-NMS count의 2D 공간에서 TP/FP decision boundary를 보여준다. `[CVPR]-[2026]-[Portable Active Learning for Object Detection].pdf`

똑같은 공간에서

- Correct
- Classification
- Localization
- Both
- Background

를 서로 다르게 표시해.

그리고

$$
q^{TP}=0.5
$$

boundary도 그린다.

그러면 visually:

> PAL이 high-entropy로 보는 decision-boundary region에는 어떤 error type이 몰려 있는가?

를 볼 수 있어.

예를 들어 localization error가 boundary와 무관하게 넓게 흩어져 있다면 굉장히 강한 그림이 된다.

---

# 4. Error type별 retrieval rate

이건 더 직접적이야.

PAL LIUS로 detections를 내림차순 정렬하고 top $K\%$를 선택한다고 하자.

각 error type $m$에 대해

$$
\boxed{
\mathrm{Recall}_m(K)
=
\frac{
\#\{\text{top-}K\%\text{에 포함된 error type }m\}
}{
\#\{\text{pool의 error type }m\}
}
}
$$

를 계산해.

그리고 $K$를 5%, 10%, 20%, ... 늘리면서 curve를 그린다.

예:

- Classification error recall
- Localization error recall
- Both error recall
- Background error recall

같은 acquisition budget에서 어떤 error가 빨리 회수되는지 볼 수 있어.

이 분석은 단순 mean entropy보다 더 **AL selection과 직접 연결**돼.

---

# 5. Missed object는 별도의 분석이 필요함

여기가 아마 PAL의 가장 구조적인 약점일 가능성이 있어.

PAL LIUS는 **존재하는 final detection**마다 계산된다. PAL은 class-specific TP/FP classifier로 predicted instances의 uncertainty를 계산한다. `[CVPR]-[2026]-[Portable Active Learning for Object Detection].pdf`

그런데 missed object는

$$
\boxed{\text{final detection 자체가 존재하지 않는다}}
$$

는 failure야.

따라서 해당 object에 대응되는 LIUS instance가 원천적으로 없다.

이걸 분석하면 MOCP의 motivation이 된다.

## 분석 5-A. Image PAL score vs missed-object count

각 image $x$에 대해

$$
N_{\mathrm{miss}}(x)
$$

를 GT로 계산하고 PAL image score와 비교한다.

Spearman:

$$
\rho
\left(
S_{\mathrm{PAL}}(x),
N_{\mathrm{miss}}(x)
\right)
$$

를 계산해.

그리고 scatter 또는 miss count별 PAL score box plot.

---

## 분석 5-B. High-miss image retrieval

Miss count가 높은 상위 10% images를

$$
\mathcal M_{\mathrm{hard}}
$$

라고 하자.

PAL top-budget selection에서 얼마나 포함되는지:

$$
\boxed{
\mathrm{Recall}_{miss}
=
\frac{
|\mathcal S_{\mathrm{PAL}}\cap\mathcal M_{\mathrm{hard}}|
}{
|\mathcal M_{\mathrm{hard}}|
}
}
$$

를 본다.

Random, entropy, PAL과 비교하면 좋아.

---

## 분석 5-C. 가장 흥미로운 경우

특히

$$
N_{\mathrm{GT}}\text{ 많음},
\qquad
N_{\mathrm{det}}\text{ 적음}
$$

인 image들을 따로 보자.

즉 detector가 여러 object를 놓친 경우.

이런 이미지에서는 PAL이 uncertainty를 계산할 predicted instance 자체가 적어.

그래서

$$
\boxed{
\text{high miss count}
+
\text{few detections}
}
$$

인 이미지의 PAL rank가 낮다면 MOCP의 필요성을 아주 직접적으로 보여줄 수 있어.

---

# 6. 실제 PAL이 선택한 이미지의 error composition

Detection 단위 분석 다음에는 image selection을 봐야 해.

같은 acquisition round에서 PAL이 선택한 $b$개 이미지에 실제로 포함된 error를 센다.

$$
\mathbf e(x)
=
[
N_{cls}(x),
N_{loc}(x),
N_{both}(x),
N_{bkg}(x),
N_{miss}(x)
].
$$

선택된 batch 평균:

$$
\bar{\mathbf e}_{PAL}
=
\frac1b
\sum_{x\in\mathcal S_{PAL}}
\mathbf e(x).
$$

그리고 pool 전체 평균과 비교해서 enrichment ratio를 만들면 좋아.

$$
\boxed{
ER_m
=
\frac{
\frac1b\sum_{x\in S}N_m(x)
}{
\frac1{|U|}\sum_{x\in U}N_m(x)
}
}
$$

예:

| Method | Cls | Loc | Both | Bkg | Miss |
|---|---:|---:|---:|---:|---:|
| Random | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Entropy | | | | | |
| PAL-LIUS | | | | | |
| PAL-full | | | | | |

heatmap으로 만들면 아주 보기 좋아.

### 중요한 주의점

**raw error count만 보면 안 돼.**

PAL이 단순히 object/detection이 많은 이미지를 선택해서 모든 error count가 많아질 수 있으니까,

추가로

$$
\frac{N_{cls}}{N_{det}},
\qquad
\frac{N_{loc}}{N_{det}},
\qquad
\frac{N_{miss}}{N_{GT}}
$$

같은 normalized error rates도 같이 봐야 해.

이건 네 EUA density 문제 분석과 같은 이유야.

---

# 7. 가장 중요한 분석: PAL로 학습하면 어떤 error type이 실제로 줄어드나

네가 말한

> “PAL은 특정 error type은 실제로 개선하지 못한다”

를 주장하려면 이 실험이 결정적이야.

PAL로 round $t$의 데이터를 선택하고 detector를 재학습한다.

재학습 전 test-set error vector:

$$
\mathbf E_t^{before}
=
[
E_{cls},
E_{loc},
E_{both},
E_{bkg},
E_{miss}
].
$$

재학습 후:

$$
\mathbf E_t^{after}.
$$

각 error type reduction:

$$
\boxed{
\Delta_m
=
\frac{
E_m^{before}-E_m^{after}
}{
E_m^{before}
}
}
$$

를 구해.

그리고 Random / Entropy / PAL을 비교한다.

예:

| Method | $\Delta_{\rm cls}$ ↑ | $\Delta_{\rm loc}$ ↑ | $\Delta_{\rm miss}$ ↑ | mAP |
|---|---:|---:|---:|---:|
| Random | | | | |
| Entropy | | | | |
| PAL | | | | |

만약 PAL이

$$
\Delta_{cls}\text{ 높음}
$$

인데

$$
\Delta_{loc},\Delta_{miss}\approx\text{Random}
$$

이면 **이게 가장 강한 문제 정의**가 돼.

---

# 8. TIDE error contribution 변화도 같이 보면 좋음

단순 error count와 AP impact가 항상 같은 건 아니니까, test set에 대해 TIDE-style error decomposition도 같이 보는 게 좋아.

예를 들어 PAL 학습 전후에

- Classification
- Localization
- Both
- Duplicate
- Background
- Missed

각 error가 AP에 미치는 영향을 비교하는 식.

그러면 단순히 “loc error가 몇 개 줄었다”가 아니라

> PAL-acquired supervision mainly reduces classification-related performance loss while leaving localization/missed-object loss relatively unchanged.

같은 더 강한 해석이 가능해.

---

# 9. PAL-full과 LIUS-only는 반드시 분리해서 봐야 함

이거 중요해.

PAL full은 LIUS만 사용하는 게 아니야.

GUIDE에

- CWIE
- RCDI
- RCSP

가 들어간다. PAL 논문도 LIUS-only ablation을 별도로 수행하고 있으며, GUIDE가 특히 early rounds에서 성능을 보완한다고 보고한다. `[CVPR]-[2026]-[Portable Active Learning for Object Detection].pdf`

따라서

> “TP/FP modeling의 한계”

를 분석하려면 **PAL-LIUS only를 먼저 분석**해야 해.

그리고 그다음 PAL-full을 분석해서

> GUIDE가 그 한계를 얼마나 보완하는가?

를 보면 돼.

분석 순서는:

$$
\boxed{
\text{PAL-LIUS}
\rightarrow
\text{PAL-full}
}
$$

이 맞아.

그렇지 않으면 localization/miss coverage 부족이 LIUS 때문인지 GUIDE 때문인지 구분이 안 돼.

---

# 10. 특히 같은 detector checkpoint에서 비교해야 함

이것도 굉장히 중요해.

Round 3의 PAL detector와 EUA detector는 이미 서로 다른 데이터로 학습됐기 때문에,

$$
\mathcal U_t
$$

와 detector 자체가 달라져 있어.

그래서 selection characteristic을 비교할 때는 가능하면 **동일한 detector + 동일한 unlabeled pool**을 사용해야 해.

가장 깨끗한 것은 round 1 acquisition이야.

모든 방법이 동일한 initial labeled set에서 시작하므로,

$$
f_{\theta_1}
$$

이 같아.

이 checkpoint 하나에서

- PAL LIUS
- PAL full
- ECPAL component scores

를 모두 계산해서 selection을 사후 분석하면 된다.

그 다음 round들도 필요하면 **공통 checkpoint를 따로 만들어 offline acquisition analysis**를 하는 게 제일 정확하다.

---

# 내가 우선순위를 정하면

처음부터 재학습 실험까지 다 하지 말고, 아래 **4개를 먼저** 할 것 같아.

### Analysis 1 - Error-type sensitivity

$$
\boxed{
\text{PAL LIUS distribution by TIDE error type}
}
$$

→ PAL uncertainty가 어떤 error에 민감한가?

### Analysis 2 - Error-type retrieval

$$
\boxed{
\text{error-type recall vs acquisition budget}
}
$$

→ PAL high-uncertainty sampling이 어떤 error를 실제로 회수하는가?

### Analysis 3 - Missed-object blind spot

$$
\boxed{
\text{PAL score/rank vs missed-object count}
}
$$

→ predicted instance가 없는 error를 PAL이 놓치는가?

### Analysis 4 - Selected batch composition

$$
\boxed{
\text{PAL-selected images' error-type enrichment}
}
$$

→ 실제 annotation batch가 어떤 failure mode에 편향되는가?

이 네 개는 **새 AL training을 거의 하지 않고도** 할 수 있어.

여기서 정말 명확한 imbalance가 보이면 그 다음에

### Analysis 5 - Error-specific learning gain

을 돌리는 거야.

---

# 가장 이상적인 결과 시나리오

만약 분석 결과가 예를 들어 이렇게 나온다고 해보자.

$$
\boxed{
\begin{aligned}
&\text{PAL LIUS strongly captures classification ambiguity},\\
&\text{but weakly captures localization errors},\\
&\text{and has no direct mechanism for missed objects}.
\end{aligned}
}
$$

그리고 실제 PAL-selected data로 학습했더니

$$
\Delta_{\rm cls}
>
\Delta_{\rm loc},
\Delta_{\rm miss}
$$

가 나온다면, 네 연구의 motivation은 상당히 강해져.

그때 ECPAL은 단순히 PAL의 classifier feature를 바꾼 게 아니라,

$$
\boxed{
\text{binary TP/FP uncertainty}
}
$$

를

$$
\boxed{
\text{classification correctness uncertainty}
+
\text{localization correctness uncertainty}
+
\text{missed-object count uncertainty}
}
$$

로 **decompose한 방법**이라고 명확하게 자리 잡을 수 있어.

다만 반대로 PAL이 세 error type을 모두 잘 회수하고 모두 잘 줄인다는 결과가 나오면, **“PAL의 error-type blind spot을 해결한다”는 서사는 버려야 해.** 그 경우에는 error-type decomposition 자체의 interpretability나 controllability, 혹은 더 정확한 uncertainty modeling 쪽에서 novelty를 찾아야 하고.

그래서 지금 단계에서는 **PAL의 약점을 가정하는 게 아니라, TP/FP binary uncertainty가 어떤 failure mode를 실제로 학습시키는지 해부하는 분석**을 먼저 하는 게 가장 중요해.



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)