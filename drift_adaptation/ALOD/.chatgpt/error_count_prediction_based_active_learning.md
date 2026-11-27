# Error-Count Prediction-Based Active Learning for Object Detection

Active learning round \(t\)에서 현재 labeled set과 unlabeled pool을 각각

\[
\mathcal L_t
=
\left\{
(x_n,\mathcal G_n)
\right\},
\qquad
\mathcal U_t
=
\left\{
x_n
\right\}
\]

라고 한다.

한 round에서 annotation할 이미지 수를 \(b\), 1단계에서 구성할 candidate image 수의 배율을 \(\delta>1\)이라고 한다. 따라서 candidate image 수는

\[
\left\lceil \delta b \right\rceil
\]

이다.

현재 detector는 labeled set으로 학습한다.

\[
\boxed{
\theta_t
=
\operatorname{TrainDetector}(\mathcal L_t)
}
\]

이후 현재 detector의 출력으로 다음 네 predictor를 학습한다.

- Foreground Detection Predictor (FDP)
- Classification-Error Count Predictor (CECP)
- Localization-Error Count Predictor (LECP)
- Missed-Object Count Predictor (MOCP)

FDP는 각 final detection이 실제 객체와 관련된 foreground detection일 확률을 예측한다. CECP와 LECP는 foreground detection이라는 조건에서 각각 classification error와 localization error가 발생할 확률을 예측한다. MOCP는 이미지별 missed-object count를 직접 예측한다.

---

# 1. Detector outputs

이미지 \(x\)에 대해 detector가 final selection 이전에 생성한 raw detections를

\[
\widetilde{\mathcal D}_t(x)
=
\left\{
\tilde d_k
=
(\tilde b_k,\tilde{\mathbf p}_k)
\right\}_{k=1}^{M_x}
\]

라고 한다.

여기서

- \(\tilde b_k\): raw bounding box
- \(\tilde{\mathbf p}_k\in[0,1]^C\): raw class-probability vector

이다.

Raw detection \(k\)의 predicted class와 maximum probability는

\[
\tilde c_k
=
\arg\max_c \tilde p_{k,c},
\]

\[
\tilde p_k^{\max}
=
\max_c \tilde p_{k,c}
\]

로 정의한다.

Detector의 final detections는

\[
\widehat{\mathcal D}_t(x)
=
\left\{
\hat d_i
=
(\hat b_i,\hat{\mathbf p}_i)
\right\}_{i=1}^{N_x}
\]

로 정의한다.

Final detection \(i\)의 predicted class와 maximum probability는

\[
\hat c_i
=
\arg\max_c \hat p_{i,c},
\]

\[
\boxed{
\hat p_i^{\max}
=
\max_c \hat p_{i,c}
}
\]

이다.

---

# 2. Support set construction

각 raw detection을 가장 많이 겹치는 final detection에 연결한다.

\[
a(k)
=
\arg\max_{1\le j\le N_x}
\operatorname{IoU}
(\tilde b_k,\hat b_j).
\]

Final detection \(i\)에 할당된 raw detection 집합은

\[
\mathcal A_i(x)
=
\left\{
\tilde d_k:
a(k)=i,\;
\operatorname{IoU}(\tilde b_k,\hat b_i)>0
\right\}
\]

이다.

일반적인 NMS 기반 detector에서는 final detection의 box가 raw detection 중 하나와 동일하다. 이 raw detection을 support에 포함하면 final detection이 자기 자신에 의해 지지되는 문제가 생긴다.

따라서 final box와 동일한 box를 갖는 raw detections를

\[
\boxed{
\mathcal Q_i(x)
=
\left\{
\tilde d_k\in\mathcal A_i(x):
\tilde b_k=\hat b_i
\right\}
}
\]

로 정의한다.

Final detection \(i\)의 support set은

\[
\boxed{
\mathcal H_i(x)
=
\mathcal A_i(x)
\setminus
\mathcal Q_i(x)
}
\]

로 정의한다.

즉, final box와 **정확히 동일한 box만** support set에서 제외한다. Final box와 IoU가 가장 크더라도 box가 동일하지 않다면 제거하지 않는다.

만약 final box와 동일한 raw box가 존재하지 않는다면

\[
\mathcal Q_i(x)=\varnothing
\]

이므로

\[
\mathcal H_i(x)=\mathcal A_i(x)
\]

가 된다.

실제 구현에서 box 좌표가 동일한 coordinate representation으로 저장된다면 좌표의 직접 equality를 사용한다. Floating-point 연산 때문에 아주 작은 수치 차이가 발생할 수 있다면 고정된 numerical tolerance를 사용할 수 있지만, 별도의 IoU threshold를 추가하지는 않는다.

---

# 3. Detection-wise common features

FDP, CECP, LECP에서 사용하는 네 개의 공통 feature를 다음과 같이 정의한다.

\[
\boxed{
\mathbf z_i^{\mathrm{common}}
=
\begin{bmatrix}
\hat p_i^{\max}\\
A_i^{\mathrm{cls}}\\
n_i^{\mathrm{sup}}\\
\mu_i^{\mathrm{IoU}}
\end{bmatrix}
}
\]

모든 feature는 값이 클수록 final detection을 지지하는 evidence가 강하다는 **긍정 기준**으로 정의한다.

FDP는 네 feature를 모두 사용한다.

CECP는 classification과 관련된 앞의 두 feature를 사용한다.

\[
\boxed{
\mathbf z_i^{\mathrm{cls}}
=
\begin{bmatrix}
\hat p_i^{\max}\\
A_i^{\mathrm{cls}}
\end{bmatrix}
}
\]

LECP는 localization과 관련된 뒤의 두 feature를 사용한다.

\[
\boxed{
\mathbf z_i^{\mathrm{loc}}
=
\begin{bmatrix}
n_i^{\mathrm{sup}}\\
\mu_i^{\mathrm{IoU}}
\end{bmatrix}
}
\]

따라서

\[
\boxed{
\mathbf z_i^{\mathrm{common}}
=
\begin{bmatrix}
\mathbf z_i^{\mathrm{cls}}\\
\mathbf z_i^{\mathrm{loc}}
\end{bmatrix}
}
\]

이다.

---

## 3.1 Maximum probability

Final detection의 maximum class probability는

\[
\boxed{
\hat p_i^{\max}
=
\max_c \hat p_{i,c}
}
\]

이다.

값이 클수록 detector가 final class assignment를 강하게 지지한다.

---

## 3.2 Support class agreement

Support-set raw detections 중 final detection과 동일한 class를 예측하는 비율을 계산한다.

\[
\boxed{
A_i^{\mathrm{cls}}
=
\begin{cases}
\displaystyle
\frac{1}{|\mathcal H_i|}
\sum_{\tilde d_k\in\mathcal H_i}
\mathbb 1
\left[
\tilde c_k=\hat c_i
\right],
&
|\mathcal H_i|>0,\\[4mm]
0,
&
|\mathcal H_i|=0.
\end{cases}
}
\]

- 모든 support detections가 final class와 같으면

\[
A_i^{\mathrm{cls}}=1
\]

- Support detections가 final class에 동의하지 않으면 값이 작아진다.
- Support set이 비어 있으면 독립적으로 final class를 지지하는 raw detection이 없으므로

\[
A_i^{\mathrm{cls}}=0
\]

으로 둔다.

---

## 3.3 Support count

\[
\boxed{
n_i^{\mathrm{sup}}
=
|\mathcal H_i|
}
\]

이다.

이는 final box와 동일한 raw detection을 제외하고, final detection을 추가로 지지하는 raw detections의 수다.

---

## 3.4 Support IoU mean

Support raw boxes와 final box 사이의 평균 IoU를 계산한다.

\[
\boxed{
\mu_i^{\mathrm{IoU}}
=
\begin{cases}
\displaystyle
\frac{1}{n_i^{\mathrm{sup}}}
\sum_{\tilde d_k\in\mathcal H_i}
\operatorname{IoU}
(\tilde b_k,\hat b_i),
&
n_i^{\mathrm{sup}}>0,\\[4mm]
0,
&
n_i^{\mathrm{sup}}=0.
\end{cases}
}
\]

값이 클수록 주변 raw boxes가 final box의 위치와 크기를 일관되게 지지한다.

---

# 4. TIDE-based detection labels

Labeled image \(x\)의 GT objects를

\[
\mathcal G(x)
=
\left\{
g_j=(b_j,c_j)
\right\}_{j=1}^{G_x}
\]

라고 한다.

Final detection \(i\)에 대해 같은 class GT와 다른 class GT에 대한 최대 IoU를 각각 정의한다.

\[
\boxed{
u_i^{\mathrm{same}}
=
\max_{j:c_j=\hat c_i}
\operatorname{IoU}
(\hat b_i,b_j)
}
\]

\[
\boxed{
u_i^{\mathrm{diff}}
=
\max_{j:c_j\neq\hat c_i}
\operatorname{IoU}
(\hat b_i,b_j)
}
\]

해당 조건을 만족하는 GT가 없으면 최대값은 0으로 둔다.

전체 maximum overlap은

\[
\boxed{
u_i^{\max}
=
\max
\left(
u_i^{\mathrm{same}},
u_i^{\mathrm{diff}}
\right)
}
\]

이다.

TIDE는 foreground IoU threshold \(t_f\)와 background IoU threshold \(t_b\)를 사용하여 classification, localization, both classification and localization, duplicate, background, missed GT error를 구분한다. TIDE의 기본 설정에서는 \(t_f=0.5\), \(t_b=0.1\)을 사용한다. fileciteturn20file0

우리 방법에서는

\[
t_f=\tau,
\qquad
t_b=0.1
\]

로 둔다. 여기서 \(\tau\)는 실험에서 사용하는 단일 foreground IoU 기준이다.

---

## 4.1 Foreground label

Final detection이 어떤 GT와라도 \(t_b\) 이상의 overlap을 가지면 foreground-related detection으로 정의한다.

\[
\boxed{
y_i^{\mathrm{fg}}
=
\mathbb 1
\left[
u_i^{\max}\ge t_b
\right]
}
\]

반대로

\[
u_i^{\max}<t_b
\]

이면 background detection이다.

\[
y_i^{\mathrm{fg}}=0.
\]

Foreground-related detection index set은

\[
\boxed{
\mathcal I_{\mathrm{fg}}(x)
=
\left\{
i:
u_i^{\max}\ge t_b
\right\}
}
\]

이다.

Classification 및 localization labels는

\[
i\in\mathcal I_{\mathrm{fg}}(x)
\]

인 detection에서만 정의한다.

---

## 4.2 Classification-error label

Foreground-related detection \(i\)에 대해 다른 class GT와의 maximum overlap이 같은 class GT와의 maximum overlap보다 크면 classification error가 있다고 정의한다.

\[
\boxed{
y_i^{\mathrm{cls}}
=
\mathbb 1
\left[
u_i^{\mathrm{diff}}
>
u_i^{\mathrm{same}}
\right],
\qquad
i\in\mathcal I_{\mathrm{fg}}(x)
}
\]

두 maximum IoU가 같으면 같은 class GT를 우선한다.

순수 classification error는

\[
u_i^{\mathrm{diff}}\ge t_f,
\qquad
u_i^{\mathrm{diff}}>u_i^{\mathrm{same}}
\]

인 경우다.

이 경우

\[
y_i^{\mathrm{cls}}=1,
\qquad
y_i^{\mathrm{loc}}=0.
\]

Classification과 localization이 모두 잘못된 경우는

\[
t_b
\le
u_i^{\mathrm{diff}}
<
t_f,
\qquad
u_i^{\mathrm{diff}}>u_i^{\mathrm{same}}
\]

이다.

이 경우

\[
y_i^{\mathrm{cls}}=1,
\qquad
y_i^{\mathrm{loc}}=1.
\]

---

## 4.3 Localization-error label

Foreground-related detection \(i\)의 maximum GT overlap이 \(t_f\)에 미달하면 localization error가 있다고 정의한다.

\[
\boxed{
y_i^{\mathrm{loc}}
=
\mathbb 1
\left[
u_i^{\max}<t_f
\right],
\qquad
i\in\mathcal I_{\mathrm{fg}}(x)
}
\]

이미 foreground detection에 대해서만 정의하므로 localization error의 실제 범위는

\[
t_b
\le
u_i^{\max}
<
t_f
\]

이다.

Class는 맞지만 위치가 부정확한 경우는

\[
t_b
\le
u_i^{\mathrm{same}}
<
t_f,
\qquad
u_i^{\mathrm{same}}
\ge
u_i^{\mathrm{diff}}
\]

이다.

이 경우

\[
y_i^{\mathrm{cls}}=0,
\qquad
y_i^{\mathrm{loc}}=1.
\]

Class와 위치가 모두 잘못된 경우는

\[
t_b
\le
u_i^{\mathrm{diff}}
<
t_f,
\qquad
u_i^{\mathrm{diff}}
>
u_i^{\mathrm{same}}
\]

이므로

\[
y_i^{\mathrm{cls}}=1,
\qquad
y_i^{\mathrm{loc}}=1.
\]

---

## 4.4 Correct and duplicate detections

같은 class GT와 충분히 겹치는 detection은

\[
u_i^{\mathrm{same}}\ge t_f,
\qquad
u_i^{\mathrm{same}}\ge u_i^{\mathrm{diff}}
\]

를 만족한다.

동일 class detections를 confidence 내림차순으로 처리할 때 해당 GT와 처음 매칭되는 detection은 correct detection이고, 동일 GT가 더 높은-confidence detection에 이미 매칭되어 있으면 duplicate detection이다.

두 경우 모두

\[
\boxed{
y_i^{\mathrm{cls}}=0,
\qquad
y_i^{\mathrm{loc}}=0
}
\]

이다.

Duplicate detections도 CECP와 LECP의 negative samples로 사용한다.

---

## 4.5 Error-type mapping

| Detection type | 정의되는 범위 | \(y_i^{\mathrm{fg}}\) | \(y_i^{\mathrm{cls}}\) | \(y_i^{\mathrm{loc}}\) |
|---|---|:---:|:---:|:---:|
| Correct | \(u_i^{\mathrm{same}}\ge t_f,\ u_i^{\mathrm{same}}\ge u_i^{\mathrm{diff}}\), GT와 최초 매칭 | 1 | 0 | 0 |
| Duplicate | \(u_i^{\mathrm{same}}\ge t_f,\ u_i^{\mathrm{same}}\ge u_i^{\mathrm{diff}}\), GT가 이미 매칭됨 | 1 | 0 | 0 |
| Classification | \(u_i^{\mathrm{diff}}\ge t_f,\ u_i^{\mathrm{diff}}>u_i^{\mathrm{same}}\) | 1 | 1 | 0 |
| Localization | \(t_b\le u_i^{\mathrm{same}}<t_f,\ u_i^{\mathrm{same}}\ge u_i^{\mathrm{diff}}\) | 1 | 0 | 1 |
| Classification + Localization | \(t_b\le u_i^{\mathrm{diff}}<t_f,\ u_i^{\mathrm{diff}}>u_i^{\mathrm{same}}\) | 1 | 1 | 1 |
| Background | \(u_i^{\max}<t_b\) | 0 | 정의하지 않음 | 정의하지 않음 |

---

# 5. Foreground Detection Predictor

Foreground Detection Predictor는 final detection이 실제 GT object와 관련된 detection일 확률을 예측한다.

FDP는 네 개의 공통 feature를 모두 사용한다.

\[
\boxed{
\mathbf z_i^{\mathrm{fg}}
=
\begin{bmatrix}
\hat p_i^{\max}\\
A_i^{\mathrm{cls}}\\
n_i^{\mathrm{sup}}\\
\mu_i^{\mathrm{IoU}}
\end{bmatrix}
}
\]

Foreground probability는

\[
q_i^{\mathrm{fg}}
=
P
\left(
y_i^{\mathrm{fg}}=1
\mid
\mathbf z_i^{\mathrm{fg}}
\right)
\]

로 정의한다.

Logistic regression을 사용하면

\[
\boxed{
q_i^{\mathrm{fg}}
=
\sigma
\left(
\eta_0
+
\eta_1\hat p_i^{\max}
+
\eta_2A_i^{\mathrm{cls}}
+
\eta_3n_i^{\mathrm{sup}}
+
\eta_4\mu_i^{\mathrm{IoU}}
\right)
}
\]

이다.

여기서

\[
\sigma(a)
=
\frac{1}{1+\exp(-a)}
\]

이다.

FDP는 labeled images의 모든 final detections를 사용하여 학습한다.

\[
\boxed{
\mathcal L_{\mathrm{FDP}}
=
-\sum_{x\in\mathcal L_t}
\sum_{i=1}^{N_x}
\left[
y_i^{\mathrm{fg}}
\log q_i^{\mathrm{fg}}
+
(1-y_i^{\mathrm{fg}})
\log
\left(
1-q_i^{\mathrm{fg}}
\right)
\right]
+
\lambda_{\mathrm{fg}}
\|\boldsymbol\eta\|_2^2
}
\]

FDP의 출력은 error-count profile의 별도 축으로 사용하지 않는다. CECP와 LECP의 conditional error probabilities를 실제 classification/localization error probabilities로 변환하기 위한 foreground gate로 사용한다.

---

# 6. Conditional Classification-Error Count Predictor

CECP는 detection이 foreground object와 관련되어 있다는 조건에서 class assignment가 잘못되었을 확률을 예측한다.

\[
q_i^{\mathrm{cls}\mid\mathrm{fg}}
=
P
\left(
y_i^{\mathrm{cls}}=1
\mid
y_i^{\mathrm{fg}}=1,
\mathbf z_i^{\mathrm{cls}}
\right).
\]

CECP는 공통 feature 중 classification과 관련된 두 feature를 사용한다.

\[
\boxed{
\mathbf z_i^{\mathrm{cls}}
=
\begin{bmatrix}
\hat p_i^{\max}\\
A_i^{\mathrm{cls}}
\end{bmatrix}
}
\]

Logistic regression은

\[
\boxed{
q_i^{\mathrm{cls}\mid\mathrm{fg}}
=
\sigma
\left(
\beta_0^{\mathrm{cls}}
+
\beta_1^{\mathrm{cls}}\hat p_i^{\max}
+
\beta_2^{\mathrm{cls}}A_i^{\mathrm{cls}}
\right)
}
\]

로 정의한다.

CECP는 labeled images의 foreground-related detections만 사용해 학습한다.

\[
\boxed{
\mathcal L_{\mathrm{CECP}}
=
-\sum_{x\in\mathcal L_t}
\sum_{i\in\mathcal I_{\mathrm{fg}}(x)}
\left[
y_i^{\mathrm{cls}}
\log q_i^{\mathrm{cls}\mid\mathrm{fg}}
+
(1-y_i^{\mathrm{cls}})
\log
\left(
1-q_i^{\mathrm{cls}\mid\mathrm{fg}}
\right)
\right]
+
\lambda_{\mathrm{cls}}
\|\boldsymbol\beta^{\mathrm{cls}}\|_2^2
}
\]

Positive samples는 classification 및 classification+localization detections다.

Negative samples는 correct, duplicate, localization-only detections다.

---

# 7. Conditional Localization-Error Count Predictor

LECP는 detection이 foreground object와 관련되어 있다는 조건에서 box localization이 부정확할 확률을 예측한다.

\[
q_i^{\mathrm{loc}\mid\mathrm{fg}}
=
P
\left(
y_i^{\mathrm{loc}}=1
\mid
y_i^{\mathrm{fg}}=1,
\mathbf z_i^{\mathrm{loc}}
\right).
\]

LECP는 공통 feature 중 localization과 관련된 두 feature를 사용한다.

\[
\boxed{
\mathbf z_i^{\mathrm{loc}}
=
\begin{bmatrix}
n_i^{\mathrm{sup}}\\
\mu_i^{\mathrm{IoU}}
\end{bmatrix}
}
\]

Logistic regression은

\[
\boxed{
q_i^{\mathrm{loc}\mid\mathrm{fg}}
=
\sigma
\left(
\beta_0^{\mathrm{loc}}
+
\beta_1^{\mathrm{loc}}n_i^{\mathrm{sup}}
+
\beta_2^{\mathrm{loc}}\mu_i^{\mathrm{IoU}}
\right)
}
\]

로 정의한다.

LECP도 labeled images의 foreground-related detections만 사용해 학습한다.

\[
\boxed{
\mathcal L_{\mathrm{LECP}}
=
-\sum_{x\in\mathcal L_t}
\sum_{i\in\mathcal I_{\mathrm{fg}}(x)}
\left[
y_i^{\mathrm{loc}}
\log q_i^{\mathrm{loc}\mid\mathrm{fg}}
+
(1-y_i^{\mathrm{loc}})
\log
\left(
1-q_i^{\mathrm{loc}\mid\mathrm{fg}}
\right)
\right]
+
\lambda_{\mathrm{loc}}
\|\boldsymbol\beta^{\mathrm{loc}}\|_2^2
}
\]

Positive samples는 localization 및 classification+localization detections다.

Negative samples는 correct, duplicate, classification-only detections다.

---

# 8. Unconditional error probabilities

Unlabeled image에서는 final detection의 실제 foreground 여부를 알 수 없다. 따라서 FDP의 foreground probability와 conditional error probability를 곱한다.

Classification error probability는

\[
\boxed{
q_i^{\mathrm{cls}}
=
q_i^{\mathrm{fg}}
q_i^{\mathrm{cls}\mid\mathrm{fg}}
}
\]

이다.

이는

\[
P(\mathrm{ClsError})
=
P(\mathrm{Foreground})
P(\mathrm{ClsError}\mid\mathrm{Foreground})
\]

에 해당한다.

Localization error probability는

\[
\boxed{
q_i^{\mathrm{loc}}
=
q_i^{\mathrm{fg}}
q_i^{\mathrm{loc}\mid\mathrm{fg}}
}
\]

이다.

이는

\[
P(\mathrm{LocError})
=
P(\mathrm{Foreground})
P(\mathrm{LocError}\mid\mathrm{Foreground})
\]

에 해당한다.

FDP가 background일 가능성이 높다고 판단한 detection은

\[
q_i^{\mathrm{fg}}\approx0
\]

이므로 classification 및 localization error counts에 거의 기여하지 않는다.

---

# 9. Expected classification- and localization-error counts

이미지 \(x\)의 expected classification-error count는

\[
\boxed{
\widehat N_{\mathrm{cls}}(x)
=
\sum_{i=1}^{N_x}
q_i^{\mathrm{cls}}
}
\]

이므로

\[
\boxed{
\widehat N_{\mathrm{cls}}(x)
=
\sum_{i=1}^{N_x}
q_i^{\mathrm{fg}}
q_i^{\mathrm{cls}\mid\mathrm{fg}}
}
\]

이다.

이미지 \(x\)의 expected localization-error count는

\[
\boxed{
\widehat N_{\mathrm{loc}}(x)
=
\sum_{i=1}^{N_x}
q_i^{\mathrm{loc}}
}
\]

이므로

\[
\boxed{
\widehat N_{\mathrm{loc}}(x)
=
\sum_{i=1}^{N_x}
q_i^{\mathrm{fg}}
q_i^{\mathrm{loc}\mid\mathrm{fg}}
}
\]

이다.

Classification+localization detection은 CECP와 LECP에서 모두 높은 조건부 error probability를 가질 수 있으므로 두 expected count에 동시에 기여한다.

---

# 10. Missed-object labels

각 final detection \(i\)와 가장 많이 겹치는 GT를

\[
\boxed{
g_i^*
=
\arg\max_{g_j\in\mathcal G(x)}
\operatorname{IoU}
(\hat b_i,b_j)
}
\]

라고 한다.

전체 maximum overlap이 \(t_b\) 이상인 detection만 해당 GT를 설명하는 것으로 본다.

GT \(g_j\)를 설명하는 final detection 집합은

\[
\boxed{
\mathcal E_j(x)
=
\left\{
i:
g_i^*=g_j,\;
u_i^{\max}\ge t_b
\right\}
}
\]

이다.

이 집합에는 다음 detections가 모두 포함된다.

\[
\mathrm{Correct},
\quad
\mathrm{Duplicate},
\quad
\mathrm{Classification},
\quad
\mathrm{Localization},
\quad
\mathrm{Classification+Localization}.
\]

Background detection은 어떤 GT도 설명하지 않는다.

GT \(g_j\)의 missed-object label은

\[
\boxed{
y_j^{\mathrm{miss}}
=
\mathbb 1
\left[
\mathcal E_j(x)=\varnothing
\right]
}
\]

이다.

이미지 \(x\)의 실제 missed-object count는

\[
\boxed{
N_{\mathrm{miss}}(x)
=
\sum_{j=1}^{G_x}
y_j^{\mathrm{miss}}
}
\]

이다.

---

# 11. Missed-Object Count Predictor

MOCP는 image-wise count regressor다. Final detections로 충분히 설명되지 않은 raw-detection evidence를 사용하여 missed-object count를 예측한다.

## 11.1 Raw-detection residual weight

Raw detection \(k\)가 final detections로 설명되는 정도를

\[
\rho_k
=
\begin{cases}
\displaystyle
\max_{1\le i\le N_x}
\operatorname{IoU}
(\tilde b_k,\hat b_i),
&
N_x>0,\\[3mm]
0,
&
N_x=0
\end{cases}
\]

로 정의한다.

Residual weight는

\[
\boxed{
r_k
=
1-\rho_k
}
\]

이다.

Raw detection과 동일하거나 매우 유사한 final detection이 존재하면

\[
\rho_k\approx1,
\qquad
r_k\approx0
\]

이다.

Raw detection이 어떤 final detection에도 반영되지 않았다면

\[
\rho_k\approx0,
\qquad
r_k\approx1
\]

이다.

---

## 11.2 Residual amount

\[
\boxed{
R_{\mathrm{amt}}(x)
=
\sum_{k=1}^{M_x} r_k
=
\sum_{k=1}^{M_x}
(1-\rho_k)
}
\]

이다.

---

## 11.3 Residual probability mean

\[
\boxed{
R_{\mathrm{prob}}(x)
=
\frac{
\sum_{k=1}^{M_x}
r_k\tilde p_k^{\max}
}{
\sum_{k=1}^{M_x}r_k+\epsilon
}
}
\]

이다.

---

## 11.4 MOCP feature and output

\[
\boxed{
\mathbf z^{\mathrm{miss}}(x)
=
\begin{bmatrix}
R_{\mathrm{amt}}(x)\\
R_{\mathrm{prob}}(x)
\end{bmatrix}
}
\]

이다.

Poisson regression을 사용하면

\[
N_{\mathrm{miss}}(x)
\mid
\mathbf z^{\mathrm{miss}}(x)
\sim
\operatorname{Poisson}(\lambda_x)
\]

이고,

\[
\boxed{
\lambda_x
=
\exp
\left(
\gamma_0
+
\gamma_1R_{\mathrm{amt}}(x)
+
\gamma_2R_{\mathrm{prob}}(x)
\right)
}
\]

이다.

따라서

\[
\boxed{
\widehat N_{\mathrm{miss}}(x)
=
\lambda_x
}
\]

이다.

Poisson negative log-likelihood는 상수항을 제외하면

\[
\boxed{
\mathcal L_{\mathrm{MOCP}}
=
\sum_{x\in\mathcal L_t}
\left[
\lambda_x
-
N_{\mathrm{miss}}(x)\log\lambda_x
\right]
+
\lambda_{\mathrm{miss}}
\|\boldsymbol\gamma\|_2^2
}
\]

이다.

---

# 12. Predictor summary

| Predictor | Input features | Training samples | Output |
|---|---|---|---|
| FDP | \([\hat p_i^{\max},A_i^{\mathrm{cls}},n_i^{\mathrm{sup}},\mu_i^{\mathrm{IoU}}]\) | 모든 final detections | \(q_i^{\mathrm{fg}}\) |
| Conditional CECP | \([\hat p_i^{\max},A_i^{\mathrm{cls}}]\) | Foreground-related detections | \(q_i^{\mathrm{cls}\mid\mathrm{fg}}\) |
| Conditional LECP | \([n_i^{\mathrm{sup}},\mu_i^{\mathrm{IoU}}]\) | Foreground-related detections | \(q_i^{\mathrm{loc}\mid\mathrm{fg}}\) |
| MOCP | \([R_{\mathrm{amt}},R_{\mathrm{prob}}]\) | Labeled images | \(\widehat N_{\mathrm{miss}}\) |

---

# 13. Predicted Detection Error-Count Profile

Unlabeled image \(x\)의 predicted classification-error count는

\[
\widehat N_{\mathrm{cls},t}(x)
=
\sum_{i=1}^{N_x}
q_{i,t}^{\mathrm{fg}}
q_{i,t}^{\mathrm{cls}\mid\mathrm{fg}}
\]

이다.

Predicted localization-error count는

\[
\widehat N_{\mathrm{loc},t}(x)
=
\sum_{i=1}^{N_x}
q_{i,t}^{\mathrm{fg}}
q_{i,t}^{\mathrm{loc}\mid\mathrm{fg}}
\]

이다.

Predicted missed-object count는

\[
\widehat N_{\mathrm{miss},t}(x)
=
\lambda_{x,t}
\]

이다.

따라서 unweighted Detection Error-Count Profile은

\[
\boxed{
\widehat{\mathbf e}_t(x)
=
\begin{bmatrix}
\widehat N_{\mathrm{cls},t}(x)\\
\widehat N_{\mathrm{loc},t}(x)\\
\widehat N_{\mathrm{miss},t}(x)
\end{bmatrix}
}
\]

이다.

이를 전개하면

\[
\boxed{
\widehat{\mathbf e}_t(x)
=
\begin{bmatrix}
\displaystyle
\sum_{i=1}^{N_x}
q_{i,t}^{\mathrm{fg}}
q_{i,t}^{\mathrm{cls}\mid\mathrm{fg}}
\\[6mm]
\displaystyle
\sum_{i=1}^{N_x}
q_{i,t}^{\mathrm{fg}}
q_{i,t}^{\mathrm{loc}\mid\mathrm{fg}}
\\[6mm]
\widehat N_{\mathrm{miss},t}(x)
\end{bmatrix}
}
\]

이다.

---

# 14. Error-count scale weights

Classification, localization, missed-object counts는 서로 다른 평균 규모를 가질 수 있다. 예를 들어 한 이미지에서 detection-wise classification error와 localization error는 여러 개 발생할 수 있지만, missed-object count는 상대적으로 작은 범위를 가질 수 있다.

따라서 각 error count의 scale을 조정하기 위한 positive weight를 사용한다.

\[
\boxed{
\mathbf w_t
=
\begin{bmatrix}
w_{\mathrm{cls},t}\\
w_{\mathrm{loc},t}\\
w_{\mathrm{miss},t}
\end{bmatrix},
\qquad
w_{m,t}>0
}
\]

가중치 행렬은

\[
\boxed{
\mathbf W_t
=
\operatorname{diag}
\left(
w_{\mathrm{cls},t},
w_{\mathrm{loc},t},
w_{\mathrm{miss},t}
\right)
}
\]

이다.

Weighted error-count profile은

\[
\boxed{
\widehat{\mathbf e}_t^{\,w}(x)
=
\mathbf W_t
\widehat{\mathbf e}_t(x)
}
\]

이다.

즉,

\[
\boxed{
\widehat{\mathbf e}_t^{\,w}(x)
=
\begin{bmatrix}
w_{\mathrm{cls},t}\widehat N_{\mathrm{cls},t}(x)\\
w_{\mathrm{loc},t}\widehat N_{\mathrm{loc},t}(x)\\
w_{\mathrm{miss},t}\widehat N_{\mathrm{miss},t}(x)
\end{bmatrix}
}
\]

이다.

## Round-wise weight estimation

각 error type의 평균 scale을 labeled set에서 계산할 수 있다.

\[
\bar N_{m,t}
=
\frac{1}{|\mathcal L_t|}
\sum_{x\in\mathcal L_t}
N_m(x),
\qquad
m\in
\{\mathrm{cls},\mathrm{loc},\mathrm{miss}\}.
\]

Inverse-scale weight는

\[
\tilde w_{m,t}
=
\frac{1}{
\bar N_{m,t}+\epsilon_w
}
\]

로 정의할 수 있다.

가중치의 전체 크기가 round마다 크게 바뀌지 않도록 평균이 1이 되게 정규화하면

\[
\boxed{
w_{m,t}
=
\frac{
3\tilde w_{m,t}
}{
\tilde w_{\mathrm{cls},t}
+
\tilde w_{\mathrm{loc},t}
+
\tilde w_{\mathrm{miss},t}
}
}
\]

이다.

이에 따라

\[
\frac{
w_{\mathrm{cls},t}
+
w_{\mathrm{loc},t}
+
w_{\mathrm{miss},t}
}{3}
=
1
\]

이 된다.

이 방식은 평균적으로 큰 error count에는 작은 weight를, 평균적으로 작은 error count에는 큰 weight를 부여하여 세 error type의 scale 차이를 보정한다.

가중치는 실험 전체에서 고정할 수도 있지만, round가 진행되면서 detector의 error distribution이 달라질 수 있으므로 labeled set을 이용한 round-wise estimation을 기본 방식으로 둘 수 있다.

---

# 15. Weighted Error-Count Amount

이미지 \(x\)에서 예상되는 전체 error-learning signal의 양을 Weighted Error-Count Amount로 정의한다.

\[
\boxed{
\operatorname{ECA}_t(x)
=
\left\|
\widehat{\mathbf e}_t^{\,w}(x)
\right\|_1
}
\]

즉,

\[
\boxed{
\operatorname{ECA}_t(x)
=
w_{\mathrm{cls},t}
\widehat N_{\mathrm{cls},t}(x)
+
w_{\mathrm{loc},t}
\widehat N_{\mathrm{loc},t}(x)
+
w_{\mathrm{miss},t}
\widehat N_{\mathrm{miss},t}(x)
}
\]

이다.

Weight가 적용되지 않은 단순 count sum이 아니라, 각 error type의 scale을 맞춘 뒤 계산한 총 error amount다.

---

# 16. Candidate images

Unlabeled pool에서 weighted ECA가 높은 상위 \(\lceil\delta b\rceil\)개 이미지를 candidates로 선택한다.

\[
\boxed{
\mathcal C_t
=
\operatorname{Top}_{\lceil\delta b\rceil}
\left\{
\operatorname{ECA}_t(x):
x\in\mathcal U_t
\right\}
}
\]

여기서 \(\mathcal C_t\)는 최종 selected images가 아니라 두 번째 diversity selection을 위한 candidate images다.

---

# 17. Weighted Error-Count Composition

Error-count composition도 raw error-count profile을 바로 정규화하지 않는다. ECA에서 사용한 동일한 weights를 먼저 적용한 뒤 정규화한다.

\[
\widehat{\mathbf e}_t^{\,w}(x)
=
\begin{bmatrix}
w_{\mathrm{cls},t}\widehat N_{\mathrm{cls},t}(x)\\
w_{\mathrm{loc},t}\widehat N_{\mathrm{loc},t}(x)\\
w_{\mathrm{miss},t}\widehat N_{\mathrm{miss},t}(x)
\end{bmatrix}
\]

이므로 weighted error-count composition은

\[
\boxed{
\boldsymbol\pi_t(x)
=
\frac{
\widehat{\mathbf e}_t^{\,w}(x)
+
\epsilon\mathbf 1
}{
\operatorname{ECA}_t(x)+3\epsilon
}
}
\]

이다.

성분별로 쓰면

\[
\boxed{
\pi_{t,m}(x)
=
\frac{
w_{m,t}\widehat N_{m,t}(x)+\epsilon
}{
\sum_{r}
w_{r,t}\widehat N_{r,t}(x)
+
3\epsilon
}
}
\]

이다.

여기서

\[
m,r
\in
\{\mathrm{cls},\mathrm{loc},\mathrm{miss}\}.
\]

따라서

\[
\boldsymbol\pi_t(x)
=
\begin{bmatrix}
\pi_{t,\mathrm{cls}}(x)\\
\pi_{t,\mathrm{loc}}(x)\\
\pi_{t,\mathrm{miss}}(x)
\end{bmatrix},
\qquad
\sum_m\pi_{t,m}(x)=1.
\]

이 composition은 error count의 원래 scale 차이를 제거한 뒤 각 error type이 차지하는 상대적인 비율을 나타낸다.

---

# 18. Error-Count Diversity

두 candidate images \(x_a\)와 \(x_b\) 사이의 weighted error-count composition 차이는 Jensen–Shannon distance로 계산한다.

먼저

\[
\mathbf m_{ab}
=
\frac{
\boldsymbol\pi_t(x_a)
+
\boldsymbol\pi_t(x_b)
}{2}
\]

라고 한다.

Jensen–Shannon divergence는

\[
\operatorname{JS}
\left(
\boldsymbol\pi_t(x_a),
\boldsymbol\pi_t(x_b)
\right)
=
\frac12
\operatorname{KL}
\left(
\boldsymbol\pi_t(x_a)
\|
\mathbf m_{ab}
\right)
+
\frac12
\operatorname{KL}
\left(
\boldsymbol\pi_t(x_b)
\|
\mathbf m_{ab}
\right)
\]

이다.

Error-Count Diversity distance는

\[
\boxed{
d_{\mathrm{ECD}}
(x_a,x_b)
=
\sqrt{
\operatorname{JS}
\left(
\boldsymbol\pi_t(x_a),
\boldsymbol\pi_t(x_b)
\right)
}
}
\]

로 정의한다.

---

# 19. Selected images

Candidates 중 weighted ECA가 가장 높은 이미지를 첫 번째 selected image로 둔다.

\[
\boxed{
x_1
=
\arg\max_{x\in\mathcal C_t}
\operatorname{ECA}_t(x)
}
\]

초기 selected-image set은

\[
\mathcal S_t^{(1)}
=
\{x_1\}
\]

이다.

현재 selected-image set이 \(\mathcal S_t^{(\ell-1)}\)일 때 다음 이미지는

\[
\boxed{
x_\ell
=
\arg\max_{
x\in
\mathcal C_t
\setminus
\mathcal S_t^{(\ell-1)}
}
\;
\min_{
x'\in
\mathcal S_t^{(\ell-1)}
}
d_{\mathrm{ECD}}(x,x')
}
\]

로 선택한다.

선택된 이미지를 set에 추가한다.

\[
\mathcal S_t^{(\ell)}
=
\mathcal S_t^{(\ell-1)}
\cup
\{x_\ell\}.
\]

이를

\[
|\mathcal S_t^{(\ell)}|=b
\]

가 될 때까지 반복한다.

ECD가 동일하면 weighted ECA가 높은 이미지를 우선한다.

최종 selected images는

\[
\boxed{
\mathcal S_t
=
\mathcal S_t^{(b)}
}
\]

이다.

---

# 20. Complete active learning round

## Step 1. Detector training

\[
\boxed{
\theta_t
=
\operatorname{TrainDetector}(\mathcal L_t)
}
\]

---

## Step 2. Labeled-image inference

각

\[
(x,\mathcal G(x))
\in\mathcal L_t
\]

에 현재 detector를 적용하여

\[
\widetilde{\mathcal D}_t(x),
\qquad
\widehat{\mathcal D}_t(x)
\]

를 얻는다.

---

## Step 3. Support-set construction

각 raw detection을 가장 많이 겹치는 final detection에 연결한다.

Final box와 동일한 box를 갖는 raw detection만 제거하여

\[
\mathcal H_i(x)
=
\mathcal A_i(x)
\setminus
\mathcal Q_i(x)
\]

를 구성한다.

---

## Step 4. Feature extraction

각 final detection에 대해

\[
\mathbf z_i^{\mathrm{common}}
=
\begin{bmatrix}
\hat p_i^{\max}\\
A_i^{\mathrm{cls}}\\
n_i^{\mathrm{sup}}\\
\mu_i^{\mathrm{IoU}}
\end{bmatrix}
\]

를 계산한다.

Classification features는

\[
\mathbf z_i^{\mathrm{cls}}
=
\begin{bmatrix}
\hat p_i^{\max}\\
A_i^{\mathrm{cls}}
\end{bmatrix}
\]

이고, localization features는

\[
\mathbf z_i^{\mathrm{loc}}
=
\begin{bmatrix}
n_i^{\mathrm{sup}}\\
\mu_i^{\mathrm{IoU}}
\end{bmatrix}
\]

이다.

각 이미지에는 MOCP features

\[
\mathbf z^{\mathrm{miss}}(x)
=
\begin{bmatrix}
R_{\mathrm{amt}}(x)\\
R_{\mathrm{prob}}(x)
\end{bmatrix}
\]

를 계산한다.

---

## Step 5. Label generation

GT를 이용해 모든 final detections에 대해

\[
y_i^{\mathrm{fg}}
\]

를 생성한다.

Foreground-related detections에는

\[
y_i^{\mathrm{cls}},
\qquad
y_i^{\mathrm{loc}}
\]

를 생성한다.

각 labeled image에는

\[
N_{\mathrm{miss}}(x)
\]

를 생성한다.

---

## Step 6. Predictor training

FDP는 모든 final detections로 학습한다.

\[
\mathrm{FDP}_t
=
\operatorname{FitLogistic}
\left(
\mathbf z^{\mathrm{common}},
y^{\mathrm{fg}}
\right).
\]

CECP는 foreground-related detections로 학습한다.

\[
\mathrm{CECP}_t
=
\operatorname{FitLogistic}
\left(
\mathbf z^{\mathrm{cls}},
y^{\mathrm{cls}}
\right).
\]

LECP도 foreground-related detections로 학습한다.

\[
\mathrm{LECP}_t
=
\operatorname{FitLogistic}
\left(
\mathbf z^{\mathrm{loc}},
y^{\mathrm{loc}}
\right).
\]

MOCP는 image-wise residual features와 missed-object counts로 학습한다.

\[
\mathrm{MOCP}_t
=
\operatorname{FitPoisson}
\left(
\mathbf z^{\mathrm{miss}},
N_{\mathrm{miss}}
\right).
\]

---

## Step 7. Error-scale weight estimation

Labeled set의 actual error counts로 각 error type의 평균 scale을 계산한다.

\[
\bar N_{m,t}
=
\frac{1}{|\mathcal L_t|}
\sum_{x\in\mathcal L_t}
N_m(x).
\]

Inverse-scale weights를 계산하고 평균이 1이 되도록 정규화한다.

\[
\tilde w_{m,t}
=
\frac{1}{\bar N_{m,t}+\epsilon_w},
\]

\[
w_{m,t}
=
\frac{
3\tilde w_{m,t}
}{
\sum_r\tilde w_{r,t}
}.
\]

---

## Step 8. Unlabeled-image inference

각

\[
x\in\mathcal U_t
\]

에 현재 detector를 한 번 적용하여

\[
\widetilde{\mathcal D}_t(x),
\qquad
\widehat{\mathcal D}_t(x)
\]

를 얻는다.

동일한 방식으로 support sets와 predictor features를 계산한다.

---

## Step 9. Detection-wise prediction

각 final detection에 대해

\[
q_{i,t}^{\mathrm{fg}}
\]

를 예측한다.

또한

\[
q_{i,t}^{\mathrm{cls}\mid\mathrm{fg}},
\qquad
q_{i,t}^{\mathrm{loc}\mid\mathrm{fg}}
\]

를 예측한다.

Unconditional error probabilities는

\[
q_{i,t}^{\mathrm{cls}}
=
q_{i,t}^{\mathrm{fg}}
q_{i,t}^{\mathrm{cls}\mid\mathrm{fg}},
\]

\[
q_{i,t}^{\mathrm{loc}}
=
q_{i,t}^{\mathrm{fg}}
q_{i,t}^{\mathrm{loc}\mid\mathrm{fg}}
\]

로 계산한다.

---

## Step 10. Error-count prediction

\[
\widehat N_{\mathrm{cls},t}(x)
=
\sum_i
q_{i,t}^{\mathrm{fg}}
q_{i,t}^{\mathrm{cls}\mid\mathrm{fg}},
\]

\[
\widehat N_{\mathrm{loc},t}(x)
=
\sum_i
q_{i,t}^{\mathrm{fg}}
q_{i,t}^{\mathrm{loc}\mid\mathrm{fg}},
\]

\[
\widehat N_{\mathrm{miss},t}(x)
=
\mathrm{MOCP}_t
\left(
\mathbf z^{\mathrm{miss}}(x)
\right)
\]

를 계산한다.

---

## Step 11. Weighted profile construction

\[
\widehat{\mathbf e}_t^{\,w}(x)
=
\begin{bmatrix}
w_{\mathrm{cls},t}
\widehat N_{\mathrm{cls},t}(x)\\
w_{\mathrm{loc},t}
\widehat N_{\mathrm{loc},t}(x)\\
w_{\mathrm{miss},t}
\widehat N_{\mathrm{miss},t}(x)
\end{bmatrix}
\]

를 계산한다.

---

## Step 12. Candidate selection

\[
\operatorname{ECA}_t(x)
=
\mathbf 1^\top
\widehat{\mathbf e}_t^{\,w}(x)
\]

를 계산하고, 상위 \(\lceil\delta b\rceil\)개 이미지를 candidates로 선택한다.

\[
\mathcal C_t
=
\operatorname{Top}_{\lceil\delta b\rceil}
\operatorname{ECA}_t(x).
\]

---

## Step 13. Diversity selection

동일하게 weighted profile을 정규화한다.

\[
\boldsymbol\pi_t(x)
=
\frac{
\widehat{\mathbf e}_t^{\,w}(x)+\epsilon\mathbf 1
}{
\operatorname{ECA}_t(x)+3\epsilon
}.
\]

Jensen–Shannon distance 기반 farthest-first selection으로 최종 \(b\)개의 selected images를 결정한다.

\[
\mathcal S_t
=
\operatorname{FarthestFirst}
\left(
\mathcal C_t,
d_{\mathrm{ECD}},
b
\right).
\]

---

## Step 14. Annotation and set update

Selected images의 GT annotations를 획득한다.

\[
\boxed{
\mathcal L_{t+1}
=
\mathcal L_t
\cup
\left\{
(x,\mathcal G(x)):
x\in\mathcal S_t
\right\}
}
\]

\[
\boxed{
\mathcal U_{t+1}
=
\mathcal U_t
\setminus
\mathcal S_t
}
\]

이후 새로운 labeled set으로 detector와 predictor들을 다시 학습하며 다음 active learning round를 진행한다.

---

# 21. Core formulation

\[
\boxed{
q_i^{\mathrm{fg}}
=
P
\left(
\mathrm{Foreground}
\mid
\hat p_i^{\max},
A_i^{\mathrm{cls}},
n_i^{\mathrm{sup}},
\mu_i^{\mathrm{IoU}}
\right)
}
\]

\[
\boxed{
q_i^{\mathrm{cls}\mid\mathrm{fg}}
=
P
\left(
\mathrm{ClsError}
\mid
\mathrm{Foreground},
\hat p_i^{\max},
A_i^{\mathrm{cls}}
\right)
}
\]

\[
\boxed{
q_i^{\mathrm{loc}\mid\mathrm{fg}}
=
P
\left(
\mathrm{LocError}
\mid
\mathrm{Foreground},
n_i^{\mathrm{sup}},
\mu_i^{\mathrm{IoU}}
\right)
}
\]

\[
\boxed{
\widehat N_{\mathrm{cls},t}(x)
=
\sum_i
q_{i,t}^{\mathrm{fg}}
q_{i,t}^{\mathrm{cls}\mid\mathrm{fg}}
}
\]

\[
\boxed{
\widehat N_{\mathrm{loc},t}(x)
=
\sum_i
q_{i,t}^{\mathrm{fg}}
q_{i,t}^{\mathrm{loc}\mid\mathrm{fg}}
}
\]

\[
\boxed{
\widehat N_{\mathrm{miss},t}(x)
=
\exp
\left(
\gamma_0
+
\gamma_1R_{\mathrm{amt}}(x)
+
\gamma_2R_{\mathrm{prob}}(x)
\right)
}
\]

\[
\boxed{
\operatorname{ECA}_t(x)
=
w_{\mathrm{cls},t}\widehat N_{\mathrm{cls},t}(x)
+
w_{\mathrm{loc},t}\widehat N_{\mathrm{loc},t}(x)
+
w_{\mathrm{miss},t}\widehat N_{\mathrm{miss},t}(x)
}
\]

\[
\boxed{
\boldsymbol\pi_t(x)
=
\frac{
\begin{bmatrix}
w_{\mathrm{cls},t}\widehat N_{\mathrm{cls},t}(x)\\
w_{\mathrm{loc},t}\widehat N_{\mathrm{loc},t}(x)\\
w_{\mathrm{miss},t}\widehat N_{\mathrm{miss},t}(x)
\end{bmatrix}
+
\epsilon\mathbf 1
}{
\operatorname{ECA}_t(x)+3\epsilon
}
}
\]

전체 과정은 다음과 같이 정리된다.

\[
\boxed{
\begin{aligned}
&\text{Detector training}\\
&\rightarrow
\text{Raw/final detection inference}\\
&\rightarrow
\text{Exact-box-excluded support construction}\\
&\rightarrow
\text{Foreground probability prediction}\\
&\rightarrow
\text{Conditional classification/localization prediction}\\
&\rightarrow
\text{Foreground-gated error-count prediction}\\
&\rightarrow
\text{Missed-object count prediction}\\
&\rightarrow
\text{Scale-weighted error-count profile}\\
&\rightarrow
\text{Weighted ECA-based candidates}\\
&\rightarrow
\text{Weighted ECD-based selected images}\\
&\rightarrow
\text{Annotation and next round}
\end{aligned}
}
\]
