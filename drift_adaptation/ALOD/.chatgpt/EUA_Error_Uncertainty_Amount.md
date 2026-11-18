# Error-Uncertainty Amount (EUA)

## 1. 기본 아이디어

기존 Error-Count Amount (ECA)는 이미지에서 발생할 것으로 예상되는 오류의 양을 측정한다.

\[
\operatorname{ECA}_t(x)
=
w_{\mathrm{cls},t}\widehat N_{\mathrm{cls},t}(x)
+
w_{\mathrm{loc},t}\widehat N_{\mathrm{loc},t}(x)
+
w_{\mathrm{miss},t}\widehat N_{\mathrm{miss},t}(x).
\]

따라서 ECA는

\[
\boxed{
\text{Expected Error Amount}
}
\]

가 큰 이미지를 선택한다.

반면 Error-Uncertainty Amount (EUA)는 각 error predictor의 출력으로부터 **해당 error의 발생 여부 또는 개수에 대한 predictive uncertainty**를 계산한다.

즉 EUA가 높은 이미지는 반드시 error가 많을 것으로 예상되는 이미지가 아니라,

\[
\boxed{
\text{현재 predictor가 detector error를 확실하게 판단하기 어려운 이미지}
}
\]

이다.

---

# 2. Predictor outputs

이미지 \(x\)의 final detections를

\[
\widehat{\mathcal D}_t(x)
=
\left\{
\hat d_i
\right\}_{i=1}^{N_x}
\]

라고 한다.

기존 ECA와 동일하게 Foreground Detection Predictor (FDP), Conditional Classification-Error Count Predictor (CECP), Conditional Localization-Error Count Predictor (LECP), Missed-Object Count Predictor (MOCP)를 사용한다.

## 2.1 Foreground probability

각 final detection \(i\)에 대해 FDP는

\[
\boxed{
q_{i,t}^{\mathrm{fg}}
=
P
\left(
y_i^{\mathrm{fg}}=1
\mid
\mathbf z_i^{\mathrm{fg}}
\right)
}
\]

를 예측한다.

입력 feature는

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

이다.

---

## 2.2 Conditional classification-error probability

CECP는 foreground-related detection이라는 조건에서 classification error일 probability를 예측한다.

\[
\boxed{
q_{i,t}^{\mathrm{cls}\mid\mathrm{fg}}
=
P
\left(
y_i^{\mathrm{cls}}=1
\mid
y_i^{\mathrm{fg}}=1,
\mathbf z_i^{\mathrm{cls}}
\right)
}
\]

where

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

이다.

---

## 2.3 Conditional localization-error probability

LECP는 foreground-related detection이라는 조건에서 localization error일 probability를 예측한다.

\[
\boxed{
q_{i,t}^{\mathrm{loc}\mid\mathrm{fg}}
=
P
\left(
y_i^{\mathrm{loc}}=1
\mid
y_i^{\mathrm{fg}}=1,
\mathbf z_i^{\mathrm{loc}}
\right)
}
\]

where

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

이다.

---

## 2.4 Missed-object count distribution

MOCP는 이미지별 missed-object count를 Poisson distribution으로 모델링한다.

\[
\boxed{
N_{\mathrm{miss}}(x)
\mid
\mathbf z^{\mathrm{miss}}(x)
\sim
\operatorname{Poisson}
\left(
\lambda_{x,t}
\right)
}
\]

where

\[
\boxed{
\lambda_{x,t}
=
\exp
\left(
\gamma_{0,t}
+
\gamma_{1,t}R_{\mathrm{amt}}(x)
+
\gamma_{2,t}R_{\mathrm{prob}}(x)
\right)
}
\]

이다.

따라서 expected missed-object count는

\[
\widehat N_{\mathrm{miss},t}(x)
=
\lambda_{x,t}
\]

이다.

ECA와 EUA는 여기까지 **완전히 동일한 predictors와 outputs**를 사용한다.

차이는 그 이후 predictor output을 image-level acquisition score로 변환하는 방법이다.

---

# 3. Bernoulli entropy

Classification과 localization error는 detection-wise binary random variable로 모델링한다.

Binary probability \(p\)에 대한 Bernoulli entropy를

\[
\boxed{
H_{\mathrm{Bern}}(p)
=
-p\log p
-
(1-p)\log(1-p)
}
\]

로 정의한다.

Natural logarithm을 사용하므로 entropy 단위는 nat이다.

\[
0
\le
H_{\mathrm{Bern}}(p)
\le
\log 2.
\]

특히

\[
H_{\mathrm{Bern}}(0)
=
H_{\mathrm{Bern}}(1)
=
0
\]

이며,

\[
\boxed{
H_{\mathrm{Bern}}(0.5)
=
\log 2
}
\]

에서 최대가 된다.

따라서

- \(p\approx0\): error가 아니라고 확신
- \(p\approx1\): error라고 확신
- \(p\approx0.5\): error인지 아닌지 가장 불확실

한 것으로 해석한다.

---

# 4. Classification-Error Uncertainty

Detection \(i\)가 foreground-related detection이라는 조건에서 classification-error status의 uncertainty는

\[
H_{\mathrm{Bern}}
\left(
q_{i,t}^{\mathrm{cls}\mid\mathrm{fg}}
\right)
\]

이다.

하지만 background detection의 classification correctness는 classification-error 정의의 대상이 아니므로, foreground probability를 이용해 각 detection의 entropy contribution을 조절한다.

Detection \(i\)의 classification uncertainty contribution을

\[
\boxed{
u_{i,t}^{\mathrm{cls}}
=
q_{i,t}^{\mathrm{fg}}
H_{\mathrm{Bern}}
\left(
q_{i,t}^{\mathrm{cls}\mid\mathrm{fg}}
\right)
}
\]

로 정의한다.

따라서 이미지 \(x\)의 전체 classification-error uncertainty는

\[
\boxed{
U_{\mathrm{cls},t}(x)
=
\sum_{i=1}^{N_x}
q_{i,t}^{\mathrm{fg}}
H_{\mathrm{Bern}}
\left(
q_{i,t}^{\mathrm{cls}\mid\mathrm{fg}}
\right)
}
\]

이다.

즉 foreground일 가능성이 높으면서

\[
q_{i,t}^{\mathrm{cls}\mid\mathrm{fg}}
\approx0.5
\]

인 detection이 많을수록 \(U_{\mathrm{cls},t}(x)\)가 커진다.

반대로

\[
q_{i,t}^{\mathrm{cls}\mid\mathrm{fg}}
\approx0
\]

또는

\[
q_{i,t}^{\mathrm{cls}\mid\mathrm{fg}}
\approx1
\]

이면 classification error 여부를 predictor가 확신하고 있으므로 EUA에 대한 contribution은 작다.

---

# 5. Localization-Error Uncertainty

Localization error도 동일하게 detection-wise Bernoulli random variable로 본다.

Detection \(i\)의 localization uncertainty contribution은

\[
\boxed{
u_{i,t}^{\mathrm{loc}}
=
q_{i,t}^{\mathrm{fg}}
H_{\mathrm{Bern}}
\left(
q_{i,t}^{\mathrm{loc}\mid\mathrm{fg}}
\right)
}
\]

이다.

따라서 이미지 \(x\)의 localization-error uncertainty는

\[
\boxed{
U_{\mathrm{loc},t}(x)
=
\sum_{i=1}^{N_x}
q_{i,t}^{\mathrm{fg}}
H_{\mathrm{Bern}}
\left(
q_{i,t}^{\mathrm{loc}\mid\mathrm{fg}}
\right)
}
\]

이다.

즉

\[
q_{i,t}^{\mathrm{loc}\mid\mathrm{fg}}
\approx0.5
\]

인 foreground-related detections가 많이 존재할수록 localization uncertainty가 커진다.

---

# 6. Foreground probability의 역할

ECA에서는 실제 classification-error probability를

\[
q_{i,t}^{\mathrm{cls}}
=
q_{i,t}^{\mathrm{fg}}
q_{i,t}^{\mathrm{cls}\mid\mathrm{fg}}
\]

로 계산했다.

EUA에서 한 가지 가능한 방법은 이 marginal probability에 직접 entropy를 취하는 것이다.

\[
H_{\mathrm{Bern}}
\left(
q_{i,t}^{\mathrm{fg}}
q_{i,t}^{\mathrm{cls}\mid\mathrm{fg}}
\right).
\]

그러나 EUA에서는 이를 사용하지 않고

\[
\boxed{
q_{i,t}^{\mathrm{fg}}
H_{\mathrm{Bern}}
\left(
q_{i,t}^{\mathrm{cls}\mid\mathrm{fg}}
\right)
}
\]

를 사용한다.

이 방식에서는 foreground/background 판단과 classification-error 판단의 역할이 분리된다.

즉 classification uncertainty가 의미하는 것은

> 실제 객체와 관련되어 있을 가능성이 높은 detection에 대해, classification이 correct인지 error인지 얼마나 불확실한가

이다.

Localization도 동일하게

\[
q_{i,t}^{\mathrm{fg}}
H_{\mathrm{Bern}}
\left(
q_{i,t}^{\mathrm{loc}\mid\mathrm{fg}}
\right)
\]

를 사용한다.

---

# 7. Poisson count distribution

MOCP는 binary event가 아니라 missed-object count 자체를 예측한다.

\[
N_{\mathrm{miss}}(x)
\mid x
\sim
\operatorname{Poisson}
\left(
\lambda_{x,t}
\right).
\]

Poisson probability mass function은

\[
\boxed{
P
\left(
N_{\mathrm{miss}}(x)=k
\mid
\lambda_{x,t}
\right)
=
\frac{
e^{-\lambda_{x,t}}
\lambda_{x,t}^{k}
}{
k!
}
}
\]

이다.

이를

\[
p_k(\lambda)
=
\frac{
e^{-\lambda}\lambda^k
}{
k!
}
\]

라고 나타내자.

---

# 8. Missed-Object Uncertainty

Missed-object uncertainty는 missed object의 존재 여부에 대한 binary entropy로 변환하지 않고, **MOCP가 예측하는 전체 count distribution의 Shannon entropy**를 사용한다.

\[
\boxed{
U_{\mathrm{miss},t}(x)
=
H_{\mathrm{Pois}}
\left(
\lambda_{x,t}
\right)
}
\]

Poisson entropy는

\[
\boxed{
H_{\mathrm{Pois}}(\lambda)
=
-
\sum_{k=0}^{\infty}
p_k(\lambda)
\log p_k(\lambda)
}
\]

로 정의한다.

따라서 이미지 \(x\)에서는

\[
\boxed{
U_{\mathrm{miss},t}(x)
=
-
\sum_{k=0}^{\infty}
\frac{
e^{-\lambda_{x,t}}
\lambda_{x,t}^{k}
}{
k!
}
\log
\left(
\frac{
e^{-\lambda_{x,t}}
\lambda_{x,t}^{k}
}{
k!
}
\right)
}
\]

이다.

---

# 9. Missed-Object Uncertainty의 의미

기존에 고려했던 binary-existence entropy는

\[
N_{\mathrm{miss}}=0
\quad\text{vs.}\quad
N_{\mathrm{miss}}>0
\]

만 구분한다.

반면 Poisson entropy는

\[
N_{\mathrm{miss}}
\in
\{0,1,2,\ldots\}
\]

전체에 대한 predictive distribution을 고려한다.

즉

\[
\boxed{
U_{\mathrm{miss},t}(x)
=
\text{predictive uncertainty of the missed-object count}
}
\]

로 해석한다.

예를 들어 특정 하나의 count에 probability가 집중된 distribution보다 여러 possible missed-object counts에 probability가 분산된 distribution에서 entropy가 크다.

---

# 10. 세 uncertainty components의 공통 원리

최종적으로 classification, localization, missed object는 모두 각 error predictor가 정의하는 **predictive distribution의 entropy**를 이용한다.

| Error component | Predictive distribution | Image-level uncertainty |
|---|---|---|
| Classification | \(\operatorname{Bernoulli}(q_i^{\mathrm{cls}\mid\mathrm{fg}})\) | Foreground-weighted detection entropy의 합 |
| Localization | \(\operatorname{Bernoulli}(q_i^{\mathrm{loc}\mid\mathrm{fg}})\) | Foreground-weighted detection entropy의 합 |
| Missed object | \(\operatorname{Poisson}(\lambda_x)\) | Poisson count entropy |

따라서

\[
\boxed{
U_{\mathrm{cls},t}(x)
=
\sum_i
q_{i,t}^{\mathrm{fg}}
H_{\mathrm{Bern}}
\left(
q_{i,t}^{\mathrm{cls}\mid\mathrm{fg}}
\right)
}
\]

\[
\boxed{
U_{\mathrm{loc},t}(x)
=
\sum_i
q_{i,t}^{\mathrm{fg}}
H_{\mathrm{Bern}}
\left(
q_{i,t}^{\mathrm{loc}\mid\mathrm{fg}}
\right)
}
\]

\[
\boxed{
U_{\mathrm{miss},t}(x)
=
H_{\mathrm{Pois}}
\left(
\lambda_{x,t}
\right)
}
\]

로 정리된다.

---

# 11. Error-Uncertainty Profile

세 uncertainty component를 이용하여 이미지 \(x\)의 unweighted Error-Uncertainty Profile을

\[
\boxed{
\mathbf u_t(x)
=
\begin{bmatrix}
U_{\mathrm{cls},t}(x)\\
U_{\mathrm{loc},t}(x)\\
U_{\mathrm{miss},t}(x)
\end{bmatrix}
}
\]

로 정의한다.

이를 완전히 전개하면

\[
\boxed{
\mathbf u_t(x)
=
\begin{bmatrix}
\displaystyle
\sum_{i=1}^{N_x}
q_{i,t}^{\mathrm{fg}}
H_{\mathrm{Bern}}
\left(
q_{i,t}^{\mathrm{cls}\mid\mathrm{fg}}
\right)
\\[7mm]
\displaystyle
\sum_{i=1}^{N_x}
q_{i,t}^{\mathrm{fg}}
H_{\mathrm{Bern}}
\left(
q_{i,t}^{\mathrm{loc}\mid\mathrm{fg}}
\right)
\\[7mm]
\displaystyle
H_{\mathrm{Pois}}
\left(
\lambda_{x,t}
\right)
\end{bmatrix}
}
\]

이다.

---

# 12. Scale weighting이 필요한 이유

세 uncertainty components의 numerical scale은 서로 다를 수 있다.

Classification과 localization에서는 이미지 내 여러 final detections에 대해 entropy를 합한다.

\[
U_{\mathrm{cls}}
=
\sum_i
q_i^{\mathrm{fg}}H_{\mathrm{Bern}}(\cdot),
\]

\[
U_{\mathrm{loc}}
=
\sum_i
q_i^{\mathrm{fg}}H_{\mathrm{Bern}}(\cdot).
\]

따라서 foreground-related detections가 많은 이미지에서는 큰 값이 가능하다.

Missed-object uncertainty는 하나의 image-wise Poisson distribution에 대한 entropy다.

\[
U_{\mathrm{miss}}
=
H_{\mathrm{Pois}}(\lambda).
\]

따라서 raw uncertainty components를 그대로 합하면 특정 component의 numerical scale이 EUA를 지배할 수 있다.

이를 보정하기 위해 ECA와 동일한 **round-wise inverse-scale weighting**을 적용한다.

---

# 13. Scale은 labeled pool에서 추정

ECA에서 error-count scale을 labeled pool \(\mathcal L_t\)의 실제 error counts를 이용해 추정했던 것과 동일하게, EUA의 uncertainty scale도 **현재 labeled pool \(\mathcal L_t\)** 에서 추정한다.

중요한 점은 EUA scale을 계산할 때 GT error label 자체를 uncertainty로 사용하는 것이 아니라, **현재 학습된 predictor outputs를 labeled images에 적용하여 EUA와 동일한 방식으로 uncertainty를 계산한다는 것**이다.

즉 모든

\[
x\in\mathcal L_t
\]

에 대해

\[
U_{\mathrm{cls},t}(x),
\qquad
U_{\mathrm{loc},t}(x),
\qquad
U_{\mathrm{miss},t}(x)
\]

를 계산한다.

여기서 classification과 localization uncertainty를 계산할 때도 실제 foreground label \(y_i^{\mathrm{fg}}\)를 곱하는 것이 아니라, acquisition 때와 동일하게 predictor의

\[
q_{i,t}^{\mathrm{fg}}
\]

를 사용한다.

그래야 labeled pool에서 측정한 scale과 unlabeled pool에서 계산할 EUA의 정의가 동일하다.

---

# 14. Labeled-pool uncertainty scale

각 component의 labeled-pool 평균 uncertainty를

\[
\boxed{
\bar U_{m,t}
=
\frac{1}{|\mathcal L_t|}
\sum_{x\in\mathcal L_t}
U_{m,t}(x)
}
\]

로 정의한다.

여기서

\[
m
\in
\{\mathrm{cls},\mathrm{loc},\mathrm{miss}\}.
\]

구체적으로,

\[
\boxed{
\bar U_{\mathrm{cls},t}
=
\frac{1}{|\mathcal L_t|}
\sum_{x\in\mathcal L_t}
U_{\mathrm{cls},t}(x)
}
\]

\[
\boxed{
\bar U_{\mathrm{loc},t}
=
\frac{1}{|\mathcal L_t|}
\sum_{x\in\mathcal L_t}
U_{\mathrm{loc},t}(x)
}
\]

\[
\boxed{
\bar U_{\mathrm{miss},t}
=
\frac{1}{|\mathcal L_t|}
\sum_{x\in\mathcal L_t}
U_{\mathrm{miss},t}(x)
}
\]

이다.

---

# 15. Uncertainty-scale weights

ECA와 동일한 inverse-mean 방식으로 각 uncertainty component의 scale weight를 계산한다.

먼저

\[
\boxed{
\tilde v_{m,t}
=
\frac{1}{
\bar U_{m,t}
+
\epsilon_v
}
}
\]

로 정의한다.

그리고 세 weights의 평균이 1이 되도록 정규화한다.

\[
\boxed{
v_{m,t}
=
\frac{
3\tilde v_{m,t}
}{
\tilde v_{\mathrm{cls},t}
+
\tilde v_{\mathrm{loc},t}
+
\tilde v_{\mathrm{miss},t}
}
}
\]

따라서

\[
\boxed{
\frac{
v_{\mathrm{cls},t}
+
v_{\mathrm{loc},t}
+
v_{\mathrm{miss},t}
}{3}
=
1
}
\]

이다.

이 weight는 error type의 중요도를 직접 지정하는 parameter가 아니라, 세 uncertainty components가 갖는 **평균적인 numerical scale 차이를 보정하기 위한 scale weight**다.

---

# 16. Scale-Weighted Error-Uncertainty Profile

Uncertainty-scale weight vector를

\[
\boxed{
\mathbf v_t
=
\begin{bmatrix}
v_{\mathrm{cls},t}\\
v_{\mathrm{loc},t}\\
v_{\mathrm{miss},t}
\end{bmatrix}
}
\]

라고 한다.

Weight matrix는

\[
\boxed{
\mathbf V_t
=
\operatorname{diag}
\left(
v_{\mathrm{cls},t},
v_{\mathrm{loc},t},
v_{\mathrm{miss},t}
\right)
}
\]

이다.

Scale-weighted Error-Uncertainty Profile은

\[
\boxed{
\mathbf u_t^{\,w}(x)
=
\mathbf V_t
\mathbf u_t(x)
}
\]

로 정의한다.

즉,

\[
\boxed{
\mathbf u_t^{\,w}(x)
=
\begin{bmatrix}
v_{\mathrm{cls},t}U_{\mathrm{cls},t}(x)\\
v_{\mathrm{loc},t}U_{\mathrm{loc},t}(x)\\
v_{\mathrm{miss},t}U_{\mathrm{miss},t}(x)
\end{bmatrix}
}
\]

이다.

이를 predictor outputs까지 전개하면

\[
\boxed{
\mathbf u_t^{\,w}(x)
=
\begin{bmatrix}
\displaystyle
v_{\mathrm{cls},t}
\sum_{i=1}^{N_x}
q_{i,t}^{\mathrm{fg}}
H_{\mathrm{Bern}}
\left(
q_{i,t}^{\mathrm{cls}\mid\mathrm{fg}}
\right)
\\[7mm]
\displaystyle
v_{\mathrm{loc},t}
\sum_{i=1}^{N_x}
q_{i,t}^{\mathrm{fg}}
H_{\mathrm{Bern}}
\left(
q_{i,t}^{\mathrm{loc}\mid\mathrm{fg}}
\right)
\\[7mm]
\displaystyle
v_{\mathrm{miss},t}
H_{\mathrm{Pois}}
\left(
\lambda_{x,t}
\right)
\end{bmatrix}
}
\]

이다.

---

# 17. Error-Uncertainty Amount

이미지 \(x\)의 전체 predictive error uncertainty를 **Error-Uncertainty Amount (EUA)**로 정의한다.

\[
\boxed{
\operatorname{EUA}_t(x)
=
\left\|
\mathbf u_t^{\,w}(x)
\right\|_1
}
\]

모든 component가 non-negative이므로

\[
\boxed{
\operatorname{EUA}_t(x)
=
v_{\mathrm{cls},t}
U_{\mathrm{cls},t}(x)
+
v_{\mathrm{loc},t}
U_{\mathrm{loc},t}(x)
+
v_{\mathrm{miss},t}
U_{\mathrm{miss},t}(x)
}
\]

이다.

완전히 전개하면

\[
\boxed{
\begin{aligned}
\operatorname{EUA}_t(x)
={}&
v_{\mathrm{cls},t}
\sum_{i=1}^{N_x}
q_{i,t}^{\mathrm{fg}}
H_{\mathrm{Bern}}
\left(
q_{i,t}^{\mathrm{cls}\mid\mathrm{fg}}
\right)
\\[2mm]
&+
v_{\mathrm{loc},t}
\sum_{i=1}^{N_x}
q_{i,t}^{\mathrm{fg}}
H_{\mathrm{Bern}}
\left(
q_{i,t}^{\mathrm{loc}\mid\mathrm{fg}}
\right)
\\[2mm]
&+
v_{\mathrm{miss},t}
H_{\mathrm{Pois}}
\left(
\lambda_{x,t}
\right).
\end{aligned}
}
\]

이 식이 EUA-based uncertainty sampling의 핵심 acquisition score다.

---

# 18. EUA-based candidate selection

현재 active-learning round의 annotation budget을 \(b\), candidate multiplier를 \(\delta>1\)라고 한다.

Candidate 수는

\[
K
=
\lceil\delta b\rceil
\]

이다.

모든

\[
x\in\mathcal U_t
\]

에 대해 EUA를 계산한 뒤 상위 \(K\)개 이미지를 candidate images로 선택한다.

\[
\boxed{
\mathcal C_t^{\mathrm{EUA}}
=
\operatorname{Top}_{\lceil\delta b\rceil}
\left\{
\operatorname{EUA}_t(x):
x\in\mathcal U_t
\right\}
}
\]

즉 EUA-based first-stage sampling은

\[
\boxed{
\text{predictive error uncertainty가 큰 이미지를 우선적으로 candidate로 선택}
}
\]

한다.

---

# 19. ECA와 EUA의 직접적인 비교

두 방법은 predictor training까지는 동일하다.

동일한

\[
q_i^{\mathrm{fg}},
\qquad
q_i^{\mathrm{cls}\mid\mathrm{fg}},
\qquad
q_i^{\mathrm{loc}\mid\mathrm{fg}},
\qquad
\lambda_x
\]

를 사용한다.

차이는 이를 acquisition score로 변환하는 방식이다.

| Component | ECA | EUA |
|---|---|---|
| Classification | \(\sum_i q_i^{\mathrm{fg}}q_i^{\mathrm{cls}\mid\mathrm{fg}}\) | \(\sum_i q_i^{\mathrm{fg}}H_{\mathrm{Bern}}(q_i^{\mathrm{cls}\mid\mathrm{fg}})\) |
| Localization | \(\sum_i q_i^{\mathrm{fg}}q_i^{\mathrm{loc}\mid\mathrm{fg}}\) | \(\sum_i q_i^{\mathrm{fg}}H_{\mathrm{Bern}}(q_i^{\mathrm{loc}\mid\mathrm{fg}})\) |
| Missed object | \(\lambda_x\) | \(H_{\mathrm{Pois}}(\lambda_x)\) |
| 의미 | Expected error amount | Predictive error uncertainty |

따라서 ECA는

\[
\boxed{
\text{How many errors are expected?}
}
\]

를 측정하고,

EUA는

\[
\boxed{
\text{How uncertain are the predicted errors?}
}
\]

를 측정한다.

---

# 20. Classification/localization에서 ECA와 EUA가 선호하는 sample

Foreground probability가

\[
q_i^{\mathrm{fg}}\approx1
\]

이라고 하자.

### \(q_i^{\mathrm{error}}\approx0\)

Detector-derived error predictor가 correct라고 확신한다.

- ECA contribution: 작음
- EUA contribution: 작음

### \(q_i^{\mathrm{error}}\approx0.5\)

Error인지 correct인지 가장 애매하다.

- ECA contribution: 중간
- EUA contribution: 최대

### \(q_i^{\mathrm{error}}\approx1\)

Error라고 거의 확신한다.

- ECA contribution: 최대에 가까움
- EUA contribution: 작음

따라서

\[
\boxed{
\mathrm{ECA}
\rightarrow
\text{confident errors를 선호}
}
\]

하는 반면,

\[
\boxed{
\mathrm{EUA}
\rightarrow
\text{ambiguous error states를 선호}
}
\]

한다.

---

# 21. Poisson entropy에 대한 해석

Missed-object uncertainty에 사용되는

\[
H_{\mathrm{Pois}}(\lambda_x)
\]

는 MOCP parameter 자체의 epistemic uncertainty가 아니다.

즉 MOCP가

\[
\lambda_x
\]

라는 값을 얼마나 자신 있게 추정했는지는 측정하지 않는다.

대신

\[
N_{\mathrm{miss}}
\mid x
\sim
\operatorname{Poisson}(\lambda_x)
\]

라는 **predicted missed-object count distribution의 uncertainty**를 측정한다.

따라서 EUA의 세 components는 모두 predictive uncertainty라는 동일한 관점으로 해석한다.

\[
\boxed{
\begin{aligned}
U_{\mathrm{cls}}
&:\text{ classification-error status predictive uncertainty}\\
U_{\mathrm{loc}}
&:\text{ localization-error status predictive uncertainty}\\
U_{\mathrm{miss}}
&:\text{ missed-object count predictive uncertainty}
\end{aligned}
}
\]

---

# 22. Round 내 EUA 계산 과정

현재 round \(t\)에서 detector와 FDP, CECP, LECP, MOCP가 학습되었다고 하자.

먼저 **labeled pool**의 각 이미지

\[
x\in\mathcal L_t
\]

에서

\[
U_{\mathrm{cls},t}(x),
\quad
U_{\mathrm{loc},t}(x),
\quad
U_{\mathrm{miss},t}(x)
\]

를 predictor outputs로 계산한다.

그 후

\[
\bar U_{\mathrm{cls},t},
\quad
\bar U_{\mathrm{loc},t},
\quad
\bar U_{\mathrm{miss},t}
\]

를 계산하고,

\[
v_{\mathrm{cls},t},
\quad
v_{\mathrm{loc},t},
\quad
v_{\mathrm{miss},t}
\]

를 결정한다.

이 weights를 고정한 상태에서 모든

\[
x\in\mathcal U_t
\]

에 대해

\[
\mathbf u_t(x)
\]

를 계산하고,

\[
\mathbf u_t^{\,w}(x)
=
\mathbf V_t\mathbf u_t(x)
\]

를 얻는다.

최종적으로

\[
\operatorname{EUA}_t(x)
=
\mathbf 1^\top
\mathbf u_t^{\,w}(x)
\]

를 계산하여 candidate images를 선택한다.

따라서 전체 EUA pipeline은

\[
\boxed{
\begin{aligned}
&\text{FDP / CECP / LECP / MOCP prediction}\\
&\rightarrow
\text{Bernoulli entropy for cls/loc}\\
&\rightarrow
\text{Poisson entropy for missed count}\\
&\rightarrow
\text{Error-Uncertainty Profile}\\
&\rightarrow
\text{Labeled-pool scale estimation}\\
&\rightarrow
\text{Scale weighting}\\
&\rightarrow
\text{Error-Uncertainty Amount (EUA)}\\
&\rightarrow
\text{High-EUA candidate images}
\end{aligned}
}
\]
