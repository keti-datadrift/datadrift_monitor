# Error Predictor Feature Design

맞아. 지금 단계에서는 **평균 vs 총합 정도만 비교 가능한 설계로 단순하게 두는 게 낫다.** \(\log(1+n)\) 같은 별도 변환을 넣으면 근거와 ablation만 늘어난다. 그리고 support 개수 자체가 중요한 정보라는 점을 반영하려면 일단 **총합**으로 가는 게 자연스럽다.

FDP는 제외하고, CECP·LECP·MOCP를 모두 **2차원 feature**로 정리하면 다음 구조가 가장 깔끔해 보여.

## 1. 공통 Support Set

Final detection \(i\)의 support set을 기존과 동일하게

\[
\mathcal H_i
\]

라고 하자. Support 개수는

\[
n_i^{\mathrm{sup}}=|\mathcal H_i|
\]

이다.

다만 \(n_i^{\mathrm{sup}}\) 자체를 CECP/LECP의 별도 input으로 넣지는 않고, **support evidence의 총합 안에 자연스럽게 포함**시킨다.

---

# 2. Classification Support Evidence

Final detection \(i\)의 predicted class를 \(\hat c_i\)라고 하면 각 support detection이 \(\hat c_i\)에 부여한 probability를 모두 더한다.

\[
\boxed{
E_i^{\mathrm{cls}}
=
\sum_{\tilde d_k\in\mathcal H_i}
\tilde p_{k,\hat c_i}
}
\]

이 값은

\[
E_i^{\mathrm{cls}}
=
n_i^{\mathrm{sup}}
\mu_i^{\mathrm{cls}}
\]

로 볼 수 있다. 여기서

\[
\mu_i^{\mathrm{cls}}
=
\frac{1}{n_i^{\mathrm{sup}}}
\sum_{\tilde d_k\in\mathcal H_i}
\tilde p_{k,\hat c_i}.
\]

따라서 \(E_i^{\mathrm{cls}}\)는

\[
\boxed{
\text{support quantity}
\times
\text{average classification consistency}
}
\]

를 한 번에 나타낸다.

즉 support가 많을수록 커지고, 그 support들이 final class를 강하게 지지할수록도 커진다.

---

# 3. Localization Support Evidence

완전히 대칭적으로 localization support evidence를

\[
\boxed{
E_i^{\mathrm{loc}}
=
\sum_{\tilde d_k\in\mathcal H_i}
\operatorname{IoU}
(\tilde b_k,\hat b_i)
}
\]

로 정의한다.

마찬가지로

\[
E_i^{\mathrm{loc}}
=
n_i^{\mathrm{sup}}
\mu_i^{\mathrm{loc}},
\]

where

\[
\mu_i^{\mathrm{loc}}
=
\frac{1}{n_i^{\mathrm{sup}}}
\sum_{\tilde d_k\in\mathcal H_i}
\operatorname{IoU}
(\tilde b_k,\hat b_i).
\]

따라서

\[
\boxed{
\text{support quantity}
\times
\text{average localization consistency}
}
\]

이다.

이제 classification과 localization이 정확히 대칭적이다.

\[
\boxed{
E_i^{\mathrm{cls}}
=
\sum \text{classification support}
}
\]

\[
\boxed{
E_i^{\mathrm{loc}}
=
\sum \text{localization support}
}
\]

---

# 4. CECP 입력

Final detection \(i\)의 detector-native confidence를

\[
s_i
\]

라고 두자.

CECP의 입력은

\[
\boxed{
\mathbf z_i^{\mathrm{cls}}
=
\begin{bmatrix}
s_i\\
E_i^{\mathrm{cls}}
\end{bmatrix}
}
\]

로 한다.

즉 CECP는 딱 두 가지를 본다.

\[
\boxed{
\text{Detection Confidence}
+
\text{Classification Support Evidence}
}
\]

Logistic regression이라면

\[
q_i^{\mathrm{cls}}
=
\sigma
\left(
\beta_0^{\mathrm{cls}}
+
\beta_1^{\mathrm{cls}}s_i
+
\beta_2^{\mathrm{cls}}E_i^{\mathrm{cls}}
\right).
\]

따라서 2D visualization도 그대로 가능하다.

- x-axis: \(s_i\)
- y-axis: \(E_i^{\mathrm{cls}}\)

---

# 5. LECP 입력

LECP도 정확히 동일한 형태로

\[
\boxed{
\mathbf z_i^{\mathrm{loc}}
=
\begin{bmatrix}
s_i\\
E_i^{\mathrm{loc}}
\end{bmatrix}
}
\]

로 한다.

즉

\[
\boxed{
\text{Detection Confidence}
+
\text{Localization Support Evidence}
}
\]

이다.

\[
q_i^{\mathrm{loc}}
=
\sigma
\left(
\beta_0^{\mathrm{loc}}
+
\beta_1^{\mathrm{loc}}s_i
+
\beta_2^{\mathrm{loc}}E_i^{\mathrm{loc}}
\right).
\]

여기도

- x-axis: \(s_i\)
- y-axis: \(E_i^{\mathrm{loc}}\)

로 decision boundary를 직접 그릴 수 있다.

그래서 CECP와 LECP가 완전히 짝을 이룬다.

| | CECP | LECP |
|---|---|---|
| Final detection evidence | \(s_i\) | \(s_i\) |
| Support evidence | \(E_i^{cls}\) | \(E_i^{loc}\) |
| Input dimension | 2 | 2 |

이 구조가 지금까지 논의한 것 중 가장 정돈돼 있다.

---

# 6. MOCP는 기존 \(R_{\mathrm{prob}}\)의 mean을 버리는 게 좋음

현재 실패한 MOCP는

\[
[R_{\mathrm{amt}},R_{\mathrm{prob}}]
\]

였고,

\[
R_{\mathrm{prob}}
=
\frac{\sum_k r_k\tilde p_k^{\max}}
{\sum_k r_k}
\]

는 **평균 residual confidence**였다.

그런데 missed-object count를 예측하려는데 평균을 쓰면 중요한 quantity 정보가 사라진다.

예를 들어 residual evidence가 2개이고 평균 confidence가 0.7인 경우와 20개이고 평균 confidence가 0.7인 경우를 \(R_{\mathrm{prob}}\)은 똑같이 본다.

Miss count와 연결하려면 총합이 더 자연스럽다.

---

# 7. Residual Amount

Raw detection \(k\)가 final detections에 의해 설명되는 정도는 기존처럼

\[
\rho_k
=
\max_i
\operatorname{IoU}
(\tilde b_k,\hat b_i)
\]

이고,

\[
\boxed{
r_k=1-\rho_k
}
\]

로 둔다.

첫 번째 image-wise feature는 그대로 residual amount를 사용한다.

\[
\boxed{
R_{\mathrm{amt}}(x)
=
\sum_{k=1}^{M_x}r_k
}
\]

이는

\[
\boxed{
\text{전체적으로 얼마나 많은 unexplained raw evidence가 존재하는가}
}
\]

를 나타낸다.

---

# 8. Residual Confidence Evidence

두 번째 feature는 기존 weighted mean 대신 **weighted sum**으로 바꾼다.

Raw detection 자체의 confidence를 \(\tilde s_k\)라고 하면

\[
\boxed{
R_{\mathrm{conf}}(x)
=
\sum_{k=1}^{M_x}
r_k\tilde s_k
}
\]

로 정의한다.

의미는

\[
\boxed{
\text{final detections로 설명되지 않은 confident detection evidence의 총량}
}
\]

이다.

이제 기존의

\[
R_{\mathrm{prob}}
=
\frac{\sum r_k\tilde s_k}{\sum r_k}
\]

와 달리 residual evidence가 많을수록 값도 커진다.

즉

\[
R_{\mathrm{amt}}
=
\text{unexplained evidence quantity},
\]

\[
R_{\mathrm{conf}}
=
\text{unexplained confidence evidence}
\]

라는 역할이다.

---

# 9. MOCP 입력

따라서 MOCP는

\[
\boxed{
\mathbf z^{\mathrm{miss}}(x)
=
\begin{bmatrix}
R_{\mathrm{amt}}(x)\\
R_{\mathrm{conf}}(x)
\end{bmatrix}
}
\]

를 사용한다.

Poisson regression은

\[
\boxed{
N_{\mathrm{miss}}(x)
\mid\mathbf z^{\mathrm{miss}}(x)
\sim
\operatorname{Poisson}(\lambda_x)
}
\]

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
\gamma_2R_{\mathrm{conf}}(x)
\right)
}
\]

가 된다.

MOCP 역시 input dimension이 2라서 visualization도 가능하다.

- x-axis: \(R_{\mathrm{amt}}\)
- y-axis: \(R_{\mathrm{conf}}\)
- point 색/크기: actual missed-object count

같이 볼 수 있다.

---

# 10. 전체 feature 설계

최종적으로 세 predictor를 나란히 놓으면 꽤 예쁘다.

| Predictor | Feature 1 | Feature 2 |
|---|---|---|
| **CECP** | Detection confidence \(s_i\) | Classification support evidence \(E_i^{cls}\) |
| **LECP** | Detection confidence \(s_i\) | Localization support evidence \(E_i^{loc}\) |
| **MOCP** | Residual amount \(R_{\rm amt}\) | Residual confidence evidence \(R_{\rm conf}\) |

세 predictor 모두 **2-dimensional representation**을 사용한다.

특히 CECP/LECP는 완전히 대칭이다.

\[
\boxed{
\mathbf z_i^{cls}
=
[s_i,E_i^{cls}]^\top
}
\]

\[
\boxed{
\mathbf z_i^{loc}
=
[s_i,E_i^{loc}]^\top
}
\]

with

\[
\boxed{
E_i^{cls}
=
\sum_{k\in\mathcal H_i}
\tilde p_{k,\hat c_i}
}
\]

\[
\boxed{
E_i^{loc}
=
\sum_{k\in\mathcal H_i}
\operatorname{IoU}(\tilde b_k,\hat b_i)
}
\]

이다.

MOCP는

\[
\boxed{
\mathbf z^{miss}(x)
=
[R_{\rm amt}(x),R_{\rm conf}(x)]^\top
}
\]

with

\[
\boxed{
R_{\rm amt}(x)=\sum_k r_k
}
\]

\[
\boxed{
R_{\rm conf}(x)=\sum_k r_k\tilde s_k
}
\]

로 간다.

한 가지 주의할 점은 \(R_{\rm conf}\)가 \(R_{\rm amt}\)와 강하게 상관될 가능성이 있다는 거야. 그래도 **현재 \(R_{\rm prob}\) mean이 거의 기능하지 않았다는 결과가 있으니, 다음 단순한 개선안으로 total confidence evidence를 시험하는 것은 충분히 합리적**이야. 지금 단계에서는 spatial clustering 같은 새로운 feature를 더 얹기보다 이 깔끔한 2-feature 설계부터 predictor 성능을 다시 확인하는 게 좋다.
