# ALOD Multi-run Metrics Dashboard

이 문서는 ALOD 실험 결과를 TensorBoard처럼 로컬 웹 페이지에서 비교하기 위한 dashboard 구현 내용을 정리한다.

## 목적

Dashboard의 목적은 `work_dirs/` 아래에 저장된 기존 실험 결과를 **read-only**로 읽어서 다음 curve를 보는 것이다.

- train loss curve
- train learning-rate curve
- validation mAP/AP50 curve
- seed 간 비교
- method 간 비교
- timestamp run 간 비교

실험 저장 방식은 수정하지 않는다. `tools/run_active_learning.py`의 logging/output logic도 수정하지 않는다.

## 실행

Dashboard dependency는 optional이다.

```powershell
pip install -r requirements/dashboard.txt
```

기본 실행은 다음과 같다.

```powershell
python tools/run_metrics_dashboard.py
```

기본적으로 `work_dirs`를 scan한다. 다른 root를 보고 싶으면 다음처럼 실행할 수 있다.

```powershell
python tools/run_metrics_dashboard.py --work_dir other_work_dirs
```

`--run_dir`도 같은 의미의 alias로 지원한다.

정상 SSL 환경에서는 Streamlit CLI를 직접 실행해도 된다.

```powershell
streamlit run tools/view_metrics.py
```

다만 일부 Windows conda 환경에서는 Streamlit/Tornado import 중 Windows certificate store를 읽다가
`ssl.SSLError: [ASN1: NOT_ENOUGH_DATA]`가 발생할 수 있다. 이 경우 `tools/run_metrics_dashboard.py` launcher가
Streamlit import 전에 narrow SSL fallback을 적용하므로 launcher 실행을 사용한다.

## 자동 Scan 규칙

`tools/common/metrics_scanner.py`는 `work_dirs` 아래에서 다음 구조를 자동으로 찾는다.

```text
work_dirs/
  <experiment_name>/
    <MM-DD-YYYY_HH;mm>/
      aggregate_summary.json
      seed_0/
        run_summary.json
        round_01/
      seed_1/
      seed_2/
```

우선순위는 다음과 같다.

1. `<run_dir>/aggregate_summary.json`
2. `<run_dir>/seed_*/run_summary.json`
3. legacy flat run의 `<run_dir>/run_summary.json`

발견된 run은 내부적으로 다음 정보로 정리한다.

```text
method
detector
dataset
experiment
run_id
seeds
rounds
budget
status
path
```

## Dashboard 구성

### Run Selection

Sidebar에서 다음 항목을 선택한다.

- method
- detector
- dataset
- experiment
- timestamp run

명령어에 여러 run path를 길게 입력하지 않는다. `work_dirs` 전체를 scan하고, 사용자가 sidebar에서 보고 싶은
것만 켜고 끄는 방식이다.

### Validation Curves

Validation chart는 active learning round 단위의 metric을 보여준다.

Source 우선순위:

1. `aggregate_summary.json`
2. `seed_*/round_*/eval_*.json`

지원 항목:

- `mAP`
- `AP50`
- seed별 curve
- mean curve
- mean +/- std band
- x-axis: `round` 또는 `labeled_images`

Validation curve는 epoch-level validation curve가 아니다. ALOD는 각 active learning round가 끝난 뒤 eval을
수행하므로, validation curve의 x축 단위는 active learning round이다.

### Train Curves

Train chart는 round 내부 training log를 보여준다.

Source 우선순위:

1. `seed_*/round_XX/*.log.json`
2. `seed_*/round_XX/logs/train.log`

지원 항목:

- `loss`
- `loss_cls`
- `loss_bbox`
- `lr`
- `grad_norm`
- log JSON에 존재하는 기타 numeric key

Train curve는 iteration-level curve이다. Sidebar에서 method/run/seed/round/loss key를 선택해 여러 curve를
한 그래프 안에서 비교할 수 있다.

## 해석 주의점

Validation mAP/AP50은 method 비교에 가장 직접적인 지표이다.

Train loss는 selected labeled pool이 method마다 다르기 때문에 method 성능의 직접 비교 지표로 보면 안 된다.
예를 들어 PAL과 PPAL이 같은 RetinaNet train config를 쓰더라도 round별 labeled data가 다르므로, train loss
scale 차이는 acquisition method의 좋고 나쁨을 바로 뜻하지 않는다.

Train loss는 주로 다음 용도로 보는 것이 좋다.

- 학습이 정상적으로 수렴했는지 확인
- seed 간 training stability 확인
- 특정 round에서 loss spike가 있었는지 확인
- learning rate schedule 확인

## 구현 파일

```text
tools/view_metrics.py
tools/run_metrics_dashboard.py
tools/common/metrics_scanner.py
tools/common/metrics_viewer.py
tools/common/metrics_logs.py
requirements/dashboard.txt
```

각 파일 역할은 다음과 같다.

| File | Role |
|---|---|
| `tools/view_metrics.py` | Streamlit UI entrypoint |
| `tools/run_metrics_dashboard.py` | Streamlit launcher with Windows SSL-store fallback |
| `tools/common/metrics_scanner.py` | `work_dirs` run discovery |
| `tools/common/metrics_viewer.py` | validation metric/labeled count parser |
| `tools/common/metrics_logs.py` | train loss/lr log parser |
| `requirements/dashboard.txt` | optional dashboard dependencies |

## Read-only 원칙

Dashboard는 다음 파일들을 읽기만 한다.

- `aggregate_summary.json`
- `run_summary.json`
- `round_summary.json`
- `eval_*.json`
- `*.log.json`
- `logs/train.log`
- `round_*/annotations/new_labeled.json`

실험 결과, checkpoint, dataset, annotation split은 수정하지 않는다.

## 향후 확장

현재 구현은 train/validation curve에 집중한다. 다음 기능은 추후 추가할 수 있다.

- error count curve overlay
- candidate score distribution
- selected image overlap 분석
- method별 final mAP summary card
- 진행 중인 log의 자동 refresh 주기 조절
