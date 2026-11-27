# Seed-Specific Initial-Pool Protocol

## 목적과 불변조건

ALOD의 initial labeled/unlabeled pool은 다음 규칙으로만 결정한다.

```text
initial_pool = f(dataset, seed)
```

Detector, active-learning method, method variant, seed 실행 방식은 initial
pool의 경로나 이미지 구성에 영향을 주지 않는다. 따라서 같은 dataset과
seed를 사용하는 모든 method는 완전히 같은 source pool에서 시작한다.

## 파일명 규칙

Runner가 요청된 seed에 대해 다음 source 파일을 자동으로 준비하고 이후
실행에서 검증하여 재사용한다.

```text
data/active_learning/voc/voc_827_labeled_seed_{seed}.json
data/active_learning/voc/voc_827_unlabeled_seed_{seed}.json

data/active_learning/coco/coco_2365_labeled_seed_{seed}.json
data/active_learning/coco/coco_2365_unlabeled_seed_{seed}.json
```

VOC는 827장, COCO는 2,365장을 initial labeled pool로 선택한다. 각 seed의
membership은 dataset oracle과 labeled 수, 실제 seed를 입력으로 하는
deterministic sampling으로 생성한다.

## 준비 및 검증 흐름

1. Runner가 CLI에서 요청된 seed 목록을 확정한다.
2. Dataset preparation이 seed worker 실행 전에 필요한 source pool을 한 번
   준비한다.
3. 기존 pool이 있으면 덮어쓰지 않고 membership과 구조를 검증한다.
4. 각 seed의 source pool을 해당 실행의 `seed_*/round_00/annotations`로
   복사한다.
5. 기존 round 0이 있으면 source와 image-ID fingerprint가 같은지 확인한다.

각 labeled/unlabeled pair는 다음 조건을 만족해야 한다.

- image ID가 각 pool 안에서 중복되지 않는다.
- labeled와 unlabeled pool이 서로 겹치지 않는다.
- 두 pool의 합집합이 oracle의 전체 이미지 집합과 같다.
- labeled image 수가 dataset protocol과 같다.
- annotation과 category 구조가 oracle과 일치한다.
- 실제 labeled membership이 파일명에 기록된 seed의 deterministic 결과와
  같다.

검증에 실패한 기존 파일은 자동 교체하지 않고 실행을 중단한다.

## 실행 방식과 Method 공유

다음 세 실행 방식에서 같은 dataset/seed의 initial pool은 동일하다.

- single-seed 실행
- multi-seed 순차 실행
- `--seed-workers`를 사용한 multi-seed 병렬 실행

이 규칙은 PPAL, PAL full/LIUS, ECPAL ECA-only/EUA-only/ECA-full/EUA-full,
Core-set, MIAL, Random, Entropy에 공통으로 적용된다. Method 구현은 initial
pool을 생성하거나 바꾸지 않으며, 공통 runner가 전달한 round 0부터 각
method 고유 acquisition을 수행한다.

## 실행 기록

각 seed의 `run_summary.json`에는 전체 image ID 목록을 중복 저장하지 않고
다음과 같은 compact provenance를 기록한다.

- initial-pool policy와 seed
- labeled/unlabeled image 수
- canonical image-ID 집합의 SHA-256 fingerprint

이 fingerprint를 이용하면 서로 다른 method 결과가 같은 dataset/seed
source pool을 사용했는지 JSON formatting이나 annotation 순서와 무관하게
확인할 수 있다.

## Legacy VOC 결과

기존 VOC 파일인 `voc_827_labeled_1.json`과
`voc_827_unlabeled_1.json`은 삭제하거나 수정하지 않지만 새 실행에서는
사용하지 않는다. 기존 `_1` split의 membership은 새 `seed_0` split과
동일하더라도, 과거 3-seed 실험은 모든 training seed가 하나의 initial
pool을 공유했다.

기존 `work_dirs` 결과도 변경하지 않는다. 따라서 protocol 의미는 다음과
같이 구분해야 한다.

```text
과거 VOC 3-seed 결과: 동일 initial pool + 서로 다른 training/acquisition seed
신규 3-seed 결과:    seed별 initial pool + 대응하는 training/acquisition seed
```

두 결과는 서로 다른 initial-pool protocol이므로 동일한 3-seed 실험으로
간주하거나 통계값을 직접 합쳐서는 안 된다.
