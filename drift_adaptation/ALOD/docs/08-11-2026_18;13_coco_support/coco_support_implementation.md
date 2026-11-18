# RetinaNet + COCO active-learning support

## Scope

ALOD의 모든 현재 method를 `RetinaNet + COCO 2017` 조합에서 동일한
PAL-paper experiment protocol로 실행할 수 있도록 dataset-aware catalog,
automatic input preparation, MMDetection configs, runner metrics, dashboard를
연결했다.

지원 method는 다음과 같다.

- PPAL: DCUS + CCMS
- PAL full: LIUS + GUIDE
- PAL LIUS-only
- ECPAL ECA-only, EUA-only, ECA-full, EUA-full
- Core-set
- MIAL
- Random
- Entropy

Method의 scoring/selection 구현은 변경하지 않았다. COCO 지원은 dataset
protocol, model config, input pool, evaluation metric을 method-independent
계층에서 공급한다.

## COCO protocol

| Item | Value |
| --- | --- |
| Training pool | COCO 2017 `train2017` |
| Validation set | COCO 2017 `val2017` |
| Initial labeled pool | 2,365 images, 2% |
| Per-round acquisition budget | 2,365 images, 2% |
| Evaluated labeled ratios | 2%, 4%, 6%, 8%, 10% |
| Train/eval rounds | 5 |
| Detector | RetinaNet R50-FPN |
| Input scale | 1333 x 800, keep ratio |
| Batch size | 1 image/GPU |
| Optimizer | SGD, LR 0.01, momentum 0.9, weight decay 0.0001 |
| Schedule | 26 epochs, decay at epoch 20 by gamma 0.1 |
| Evaluation | COCO bbox AP |
| Aggregated metrics | `bbox_mAP`, `bbox_mAP_50`, `bbox_mAP_75` |

Round 1은 initial 2% pool로 학습하고 평가한다. 각 round의 acquisition은
다음 round pool을 만든다. 기존 ALOD 정책과 같이 round 5의 acquisition도
보존하므로, 10% 평가가 끝난 뒤 분석용 12% pool artifact가 추가로 생성된다.

## Data layout and preparation

사용자는 원본 COCO 파일만 다음 위치에 둔다.

```text
data/coco/train2017/
data/coco/val2017/
data/coco/annotations/instances_train2017.json
data/coco/annotations/instances_val2017.json
```

`tools/run_active_learning.py`가 실행 전에 source path를 확인하고 요청된
seed별 initial pool을 자동 생성한다.

```text
data/active_learning/coco/
  coco_2365_labeled_seed_0.json
  coco_2365_unlabeled_seed_0.json
  coco_2365_labeled_seed_1.json
  coco_2365_unlabeled_seed_1.json
  coco_2365_labeled_seed_2.json
  coco_2365_unlabeled_seed_2.json
```

COCO에서는 실제 experiment seed마다 독립적으로 `RandomState(seed)`를
사용한다. 따라서 `--seeds 0 1 2`는 서로 다른 initial 2% pool을 사용한다.
동일 seed split이 이미 있으면 다시 쓰지 않는다. Labeled JSON은 해당
image의 GT annotation을 포함하고 unlabeled JSON은 image/category metadata만
포함한다. 기존 split은 재사용 전에 labeled count, image ID 중복,
labeled/unlabeled overlap, 전체 COCO train image count, annotation 참조를
검증한다.

`data/coco` 또는 그 하위 source path는 repository 안의 junction/symlink를
통해 외부 dataset 저장소를 가리킬 수 있다. Dataset input만 이 동작을
허용하며 config, output, checkpoint 경로는 계속 repository 내부로 제한한다.
`code_refs/`를 runtime input으로 사용하는 것은 허용하지 않는다.

VOC는 기존 재현성을 보존하기 위해 이전과 동일한 indexed split과 RNG
흐름을 유지한다. 공통 COCO-style subset writing만
`tools/common/dataset_pools.py`로 이동했다.

PAL full이 필요한 Google ViT embedding cache는 dataset별로 분리된다.

```text
work_dirs/pal_embeddings/voc_google_vit_embeddings.npy
work_dirs/pal_embeddings/coco_google_vit_embeddings.npy
```

Embedding 대상 image 목록은 initial split 두 파일이 아니라 dataset oracle
JSON 하나에서 읽는다. 이렇게 하면 seed와 관계없이 전체 train pool을 한 번
계산하고 재사용한다.

## Catalog ownership

`configs/catalog/datasets.py`가 dataset protocol의 단일 source다.

- round count, budget, percentage range
- oracle/train/validation paths
- seed-specific initial-pool path templates
- MMDetection evaluation metric
- aggregate summary metric names
- automatic dataset preparation declaration

`configs/catalog/detectors.py`는 `(detector, dataset)`에 맞는 train/inference
config path를 선택한다. `configs/catalog/methods.py`는 method 고유
hyperparameter와 artifact 이름만 담당하며, budget과 output protocol 이름은
dataset catalog에서 받는다.

Preset은 catalog 조합으로 자동 생성된다. 기존 VOC preset 11개와 COCO preset
11개가 함께 노출된다.

## MMDetection configs

`configs/alod_mmdet/retinanet_coco_base.py`는 COCO 공통 model/data pipeline을
정의한다. 별도 `CLASSES` tuple을 재정의하지 않고 MMDetection의 표준
`CocoDataset` class mapping을 사용한다.

Method별 최소 config는 다음과 같다.

| Config | Consumer |
| --- | --- |
| `retinanet_coco_train_quality_ema_26e.py` | PPAL, PAL, ECPAL, Core-set, Random, Entropy |
| `retinanet_coco_infer_uncertainty.py` | PPAL DCUS, Entropy |
| `retinanet_coco_infer_detection_features.py` | PPAL CCMS |
| `retinanet_coco_infer_image_features.py` | Core-set |
| `retinanet_coco_infer_pal_detections.py` | PAL |
| `retinanet_coco_infer_ecpal_detections.py` | ECPAL |
| `retinanet_coco_train_mial.py` | MIAL |

Inference config는 모두 train pool의 real COCO image ID를
`AddImageIdToMeta`로 전달한다. 따라서 method별 filename-derived ID 변환 없이
공통 pool update 코드와 직접 연결된다.

## Runner and metrics

Runner는 각 seed의 round 0을 만들 때 catalog의 seed path template를 해석한다.
Train/inference 계획은 COCO JSON의 image count로 진행률 total을 계산한다.
Evaluation command는 dataset의 `eval_metric`을 사용한다.

- VOC: `--eval mAP`
- COCO: `--eval bbox`

`aggregate_summary.json`도 dataset별 metric 목록을 사용한다. Dashboard는
VOC `mAP`/`AP50`와 COCO `bbox_mAP`/`bbox_mAP_50`/`bbox_mAP_75`를 동일한
validation curve UI에서 선택하고 비교할 수 있다. `labeled_images` x축은
round N acquisition 이후 pool이 아니라 round N train/eval에 실제 사용된
pool을 읽는다. `method_arg`도 유지하므로 PAL LIUS와 ECPAL ECA/EUA variants가
서로 다른 curve/filter 이름으로 표시된다.

## Commands

Three-seed PAL run:

```powershell
python -B tools/run_active_learning.py --method pal --detector retinanet --dataset coco --gpus 1 --seeds 0 1 2
```

Parallel seed pipelines with CPU affinity:

```powershell
python -B tools/run_active_learning.py --method pal --detector retinanet --dataset coco --gpus 1 --seeds 0 1 2 --seed-workers 3 --seed-cpu-cores 4
```

Other methods use the same command shape by changing only `--method`, for
example `ppal`, `pal:lius`, `ecpal:eca-only`, `ecpal:eua-only`,
`ecpal:eca-full`, `ecpal:eua-full`, `coreset`, `mial`, `random`, or `entropy`.

## Validation

Implementation validation covers:

- Python compilation of the catalog, preparation, runner, dashboard, and all
  eight COCO MMDetection config files
- resolution of all 11 COCO method configs with round count 5, budget 2,365,
  `bbox` evaluation, and expected output directory names
- listing of 22 total VOC/COCO presets
- synthetic COCO seed split membership, metadata, no-annotation unlabeled
  pools, existing-pair integrity, idempotent rerun, and missing-path errors
- VOC indexed split RNG and membership regression
- dashboard labeled-image round alignment and method-variant discovery
- MMDetection config inheritance and model graph construction for all eight
  COCO configs in an environment containing the project MMDetection stack

Full COCO training/inference was not run locally. It remains the final runtime
validation because the local base environment used for integration does not
contain `mmcv`, and a complete five-round COCO run is a long GPU experiment.
