# SACIL Preliminary Class Orders

이 디렉터리는 SACIL의 **초기 아이디어 타당성 검증**에 사용할 class order를
기계가 읽을 수 있는 JSON으로 고정한다.

현재 원칙은 데이터셋마다 하나의 대표 protocol과 하나의 대표 order만 사용하는
것이다. 논문용 multi-order robustness, 여러 step 크기, fixed-total memory,
ImageNet-1K scalability는 초기 검증 이후로 미룬다.

| Dataset | Manifest | Preliminary protocol | Order 출처 |
|---|---|---|---|
| CIFAR-100 | `cifar100_b50_t10_afc_order1.json` | 50 base + 10 steps × 5 classes | PODNet/AFC published order 1 |
| ImageNet-100 | `imagenet100_b50_t10_afc_order1.json` | 50 base + 10 steps × 5 classes | PODNet/AFC/R-DFCIL 공통 공개 order |
| CUB-200-2011 | `cub200_b0_t10_seed1993.json` | 10 equal tasks × 20 classes | SACIL NumPy seed-1993 order |
| FGVC-Aircraft-100 | `aircraft100_b0_t10_seed1993.json` | 10 equal tasks × 10 classes | SACIL NumPy seed-1993 order |

## JSON 해석

- `class_order`는 원래 dataset class ID의 전체 permutation이다.
- `session_slices`의 `start`는 inclusive, `stop`은 exclusive이다.
- 실제 session class 목록은 `class_order[start:stop]`으로 얻는다.
- CIFAR-100과 ImageNet-100의 `session_id=0`은 50-class base session이다.
- CUB와 Aircraft는 모든 task 크기가 같으며 `session_id=0`도 첫 20/10-class
  task이다.

```python
import json

with open(manifest_path, encoding="utf-8") as f:
    protocol = json.load(f)

order = protocol["class_order"]
session_classes = [
    order[item["start"]:item["stop"]]
    for item in protocol["session_slices"]
]
```

## Label mapping

- CIFAR-100: `datasets/cifar-100-python/meta`의 `fine_label_names`
- ImageNet-100: `datasets/ImageNet100/metadata/class_index.csv`
- CUB-200-2011: `datasets/CUB_200_2011/classes.txt`; manifest ID는 원본
  1-based class ID에서 1을 뺀 값
- FGVC-Aircraft: `datasets/fgvc-aircraft-2013b/data/variants.txt`; manifest
  ID는 0-based line index

## 사용 시 주의사항

1. `class_order`를 다시 shuffle하지 않는다.
2. 모든 baseline과 SACIL에 같은 manifest를 전달한다.
3. dataset loader 내부에서 label을 재매핑했다면 원래 ID와의 대응을 검사한다.
4. 결과 파일에 manifest의 `protocol_id`와 SHA-256을 기록한다.
5. CUB/Aircraft order는 특정 논문의 exact published permutation이라고
   주장하지 않는다. 대표 task 크기를 사용한 SACIL 고정 진단 order이다.

현재 manifest의 SHA-256은 [`SHA256SUMS`](SHA256SUMS)에 고정되어 있다.
