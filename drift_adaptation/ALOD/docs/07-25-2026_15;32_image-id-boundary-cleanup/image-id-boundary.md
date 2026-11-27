# Image ID Boundary

## Goal

ALOD method code uses one canonical image identity:

```text
canonical image id = COCO annotation images[].id
```

Detector artifacts may expose image ids in a different form. Those ids must be
converted to canonical ids at the artifact boundary before method acquisition
logic updates labeled or unlabeled pools.

## Artifact Inventory

| Artifact | Producer | Image id source | Canonical |
| --- | --- | --- | --- |
| VOC oracle JSON | `tools/common/voc.py` | `images[].id` assigned as stable `0..N` ids | Yes |
| Initial labeled/unlabeled pools | `tools/common/voc.py` | oracle `images[].id` | Yes |
| PPAL uncertainty bbox JSON | `mmdet/alod/datasets/al_voc.py` | `self.img_ids[idx]` from annotation JSON | Yes |
| PAL bbox JSON | `mmdet/alod/datasets/pal_coco.py` | `self.img_ids[idx]` from annotation JSON | Yes |
| Entropy bbox JSON | `mmdet/alod/datasets/al_voc.py` or `al_coco.py` | `self.img_ids[idx]` from annotation JSON | Yes |
| PAL embedding cache | `methods/pal/vit_embeddings.py` and `methods/pal/embeddings.py` | COCO `image['id']` | Yes |
| PPAL `image_dis.npy` | `mmdet/alod/models/retinanet/ppal_feat_head.py` | integer parsed from image filename stem | No |

## Common Boundary

`methods/common/image_identity.py` owns method-runtime image identity helpers:

- `normalize_image_id(...)`
- `canonical_image_ids(...)`
- `image_file_stem_id(...)`
- `build_image_id_alias_index(...)`
- `map_artifact_image_ids_to_canonical(...)`
- `validate_image_ids_subset(...)`

This module does not import MMDetection or runner code.

## PPAL CCMS

PPAL's `RetinaHeadFeat` follows the original PPAL behavior and stores ids from
the image filename stem. For VOC, examples are:

- `VOC2007/JPEGImages/000001.jpg -> 1`
- `VOC2012/JPEGImages/2007_000033.jpg -> 2007000033`

ALOD's VOC0712 oracle uses canonical COCO image ids, so
`methods/ppal/inference.py` maps `image_dis.npy` ids to canonical ids before
`methods/ppal/ccms.py` performs CCMS selection.

The PPAL algorithm is unchanged: uncertainty candidate generation, feature
distance computation, and CCMS centroid selection are the same process. Only
the artifact id representation is normalized before pool JSON files are
updated.

## Placement Rule

- Put generic image id and alias mapping utilities in `methods/common`.
- Keep PPAL artifact names such as `image_dis.npy` in `methods/ppal/inference.py`.
- Keep CCMS itself focused on distance-matrix selection, not filename parsing.
- Do not change `mmdet/alod/models/retinanet/ppal_feat_head.py` only to hide
  this issue; that file intentionally remains close to the original PPAL
  detector artifact behavior.

## Validation

Use compile checks and a small inline mapping check after boundary edits:

```powershell
python -m compileall -q methods tools configs
python -B tools/run_active_learning.py --help
```

Inline check:

```python
from methods.common.image_identity import map_artifact_image_ids_to_canonical

images = [
    {'id': 0, 'file_name': 'VOC2007/JPEGImages/000001.jpg'},
    {'id': 2, 'file_name': 'VOC2012/JPEGImages/2007_000033.jpg'},
]
assert map_artifact_image_ids_to_canonical([1, 2007000033], images) == [0, 2]
```
