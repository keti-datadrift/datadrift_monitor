# PAL LIUS Export Integration

## Implemented Slice

- Added `RetinaHeadPAL` for RetinaNet inference with PAL settings.
- Added `PALCocoDataset` to write `pre_nms_count` without changing PPAL's uncertainty JSON schema.
- Extended `ALRetinaNet.simple_test` to accept PPAL uncertainty tuples, PAL pre-NMS-count tuples, and ordinary detector tuples.
- Wired `tools/run_active_learning.py --method pal` to run PAL inference on both labeled and unlabeled pools before LIUS acquisition.
- Added `configs/voc_active_learning/al_inference/retinanet_pal_lius.py`.
- Added `configs/experiments/pal_lius_retinanet_voc_smoke.py` for user-run smoke validation.

## PAL Inference Settings

- `nms_pre=1000`
- `score_thr=0.3`
- `nms.iou_threshold=0.5`
- `pre_nms_count_iou_thr=0.5`

## Output Files

Each round writes:

- `pal_labeled_detections.bbox.json`
- `pal_unlabeled_detections.bbox.json`

Each detection record contains:

- `image_id`
- `bbox`
- `score`
- `pre_nms_count`
- `category_id`
- `class_scores` when exported by `RetinaHeadPAL`

## Remaining PAL Work

- Implement full GUIDE terms: CWIE, RCDI, RCSP.
- Add image embedding extraction/cache for RCSP.
- Add full PAL score and duplicate/fill policy diagnostics.
