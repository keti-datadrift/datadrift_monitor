# Feature Export Head Split

## Purpose

ALOD no longer uses one configurable feature-export head for both Core-set and
PPAL. The artifact contracts are method-specific, so the producer classes and
MMDetection configs are now method-specific as well.

## Heads

`RetinaImageFeatureExportHead` is for Core-set.

- Input feature source: RetinaNet classification tower feature maps.
- Exported artifact keys: `image_ids`, `image_features`, `metadata_json`.
- Image feature definition: spatial mean pooling per FPN level, then averaging
  across levels.
- It does not allocate detection queues and does not export detection arrays.

`RetinaDetectionFeatureExportHead` is for PPAL.

- Input feature source: RetinaNet classification tower feature maps, matching
  PPAL `RetinaHeadFeat`.
- Exported artifact keys: `image_ids`, `det_labels`, `det_scores`,
  `det_features`, `det_valid`, `metadata_json`.
- It samples detection features after NMS from the selected FPN level with
  `get_inter_feats`.
- It does not export `image_features`.

The shared RetinaNet decoding and NMS flow lives in `feature_export_utils.py`.

## Config Routing

- Core-set uses `image_feature_infer_config`.
- PPAL uses `detection_feature_infer_config`.

The old `feature_infer_config`, `RetinaFeatureExportHead`, and
`export_detection_features` switch are removed from the active code path.

## Artifact Loading

`methods.common.feature_artifacts.load_feature_artifact(...)` now supports
separate image-feature and detection-feature requirements:

- Core-set loads with image features required.
- PPAL loads with detection features required and image features not required.

This keeps PPAL's CCMS distance computation detection-based while allowing
Core-set artifacts to stay compact.
