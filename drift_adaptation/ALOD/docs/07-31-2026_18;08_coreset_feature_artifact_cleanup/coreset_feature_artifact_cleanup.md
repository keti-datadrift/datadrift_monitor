# Core-set Feature Artifact Cleanup

## Scope

This cleanup separates Core-set and PPAL feature artifact needs while keeping
their RetinaNet feature source aligned.

## PPAL Reference Check

The PPAL reference implementation does not define a separate image-level
embedding for CCMS. PPAL diversity uses detection-level features:

- `RetinaHeadFeat` samples detection features from the RetinaNet classification
  tower feature maps.
- `get_img_score_distance_matrix_slow(...)` computes an image-distance matrix
  from detection labels, detection scores, and detection features.
- `DiversitySampler` consumes the precomputed image-distance matrix.

Therefore ALOD keeps PPAL's detection-level path intact. PPAL feature inference
now uses `RetinaDetectionFeatureExportHead` and exports detection labels,
scores, features, and validity masks.

## Core-set Artifact Policy

Core-set only needs one image-level feature vector per image for greedy
k-center selection. Core-set feature inference now uses
`RetinaImageFeatureExportHead` and writes only:

- `image_ids`
- `image_features`
- `metadata_json`

This avoids storing `det_labels`, `det_scores`, `det_features`, and `det_valid`
for Core-set runs.

The Core-set image feature is an ALOD image-level aggregate from the same
RetinaNet classification-tower feature source used by PPAL detection features:
spatial mean pooling per FPN level followed by level averaging.

## Candidate Records

Core-set candidate records no longer duplicate selection state. The common
candidate artifact owns:

- `selected`
- `selection_rank`

Core-set k-center records now keep method-specific scoring information only:

- `score`: initial distance to the labeled center set
- `components.final_min_distance`: distance after selected centers are added

## Core-set Distance Policy

Core-set follows the reference code's raw squared Euclidean distance behavior.
The user-facing `coreset_normalize_features` option was removed. The remaining
Core-set batch-size values are implementation controls for memory and speed,
not acquisition hyperparameters.
