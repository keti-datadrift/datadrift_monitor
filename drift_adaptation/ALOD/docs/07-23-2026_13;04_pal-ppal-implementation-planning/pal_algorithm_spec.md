# PAL Algorithm Spec for RetinaNet + PASCAL VOC

## Goal

Implement PAL as an acquisition method on top of the local PPAL-style RetinaNet + PASCAL VOC active learning protocol. The implementation should train and evaluate the detector exactly like PPAL, then replace PPAL's DCUS plus diversity acquisition with PAL's LIUS plus GUIDE acquisition.

PAL should operate from inference outputs rather than training internals.

## Required Inference Output

For each post-NMS detection, store:

- `image_id`
- `bbox`
- `category_id`
- `score`
- `pre_nms_count`
- optionally, class probability vector or per-class scores if available
- for labelled inference only, `is_tp` after ground-truth matching

PAL needs inference on both:

- labelled set: to train class-specific logistic classifiers.
- unlabelled set: to score candidate detections/images.

PAL paper RetinaNet LIUS inference settings:

- pre-NMS boxes: 1000.
- NMS IoU threshold: 0.5.
- score threshold: 0.3.

## LIUS

Function: `compute_pre_nms_count(pre_nms_boxes, final_detections, iou_threshold) -> count_per_detection`

- Extract pre-NMS boxes for the full image.
- Run NMS to get final detections.
- Assign pre-NMS boxes to each final detection using an IoU threshold.
- The assigned box count is `pre_nms_count`.

Implementation decision for first version:

- Use IoU threshold `0.5` for assigning pre-NMS boxes to final detections unless later paper/code evidence says otherwise.

Function: `match_labeled_detections_to_gt(detections, gt, iou_threshold) -> is_tp`

- For labelled detections, determine TP/FP using class-aware detection-to-GT matching.
- Suggested first version: VOC-style class match plus IoU >= 0.5, one detection matched per GT.

Function: `train_class_logistic_classifiers(labeled_detections) -> classifiers_by_class`

For each class `c`, train a binary logistic classifier:

```text
x = [pre_nms_count, confidence]
y = 1 for TP, 0 for FP
P(Y = 1 | x) = 1 / (1 + exp(-(b0 + b1*x1 + b2*x2)))
```

Function: `compute_lius(classifier, detection) -> float`

For an unlabelled detection `j` in image `I`:

```text
p = P(Y_j = 1 | x_j)
LIUS(I_j) = -p * log(p) - (1 - p) * log(1 - p)
```

The PAL paper writes LIUS as Shannon entropy over `Y_j in {0, 1}`. Use binary entropy with numerical clipping.

## Class Budget

Function: `compute_class_weights(labeled_counts, unlabeled_counts) -> r_c`

PAL Eq. 5:

```text
r_c = 1 - 0.5 * (n_c,l / N_l + n_c,u / N_u)
```

where:

- `n_c,l`: labelled instance count for class `c`.
- `n_c,u`: unlabelled detected instance count for class `c`.
- `N_l`: total labelled detections/instances used for the budget calculation.
- `N_u`: total unlabelled detections/instances used for the budget calculation.

Function: `allocate_class_budget(r_c, n_c,u, total_budget) -> b_c`

PAL Eq. 6:

```text
b_c = min(n_c,u, b * r_c / sum_c(r_c))
sum_c(b_c) = b
```

Implementation decision for first version:

- Compute floating budgets, floor them, then distribute the remaining slots by largest fractional remainder among classes with available unlabelled candidates.
- Ensure deterministic tie-breaking by class index.
- Classes with no valid classifier or no unlabelled detections receive zero direct budget, then their unused budget is redistributed.

## Candidate Generation

Function: `build_class_candidates(unlabeled_detections, b_c) -> candidates_by_class`

- For each class `c`, sort detections by LIUS descending.
- Convert detections to image candidates.
- Select top `2 * b_c` candidate images containing the highest-LIUS instances for that class.
- Retain the detection `j` that caused image `I` to become a candidate, because final Eq. 11 uses `LIUS(I_j)`.

Implementation decision for first version:

- If one image has multiple detections for the same class, keep the maximum LIUS detection for that class/image pair.

## GUIDE

GUIDE combines three image-level terms.

### CWIE

Function: `compute_cwie(image_detections, class_weights) -> float`

PAL Eq. 7:

```text
CWIE(I) = - sum_{i in objects} r_{c_i} * sum_{j in classes} p_ij * log(p_ij)
```

where `r_{c_i}` is the class weight from Eq. 5.

Implementation decision for first version:

- Prefer full class score vectors from RetinaNet before class argmax if practical.
- If full vectors are not available in the first PAL inference head, use binary entropy of each final detection confidence as a temporary fallback and document it as a non-final approximation.

Normalize CWIE over the `2 * b_c` class candidates with PAL Eq. 8:

```text
X_std = (X - X_min) / (X_max - X_min)
```

The paper states `X_min = 0` and `X_max` is the maximum candidate score.

### RCDI

Function: `compute_rcdi(image_unique_classes, class_weights) -> float`

PAL Eq. 9:

```text
RCDI(I) = sum_{k in K} r_k
```

where `K` is the set of unique predicted classes in image `I`. Normalize with the same min-max scheme as CWIE over the per-class candidate set.

### RCSP

Function: `compute_rcsp(ranked_candidates, image_embeddings) -> float`

- Rank candidates for each class by LIUS descending.
- Assign the top-ranked image a diversity score of 1.
- For candidate rank `i > 1`, compare image embedding `e_i` to embeddings of all higher-ranked images.

PAL Eq. 10:

```text
RCSP(I_i) = 1 - max_{m in [1, i-1]} cosine(e_i, e_m)
```

Implementation decision for first version:

- Use a cached image embedding table keyed by image id.
- Prefer Google ViT if dependency/setup is available.
- If Google ViT is not available for the first smoke test, allow an explicitly configured encoder substitute and record that the result is not paper-faithful.

## Final Selection Score

Function: `compute_pal_score(lius, cwie, rcdi, rcsp) -> float`

PAL Eq. 11:

```text
Score(I) = alpha * LIUS(I_j) + gamma * RCSP(I) + beta * (CWIE(I) + RCDI(I))
```

Use paper settings:

```text
alpha = 0.9
beta = 0.04
gamma = 0.02
d = 0.1
```

For each class, select the top `b_c` images ranked by final PAL score.

## Duplicate Image Handling

The paper states that the top `b_c` images per class are sent to the oracle, but it does not fully specify how to handle the same image selected by multiple class budgets.

Suggested first faithful implementation:

- Maintain a global selected image set.
- Process classes in descending `r_c` order, with deterministic class-index tie-breaking.
- Add an image only once.
- If duplicates cause the final selected count to fall below the total budget, fill remaining slots from a global list of unselected candidates ranked by their best PAL score across classes.
- Record per-image selected class and score provenance for debugging.

## Rare-Class and Classifier Fallbacks

Ambiguous or under-specified cases:

- A class may have only TP or only FP labelled detections, making logistic classifier training ill-posed.
- A class may have too few labelled detections.
- A class may have no unlabelled detections after PAL's score threshold.
- Rare classes may be important but absent from the current labelled set.

Suggested first implementation:

- Require at least one TP and one FP to train a class logistic classifier.
- If a classifier cannot be trained, use a documented fallback score for that class:
  - Option A, preferred for stability: binary entropy of detection confidence.
  - Option B, stricter reproduction mode: skip direct LIUS selection for that class and redistribute its budget.
- Make the fallback configurable and log which classes used it per round.

## Ambiguities Requiring Confirmation

- TP/FP IoU threshold: PAL says labelled detections record true/false positive status but does not specify the exact matching threshold. Use class-aware IoU >= 0.5 for VOC unless contradicted later.
- Pre-NMS assignment IoU threshold: PAL says pre-NMS boxes are assigned to final detections using an IoU threshold but does not specify the value. Use 0.5 initially.
- CWIE class probability vector: PAL Eq. 7 uses `p_ij` over classes, but common MMDetection JSON output contains only final class and confidence. Full class score vectors require a custom inference output.
- Duplicate image handling across class budgets is not fully specified.
- Budget integer rounding is not specified.
- Exact Google ViT checkpoint/preprocessing is not specified in the extracted text.
- Whether `n_c,l` and `n_c,u` should use GT instance counts, predicted detection counts, or TP/FP-filtered detection counts is not fully explicit. First implementation should use available labelled GT counts for labelled data and predicted detection counts for unlabelled data only if this is documented and kept consistent.

## Minimal Build Order

1. Implement PAL inference JSON for RetinaNet with `pre_nms_count`.
2. Implement labelled TP/FP matching.
3. Implement LIUS logistic classifiers and LIUS-only selection.
4. Add class budget allocation and candidate generation.
5. Add CWIE and RCDI.
6. Add RCSP with image embedding cache.
7. Add final PAL score and duplicate/fill policy.

This order gives a runnable `PAL-LIUS-only` milestone before the full GUIDE implementation.
