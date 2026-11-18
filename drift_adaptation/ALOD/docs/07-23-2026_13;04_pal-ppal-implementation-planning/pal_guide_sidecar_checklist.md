# PAL GUIDE Sidecar Checklist

## Formula Decisions

- Class rarity weight:

```text
r_c = 1 - 0.5 * (n_c,l / N_l + n_c,u / N_u)
```

- Class budget:

```text
b_c = min(n_c,u, b * r_c / sum_c(r_c))
```

The implementation uses deterministic largest-remainder rounding.

- Per-class candidate pool: rank detections by LIUS descending, keep one best
  detection per `(class, image)`, then keep the top `2 * b_c` image candidates.

- CWIE:

```text
CWIE(I) = - sum_i r_{c_i} * sum_j p_ij log(p_ij)
```

The implementation uses exported `class_scores`, normalized into a probability
vector per detection. CWIE is normalized per class-candidate set with minimum 0.

- RCDI:

```text
RCDI(I) = sum_{k in unique predicted classes of I} r_k
```

RCDI is normalized per class-candidate set with minimum 0.

- RCSP:

```text
RCSP(I_1) = 1
RCSP(I_i) = 1 - max_{m < i} cosine(e_i, e_m)
```

Candidates are ranked by LIUS before RCSP calculation.

- Final PAL score:

```text
Score(I) = alpha * LIUS(I_j) + gamma * RCSP(I) + beta * (CWIE(I) + RCDI(I))
alpha = 0.9, beta = 0.04, gamma = 0.02
```

## Duplicate Handling

- Process classes in descending `r_c`, with deterministic class id tie-breaking.
- Add each image at most once.
- If duplicate removal leaves a short budget, refill from the best remaining
  GUIDE candidates ranked by `pal_score`.
- If the GUIDE candidate pool is still short, refill by global LIUS, then by
  deterministic random sampling from the unlabeled pool.

## Current Detection JSON Sufficiency

The current PAL detection JSON has enough fields for LIUS, CWIE, and RCDI:

- `image_id`
- `bbox`
- `category_id`
- `score`
- `pre_nms_count`
- `class_scores`

Labeled detection TP/FP status is derived from the labeled pool annotations with
class-aware IoU matching. RCSP still needs image embeddings. The implementation
supports an external embedding cache for reproduction and a detection-record
embedding backend for smoke testing only.

## Ambiguities Kept Explicit

- The paper does not fully specify TP/FP matching IoU; VOC-style class-aware
  IoU `0.5` is used.
- The paper does not fully specify the pre-NMS assignment IoU; `0.5` is used in
  the detector export config.
- Exact Google ViT checkpoint/preprocessing is not implemented in this pass.
  Full reproduction config therefore requires an external ViT embedding cache.
- Detection-record embeddings are not paper-faithful and are used only by the
  smoke config.
