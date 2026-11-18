# Experiment Progress Recap

## Objective

Reconstruct the current ALOD research state from the experiment artifacts and
historical notes, with emphasis on the ECPAL ECA/EUA ablations.

## Measured Experiment Inventory

The current `work_dirs` contains seven completed RetinaNet + VOC experiments.
Each experiment has seeds 0, 1, and 2 and completed rounds 1 through 7.

| Current method name | Historical `method_arg` | Run | Final mAP mean | Std |
|---|---|---|---:|---:|
| ECPAL EUA-only | `ecpal:eua-only` | `08-07-2026_18;32` | 0.710877 | 0.000930 |
| PAL full | `pal` | `07-28-2026_17;23` | 0.708557 | 0.001511 |
| PAL LIUS-only | `pal:lius` | `08-04-2026_09;05` | 0.707096 | 0.001154 |
| PPAL | `ppal` | `07-30-2026_12;34` | 0.707064 | 0.003505 |
| ECPAL ECA-only | `ecpal:eca` | `08-05-2026_16;35` | 0.706369 | 0.001446 |
| ECPAL ECA-full | `ecpal` | `08-02-2026_19;46` | 0.704726 | 0.000764 |
| Core-set | `coreset` | `07-31-2026_23;51` | 0.671498 | 0.003661 |

The historical aliases `ecpal` and `ecpal:eca` were later renamed to the
explicit current aliases `ecpal:eca-full` and `ecpal:eca-only`.

No completed result currently exists for:

- ECPAL EUA-full
- MIAL
- Random
- Entropy
- any COCO method

MIAL and COCO support were implemented, but no corresponding completed run is
present in the current result tree. COCO seed-specific 2% pool files are
prepared under `data/active_learning/coco/`.

## ECPAL Ablation State

The intended ECPAL comparison is a two-by-two design:

| First-stage signal | No diversity | JS-profile diversity |
|---|---|---|
| ECA: expected error amount | ECA-only: complete | ECA-full: complete |
| EUA: predictive error uncertainty | EUA-only: complete | EUA-full: not run |

Round-wise measured mAP means:

| Round | ECA-only | ECA-full | EUA-only |
|---:|---:|---:|---:|
| 1 | 0.4677 | 0.4677 | 0.4677 |
| 2 | 0.5941 | 0.5914 | 0.5973 |
| 3 | 0.6381 | 0.6394 | 0.6439 |
| 4 | 0.6630 | 0.6690 | 0.6705 |
| 5 | 0.6871 | 0.6849 | 0.6886 |
| 6 | 0.6966 | 0.6988 | 0.7046 |
| 7 | 0.7064 | 0.7047 | 0.7109 |

Measured differences at the final round:

- EUA-only minus ECA-only: +0.0045 mAP.
- EUA-only minus ECA-full: +0.0062 mAP.
- ECA-only minus ECA-full: +0.0016 mAP.
- EUA-only minus PAL full: +0.0023 mAP.

ECA-only selected images with high actual detector-error counts, showing that
the error-count predictor worked as intended. The highest error count did not
produce the highest final mAP. EUA-only selected predictive error uncertainty
instead and produced the best local final result. Its selected uncertainty
profile was dominated by localization uncertainty, followed by classification
and missed-object uncertainty. Predictor fallback count was zero throughout the
completed ECPAL experiments.

ECA-full was better than ECA-only in some middle rounds but worse at the final
round. This suggests that the existing hard JS-diversity stage may replace too
many high-value score-ranked samples late in training. Whether diversity is
useful for the stronger EUA signal remains unknown because EUA-full is the
missing ablation.

## PAL/PPAL Context

- PAL LIUS-only finished only 0.0015 mAP below PAL full. LIUS therefore accounts
  for most of PAL's measured performance, while GUIDE supplied a small overall
  benefit and a larger benefit in some middle rounds.
- PPAL and PAL LIUS-only finished at essentially the same mean mAP.
- ECPAL selection overlapped little with PAL/PPAL after the first round, so the
  methods reached similar performance using substantially different acquisition
  trajectories.
- Core-set was materially weaker than the uncertainty/error-driven methods in
  this setup.

## Protocol Boundary

All seven completed VOC runs used exactly the same 827-image initial labeled
membership for every method and all three training seeds. Canonical image-ID
hashing found one unique membership across all 21 seed runs:
`463ac8b3a7e9` (truncated SHA-256).

The repository later adopted a seed-specific initial-pool protocol. Therefore:

- the existing results remain internally comparable as legacy
  shared-initial-pool experiments;
- their seed variation measures training/acquisition randomness, not initial
  pool sampling variability;
- a new seed-specific run must not be combined directly with the old aggregate
  table as if it used the same protocol.

The old runs also used three seed workers concurrently. The current runner now
executes seeds sequentially, so historical commands containing
`--seed-workers` are no longer current commands.

## Current Research Position

The strongest supported working hypothesis is that selecting cases where the
error predictors are uncertain is more useful than simply maximizing predicted
error amount. The immediate unresolved method question is whether EUA-profile
diversity improves EUA-only; this is the missing EUA-full experiment.

For a defensible next comparison under the current seed-specific protocol, the
minimum useful group is a current baseline plus the ECPAL variants needed for
the claim. Running only EUA-full now would answer neither the old-protocol
ablation nor a complete new-protocol comparison. At minimum, rerun PAL full,
EUA-only, and EUA-full under one protocol; include ECA-only and ECA-full to
reproduce the full two-by-two ECPAL ablation.

## Sources and Validation

Measured data came from the seven current `aggregate_summary.json` files,
their seed/round summaries, round-0 annotation pools, and the result-analysis
records under:

- `docs/08-04-2026_13;04_ecpal-pal-ppal-selected-error-analysis/`
- `docs/08-05-2026_17;12_pal-lius-result-analysis/`
- `docs/08-07-2026_15;12_ecpal-eca-result-analysis/`
- `docs/08-10-2026_17;08_ecpal-eua-only-result-analysis/`
- `docs/08-11-2026_19;07_seeded_initial_pool_protocol/`

All aggregate values were recomputed from the current JSON artifacts. The
initial-pool claim was verified from canonical image-ID membership rather than
raw JSON byte equality. No experiment output was modified during this audit.
