# ECPAL EUA Branches

## Goal

ECPAL now separates the uncertainty score and the diversity stage explicitly.
The bare `ecpal` method alias is removed so experiments must choose one of the
four named branches.

## Branches

| CLI method | First-stage score | Candidate size | Diversity |
|---|---|---:|---|
| `ecpal:eca-only` | ECA expected error count | `1 * budget` | none |
| `ecpal:eua-only` | EUA predictive error uncertainty | `1 * budget` | none |
| `ecpal:eca-full` | ECA expected error count | `2 * budget` | ECA-profile JS distance |
| `ecpal:eua-full` | EUA predictive error uncertainty | `2 * budget` | EUA-profile JS distance |

## Output Directories

```text
work_dirs/retinanet_voc_ecpal_eca_only_7rounds_5percent_to_20percent
work_dirs/retinanet_voc_ecpal_eua_only_7rounds_5percent_to_20percent
work_dirs/retinanet_voc_ecpal_eca_full_7rounds_5percent_to_20percent
work_dirs/retinanet_voc_ecpal_eua_full_7rounds_5percent_to_20percent
```

The old historical output directory:

```text
work_dirs/retinanet_voc_ecpal_7rounds_5percent_to_20percent
```

corresponds to the old full ECPAL behavior, now named `ecpal:eca-full`. Rename it
manually only when the destination does not already exist:

```powershell
Rename-Item `
  "work_dirs/retinanet_voc_ecpal_7rounds_5percent_to_20percent" `
  "retinanet_voc_ecpal_eca_full_7rounds_5percent_to_20percent"
```

The old ECA-only output directory:

```text
work_dirs/retinanet_voc_ecpal_eca_7rounds_5percent_to_20percent
```

maps to:

```text
work_dirs/retinanet_voc_ecpal_eca_only_7rounds_5percent_to_20percent
```

## EUA Score

EUA uses the same predictors as ECA:

- FDP: foreground probability
- CECP: conditional classification-error probability
- LECP: conditional localization-error probability
- MOCP: missed-object count

The score conversion differs:

```text
U_cls  = sum q_fg * H_Bern(q_cls_given_fg)
U_loc  = sum q_fg * H_Bern(q_loc_given_fg)
U_miss = H_Poisson(lambda_miss)
EUA    = v_cls * U_cls + v_loc * U_loc + v_miss * U_miss
```

The EUA scale weights are estimated from the current labeled pool by applying
the trained predictors to labeled feature records and averaging the EUA profile.
This keeps the labeled and unlabeled EUA definitions identical.

## Dependencies

EUA uses `scipy.stats.poisson.entropy` for the missed-object Poisson entropy.
`requirements/runtime.txt` now lists `scipy==1.10.1`.
