# PAL LIUS-only and ECPAL ECA-only Comparison Setup

## Purpose

This change adds a direct uncertainty-only comparison path between PAL and
ECPAL:

```powershell
python -B tools/run_active_learning.py --method pal:lius --detector retinanet --dataset voc --gpus 1 --seeds 0 1 2
python -B tools/run_active_learning.py --method ecpal:eca --detector retinanet --dataset voc --gpus 1 --seeds 0 1 2
```

The goal is to separate two questions that are mixed in full-method results:

1. Does the uncertainty score itself select useful training images?
2. Does the diversity step improve or hurt the uncertainty-selected pool?

For this comparison, both methods select exactly the training budget from their
uncertainty signal, without expanding to a `2 * budget` candidate pool for the
final diversity step.

## Method Meaning

### PAL LIUS-only

`pal:lius` now means:

- train PAL's LIUS logistic models from labeled detections;
- score unlabeled detections with LIUS;
- allocate class-wise budgets as in the PAL LIUS path;
- collect only `1 * class_budget` image candidates per class;
- select the final `budget` images directly from LIUS scores;
- do not run GUIDE, CWIE, RCDI, RCSP, image embeddings, or diversity selection.

Important detail: this is still PAL's class-wise LIUS acquisition, not a purely
global top-LIUS ranking over all detections. The change removes the previous
`2 * class_budget` intermediate candidate expansion from the LIUS-only path.

Full PAL is unchanged:

- `pal`, `pal:guide`, and `pal/full` still run LIUS + GUIDE;
- full PAL still builds GUIDE candidates with `candidate_multiplier=2`.

### ECPAL ECA-only

`ecpal:eca` is a new ECPAL acquisition mode:

- run the same ECPAL labeled and unlabeled inference as full ECPAL;
- train the same ECPAL predictors;
- compute ECA scores from predicted error counts;
- build only `1 * budget` candidates;
- select the top `budget` images by ECA score;
- do not run JS distance or farthest-first diversity selection.

Full ECPAL is unchanged:

- `ecpal` still runs ECA candidate scoring followed by ECD JS diversity;
- full ECPAL still uses `ecpal_candidate_expand_ratio=2`.

## Code Changes

### Catalog

Changed files:

- `configs/catalog/methods.py`
- `configs/catalog/experiments.py`

Added ECPAL ECA-only aliases:

```python
ECPAL_ECA_ALIASES = (
    'ecpal:eca',
    'ecpal/eca',
    'ecpal:uncertainty',
    'ecpal/uncertainty',
)
```

Added method spec:

```python
MethodSpec(
    key='ecpal_eca',
    method='ecpal',
    aliases=ECPAL_ECA_ALIASES,
    description='ECPAL ECA-only uncertainty acquisition.',
    output_name='retinanet_voc_ecpal_eca_7rounds_5percent_to_20percent',
    cfg_overrides={
        'ecpal_mode': 'eca',
        'ecpal_candidate_expand_ratio': 1,
        'ecpal_diagnostics_file': 'ecpal_eca_diagnostics.json',
        'ecpal_candidates_file': 'ecpal_eca_candidates.json',
    },
)
```

Added preset name:

```python
'ecpal_eca': 'ecpal-eca-retinanet-voc'
```

Changed `pal:lius` catalog config:

```python
'pal_candidate_multiplier': 1
```

### PAL Acquisition

Changed file:

- `methods/pal/acquisition.py`

`select_lius_images()` now accepts:

```python
candidate_multiplier: int = 1
```

The previous hard-coded LIUS-only candidate expansion:

```python
class_candidates[:2 * class_budget]
```

is now:

```python
class_candidates[:candidate_multiplier * class_budget]
```

For `pal:lius`, the catalog sets `candidate_multiplier=1`. Full PAL still uses
`candidate_multiplier=2` inside `select_full_pal_images()`.

### ECPAL Acquisition

Changed files:

- `methods/ecpal/acquisition.py`
- `methods/ecpal/scoring.py`

`select_ecpal_images()` and `sample_ecpal_from_files()` now accept:

```python
mode: str = 'ecd'
```

Supported ECPAL acquisition modes:

- `ecd`: full ECPAL, ECA candidate pool followed by JS diversity.
- `eca`: ECA-only, top budget by predicted error-count score.

For `mode='eca'`:

- effective candidate expand ratio is forced to `1.0`;
- `farthest_first_select()` is not called;
- candidate source is written as `eca`;
- selected ids are the first `budget` candidate ids after ECA ranking.

For `mode='ecd'`:

- existing full ECPAL behavior is preserved;
- candidate source is written as `ecd`;
- JS diversity and farthest-first selection are still used.

### Runner

Changed file:

- `tools/run_active_learning.py`

PAL runner now passes:

```python
candidate_multiplier=int(cfg.get('pal_candidate_multiplier', 1))
```

ECPAL runner now passes:

```python
candidate_expand_ratio=float(cfg.get('ecpal_candidate_expand_ratio', 2))
mode=str(cfg.get('ecpal_mode', 'ecd'))
```

ECPAL stage is read from acquisition diagnostics instead of being hard-coded:

```python
diagnostics_stage = str(diagnostics.get('stage', cfg.get('ecpal_mode', 'ecd')))
```

The CLI help text now includes `ecpal:eca`.

## Expected Outputs

### PAL LIUS-only

Output directory:

```text
work_dirs/retinanet_voc_pal_lius_7rounds_5percent_to_20percent/<timestamp>/
```

Round artifacts include:

```text
pal_lius_diagnostics.json
pal_lius_candidates.json
```

Expected behavior:

- stage: `lius`
- selected count: `budget`
- GUIDE/embedding outputs are not used
- candidate count should reflect `1 * class_budget` collection plus any refill
  candidates if needed

### ECPAL ECA-only

Output directory:

```text
work_dirs/retinanet_voc_ecpal_eca_7rounds_5percent_to_20percent/<timestamp>/
```

Round artifacts include:

```text
ecpal_eca_diagnostics.json
ecpal_eca_candidates.json
```

Expected behavior:

- stage: `eca`
- candidate expand ratio: `1.0`
- candidate count: `min(budget, unlabeled_count)`
- selected count: `budget`, unless the unlabeled pool is smaller
- no JS diversity metadata such as `nearest_selected_distance`
- selected ids match ECA score top budget before common budget refill

### Full ECPAL

Output directory remains:

```text
work_dirs/retinanet_voc_ecpal_7rounds_5percent_to_20percent/<timestamp>/
```

Expected behavior remains:

- stage: `ecd`
- candidate expand ratio: `2`
- candidate count: `min(2 * budget, unlabeled_count)`
- JS farthest-first diversity is used
- selected candidates can include JS diversity metadata

## Validation Performed

Syntax validation without writing bytecode:

```powershell
python -B -c "from pathlib import Path; files=['configs/catalog/methods.py','configs/catalog/experiments.py','methods/pal/acquisition.py','methods/ecpal/acquisition.py','methods/ecpal/scoring.py','tools/run_active_learning.py']; [compile(Path(f).read_text(encoding='utf-8'), f, 'exec') for f in files]; print('ok')"
```

Catalog resolution checks:

- `pal:lius` resolves to:
  - `method='pal'`
  - `pal_mode='lius'`
  - `pal_candidate_multiplier=1`
- `pal` resolves to full PAL:
  - `pal_mode='full'`
- `ecpal:eca` resolves to:
  - `method='ecpal'`
  - `ecpal_mode='eca'`
  - `ecpal_candidate_expand_ratio=1`
  - `ecpal_candidates_file='ecpal_eca_candidates.json'`
- `ecpal` resolves to full ECPAL:
  - `ecpal_mode='ecd'`
  - `ecpal_candidate_expand_ratio=2`
  - `ecpal_candidates_file='ecpal_candidates.json'`

Preset listing check:

```powershell
python -B tools/run_active_learning.py --list-presets
```

Confirmed the new preset appears:

```text
ecpal-eca-retinanet-voc
```

Function-level synthetic checks:

- PAL LIUS-only with one class and budget `2`:
  - `candidate_multiplier=1` keeps top `2` class candidates;
  - `candidate_multiplier=2` keeps top `4` class candidates;
  - selected ids are top LIUS-scored ids.
- ECPAL ECA-only with budget `2`:
  - selected ids are top `2` ECA-scored ids;
  - `farthest_first_select()` is not called;
  - stage is `eca`;
  - candidate expand ratio is `1.0`;
  - candidate source is `eca`.
- Full ECPAL regression:
  - stage remains `ecd`;
  - candidate expand ratio remains `2.0`;
  - candidate count is `2 * budget` when enough unlabeled images exist;
  - `farthest_first_select()` is called.

## Interpretation Plan

Recommended comparison order:

1. Run uncertainty-only methods:

```powershell
python -B tools/run_active_learning.py --method pal:lius --detector retinanet --dataset voc --gpus 1 --seeds 0 1 2
python -B tools/run_active_learning.py --method ecpal:eca --detector retinanet --dataset voc --gpus 1 --seeds 0 1 2
```

2. Compare:

- round-wise mAP/AP50;
- selected image overlap;
- selected GT object counts;
- selected actual cls/loc/miss error counts;
- candidate score vs actual error-count correlation.

3. Interpret:

- If `ecpal:eca` beats `pal:lius`, ECPAL's error-count score is likely a
  stronger uncertainty signal than PAL LIUS for this setting.
- If `ecpal:eca` is strong but full `ecpal` is weaker, the JS diversity step is
  the likely improvement target.
- If `pal:lius` beats `ecpal:eca`, ECPAL predictor labels, features, weighting,
  or error-count score definition should be inspected before changing diversity.
