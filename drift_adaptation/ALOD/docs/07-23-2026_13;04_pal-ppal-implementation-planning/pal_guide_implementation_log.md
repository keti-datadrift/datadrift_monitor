# PAL GUIDE Implementation Log

## Implemented

- Integrated full PAL GUIDE acquisition into `methods/pal/sampler.py`.
- Added `sample_pal_from_files()` with `mode='lius'` and `mode='full'`.
- Kept `sample_lius_only_from_files()` for backward compatibility.
- Connected `tools/run_active_learning.py` to pass PAL mode, weights, and
  embedding settings into the sampler.
- Added full PAL configs:
  - `configs/experiments/pal_retinanet_voc_smoke.py`
  - `configs/experiments/pal_retinanet_voc.py`
- Made existing PAL-LIUS configs explicit with `pal_mode = 'lius'`.
- Added synthetic GUIDE tests in `tests/test_pal_guide.py`.

## Full PAL Behavior

Full mode now performs:

1. Labeled detection to GT matching.
2. Class-wise LIUS model training.
3. Unlabeled detection LIUS scoring.
4. Class rarity weight and class budget allocation.
5. Per-class `2 * b_c` candidate generation.
6. CWIE, RCDI, RCSP, and final PAL score calculation.
7. Duplicate-safe unique image selection with deterministic refill.
8. Diagnostics output containing selected images, class budgets, class weights,
   selected candidate details, and GUIDE candidate scores.

## Embedding Policy

- Reproduction config uses `pal_embedding_source = 'external'` and requires
  `work_dirs/pal_embeddings/voc_google_vit_embeddings.npy`.
- Smoke config uses `pal_embedding_source = 'detection'` to exercise RCSP without
  adding a ViT dependency. This is not paper-faithful.
- The ViT backend is intentionally not hidden behind a silent fallback.

## Validation

No Python, test, training, inference, or smoke commands were run in this pass.
The user requested to run code themselves.

Suggested user-run checks:

```powershell
python -B -m unittest tests.test_pal_guide tests.test_pal_embeddings tests.test_pal_vit_embeddings
python -B tools/run_active_learning.py configs/experiments/pal_retinanet_voc_smoke.py --method pal --rounds 1 --gpus 1
python -B tools/run_active_learning.py configs/experiments/pal_retinanet_voc_smoke.py --method pal --rounds 1 --gpus 1 --execute
```

## Remaining Limitation

The Google ViT embedding cache builder is now implemented in
`tools/build_pal_vit_embeddings.py`. The paper text does not specify the exact
Google ViT checkpoint, processor, or embedding layer, so the local default is
documented as `google/vit-base-patch16-224-in21k` with Hugging Face processor
defaults and `pooler_output`/CLS image-level vectors.

## User-Run Validation

The user ran the requested GUIDE checks and smoke pipeline on 2026-07-23:

```powershell
python -B -m unittest tests.test_pal_guide tests.test_pal_embeddings
python -B tools/run_active_learning.py configs/experiments/pal_retinanet_voc_smoke.py --method pal --rounds 1 --gpus 1
python -B tools/run_active_learning.py configs/experiments/pal_retinanet_voc_smoke.py --method pal --rounds 1 --gpus 1 --execute
```

Observed result:

- Unit tests passed: `Ran 8 tests ... OK`.
- Full PAL/GUIDE smoke completed train, eval, labeled PAL inference,
  unlabeled PAL inference, and acquisition.
- Final smoke selection: `[10219, 5693]`.
- Diagnostics were written to
  `work_dirs/smoke_retinanet_voc_pal_guide_1round/round_01/pal_diagnostics.json`.

The Windows `tail`/`gcc` messages and the checkpoint `unexpected key` warning
were non-fatal in this run.
