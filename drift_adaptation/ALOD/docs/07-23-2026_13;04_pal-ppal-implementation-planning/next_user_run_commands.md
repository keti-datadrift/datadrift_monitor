# Next User Run Commands

Codex did not execute project Python, tests, training, or smoke commands in this pass.

## Dependency Pin

If your current `alod` environment still has NumPy 1.24, downgrade it before running the restored PPAL code:

```powershell
pip install numpy==1.23.5
```

## PAL-LIUS Smoke Plan

Print the command plan without launching train/test/acquisition:

```powershell
python -B tools/run_active_learning.py configs/experiments/pal_lius_retinanet_voc_smoke.py --method pal --rounds 1 --gpus 1
```

Execute the tiny PAL-LIUS smoke pipeline:

```powershell
python -B tools/run_active_learning.py configs/experiments/pal_lius_retinanet_voc_smoke.py --method pal --rounds 1 --gpus 1 --execute
```

Expected PAL round outputs:

```text
work_dirs/smoke_retinanet_voc_pal_lius_1round/round_01/pal_labeled_detections.bbox.json
work_dirs/smoke_retinanet_voc_pal_lius_1round/round_01/pal_unlabeled_detections.bbox.json
work_dirs/smoke_retinanet_voc_pal_lius_1round/round_01/pal_lius_diagnostics.json
work_dirs/smoke_retinanet_voc_pal_lius_1round/round_01/annotations/new_labeled.json
work_dirs/smoke_retinanet_voc_pal_lius_1round/round_01/annotations/new_unlabeled.json
```

The smoke config lowers the PAL inference score threshold to `0.0` and caps
`max_per_img` at `50` so the LIUS path is exercised even when the tiny
1-epoch detector has no confident predictions. The full VOC config keeps the
paper setting `score_thr=0.3`.

## Full PAL/GUIDE Smoke Plan

Run the lightweight unit checks for the new GUIDE and embedding helpers:

```powershell
python -B -m unittest tests.test_pal_guide tests.test_pal_embeddings tests.test_pal_vit_embeddings
```

Print the full PAL/GUIDE smoke command plan:

```powershell
python -B tools/run_active_learning.py configs/experiments/pal_retinanet_voc_smoke.py --method pal --rounds 1 --gpus 1
```

Execute the tiny full PAL/GUIDE smoke pipeline:

```powershell
python -B tools/run_active_learning.py configs/experiments/pal_retinanet_voc_smoke.py --method pal --rounds 1 --gpus 1 --execute
```

Expected full PAL round outputs:

```text
work_dirs/smoke_retinanet_voc_pal_guide_1round/round_01/pal_labeled_detections.bbox.json
work_dirs/smoke_retinanet_voc_pal_guide_1round/round_01/pal_unlabeled_detections.bbox.json
work_dirs/smoke_retinanet_voc_pal_guide_1round/round_01/pal_diagnostics.json
work_dirs/smoke_retinanet_voc_pal_guide_1round/round_01/annotations/new_labeled.json
work_dirs/smoke_retinanet_voc_pal_guide_1round/round_01/annotations/new_unlabeled.json
```

The full PAL smoke config uses detector-record embeddings only so RCSP can be
exercised without the paper's Google ViT embedding cache. This is a plumbing
test, not a reproduction setting.

## Full PAL-LIUS VOC Plan

Dry-run full VOC PAL-LIUS:

```powershell
python -B tools/run_active_learning.py configs/experiments/pal_lius_retinanet_voc.py --method pal --rounds 1 --gpus 1
```

Run full VOC PAL-LIUS:

```powershell
python -B tools/run_active_learning.py configs/experiments/pal_lius_retinanet_voc.py --method pal --rounds 1 --gpus 1 --execute
```

Use a larger `--gpus` value only after confirming the single-GPU plan.

## Full PAL/GUIDE VOC Plan

The reproduction config expects a prepared image embedding cache:

```text
work_dirs/pal_embeddings/voc_google_vit_embeddings.npy
```

Build the cache with the local Google ViT generator:

```powershell
pip install -r requirements/pal_embeddings.txt
```

The first run may download the Hugging Face model unless `--model-name` points
to a local model directory.

```powershell
python -B tools/build_pal_vit_embeddings.py `
  --ann-file data/VOC0712/annotations/trainval_0712.json `
  --image-root data/VOCdevkit `
  --output work_dirs/pal_embeddings/voc_google_vit_embeddings.npy `
  --model-name google/vit-base-patch16-224-in21k `
  --device cuda `
  --batch-size 16
```

After that cache exists, dry-run full VOC PAL/GUIDE:

```powershell
python -B tools/run_active_learning.py configs/experiments/pal_retinanet_voc.py --method pal --rounds 1 --gpus 1
```

Run full VOC PAL/GUIDE:

```powershell
python -B tools/run_active_learning.py configs/experiments/pal_retinanet_voc.py --method pal --rounds 1 --gpus 1 --execute
```
