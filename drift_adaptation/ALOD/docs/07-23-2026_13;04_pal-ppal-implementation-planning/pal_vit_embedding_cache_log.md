# PAL ViT Embedding Cache Log

## Why This Exists

PAL full/GUIDE uses RCSP to penalize near-duplicate images among LIUS-ranked
class candidates. RCSP needs image-level embeddings for all candidate images.
The paper states that PAL uses a pre-trained vision transformer encoder and
uses Google ViT in the main method, but the extracted paper text does not state
an exact checkpoint, processor configuration, layer, or embedding dimension.

## Implemented

- Added `methods/pal/vit_embeddings.py` for COCO image path collection and
  Hugging Face ViT extraction helpers.
- Added `tools/build_pal_vit_embeddings.py` to build `.npy` or `.json` caches
  compatible with `methods.pal.embeddings.read_embedding_cache()`.
- Added `tests/test_pal_vit_embeddings.py` for path collection and validation
  helpers. These tests do not download a model or require transformers.
- Added `requirements/pal_embeddings.txt` for `Pillow` and `transformers`.

## Default Reproduction Choice

Because the paper does not identify an exact checkpoint, the local default is:

```text
google/vit-base-patch16-224-in21k
```

The default cached vector is the Hugging Face model `pooler_output` when
available, falling back to the CLS token if the pooler is absent. Vectors are
stored as float32 and L2-normalized by default. The chosen settings are written
beside the cache in `<output>.meta.json`.

## VOC Cache Command

Install optional extraction dependencies if they are not already present:

```powershell
pip install -r requirements/pal_embeddings.txt
```

The first run may download the Hugging Face model unless `--model-name` points
to an existing local model directory.

```powershell
python -B tools/build_pal_vit_embeddings.py `
  --ann-file data/VOC0712/annotations/trainval_0712.json `
  --image-root data/VOCdevkit `
  --output work_dirs/pal_embeddings/voc_google_vit_embeddings.npy `
  --model-name google/vit-base-patch16-224-in21k `
  --device cuda `
  --batch-size 16
```

Use `--device auto` to choose CUDA when available and CPU otherwise. Use
`--max-images` only for checking the generator mechanics, not for reproduction.

## Boundary

The documented command reads ALOD-owned annotation/image paths and writes the
cache under `work_dirs/` for the full PAL experiment config. The tool also
accepts absolute external dataset/model paths when needed, but refuses
`code_refs/` input/output.
