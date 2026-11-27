# Subagent Execution Log

Codex did not run project Python, tests, training, inference, or smoke commands.
The work below was limited to file inspection, static integration review, and
source/document edits.

## Roles

- Paper/settings sidecar: checked PAL GUIDE settings against the current
  PPAL-based VOC/RetinaNet plan and recorded reproduction caveats.
- GUIDE code worker: implemented pure-Python PAL GUIDE scoring helpers and
  focused synthetic tests.
- Embedding code worker: implemented PAL image embedding cache utilities and
  a deterministic detector-record embedding backend for smoke validation.
- ViT paper/settings sidecar: confirmed that the PAL paper requires
  image-level ViT embeddings for RCSP but does not specify an exact Google ViT
  checkpoint, preprocessing recipe, layer, or embedding dimension.
- ViT embedding code worker: started COCO image path helper implementation;
  the main agent completed the CLI generator, metadata, tests, and docs.

## Integrated Outputs

- `methods/pal/guide.py`
- `methods/pal/embeddings.py`
- `methods/pal/sampler.py`
- `tools/run_active_learning.py`
- `configs/experiments/pal_retinanet_voc_smoke.py`
- `configs/experiments/pal_retinanet_voc.py`
- `tests/test_pal_guide.py`
- `tests/test_pal_embeddings.py`
- `methods/pal/vit_embeddings.py`
- `tools/build_pal_vit_embeddings.py`
- `tests/test_pal_vit_embeddings.py`
- `requirements/pal_embeddings.txt`
- `docs/07-23-2026_13;04_pal-ppal-implementation-planning/pal_vit_embedding_cache_log.md`
- `docs/07-23-2026_13;04_pal-ppal-implementation-planning/pal_guide_implementation_log.md`
- `docs/07-23-2026_13;04_pal-ppal-implementation-planning/next_user_run_commands.md`

## Boundary

`code_refs/` was used only as reference material. The new PAL path runs from
ALOD-owned files and keeps external embeddings in `work_dirs/`.
