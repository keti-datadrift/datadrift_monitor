# Common Boundary Cleanup

## Goal

ALOD has two common layers:

- `methods/common`: method-runtime utilities shared by active learning methods.
- `tools/common`: runner and preparation utilities shared by command-line tools.

The boundary keeps paper method logic independent from runner setup, downloads,
repo path handling, and other execution side effects.

## Layer Responsibilities

### methods/common

Use `methods/common` for utilities that active learning methods call during
runtime:

- COCO pool loading and pool updates
- detection-result JSON loading
- image-id selection, ranking, and deterministic refill
- detection/ground-truth matching
- acquisition result payloads
- numeric vector helpers

`methods/common` must not assume a repo root, CLI arguments, catalog selection,
download access, or preparation side effects.

### tools/common

Use `tools/common` for runner and setup utilities:

- repo-relative path resolution
- `code_refs` read/write guard
- automatic input preparation
- pretrained checkpoint download/checking
- VOC oracle and initial active learning pool preparation
- PAL embedding cache preparation
- preparation summaries

`tools/common` may call pure utilities from `methods/common`.

## Import Rules

Allowed:

- `tools/common -> methods/common`
- `tools/run_active_learning.py -> tools/common`
- `methods/pal|ppal|random|entropy -> methods/common`

Forbidden:

- `methods/common -> tools/common`
- `methods/pal|ppal|random|entropy -> tools/common`

Current narrow exception:

- `tools/common/pal_embeddings.py` calls PAL public embedding-cache helpers
  because it prepares a PAL-owned artifact for `pal_embedding_source="external"`.
  This exception should stay limited to artifact format/read-write helpers, not
  acquisition logic.

## Placement Rules

Put a helper in `methods/common` when:

- it is used by method acquisition logic
- it is independent of repo layout and CLI
- it does not download or prepare files before a run
- it can be tested without runner/catalog state

Put a helper in `tools/common` when:

- it prepares files before a run
- it downloads or verifies external assets
- it resolves repo paths or protects `code_refs`
- it depends on catalog or runner config
- it writes preparation summaries

## Cleanup Applied

- `is_relative_to` is centralized in `methods/common/paths.py`.
- `tools/common/paths.py` imports that helper and keeps only tool-specific path
  responsibilities such as repo path resolution and `code_refs` guarding.
- VOC XML parsing and oracle creation stay in `tools/common/voc.py`.
- VOC initial split writing now reuses `methods.common.coco_pool.build_coco_subset`
  instead of maintaining a duplicate subset helper.
- `tests/test_common_boundaries.py` prevents methods code from importing
  `tools.common`.

## Validation

Use these checks after boundary-related changes:

```powershell
python -m compileall -q methods tools configs tests
python -m unittest discover -s tests
rg "from tools.common|import tools.common" methods
rg "def is_relative_to" methods/common tools/common
```
