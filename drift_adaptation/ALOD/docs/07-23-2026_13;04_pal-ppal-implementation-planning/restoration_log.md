# Restoration Log

## Scope

Smoke-only changes were removed from algorithmic PPAL code. `code_refs/` remains a read-only reference.

## Restored

- `mmdet/ppal/sampler/diversity_sampler.py` follows the PPAL reference behavior again.
- NumPy alias compatibility edits in copied PPAL source were reverted; the runtime requirement is pinned instead.
- `requirements/runtime.txt` and `requirements/build.txt` use `numpy==1.23.5` for MMDetection 2.x compatibility.

## Kept

- Single-process distributed guards in `mmdet/ppal/models/utils.py` remain. They are identity behavior for world size 1 and allow Windows `--launcher none` execution.
- `tools/train.py` and `tools/test.py` compatibility bootstrapping remains for local imports and newer `yapf`.

## Follow-up

For future smoke tests with the original diversity sampler, prepare a smoke pool large enough that PPAL's clustering assumptions hold instead of weakening sampler assertions.
