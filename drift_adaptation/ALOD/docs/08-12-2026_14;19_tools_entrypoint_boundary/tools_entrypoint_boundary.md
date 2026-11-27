# Tools Entrypoint Boundary

## Purpose

The top level of `tools/` exposes only commands intended for users. Executable
backend scripts live under `tools/internal/`, while reusable runner and input
preparation code remains under `tools/common/`.

## Path Changes

The internal entrypoints moved as follows:

| Previous path | Current path | Role |
| --- | --- | --- |
| `tools/train.py` | `tools/internal/train_detector.py` | Standard detector training |
| `tools/test.py` | `tools/internal/infer_detector.py` | Evaluation and acquisition inference |
| `tools/train_mial.py` | `tools/internal/train_mial_detector.py` | MIAL phase-based detector training |
| `tools/view_metrics.py` | `tools/internal/metrics_dashboard.py` | Streamlit dashboard application |

The previous paths are removed without compatibility wrappers. Internal
entrypoints are implementation details and are invoked by the public launchers.

## Public Commands

Run active-learning experiments with:

```powershell
python -B tools/run_active_learning.py --method pal --detector retinanet --dataset voc --gpus 1
```

Open the metrics dashboard with:

```powershell
python tools/run_metrics_dashboard.py
```

Users should not invoke files under `tools/internal/` directly.

## Dependency Rules

The allowed dependency direction is:

```text
tools/run_*.py -> tools/internal/*.py (subprocess or Streamlit execution)
tools/run_*.py -> tools/common/*.py
tools/internal/*.py -> tools/common/, methods/, mmdet/
```

The following dependencies are prohibited:

```text
tools/common/*.py -> tools/internal/*.py
methods/* -> tools/internal/*.py
methods/* -> tools/common/*.py
```

`tools/common/` contains importable, reusable libraries. `tools/internal/`
contains executable backend entrypoints and must not become a shared utility
layer. It intentionally has no `__init__.py` because its scripts are executed
by the public launchers rather than imported as a package.

## Repository Bootstrap

Moving an executable one directory deeper changes its filesystem-relative root.
Each internal entrypoint must resolve the repository root with:

```python
REPO_ROOT = Path(__file__).resolve().parents[2]
```

It must add that root to `sys.path` before importing repository-local modules.
This preserves direct local-source execution without installing ALOD as a
package.

## Runtime Compatibility

The active-learning runner continues to build the same train, evaluation,
inference, and acquisition plans. Only the executable paths stored in newly
generated command plans change to `tools/internal/`.

Progress labels, round ordering, command arguments, log filenames, artifact
formats, and output directory structure remain unchanged. Existing experiment
outputs and their saved command plans are historical records and are not
rewritten.

## Validation Criteria

The refactor is complete when:

1. The public launchers and every internal entrypoint compile successfully.
2. Detector training, inference, and MIAL entrypoints accept their existing CLI arguments.
3. Every newly generated command plan references `tools/internal/` paths.
4. The dashboard launcher starts the internal Streamlit application and scans the requested work directory.
5. No live source or README reference uses the four previous paths.
6. No compatibility wrapper remains at a previous path.
7. `tools/common/` and `methods/` do not import `tools/internal/`.
8. `git diff --check` reports no whitespace errors.

Older timestamped documents describe the repository state at the time they were
written. They remain historical records; this document supersedes their tool
paths and entrypoint-boundary guidance.
