# PPAL Port Plan

## Scope

This is a documentation-only plan for porting `code_refs/PPAL` into editable ALOD code. The reference implementation under `code_refs/` must remain read-only. The first implementation target is a local PPAL baseline for `RetinaNet + PASCAL VOC`; PAL should be added only after that baseline can run from ALOD-owned code.

## Reference Files Inspected

- `code_refs/PPAL/tools/train.py`
- `code_refs/PPAL/tools/test.py`
- `code_refs/PPAL/tools/run_al_voc.py`
- `code_refs/PPAL/al_configs/voc/ppal_retinanet_voc.py`
- `code_refs/PPAL/configs/voc_active_learning/al_train/retinanet_26e.py`
- `code_refs/PPAL/configs/voc_active_learning/al_inference/retinanet_uncertainty.py`
- `code_refs/PPAL/configs/voc_active_learning/al_inference/retinanet_diversity.py`
- `code_refs/PPAL/configs/voc_active_learning/bases/al_retinanet_base.py`
- `code_refs/PPAL/configs/voc_active_learning/bases/al_retinanet_inference_base.py`
- `code_refs/PPAL/configs/voc_active_learning/bases/models/retinanet_r50_fpn.py`
- `code_refs/PPAL/mmdet/__init__.py`
- `code_refs/PPAL/mmdet/version.py`
- `code_refs/PPAL/mmdet/ppal/**`

## Local Copy Target

The editable ALOD baseline should copy these reference directories/files into the ALOD root:

```text
ALOD/
  mmdet/                         # copy from code_refs/PPAL/mmdet
  tools/
    train.py                      # copy from code_refs/PPAL/tools/train.py
    test.py                       # copy from code_refs/PPAL/tools/test.py
    run_active_learning.py        # new ALOD-owned replacement for run_al_voc.py
  configs/
    _base_/                       # copy from code_refs/PPAL/configs/_base_ if present/needed
    voc_active_learning/          # copy from code_refs/PPAL/configs/voc_active_learning
    experiments/
      ppal_retinanet_voc.py       # new ALOD-owned active-learning experiment config
  methods/
    common/
      active_learning_io.py       # new shared JSON/path/subprocess helpers
      sampler_registry.py         # new or wrapper around mmdet.ppal.builder.SAMPLER
    ppal/
      __init__.py
      sampler/
      README.md
    pal/
      __init__.py
```

Also copy these package metadata files when ALOD needs an editable install of the local MMDetection fork:

```text
setup.py
setup.cfg
MANIFEST.in
requirements/
README.md                         # optional, but setup.py expects it
model-index.yml                    # optional for MIM compatibility
```

The safest first port is to copy the entire `code_refs/PPAL/mmdet` package, not only `mmdet/ppal`. `tools/train.py` and `tools/test.py` import `mmdet` directly, and a partial copy would increase the risk of silently using the installed `mmdet` package for core code while using local PPAL extensions for only some modules.

Do not copy `code_refs/PPAL/mmdet.egg-info`; it is generated packaging state.

## What Stays Inside `mmdet/`

For the first PPAL baseline milestone, keep these PPAL-specific modules inside local `mmdet/ppal/` unchanged except for import hygiene if needed:

```text
mmdet/ppal/builder.py
mmdet/ppal/datasets/al_coco.py
mmdet/ppal/datasets/al_voc.py
mmdet/ppal/models/utils.py
mmdet/ppal/models/retinanet_al/al_retinanet.py
mmdet/ppal/models/retinanet_al/al_retinanet_feat_head.py
mmdet/ppal/models/retinanet_al/retinanet_quality_head.py
mmdet/ppal/models/retinanet_al/retinanet_uncertainty_head.py
mmdet/ppal/utils/dataset_info.py
mmdet/ppal/utils/running_checks.py
```

Reasoning:

- Dataset/model classes are registered through MMDetection registries (`DATASETS`, `DETECTORS`, `HEADS`) as import side effects.
- Configs refer to class names such as `ALVOCDataset`, `ALCocoDataset`, `ALRetinaNet`, `RetinaQualityEMAHead`, `RetinaHeadUncertainty`, and `RetinaHeadFeat`.
- Moving these files immediately would require broad import rewrites and create an avoidable source of reproduction drift.

## What Should Move To `methods/ppal/`

After the local PPAL baseline runs, move or wrap the acquisition-only code under `methods/ppal/`:

```text
mmdet/ppal/sampler/al_sampler_base.py
mmdet/ppal/sampler/difficulty_calibrated_uncertainty_sampler.py
mmdet/ppal/sampler/diversity_sampler.py
```

Suggested target:

```text
methods/ppal/sampler/base.py
methods/ppal/sampler/dcus.py
methods/ppal/sampler/diversity.py
```

Keep a compatibility bridge in `mmdet/ppal/sampler/` during the transition:

```python
from methods.ppal.sampler.dcus import DCUSSampler
from methods.ppal.sampler.diversity import DiversitySampler
```

This allows existing PPAL configs and `builder_al_sampler` calls to keep working while ALOD gradually adopts a method-centric layout. The acquisition samplers are the right first candidates to move because they do not need MMDetection model registries; they use PPAL's own `SAMPLER` registry from `mmdet.ppal.builder`.

Do not move `RetinaQualityEMAHead`, `RetinaHeadUncertainty`, or `RetinaHeadFeat` in the first pass. PAL may introduce sibling heads later, but the PPAL heads should remain stable until baseline parity is established.

## Tool Registration Plan

Current reference behavior:

- `tools/train.py` imports `from mmdet.ppal.datasets import *` and `from mmdet.ppal.models import *`.
- `tools/test.py` imports the same PPAL datasets/models.
- `tools/run_al_voc.py` imports `from mmdet.ppal.sampler import *` and builds samplers via `mmdet.ppal.builder.builder_al_sampler`.

For ALOD, replace wildcard imports with an explicit method registration hook:

```python
def register_method_modules(method: str) -> None:
    if method == "ppal":
        import mmdet.ppal.datasets  # registers ALVOCDataset, ALCocoDataset
        import mmdet.ppal.models    # registers PPAL RetinaNet heads/detector
        import mmdet.ppal.sampler   # registers DCUSSampler, DiversitySampler
    elif method == "pal":
        import mmdet.ppal.datasets
        import mmdet.ppal.models
        import methods.pal.models
        import methods.pal.sampler
    else:
        raise ValueError(f"Unknown method: {method}")
```

Add a CLI option to `tools/train.py` and `tools/test.py`:

```text
--method {ppal,pal,random,entropy}
```

Default it to `ppal` for backward-compatible first-pass behavior. `tools/run_active_learning.py` should pass `--method ppal` for the PPAL baseline and later `--method pal` for PAL inference/configs.

For the initial baseline, it is acceptable to keep the existing PPAL wildcard imports, then convert to the hook in the next commit. The hook is important before adding PAL because PAL will need its own model head and sampler registrations without making `tools/train.py` import every method unconditionally.

## Active Learning Runner Replacement

Do not directly copy `tools/run_al_voc.py` as the final runner. It should be used as the behavioral reference for a new ALOD runner:

```text
tools/run_active_learning.py
```

The new runner should preserve the PPAL round protocol:

```text
roundN/annotations/labeled.json
roundN/annotations/unlabeled.json
roundN/unlabeled_inference_result.bbox.json
roundN/annotations/uncertainty_new_labeled.json
roundN/annotations/uncertainty_new_unlabeled.json
roundN/image_dis.npy
roundN/diversity_inference_result.bbox.json
roundN/annotations/new_labeled.json
roundN/annotations/new_unlabeled.json
roundN/eval.txt
roundN/latest.pth
```

But it should replace shell-specific commands with Python equivalents:

- `os.makedirs(..., exist_ok=True)` instead of `mkdir -p`
- `shutil.copy2(...)` instead of `cp`
- `Path.unlink(missing_ok=True)` or guarded unlink instead of `rm -f`
- `subprocess.run([...], check=True)` instead of string-built `os.system(...)`

This is required for Windows compatibility and for reliable error propagation.

## Config Rewrite Plan

Create local active-learning experiment config:

```text
configs/experiments/ppal_retinanet_voc.py
```

It should replace `code_refs/PPAL/al_configs/voc/ppal_retinanet_voc.py` and point only to ALOD-owned config paths:

```python
config_dir = "configs/voc_active_learning/"
train_config = config_dir + "al_train/retinanet_26e.py"
uncertainty_infer_config = config_dir + "al_inference/retinanet_uncertainty.py"
diversity_infer_config = config_dir + "al_inference/retinanet_diversity.py"
```

Keep dataset/pretrain paths configurable and relative to the ALOD root:

```python
oracle_path = "data/VOC0712/annotations/trainval_0712.json"
init_label_json = "data/active_learning/voc/voc_827_labeled_1.json"
init_unlabeled_json = "data/active_learning/voc/voc_827_unlabeled_1.json"
output_dir = "work_dirs/retinanet_voc_ppal_7rounds_5percent_to_20percent"
```

The MMDetection model config stack should remain structurally identical for the first milestone:

```text
configs/voc_active_learning/al_train/retinanet_26e.py
configs/voc_active_learning/al_inference/retinanet_uncertainty.py
configs/voc_active_learning/al_inference/retinanet_diversity.py
configs/voc_active_learning/bases/al_retinanet_base.py
configs/voc_active_learning/bases/al_retinanet_inference_base.py
configs/voc_active_learning/bases/models/retinanet_r50_fpn.py
```

Fix base paths only if needed. `bases/al_retinanet_base.py` uses:

```python
mmdet_base = "../../_base_"
```

This requires `configs/_base_` to exist at the same relative depth as in the PPAL repo. If copying all `configs/` from PPAL, this remains valid.

Avoid hard-coded absolute paths. If ALOD later needs machine-specific locations, add CLI overrides or a local ignored config file rather than committing absolute paths.

## Minimal First Milestone

Goal: run one PPAL `RetinaNet + VOC` active-learning round from editable ALOD code, without PAL.

Acceptance criteria:

1. `python -c "import mmdet; print(mmdet.__file__, mmdet.__version__)"` resolves to `D:\DataDrift\GitHub\ALOD\mmdet`, not the conda site-packages installation.
2. `python tools/train.py configs/voc_active_learning/al_train/retinanet_26e.py --method ppal --cfg-options data.train.ann_file=<small_or_real_labeled_json> labeled_data=<same> unlabeled_data=<unlabeled_json> --work-dir <work_dir> --gpus 1` imports and builds the model config.
3. `python tools/test.py configs/voc_active_learning/al_inference/retinanet_uncertainty.py <checkpoint> --method ppal --format-only --eval-options jsonfile_prefix=<prefix> --cfg-options data.test.ann_file=<unlabeled_json> unlabeled_data=<unlabeled_json>` writes a `.bbox.json` containing `cls_uncertainty`.
4. The PPAL uncertainty sampler can create `uncertainty_new_labeled.json`.
5. The PPAL diversity inference can write `image_dis.npy`.
6. The PPAL diversity sampler can create final `new_labeled.json` and `new_unlabeled.json`.

For a fast smoke test, start with config/import/build checks and sampler tests against tiny synthetic COCO-style JSONs before running a full VOC training round.

## Key Risks

### Installed `mmdet` Shadowing Local `mmdet`

The `alod` conda environment already has `mmdet 2.20.0` installed from the PPAL reference. If ALOD root is not first on `PYTHONPATH`, imports may resolve to site-packages instead of local `ALOD/mmdet`.

Mitigation:

- Run commands from the ALOD root.
- Add a startup check in `tools/train.py`, `tools/test.py`, and `tools/run_active_learning.py` that logs `mmdet.__file__`.
- Prefer `python -m pip install -e .` from the ALOD root after local copy.
- Avoid executing from inside `code_refs/PPAL`.

### Windows Distributed Launch

`run_al_voc.py` assumes Linux shell commands and uses:

```text
python -m torch.distributed.launch --nproc_per_node=<gpus> ...
mkdir -p
cp
rm -f
stdout redirection with >
```

These are not portable to PowerShell/Windows. Distributed execution also needs extra care on Windows.

Mitigation:

- Make `tools/run_active_learning.py` use `subprocess.run` with argument lists.
- Support `gpus=1` and `--launcher none` as the first Windows milestone.
- Add distributed launch only after single-GPU PPAL works.
- If distributed is required later, prefer `torchrun` or a version-compatible launcher and test it explicitly on Windows.

### Torch/MMCV Version Constraints

Local `mmdet/__init__.py` enforces:

```text
mmcv >= 1.3.17 and <= 1.5.0
mmdet == 2.20.0
```

The working environment currently validated:

```text
torch 1.10.0
CUDA 11.3
mmcv-full 1.4.8
mmdet 2.20.0
```

Mitigation:

- Keep this version set for the first milestone.
- Do not upgrade PyTorch/MMCV while porting.
- Document exact environment in the experiment README once code is copied.

### Pycocotools and OpenCV

Runtime requirements include `pycocotools`; PPAL utils import `cv2`. Windows installs may fail or differ depending on wheels.

Mitigation:

- Keep `pycocotools` installed from conda-forge if pip fails.
- Keep `opencv-python` installed and verify `import cv2`.
- Add import smoke tests for `torch`, `cv2`, `mmcv`, and local `mmdet`.

### PPAL Diversity Assumptions

`RetinaHeadFeat` assumes:

- `total_images % world_size == 0`
- image id can be parsed from `os.path.split(img_meta["filename"])[-1].split(".")[0]`
- diversity inference uses distributed gather

The VOC PPAL config pads `uncertainty_pool_size` to be divisible by `gpus`, which satisfies the first assumption for the reference setup. The filename parsing assumption should be checked against VOC/COCO JSON image ids before changing dataset layout.

Mitigation:

- Preserve PPAL's padded `uncertainty_pool_size` calculation for the baseline.
- Use the reference VOC active-learning JSON format.
- Defer any robust image-id rewrite until after baseline parity.

## Suggested Commit/Phase Sequence

### Phase 1: Local PPAL Copy

- Copy `mmdet/`, `tools/train.py`, `tools/test.py`, `configs/`, `setup.py`, `setup.cfg`, `MANIFEST.in`, and `requirements/` from `code_refs/PPAL` into ALOD root.
- Do not restructure `mmdet/ppal` yet.
- Verify local import resolves to `ALOD/mmdet`.

### Phase 2: Local PPAL VOC Config

- Add `configs/experiments/ppal_retinanet_voc.py`.
- Rewrite paths to ALOD-local `configs/voc_active_learning`.
- Keep VOC split, budget, round count, and PPAL hyperparameters identical to reference.

### Phase 3: Windows-Safe Active Learning Runner

- Add `tools/run_active_learning.py` based on `run_al_voc.py` behavior.
- Replace shell commands with `pathlib`, `shutil`, and `subprocess.run`.
- Support `--method ppal`, `--resume`, `--config`, and single-GPU execution first.

### Phase 4: Explicit Method Registration

- Add method registration hook for `ppal` and future `pal`.
- Update `tools/train.py`, `tools/test.py`, and `tools/run_active_learning.py` to call it.
- Keep default method as `ppal`.

### Phase 5: PPAL Acquisition Module Cleanup

- Move or wrap PPAL sampler code under `methods/ppal/sampler`.
- Keep compatibility imports in `mmdet/ppal/sampler`.
- Add a shared sampler registry or an adapter around `mmdet.ppal.builder.SAMPLER`.

### Phase 6: PPAL Baseline Validation

- Run import smoke tests.
- Run config loading/build tests.
- Run sampler tests with tiny JSON fixtures.
- Run one VOC round when data/checkpoints are available.

### Phase 7: PAL Extension

- Add PAL-specific inference head and sampler under `methods/pal`.
- Register PAL modules through the same method hook.
- Reuse the validated PPAL train/eval/round protocol.

