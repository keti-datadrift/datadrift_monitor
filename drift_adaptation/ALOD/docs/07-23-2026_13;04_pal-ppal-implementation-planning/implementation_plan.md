# PAL/PPAL Implementation Plan

## Goal

Implement and run both PPAL and PAL from editable local ALOD code, while keeping `code_refs/` as read-only reference material. The first target experiment is RetinaNet on PASCAL VOC, because the PPAL reference already provides VOC active learning configuration and RetinaNet-specific PPAL components.

The initial implementation should prioritize reproducibility over broad framework abstraction:

- Copy PPAL code into the ALOD root as local editable code.
- Do not import or execute modules from `code_refs/`.
- Preserve PPAL's active learning protocol for training, inference, sampling, JSON updates, and evaluation.
- Add PAL as a method implemented on top of the same local MMDetection/RetinaNet base.
- Build PAL incrementally: PPAL baseline, common AL loop, PAL-LIUS-only, then full PAL with GUIDE.

## Root-Level Project Structure

The project should not introduce an `alod/` package directory. The repository root should be the code root.

```text
ALOD/
  code_refs/
    PPAL/                         # Read-only reference. Never edited.
    PAL/                          # Read-only paper/reference material.
    ...

  mmdet/                          # Local editable copy of PPAL's MMDetection fork.
  configs/
    mmdet/
      voc/
        retinanet_train.py
        retinanet_ppal_uncertainty.py
        retinanet_ppal_diversity.py
        retinanet_pal_inference.py
    experiments/
      ppal_retinanet_voc.yaml
      pal_lius_retinanet_voc.yaml
      pal_retinanet_voc.yaml
      random_retinanet_voc.yaml
      entropy_retinanet_voc.yaml

  tools/
    train.py                      # Local editable copy derived from PPAL/MMDetection.
    test.py                       # Local editable copy derived from PPAL/MMDetection.
    run_active_learning.py        # Common active learning round entry point.

  methods/
    common/
      annotation_pool.py          # Labeled/unlabeled COCO-style JSON updates.
      matching.py                 # Detection-GT TP/FP matching for PAL LIUS.
      scoring.py                  # Entropy, normalization, aggregation helpers.
      registry.py                 # Method selection and registration.

    ppal/
      sampler.py                  # PPAL DCUS and diversity sampling entry points.
      datasets/
      models/
      configs/

    pal/
      sampler.py                  # PAL acquisition entry point.
      lius.py                     # Logistic classifier and LIUS scoring.
      guide.py                    # CWIE, RCDI, RCSP, and PAL score combination.
      embeddings.py               # Image embedding cache for RCSP.
      inference.py                # PAL inference output schema and parsing.
      datasets/
      models/
        pal_retinanet_head.py
      configs/

    random/
      sampler.py

    entropy/
      sampler.py

  datasets/
    create_initial_split.py
    coco_json.py
    voc_json.py

  data/                           # Dataset location, treated as user-managed.
  work_dirs/                      # Experiment outputs.
  docs/
```

This structure keeps detector framework code (`mmdet/`, `tools/train.py`, `tools/test.py`) separate from active learning method differences (`methods/`). PPAL and PAL can share the same detector, dataset format, split files, and round loop while differing only in acquisition logic and required inference output.

## Why `tools/run_active_learning.py` Replaces `runners/` and `backends/`

Earlier planning used the terms `runners/` and `backends/`. For this project stage, both are unnecessary abstractions.

`tools/run_active_learning.py` should be the single active learning orchestration entry point. Its role is to run the protocol:

```text
prepare round state
train detector on labeled.json
evaluate detector
run method-specific inference
run method-specific sampler
write next labeled/unlabeled JSON files
advance to the next round
```

The script should not implement PAL or PPAL math directly. It should delegate method behavior through a small method interface, for example:

```text
method.prepare_configs(...)
method.run_inference(...)
method.sample(...)
method.update_round(...)
```

A `backends/` directory is not needed yet because the first reproducible target is a single detector stack: PPAL's MMDetection fork with RetinaNet. If future work adds Detectron2, YOLOX, or Ultralytics backends, then a backend abstraction may become useful. Until then, `mmdet/` and `tools/` are the detector framework.

## Work Phases

### Phase 1: PPAL Local Copy Baseline

Objective: create a local editable copy of PPAL's code and verify that the PPAL RetinaNet + VOC protocol can run without importing from `code_refs/`.

Scope:

- Copy PPAL's `mmdet/`, `tools/train.py`, `tools/test.py`, relevant config files, and PPAL-specific model/dataset/sampler code into root-level local locations.
- Keep the copied code as close as possible to the PPAL reference for the first baseline.
- Adjust import paths only where necessary for the new local layout.
- Add local VOC RetinaNet PPAL configs under `configs/mmdet/voc/` and experiment YAML under `configs/experiments/`.

Acceptance criteria:

- `code_refs/` remains unmodified.
- Local `tools/train.py` and `tools/test.py` import from local `mmdet/`.
- Local PPAL sampler/model/dataset components are registered without importing from `code_refs/`.
- A config-load smoke test works for the local VOC PPAL train, uncertainty inference, and diversity inference configs.
- A minimal PPAL dry run can reach the first train/inference command construction stage.

### Phase 2: Common Active Learning Loop

Objective: implement a method-switchable active learning loop in `tools/run_active_learning.py`.

Scope:

- Define round state layout.
- Standardize COCO-style labeled/unlabeled JSON inputs and outputs.
- Provide shared helpers for annotation pool updates.
- Support `ppal`, `random`, and `entropy` method names initially.
- Keep image files in place; update annotation JSON files rather than moving dataset files.

Recommended round directory layout:

```text
work_dirs/<experiment_name>/
  round_00/
    annotations/
      labeled.json
      unlabeled.json
    checkpoints/
    inference/
    metrics.json
  round_01/
    annotations/
      labeled.json
      unlabeled.json
      selected.json
    checkpoints/
    inference/
    metrics.json
```

Acceptance criteria:

- The runner can initialize round 0 from configured labeled/unlabeled JSON files.
- The runner can invoke local train/test scripts through deterministic commands.
- Each method returns selected image ids using a common sampler contract.
- The runner writes next-round `labeled.json` and `unlabeled.json` without changing source dataset files.
- Random sampling is deterministic when a seed is provided.

### Phase 3: PAL-LIUS-Only

Objective: implement the minimum PAL core needed to select images using LIUS without GUIDE.

Scope:

- Add PAL inference output that records post-NMS detections plus PAL-specific features.
- Compute or approximate each detection's `pre_nms_count` from RetinaNet output.
- Match labeled detections to ground truth as TP/FP examples.
- Train class-wise logistic classifiers using:

```text
features = [pre_nms_count, confidence]
target = TP or FP
```

- Score unlabeled detections with TP probability uncertainty.
- Aggregate detection-level LIUS to image-level and class-level candidate pools.
- Implement class budget allocation compatible with PAL's described protocol.

Acceptance criteria:

- PAL inference produces a documented JSON or JSONL schema.
- TP/FP matching has unit tests on synthetic boxes.
- LIUS training handles classes with insufficient positive or negative examples through an explicit documented fallback.
- PAL-LIUS-only can select exactly the configured budget of unique image ids from a small synthetic pool.
- PAL-LIUS-only integrates with `tools/run_active_learning.py` as a method.

### Phase 4: PAL Full GUIDE

Objective: add PAL GUIDE components and combine them with LIUS for full PAL acquisition.

Scope:

- Implement CWIE.
- Implement RCDI.
- Implement RCSP using an image embedding cache.
- Implement final PAL score combination using paper settings:

```text
Score = alpha * LIUS + gamma * RCSP + beta * (CWIE + RCDI)
alpha = 0.9
beta = 0.04
gamma = 0.02
```

- Add deterministic duplicate handling when the same image is selected by multiple class-specific candidate lists.
- Cache embeddings in `work_dirs/`, not in `data/` or `code_refs/`.

Acceptance criteria:

- GUIDE components have synthetic tests for shape, determinism, and normalization behavior.
- Embedding cache keys include image identity and encoder settings.
- PAL full sampler returns a stable selection with a fixed seed.
- PAL full method can be selected from the common runner.
- PAL full output includes per-image diagnostic scores for LIUS, CWIE, RCDI, RCSP, and final score.

### Phase 5: Validation

Objective: verify that the local implementation is structurally correct before running expensive experiments.

Scope:

- Import smoke tests.
- Config load tests.
- Synthetic sampler tests.
- VOC mini-pool dry run.
- Read-only check for `code_refs/`.

Acceptance criteria:

- No local command modifies `code_refs/`.
- All new Python files compile.
- Sampler tests pass.
- A one-round VOC mini-pool run completes or reaches a clearly documented external dependency limit.
- PPAL and PAL configurations produce comparable round directories and metrics files.

## Agent and Worker Plan

### Main Orchestration Agent

Write scope:

```text
docs/07-23-2026_13;04_pal-ppal-implementation-planning/
```

Responsibilities:

- Maintain implementation plan and integration checklist.
- Define worker write scopes and dependency order.
- Record ambiguous PAL implementation points.
- Keep `code_refs/` read-only.

### Paper/Settings Worker

Write scope:

```text
docs/<timestamp>_pal-ppal-paper-settings/
```

Responsibilities:

- Read PAL CVPR 2026 PDF and PPAL reference config.
- Extract VOC/RetinaNet settings, budgets, thresholds, schedule, and PAL hyperparameters.
- Identify all paper-to-code mapping decisions.

Depends on:

- This orchestration plan.

### PPAL Port Worker

Write scope:

```text
mmdet/
tools/train.py
tools/test.py
configs/mmdet/
methods/ppal/
```

Responsibilities:

- Copy PPAL code from `code_refs/PPAL` into editable local code.
- Adjust imports for the local layout.
- Preserve behavior for the first baseline.
- Avoid algorithmic changes during porting.

Depends on:

- This orchestration plan.

### Common AL Loop Worker

Write scope:

```text
tools/run_active_learning.py
methods/common/
methods/random/
methods/entropy/
datasets/
configs/experiments/
```

Responsibilities:

- Implement the active learning round protocol.
- Define method interface.
- Implement baseline random and entropy samplers.
- Manage labeled/unlabeled JSON updates.

Depends on:

- PPAL Port Worker for local train/test command compatibility.

### PAL-LIUS Worker

Write scope:

```text
methods/pal/lius.py
methods/pal/inference.py
methods/pal/models/
methods/common/matching.py
configs/mmdet/voc/retinanet_pal_inference.py
```

Responsibilities:

- Implement PAL inference schema.
- Add pre-NMS count support.
- Implement TP/FP matching and class-wise LIUS classifiers.
- Implement LIUS-only image selection.

Depends on:

- PPAL Port Worker.
- Common AL Loop Worker.
- Paper/Settings Worker for exact PAL settings.

### PAL-GUIDE Worker

Write scope:

```text
methods/pal/guide.py
methods/pal/embeddings.py
methods/pal/sampler.py
```

Responsibilities:

- Implement CWIE, RCDI, RCSP.
- Add image embedding cache.
- Combine LIUS and GUIDE into final PAL sampling.

Depends on:

- PAL-LIUS Worker.
- Paper/Settings Worker.

### Verification Worker

Write scope:

```text
tests/
docs/<timestamp>_pal-ppal-verification/
```

Responsibilities:

- Add targeted tests and smoke checks.
- Verify imports, config loading, JSON updates, sampler behavior, and read-only reference integrity.
- Record validation commands and outcomes.

Depends on:

- Each implementation phase as it lands.

## Dependency Order

```text
1. Orchestration Agent
2. Paper/Settings Worker and PPAL Port Worker in parallel
3. Main integration review of PPAL local baseline
4. Common AL Loop Worker
5. PAL-LIUS Worker
6. Verification Worker pass 1
7. PAL-GUIDE Worker
8. Verification Worker pass 2
9. Main final integration review
```

## PAL Risks and Ambiguous Points

These decisions must be resolved and documented before claiming PAL reproduction.

- TP/FP matching threshold: likely class match plus IoU >= 0.5, but the exact paper/code assumption must be recorded.
- Pre-NMS count definition: whether it counts candidate boxes for the same predicted class, all classes, or boxes surviving score threshold before NMS.
- Confidence feature for LIUS: whether to use post-NMS score, pre-NMS class score, or calibrated score from the detector head.
- Class-wise logistic fallback: rare classes may have only TP or only FP examples in early rounds.
- Class budget formula: exact implementation of PAL's class budget equations must be mapped from paper notation to VOC class counts.
- Duplicate selected images across classes: de-duplication and budget refill policy must be deterministic.
- CWIE probability source: full class probability vector versus selected detection score distribution.
- RCDI rare-class definition: whether rarity is based on current labeled counts, unlabeled predictions, or full oracle counts.
- RCSP image encoder: PAL paper settings must be matched as closely as possible; any replacement encoder changes reproducibility.
- Score normalization: LIUS, CWIE, RCDI, and RCSP must be normalized consistently before weighted combination.
- Runtime cost: PAL inference and embedding extraction may dominate VOC runs if not cached.

## Key Decisions

- Use RetinaNet + PASCAL VOC as the first experiment target.
- Keep `code_refs/` read-only.
- Use local editable copies of PPAL code instead of importing from `code_refs/`.
- Avoid an `alod/` package directory.
- Use `tools/run_active_learning.py` as the common active learning entry point.
- Defer `backends/` until a second detector framework is required.
- Implement PAL incrementally, with PAL-LIUS-only as the first PAL milestone.
