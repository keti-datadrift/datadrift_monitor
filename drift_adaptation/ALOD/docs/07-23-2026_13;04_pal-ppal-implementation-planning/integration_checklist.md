# PAL/PPAL Integration Checklist

## Repository Safety

- [ ] `code_refs/` is treated as read-only reference material.
- [ ] No command writes to `code_refs/`.
- [ ] No local code imports from `code_refs/`.
- [ ] No generated output is written under `code_refs/`.
- [ ] Dataset files under `data/` are not moved or modified by active learning rounds.
- [ ] Experiment outputs are written under `work_dirs/`.
- [ ] Documentation is written under timestamped `docs/<MM-DD-YYYY_HH;MM_title>/` folders, not directly as `docs/*.md`.

## Target Experiment

- [ ] First target is RetinaNet + PASCAL VOC.
- [ ] PPAL VOC reference config is mapped to local config files.
- [ ] Initial labeled/unlabeled JSON paths are configurable.
- [ ] Oracle annotation path is configurable.
- [ ] Round count, budget, and budget expansion ratio are configurable.
- [ ] Random seeds are explicit for split and sampler behavior.

## Phase 1: PPAL Local Copy Baseline

- [ ] Local `mmdet/` exists as an editable copy derived from PPAL.
- [ ] Local `tools/train.py` exists.
- [ ] Local `tools/test.py` exists.
- [ ] Local PPAL model components are registered.
- [ ] Local PPAL dataset components are registered.
- [ ] Local PPAL sampler components are registered.
- [ ] Local VOC train config exists.
- [ ] Local VOC PPAL uncertainty inference config exists.
- [ ] Local VOC PPAL diversity inference config exists.
- [ ] Config loading does not require `code_refs/`.
- [ ] Import smoke test passes.
- [ ] No algorithmic behavior is intentionally changed during the first port.

Acceptance gate:

- [ ] A local PPAL VOC baseline can construct train, uncertainty inference, diversity inference, and sampler steps without importing from `code_refs/`.

## Phase 2: Common Active Learning Loop

- [ ] `tools/run_active_learning.py` exists.
- [ ] The script accepts an experiment config path.
- [ ] The script accepts or derives the method name.
- [ ] Round 0 state is initialized from configured labeled/unlabeled JSON files.
- [ ] Round directories are deterministic and clearly named.
- [ ] Training command construction is logged.
- [ ] Evaluation command construction is logged.
- [ ] Method-specific inference command construction is logged.
- [ ] Sampler output is recorded as selected image ids.
- [ ] Next-round `labeled.json` and `unlabeled.json` are written.
- [ ] Image files are not moved.
- [ ] The common sampler interface is documented in code or method README.

Acceptance gate:

- [ ] `ppal`, `random`, and `entropy` methods can be selected through the same runner interface.

## Phase 3: PAL-LIUS-Only

- [x] PAL inference output schema is documented.
- [x] PAL inference records image id, bbox, category id, confidence, and pre-NMS count.
- [x] Labeled detection to GT matching is implemented.
- [x] Matching tests cover exact match, no match, class mismatch, and overlapping detections.
- [x] Class-wise logistic classifier training is implemented.
- [x] Insufficient-class fallback behavior is explicit and deterministic.
- [x] Detection-level LIUS score is computed.
- [x] Detection-level LIUS is aggregated to image-level score.
- [x] PAL class budget allocation is implemented.
- [x] Candidate images are selected per class.
- [x] Duplicate image handling is deterministic.
- [x] PAL-LIUS-only writes diagnostic score output.

Acceptance gate:

- [x] PAL-LIUS-only can select the configured number of unique images from a synthetic unlabeled pool and integrate with `tools/run_active_learning.py`.

## Phase 4: PAL Full GUIDE

- [x] CWIE is implemented.
- [x] RCDI is implemented.
- [x] RCSP is implemented.
- [x] Image embedding source is configurable.
- [x] Image embeddings are cached under `work_dirs/`.
- [x] Cache metadata records encoder configuration and image identity.
- [x] Score normalization is deterministic.
- [x] Final PAL score uses configured alpha, beta, and gamma.
- [x] Per-image diagnostics include LIUS, CWIE, RCDI, RCSP, and final score.
- [x] PAL full sampler handles duplicate class selections and budget refill deterministically.

Acceptance gate:

- [x] Full PAL can run through the common sampler interface and produce a selected image list plus diagnostic score file.

## Phase 5: Validation

- [ ] Python compile check runs for new Python files.
- [ ] Import smoke test runs for local `mmdet`, `methods.ppal`, and `methods.pal`.
- [ ] Config load smoke tests run for PPAL and PAL VOC configs.
- [ ] Synthetic annotation pool update test passes.
- [ ] Synthetic TP/FP matching test passes.
- [ ] Synthetic PPAL/Random/Entropy/PAL sampler tests pass.
- [ ] One-round VOC mini-pool dry run is attempted after data paths are configured.
- [ ] Validation log records exact commands and results.
- [ ] Any skipped validation includes a concrete reason.

Acceptance gate:

- [ ] PPAL and PAL produce comparable round output structures under `work_dirs/`.

## Worker Write Scopes

Orchestration Agent:

```text
docs/07-23-2026_13;04_pal-ppal-implementation-planning/
```

Paper/Settings Worker:

```text
docs/<timestamp>_pal-ppal-paper-settings/
```

PPAL Port Worker:

```text
mmdet/
tools/train.py
tools/test.py
configs/mmdet/
methods/ppal/
```

Common AL Loop Worker:

```text
tools/run_active_learning.py
methods/common/
methods/random/
methods/entropy/
datasets/
configs/experiments/
```

PAL-LIUS Worker:

```text
methods/pal/lius.py
methods/pal/inference.py
methods/pal/models/
methods/common/matching.py
configs/mmdet/voc/retinanet_pal_inference.py
```

PAL-GUIDE Worker:

```text
methods/pal/guide.py
methods/pal/embeddings.py
methods/pal/sampler.py
```

Verification Worker:

```text
tests/
docs/<timestamp>_pal-ppal-verification/
```

## Dependency Order

- [ ] Orchestration Agent finishes planning docs.
- [ ] Paper/Settings Worker extracts exact PAL and PPAL settings.
- [ ] PPAL Port Worker creates local editable PPAL baseline.
- [ ] Main agent reviews PPAL port against reference behavior.
- [ ] Common AL Loop Worker implements method-switchable round loop.
- [ ] PAL-LIUS Worker implements minimum PAL acquisition.
- [ ] Verification Worker performs first validation pass.
- [ ] PAL-GUIDE Worker implements full PAL scoring.
- [ ] Verification Worker performs second validation pass.
- [ ] Main agent performs final integration review.

## Ambiguity Resolution Checklist

- [ ] TP/FP IoU threshold selected and documented.
- [ ] Pre-NMS count definition selected and documented.
- [ ] LIUS confidence source selected and documented.
- [ ] Class budget formula mapped from paper to implementation.
- [ ] Fallback for one-class logistic training selected and documented.
- [ ] Duplicate image de-duplication/refill policy selected and documented.
- [ ] CWIE probability source selected and documented.
- [ ] RCDI rare-class source selected and documented.
- [ ] RCSP image encoder selected and documented.
- [ ] Score normalization strategy selected and documented.
- [ ] Any deviation from PAL paper settings is recorded as a reproducibility limitation.
