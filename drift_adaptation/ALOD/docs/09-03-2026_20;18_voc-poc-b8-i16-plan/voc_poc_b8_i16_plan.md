# VOC PoC Batch-8 / Inference-16 Implementation Record

## Objective

Directly change the current RetinaNet/VOC experiment settings for the PoC
stage. Keep the existing seven-round protocol, use one seed by default, train
with batch 8 and AMP, run evaluation and acquisition inference with batch 16 in
FP32, and skip only the unused acquisition after protocol round 7.

No `--profile` option, paper-mode branch, PoC output suffix, or three-round
screening workflow will be added. COCO settings and experiment files are
outside this work item.

The runtime source, configuration, and public usage documentation were updated
in this work item. Historical experiment outputs were not modified.

## Fixed Runtime Settings

| Setting | New VOC value |
|---|---:|
| Protocol rounds | 7, unchanged |
| Epochs per round | 26, unchanged |
| Default seeds per invocation | 1, existing seed-0 default |
| Training samples per GPU | 8 |
| Evaluation/acquisition samples per GPU | 16 |
| Workers per GPU | 4 |
| Persistent workers | enabled |
| Training precision | AMP with dynamic loss scaling |
| Evaluation/acquisition precision | FP32 |
| Deterministic training | enabled, unchanged |
| Learning rate | 0.016 |
| Warm-up iterations | 63 |
| LR step | epoch 20, unchanged |
| Quality EMA base momentum | 0.9227446944279201 (`0.99 ** 8`) |
| Terminal acquisition | skipped after VOC round 7 |
| Pin memory | `False`, unchanged |

Batch 8 and inference batch 16 are fixed defaults. The implementation must not
silently reduce them on CUDA OOM. A failure must remain visible so any later
batch change is an explicit experiment-protocol decision.

## Implemented Changes

### 1. Directly update the common VOC config

Modify:

- `configs/alod_mmdet/retinanet_voc_base.py`

Set:

- top-level `data.samples_per_gpu=8`;
- `data.workers_per_gpu=4`;
- `data.persistent_workers=True`;
- `data.val.samples_per_gpu=16`;
- `data.test.samples_per_gpu=16`.

All current VOC acquisition inference configs inherit this base config, so
batch 16 will reach uncertainty, PAL, ECPAL, image-feature, and
detection-feature inference without duplicating the setting across six files.
The shared worker count also becomes 4.

Do not change image resolution, augmentation, model architecture, test
thresholds, NMS settings, or dataset paths.

### 2. Directly update the current VOC QualityEMA train config

Modify:

- `configs/alod_mmdet/retinanet_voc_train_quality_ema_26e.py`

Set the exact effective values:

```python
model = dict(
    bbox_head=dict(
        type='RetinaQualityEMAHead',
        base_momentum=0.9227446944279201,
    )
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(ann_file=None),
)

optimizer = dict(
    type='SGD',
    lr=0.016,
    momentum=0.9,
    weight_decay=0.0001,
)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=63,
    warmup_ratio=0.001,
    step=[20],
)

fp16 = dict(loss_scale='dynamic')
```

Keep the current 26 epochs, gradient clipping, final-only checkpoint, and
disabled in-training evaluation.

The EMA value preserves approximately the old decay per image exposure; it
does not make batch-8 training mathematically identical to batch-1 training.
This is an intentional PoC protocol change.

### 3. Keep evaluation and acquisition inference in FP32

The round evaluator loads the train config. Adding `fp16` to the train config
would therefore also make evaluation FP16 unless it is explicitly overridden.

Modify the VOC entry in:

- `configs/catalog/datasets.py`

Add a VOC-only evaluation override:

```python
mmdet_eval_cfg_options=dict(fp16=None)
```

The batch-16 value still comes from the inherited VOC base config. Acquisition
inference uses the separate VOC inference configs, which do not define
`fp16`, so it remains FP32.

No COCO dataset specification or `retinanet_coco_*.py` file will change.

### 4. Make persistent workers effective in inference

Modify:

- `tools/internal/infer_detector.py`

Pass:

```python
persistent_workers=cfg.data.get('persistent_workers', False)
```

to `build_dataloader`. This honors the direct VOC base setting. It has little
expected benefit for a one-pass inference iterator, but keeps the effective
DataLoader configuration consistent with the requested settings. COCO retains
the default `False` because its config is unchanged.

`pin_memory` remains unchanged because the local MMDetection dataloader
currently hard-codes it to `False`.

### 5. Preserve MIAL's separate training behavior

MIAL uses `retinanet_voc_train_mial.py` and a custom manual optimization loop
that does not currently implement the standard MMCV AMP hook. It is not part of
the current EUA research path.

Modify:

- `configs/alod_mmdet/retinanet_voc_train_mial.py`

Add `persistent_workers=False` explicitly so the new base setting does not
accidentally leak into MIAL. Keep MIAL batch 1, workers 2, LR, and FP32 behavior
unchanged. Batch 8/AMP applies to the common QualityEMA path used by EUA/ECA,
PAL, PPAL, Core-set, Random, and Entropy.

### 6. Skip acquisition only after the real VOC final round

Modify:

- `configs/catalog/datasets.py`
- `tools/run_active_learning.py`

Add a dataset setting such as `skip_terminal_acquisition=True` only to VOC.
Use `cfg['round_num']`, which remains 7, as the protocol-final round.

Refactor `build_round_plan` so:

- rounds 1 through 6 retain their current complete plans;
- round 7 contains train and evaluation only;
- the MIAL branch follows the same terminal decision even though its training
  plan is separate;
- arbitrary CLI partial execution does not redefine the terminal round;
- COCO behavior remains unchanged because it does not enable the setting.

The `--rounds` CLI remains available and its defaults are unchanged. No
three-round workflow or documentation will be added.

### 7. Keep terminal-round artifacts and summaries truthful

Acquisition currently creates each round's annotation output. When round 7 has
no acquisition, the summary must not advertise a newly selected 22.5% pool.

For VOC round 7:

- carry the round-6 labeled and unlabeled pools forward into the round-7
  annotation locations without selecting new images;
- record `selected_count=None`;
- record an explicit state such as
  `acquisition_status='skipped_terminal_round'`;
- record `terminal_round=True`;
- do not create candidate or acquisition-diagnostics artifacts;
- keep validation metrics and total-duration TensorBoard points;
- omit the round-7 `acquisition/selected_count` scalar.

Carrying the pool files forward preserves the existing
`round_XX/annotations` layout while making round 7's final pool exactly the
20% training pool rather than an unused 22.5% pool.

Update aggregate output so round 7's missing selection scalar is recognized as
an intentional terminal skip rather than an incomplete run.

During the runner audit, an existing related defect was found:
`_round_metrics` accepts three arguments but the aggregate builder passes a
fourth `round_summary` argument. Remove that extra argument as a minimal
required fix so a completed new run can generate its aggregate summary.

### 8. Keep the public experiment command simple

Update `README.md` to describe the new direct VOC defaults and the absence of
round-7 acquisition. The normal command remains:

```powershell
python -B tools/run_active_learning.py --method ecpal:eua-only --detector retinanet --dataset voc --gpus 1
```

It uses the existing default seed 0 and the existing seven rounds. Later
three-seed confirmation remains explicit and sequential:

```powershell
python -B tools/run_active_learning.py --method ecpal:eua-only --detector retinanet --dataset voc --gpus 1 --seeds 0 1 2
```

No `--profile` examples or three-round screening commands will be added.

## Subagent Execution

Three bounded subagent roles were used with non-overlapping file ownership
because all agents share one worktree.

### Subagent A: VOC configuration

Own:

- `configs/alod_mmdet/retinanet_voc_base.py`;
- `configs/alod_mmdet/retinanet_voc_train_quality_ema_26e.py`;
- `configs/alod_mmdet/retinanet_voc_train_mial.py`;
- `configs/catalog/datasets.py`;
- `tools/internal/infer_detector.py`.

Responsibilities:

- apply exact batch/LR/warm-up/EMA/AMP/worker values;
- add the VOC-only FP32 evaluation override and terminal flag;
- keep every COCO file and value untouched;
- report resolved config values to the main agent.

### Subagent B: runner terminal-round behavior

Own:

- `tools/run_active_learning.py`.

Responsibilities:

- skip inference/acquisition only at configured VOC round 7;
- carry forward the final 20% pool;
- update round/aggregate summaries and TensorBoard semantics;
- fix the related `_round_metrics` call-arity defect;
- preserve rounds 1 through 6 and non-enabled dataset behavior.

### Subagent C: independent read-only reviewer

Own no files.

Responsibilities:

- inspect the combined diff after A and B finish;
- verify batch-16 propagation through evaluation and every acquisition
  inference plan;
- verify FP32 evaluation/acquisition despite AMP training;
- verify no COCO behavior or files changed;
- verify terminal summaries refer only to real artifacts;
- report conflicts or omissions to the main agent.

### Main agent

Own:

- integration decisions;
- `README.md`;
- this task record under `docs/`;
- final source/config/diff validation.

The two writing subagents completed their separate areas before integration.
The main agent inspected and validated the combined changes, and the third
subagent performed a read-only integrated review. No subagent ran experiments
or modified `work_dirs`.

## Validation

### Performed checks

- Compiled the six modified Python files in memory with `compile(...)`: passed.
- Loaded the resolved QualityEMA and MIAL configs with the `alod` environment's
  `mmcv.Config`: all fixed batch, worker, precision, LR, warm-up, EMA, epoch,
  and LR-step assertions passed.
- Loaded all five VOC inference configs: batch 16, workers 4, persistent
  workers, and no effective FP16 passed for every config.
- Resolved the VOC and COCO catalog configurations: the terminal flag and FP32
  evaluation override are VOC-only; COCO retains its previous acquisition
  behavior.
- Verified MMCV 1.4.8 parses CLI `fp16=None` as the string `"None"`, and
  verified the inference entry point normalizes that value to Python `None`
  before deciding whether to wrap the model for FP16.
- Built VOC ECPAL plans in memory for rounds 1, 6, and 7. Rounds 1 and 6 retain
  acquisition, while round 7 contains only training and evaluation. Random and
  MIAL round-7 plans follow the same terminal rule. A COCO round-5 plan retains
  acquisition.
- Built a terminal round summary and aggregate in memory: round-7 validation,
  labeled-pool, and runtime TensorBoard points remain; the acquisition count
  point is absent; `selected_count` is not applicable; round-7 metrics remain
  in the aggregate.
- Inspected the carry-forward path: both labeled and unlabeled files are copied
  from round 6 to round 7, and their image counts plus sorted image-ID hashes
  are checked before an existing file is accepted and after a new copy.
- Ran `git diff --check`: passed, with only Git's existing LF-to-CRLF working
  tree warnings.
- Inspected the full diff and changed-file list. No COCO config file,
  experiment output, checkpoint, dataset, or TensorBoard event was changed.

No smoke, synthetic, disposable, or real GPU experiment was run, in accordance
with the repository policy. Consequently, batch-8 training and batch-16 FP32
inference fit on the RTX 3090 only after a real run proves it. There is no
automatic OOM fallback. The first actual run should monitor peak VRAM, AMP
loss-scale behavior, NaN/Inf loss, throughput, image-ID coverage, and metrics.

## Result and Compatibility

- The ordinary seven-round VOC command directly uses the faster PoC settings.
- There is no profile selection or preserved paper execution mode.
- Existing historical runs and artifacts remain unchanged, but new runs are a
  different training protocol and must not be numerically combined with old
  batch-1 aggregates.
- All current non-MIAL RetinaNet/VOC methods use the same batch-8 AMP training
  and batch-16 FP32 inference settings.
- Round 7 ends at the final 20% evaluation without unused acquisition.
- COCO remains unchanged.
