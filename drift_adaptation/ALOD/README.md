# ALOD

ALOD is a local-source implementation workspace for active learning in object
detection. It is organized around one user choice:

```text
method + detector + dataset
```

The runner resolves the cataloged experiment protocol and current runtime
settings from that choice, then uses the local source tree directly. This
repository is not installed with `pip install -e .` or `setup.py install`.

## Supported Targets

Current implemented targets:

| Method | Detector | Dataset | Notes |
| --- | --- | --- | --- |
| PPAL | RetinaNet | PASCAL VOC, COCO | DCUS + CCMS reproduction path |
| PAL | RetinaNet | PASCAL VOC, COCO | LIUS + GUIDE reproduction path |
| EPAL | RetinaNet | PASCAL VOC, COCO | Error Prediction-based Active Learning (EUA) |
| Core-set | RetinaNet | PASCAL VOC, COCO | Greedy k-center acquisition |
| MIAL | RetinaNet | PASCAL VOC, COCO | MI-AOD training and acquisition path |
| Random | RetinaNet | PASCAL VOC, COCO | Baseline acquisition |
| Entropy | RetinaNet | PASCAL VOC, COCO | Baseline acquisition |

`code_refs/` contains archived upstream/reference implementations only. Runtime
code is copied into this repository and should be imported from the local ALOD
source tree, not from `code_refs/`.

## Repository Layout

- `tools/run_active_learning.py`: public runner for executing AL rounds.
- `tools/internal/`: private detector training, inference, and MIAL training
  entrypoints invoked by the public runner.
- `tools/common/`: reusable runner and input-preparation libraries; these modules
  do not depend on private entrypoints.
- `configs/catalog/`: supported method/detector/dataset presets.
- `configs/alod_mmdet/`: minimal MMDetection configs used by ALOD runs.
- `methods/ppal/`: PPAL method implementation.
- `methods/pal/`: PAL method implementation.
- `methods/epal/`: EPAL method implementation.
- `methods/coreset/`: Core-set method implementation.
- `mmdet/`: local MMDetection backend used by the private detector entrypoints;
  method code consumes saved artifacts instead of importing it.
- `docs/`: implementation notes, plans, and run logs.

## Prepare Environment

Activate the environment:

```powershell
conda activate alod
```

Install the framework stack first. The currently validated target is
PyTorch 1.10, torchvision 0.11, and `mmcv-full` 1.3.17-1.5.0; install wheels
that match the local CUDA/driver setup. Then install ALOD's pip-level runtime:

```powershell
pip install -r requirements.txt
```

For PAL GUIDE with Google ViT embeddings, install the optional embedding
dependencies before running PAL full mode:

```powershell
pip install -r requirements/pal_embeddings.txt
```

## Prepare Data

Place the original VOC data under:

```text
data/VOCdevkit/VOC2007
data/VOCdevkit/VOC2012
```

Place the original COCO 2017 data under:

```text
data/coco/train2017
data/coco/val2017
data/coco/annotations/instances_train2017.json
data/coco/annotations/instances_val2017.json
```

The runner automatically prepares the dataset-specific initial active-learning
pools, RetinaNet ResNet-50 backbone checkpoint, and PAL GUIDE embedding cache
when the selected experiment needs them. It also builds the VOC0712 oracle JSON
from the original VOC files. Source datasets are not downloaded automatically.
Repository-local junctions or symlinks may point the dataset source paths to an
external dataset drive; generated pools and experiment outputs retain their
catalog locations.

The catalog applies the paper protocol for the selected dataset. VOC uses 827
initial labeled images, a 414-image budget, and seven evaluated pools from 5%
through 20%. COCO uses 2,365 initial labeled images, a 2,365-image budget, and
five evaluated pools from 2% through 10%. For both supported datasets, the
runner deterministically generates an independent initial pool for each
requested seed. Every method uses the same source pool for a given dataset and
seed. The generated files follow `voc_827_*_seed_{seed}.json` and
`coco_2365_*_seed_{seed}.json`; the run commands are unchanged.

The current VOC configuration is optimized for single-GPU PoC iteration. The
standard QualityEMA training path uses batch 16, four persistent workers, AMP
with dynamic loss scaling, learning rate 0.008, 32 warm-up iterations, and a
Quality EMA base momentum of 0.8514577710948755. VOC evaluation and acquisition
inference use batch 16 with four workers in FP32. The seven evaluated pools and
26-epoch schedule are unchanged. After evaluating round 7 at the final 20%
labeled pool, the runner carries that pool into the round-7 artifact layout and
skips the unused acquisition for a 22.5% pool. The runner also stops a VOC
training step after 20 consecutive non-finite gradient norms instead of
allowing a diverged model to complete. COCO settings are unchanged.

## Run

List supported presets:

```powershell
python -B tools/run_active_learning.py --list-presets
```

Run one PAL round:

```powershell
python -B tools/run_active_learning.py --method pal --detector retinanet --dataset voc --rounds 1 --gpus 1
```

Run PAL with a specific seed:

```powershell
python -B tools/run_active_learning.py --method pal --detector retinanet --dataset voc --gpus 1 --seed 0
```

Run EPAL with all three error predictors, the default seed 0, and all seven
VOC rounds:

```powershell
python -B tools/run_active_learning.py --method epal --detector retinanet --dataset voc --gpus 1
```

EPAL selects images directly by error-prediction entropy. Choose the predictors
with `--epal-components`:

| Variant | Option |
|---|---|
| CECP only | `--epal-components cecp` |
| LECP only | `--epal-components lecp` |
| MOCP only | `--epal-components mocp` |
| CECP + LECP | `--epal-components cecp+lecp` |
| All three (default) | `--epal-components cecp+lecp+mocp` |

For example:

```powershell
python -B tools/run_active_learning.py --method epal --epal-components cecp+lecp --detector retinanet --dataset voc --gpus 1
```

Only selected predictors are fitted and scored. Their inverse-mean entropy
weights sum to the number of active components; a single component has weight
one. EPAL has no FDP, ECA mode, or diversity reranking.

All predictors have two inputs: CECP uses detection score and the sum of support
probabilities for the final predicted class; LECP uses detection score and the
sum of support IoUs; MOCP uses residual amount and the sum of residual-weighted
native candidate confidence. CECP/LECP retain the existing foreground training
eligibility mask (GT IoU >= 0.1), but their entropy is applied to all retained
detections without a foreground-probability gate.

Feature JSON uses `schema=epal`, `schema_version=1`, and an `images` list.
Legacy ECPAL features cannot reconstruct the new evidence exactly and are
rejected: rerun EPAL feature inference using a compatible detector checkpoint.
Old method aliases and mode/config keys are rejected with migration guidance.

Generated run names include the canonical component combination, such as
`retinanet_voc_epal_cecp_lecp_7rounds_5percent_to_20percent`. Custom config
output roots receive an `epal_<components>` subdirectory. The resolved
combination appears in acquisition plans, run/round/aggregate summaries and
the seed TensorBoard stream. Predictor coefficients, scaler parameters,
sample counts, fallback reasons and entropy weights are saved in
`epal_diagnostics.json`; disabled predictors are marked explicitly.
Historical ECPAL results and documentation retain their original names.

Run the three reporting seeds sequentially:

```powershell
python -B tools/run_active_learning.py --method pal --detector retinanet --dataset voc --gpus 1 --seeds 0 1 2
```

Run the same PAL protocol on COCO:

```powershell
python -B tools/run_active_learning.py --method pal --detector retinanet --dataset coco --gpus 1 --seed 0
```

The default terminal output is concise. It prepares missing inputs, prints the
resolved run and output directory, then shows separate progress bars for the
current round's train, eval, and method inference steps. Acquisition is printed
as a short result line. Detailed command arguments, MMDetection logs, inference
logs, input preparation details, and acquisition details are saved under the run
directory instead of being streamed to the terminal. When multiple values are
passed through `--seeds`, each complete seed pipeline runs sequentially. Seed
pipelines never share a GPU concurrently.

All train, evaluation, and inference progress bars use image units. A completed
batch advances the bar by the effective batch size, so batch 16 is displayed as
`16/N`, `32/N`, and so on; a partial final inference batch is capped at the
dataset size. The training total counts every image slot processed across all
epochs, including samples repeated by MMDetection's grouped sampler padding.

New runs are stored under `work_dirs/current_work`. Runs archived after the
September 3 PoC setting changes are stored separately under
`work_dirs/backup_09-03-2026`. Single-seed and sequential multi-seed runs use
the same relative layout in either area:

```text
work_dirs/
  current_work/
    <experiment_name>/<MM-DD-YYYY_HH;mm>/
  backup_09-03-2026/
    <experiment_name>/<MM-DD-YYYY_HH;mm>/
  pal_embeddings/
```

Each timestamped run contains:

```text
<experiment_name>/<MM-DD-YYYY_HH;mm>/
  seed_0/
    active_learning_plan.json
    run_summary.json
    round_00/
    round_01/
  seed_1/
  seed_2/
  aggregate_summary.json
```

Important output files:

- `work_dirs/current_work/.../seed_*/active_learning_plan.json`: full command/acquisition plan.
- `work_dirs/current_work/.../seed_*/run_summary.json`: resolved method, detector, dataset, rounds,
  budget, output directory, prepared inputs, and round summary paths.
- `work_dirs/current_work/.../seed_*/round_XX/round_summary.json`: per-round step status, durations,
  TensorBoard location, and acquisition outputs. VOC round 7 records the terminal acquisition
  skip explicitly and keeps `selected_count` empty.
- `work_dirs/current_work/.../seed_*/events.out.tfevents.*`: seed-wide
  TensorBoard stream containing round-scoped console output and training
  scalars together with active-learning metrics.
- `work_dirs/current_work/.../seed_*/round_XX/*_diagnostics.json`: method-specific acquisition
  diagnostics.
- `work_dirs/current_work/.../seed_*/round_XX/*_candidates.json`: compact candidate rankings and
  final selection flags for method analysis.
- `work_dirs/current_work/.../aggregate_summary.json`: seed-level run paths plus round-wise
  metric means and standard deviations. VOC records `mAP`/`AP50`; COCO records
  `bbox_mAP`, `bbox_mAP_50`, and `bbox_mAP_75`.

Use `--verbose` when debugging to print the full plan and stream subprocess
output:

```powershell
python -B tools/run_active_learning.py --method pal --detector retinanet --dataset voc --rounds 1 --gpus 1 --verbose
```

## View Metrics with TensorBoard

Install the ALOD dependencies, including TensorBoard:

```powershell
pip install -r requirements.txt
```

New experiments write TensorBoard events directly while they run. The public
runner records command output, per-iteration training scalars, validation, pool,
acquisition, and duration scalars. Plain `.log` and `.log.json` files are not
retained.

Launch TensorBoard from the repository root:

```powershell
tensorboard --logdir work_dirs/current_work
```

If the console script is not on `PATH`, use the equivalent module command:

```powershell
python -m tensorboard.main --logdir work_dirs/current_work
```

Use `work_dirs/backup_09-03-2026` as the log directory only when inspecting
the archived pre-change experiments. `work_dirs/pal_embeddings` remains a
shared prepared-input cache and is not an experiment result directory.

TensorBoard discovers all generated event files recursively. The Scalars page
contains VOC mAP/AP50 or COCO bbox AP by active-learning round, labeled and
selected image counts, round durations, and per-round training loss/lr curves
for each seed. Each seed owns one TensorBoard event stream directly in its
existing `seed_*` directory. Round-specific values use tags such as
`round_01/train/loss` and `round_01/console/train`; no TensorBoard event is
written to the timestamp directory or `round_XX/logs/`. Therefore the Runs
selector shows one entry per seed instead of one entry per round. Cross-seed
mean/std values remain available in `aggregate_summary.json`.

Useful method aliases:

- `ppal`: PPAL DCUS + CCMS.
- `pal`, `pal:guide`, `pal/full`: PAL LIUS + GUIDE.
- `pal:lius`: PAL LIUS only.
- `epal`: EUA acquisition with configurable CECP/LECP/MOCP components.
- `coreset`, `core-set`, `kcenter`: Core-set greedy k-center.
- `mial`, `mi-aod`, `miaod`: MIAL/MI-AOD instance-discrepancy acquisition.
- `random`: random acquisition baseline.
- `entropy`: entropy acquisition baseline.
