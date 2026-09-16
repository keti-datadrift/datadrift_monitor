# PAL-only error audit

## Objective and scope

Execute the accepted PAL-only protocol: measure TP/FP predictor discrimination,
uncertainty by error type, actual acquired-batch coverage, and seven-round test
error trajectories. No EPAL analysis, new acquisition or detector training.

Source run (read-only):
`work_dirs/current_work/retinanet_voc_pal_7rounds_5percent_to_20percent/09-04-2026_19;37/seed_0`.

New outputs:
`work_dirs/analysis/09-09-2026_12;28_pal_error_audit/`.

## Implementation and environment

- `tools/analysis/infer_pal_audit.py`: reuse each round's dumped configuration,
  checkpoint and recorded evaluation overrides; export raw test predictions
  and native evaluation to the separate analysis directory.
- `tools/analysis/pal_error_audit.py`: refit the existing PAL classwise models
  on saved labeled detections, score saved unlabeled detections, and compute
  retrospective TIDE/predictor/batch/test diagnostics.
- `tools/analysis/summarize_pal_audit.py`: verify probabilities/entropy against
  the original 828 candidates per round, compute class-conditioned diagnostics
  and zero-detection visibility, then produce transition tables and figures.
- Analysis dependency installed in the existing `alod` environment:
  `tidecv==1.0.1`, with previously absent import dependencies
  `pandas==1.5.3`, `seaborn==0.12.2`, `appdirs==1.4.4` and `pytz`.
  Runtime requirements and method implementations are not changed.
- Python is run with `-B`; child processes inherit
  `PYTHONDONTWRITEBYTECODE=1`. `python -B -m pip check`: passed.

## Measurement conventions

- Acquisition: original score threshold 0.3 / NMS 0.5 / max 200 detections.
  Test: original threshold 0.01 / NMS 0.4 / max 100, batch 16, FP32.
- Predictor fitting exactly retains the repository's existing greedy target
  matching, including its lack of explicit difficult-object exclusion.
  Report native PAL-target performance separately from independent error labels.
- TIDE uses IoU 0.5 and background threshold 0.1. Before TIDE, remove difficult
  GT and detections whose highest-overlap same-class GT is difficult at IoU>=0.5,
  matching native VOC's ignore decision. Do not use TIDE crowd-ignore IoA for
  VOC difficult objects. TIDE matching and error-fixing oracles are unmodified.
- Test boxes are converted from inclusive zero-based xyxy to xywh with +1 width
  and height. Acquisition JSON keeps its existing continuous xywh convention.
- Native VOC07 AP is 11-point; TIDE AP/dAP is 101-point. dAP is in percentage
  points and its components are not additive. TIDE matching also differs from
  native VOC matching; use native mAP for reproduction checks.
  Reference implementation: [official TIDE](https://github.com/dbolya/tide).
  Error categories are operational TIDE assignments, not independently proven
  physical causes of a detector's failure.
  In particular, TIDE's Miss oracle removes irrecoverable missed GT from the
  AP denominator; its dAP is not the AP gain from training a real missing-object
  detector. FN also includes GT potentially recoverable by another error oracle.
- Detection-type distributions exclude ignored detections. Missing GT has no
  detection entropy. FN and TIDE Miss are separate quantities.
- Size strata use original-image box areas: small <32^2, medium [32^2,96^2),
  large >=96^2 pixels. These are exploratory COCO-style bins on VOC, not a
  change to native VOC evaluation.
- The supplementary score>=0.3 test counts filter predictions after the native
  test export and TIDE assignment; they retain test NMS=0.4, unlike acquisition
  NMS=0.5. They are not presented as acquisition outputs.
- Acquired-batch enrichment is measured against that round's full unlabeled
  pool, with both per-image and per-GT normalization. Round t selection is
  paired with test changes t -> t+1, not with round t's already-trained model.

## Measured findings: predictor and acquired batches

### 1. TP/FP discrimination is generally useful, not a wholesale failure

The positive label below is FP; the score is q_FP = 1 - p_TP. Independent
TIDE labels are used, with difficult-matched detections excluded.

| Acquisition round | Evaluated detections | Micro AUROC | Class-macro AUROC | FP AP | FP prevalence | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 48,786 | .8096 | .7167 | .8195 | .5902 | .1786 | .0991 |
| 2 | 46,945 | .8378 | .8024 | .7968 | .5052 | .1746 | .1101 |
| 3 | 48,949 | .8604 | .8373 | .8253 | .5076 | .1592 | .0992 |
| 4 | 46,256 | .8728 | .8501 | .8328 | .4846 | .1518 | .0935 |
| 5 | 41,944 | .8746 | .8557 | .8132 | .4448 | .1514 | .0951 |
| 6 | 38,810 | .8837 | .8671 | .8141 | .4193 | .1445 | .0916 |

There are zero fallback-scored detections across these six rounds. The original
saved candidates' probabilities and entropy reproduce with maximum absolute
difference 5.55e-17 in every round (828 candidates per round, 4,968 total).
This is a replay of the actual PAL scorer, not an independently designed proxy.

Native PAL-target AUROC (including difficult objects as the production matcher
does) rises .8050 -> .8767. Among nonignored detections, independent TIDE and
native targets disagree only 1-2 times per round; the difference in headline
AUROC mainly reflects the excluded difficult-associated detections.

The simple untrained ranking baseline 1 - detection score reaches AUROC .8061
in round 1 and .8822 in round 6. PAL's pooled AUROC advantage is only .0015-.0067
across the six rounds. This is a diagnostic comparator, **not** an acquisition
ablation: it does not prove the pre-NMS-count feature or classwise calibration
is unnecessary for downstream learning.

### 2. Localization is relatively harder; classification errors are not invisible

Round 6, subtype-versus-TP discrimination:

| Type | Count | Mean q_FP | AUROC vs TP | Class-macro AUROC | q_FP >= .5 | Coverage by top-10% entropy |
|---|---:|---:|---:|---:|---:|---:|
| TP | 22,538 | .1700 | — | — | 14.74% | 7.20% |
| Cls | 4,426 | .7775 | .9113 | .8993 | 82.72% | 11.70% |
| Loc | 6,436 | .6693 | .8648 | .8478 | 70.39% | 14.68% |
| Both | 1,382 | .7616 | .9053 | .8887 | 81.69% | 13.68% |
| Dupe | 1,548 | .7044 | .8895 | .8538 | 74.68% | 15.50% |
| Bkg | 2,480 | .6784 | .8678 | .8639 | 71.13% | 14.76% |

The top-10% calculation ranks all nonignored detections by LIUS entropy; it is
not the actual GUIDE-selected image batch. Classwise top-10% versions and
per-class subtype results are in `predictor_class_types.csv`.

Among detections given q_FP < .1 in round 6, 1,562 are actually FP; 830 of these
(53.1%) are localization errors. However, localization errors as a whole are
not neglected by entropy: 14.68% coverage exceeds the 10% uniform reference.
Do not claim that PAL fails simply because it does not distinguish one FP
subtype from another: its supervised task is TP/FP, not multiclass error typing.

### 3. Calibration is a more credible weakness than gross discrimination

Round 6 ECE is .0916 (10 equal-width probability bins). Examples:

- q_FP in [0, .1): 17,728 detections; predicted mean .0244, observed FP .0881.
- q_FP in [.2, .3): 1,304 detections; predicted .2480, observed FP .4709.
- q_FP in [.9, 1]: 8,818 detections; predicted .9596, observed FP .8459.

The probabilities are too extreme in these bins, even though the aggregate
predicted FP fraction .3969 is fairly close to the observed .4193. Averages can
hide bin-level miscalibration. The round-6 q_FP [.4, .6) region has 58.94% actual
FP, not 50%. This can affect the interpretation of maximum binary entropy as
uncertainty; these observations do not yet establish a downstream AP loss.

Classwise overall AUROC in round 6 is weakest for diningtable (.7664),
pottedplant (.7693), boat (.7737) and sofa (.7754), versus cat (.9206).
These classes have 189-3,465 labeled fit detections across all classes in this
round; their performance is not explained by constant fallback behavior.

### 4. Actual selected batches emphasize classification errors more than Miss

The following ratios compare error count per ordinary GT object in selected
images with the corresponding count per GT in the full round-specific pool.
A value of 1 is equal burden after this density adjustment, not a randomized
control estimate.

| Acquisition round | Cls enrichment | Loc enrichment | Miss enrichment |
|---|---:|---:|---:|
| 1 | 2.305 | 1.252 | .812 |
| 2 | 1.520 | 1.094 | .924 |
| 3 | 1.737 | 1.092 | .987 |
| 4 | 1.587 | 1.323 | .868 |
| 5 | 1.529 | 1.414 | .967 |
| 6 | 1.865 | 1.328 | .909 |

Round 6 selects 395 Cls, 409 Loc and 257 Miss instances in its 414 images.
Per-image enrichment is 2.943x, 2.096x and 1.435x respectively; selected images
also contain 1.578x as many GT objects per image. Thus Miss **does** appear in
selected batches, but its increased raw count does not survive GT-density
normalization. This distinction is essential; saying PAL never selects images
with missed objects would be false.

### 5. A measurable zero-detection candidate blind spot

| Acquisition round | Pool images with zero exported detections | Ordinary GT in those images | Selected images from this group |
|---|---:|---:|---:|
| 1 | 1,557 | 2,430 | 0 |
| 2 | 955 | 1,332 | 0 |
| 3 | 593 | 828 | 0 |
| 4 | 493 | 711 | 0 |
| 5 | 470 | 658 | 0 |
| 6 | 397 | 572 | 0 |

These are images without detections **after the actual acquisition threshold
and NMS**, not images with no proposals anywhere in the network. Their objects
cannot receive a per-detection LIUS score. Existing GUIDE/class-quota selection
did not reach this group in any measured round. Generalization still improves:
the group shrinks without direct acquisition, so this is not proof of permanent
learning failure or of how valuable those annotations would have been.
The implementation has a last-resort random refill if detection-based
candidates cannot fill the budget. No selected image in this audit came from
the zero-detection group; the structural observation concerns the ordinary
LIUS/GUIDE candidate path, not an assertion that every possible execution of
the code excludes these images.

## Measured findings: seven-round test trajectory

All seven checkpoints were inferred on all 4,952 VOC2007 test images. Every
native AP50 exactly reproduces its original scalar (maximum difference 0.0pp).

| Round | Native AP50 | Native AP75 | TIDE AP50 | Cls dAP | Loc dAP | Both dAP | Dupe dAP | Bkg dAP | Miss dAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 34.316 | 15.596 | 33.033 | 18.776 | 3.191 | 1.054 | .081 | 1.139 | 1.287 |
| 2 | 54.694 | 27.528 | 54.963 | 9.329 | 4.894 | 1.361 | .154 | 1.714 | 1.794 |
| 3 | 61.754 | 31.999 | 62.714 | 6.877 | 4.746 | 1.057 | .124 | 2.282 | 1.786 |
| 4 | 64.417 | 34.702 | 65.758 | 5.835 | 5.145 | 1.090 | .102 | 2.204 | 1.727 |
| 5 | 67.527 | 38.068 | 69.017 | 4.810 | 5.409 | 1.048 | .075 | 2.447 | 1.836 |
| 6 | 69.647 | 41.148 | 71.335 | 3.908 | 5.551 | .936 | .089 | 2.009 | 2.018 |
| 7 | 70.347 | 42.238 | 72.059 | 4.097 | 5.135 | .924 | .109 | 2.163 | 1.493 |

AP columns are percentages; dAP columns are percentage points. Native AP and
TIDE AP are different metrics and should not be numerically combined. Each
dAP independently fixes one category relative to its own TIDE baseline.

### Interpretation

- PAL's final detector metrics improve throughout: AP50 34.316 -> 70.347,
  AP75 15.596 -> 42.238. This is not an aggregate final-evaluation collapse.
- Classification's oracle impact falls sharply, 18.776 -> 4.097pp. It is not
  credible to say that this PAL run failed to learn classification errors.
- Localization becomes the largest residual category from round 5 onward.
  Its impact remains about 5pp in the later rounds, rather than falling like Cls.
  **This does not mean localization never improved**: native AP75 increases
  substantially. dAP depends on each round's underlying PR curve, so an
  increasing dAP is not on its own proof that localization became worse.
- Background remains meaningful (2.163pp in round 7); Both and duplicate
  contributions are smaller (.924pp and .109pp). Miss is 1.493pp at the final
  low-threshold test view, not the largest global test bottleneck.

### Absolute counts, thresholds and recall

Ordinary test GT count is fixed at 12,032 throughout.

| Round | Cls count, score>=.3 | Loc count, score>=.3 | Bkg count, score>=.3 | FN at native threshold | TIDE Miss | Recall at native threshold |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 4,668 | 1,868 | 801 | 1,784 | 516 | 85.17% |
| 2 | 2,699 | 1,956 | 970 | 1,345 | 407 | 88.82% |
| 3 | 2,766 | 2,106 | 1,388 | 1,159 | 344 | 90.37% |
| 4 | 2,506 | 2,086 | 1,249 | 1,114 | 322 | 90.74% |
| 5 | 1,895 | 1,977 | 1,309 | 1,041 | 310 | 91.35% |
| 6 | 1,653 | 2,003 | 1,209 | 1,047 | 311 | 91.30% |
| 7 | 1,878 | 1,996 | 1,316 | 957 | 268 | 92.05% |

Full-threshold localization FP count changes 44,954 -> 42,949, but the score>=.3
subset changes 1,868 -> 1,996. This supports persistent confidently emitted
location errors, not a simple all-error-count reduction. Confidence and the
number of emitted boxes also change with training; these counts do not alone
measure causal effects of acquisition. Full per-type counts and score>=.3
recall are available in `test_rounds.csv`.

FN falls 1,784 -> 957; TIDE Miss falls 516 -> 268. Thus the density-adjusted
Miss underrepresentation in acquired batches does **not** imply that test
misses fail to improve. Missing-object acquisition and test model coverage are
related but not interchangeable measurements.

### Class and size strata

Round 7 localization dAP is especially high for pottedplant (12.923pp), boat
(10.526pp), person (8.923pp), diningtable (6.779pp), train (6.658pp), and
motorbike (6.369pp). Round-6 predictor localization-vs-TP AUROC is .7279 for
pottedplant and .7090 for boat; these are examples of predictor weakness and
final localization burden occurring in the same classes, not proof of causality.

| Final-round object size | GT count | FN | TIDE Miss | Recall |
|---|---:|---:|---:|---:|
| Small | 516 | 167 | 87 | 67.64% |
| Medium | 3,813 | 493 | 159 | 87.07% |
| Large | 7,703 | 297 | 22 | 96.14% |

Small objects are 4.29% of test GT but account for 32.46% of TIDE Miss. This
identifies a remaining detector weakness; a same-protocol comparison is needed
before calling it a PAL-specific weakness rather than a common detector effect.

## Acquisition-to-next-round bridge

The following changes correctly pair acquisition t with evaluated model t+1.
Negative dAP change means the error oracle's impact decreased, not necessarily
a direct improvement in a fixed error count.

| Transition | AP50 gain | Cls dAP change | Loc dAP change | Miss dAP change |
|---|---:|---:|---:|---:|
| 1 -> 2 | +20.378 | -9.446 | +1.703 | +.507 |
| 2 -> 3 | +7.060 | -2.452 | -.148 | -.008 |
| 3 -> 4 | +2.663 | -1.042 | +.399 | -.059 |
| 4 -> 5 | +3.110 | -1.025 | +.264 | +.109 |
| 5 -> 6 | +2.120 | -.902 | +.142 | +.182 |
| 6 -> 7 | +.699 | +.190 | -.417 | -.525 |

Although Cls-enriched batches accompany large early reductions in Cls impact,
the last transition reduces Loc and Miss rather than Cls. There is no simple
law that more selected errors of one type necessarily reduces that type's
next-round dAP. `batch_to_test_transitions.csv` retains all six types, selected
counts and both enrichment measures for every transition.

## Supportable conclusions and limitations

Measured evidence supports these narrower problem statements:

1. PAL TP/FP prediction distinguishes classification errors reasonably well,
   while localization errors are relatively harder and are the largest late
   test residual. Some difficult classes show both patterns.
2. Binary TP/FP probabilities are materially miscalibrated in the unlabeled
   pool, so high discrimination does not guarantee reliable entropy values.
3. The actual detection-based candidate path has a zero-detection blind spot;
   Miss burden in acquired images is not enriched after GT-density adjustment.

Not supported: PAL broadly cannot predict errors; PAL never learns localization
or missed objects; every image with a miss is excluded; TP/FP binarization causes
the observed residuals; another method is better; or these patterns establish
novelty/SoTA superiority. Low entropy on a confidently identified FP is expected
and is not by itself a defect. Class composition, object difficulty, annotation
budget, learning dynamics and score thresholds remain confounders.

This is one **local PAL implementation, VOC/RetinaNet, seed 0, accelerated PoC
protocol**. It is not a multi-seed paper-protocol estimate. No EPAL run or
alternative acquisition/training was executed. Image-level GT-density
normalization does not fully control class or difficulty composition.

## Validation and completion

- Seven full real checkpoint inference jobs completed (34,664 image forwards);
  source-native AP50 reproduced with maximum absolute difference **0.0pp**.
- Predictor fits and retrospective evaluation completed for rounds 1-6;
  271,690 nonignored detection records evaluated across rounds (not independent
  samples across rounds). All 4,968 saved candidate scores reproduce within
  **5.55e-17**; actual 414-image batches match the labeled-pool additions.
- Runtime checks passed for pool disjointness/membership, original detection
  counts, TIDE prediction accounting, TP+FN=GT, and completed input exports.
  Calibration-bin counts cover every evaluated detection in each round.
- `python -B -m pip check`: no broken requirements.
- AST parsing of all three new Python tools: passed. `git diff --check` and
  a separate trailing-whitespace scan of the new untracked files: clean.
- Four figures were generated from actual outputs and all four were visually
  inspected. No synthetic/smoke/test-suite workflow ran.
- No new `.pyc` files found under tools/methods/mmdet/configs after task start;
  older pre-existing bytecode was preserved. No real experiment output removed
  or overwritten. Only new analysis tools, this record and separate generated
  analysis outputs were added.

## Output index

All generated paths below are relative to
`work_dirs/analysis/09-09-2026_12;28_pal_error_audit/`:

- `test_rounds.csv`, `test_classes.csv`, `test_sizes.csv`: detector metrics/errors.
- `predictor_rounds.csv`, `predictor_classes.csv`, `predictor_types.csv`,
  `predictor_class_types.csv`: predictor discrimination/type sensitivity.
- `predictor_calibration.csv`, `predictor_calibration_bias.csv`,
  `predictor_probability_strata.csv`: probability reliability.
- `selected_batch_errors.csv`, `acquisition_visibility.csv`,
  `batch_to_test_transitions.csv`: selected batches and next-round bridge.
- `candidate_reproduction.csv`, `inference_manifest.json`, `completion.json`:
  replay validation and provenance.
- `round_XX/test_predictions.pkl`, native `eval_*.json`, `inference.log`,
  `test_error_summary.json`: new test exports, not modifications to the run.
- Rounds 1-6 also have `predictor_detections.npz`,
  `acquisition_image_errors.csv`, `predictor_summary.json`.
- `test_error_trajectory.png`, `predictor_round6.png`,
  `selected_batch_enrichment.png`, `predictor_calibration.png`: figures.

## Reproduction commands (PowerShell, repository root)

The analysis tools require the existing `alod` runtime plus:

```powershell
python -B -m pip install tidecv==1.0.1 pandas==1.5.3 seaborn==0.12.2 appdirs==1.4.4
$env:PYTHONDONTWRITEBYTECODE = '1'
$palRun = 'work_dirs/current_work/retinanet_voc_pal_7rounds_5percent_to_20percent/09-04-2026_19;37/seed_0'
$palAudit = 'work_dirs/analysis/09-09-2026_12;28_pal_error_audit'
python -B tools/analysis/infer_pal_audit.py --run $palRun --output $palAudit
python -B tools/analysis/pal_error_audit.py --stage predictor --run $palRun --output $palAudit
python -B tools/analysis/pal_error_audit.py --stage test --run $palRun --output $palAudit
python -B tools/analysis/summarize_pal_audit.py --stage predictor --run $palRun --output $palAudit
python -B tools/analysis/summarize_pal_audit.py --stage report --run $palRun --output $palAudit
```

Existing completed inference exports are reused by the inference tool. Choose
a new analysis output directory if changing the input run or evaluation setup.
For this execution, the CPU test stage ran concurrently with sequential GPU
inference using `--wait-for-exports`; `OMP_NUM_THREADS=2` and `MKL_NUM_THREADS=2`
were set for CPU analysis subprocesses. All detector-training and acquisition
configuration files, original logs, annotations and checkpoints are untouched.
