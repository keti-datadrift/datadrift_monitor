# EPAL implementation

## Objective

Implement the approved EPAL (Error Prediction-based Active Learning) plan in
`docs/09-08-2026_20;07_epal-implementation-plan/implementation_plan.md` using
`.chatgpt/error_predictor_feature_design.md` as the feature specification.

## Behavior and affected files

- Renamed `methods/ecpal` to `methods/epal`, the inference head to
  `RetinaHeadEPAL`, and the dataset exporter to `EPALFeatureCocoDataset`.
  Updated model/dataset package registrations, ALRetinaNet result conversion,
  both dataset inference configs, catalog references, runner stages and README.
- Removed ECA score branches, FDP fitting/prediction and foreground probability
  gating, the ECPAL diversity module, candidate expansion and score-mode aliases.
  Removed code is recoverable from Git; existing runs were not deleted or moved.
- CECP uses `[score, E_cls]`, LECP `[score, E_loc]`, and MOCP
  `[R_amt, R_conf]`. Classification support sums the unfactored probability for
  the final detection's class across its existing support set. Localization
  support sums IoU. Residual confidence sums residual times the native candidate
  score, including existing score factors. Final confidence comes from the
  exported detection score, not the maximum class probability.
- Kept candidate population, support assignment, NMS settings, label matching,
  foreground training eligibility and miss targets unchanged. Foreground
  eligibility is a label mask, not a learned FDP. Conditional CECP/LECP entropy
  now applies to all final detections, which can include background detections.
- Added `methods/epal/components.py` and `--epal-components` with the five
  approved combinations. Default is `cecp+lecp+mocp`; names/order normalize,
  duplicate/unknown/unsupported combinations fail before preparation, and
  other methods reject the option. CLI overrides config values. Only enabled
  predictors are fitted and scored, and only their labeled mean entropies enter
  inverse-scale weighting. Weights sum to the enabled component count.
- Added versioned compact feature output: `{schema: epal, schema_version: 1,
  images: [...]}`. Strict reader/exporter checks cover missing/non-finite values,
  detection/feature shapes and image coverage. Legacy artifacts require fresh
  inference; they cannot reconstruct the new support evidence exactly.
- Selected images are the highest EUA scores with existing image-ID tie
  ordering, capped by budget/pool size. Candidate artifacts retain selected
  rankings and component scores, without diversity metadata.
- Updated `tools/run_active_learning.py` and `configs/catalog/` for canonical
  component-specific result folders under `work_dirs/current_work`. Custom
  config roots receive an `epal_<components>` subdirectory. Acquisition plans,
  run/round/aggregate summaries and seed TensorBoard include the resolved
  combination. Round TensorBoard also includes active uncertainty means,
  weights and predictor sample/target statistics.
- Diagnostics include schema, feature order, scaler mean/scale, coefficients,
  intercept, constant/fallback state, sample counts, target rates and selections.
  Disabled predictors have `enabled: false` and no fabricated measured values.
- Existing SciPy, scikit-learn and NumPy requirements remain necessary and
  sufficient. Shared predictor wrapper changes are only EPAL naming in error
  messages; PAL behavior is preserved.

## Validation performed

All Python commands used `C:/Users/user/.conda/envs/alod/python.exe -B`.

- `tools/run_active_learning.py --help` and `--list-presets`: passed; only one
  EPAL method/preset per dataset and the five approved component choices.
- Parsed all 22 changed/new Python files with `ast.parse` and checked their
  whitespace: passed. `git -c core.safecrlf=false diff --check`: passed.
- Imported EPAL acquisition, renamed MMDetection head and dataset exporter;
  verified HEADS/DATASETS registration: passed.
- Resolved all 16 catalog presets and validated configured paths: passed.
- Loaded VOC/COCO EPAL inference configs using MMCV: passed. VOC inference
  batch remains 16; COCO remains 1. Loaded VOC training config and confirmed
  batch 16, learning rate 0.008, 26 epochs.
- Resolved all five actual CLI component configurations without executing
  experiments: canonical options and distinct expected output paths passed.
- Executed invalid CLI paths for the removed `ecpal:eua-only` name, duplicate
  `cecp+cecp`, unsupported `lecp+mocp`, and PAL with EPAL components. Each
  returned a nonzero status before data preparation. Initial inspection found
  argparse masking normalization details; a narrow ArgumentTypeError adapter
  fixed the message and the checks then passed.
- Reviewed exporter native-score provenance, support class indexing, empty
  candidate/detection handling, selected-only predictors/weights, diagnostic
  envelope and terminal-round TensorBoard handling.
- Searched remaining runtime ECPAL mentions: only explicit migration errors;
  README also documents legacy compatibility. Existing bytecode files inspected
  in methods/configs/mmdet-alod/tools predate the task; none was modified during
  validation. No test suite or experiment validation artifacts were created.

## Compatibility and follow-up

Existing experiments, checkpoints, source design and historical docs are
unchanged. ECPAL commands/config keys and feature JSON are intentionally not
accepted as EPAL. Renaming the inference head adds no learned parameters;
compatible detector state dictionaries can be used for fresh feature inference.

VOC batch 16/16, LR 0.008, AMP/FP32 split, seed policy, seven evaluated pools,
terminal acquisition skip and one TensorBoard stream per seed remain intact.
COCO receives shared-name/exporter integration, not protocol tuning.

No GPU forward pass, predictor fitting on newly exported features, or complete
training/acquisition experiment was run. End-to-end numerical validation and
predictor accuracy therefore remain to be established by the next real run.
The revised support sums retain detection-density dependence, and the retained
conditional error labels still warrant separate analysis.
