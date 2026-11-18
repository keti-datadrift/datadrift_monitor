# PAL limitations and EPAL novelty: experiment plan

## Objective and evidence status

Translate `.chatgpt/ChatGPT-AL for OD 아이디어 구체화-20260908-2052.md`
into falsifiable experiments for EPAL. The document was read completely in
UTF-8, then checked against the PAL paper, TIDE paper/toolbox and local code.
This task plans experiments; it does not run them or change runtime settings.

Working question: does separately modeling classification, localization and
missed-object errors supply acquisition information and learning gains beyond
PAL's TP/FP uncertainty? A PAL weakness alone does not establish novelty, and
unequal error coverage alone does not establish an acquisition failure.

PAL's established contributions are an important comparison, not proof that it
is the universally best current method. A separate related-work comparison is
still needed before claiming an unprecedented failure-mode decomposition.

## Confirmed local implementation and artifacts

- PAL fits class-specific logistic TP/FP classifiers on native detection score
  and pre-NMS count. TP matching is score-ordered, same-class, one-to-one at
  IoU 0.5. Missing-class models yield probability 0.5; fit fallbacks must be
  audited because these produce high entropy without learned evidence.
- Local `pal:lius` includes class budgeting, maximum instance entropy per
  image/class, deduplication and refill. It is not a global entropy sum.
  PAL-full adds GUIDE and a two-times per-class candidate pool.
- PAL's count is same-class overlap count including the final box; EPAL support
  is maximum-overlap assignment, excludes identical boxes and includes class
  probabilities across labels. Neither count is a substitute for the other.
- PAL and EPAL acquisition configs use score threshold 0.3, NMS IoU 0.5,
  nms_pre 1000, max_per_img 200. Native VOC evaluation instead uses threshold
  0.01, NMS 0.4, nms_pre 5000, max_per_img 100. These are different diagnostic
  populations and must not be silently combined.
- EPAL's CECP/LECP are global (not class-specific) models fitted only on the
  foreground eligibility subset (GT IoU >= 0.1), then applied to all retained
  detections without FDP. Its overlapping proxy errors are not standard TIDE
  errors. MOCP's proxy miss count is not automatically standard false negatives.

Primary PAL run:

`work_dirs/current_work/retinanet_voc_pal_7rounds_5percent_to_20percent/09-04-2026_19;37/seed_0`

All rounds 1-7 retain latest/epoch-26 checkpoints. Rounds 1-6 retain both
`pal_labeled_detections.bbox.json` and `pal_unlabeled_detections.bbox.json`,
candidate lists and diagnostics. Round 7 correctly has no acquisition exports.
Inspected evaluation files contain aggregate metrics, not test box predictions.

New EPAL source:

`work_dirs/current_work/retinanet_voc_epal_cecp_7rounds_5percent_to_20percent/09-08-2026_20;22/seed_0/round_01`

Both versioned labeled/unlabeled feature files and predictor diagnostics exist.
All three feature pairs are exported even in CECP-only mode, so the five
predictor combinations can be fitted offline from those files. The run summary
was still marked running when inspected; this is not a completed-run analysis.

Random comparator:

`work_dirs/current_work/retinanet_voc_random_7rounds_5percent_to_20percent/09-05-2026_19;11/seed_0`

Existing result records show PAL's seven-round mean mAP 0.603862 versus old
ECPAL EUA-only 0.577252 and Random 0.574941. This is evidence against assuming
general PAL failure, not evidence for or against a specific error-type gap.
The old ECPAL predictor audit does not measure the new EPAL features.

## P0. Freeze analysis inputs and diagnostic definitions

Use PAL round 1 as the initial frozen state (one checkpoint, one L/U split).
Refit PAL predictors only from L and reproduce stored PAL-full selections
exactly before interpreting weaknesses. Record configuration, model weights,
image-ID hashes, class mappings, dataset ignore conventions and extractor
version. Identical mAP/seed/initial pools do not prove identical weights.
Compare actual detector state tensors or their canonical hashes before reusing
the new EPAL round-1 features. Otherwise export EPAL features on the chosen PAL
checkpoint. Do not equate whole checkpoint-file hashes with weight identity;
serialization metadata may differ.

For all later matched comparisons, fix a checkpoint AND its exact labeled and
unlabeled pools. PAL round 1-6 can provide six reference states without new
detector training. Replicate key findings on EPAL-origin checkpoints when
available to check dependence on the source trajectory.

Use official TIDE to externally label Cls, Loc, Both, Dupe, Bkg, Miss and TP,
starting with foreground/background IoU 0.5/0.1. Keep TP/ignore status separate
from error categories. Distinguish three GT quantities:

1. EPAL training proxy misses (current label definition).
2. Standard unmatched GT / false negatives under the evaluation matching rule.
3. TIDE Miss, which excludes GT accounted for by classification/localization
   errors. These are not interchangeable.

Map VOC difficult annotations deliberately: local pool conversion uses
iscrowd for difficult, whereas native evaluation uses VOC ignore behavior.
Verify box coordinate conventions, class IDs, image coverage and ignore rules.
Exclude ignored GT/detections from relevant diagnostic denominators. Do not
silently repair training labels as part of an analysis.

Keep two prediction views:

- acquisition view: exact 0.3/0.5/200 settings for selection mechanism;
- evaluation view: native evaluation settings for before/after AP and TIDE.

Add a separate acquisition-threshold sensitivity export at 0.05, with all other
acquisition settings unchanged, only if needed to distinguish threshold-caused
misses. This changes the candidate population, including pre-NMS support: it
must be recomputed, not approximated from existing compact features. Do not
attribute differences between 0.3 acquisition and 0.01 evaluation solely to
score threshold, since their NMS/top-k settings also differ.

## P1. Does PAL binary uncertainty hide useful error-type structure?

No detector retraining. Use complete stored PAL pools in all six rounds.

Measure global TP/FP discrimination/calibration first: classwise and macro
AUROC, AP with prevalence, Brier skill against labeled class prevalence,
calibration plots, sample counts and fallback frequencies. Evaluate on the
unlabeled pool using GT only retrospectively. Detector training examples are
not an unbiased validation set for auxiliary predictors.

For each external error category, report q_FP=1-q_TP and entropy distributions
(median, quantiles, histogram) and top-5/10/20-percent detection coverage.
Report AUC/AP for error-type versus TP; report type-versus-rest separately as
a descriptive category-separation test. A single q_FP cannot be expected to
separate one FP cause from all other FP causes, and low type-versus-rest AUC
alone is not proof of a bad binary predictor. Error probability calibration
must use aligned targets, not compare q_FP directly to a subtype indicator.

Plot one boundary per predicted class in [score, PAL pre-NMS count] coordinates
and color observations by external error type. Never pool class-specific
boundaries into one artificial classifier. Show all classes in supplements;
preselect main-figure classes by support frequency, not by the desired outcome.

Most informative contrast: within narrow PAL q_TP/entropy bins, does LECP/CECP
still distinguish their own errors and retrieve examples that PAL ranks low?
Condition or stratify by predicted class, confidence, support count, image
detection count and object size. Low entropy for confidently known FP is
expected for entropy sampling; only a later learning-gain experiment can show
that skipping those examples is harmful.

Outputs: per-round/class predictor table, error distributions, boundary panels,
conditional EPAL additional-signal table, fallback audit.

## P2. What does actual image acquisition retrieve, and where does GUIDE help?

Compare Random, confidence entropy, detection-count ranking, actual PAL-LIUS,
actual PAL-full and all five EPAL components on each fixed state. Use the same
final detections for score comparisons; verify detection ID/box/class/score
alignment across exporters first. Label synthetic score aggregations as
diagnostic controls, not native PAL.

Evaluate exact 414-image selection. Also recompute policy selection for budgets
100, 200, 414 and 828; PAL quotas/normalization can change with budget, so do not
assume all batches are prefixes of one global ranking. Detection-percent
retrieval from P1 is diagnostic, not an annotation budget.

For each error type m, with image count n_m(x), report:

`coverage_m = sum_{x in S} n_m(x) / sum_{x in U} n_m(x)`

`enrichment_m = mean_{x in S} n_m(x) / mean_{x in U} n_m(x)`

Also report total GT per image, detections per image, class composition,
duplicate/background burden, error rates per detection and misses per eligible
GT. Give both micro ratios and macro image/class results; zero denominators
are N/A, not zero. Random enrichment has expectation 1, not exactly 1 per draw;
use 100 pre-seeded selections for the CPU reference distribution.

Control detection density with count-stratified Random and within-density-bin
comparisons. Report PAL sum/max and EPAL sum/mean as labeled aggregation
controls. Revisit annotation cost using eligible GT box counts as a proxy,
explicitly not measured annotation time. Higher total error due only to more
boxes is not evidence for better failure-specific information.

Trace the PAL-full selection funnel: all pool images -> images eligible under
predicted classes -> LIUS candidate union -> GUIDE final batch -> refill.
Measure each stage's error coverage and class composition. This identifies
whether missed opportunities originate in the binary score, class quotas,
candidate filtering or GUIDE. A low candidate coverage cannot be repaired by
reranking only those candidates, although incidental/refill selection can
still acquire missed-object images.

For miss analysis, report ranks/selection rates for zero-detection images,
few-detection/high-miss images, and upper-decile miss counts. Predefine strata
and handle ties deterministically (or include all threshold ties and disclose
the resulting size). Condition on GT density and object size retrospectively.
Measure total FN and TIDE Miss separately. PAL has no direct per-instance term
for an absent detection, but it can select the image through other detections;
do not claim complete blindness. EPAL MOCP also has no discriminating evidence
when raw candidates are absent and both residual features equal zero.

Outputs: image-budget coverage curves, component enrichment/rate tables,
GUIDE funnel table, miss strata, density-controlled comparisons.

## P3. Are the new predictors valid on both their targets and external errors?

Fit on L, evaluate on U, using the complete new feature schema. CECP/LECP:
AUROC, AP/prevalence, Brier/log loss, calibration, classwise and macro metrics.
MOCP: MAE, Poisson deviance, R2, Spearman, high-miss retrieval and mean bias.
Count baselines must include constant labeled mean and detection-density
models, not only zero prediction. Evaluate each binary model first on its
foreground training domain, then audit all scored detections, especially Bkg.

Use an explicit cross-tab between EPAL proxy targets and external TIDE labels.
A good proxy predictor may still fail to locate standard localization/miss
errors. Do not report q_loc as calibrated probability of a different TIDE
event. Quantify the extra unmodeled Bkg/Dupe entropy after FDP removal.

Compare two-feature models with score-only/support-only (CECP/LECP) and
R_amt-only/R_conf-only (MOCP); measure feature correlation. This establishes
whether the second feature adds information rather than merely scale.

MOCP-only entropy and lambda rank identically (Poisson entropy is monotonic).
Treat this as a count-prediction component, and test its marginal benefit to
Cls+Loc. Entropy is predictive-distribution uncertainty, not measured parameter
or epistemic uncertainty. Record entropy/error and entropy/predictor-residual
associations as separate questions.

Use image-grouped labeled holdouts for selecting auxiliary hyperparameters;
unlabeled oracle labels are for post-hoc evaluation, never for actual ranking
or tuning. The prior ECPAL metrics must not be reused as new EPAL measurements.

## P4. Separate new features from error decomposition (core novelty ablation)

Use a controlled two-by-two diagnostic experiment, separate from native
algorithm comparisons. Fix training image support, optimizer regularization,
global versus classwise fitting, instance set and image aggregation.

| Diagnostic | Inputs | Prediction targets |
|---|---|---|
| A | PAL score/count | one TP/FP target |
| B | EPAL score/E_cls/E_loc union | one TP/FP target |
| C | PAL score/count | separate Cls and Loc targets |
| D | EPAL score/E_cls/E_loc union | separate Cls and Loc targets |

The shared feature union isolates target factorization; the deployable EPAL
with two features per head is evaluated alongside it. D vs B and C vs A test
decomposition at fixed inputs. B vs A tests feature improvement. Run a primary
matched-foreground-domain comparison, explicitly not original PAL, and a
class-conditioning control because native PAL has per-class models but EPAL
uses global ones. Include a matched-flexibility TP/FP control if the gain could
be attributed to extra heads/parameters. Keep the actual PAL-full/LIUS
baselines intact and report them separately.

Then compare D or native Cls+Loc with Cls+Loc+MOCP. Do not attach MOCP only to
the proposed side and infer all improvement came from Cls/Loc factorization.
Carry forward only informative ablations into expensive detector retraining.

Support for novelty: separate targets improve external error coverage and
later learning gain over a TP/FP predictor with the same richer inputs. If B
matches D, feature engineering is the better-supported explanation.

## P5. Does the additional supervision actually fix the claimed errors?

Run a controlled one-acquisition fork from one frozen L/U/checkpoint. Each
policy selects 414 images; train on L union S using the same existing 26-epoch
initialization/training protocol, paired RNG seeds and unchanged hyperparameters.
Do not fine-tune one branch from the acquisition checkpoint and initialize
another from the backbone. A fork is a separate causal experiment, not a
reduction of the normal seven-round run count.

Minimum branches: Random, PAL-LIUS, PAL-full, and the strongest justified EPAL
combination. Add Cls+Loc versus all-three for a miss claim; add the matched-input
TP/FP control from P4 for a decomposition claim. Use round 1 first and a
preselected later state, such as PAL round 4, for confirmation. Saved next-round
results are reusable only if selection IDs and training conditions match.

Evaluate every branch on the same held-out images. Report native VOC mAP/AP50,
diagnostic AP75 and consistently defined AP50:95, recall, and each TIDE dAP.
TIDE AP conventions can differ from native VOC, so label metric implementations
separately. Save per-image predictions; the existing scalar eval JSON is
insufficient. Evaluate per-error counts as secondary outcomes because fewer
detections can reduce FP counts while increasing missed GT.

For error contribution D_m (smaller is better):

`gain_m(policy) = D_m(before) - D_m(after_policy)`

`incremental_gain_m = gain_m(EPAL) - gain_m(PAL)`

Relative error reduction is secondary and undefined when the baseline is zero.
TIDE dAP terms are counterfactual contributions, not additive percentages of
100 - AP; do not sum their improvements to claim the AP gain is explained.
Include class AP and GT-matched localization/recall diagnostics to inspect
transitions across error categories rather than assume stable FP identities.

## P6. Confirm full active-learning performance

Retain seven evaluated pools, seed 0 for PoC, and three sequential seeds
(0, 1, 2) for shortlisted reporting comparisons. Priority baselines: Random,
PAL-LIUS, PAL-full and chosen EPAL; retain all five EPAL variants for component
ablation as resources permit. If novelty depends on MOCP or factorization,
include the corresponding ablation rather than only the best final model.

Report complete AP and TIDE trajectories, trapezoidal AP-versus-label-budget
area, final AP, inference/acquisition/training cost and annotated-box proxy
cost. Do not infer per-acquisition causality by comparing independently evolved
round-3 PAL and EPAL models. Formal cross-detector/dataset portability and
related-method coverage are later requirements, not established by VOC seed 0.

## Statistics, priorities and stopping interpretations

Bootstrap image clusters (e.g. 1000 fixed-seed resamples) for predictor/coverage
uncertainty; detections in an image are not independent. Distinguish image
bootstrap intervals from training-seed variability. Use paired seeds for model
comparisons; treat multiple rounds on one trajectory as correlated observations.
Present per-class sample sizes, all preregistered rounds and uncertainty, not
only significant or favorable subsets. Predefine primary external Loc/Miss
outcomes and AP noninferiority expectations before reading outcomes.

Execution order:

1. P0/P1 and PAL-only P2 on six saved rounds: CPU analysis, no detector training.
2. Match the round-1 EPAL artifacts or run one missing feature inference, then
   P2/P3/P4 on that state. New GPU inference only when artifacts are insufficient.
3. Export test predictions for saved PAL/Random checkpoints to describe existing
   error trajectories; this is inference, not retraining and not causal evidence.
4. P5 on the short list, then P6 full seven-round confirmation.

Decisions: a lower Loc/Miss coverage is motivation only if controlling class,
density and matching does not remove it, and adding the retrieved supervision
actually improves held-out outcomes. If PAL already captures/corrects those
errors comparably, drop the PAL-blind-spot claim. If new features explain the
gain, claim features rather than decomposition. If MOCP does not improve real
miss diagnostics and learning, do not claim missed-object novelty. Greater
error count alone is neither a learning gain nor an oracle upper bound.

## Sources and validation

- PAL primary paper, especially sections 3.2.1-3.2.3:
  https://arxiv.org/html/2605.10349v1
- TIDE primary paper: https://dbolya.github.io/tide/paper.pdf
- Official TIDE implementation: https://github.com/dbolya/tide
- Local source: `methods/pal/{lius,acquisition,inference}.py`,
  `methods/common/matching.py`, `mmdet/alod/models/retinanet/pal_head.py`,
  PAL/EPAL inference configs, VOC base/train configs, EPAL labels/predictors,
  entropy sampler, VOC conversion and evaluation source.
- Prior result records: September 5 PAL analysis and September 7 Random/ECA
  analysis. Artifact existence checked directly for all seven PAL rounds and
  the new EPAL round-1 files.

No predictor metrics were newly computed, no detector inference/training was
run, and no runtime or experiment outputs were changed. Only this experiment
plan was added. Git diff whitespace validation is performed for the record.
