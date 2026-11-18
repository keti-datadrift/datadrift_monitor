# PAL-only diagnostic scope

## Objective

Narrow the September 8 novelty experiment plan to the user's immediate
priority: understand PAL alone, before testing or comparing EPAL. This is a
scope/protocol decision, not a report of newly measured predictor performance.

Reference PAL run:
`work_dirs/current_work/retinanet_voc_pal_7rounds_5percent_to_20percent/09-04-2026_19;37/seed_0`.

## Track A: TP/FP predictor and uncertainty by error type

Analyze acquisition rounds 1-6 on their own labeled/unlabeled pools. Refit PAL's
classwise logistic predictors using only labeled detections; use unlabeled GT
retrospectively for diagnosis. Keep model fit samples, class prevalence,
missing-class behavior and constant-model fallbacks explicit.

Evaluate TP/FP AUROC, AP/prevalence and Brier/calibration. Then label detections
with independent standard TIDE categories (TP, Cls, Loc, Both, Dupe, Bkg) and
report q_FP and entropy distributions, error-versus-TP discrimination, and
top-5/10/20-percent entropy coverage, per class and macro-averaged.

Distinguish error detection from uncertainty: a confidently identified FP has
low entropy by design. A subtype-versus-all-other-FPs AUC is not a fair primary
measure of TP/FP classification quality. Unequal subtype coverage is a pattern
to investigate, not intrinsically a defect. Miss is GT-level and cannot be
assigned a missing detection's per-instance q_FP.

As a lightweight link to training outcomes, use saved PAL-full selected IDs
to summarize error coverage/rates in each actual batch, versus its current
pool. This keeps the analysis about PAL and shows whether GUIDE/class quotas
compensate for a LIUS pattern. No new alternative acquisition or training is
required for this bridge.

## Track B: errors in the actual PAL-trained detector across rounds

Run each of the seven saved checkpoints on the same complete VOC2007 test
set, with identical native evaluation settings across checkpoints. Export
per-image boxes/scores/classes, then evaluate standard TIDE errors and native
VOC mAP/AP50, diagnostic AP75 and recall. Existing scalar eval JSON cannot
reconstruct error types. Additional inference may be necessary; detector
retraining is not needed.

Show round-by-error tables and curves for TIDE dAP and absolute/normalized
counts, not only error composition percentages. A type's share can rise while
its absolute burden falls; counts can fall by suppressing detections while
recall deteriorates. Keep FN, TIDE Miss and ignored GT distinct. Report class
and object-size strata to check composition effects. Mark native VOC versus
TIDE AP definitions separately; dAP contributions are not additive.

Use the acquisition/evaluation views separately: the current acquisition
threshold/NMS/max-per-image are 0.3/0.5/200; evaluation uses 0.01/0.4/100.
Respect VOC difficult/ignore and legacy box-coordinate conventions. Do not
diagnose the full detector using only acquisition-threshold boxes.

The local ALVOCDataset JSON export assumes an extra uncertainty column, while
ordinary evaluation returns standard bbox results. A future diagnostic tool
must use raw prediction output and an explicit dataset adapter or deliberately
extend the exporter; do not assume --format-only alone yields suitable boxes.

Round t's acquisition influences the model evaluated in round t+1. Pair the
six acquisition batches with transitions 1->2 through 6->7. Round 7 has no
subsequent acquisition.

## Interpretation and deliverables

Deliver: (1) predictor calibration/type-sensitivity tables, (2) seven-round
detector error trajectories, (3) six batch-to-next-round transition summaries.
Prioritize Track B's overall error trajectory and use Track A to interpret it;
both contribute different evidence and can be prepared independently.

Supportable conclusions include persistent PAL residual errors, weak predictor
discrimination for a subtype, or mismatch between LIUS coverage and residual
error burden. These results alone cannot establish that TP/FP binarization
causes an error plateau or that another method would improve it. Such claims
would later require controlled comparisons. EPAL analysis, new component
ablations, baseline training and full fork experiments are deferred.

## Validation and affected files

Re-read repository instructions, PAL classwise scoring and native VOC
evaluation/export source; checked worktree status. Only this protocol record
is added; prior plans and all experiment outputs remain unchanged. No predictor
fit, inference or new numerical analysis was run. Git diff whitespace check is
performed after writing the record.
