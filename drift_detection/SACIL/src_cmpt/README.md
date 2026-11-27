# CMPT source snapshot

`src_cmpt/sacil` was mechanically copied from `src_explore/sacil` on
2026-08-07 so the successful prototype-transport exploration remains the
implementation baseline. Python cache directories were not copied.

The paper-facing checkpoint-frozen evaluator is implemented in
`sacil/cmpt/evaluator.py`. The entry point `scripts/evaluate_cmpt.py` prepends
`src_cmpt` to `sys.path`; it does not import the training package from `src`,
execute `ref_codes`, or modify the saved learner checkpoints.

The evaluator reconstructs each checkpoint's exact dynamic model structure,
reproduces its stored NME result, computes introduction-time full-data class
prototypes, fits a centered orthogonal rigid map from paired exemplar
features, and reports the paired CMPT-NCM result. At evaluation it replaces
old-class means only; current-session classes retain the control NME means, so
S0 is identical by construction.

`scripts/train_cmpt_common.py` is the isolated training entry point for the
controlled geometry/topology-preserving learner experiment. Its configs live
under `configs/cmpt/common_recipe/`. LUCIR, FGP-ICL, and CaSpeR-IL retain their
method-specific objectives while sharing the earlier LUCIR-like nuisance
recipe (80 incremental epochs, fixed 50:50 replay, and weight decay 5e-4).
The existing native-recipe trajectories and configs are not modified.
