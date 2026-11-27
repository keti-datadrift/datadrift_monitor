# MIAL Implementation Notes

## Goal

MIAL is the ALOD method name for MI-AOD. The implementation ports the MI-AOD
RetinaNet/VOC active-learning flow into ALOD while keeping the same experiment
protocol used by the other ALOD methods.

## Scope

MIAL is training-integrated. It is not only a post-training sampler like PAL,
PPAL, ECPAL, or Core-set. Each round trains a detector with MI-AOD's alternating
instance-uncertainty phases, exports image-level uncertainty for the current
unlabeled pool, evaluates the checkpoint, then acquires the next labeled pool.

The main runner remains `tools/run_active_learning.py`. MIAL-specific training
logic is isolated in `tools/train_mial.py`.

## Code Structure

- `tools/run_active_learning.py`
  - Adds `mial` as a supported method branch.
  - Uses `tools/train_mial.py` instead of the standard `tools/train.py`.
  - Keeps the common eval, acquisition result, candidate artifact, and pool
    update flow.

- `tools/train_mial.py`
  - Runs the MI-AOD phase schedule for one active-learning round.
  - Samples an unlabeled training subset with the same size as the labeled pool.
  - Saves `latest.pth`.
  - Exports `mial_uncertainty.json` from the full current unlabeled pool.

- `mmdet/alod/models/retinanet/mial_head.py`
  - Adds `RetinaHeadMIAL`.
  - Implements the MI-AOD two classifier branches, shared regression branch,
    MIL branch, min/max discrepancy phases, and top-k discrepancy uncertainty.

- `methods/mial/inference.py`
  - Loads and validates the compact MIAL uncertainty artifact.

- `methods/mial/scoring.py`
  - Ranks unlabeled images by image-level MI-AOD instance-discrepancy score.

- `methods/mial/acquisition.py`
  - Selects the top-budget ranked images.
  - Returns diagnostics and candidate records for common artifact writing.

- `configs/alod_mmdet/retinanet_voc_train_mial.py`
  - MIAL RetinaNet/VOC config.
  - Uses the same ALOD RetinaNet/VOC training protocol values as the other
    methods where applicable.
  - Adds MIAL-specific `mial_topk=10000` and `mial_lambda=0.5`.

- `configs/catalog/methods.py`
  - Adds the user-facing aliases `mial`, `mi-aod`, and `miaod`.

## Round Flow

For each round:

1. `mial train`
   - Train with `RetinaHeadMIAL`.
   - Use deterministic seed setup.
   - Alternate phases:
     - detection training on labeled pool,
     - instance uncertainty minimization,
     - instance uncertainty maximization,
     - detection training on labeled pool.
   - Save `latest.pth`.
   - Export `mial_uncertainty.json`.

2. `eval`
   - Evaluate `latest.pth` with the standard VOC mAP evaluation path.

3. `mial acquisition`
   - Load `mial_uncertainty.json`.
   - Rank unlabeled images by top-k instance discrepancy.
   - Select `budget` images.
   - Write `mial_candidates.json`, `mial_diagnostics.json`,
     `annotations/new_labeled.json`, and `annotations/new_unlabeled.json`.

## Experiment Settings

The active-learning protocol matches the existing ALOD RetinaNet/VOC setup:

- Detector: RetinaNet R50-FPN
- Dataset: VOC2007+2012 trainval, VOC2007 test
- Initial labeled pool: 827 images
- Acquisition budget: 414 images per round
- Rounds: 7
- Training batch size: same as ALOD RetinaNet/VOC config
- Optimizer and learning-rate schedule: same as ALOD RetinaNet/VOC MIAL config

MIAL-specific values:

- `mial_lambda = 0.5`
- `mial_topk = 10000`
- unlabeled training subset size equals the current labeled pool size
- one GPU per seed in the current `tools/train_mial.py` implementation

## Notes

The original MI-AOD code uses a custom training loop with global loss-state
switching. ALOD keeps that logic explicit through `tools/train_mial.py` and
`RetinaHeadMIAL.set_mial_phase(...)`, so the runner remains readable and the
method-specific behavior stays isolated.
