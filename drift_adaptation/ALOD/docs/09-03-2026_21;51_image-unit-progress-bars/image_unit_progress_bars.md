# Image-Unit Progress Bars

## Objective

Use one progress convention for training, evaluation, and acquisition
inference: the denominator is the total number of images processed by the step,
and each completed model batch advances the bar by its image batch size.

For a 1,000-image inference set with batch 16, the visible sequence is
`16/1000`, `32/1000`, and so on, ending at `1000/1000` for the partial final
batch.

## Affected Behavior and Files

Modified `tools/run_active_learning.py`:

- Resolve the effective train or test `samples_per_gpu` from each command's
  MMDetection config.
- Multiply by the GPU count to obtain the effective global image batch.
- Display every model-execution progress bar with unit `img`.
- Convert training's global batch-iteration records to processed-image counts.
- Coalesce MMDetection's per-image evaluation/inference records at batch
  boundaries, while always emitting the final dataset count.

Modified `README.md` to document the common image-unit convention.

The current VOC standard training example changes from `1/1378`, `2/1378`,
and so on to `16/22048`, `32/22048`, and so on. The denominator includes the
26 epochs and the full batches produced by grouped-sampler padding. This is the
actual number of image slots passed through the model; repeated padding samples
are intentionally counted. TensorBoard scalar step numbering and experiment
math are unchanged.

The active run started at `09-03-2026_21;27` already loaded the old runner code
and is not modified. The new display applies from the next runner invocation.

## Validation

- Resolved an ECPAL/VOC train command and evaluation command to effective batch
  16.
- Fed synthetic progress records to the existing in-memory progress parser:
  training records advanced `16/1600`, then `32/1600`; a 1,000-image inference
  sequence advanced only at 16-image boundaries and finished with the final
  8-image increment.
- Compiled `tools/run_active_learning.py` in memory.
- Ran `git diff --check`.

No experiment, smoke workflow, or output artifact was created or modified.
