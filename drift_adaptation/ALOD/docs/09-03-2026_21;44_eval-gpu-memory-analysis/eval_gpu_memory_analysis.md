# Round-1 Evaluation GPU Memory Analysis

## Objective

Explain the increase in GPU memory observed during round-1 VOC evaluation
without interrupting or modifying the running experiment.

## Source Run

- Run: `work_dirs/retinanet_voc_ecpal_eua_only_7rounds_5percent_to_20percent/09-03-2026_21;27`
- Seed/round: `seed_0/round_01`
- Step: evaluation of 4,952 VOC2007 test images
- Runtime command: `tools/internal/infer_detector.py` with the QualityEMA
  training config, batch 16, and `fp16=None`

## Measured Evidence

- At approximately 49% progress, `nvidia-smi` reported 20,155 MiB used and
  4,171 MiB free on the 24,576 MiB RTX 3090.
- Six samples taken two seconds apart remained at exactly 20,155 MiB while GPU
  utilization varied between 19% and 100%. The observed memory had plateaued
  rather than continuing to grow during that interval.
- The evaluation process converted every final detection result from CUDA
  tensors to CPU NumPy arrays before appending it to the dataset-wide result
  list.
- The evaluation process ended successfully and the next ECPAL inference
  process started. Between processes, reported GPU usage fell to 702 MiB. The
  evaluation allocation therefore did not survive the process boundary.
- A read-only pass over the 4,952 VOC2007 test images reproduced the resize,
  divisor-32 pad, and batch-16 grouping. The largest collated spatial shape,
  608 x 1024, first occurred in batch 3. Around images 2,401-2,432 the batches
  were 608 x 928. A newly encountered maximum input shape at 49% does not
  explain the observed later increase.

## Interpretation

The evidence is most consistent with normal PyTorch CUDA caching allocator
behavior, possibly combined with allocation fragmentation or a later peak in
RetinaNet post-processing work:

1. Batch-16 FP32 evaluation allocates large FPN activations and dense RetinaNet
   class/box predictions.
2. Batches alternate between padded widths such as 928 and 1024. Temporary
   blocks of different sizes are also created by score filtering, box decode,
   and batched NMS. The number of candidates above the 0.01 score threshold can
   vary with image content.
3. If an existing cached block cannot satisfy a later request, PyTorch obtains
   another CUDA segment. Freed blocks normally remain reserved for reuse, so
   `nvidia-smi` can show a step upward that does not fall after the batch.

This does not currently look like predictions accumulating on the GPU or a
monotonic tensor-reference leak. The all-image result list grows on CPU, while
the sampled GPU usage plateaued and was released when evaluation exited.

This interpretation is strong but not a direct allocator measurement. The
running process did not expose `torch.cuda.memory_allocated()` and
`torch.cuda.memory_reserved()`, so the exact split between live tensors and
cached blocks cannot be reconstructed afterward.

## Follow-up

No runtime change was made. Calling `torch.cuda.empty_cache()` after every
batch is not recommended: it would discard reusable blocks and can slow
evaluation without reducing the live tensor requirement. If a later run grows
continuously or reaches OOM, add periodic allocated/reserved/peak CUDA counters
inside the evaluation loop to distinguish a real tensor leak from allocator
caching precisely.
