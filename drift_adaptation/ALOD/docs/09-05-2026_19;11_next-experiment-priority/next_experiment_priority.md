# Next Experiment Priority

## Objective

Choose the next VOC experiment after completing PAL, EUA-only, and EUA-full
under the current fast single-seed PoC protocol.

## Evidence

The three matching seed-0 runs finish close together:

| Method | Final mAP | Seven-round mean |
|---|---:|---:|
| EUA-full | 0.710689 | 0.583702 |
| EUA-only | 0.710058 | 0.577252 |
| PAL | 0.703466 | 0.603862 |

EUA-profile diversity changes nearly half of the initial EUA top-414 batch but
improves final mAP over EUA-only by only 0.000631. No Random result exists under
the current fast protocol. Without that baseline, it is not yet possible to
determine whether the small final differences among these active-learning
methods represent a meaningful advantage over uninformed sampling.

## Decision

The next GPU experiment should be Random with the same seed and protocol:

```powershell
python -B tools/run_active_learning.py --method random --detector retinanet --dataset voc --gpus 1
```

This is the highest-value falsification check for the current method results.
The initial round should match the other seed-0 runs exactly; rounds 2 through
7 provide the comparison.

If EUA-only remains meaningfully stronger than Random, the next trained method
should be ECA-only:

```powershell
python -B tools/run_active_learning.py --method ecpal:eca-only --detector retinanet --dataset voc --gpus 1
```

That comparison isolates whether predictive error uncertainty (EUA) is more
useful than predicted error amount (ECA) under the current protocol. ECA-full
can follow if a complete current-protocol two-by-two ECA/EUA and
only/diversity ablation is needed.

Separately, the natural-diversity claim should be tested with an offline audit
of the existing EUA selections in the cached ViT embedding space. This does
not require another detector-training run and should compare EUA-only with
Random, PAL, and the frozen-state EUA-full reranking using dispersion,
redundancy, and pool-coverage metrics.

## Compatibility and Validation

- The recommendation uses only current-protocol runs for direct comparison.
- Historical batch-1 or shared-initial-pool results are not mixed into the
  decision table.
- Confirmed that `random` and `ecpal:eca-only` are registered method aliases in
  `configs/catalog/methods.py`.

No experiment was launched and no runtime code or configuration was changed.
