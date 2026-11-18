# RetinaNet + PASCAL VOC Settings for PAL/PPAL

## Scope

This note narrows the implementation target to the lightest shared setting for PPAL and PAL:

- Detector: RetinaNet with ResNet-50 FPN.
- Dataset: PASCAL VOC, 20 classes.
- Training data: VOC2007 trainval + VOC2012 trainval, represented by `data/VOC0712/annotations/trainval_0712.json`.
- Evaluation data: VOC2007 test, configured as `data/VOCdevkit/VOC2007/ImageSets/Main/test.txt`.
- Reference protocol source: `code_refs/PPAL/al_configs/voc/ppal_retinanet_voc.py` and `code_refs/PPAL/configs/voc_active_learning/`.

PAL states that its COCO and PASCAL VOC RetinaNet experiments follow the PPAL training and dataset protocol. Therefore, the first faithful PAL implementation should keep PPAL's VOC RetinaNet protocol unchanged and replace only the acquisition method.

## PPAL VOC Active Learning Protocol

From `code_refs/PPAL/al_configs/voc/ppal_retinanet_voc.py`:

- Initial labelled set: `data/active_learning/voc/voc_827_labeled_1.json`.
- Initial unlabelled set: `data/active_learning/voc/voc_827_unlabeled_1.json`.
- Initial labelled image count implied by filename: 827 images, corresponding to 5 percent of VOC0712 trainval.
- Round count: `round_num = 7`.
- Per-acquisition budget: `budget = 414`, corresponding to about 2.5 percent of VOC0712 trainval.
- Expanded uncertainty candidate pool: `budget_expand_ratio = 4`.
- Uncertainty pool size: `budget * 4`, padded to be divisible by `gpus`.
- GPU count in reference config: `gpus = 8`.
- Output directory: `work_dirs/retinanet_voc_ppal_7rounds_5percent_to_20percent`.

From `code_refs/PPAL/tools/run_al_voc.py`, `round_num=7` means rounds 1 through 7 are trained/evaluated, while sampling is skipped in the final round. Therefore the active acquisition count is 6:

- Round 1: train/evaluate initial 827-image labelled set.
- Rounds 1-6: after evaluation, acquire 414 images for the next round.
- Round 7: train/evaluate the final labelled set and stop.
- Final labelled size: `827 + 6 * 414 = 3311` images, matching the output name's 5 percent to 20 percent protocol.

## RetinaNet Training Config

From `code_refs/PPAL/configs/voc_active_learning/al_train/retinanet_26e.py` and base configs:

- Model: RetinaNet, ResNet-50 backbone, FPN neck.
- Backbone init: `data/pretrain_models/resnet50-19c8e357.pth`.
- Number of classes: 20.
- Training head for PPAL: `RetinaQualityEMAHead` with `base_momentum = 0.99`.
- Epochs: 26.
- Optimizer: SGD.
- Learning rate: `0.002`.
- Momentum: `0.9`.
- Weight decay: `0.0001`.
- LR schedule: step policy, warmup 500 iterations, step at epoch 20.
- Batch config: `samples_per_gpu = 1`, `workers_per_gpu = 2`.
- Training image scale: `(1000, 600)`, keep ratio.
- Evaluation metric: VOC mAP via `--eval mAP`.

For PAL, the first implementation should keep this training config and schedule unchanged. PAL's paper emphasizes that PAL uses inference outputs and does not require detector training-pipeline changes.

## PPAL Inference Settings

PPAL uncertainty inference config, `retinanet_uncertainty.py`:

- Detector wrapper: `ALRetinaNet`.
- Head: `RetinaHeadUncertainty`.
- `nms_pre = 3000`.
- `score_thr = 0.01`.
- NMS IoU threshold: `0.5`.
- `max_per_img = 200`.

PPAL diversity inference config, `retinanet_diversity.py`:

- Head: `RetinaHeadFeat`.
- `max_det = 100`.
- `feat_dim = 256`.
- Produces image distance data for the diversity sampler.

PPAL sampler config:

- Uncertainty sampler: `DCUSSampler`.
- Uncertainty sampler score threshold: `score_thr = 0.05`.
- Difficulty/class calibration: `class_weight_ub = 0.2`, `class_weight_alpha = 0.3`.
- Diversity sampler: `DiversitySampler`.

These PPAL-specific settings should be preserved for a PPAL baseline, but PAL should not depend on `RetinaQualityEMAHead`, `RetinaHeadUncertainty`, or `RetinaHeadFeat` except where local code reuse is useful.

## PAL Settings to Use in This Protocol

From the PAL paper:

- PAL runs RetinaNet on PASCAL VOC using PPAL's dataset and training protocol.
- Full-dataset detections for LIUS use RetinaNet inference hyperparameters:
  - pre-NMS boxes: 1000.
  - NMS IoU threshold: 0.5.
  - score threshold: 0.3.
- Selection weights:
  - `alpha = 0.9`.
  - `beta = 0.04`.
  - `gamma = 0.02`.
  - `d = 0.1`, with `2 * beta + gamma = d`.
- PAL uses class-specific logistic classifiers for LIUS.
- PAL uses a vision transformer image encoder for RCSP. The paper reports Google ViT as the default/best-performing encoder in its ablation.

## Suggested First Implementation Decisions

- Keep PPAL's VOC split files, round schedule, budget, training config, and evaluation protocol exactly as local defaults.
- Implement PAL as a separate acquisition method that consumes PAL-specific inference output.
- Use PAL's inference thresholds for PAL LIUS data collection: `nms_pre=1000`, `score_thr=0.3`, NMS IoU `0.5`.
- Do not reuse PPAL uncertainty inference thresholds for PAL scoring, because PAL explicitly reports separate LIUS inference settings.
- Keep the 7-round PPAL runner semantics: 7 train/eval rounds and 6 acquisition steps.
- Report VOC results as mAP@0.5 to match the PPAL/PAL VOC setting.

## Source Files Read

- `code_refs/PPAL/al_configs/voc/ppal_retinanet_voc.py`
- `code_refs/PPAL/tools/run_al_voc.py`
- `code_refs/PPAL/configs/voc_active_learning/al_train/retinanet_26e.py`
- `code_refs/PPAL/configs/voc_active_learning/al_inference/retinanet_uncertainty.py`
- `code_refs/PPAL/configs/voc_active_learning/al_inference/retinanet_diversity.py`
- `code_refs/PPAL/configs/voc_active_learning/bases/al_retinanet_base.py`
- `code_refs/PPAL/configs/voc_active_learning/bases/al_retinanet_inference_base.py`
- `code_refs/PPAL/configs/voc_active_learning/bases/models/retinanet_r50_fpn.py`
- `code_refs/PAL/[CVPR]-[2026]-[Portable Active Learning for Object Detection].pdf`
