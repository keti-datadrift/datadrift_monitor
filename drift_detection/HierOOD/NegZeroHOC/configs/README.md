# Experiment Config Layout

The configuration tree follows the order in which the NegZeroHOC experiments
were developed.

```text
configs/
  00_baselines/             Frozen CLIP child-only and manual-unknown baselines
  01_feature_probes/        Linear/MLP probes on cached CLIP image features
  02_idea3_cached_prompt/   Idea 3 prompt learning on cached features
  03_sparse_path/           Sparse-path loss and Global Path MAP experiments
  04_joint_vision_lora/     Raw-image true Vision LoRA + prompt joint training
  05_idea4_negative_prompt/ Frozen positive/LoRA + parent-local unknown prompts
  06_idea5_hier_negprompt/  Hierarchical NegPrompt paper/stop ablations
  07_idea6_hc_negprompt/    Hierarchy-constrained negative prompts
  08_idea7_virtual_open_negprompt/ Sibling virtual open-space experiments
  09_image_metric_joint_prompts/ Image-metric and staged prompt experiments
  10_negative_text_metric_terminal/ Negative text/global terminal ablations
  11_paper_negprompt/       Published NegPrompt vs repulsive-NPD control
  12_tree_virtual_unknown/  Augmented-tree virtual-sibling unknown prompts
  13_tree_loco_multiprompt/ LOCO-trained multi-prototype virtual siblings
  14_hierarchical_support/  Negative-prompt-free image-support factorization
  15_multidepth_feature_heads/ Frozen-feature ProHOC probability fusion
  16_probabilistic_support/ Threshold-free shared masked-support likelihood
  17_probabilistic_mixture/ Coherent shared K+1 prototype-mixture posterior
  18_crossfit_class_holdout/ Class-disjoint hierarchical pseudo-OOD encoders
  18_reference_sample_mixture/ Non-parametric reference-image vMF mixtures
  19_cf_rpep/                Route-preserving class-held-out OOF posterior
  22_cf_fshp/                Factorized selective hierarchical OOF posterior
  20_hierarchical_hazard/    Normalized top-down unknown stopping posterior
  21_relational_hazard/      Shared node-agnostic relational hazard model
  22_class_loco_relational/   Retained-class episodic LOCO meta-head
  references/               External/reference experiment configs
```

The main current FGVC-Aircraft experiment is:

```powershell
python scripts\train_idea3_joint_vision_lora.py --config configs\04_joint_vision_lora\idea3_fgvc_aircraft_b16_joint_vision_lora.yaml
python scripts\infer_idea3_joint_vision_lora.py --config configs\04_joint_vision_lora\idea3_fgvc_aircraft_b16_joint_vision_lora.yaml
```

`base_config` paths are resolved relative to the config file containing them.

Idea V CF-FSHP consumes the actual-OOD-free OOF bundles persisted by the
Idea O/P CF-RPEP checkpoint. It does not re-encode images or expose an
official-OOD loader. Its six shared scalars, calibration-only feature
normalization, fold-pruned augmented-tree Brier loss, and known-first MAP
tie-break are locked by `22_cf_fshp/fgvc_aircraft_oof_screen_gpu0.yaml`.
Because the method was designed after inspecting the current outer manifest,
this run is a method-development screen and cannot unlock official OOD even
when every numerical gate passes.

The `lora` configs under `03_sparse_path/` are feature-space low-rank adapters.
Only `04_joint_vision_lora/` modifies the actual CLIP vision transformer.

The joint trainer validates every configured interval and runs final inference
automatically. The separate inference command regenerates the metrics from the
compact positive-prompt and Vision-LoRA checkpoint without retraining.

The image-metric Vision-LoRA trainer writes an atomic resumable
`last_checkpoint` after every completed epoch. Automatic resume remains opt-in:

```yaml
image_metric_training:
  resume:
    enabled: true
    checkpoint: null  # defaults to validation.last_checkpoint
```

Exact epoch-boundary continuation requires `dataloader.num_workers: 0`. A
checkpoint marked complete skips training and reruns finalization from its
stored validation-best state. The primary last checkpoint retains one
`-previous` generation for corruption fallback.

Idea 4 starts from the strongest Idea 3 positive Vision-LoRA checkpoint. It
freezes the CLIP base, Vision LoRA, and positive prompts, then trains only a
non-root parent-local unknown prompt with leave-child-out supervision. Its
primary inference compares ID leaves and local unknown terminal paths with
exact global-path MAP; local greedy inference is saved as an ablation.

`idea3_fgvc_aircraft_b16_joint_vision_lora_global_depth.yaml` is the GPU 1
parent-context ablation. It inherits the main joint experiment and changes only
the prompt ablation, GPU, experiment name, and output paths.

Experiment outputs use this layout:

```text
outputs/
  shared/features/<dataset>/<clip-model>/
  experiments/<experiment-name>/
    checkpoints/
    results/
    diagnostics/
```

CLIP feature caches are shared because several experiments reuse the same
frozen features. All trainable checkpoints and evaluation artifacts are grouped
under the experiment that produced them.
