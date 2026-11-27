$Methods = @(
  "entcompprob",
  "normentropy_compprob",
  "depth_weighted_raw",
  "depth_weighted_norm",
  "fixedbeta_norm"
)

python $PROHOC/gather_hinference.py `
  --basedir $TRAINDIR/$DSET/ `
  --uncertainty_methods $Methods `
  --device cpu `
  --depth_alpha 1.0 1.0 `
  --depth_beta 1.0 1.0 `
  --beta_rule inv_log `
  --result_suffix score-ablation-v1 `
  --output_path $PROHOC/results/hinference-$DSET-score-ablation-v1.result `
  --id_split $PROHOC/data/$DSET-id-labels.csv `
  --hierarchy $PROHOC/hierarchies/$DSET.json
