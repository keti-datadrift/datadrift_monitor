python $PROHOC/gather_hinference.py `
  --basedir $TRAINDIR/$DSET/ `
  --uncertainty_methods compprob entcompprob normentropy_compprob `
  --device cpu `
  --output_path $PROHOC/results/hinference-$DSET.result `
  --id_split $PROHOC/data/$DSET-id-labels.csv `
  --hierarchy $PROHOC/hierarchies/$DSET.json
