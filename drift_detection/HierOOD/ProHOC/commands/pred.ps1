python $PROHOC/gather_hinference.py `
  --basedir $TRAINDIR/$DSET/ `
  --uncertainty_methods compprob entcompprob `
  --id_split $PROHOC/data/$DSET-id-labels.csv `
  --hierarchy $PROHOC/hierarchies/$DSET.json