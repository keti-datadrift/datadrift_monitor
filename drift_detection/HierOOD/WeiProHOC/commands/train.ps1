foreach ($HEIGHT in $HEIGHTS) {
    python "$PROHOC\main_multidepth.py" `
        --datadir "$PROHOCDATA\$DSET" `
        --hierarchy "$PROHOC\hierarchies\$DSET.json" `
        --traindir "$TRAINDIR\$DSET\H$HEIGHT" `
        --id_split "$PROHOC\data\$DSET-id-labels.csv" `
        --height $HEIGHT `
        --epochs 90 `
        --lr 0.05 `
        --batch_size 128 `
        --num_workers 8 `
        --multi_gpu `
        --seed 42
}