foreach ($HEIGHT in $HEIGHTS) {
    python "$PROHOC\gather_vallogits_multidepth.py" `
        --datadir "$PROHOCDATA\$DSET" `
        --traindir "$TRAINDIR\$DSET\H$HEIGHT" `
        --height $HEIGHT `
        --id_split "$PROHOC\data\$DSET-id-labels.csv" `
        --hierarchy "$PROHOC\hierarchies\$DSET.json"
}