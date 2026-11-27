foreach ($HEIGHT in $HEIGHTS) {
    if (-not $env:NPROC_PER_NODE) {
        $env:NPROC_PER_NODE = "2"
    }
    $env:USE_LIBUV = "0"
    $env:MASTER_ADDR = "127.0.0.1"
    $env:MASTER_PORT = "29500"
    $GLOBAL_BATCH_SIZE = 128
    $BATCH_SIZE_PER_GPU = [Math]::Max(1, [int]($GLOBAL_BATCH_SIZE / [int]$env:NPROC_PER_NODE))

    python "$PROHOC\main_multidepth.py" `
        --datadir "$PROHOCDATA\$DATASOURCE" `
        --hierarchy "$PROHOC\hierarchies\$DSET.json" `
        --traindir "$TRAINDIR\$DSET\H$HEIGHT" `
        --id_split "$PROHOC\data\$DSET-id-labels.csv" `
        --height $HEIGHT `
        --epochs 90 `
        --lr 0.05 `
        --batch_size $BATCH_SIZE_PER_GPU `
        --num_workers 8 `
        --multi_gpu `
        --seed 42
}
