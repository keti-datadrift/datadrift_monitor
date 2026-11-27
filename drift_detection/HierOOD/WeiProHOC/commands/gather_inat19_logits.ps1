param(
    [string]$DataDir,
    [string]$TrainDir = "ckpts/inat19",
    [string]$IdSplit = "data/inat19-id-labels.csv",
    [string]$Hierarchy = "hierarchies/inat19.json",
    [int[]]$Heights = @(0, 1, 2, 3, 4, 5),
    [int]$BatchSize = 128
)

if (-not $DataDir) {
    throw "DataDir is required. Example: -DataDir D:\\HY_Data\\HierOOD\\inat19"
}

foreach ($Height in $Heights) {
    $TrainSubdir = Join-Path $TrainDir "H$Height"
    Write-Host "Gathering logits for H$Height from $TrainSubdir"
    python gather_vallogits_multidepth.py `
        --datadir $DataDir `
        --traindir $TrainSubdir `
        --height $Height `
        --batch_size $BatchSize `
        --id_split $IdSplit `
        --hierarchy $Hierarchy
}
