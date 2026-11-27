$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\HOYOUNG\datadrift_1\drift_detection\SACIL"
$PythonExe = "C:\Users\user\anaconda3\envs\hoyoung\python.exe"
$RunName = "screen_s2_icarl_feature_cosine_lucir_routed_bgs_20260804"
$StdoutLog = Join-Path $ProjectRoot "outputs\runner_logs\$RunName.stdout.log"
$StderrLog = Join-Path $ProjectRoot "outputs\runner_logs\$RunName.stderr.log"

Set-Location $ProjectRoot

& $PythonExe scripts\train_table1_explore.py `
    configs\explore\cifar100\icarl_feature_cosine_lucir_routed_bgs.yaml `
    --device cuda:1 `
    --seed 1 `
    --max-sessions 3 `
    --base-checkpoint outputs\unified\ablations\shared_base_icarl_c100_b50_inc5_r32\seed_1\checkpoints\session_00.pt `
    --run-name $RunName `
    1> $StdoutLog `
    2> $StderrLog

exit $LASTEXITCODE
