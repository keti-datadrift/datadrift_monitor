$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonExe = "C:\Users\user\anaconda3\envs\hoyoung\python.exe"
$BaseCheckpoint = "outputs\unified\ablations\shared_base_icarl_c100_b50_inc5_r32\seed_1\checkpoints\session_00.pt"
$QueueLog = Join-Path $ProjectRoot "outputs\runner_logs\codex_explore_gpu0_selective.log"
$Trainer = Join-Path $ProjectRoot "src_explore\sacil\engine\table1_trainer.py"
$ExpectedTrainerSha256 = "47C72CCD38EB2F89CF196A8E2676798558433328834BFE38D921CD0AA99BAF55"
$RunName = "screen_s2_icarl_htpl_selective_kd_t0_gpu0_20260804"

function Write-QueueLog([string]$Message) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $QueueLog -Value "[$stamp] $Message"
}

$actual = (Get-FileHash -LiteralPath $Trainer -Algorithm SHA256).Hash
if ($actual -ne $ExpectedTrainerSha256) {
    throw "src_explore trainer changed: expected=$ExpectedTrainerSha256 actual=$actual"
}

$metrics = Join-Path $ProjectRoot "outputs\explore\selective_kd\$RunName\seed_1\metrics.json"
if (Test-Path -LiteralPath $metrics) {
    Write-QueueLog "skip completed $RunName"
    exit 0
}

$stdout = Join-Path $ProjectRoot "outputs\runner_logs\$RunName.stdout.log"
$stderr = Join-Path $ProjectRoot "outputs\runner_logs\$RunName.stderr.log"
$arguments = @(
    "scripts\train_table1_explore.py",
    "configs\explore\cifar100\icarl_htpl_selective_kd_t0.yaml",
    "--device", "cuda:0", "--seed", "1", "--max-sessions", "3",
    "--run-name", $RunName, "--base-checkpoint", $BaseCheckpoint
)

Write-QueueLog "start $RunName"
$process = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList $arguments `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru `
    -Wait
if ($process.ExitCode -ne 0) {
    Write-QueueLog "failed $RunName exit=$($process.ExitCode)"
    throw "$RunName failed with exit code $($process.ExitCode)"
}
Write-QueueLog "complete $RunName"
