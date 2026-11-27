$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonExe = "C:\Users\user\anaconda3\envs\hoyoung\python.exe"
$CurrentGpuPid = 22836
$BaseCheckpoint = "outputs\unified\ablations\shared_base_icarl_c100_b50_inc5_r32\seed_1\checkpoints\session_00.pt"
$QueueLog = Join-Path $ProjectRoot "outputs\runner_logs\codex_explore_gpu1_queue.log"

function Write-QueueLog([string]$Message) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $QueueLog -Value "[$stamp] $Message"
}

function Run-Screen(
    [string]$Config,
    [string]$RunName,
    [string]$OutputGroup
) {
    $metrics = Join-Path $ProjectRoot "outputs\explore\$OutputGroup\$RunName\seed_1\metrics.json"
    if (Test-Path -LiteralPath $metrics) {
        Write-QueueLog "skip completed $RunName"
        return
    }

    $stdout = Join-Path $ProjectRoot "outputs\runner_logs\$RunName.stdout.log"
    $stderr = Join-Path $ProjectRoot "outputs\runner_logs\$RunName.stderr.log"
    $arguments = @(
        "scripts\train_table1_explore.py",
        $Config,
        "--device", "cuda:1",
        "--seed", "1",
        "--max-sessions", "3",
        "--run-name", $RunName,
        "--base-checkpoint", $BaseCheckpoint
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
}

Write-QueueLog "queue initialized; waiting for current GPU1 PID $CurrentGpuPid"
while (Get-Process -Id $CurrentGpuPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 15
}

Run-Screen `
    "configs\explore\cifar100\icarl_edgecorr_r20_lambda05.yaml" `
    "screen_s2_icarl_edgecorr_global_r20_l05_u10" `
    "edge_correlation"

Write-QueueLog "GPU1 queue finished"
