$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonExe = "C:\Users\user\anaconda3\envs\hoyoung\python.exe"
$CurrentQueuePid = 22688
$BaseCheckpoint = "outputs\unified\ablations\shared_base_icarl_c100_b50_inc5_r32\seed_1\checkpoints\session_00.pt"
$QueueLog = Join-Path $ProjectRoot "outputs\runner_logs\codex_explore_gpu1_kdoff_queue.log"
$Trainer = Join-Path $ProjectRoot "src_explore\sacil\engine\table1_trainer.py"
$ExpectedTrainerSha256 = "47C72CCD38EB2F89CF196A8E2676798558433328834BFE38D921CD0AA99BAF55"

function Write-QueueLog([string]$Message) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $QueueLog -Value "[$stamp] $Message"
}

function Assert-FrozenSource {
    $actual = (Get-FileHash -LiteralPath $Trainer -Algorithm SHA256).Hash
    if ($actual -ne $ExpectedTrainerSha256) {
        throw "src_explore trainer changed: expected=$ExpectedTrainerSha256 actual=$actual"
    }
}

function Run-Screen([string]$Config, [string]$RunName) {
    Assert-FrozenSource
    $metrics = Join-Path $ProjectRoot "outputs\explore\kd_off_tpl\$RunName\seed_1\metrics.json"
    if (Test-Path -LiteralPath $metrics) {
        Write-QueueLog "skip completed $RunName"
        return
    }
    $stdout = Join-Path $ProjectRoot "outputs\runner_logs\$RunName.stdout.log"
    $stderr = Join-Path $ProjectRoot "outputs\runner_logs\$RunName.stderr.log"
    $arguments = @(
        "scripts\train_table1_explore.py", $Config,
        "--device", "cuda:1", "--seed", "1", "--max-sessions", "3",
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
}

Write-QueueLog "waiting for GPU1 follow-up queue PID $CurrentQueuePid"
while (Get-Process -Id $CurrentQueuePid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 15
}

Run-Screen `
    "configs\explore\cifar100\icarl_kdoff_control.yaml" `
    "screen_s2_icarl_kdoff_control_frozen_20260804"

Run-Screen `
    "configs\explore\cifar100\icarl_kdoff_edgecorr_r20_lambda15.yaml" `
    "screen_s2_icarl_kdoff_edgecorr_r20_l15_frozen_20260804"

Write-QueueLog "GPU1 KD-off queue finished"
