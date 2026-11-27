param(
    [int[]]$ValidationProcessIds = @(5492, 22288),
    [int]$ExistingBaseProcessId = 0,
    [string]$PythonExe = "C:\Users\user\anaconda3\envs\hoyoung\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Runner = Join-Path $ProjectRoot "scripts\train_pycil.py"
$LogRoot = Join-Path $ProjectRoot "outputs\pycil\runner_logs"
$BaseCheckpoint = Join-Path `
    $ProjectRoot `
    "outputs\pycil\cifar100_shared_base_b50_order1\task_00.pt"

function Write-Status {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    [Console]::Out.WriteLine("$timestamp $Message")
}

function Wait-ForValidation {
    foreach ($processId in $ValidationProcessIds) {
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($null -ne $process) {
            Write-Status "Waiting for validation PID $processId"
            Wait-Process -Id $processId
        }
    }
}

function Read-LastCurve {
    param(
        [string]$Path,
        [string]$CurveName
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing validation log: $Path"
    }
    $match = Select-String `
        -LiteralPath $Path `
        -Pattern "$CurveName`: \[([^\]]+)\]" |
        Select-Object -Last 1
    if ($null -eq $match) {
        throw "Validation did not finish: $Path lacks $CurveName"
    }
    if ($match.Line -notmatch "$CurveName`: \[([^\]]+)\]") {
        throw "Could not parse $CurveName from $Path"
    }
    return @(
        $Matches[1].Split(",") |
        ForEach-Object {
            $value = $_.Trim()
            if ($value -match "^np\.float64\(([-+0-9.eE]+)\)$") {
                [double]$Matches[1]
            }
            else {
                [double]$value
            }
        }
    )
}

function Assert-CurvesMatch {
    param(
        [double[]]$Official,
        [double[]]$Control,
        [string]$Name
    )
    if ($Official.Count -ne 2 -or $Control.Count -ne 2) {
        throw "$Name validation must contain exactly two tasks"
    }
    for ($index = 0; $index -lt 2; $index++) {
        if ([math]::Abs($Official[$index] - $Control[$index]) -gt 0.01) {
            throw (
                "$Name mismatch at task $index`: official=" +
                "$($Official[$index]), control=$($Control[$index])"
            )
        }
    }
}

function Start-Experiment {
    param(
        [string]$Label,
        [string]$Config,
        [string]$Gpu
    )
    $stdout = Join-Path $LogRoot "$Label.stdout.log"
    $stderr = Join-Path $LogRoot "$Label.stderr.log"
    $env:CUDA_VISIBLE_DEVICES = $Gpu
    Write-Status "Starting $Label on physical GPU $Gpu"
    return Start-Process `
        -FilePath $PythonExe `
        -ArgumentList @($Runner, "--config", $Config) `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru
}

function Wait-ForExperiments {
    param(
        [System.Diagnostics.Process[]]$Processes,
        [string]$Stage
    )
    foreach ($process in $Processes) {
        $process.WaitForExit()
        $process.Refresh()
        $exitCode = $process.ExitCode
        if ($null -ne $exitCode -and $exitCode -ne 0) {
            throw "$Stage failed: PID $($process.Id), exit $($process.ExitCode)"
        }
    }
    Write-Status "Completed $Stage"
}

function Test-ExperimentComplete {
    param(
        [string]$Label,
        [int]$ExpectedTasks = 11
    )
    $path = Join-Path $LogRoot "$Label.stdout.log"
    try {
        $curve = @(Read-LastCurve $path "CNN top1 curve")
        return $curve.Count -eq $ExpectedTasks
    }
    catch {
        return $false
    }
}

function Assert-ExperimentComplete {
    param(
        [string]$Label,
        [int]$ExpectedTasks = 11
    )
    if (-not (Test-ExperimentComplete $Label $ExpectedTasks)) {
        throw "$Label lacks a completed $ExpectedTasks-task CNN curve"
    }
}

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
Wait-ForValidation

$officialLog = Join-Path `
    $LogRoot "validation_icarl_official.stdout.log"
$controlLog = Join-Path `
    $LogRoot "validation_icarl_control.stdout.log"
$officialCnn = Read-LastCurve $officialLog "CNN top1 curve"
$controlCnn = Read-LastCurve $controlLog "CNN top1 curve"
$officialNme = Read-LastCurve $officialLog "NME top1 curve"
$controlNme = Read-LastCurve $controlLog "NME top1 curve"
Assert-CurvesMatch $officialCnn $controlCnn "CNN"
Assert-CurvesMatch $officialNme $controlNme "NME"
Write-Status (
    "Validated official/control parity: CNN [$officialCnn], " +
    "NME [$officialNme]"
)

if (Test-Path -LiteralPath $BaseCheckpoint) {
    Write-Status "Reusing completed shared-base checkpoint"
}
else {
    $base = $null
    if ($ExistingBaseProcessId -gt 0) {
        $base = Get-Process `
            -Id $ExistingBaseProcessId `
            -ErrorAction SilentlyContinue
    }
    if ($null -ne $base) {
        Write-Status "Adopting running shared-base PID $ExistingBaseProcessId"
    }
    else {
        $base = Start-Experiment `
            "core_shared_base" `
            "configs/pycil/cifar100/shared_base_b50.json" `
            "0"
    }
    Wait-ForExperiments @($base) "shared base"
    if (-not (Test-Path -LiteralPath $BaseCheckpoint)) {
        throw "Shared base completed without checkpoint: $BaseCheckpoint"
    }
}

$stageProcesses = @()
if (-not (Test-ExperimentComplete "core_icarl_control")) {
    $stageProcesses += Start-Experiment `
        "core_icarl_control" `
        "configs/pycil/cifar100/icarl_control_b50_inc5.json" `
        "0"
}
if (-not (Test-ExperimentComplete "core_global_hap")) {
    $stageProcesses += Start-Experiment `
        "core_global_hap" `
        "configs/pycil/cifar100/global_hap_b50_inc5.json" `
        "1"
}
if ($stageProcesses.Count -gt 0) {
    Wait-ForExperiments $stageProcesses "iCaRL control + Global-HAP"
}
else {
    Write-Status "Reusing completed iCaRL control + Global-HAP"
}
Assert-ExperimentComplete "core_icarl_control"
Assert-ExperimentComplete "core_global_hap"

$stageProcesses = @()
if (-not (Test-ExperimentComplete "core_flat_lrhap")) {
    $stageProcesses += Start-Experiment `
        "core_flat_lrhap" `
        "configs/pycil/cifar100/flat_lrhap_b50_inc5.json" `
        "0"
}
if (-not (Test-ExperimentComplete "core_sacil")) {
    $stageProcesses += Start-Experiment `
        "core_sacil" `
        "configs/pycil/cifar100/sacil_b50_inc5.json" `
        "1"
}
if ($stageProcesses.Count -gt 0) {
    Wait-ForExperiments $stageProcesses "Flat-LRHAP + SACIL"
}
else {
    Write-Status "Reusing completed Flat-LRHAP + SACIL"
}
Assert-ExperimentComplete "core_flat_lrhap"
Assert-ExperimentComplete "core_sacil"

$stageProcesses = @()
if (-not (Test-ExperimentComplete "core_replay_control")) {
    $stageProcesses += Start-Experiment `
        "core_replay_control" `
        "configs/pycil/cifar100/replay_control_b50_inc5.json" `
        "0"
}
if (-not (Test-ExperimentComplete "core_replay_sacil")) {
    $stageProcesses += Start-Experiment `
        "core_replay_sacil" `
        "configs/pycil/cifar100/replay_sacil_b50_inc5.json" `
        "1"
}
if ($stageProcesses.Count -gt 0) {
    Wait-ForExperiments $stageProcesses `
        "Replay-CE control + Replay-CE SACIL"
}
else {
    Write-Status "Reusing completed Replay-CE pair"
}
Assert-ExperimentComplete "core_replay_control"
Assert-ExperimentComplete "core_replay_sacil"

Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
Write-Status "All queued core experiments completed"
