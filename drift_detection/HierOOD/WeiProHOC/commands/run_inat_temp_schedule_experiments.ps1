param(
    [string]$Basedir = "ckpts/inat19",
    [string]$IdSplit = "data/inat19-id-labels.csv",
    [string]$Hierarchy = "hierarchies/inat19.json",
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cuda",
    [ValidateSet("scheduled_raw", "scheduled_norm")]
    [string]$Method = "scheduled_raw",
    [ValidateSet("default", "t0_1p0", "t0_2p0", "strong")]
    [string]$Mode = "default"
)

$outputDir = ""
$extraArgs = @()

switch ($Mode) {
    "default" {
        $outputDir = "results/schedule_experiments/inat19/scheduled_raw_full"
        $extraArgs += @("--exp_beta_gamma", "0.5", "--temp_t0", "1.5", "--temp_linear_k", "0.5", "--temp_exp_r", "1.25")
    }
    "t0_1p0" {
        $outputDir = "results/schedule_experiments/inat19/scheduled_raw_full_T0_1p0"
        $extraArgs += @("--exp_beta_gamma", "0.5", "--temp_t0", "1.0", "--temp_linear_k", "0.5", "--temp_exp_r", "1.25")
    }
    "t0_2p0" {
        $outputDir = "results/schedule_experiments/inat19/scheduled_raw_full_T0_2p0"
        $extraArgs += @("--exp_beta_gamma", "0.5", "--temp_t0", "2.0", "--temp_linear_k", "0.5", "--temp_exp_r", "1.25")
    }
    "strong" {
        $outputDir = "results/schedule_experiments/inat19/scheduled_raw_full_temp_strong"
        $extraArgs += @("--exp_beta_gamma", "0.5", "--temp_t0", "1.5", "--temp_linear_k", "1.5", "--temp_exp_r", "2.0")
    }
}

$cmd = @(
    "python", "schedule_experiment_runner.py",
    "--basedir", $Basedir,
    "--id_split", $IdSplit,
    "--hierarchy", $Hierarchy,
    "--device", $Device,
    "--method", $Method,
    "--output_dir", $outputDir
) + $extraArgs

Write-Host "Running:" ($cmd -join " ")
& $cmd[0] $cmd[1..($cmd.Length - 1)]

