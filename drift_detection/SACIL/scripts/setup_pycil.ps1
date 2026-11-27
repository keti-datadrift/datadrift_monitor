param(
    [string]$Target = "ref_codes\00_frameworks\PyCIL"
)

$ErrorActionPreference = "Stop"
$Commit = "f3509b8ca3f20660ce4aa13f19d5283de81b4b35"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$TargetPath = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $Target))

if (-not $TargetPath.StartsWith(
    $ProjectRoot,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "PyCIL target must stay inside the SACIL workspace: $TargetPath"
}

if (Test-Path -LiteralPath $TargetPath) {
    if (-not (Test-Path -LiteralPath (Join-Path $TargetPath ".git"))) {
        throw "Target exists but is not a Git checkout: $TargetPath"
    }
    $Dirty = git -C $TargetPath status --porcelain
    if ($Dirty) {
        throw "Refusing to update a dirty PyCIL checkout: $TargetPath"
    }
} else {
    git clone https://github.com/LAMDA-CL/PyCIL.git $TargetPath
}

git -C $TargetPath fetch --depth 1 origin $Commit
git -C $TargetPath checkout --detach $Commit
Write-Host "PyCIL pinned at $Commit in $TargetPath"
