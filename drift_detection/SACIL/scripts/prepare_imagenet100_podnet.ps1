param(
    [string]$ChallengeRoot = 'D:\HY_Data\HierOOD\FullDataset\imagenet-object-localization-challenge',
    [string]$DestinationRoot = (Join-Path $PSScriptRoot '..\datasets\ImageNet100'),
    [int]$RobocopyThreads = 16
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Write-Status {
    param([string]$Message)
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Write-Output "[$timestamp] $Message"
    [Console]::Out.Flush()
}

function Read-PodNetSplit {
    param(
        [string]$Path,
        [ValidateSet('train', 'val')]
        [string]$Split
    )

    $lineNumber = 0
    $rows = foreach ($line in [System.IO.File]::ReadLines($Path)) {
        $lineNumber++
        if ($line -notmatch '^(?<relative>\S+)\s+(?<label>\d+)$') {
            throw "Invalid metadata line at ${Path}:${lineNumber}: $line"
        }

        $relativePath = $Matches.relative
        $label = [int]$Matches.label
        $parts = $relativePath -split '/'
        if ($parts.Count -ne 3 -or $parts[0] -ne $Split -or $parts[1] -notmatch '^n\d{8}$') {
            throw "Unexpected $Split path at ${Path}:${lineNumber}: $relativePath"
        }

        [pscustomobject]@{
            RelativePath = $relativePath
            Split        = $parts[0]
            Wnid         = $parts[1]
            FileName     = $parts[2]
            Label        = $label
        }
    }

    return @($rows)
}

function Get-FileMap {
    param([string]$Directory)

    $map = [System.Collections.Generic.Dictionary[string, System.IO.FileInfo]]::new(
        [System.StringComparer]::OrdinalIgnoreCase
    )
    foreach ($file in Get-ChildItem -LiteralPath $Directory -File -Filter '*.JPEG') {
        $map.Add($file.Name, $file)
    }
    return $map
}

$challengeRootResolved = (Resolve-Path -LiteralPath $ChallengeRoot).Path
$destinationRootFull = [System.IO.Path]::GetFullPath($DestinationRoot)
$sourceDataRoot = Join-Path $challengeRootResolved 'ILSVRC\Data\CLS-LOC'
$sourceTrainRoot = Join-Path $sourceDataRoot 'train'
$sourceValRoot = Join-Path $sourceDataRoot 'val'
$valSolutionPath = Join-Path $challengeRootResolved 'LOC_val_solution.csv'
$synsetMappingPath = Join-Path $challengeRootResolved 'LOC_synset_mapping.txt'
$metadataRoot = Join-Path $destinationRootFull 'metadata\podnet_original'
$trainMetadataPath = Join-Path $metadataRoot 'train_100.txt'
$valMetadataPath = Join-Path $metadataRoot 'val_100.txt'
$destinationTrainRoot = Join-Path $destinationRootFull 'train'
$destinationValRoot = Join-Path $destinationRootFull 'val'

foreach ($requiredPath in @(
    $sourceTrainRoot,
    $sourceValRoot,
    $valSolutionPath,
    $synsetMappingPath,
    $trainMetadataPath,
    $valMetadataPath
)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required input does not exist: $requiredPath"
    }
}

New-Item -ItemType Directory -Force -Path $destinationTrainRoot, $destinationValRoot | Out-Null

Write-Status 'Reading the official PODNet ImageNet-100 split.'
$trainRows = Read-PodNetSplit -Path $trainMetadataPath -Split 'train'
$valRows = Read-PodNetSplit -Path $valMetadataPath -Split 'val'

$trainClassGroups = @($trainRows | Group-Object Wnid | Sort-Object Name)
$valClassGroups = @($valRows | Group-Object Wnid | Sort-Object Name)
$labelByWnid = @{}
foreach ($group in $trainClassGroups) {
    $labels = @($group.Group.Label | Sort-Object -Unique)
    if ($labels.Count -ne 1) {
        throw "WNID $($group.Name) maps to multiple labels in train metadata."
    }
    $labelByWnid[$group.Name] = $labels[0]
}

$labels = @($labelByWnid.Values | Sort-Object -Unique)
if ($trainClassGroups.Count -ne 100 -or $labels.Count -ne 100 -or $labels[0] -ne 0 -or $labels[-1] -ne 99) {
    throw 'The train split must contain exactly 100 classes with labels 0 through 99.'
}
if ($valClassGroups.Count -ne 100) {
    throw 'The validation split must contain exactly 100 classes.'
}
foreach ($group in $valClassGroups) {
    if (-not $labelByWnid.ContainsKey($group.Name)) {
        throw "Validation WNID is absent from train metadata: $($group.Name)"
    }
    $valLabels = @($group.Group.Label | Sort-Object -Unique)
    if ($valLabels.Count -ne 1 -or $valLabels[0] -ne $labelByWnid[$group.Name]) {
        throw "Train/validation label mismatch for WNID $($group.Name)."
    }
}

Write-Status "Split parsed: $($trainRows.Count) train images, $($valRows.Count) validation images, 100 classes."
Write-Status 'Preflighting all selected train files and calculating their size.'

$trainBytes = [int64]0
$trainSourceMaps = @{}
$classNumber = 0
foreach ($group in $trainClassGroups) {
    $classNumber++
    $sourceClassDirectory = Join-Path $sourceTrainRoot $group.Name
    if (-not (Test-Path -LiteralPath $sourceClassDirectory -PathType Container)) {
        throw "Source train class directory is missing: $sourceClassDirectory"
    }

    $actualFiles = Get-FileMap -Directory $sourceClassDirectory
    $expectedNames = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::OrdinalIgnoreCase
    )
    foreach ($row in $group.Group) {
        [void]$expectedNames.Add($row.FileName)
    }

    if ($actualFiles.Count -ne $expectedNames.Count) {
        throw "Train file count mismatch for $($group.Name): source=$($actualFiles.Count), metadata=$($expectedNames.Count)."
    }
    foreach ($expectedName in $expectedNames) {
        if (-not $actualFiles.ContainsKey($expectedName)) {
            throw "Selected train image is missing: $(Join-Path $sourceClassDirectory $expectedName)"
        }
        $trainBytes += $actualFiles[$expectedName].Length
    }
    $trainSourceMaps[$group.Name] = $actualFiles

    if (($classNumber % 20) -eq 0) {
        Write-Status "Train preflight: $classNumber/100 classes checked."
    }
}

Write-Status 'Cross-checking selected validation images against LOC_val_solution.csv.'
$selectedValByImageId = @{}
foreach ($row in $valRows) {
    $imageId = [System.IO.Path]::GetFileNameWithoutExtension($row.FileName)
    $selectedValByImageId[$imageId] = $row
}

$validatedValIds = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)
foreach ($solutionRow in Import-Csv -LiteralPath $valSolutionPath) {
    if (-not $selectedValByImageId.ContainsKey($solutionRow.ImageId)) {
        continue
    }
    $groundTruthWnid = ($solutionRow.PredictionString -split '\s+')[0]
    $metadataRow = $selectedValByImageId[$solutionRow.ImageId]
    if ($groundTruthWnid -ne $metadataRow.Wnid) {
        throw "Validation ground-truth mismatch for $($solutionRow.ImageId): metadata=$($metadataRow.Wnid), CSV=$groundTruthWnid."
    }
    [void]$validatedValIds.Add($solutionRow.ImageId)
}
if ($validatedValIds.Count -ne $valRows.Count) {
    throw "Only $($validatedValIds.Count) of $($valRows.Count) selected validation images were verified."
}

$valBytes = [int64]0
foreach ($row in $valRows) {
    $sourceFile = Join-Path $sourceValRoot $row.FileName
    if (-not (Test-Path -LiteralPath $sourceFile -PathType Leaf)) {
        throw "Selected validation image is missing: $sourceFile"
    }
    $valBytes += (Get-Item -LiteralPath $sourceFile).Length
}

$totalBytes = $trainBytes + $valBytes
$totalGiB = [Math]::Round($totalBytes / 1GB, 2)
$driveRoot = [System.IO.Path]::GetPathRoot($destinationRootFull)
$freeBytes = ([System.IO.DriveInfo]::new($driveRoot)).AvailableFreeSpace
if ($freeBytes -lt ($totalBytes + 1GB)) {
    throw "Insufficient destination space. Required data size is approximately $totalGiB GiB."
}
Write-Status "Preflight passed. Selected image data size: $totalGiB GiB."

Write-Status 'Copying train classes. Existing matching files are reused by robocopy.'
$classNumber = 0
foreach ($group in $trainClassGroups) {
    $classNumber++
    $sourceClassDirectory = Join-Path $sourceTrainRoot $group.Name
    $destinationClassDirectory = Join-Path $destinationTrainRoot $group.Name
    New-Item -ItemType Directory -Force -Path $destinationClassDirectory | Out-Null

    & robocopy.exe `
        $sourceClassDirectory `
        $destinationClassDirectory `
        '*.JPEG' `
        /E `
        /COPY:DAT `
        /DCOPY:DAT `
        /R:2 `
        /W:1 `
        "/MT:$RobocopyThreads" `
        /NFL `
        /NDL `
        /NJH `
        /NJS `
        /NP | Out-Null
    $robocopyExitCode = $LASTEXITCODE
    if ($robocopyExitCode -gt 7) {
        throw "robocopy failed for $($group.Name) with exit code $robocopyExitCode."
    }

    if (($classNumber % 5) -eq 0 -or $classNumber -eq 100) {
        Write-Status "Train copy: $classNumber/100 classes completed."
    }
}

Write-Status 'Copying and organizing validation images by WNID.'
$valNumber = 0
foreach ($row in $valRows) {
    $valNumber++
    $sourceFile = Join-Path $sourceValRoot $row.FileName
    $destinationClassDirectory = Join-Path $destinationValRoot $row.Wnid
    $destinationFile = Join-Path $destinationClassDirectory $row.FileName
    if (-not (Test-Path -LiteralPath $destinationClassDirectory)) {
        New-Item -ItemType Directory -Force -Path $destinationClassDirectory | Out-Null
    }

    $copyRequired = $true
    if (Test-Path -LiteralPath $destinationFile -PathType Leaf) {
        $sourceLength = (Get-Item -LiteralPath $sourceFile).Length
        $destinationLength = (Get-Item -LiteralPath $destinationFile).Length
        $copyRequired = $sourceLength -ne $destinationLength
    }
    if ($copyRequired) {
        [System.IO.File]::Copy($sourceFile, $destinationFile, $true)
        (Get-Item -LiteralPath $destinationFile).LastWriteTimeUtc = (Get-Item -LiteralPath $sourceFile).LastWriteTimeUtc
    }

    if (($valNumber % 500) -eq 0) {
        Write-Status "Validation copy: $valNumber/$($valRows.Count) images completed."
    }
}

Copy-Item -LiteralPath $trainMetadataPath -Destination (Join-Path $destinationRootFull 'train_100.txt') -Force
Copy-Item -LiteralPath $valMetadataPath -Destination (Join-Path $destinationRootFull 'val_100.txt') -Force

$synsetNames = @{}
foreach ($line in [System.IO.File]::ReadLines($synsetMappingPath)) {
    if ($line -match '^(?<wnid>n\d{8})\s+(?<name>.+)$') {
        $synsetNames[$Matches.wnid] = $Matches.name
    }
}

$classIndex = foreach ($group in $trainClassGroups) {
    $wnid = $group.Name
    [pscustomobject]@{
        label       = $labelByWnid[$wnid]
        wnid        = $wnid
        class_name  = $synsetNames[$wnid]
        train_count = $group.Count
        val_count   = ($valClassGroups | Where-Object Name -eq $wnid).Count
    }
}
$classIndex = @($classIndex | Sort-Object label)
$classIndexPath = Join-Path $destinationRootFull 'metadata\class_index.csv'
$classIndex | Export-Csv -LiteralPath $classIndexPath -NoTypeInformation -Encoding utf8

$cilClassOrder = @(
    68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38,
    58, 70, 77, 1, 85, 19, 17, 50, 28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44,
    91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96, 98, 97, 2, 64, 66,
    42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46,
    62, 69, 36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33
)
$orderDocument = [ordered]@{
    description  = 'ImageNet-100 seed-1 class order used by the local AFC and R-DFCIL reference code'
    source_label = 'PODNet label in class_index.csv'
    seed         = 1
    class_order  = $cilClassOrder
}
$orderDocument |
    ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $destinationRootFull 'metadata\class_order_afc_rdfcil_seed1.json') -Encoding utf8

Write-Status 'Validating copied dataset.'
$destinationTrainFiles = @(Get-ChildItem -LiteralPath $destinationTrainRoot -File -Recurse -Filter '*.JPEG')
$destinationValFiles = @(Get-ChildItem -LiteralPath $destinationValRoot -File -Recurse -Filter '*.JPEG')
$destinationTrainClasses = @(Get-ChildItem -LiteralPath $destinationTrainRoot -Directory)
$destinationValClasses = @(Get-ChildItem -LiteralPath $destinationValRoot -Directory)

if (
    $destinationTrainFiles.Count -ne $trainRows.Count -or
    $destinationValFiles.Count -ne $valRows.Count -or
    $destinationTrainClasses.Count -ne 100 -or
    $destinationValClasses.Count -ne 100
) {
    throw "Destination validation failed: train=$($destinationTrainFiles.Count), val=$($destinationValFiles.Count), train_classes=$($destinationTrainClasses.Count), val_classes=$($destinationValClasses.Count)."
}

$trainMetadataHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $trainMetadataPath).Hash
$valMetadataHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $valMetadataPath).Hash
$manifest = [ordered]@{
    dataset                      = 'ImageNet-100'
    protocol                     = 'PODNet official ImageNet-100 split'
    prepared_at                  = (Get-Date).ToString('o')
    source_challenge_root        = $challengeRootResolved
    destination_root             = $destinationRootFull
    storage_mode                 = 'physical copy'
    classes                      = 100
    train_images                 = $trainRows.Count
    validation_images            = $valRows.Count
    selected_image_bytes         = $totalBytes
    selected_image_gib           = $totalGiB
    train_metadata_sha256        = $trainMetadataHash
    validation_metadata_sha256   = $valMetadataHash
    validation_ground_truth_csv  = $valSolutionPath
    validation_ground_truth_check = 'passed'
}
$manifest |
    ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $destinationRootFull 'manifest.json') -Encoding utf8

Write-Status "DONE: ImageNet-100 is ready at $destinationRootFull"
Write-Status "Final counts: train=$($trainRows.Count), val=$($valRows.Count), classes=100, size=$totalGiB GiB."
