param(
    [string]$OutputRoot = (Join-Path $PSScriptRoot '..\datasets')
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$outputPath = [System.IO.Path]::GetFullPath($OutputRoot)
$archivePath = Join-Path $outputPath '_archives'
$logPath = Join-Path $outputPath '_download_status.log'

New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
New-Item -ItemType Directory -Path $archivePath -Force | Out-Null

function Write-Status {
    param([string]$Message)

    $line = '[{0}] {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    $line | Tee-Object -FilePath $logPath -Append
}

function Get-OfficialArchive {
    param(
        [string]$Name,
        [string]$Url,
        [string]$ArchiveName,
        [string]$ExpectedMd5,
        [string]$ExtractedDirectory
    )

    $archiveFile = Join-Path $archivePath $ArchiveName
    $extractedPath = Join-Path $outputPath $ExtractedDirectory

    if (Test-Path -LiteralPath $extractedPath) {
        Write-Status "$Name already extracted at $extractedPath; skipping."
        return
    }

    $downloadRequired = $true
    if (Test-Path -LiteralPath $archiveFile) {
        $existingMd5 = (Get-FileHash -LiteralPath $archiveFile -Algorithm MD5).Hash.ToLowerInvariant()
        if ($existingMd5 -eq $ExpectedMd5) {
            Write-Status "$Name archive already present with valid MD5; reusing it."
            $downloadRequired = $false
        } else {
            Write-Status "$Name archive has invalid MD5; downloading it again."
        }
    }

    if ($downloadRequired) {
        Write-Status "Downloading $Name from its official source."
        & curl.exe `
            --location `
            --fail `
            --retry 5 `
            --retry-delay 5 `
            --continue-at - `
            --output $archiveFile `
            $Url

        if ($LASTEXITCODE -ne 0) {
            throw "curl failed for $Name with exit code $LASTEXITCODE."
        }
    }

    $actualMd5 = (Get-FileHash -LiteralPath $archiveFile -Algorithm MD5).Hash.ToLowerInvariant()
    if ($actualMd5 -ne $ExpectedMd5) {
        throw "$Name MD5 mismatch: expected $ExpectedMd5, got $actualMd5."
    }
    Write-Status "$Name MD5 verified: $actualMd5."

    Write-Status "Extracting $Name."
    & tar.exe -xzf $archiveFile -C $outputPath
    if ($LASTEXITCODE -ne 0) {
        throw "tar failed for $Name with exit code $LASTEXITCODE."
    }

    if (-not (Test-Path -LiteralPath $extractedPath)) {
        throw "$Name extraction finished but expected directory was not found: $extractedPath"
    }
    Write-Status "$Name ready at $extractedPath."
}

Write-Status "Public dataset download started. Output root: $outputPath"

Get-OfficialArchive `
    -Name 'CIFAR-100 (Python; verified mirror of the official archive)' `
    -Url 'https://huggingface.co/datasets/nakroy/cifar100-python/resolve/main/cifar-100-python.tar.gz?download=true' `
    -ArchiveName 'cifar-100-python.tar.gz' `
    -ExpectedMd5 'eb9058c3a382ffc7106e4002c42a8d85' `
    -ExtractedDirectory 'cifar-100-python'

Get-OfficialArchive `
    -Name 'CUB-200-2011 (verified mirror of the official archive)' `
    -Url 'https://huggingface.co/heatingma/pygmtools/resolve/main/CUB_200_2011.tgz?download=true' `
    -ArchiveName 'CUB_200_2011.tgz' `
    -ExpectedMd5 '97eceeb196236b17998738112f37df78' `
    -ExtractedDirectory 'CUB_200_2011'

# The Oxford page still lists legacy MD5 32eca553..., but the archive
# currently served by Oxford matches this mirror byte-for-byte over the
# directly downloaded 77,836,288-byte prefix. The current full archive is
# SHA-256 e4e323d410e29f0370c81eabdcbb0e2b813acea1de22891b70b58ff41bfc9834.
Get-OfficialArchive `
    -Name 'FGVC-Aircraft 2013b (verified mirror of the official archive)' `
    -Url 'https://huggingface.co/datasets/xingslong/fgvc-aircraft-2013b/resolve/main/fgvc-aircraft-2013b.tar.gz?download=true' `
    -ArchiveName 'fgvc-aircraft-2013b.tar.gz' `
    -ExpectedMd5 'd4acdd33327262359767eeaa97a4f732' `
    -ExtractedDirectory 'fgvc-aircraft-2013b'

Write-Status 'All requested public datasets are downloaded, verified, and extracted.'
