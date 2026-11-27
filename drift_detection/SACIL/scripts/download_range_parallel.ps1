param(
    [Parameter(Mandatory = $true)]
    [string]$Url,

    [Parameter(Mandatory = $true)]
    [string]$OutputFile,

    [Parameter(Mandatory = $true)]
    [long]$ExpectedSize,

    [Parameter(Mandatory = $true)]
    [string]$ExpectedMd5,

    [ValidateRange(1, 64)]
    [int]$Connections = 16
)

$ErrorActionPreference = 'Stop'

$outputPath = [System.IO.Path]::GetFullPath($OutputFile)
$partDirectory = "$outputPath.parts"
$statusFile = "$outputPath.parallel.log"
$assembledFile = "$outputPath.assembled"

New-Item -ItemType Directory -Path $partDirectory -Force | Out-Null

function Write-Status {
    param([string]$Message)

    $line = '[{0}] {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    $line | Tee-Object -FilePath $statusFile -Append
}

$chunkSize = [long][math]::Ceiling($ExpectedSize / $Connections)
$downloads = @()

Write-Status "Starting $Connections range downloads for $outputPath."

for ($index = 0; $index -lt $Connections; $index++) {
    $start = [long]$index * $chunkSize
    $end = [math]::Min($ExpectedSize - 1, $start + $chunkSize - 1)
    if ($start -gt $end) {
        break
    }

    $expectedPartSize = $end - $start + 1
    $partFile = Join-Path $partDirectory ('part-{0:D3}' -f $index)
    $stderrFile = "$partFile.stderr.log"

    if ((Test-Path -LiteralPath $partFile) -and
        ((Get-Item -LiteralPath $partFile).Length -eq $expectedPartSize)) {
        Write-Status "Part $index already complete; reusing it."
        $downloads += [pscustomobject]@{
            Index = $index
            Start = $start
            End = $end
            ExpectedSize = $expectedPartSize
            PartFile = $partFile
            Process = $null
        }
        continue
    }

    $process = Start-Process `
        -FilePath 'curl.exe' `
        -ArgumentList @(
            '--location',
            '--fail',
            '--retry', '8',
            '--retry-all-errors',
            '--retry-delay', '5',
            '--range', "$start-$end",
            '--output', $partFile,
            $Url
        ) `
        -WindowStyle Hidden `
        -RedirectStandardError $stderrFile `
        -PassThru

    $downloads += [pscustomobject]@{
        Index = $index
        Start = $start
        End = $end
        ExpectedSize = $expectedPartSize
        PartFile = $partFile
        Process = $process
    }
}

$lastReport = Get-Date
while ($true) {
    $running = @(
        $downloads |
            Where-Object { $_.Process -ne $null -and -not $_.Process.HasExited }
    )
    if ($running.Count -eq 0) {
        break
    }

    Start-Sleep -Seconds 5
    if (((Get-Date) - $lastReport).TotalSeconds -ge 30) {
        $downloadedBytes = 0L
        foreach ($download in $downloads) {
            if (Test-Path -LiteralPath $download.PartFile) {
                $downloadedBytes += (Get-Item -LiteralPath $download.PartFile).Length
            }
        }
        $percent = 100.0 * $downloadedBytes / $ExpectedSize
        Write-Status ('Downloaded {0:N1} MB / {1:N1} MB ({2:N1}%).' -f
            ($downloadedBytes / 1MB), ($ExpectedSize / 1MB), $percent)
        $lastReport = Get-Date
    }
}

foreach ($download in $downloads) {
    if ($download.Process -ne $null) {
        $download.Process.WaitForExit()
    }

    if (-not (Test-Path -LiteralPath $download.PartFile)) {
        throw "Part $($download.Index) is missing."
    }

    $actualPartSize = (Get-Item -LiteralPath $download.PartFile).Length
    if ($actualPartSize -ne $download.ExpectedSize) {
        throw "Part $($download.Index) size mismatch: expected $($download.ExpectedSize), got $actualPartSize."
    }
}

Write-Status 'All ranges downloaded. Assembling archive.'

$outputStream = [System.IO.File]::Open(
    $assembledFile,
    [System.IO.FileMode]::Create,
    [System.IO.FileAccess]::Write,
    [System.IO.FileShare]::None
)
try {
    foreach ($download in ($downloads | Sort-Object Index)) {
        $inputStream = [System.IO.File]::OpenRead($download.PartFile)
        try {
            $inputStream.CopyTo($outputStream)
        } finally {
            $inputStream.Dispose()
        }
    }
} finally {
    $outputStream.Dispose()
}

$assembledSize = (Get-Item -LiteralPath $assembledFile).Length
if ($assembledSize -ne $ExpectedSize) {
    throw "Assembled size mismatch: expected $ExpectedSize, got $assembledSize."
}

$actualMd5 = (Get-FileHash -LiteralPath $assembledFile -Algorithm MD5).Hash.ToLowerInvariant()
if ($actualMd5 -ne $ExpectedMd5.ToLowerInvariant()) {
    throw "Assembled MD5 mismatch: expected $ExpectedMd5, got $actualMd5."
}

Move-Item -LiteralPath $assembledFile -Destination $outputPath -Force
Write-Status "Archive verified and ready: MD5 $actualMd5."

foreach ($download in $downloads) {
    Remove-Item -LiteralPath $download.PartFile -Force
    $stderrFile = "$($download.PartFile).stderr.log"
    if (Test-Path -LiteralPath $stderrFile) {
        Remove-Item -LiteralPath $stderrFile -Force
    }
}
Remove-Item -LiteralPath $partDirectory -Force
Write-Status 'Temporary range parts removed.'
