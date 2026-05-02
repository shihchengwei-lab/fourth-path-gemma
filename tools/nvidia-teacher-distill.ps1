param(
    [string]$InputFile = "data\main_agent_hard_seed.jsonl",
    [string]$OutputFile = "runs\main-agent-nvidia-teacher.jsonl",
    [int]$LimitRecords = 3,
    [int]$SamplesPerModel = 1,
    [double]$RequestsPerMinute = 36,
    [int]$Timeout = 1200,
    [string[]]$Model = @(),
    [switch]$SkipReport,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Set-Location $repoRoot

if (-not $env:NVIDIA_BASE_URL) {
    $env:NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
}

if (-not $DryRun -and -not $env:NVIDIA_API_KEY) {
    $secureKey = Read-Host "Paste NVIDIA API key for this run" -AsSecureString
    if ($secureKey.Length -eq 0) {
        throw "NVIDIA API key is empty."
    }

    $keyPtr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    try {
        $env:NVIDIA_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPtr)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPtr)
    }
}

$exportArgs = @(
    "main.py",
    "main-nvidia-teacher-export",
    "--input-file",
    $InputFile,
    "--output-file",
    $OutputFile,
    "--limit-records",
    "$LimitRecords",
    "--samples-per-model",
    "$SamplesPerModel",
    "--requests-per-minute",
    "$RequestsPerMinute",
    "--timeout",
    "$Timeout",
    "--json"
)

foreach ($modelId in $Model) {
    if ($modelId.Trim()) {
        $exportArgs += @("--model", $modelId)
    }
}

if ($DryRun) {
    Write-Host "Dry run only. NVIDIA_API_KEY was not read."
    Write-Host ("python " + ($exportArgs -join " "))
    if (-not $SkipReport) {
        Write-Host (
            "python main.py main-training-data-report --input-file " +
            "$OutputFile --require-system --require-generated-metadata --json"
        )
    }
    exit 0
}

& python @exportArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if (-not $SkipReport) {
    & python main.py main-training-data-report `
        --input-file $OutputFile `
        --require-system `
        --require-generated-metadata `
        --json
    exit $LASTEXITCODE
}
