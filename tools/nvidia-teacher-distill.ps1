param(
    [string]$InputFile = "data\main_agent_hard_seed.jsonl",
    [string]$OutputFile = "runs\main-agent-nvidia-teacher.jsonl",
    [string]$SummaryFile = "runs\main-agent-nvidia-teacher-summary.json",
    [string]$ReportFile = "runs\main-agent-nvidia-teacher-report.json",
    [int]$LimitRecords = 3,
    [int]$SamplesPerModel = 1,
    [double]$RequestsPerMinute = 36,
    [int]$Timeout = 180,
    [string[]]$Model = @(),
    [switch]$PersistUserKey,
    [switch]$SkipReport,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Set-Location $repoRoot

if (-not $env:NVIDIA_BASE_URL) {
    $storedBaseUrl = [Environment]::GetEnvironmentVariable("NVIDIA_BASE_URL", "User")
    if ($storedBaseUrl) {
        $env:NVIDIA_BASE_URL = $storedBaseUrl
    }
    else {
        $env:NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
    }
}

if (-not $env:NVIDIA_API_KEY) {
    $storedKey = [Environment]::GetEnvironmentVariable("NVIDIA_API_KEY", "User")
    if ($storedKey) {
        $env:NVIDIA_API_KEY = $storedKey
        Write-Host "Loaded NVIDIA_API_KEY from Windows user environment."
    }
}

if (-not $DryRun -and -not $env:NVIDIA_API_KEY) {
    $prompt = "Paste NVIDIA API key for this run"
    if ($PersistUserKey) {
        $prompt = "Paste NVIDIA API key to store in Windows user environment"
    }
    $secureKey = Read-Host $prompt -AsSecureString
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

if (-not $DryRun -and $PersistUserKey) {
    [Environment]::SetEnvironmentVariable("NVIDIA_API_KEY", $env:NVIDIA_API_KEY, "User")
    [Environment]::SetEnvironmentVariable("NVIDIA_BASE_URL", $env:NVIDIA_BASE_URL, "User")
    Write-Host "Stored NVIDIA_API_KEY and NVIDIA_BASE_URL in Windows user environment."
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
    "--progress",
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
        Write-Host "Export summary file: $SummaryFile"
        Write-Host "Training report file: $ReportFile"
    }
    exit 0
}

Write-Host "Starting NVIDIA teacher distillation from $repoRoot"
Write-Host "Input: $InputFile"
Write-Host "Output: $OutputFile"
Write-Host "Summary: $SummaryFile"
Write-Host "LimitRecords: $LimitRecords; SamplesPerModel: $SamplesPerModel; RequestsPerMinute: $RequestsPerMinute; Timeout: $Timeout"

$summaryParent = Split-Path -Parent $SummaryFile
if ($summaryParent) {
    New-Item -ItemType Directory -Path $summaryParent -Force | Out-Null
}

& python @exportArgs | Tee-Object -FilePath $SummaryFile
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if (-not $SkipReport) {
    $reportParent = Split-Path -Parent $ReportFile
    if ($reportParent) {
        New-Item -ItemType Directory -Path $reportParent -Force | Out-Null
    }
    & python main.py main-training-data-report `
        --input-file $OutputFile `
        --require-system `
        --require-generated-metadata `
        --json |
        Tee-Object -FilePath $ReportFile
    exit $LASTEXITCODE
}
