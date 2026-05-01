param(
    [ValidateSet("raw", "main", "pipeline", "all")]
    [string]$Target = "raw",

    [string]$Tasks = "ifeval,gsm8k",
    [int]$Limit = 50,
    [string]$Profile = "qwen3-8b-s2t-lite",
    [string]$RawModel = "qwen3:8b",
    [int]$Port = 8008,
    [string]$Python = "python",
    [string]$OutputRoot = "runs\public-bench",
    [switch]$NoLimit
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$repo = Split-Path -Parent $PSScriptRoot
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outputRootPath = Join-Path $repo $OutputRoot
New-Item -ItemType Directory -Force -Path $outputRootPath | Out-Null

function Test-LmEval {
    & $Python -m lm_eval --help *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "lm-eval is not installed for this Python. Install with: $Python -m pip install `"lm-eval[api,ifeval]`""
    }
}

function Invoke-LmEvalRun {
    param(
        [string]$Name,
        [string]$ModelName,
        [string]$BaseUrl
    )

    $out = Join-Path $outputRootPath "$Name-$stamp"
    $modelArgs = "model=$ModelName,base_url=$BaseUrl,num_concurrent=1,max_retries=3,tokenized_requests=False"
    $cmd = @(
        "-m", "lm_eval", "run",
        "--model", "local-chat-completions",
        "--model_args", $modelArgs,
        "--tasks", $Tasks,
        "--apply_chat_template",
        "--output_path", $out,
        "--log_samples"
    )
    if (-not $NoLimit) {
        $cmd += @("--limit", "$Limit")
    }

    Write-Host "Running $Name"
    & $Python @cmd
    if ($LASTEXITCODE -ne 0) {
        throw "lm-eval failed for $Name"
    }
}

function Invoke-WrappedRun {
    param(
        [ValidateSet("main", "pipeline")]
        [string]$Mode
    )

    $serverArgs = @(
        (Join-Path $repo "tools\public_bench_server.py"),
        "--profile", $Profile,
        "--mode", $Mode,
        "--port", "$Port",
        "--model-alias", "$Profile-$Mode"
    )
    $server = Start-Process -FilePath $Python -ArgumentList $serverArgs -PassThru -WindowStyle Hidden -WorkingDirectory $repo
    try {
        Start-Sleep -Seconds 5
        Invoke-LmEvalRun -Name "$Profile-$Mode" -ModelName "$Profile-$Mode" -BaseUrl "http://127.0.0.1:$Port/v1/chat/completions"
    }
    finally {
        if ($null -ne $server -and -not $server.HasExited) {
            Stop-Process -Id $server.Id -Force
        }
    }
}

Test-LmEval

if ($Target -eq "raw" -or $Target -eq "all") {
    Invoke-LmEvalRun -Name "$($RawModel.Replace(':','-'))-raw" -ModelName $RawModel -BaseUrl "http://localhost:11434/v1/chat/completions"
}

if ($Target -eq "main" -or $Target -eq "all") {
    Invoke-WrappedRun -Mode "main"
}

if ($Target -eq "pipeline" -or $Target -eq "all") {
    Invoke-WrappedRun -Mode "pipeline"
}
