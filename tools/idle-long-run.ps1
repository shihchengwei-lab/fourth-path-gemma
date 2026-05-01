param(
    [string]$Python = "python",
    [string]$RunsDir = "runs",
    [switch]$IncludeModelTrials
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

if (-not (Test-Path -LiteralPath $RunsDir)) {
    New-Item -ItemType Directory -Path $RunsDir | Out-Null
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $RunsDir "idle-long-run-$stamp.log"

function Invoke-LoggedStep {
    param(
        [string]$Name,
        [string[]]$CommandArgs
    )

    $started = Get-Date
    "[$($started.ToString('s'))] START $Name" | Tee-Object -FilePath $logPath -Append
    "& $Python $($CommandArgs -join ' ')" | Tee-Object -FilePath $logPath -Append
    $resolvedRunsDir = (Resolve-Path -LiteralPath $RunsDir).Path
    $safeName = $Name -replace "[^A-Za-z0-9_.-]", "_"
    $stdoutPath = Join-Path $resolvedRunsDir "idle-step-$stamp-$safeName.out"
    $stderrPath = Join-Path $resolvedRunsDir "idle-step-$stamp-$safeName.err"

    $process = Start-Process -FilePath $Python -ArgumentList $CommandArgs `
        -Wait -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath

    if ((Test-Path -LiteralPath $stdoutPath) -and ((Get-Item -LiteralPath $stdoutPath).Length -gt 0)) {
        Get-Content -LiteralPath $stdoutPath | Tee-Object -FilePath $logPath -Append
    }
    if ((Test-Path -LiteralPath $stderrPath) -and ((Get-Item -LiteralPath $stderrPath).Length -gt 0)) {
        Get-Content -LiteralPath $stderrPath | Tee-Object -FilePath $logPath -Append
    }
    Remove-Item -LiteralPath $stdoutPath, $stderrPath -ErrorAction SilentlyContinue

    $exitCode = $process.ExitCode
    $ended = Get-Date
    "[$($ended.ToString('s'))] END $Name exit=$exitCode seconds=$([int]($ended - $started).TotalSeconds)" |
        Tee-Object -FilePath $logPath -Append
    if ($exitCode -ne 0) {
        throw "Step failed: $Name"
    }
}

"Idle long run started at $(Get-Date -Format s)" | Tee-Object -FilePath $logPath -Append
"Log: $logPath" | Tee-Object -FilePath $logPath -Append

Invoke-LoggedStep -Name "unit tests" -CommandArgs @("-m", "unittest", "discover", "-s", "tests", "-v")
Invoke-LoggedStep -Name "syntax check" -CommandArgs @("-m", "py_compile", "main.py")
Invoke-LoggedStep -Name "architecture check" -CommandArgs @("main.py", "architecture-check", "--json")
Invoke-LoggedStep -Name "architecture adversarial seed check" -CommandArgs @(
    "main.py", "architecture-adversarial-check", "--min-total", "12", "--min-layer", "6", "--json"
)
Invoke-LoggedStep -Name "main seed check" -CommandArgs @("main.py", "main-check", "--min-total", "40", "--min-category", "1", "--json")
Invoke-LoggedStep -Name "cold eyes seed check" -CommandArgs @("main.py", "distill-check", "--min-pass", "19", "--min-fail", "25", "--min-clause", "8", "--json")
Invoke-LoggedStep -Name "main sft export" -CommandArgs @("main.py", "main-sft-export", "--json", "--output-file", (Join-Path $RunsDir "main-agent-sft-seed-$stamp.jsonl"))
Invoke-LoggedStep -Name "qwen warm" -CommandArgs @("main.py", "warm", "--profile", "qwen3-8b-local-max", "--json", "--timeout", "900")
Invoke-LoggedStep -Name "architecture adversarial eval local max" -CommandArgs @(
    "main.py", "architecture-adversarial-eval", "--profile", "qwen3-8b-local-max", "--json", "--timeout", "900",
    "--output-file", (Join-Path $RunsDir "architecture-adversarial-eval-qwen3-8b-local-max-idle-$stamp.json")
)
Invoke-LoggedStep -Name "main eval local max" -CommandArgs @(
    "main.py", "main-eval", "--profile", "qwen3-8b-local-max", "--json", "--timeout", "900",
    "--max-length-ratio", "4", "--output-file", (Join-Path $RunsDir "main-eval-qwen3-8b-local-max-idle-$stamp.json")
)
Invoke-LoggedStep -Name "main eval deliberate" -CommandArgs @(
    "main.py", "main-eval", "--profile", "qwen3-8b-deliberate", "--json", "--timeout", "900",
    "--max-length-ratio", "4", "--output-file", (Join-Path $RunsDir "main-eval-qwen3-8b-deliberate-idle-$stamp.json")
)
Invoke-LoggedStep -Name "main eval reasoning" -CommandArgs @(
    "main.py", "main-eval", "--profile", "qwen3-8b-reasoning", "--json", "--timeout", "900",
    "--max-length-ratio", "4", "--output-file", (Join-Path $RunsDir "main-eval-qwen3-8b-reasoning-idle-$stamp.json")
)
Invoke-LoggedStep -Name "main eval search" -CommandArgs @(
    "main.py", "main-eval", "--profile", "qwen3-8b-search", "--json", "--timeout", "900",
    "--max-length-ratio", "4", "--output-file", (Join-Path $RunsDir "main-eval-qwen3-8b-search-idle-$stamp.json")
)
Invoke-LoggedStep -Name "bench local max" -CommandArgs @(
    "main.py", "bench", "--profile", "qwen3-8b-local-max", "--warmup", "--json", "--timeout", "900",
    "--output-file", (Join-Path $RunsDir "bench-qwen3-8b-local-max-idle-$stamp.json")
)
Invoke-LoggedStep -Name "bench deliberate" -CommandArgs @(
    "main.py", "bench", "--profile", "qwen3-8b-deliberate", "--warmup", "--json", "--timeout", "900",
    "--output-file", (Join-Path $RunsDir "bench-qwen3-8b-deliberate-idle-$stamp.json")
)
Invoke-LoggedStep -Name "bench reasoning" -CommandArgs @(
    "main.py", "bench", "--profile", "qwen3-8b-reasoning", "--warmup", "--json", "--timeout", "900",
    "--output-file", (Join-Path $RunsDir "bench-qwen3-8b-reasoning-idle-$stamp.json")
)
Invoke-LoggedStep -Name "bench search" -CommandArgs @(
    "main.py", "bench", "--profile", "qwen3-8b-search", "--warmup", "--json", "--timeout", "900",
    "--output-file", (Join-Path $RunsDir "bench-qwen3-8b-search-idle-$stamp.json")
)
Invoke-LoggedStep -Name "distill eval local max" -CommandArgs @(
    "main.py", "distill-eval", "--profile", "qwen3-8b-local-max", "--json", "--timeout", "900",
    "--require-exact", "--min-exact-accuracy", "1", "--min-mechanical-cases", "25",
    "--output-file", (Join-Path $RunsDir "distill-eval-qwen3-8b-local-max-idle-$stamp.json")
)

if ($IncludeModelTrials) {
    Invoke-LoggedStep -Name "bench split audit" -CommandArgs @(
        "main.py", "bench", "--profile", "qwen3-8b-split-audit", "--warmup", "--json", "--timeout", "900",
        "--output-file", (Join-Path $RunsDir "bench-qwen3-8b-split-audit-idle-$stamp.json")
    )
    Invoke-LoggedStep -Name "bench gemma pressure" -CommandArgs @(
        "main.py", "bench", "--profile", "gemma3-12b-pressure", "--warmup", "--json", "--timeout", "900",
        "--output-file", (Join-Path $RunsDir "bench-gemma3-12b-pressure-idle-$stamp.json")
    )
}

"Idle long run completed at $(Get-Date -Format s)" | Tee-Object -FilePath $logPath -Append
