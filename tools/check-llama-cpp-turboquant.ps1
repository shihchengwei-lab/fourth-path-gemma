param(
    [string]$RepoPath = "C:\Users\kk789\Desktop\llama-cpp-turboquant",
    [string]$ModelPath = "E:\ollama-models\blobs\sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f"
)

$ErrorActionPreference = "Stop"

function Test-Command {
    param([string]$Name)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Write-Item {
    param(
        [string]$Name,
        [string]$Value
    )
    Write-Output ("{0}: {1}" -f $Name, $Value)
}

Write-Output "llama-cpp-turboquant readiness"
Write-Output ""

Write-Item "repo_path" $RepoPath
Write-Item "repo_exists" ([string](Test-Path -LiteralPath $RepoPath))
if (Test-Path -LiteralPath $RepoPath) {
    Push-Location $RepoPath
    try {
        Write-Item "git_status" ((git status --short --branch) -join " | ")
        Write-Item "git_head" ((git log -1 --oneline) -join "")
        $bins = Get-ChildItem -Recurse -File -Include llama-cli.exe,llama-server.exe,llama-bench.exe,test-turbo-quant.exe -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty FullName
        Write-Item "binaries_found" ([string]($bins.Count))
        foreach ($bin in $bins) {
            Write-Output ("binary: {0}" -f $bin)
        }
    } finally {
        Pop-Location
    }
}

Write-Output ""
Write-Item "cmake" ([string](Test-Command "cmake"))
Write-Item "ninja" ([string](Test-Command "ninja"))
Write-Item "nvcc" ([string](Test-Command "nvcc"))
Write-Item "cl" ([string](Test-Command "cl"))
Write-Item "wsl" ([string](Test-Command "wsl"))

Write-Output ""
Write-Item "model_path" $ModelPath
Write-Item "model_exists" ([string](Test-Path -LiteralPath $ModelPath))
if (Test-Path -LiteralPath $ModelPath) {
    $bytes = Get-Content -LiteralPath $ModelPath -Encoding Byte -TotalCount 4
    $magic = [System.Text.Encoding]::ASCII.GetString($bytes)
    Write-Item "model_magic" $magic
    Write-Item "model_is_gguf" ([string]($magic -eq "GGUF"))
}
