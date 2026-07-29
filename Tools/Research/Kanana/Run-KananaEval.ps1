<#
.SYNOPSIS
  Scores a Hugging Face causal LM on this project's 20-case Korean tool-routing
  benchmark, on the desktop GPU.

.DESCRIPTION
  The LiteRT CLI only loads .litertlm bundles, so a candidate with no conversion
  path cannot be scored with the normal harness — which is exactly when a number
  is most useful, because the conversion is the expensive part. This wrapper runs
  the same 20 questions with the same tools and grading through transformers.

  The pass rate is comparable to docs/benchmarks/fc-model-benchmark.md.
  The latency is NOT: this is an RTX 4090 in bf16, not kona.

.PARAMETER Bootstrap
  Create Tools/Research/Kanana/KananaBench/.venv and install torch (CUDA 12.8)
  plus transformers. Needed once. Python 3.12 or 3.13; the py launcher defaults
  to 3.14 on this machine, which is avoided deliberately.

.EXAMPLE
  .\Tools\Research\Kanana\Run-KananaEval.ps1 -Bootstrap
  .\Tools\Research\Kanana\Run-KananaEval.ps1
  .\Tools\Research\Kanana\Run-KananaEval.ps1 -Model Qwen/Qwen3-0.6B -Label qwen3-0.6b-bf16
#>
[CmdletBinding()]
param(
    [string]$Model = "kakaocorp/kanana-2-1.3b-instruct",
    [string]$Label = "",
    [ValidateSet("bfloat16", "float16", "float32")]
    [string]$Dtype = "bfloat16",
    [string]$Device = "cuda",
    [int]$MaxNewTokens = 256,
    [string]$OutFile = "",
    [switch]$NoToolsApi,
    [switch]$ChatOnly,
    [switch]$SkipChat,
    [switch]$Bootstrap
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$BenchDir = Join-Path $PSScriptRoot "KananaBench"
$VenvPython = Join-Path $BenchDir ".venv\Scripts\python.exe"

if ($Bootstrap) {
    if (Test-Path $VenvPython) {
        Write-Host "venv already exists at $BenchDir\.venv — reusing it." -ForegroundColor Yellow
    }
    else {
        # py -3.12 is not registered here; find an interpreter the hard way.
        $interpreter = $null
        foreach ($version in @("3.12", "3.13")) {
            $found = & py "-$version" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $found) { $interpreter = $found.Trim(); break }
        }
        if (-not $interpreter) {
            $listed = & py -0p 2>$null | Where-Object { $_ -match "3\.1[23]" } | Select-Object -First 1
            if ($listed -and $listed -match "([A-Za-z]:\\[^\s].*python\.exe)") { $interpreter = $Matches[1] }
        }
        if (-not $interpreter) {
            throw "No Python 3.12/3.13 found. 3.14 is the py default here and is not supported by these wheels."
        }
        Write-Host "Creating venv with $interpreter" -ForegroundColor Cyan
        & $interpreter -m venv (Join-Path $BenchDir ".venv")
        if ($LASTEXITCODE -ne 0) { throw "venv creation failed." }
    }

    Write-Host "Installing torch (CUDA 12.8) — the PyPI wheel is CPU-only." -ForegroundColor Cyan
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
    if ($LASTEXITCODE -ne 0) { throw "torch install failed." }
    & $VenvPython -m pip install -r (Join-Path $BenchDir "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "requirements install failed." }
    & $VenvPython -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"
}

if (-not (Test-Path $VenvPython)) {
    throw "No venv. Run with -Bootstrap first."
}

if (-not $Label) { $Label = ($Model -split "/")[-1] }
if (-not $OutFile) {
    $OutFile = Join-Path $ProjectRoot "Builds\Logs\$Label-fc-bench.jsonl"
}
New-Item -ItemType Directory -Force (Split-Path $OutFile) | Out-Null

$arguments = @(
    (Join-Path $BenchDir "kanana_fc_bench.py"),
    "--model", $Model,
    "--label", $Label,
    "--dtype", $Dtype,
    "--device", $Device,
    "--max-new-tokens", $MaxNewTokens,
    "--out", $OutFile
)
if ($NoToolsApi) { $arguments += "--no-tools-api" }
if ($ChatOnly) { $arguments += "--chat-only" }
if ($SkipChat) { $arguments += "--skip-chat" }

$env:PYTHONIOENCODING = "utf-8"
Write-Host "Scoring $Model ($Dtype on $Device)" -ForegroundColor Cyan
& $VenvPython @arguments
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Host "Results: $OutFile" -ForegroundColor Green
}
exit $exitCode
