<#
.SYNOPSIS
  Korean retrieval benchmark for the RAG sample: embedder comparison, the LiteRT
  e5 encoder's parity/recall, and the SentencePiece id check for the device path.

.PARAMETER Bootstrap
  Create Tools/Research/Rag/RagBench/.venv (Python 3.12/3.13 — never the 3.14 py
  default) with torch cu128 + requirements.txt.

.PARAMETER Export
  Convert multilingual-e5 to fixed-shape .tflite (fp32/w8/i8) inside the
  litert-torch converter image (built by Convert-FunctionGemmaToLiteRtLm.ps1 -Bootstrap).

.EXAMPLE
  .\Tools\Research\Rag\Run-RagBench.ps1 -Bootstrap
  .\Tools\Research\Rag\Run-RagBench.ps1 bench --models e5-small,e5-base
  .\Tools\Research\Rag\Run-RagBench.ps1 -Export -Model intfloat/multilingual-e5-small -Lengths 64,256
  .\Tools\Research\Rag\Run-RagBench.ps1 tflite --dir External\rag-work\e5-small
  .\Tools\Research\Rag\Run-RagBench.ps1 tokcheck
#>
[CmdletBinding(PositionalBinding = $false)]  # bare words go to rag_eval.py, not to -Model/-Lengths
param(
    [switch]$Bootstrap,
    [switch]$Export,
    [string]$Model = "intfloat/multilingual-e5-small",
    [int[]]$Lengths = @(64, 256),
    [string]$OutDir = "",
    [string]$Image = "litert-torch-converter:0.9.4",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$PassThru = @()
)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$BenchDir = Join-Path $PSScriptRoot "RagBench"
$VenvPython = Join-Path $BenchDir ".venv\Scripts\python.exe"

$line = Get-Content (Join-Path $ProjectRoot ".env") -ErrorAction SilentlyContinue | Where-Object { $_ -match '^\s*HF_TOKEN\s*=' } | Select-Object -First 1
if ($line) { $env:HF_TOKEN = ($line -replace '^\s*HF_TOKEN\s*=\s*', '').Trim('"', "'", ' ') }  # gated embeddinggemma; never printed

if ($Export) {
    if (-not $OutDir) { $OutDir = Join-Path $ProjectRoot ("External\rag-work\" + ($Model -split "/")[-1]) }
    New-Item -ItemType Directory -Force $OutDir | Out-Null
    $mounts = @("-v", "fg-hf-cache:/root/.cache/huggingface", "-v", "${OutDir}:/out", "-v", "${BenchDir}:/bench:ro")
    docker run --rm -e HF_TOKEN @mounts $Image python /bench/export_e5.py --model $Model --lengths ($Lengths -join ",") --out /out
    exit $LASTEXITCODE
}

if ($Bootstrap) {
    if (-not (Test-Path $VenvPython)) {
        $interpreter = $null
        foreach ($version in @("3.13", "3.12")) {
            $found = & py "-$version" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $found) { $interpreter = $found.Trim(); break }
        }
        if (-not $interpreter) { throw "No Python 3.12/3.13 found (3.14 has no numpy wheel here)." }
        & $interpreter -m venv (Join-Path $BenchDir ".venv")
        if ($LASTEXITCODE -ne 0) { throw "venv creation failed." }
    }
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
    if ($LASTEXITCODE -ne 0) { throw "torch install failed." }
    & $VenvPython -m pip install -r (Join-Path $BenchDir "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "requirements install failed." }
    & $VenvPython -c "import torch, transformers, sentence_transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__, 'st', sentence_transformers.__version__)"
    if (-not $PassThru) { exit 0 }
}
if (-not (Test-Path $VenvPython)) { throw "No venv. Run with -Bootstrap first." }
$env:PYTHONIOENCODING = "utf-8"
Push-Location $ProjectRoot
try { & $VenvPython (Join-Path $BenchDir "rag_eval.py") @PassThru; exit $LASTEXITCODE }
finally { Pop-Location }
