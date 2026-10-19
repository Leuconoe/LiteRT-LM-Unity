<#
.SYNOPSIS
  Trains and scores the jevlike decision head on frozen gemma-3-270m-it layer-12
  states (the offline Jev sample), or checks an exported trunk .tflite against HF.

.PARAMETER Bootstrap
  Create Tools/Research/Jev/JevBench/.venv (Python 3.12/3.13 — never the 3.14 py
  default) with torch cu128 + requirements.txt.

.EXAMPLE
  .\Tools\Research\Jev\Run-JevHead.ps1 -Bootstrap
  .\Tools\Research\Jev\Run-JevHead.ps1 train
  .\Tools\Research\Jev\Run-JevHead.ps1 parity --tflite External\jev-work\trunk\gemma3_270m_l12_t48_w8.tflite
#>
[CmdletBinding()]
param(
    [switch]$Bootstrap,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$PassThru = @()
)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$BenchDir = Join-Path $PSScriptRoot "JevBench"
$VenvPython = Join-Path $BenchDir ".venv\Scripts\python.exe"

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
    & $VenvPython -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"
    if (-not $PassThru) { exit 0 }
}
if (-not (Test-Path $VenvPython)) { throw "No venv. Run with -Bootstrap first." }

# Gated model: the HF token lives in the gitignored repo-root .env — never printed.
$envFile = Join-Path $ProjectRoot ".env"
if (Test-Path $envFile) {
    $line = Get-Content $envFile | Where-Object { $_ -match '^\s*HF_TOKEN\s*=' } | Select-Object -First 1
    if ($line) { $env:HF_TOKEN = ($line -replace '^\s*HF_TOKEN\s*=\s*', '').Trim('"', "'", ' ') }
}
$env:PYTHONIOENCODING = "utf-8"
Push-Location $ProjectRoot
try { & $VenvPython (Join-Path $BenchDir "jev_head.py") @PassThru; exit $LASTEXITCODE }
finally { Pop-Location }
