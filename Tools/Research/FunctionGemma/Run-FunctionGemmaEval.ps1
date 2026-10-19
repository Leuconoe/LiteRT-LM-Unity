<#
.SYNOPSIS
  Scores FunctionGemma (stock or fine-tuned) on voice-command eval cases, and runs the fine-tune.

.DESCRIPTION
  The tool contract is an OpenAI-style tools JSON (default: FunctionGemmaBench\sample_contract.json),
  rendered through FunctionGemma's own chat template. Pass rate transfers to the device once the
  model is converted; latency here is the desktop and does NOT.

.PARAMETER Bootstrap
  Create FunctionGemmaBench/.venv and install torch (CUDA 12.8) plus requirements.txt. Needed once.
  Python 3.12 or 3.13 — 3.14 has no wheels for this stack yet.

.PARAMETER Train
  Run fg_train.py instead of fg_eval.py. Everything in -PassThru goes to the script.

.EXAMPLE
  .\Tools\Research\FunctionGemma\Run-FunctionGemmaEval.ps1 -Bootstrap
  .\Tools\Research\FunctionGemma\Run-FunctionGemmaEval.ps1 -DeclMode full                    # stock model, declarations in the prompt
  .\Tools\Research\FunctionGemma\Run-FunctionGemmaEval.ps1 -Model External\functiongemma-work\ft-functiongemma-none\merged
  .\Tools\Research\FunctionGemma\Run-FunctionGemmaEval.ps1 -Train -PassThru @('--train','Tools\Research\FunctionGemma\FunctionGemmaBench\data\train.jsonl','--val','Tools\Research\FunctionGemma\FunctionGemmaBench\data\val.jsonl')
#>
[CmdletBinding()]
param(
    [string]$Model = "",  # default: google/functiongemma-270m-it
    [ValidateSet("full", "compact", "none")]
    [string]$DeclMode = "none",
    [string]$Cases = "sample_eval_cases.json",
    [string]$Contract = "",  # default: FunctionGemmaBench\sample_contract.json
    [string]$Label = "",
    [string]$OutFile = "",
    [switch]$Train,
    [string[]]$PassThru = @(),
    [switch]$Bootstrap
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$BenchDir = Join-Path $PSScriptRoot "FunctionGemmaBench"
$VenvPython = Join-Path $BenchDir ".venv\Scripts\python.exe"

if ($Bootstrap) {
    if (Test-Path $VenvPython) {
        Write-Host "venv already exists at $BenchDir\.venv — reusing it." -ForegroundColor Yellow
    }
    else {
        $interpreter = $null
        foreach ($version in @("3.13", "3.12")) {
            $found = & py "-$version" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $found) { $interpreter = $found.Trim(); break }
        }
        if (-not $interpreter) {
            $listed = & py -0p 2>$null | Where-Object { $_ -match "3\.1[23]" } | Select-Object -First 1
            if ($listed -and $listed -match "([A-Za-z]:\\[^\s].*python\.exe)") { $interpreter = $Matches[1] }
        }
        if (-not $interpreter) {
            throw "No Python 3.12/3.13 found (3.14 is not supported by these wheels)."
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
    & $VenvPython -c "import torch, transformers, trl; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__, 'trl', trl.__version__)"
    exit $LASTEXITCODE
}

if (-not (Test-Path $VenvPython)) {
    throw "No venv. Run with -Bootstrap first."
}

# google/functiongemma-270m-it is gated; put HF_TOKEN in the environment or in a gitignored .env.
$envFile = Join-Path $ProjectRoot ".env"
if (-not $env:HF_TOKEN -and (Test-Path $envFile)) {
    $line = Get-Content $envFile | Where-Object { $_ -match '^\s*HF_TOKEN\s*=' } | Select-Object -First 1
    if ($line) { $env:HF_TOKEN = ($line -split "=", 2)[1].Trim().Trim('"').Trim("'") }
}

$env:PYTHONIOENCODING = "utf-8"

if ($Train) {
    & $VenvPython (Join-Path $BenchDir "fg_train.py") @PassThru
    exit $LASTEXITCODE
}

$modelName = if ($Model) { ($Model -split '[\\/]')[-1] } else { "functiongemma-270m-it" }
if (-not $Label) { $Label = "$modelName-$DeclMode" }
if (-not $OutFile) { $OutFile = Join-Path $ProjectRoot "Builds\Logs\fg-$Label-$([IO.Path]::GetFileNameWithoutExtension($Cases)).jsonl" }
New-Item -ItemType Directory -Force (Split-Path $OutFile) | Out-Null

$casesPath = if ([IO.Path]::IsPathRooted($Cases)) { $Cases } elseif (Test-Path $Cases) { (Resolve-Path $Cases).Path } else { Join-Path $BenchDir $Cases }
$extra = @(if ($Model) { "--model", $Model }) + @(if ($Contract) { "--contract", (Resolve-Path $Contract).Path })

& $VenvPython (Join-Path $BenchDir "fg_eval.py") @extra --cases $casesPath --decl-mode $DeclMode --label $Label --out $OutFile @PassThru
exit $LASTEXITCODE
