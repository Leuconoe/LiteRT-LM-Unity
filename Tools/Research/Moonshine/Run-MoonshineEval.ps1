<#
.SYNOPSIS
  Scores Moonshine Korean ASR models (and stock Whisper through the same harness)
  on this project's Korean sets, on the desktop GPU.

.DESCRIPTION
  Moonshine has no Korean .tflite release and no LiteRT driver in the AAR yet, so
  the cheap question — is it more accurate than the Whisper tiers we ship? — is
  answered here through transformers before any conversion work is spent.

  Buckets: ko_short (40-clip TTS gate, ranking only), real_cmd (the Korean clips
  in TestAssets/Audio), fleurs_ko (real read speech, absolute check against the
  model card). CER uses the training queue's definition; a punctuation-stripped
  CER is reported alongside because the shipped matcher ignores punctuation.

  Latency is an RTX 4090 number, not kona.

.PARAMETER Bootstrap
  Create MoonshineBench/.venv (Python 3.12 via uv) with torch cu128 and
  transformers. Needed once.

.EXAMPLE
  .\Tools\Research\Moonshine\Run-MoonshineEval.ps1 -Bootstrap
  .\Tools\Research\Moonshine\Run-MoonshineEval.ps1
  .\Tools\Research\Moonshine\Run-MoonshineEval.ps1 -Models moonshine-base-ko,whisper-turbo -Buckets ko_short
#>
[CmdletBinding()]
param(
    [string[]]$Models = @("moonshine-tiny-ko", "moonshine-base-ko", "whisper-tiny", "whisper-base"),
    [string[]]$Buckets = @("ko_short", "real_cmd", "fleurs_ko"),
    [int]$FleursLimit = 100,
    [string]$Device = "cuda",
    [ValidateSet("float32", "float16", "bfloat16")]
    [string]$Dtype = "float32",
    [double]$PadTo = 0,
    [string]$OutFile = "",
    [switch]$Bootstrap
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$BenchDir = Join-Path $PSScriptRoot "MoonshineBench"
$VenvPython = Join-Path $BenchDir ".venv\Scripts\python.exe"

if ($Bootstrap) {
    if (Test-Path $VenvPython) {
        Write-Host "venv already exists at $BenchDir\.venv — reusing it." -ForegroundColor Yellow
    }
    else {
        Push-Location $BenchDir
        try {
            uv venv --python 3.12 .venv
            uv pip install --python .venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
            uv pip install --python .venv\Scripts\python.exe -r requirements.txt
        }
        finally { Pop-Location }
    }
    & $VenvPython -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| transformers', transformers.__version__)"
    if (-not $PSBoundParameters.ContainsKey('Models')) { return }
}

if (-not (Test-Path $VenvPython)) {
    throw "No venv at $BenchDir\.venv — run with -Bootstrap first."
}

if (-not $OutFile) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutFile = Join-Path $ProjectRoot "Builds\Logs\MoonshineEval\moonshine-eval-$stamp.json"
}
New-Item -ItemType Directory -Force (Split-Path -Parent $OutFile) | Out-Null

# The HF token lives in the gitignored .env (HF_TOKEN=...); export it for the run only.
$envFile = Join-Path $ProjectRoot ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$') {
            Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2].Trim('"')
        }
    }
}

$argv = @((Join-Path $BenchDir "eval_moonshine.py"),
    "--models") + $Models + @("--buckets") + $Buckets + @(
    "--fleurs-limit", $FleursLimit,
    "--device", $Device,
    "--dtype", $Dtype,
    "--pad-to", $PadTo,
    "--out", $OutFile)

Write-Host "> $VenvPython $($argv -join ' ')" -ForegroundColor DarkGray
& $VenvPython @argv
if ($LASTEXITCODE -ne 0) { throw "eval_moonshine.py exited with $LASTEXITCODE" }
Write-Host "Results: $OutFile" -ForegroundColor Green
