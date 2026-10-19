<#
.SYNOPSIS
  Scores SenseVoice (stock, fine-tuned, ONNX or converted tflite) on the same Korean
  sets the whisper tiers were scored on.

.DESCRIPTION
  Thin wrapper over SenseVoiceBench/sv_eval.py — every argument after the switches is
  passed through. Sets: gate40 (40-clip TTS short-command holdout = the ACFT gate,
  ranking only), real (recorded command clips), zeroth (test, 457), fleurs_ko (dev 64).

.PARAMETER Bootstrap
  Create SenseVoiceBench/.venv (Python 3.12/3.13, never 3.14), install torch +
  torchaudio from the cu128 index, then requirements.txt.

.EXAMPLE
  .\Tools\Research\SenseVoice\Run-SenseVoiceEval.ps1 -Bootstrap
  .\Tools\Research\SenseVoice\Run-SenseVoiceEval.ps1 --backend funasr --model FunAudioLLM/SenseVoiceSmall --label stock
  .\Tools\Research\SenseVoice\Run-SenseVoiceEval.ps1 --backend tflite --model External\sensevoice-work\tflite\stock\w8 --label stock-w8
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [switch]$Bootstrap,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$EvalArgs = @()
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$BenchDir = Join-Path $PSScriptRoot "SenseVoiceBench"
$VenvPython = Join-Path $BenchDir ".venv\Scripts\python.exe"

if ($Bootstrap) {
    if (!(Test-Path $VenvPython)) {
        $base = "C:\Users\user\AppData\Roaming\uv\python\cpython-3.12.13-windows-x86_64-none\python.exe"
        if (!(Test-Path $base)) { $base = (& py -3.13 -c "import sys; print(sys.executable)").Trim() }
        & $base -m venv (Join-Path $BenchDir ".venv")
        if ($LASTEXITCODE -ne 0) { throw "venv creation failed." }
    }
    & $VenvPython -m pip install --upgrade pip | Out-Null
    & $VenvPython -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
    if ($LASTEXITCODE -ne 0) { throw "torch install failed." }
    & $VenvPython -m pip install -r (Join-Path $BenchDir "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "requirements install failed." }
    & $VenvPython -c "import torch, funasr, onnxruntime, ai_edge_litert; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'funasr', funasr.__version__, 'ort', onnxruntime.__version__)"
    if (!$EvalArgs) { exit 0 }
}

if (!(Test-Path $VenvPython)) { throw "No venv. Run with -Bootstrap first." }

# HF token (gitignored .env) for hub downloads; never printed.
$envFile = Join-Path $ProjectRoot ".env"
if (Test-Path $envFile) {
    $line = Get-Content $envFile | Where-Object { $_ -match '^\s*HF_TOKEN\s*=' } | Select-Object -First 1
    if ($line) { $env:HF_TOKEN = ($line -split '=', 2)[1].Trim().Trim('"') }
}
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Push-Location $ProjectRoot
try {
    & $VenvPython (Join-Path $BenchDir "sv_eval.py") @EvalArgs
    exit $LASTEXITCODE
}
finally { Pop-Location }
