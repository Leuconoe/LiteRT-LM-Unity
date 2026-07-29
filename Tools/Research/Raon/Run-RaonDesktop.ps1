<#
.SYNOPSIS
  Runs KRAFTON Raon-Speech on this workstation's GPU and measures STT and TTS.

.DESCRIPTION
  Raon is the only candidate evaluated here that does speech recognition and
  speech synthesis in one model. It has no on-device path — 9 B parameters,
  CUDA-only AWQ kernels, and a custom `RaonModel` architecture with no LiteRT
  conversion route — so this exists to answer the PC half of the question:
  does it run here, how fast, and is the Korean good enough to justify a
  companion machine.

  Requires a CUDA GPU with roughly 8 GB free for the AWQ-INT4 build.

  Licence: CC-BY-NC-4.0. Evaluation only — this cannot ship in a delivered
  product. See docs/tts-model-research.md.

.PARAMETER Bootstrap
  Create the venv and install torch (CUDA 12.8) plus the model's requirements.

.EXAMPLE
  .\Tools\Research\Raon\Run-RaonDesktop.ps1 -Bootstrap
  .\Tools\Research\Raon\Run-RaonDesktop.ps1
  .\Tools\Research\Raon\Run-RaonDesktop.ps1 -Model KRAFTON/Raon-Speech-9B -Dtype bfloat16
#>
[CmdletBinding()]
param(
    [string]$Model = "KRAFTON/Raon-Speech-9B-AWQ-INT4",
    [string]$Device = "cuda",
    [string]$Dtype = "bfloat16",
    [string]$SpeakerAudio = "",
    [string]$OutDir = "",
    [switch]$SkipStt,
    [switch]$SkipTts,
    [switch]$Bootstrap
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$BenchDir = Join-Path $PSScriptRoot "RaonBench"
$VenvPython = Join-Path $BenchDir ".venv\Scripts\python.exe"

if ($Bootstrap) {
    if (-not (Test-Path $VenvPython)) {
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
            throw "No Python 3.12/3.13 found. 3.14 is the py default here and has no wheels for this stack."
        }
        Write-Host "Creating venv with $interpreter" -ForegroundColor Cyan
        & $interpreter -m venv (Join-Path $BenchDir ".venv")
        if ($LASTEXITCODE -ne 0) { throw "venv creation failed." }
    }

    & $VenvPython -m pip install --upgrade pip
    Write-Host "Installing torch + torchaudio (CUDA 12.8)" -ForegroundColor Cyan
    & $VenvPython -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
    if ($LASTEXITCODE -ne 0) { throw "torch install failed." }
    & $VenvPython -m pip install -r (Join-Path $BenchDir "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "requirements install failed." }
    & $VenvPython -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"
}

if (-not (Test-Path $VenvPython)) {
    throw "No venv. Run with -Bootstrap first."
}

if (-not $OutDir) { $OutDir = Join-Path $ProjectRoot "Builds\Logs\RaonDesktop" }
New-Item -ItemType Directory -Force $OutDir | Out-Null

$arguments = @(
    (Join-Path $BenchDir "raon_desktop_smoke.py"),
    "--model", $Model,
    "--device", $Device,
    "--dtype", $Dtype,
    "--audio-root", (Join-Path $ProjectRoot "Assets\StreamingAssets\TestAssets\Audio"),
    "--out-dir", $OutDir
)
if ($SpeakerAudio) { $arguments += @("--speaker-audio", $SpeakerAudio) }
if ($SkipStt) { $arguments += "--skip-stt" }
if ($SkipTts) { $arguments += "--skip-tts" }

$env:PYTHONIOENCODING = "utf-8"
Write-Host "Running $Model on $Device" -ForegroundColor Cyan
& $VenvPython @arguments
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Host "Audio and records: $OutDir" -ForegroundColor Green
}
exit $exitCode
