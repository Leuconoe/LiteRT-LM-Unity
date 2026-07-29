<#
.SYNOPSIS
Convert Supertonic TTS from ONNX to tflite and run it on LiteRT.

.DESCRIPTION
End goal: serve TTS from LiteRT, the runtime this project already ships for the
LLM and ASR paths, instead of adding onnxruntime as a third runtime.

Pipeline:
  1. fp32 ONNX (Supertone/supertonic-2)  →  onnx2tf  →  tflite fp32/fp16,
     each graph verified against onnxruntime during conversion (-cotof).
  2. Run the four graphs on LiteRT with the MIT reference pipeline
     (Tools/Windows/TtsBench/supertonic_litert.py).
  3. Compare against the onnxruntime baseline from Run-SupertonicTts.ps1 —
     same text, same seed — on RTF and on round-trip ASR.

Toolchain lives in its own venv (Tools/Windows/TtsBench/.venv-convert, Python
3.12: TF and onnx2tf are heavy and must not disturb the ASR bench venv).
-Bootstrap creates it.

.EXAMPLE
  .\Tools\Windows\Convert-SupertonicToLiteRt.ps1 -Bootstrap -Convert
  .\Tools\Windows\Convert-SupertonicToLiteRt.ps1 -Run -Text "고도 백 미터로 상승합니다."
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$OnnxDir = "",
    [string]$TfliteDir = "",
    [string]$Voice = "",
    [string]$Text = "고도 백 미터로 상승합니다.",
    [string]$Lang = "ko",
    [int]$Steps = 8,
    [int]$Threads = 4,
    [switch]$Bootstrap,
    [switch]$Describe,
    [switch]$Convert,
    [switch]$Run,
    [switch]$RoundTrip
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$BenchDir = Join-Path $PSScriptRoot "TtsBench"
$VenvPython = Join-Path $BenchDir ".venv-convert\Scripts\python.exe"

$env:PYTHONIOENCODING = "utf-8"
$env:TF_ENABLE_ONEDNN_OPTS = "0"
$env:TF_CPP_MIN_LOG_LEVEL = "2"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if (!$OnnxDir) { $OnnxDir = Join-Path $ProjectRoot "External\tts-work\supertonic-2-fp32" }
if (!$TfliteDir) { $TfliteDir = Join-Path $ProjectRoot "External\tts-work\supertonic-tflite" }
if (!$Voice) { $Voice = Join-Path $OnnxDir "F1.json" }

function Invoke-Bootstrap {
    # onnx2tf needs TF, which has no cp314 wheel; 3.12 is the verified line here.
    $base = "C:\Users\user\AppData\Roaming\uv\python\cpython-3.12.13-windows-x86_64-none\python.exe"
    if (!(Test-Path $base)) {
        foreach ($v in @("3.12", "3.13", "3.11")) {
            & py "-$v" -c "import sys" 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) { $base = "py"; $baseArgs = @("-$v"); break }
        }
    }
    if (!$base) { throw "No Python 3.11-3.13 found for the conversion venv." }

    $venvDir = Join-Path $BenchDir ".venv-convert"
    if (!(Test-Path $VenvPython)) {
        if ($baseArgs) { & $base @($baseArgs + @("-m", "venv", $venvDir)) } else { & $base -m venv $venvDir }
        if ($LASTEXITCODE -ne 0) { throw "venv creation failed." }
    }
    & $VenvPython -m pip install --upgrade pip | Out-Null
    & $VenvPython -m pip install onnx onnxruntime onnx2tf "tensorflow>=2.18,<2.21" ai-edge-litert `
        onnx-graphsurgeon sng4onnx simple_onnx_processing_tools psutil soundfile
    if ($LASTEXITCODE -ne 0) { throw "conversion toolchain install failed." }
}

if ($Bootstrap) { Invoke-Bootstrap }
if (!(Test-Path $VenvPython)) {
    Write-Host "Conversion venv missing. Run with -Bootstrap first." -ForegroundColor Yellow
    exit 2
}

if (!(Test-Path $OnnxDir)) {
    Write-Host "fp32 ONNX not found: $OnnxDir" -ForegroundColor Yellow
    Write-Host "Fetch it from https://huggingface.co/Supertone-2 (onnx/ + voice_styles/)."
    exit 2
}

if ($Describe) {
    & $VenvPython (Join-Path $BenchDir "convert_supertonic_to_tflite.py") `
        --onnx-dir $OnnxDir --out-dir $TfliteDir --describe-only
}

if ($Convert) {
    & $VenvPython (Join-Path $BenchDir "convert_supertonic_to_tflite.py") `
        --onnx-dir $OnnxDir --out-dir $TfliteDir --keep-going
    if ($LASTEXITCODE -ne 0) { Write-Host "Conversion reported failures." -ForegroundColor Red }
}

if ($Run) {
    $logDir = Join-Path $ProjectRoot "Builds\Logs\TtsSupertonic"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $wav = Join-Path $logDir "litert-$Lang.wav"

    $line = & $VenvPython (Join-Path $BenchDir "supertonic_litert.py") `
        --tflite-dir $TfliteDir --assets-dir $OnnxDir --voice $Voice `
        --text $Text --out $wav --lang $Lang --steps $Steps --threads $Threads
    if ($LASTEXITCODE -ne 0 -or !$line) { Write-Host "LiteRT run failed." -ForegroundColor Red; exit 1 }

    Add-Content -Path (Join-Path $logDir "litert-runs.jsonl") -Value $line -Encoding utf8
    $r = $line | ConvertFrom-Json
    "{0,5:N2}s audio  {1,6:N3}s synth  RTF {2,6:N3}  (dp {3:N3} / enc {4:N3} / vec {5:N3} = {6:N3}/step / voc {7:N3})" -f `
        $r.audio_s, $r.synth_s, $r.rtf, $r.duration_s, $r.text_encoder_s, `
        $r.vector_estimator_s, $r.vector_estimator_per_step_s, $r.vocoder_s | Write-Host

    if ($RoundTrip) {
        $asr = (& (Join-Path $PSScriptRoot "..\Whisper\Run-WhisperTfliteWindows.ps1") `
            -Model (Join-Path $ProjectRoot "Assets\StreamingAssets\ASR\whisper-turbo-acft-ko\acft_turbo_5s_drq.tflite") `
            -Audio $wav -Lang $Lang) *>&1 | Out-String
        $heard = ""
        foreach ($asrLine in ($asr -split "`r?`n")) {
            if ($asrLine -match "dec\s+[\d\.]+s\s+(.+)$") { $heard = $Matches[1].Trim() }
        }
        Write-Host "      said : $Text"
        Write-Host "      heard: $heard" -ForegroundColor Cyan
    }
}

if (!$Bootstrap -and !$Describe -and !$Convert -and !$Run) {
    Write-Host "Nothing to do. Pass -Bootstrap, -Describe, -Convert or -Run."
}
