<#
.SYNOPSIS
Assemble the Supertonic-on-LiteRT bucket ladder and stage it into StreamingAssets.

.DESCRIPTION
Collects the pieces the conversion and quantization work settled on into the
layout the Unity runtime expects:

  Assets/StreamingAssets/TTS/supertonic-litert/
    dynamic/{duration_predictor,text_encoder}/*.tflite   fp32, dynamic shapes
    st-b64/{vector_estimator,vocoder}/*.tflite           weight-only int8, fixed
    st-b128/…                                            "
    st-b256/…                                            "
    assets/{tts.json,unicode_indexer.json,<voice>.json}

Why this shape:
  * `duration_predictor` and `text_encoder` do not survive the fixed-shape
    rewrite, and they are the cheap half (10 ms + 109 ms), so they stay dynamic.
  * `vector_estimator` and `vocoder` are the expensive half and do convert, which
    is what lets XNNPACK attach — 6.6x end to end.
  * **weight_only_wi8_afp32**, not dynamic i8: quantizing the vocoder's
    activations drops mel correlation to 0.694, worse than halving the flow steps.
    Weight-only keeps 0.983 at the same 3.9x saving.
  * Buckets exist because fixed shapes need the text padded to a converted size;
    the runtime picks the smallest that fits.

.EXAMPLE
  .\Tools\Windows\Deploy-SupertonicLiteRt.ps1
  .\Tools\Windows\Deploy-SupertonicLiteRt.ps1 -Verify
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$WorkRoot = "",
    [string]$Destination = "",
    [int[]]$Buckets = @(64, 128, 256),
    [string]$Voice = "F1",
    [switch]$Verify
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if (!$WorkRoot) { $WorkRoot = Join-Path $ProjectRoot "External\tts-work" }
if (!$Destination) { $Destination = Join-Path $ProjectRoot "Assets\StreamingAssets\TTS\supertonic-litert" }
$AssetsSource = Join-Path $WorkRoot "supertonic-2-fp32"
$DynamicSource = Join-Path $WorkRoot "supertonic-tflite"

function Copy-Graph($sourceFile, $targetDir) {
    if (!(Test-Path $sourceFile)) { throw "missing: $sourceFile" }
    New-Item -ItemType Directory -Force $targetDir | Out-Null
    Copy-Item $sourceFile (Join-Path $targetDir (Split-Path $sourceFile -Leaf)) -Force
}

# Dynamic half.
foreach ($stem in @("duration_predictor", "text_encoder")) {
    Copy-Graph (Join-Path $DynamicSource "$stem\${stem}_float32.tflite") `
               (Join-Path $Destination "dynamic\$stem")
}

# Bucketed half, weight-only int8.
foreach ($bucket in $Buckets) {
    foreach ($stem in @("vector_estimator", "vocoder")) {
        Copy-Graph (Join-Path $WorkRoot "st-b$bucket-w8\${stem}_w8.tflite") `
                   (Join-Path $Destination "st-b$bucket\$stem")
    }
}

# Front-end tables and the voice style.
$assetDir = Join-Path $Destination "assets"
New-Item -ItemType Directory -Force $assetDir | Out-Null
foreach ($name in @("tts.json", "unicode_indexer.json", "$Voice.json", "LICENSE")) {
    $source = Join-Path $AssetsSource $name
    if (Test-Path $source) { Copy-Item $source (Join-Path $assetDir $name) -Force }
    elseif ($name -ne "LICENSE") { throw "missing asset: $source" }
}

$total = (Get-ChildItem $Destination -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Write-Host "Staged to $Destination"
Get-ChildItem $Destination -Recurse -Filter *.tflite |
    Sort-Object FullName |
    ForEach-Object {
        "{0,-56} {1,7:N1} MB" -f ($_.FullName.Substring($Destination.Length + 1)), ($_.Length / 1MB)
    }
"{0,-56} {1,7:N1} MB" -f "TOTAL", $total

if ($Verify) {
    Write-Host ""
    Write-Host "=== verifying from StreamingAssets ==="
    $venvPython = Join-Path $PSScriptRoot "TtsBench\.venv-convert\Scripts\python.exe"
    if (!(Test-Path $venvPython)) { throw "conversion venv missing" }
    $env:PYTHONIOENCODING = "utf-8"
    $env:TF_CPP_MIN_LOG_LEVEL = "2"

    $logDir = Join-Path $ProjectRoot "Builds\Logs\TtsSupertonic"
    New-Item -ItemType Directory -Force $logDir | Out-Null

    $sentences = @(
        "고도 백 미터로 상승합니다.",
        "고도 백 미터로 상승합니다. 배터리 잔량 칠십 퍼센트.",
        "경고. 강풍이 감지되었습니다. 고도를 낮춥니다. 귀환을 시작합니다. 예상 소요 시간 삼 분."
    )
    $index = 0
    foreach ($sentence in $sentences) {
        $index++
        $wav = Join-Path $logDir "deployed-$index.wav"
        $line = & $venvPython (Join-Path $PSScriptRoot "TtsBench\supertonic_litert.py") `
            --tflite-dir (Join-Path $Destination "dynamic") `
            --bucket-root $Destination `
            --assets-dir (Join-Path $Destination "assets") `
            --voice (Join-Path $Destination "assets\$Voice.json") `
            --text $sentence --out $wav --steps 4 2>$null
        if (!$line) { Write-Host "FAILED  $sentence" -ForegroundColor Red; continue }
        $r = $line | ConvertFrom-Json

        $asr = (& (Join-Path $PSScriptRoot "..\Whisper\Run-WhisperTfliteWindows.ps1") `
            -Model (Join-Path $ProjectRoot "Assets\StreamingAssets\ASR\whisper-base\whisper_base_30s_i8.tflite") `
            -Audio $wav -Lang ko) *>&1 | Out-String
        $heard = ""
        foreach ($asrLine in ($asr -split "`r?`n")) {
            if ($asrLine -match "dec\s+[\d\.]+s\s+(.+)$") { $heard = $Matches[1].Trim() }
        }
        "bucket {0,4}  RTF {1,6:N3}  audio {2,5:N2}s  synth {3,5:N2}s" -f $r.bucket, $r.rtf, $r.audio_s, $r.synth_s
        "   heard: $heard"
    }
}
