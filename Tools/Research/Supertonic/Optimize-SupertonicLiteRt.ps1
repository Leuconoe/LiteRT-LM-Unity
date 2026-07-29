<#
.SYNOPSIS
Find the cheapest Supertonic-on-LiteRT setting that still speaks clearly.

.DESCRIPTION
`vector_estimator` runs once per flow-matching step and is ~78 % of synthesis
time, so the step count is the one setting that changes RTF proportionally.
Quality is judged by round-trip ASR with the accuracy-best tier rather than by
ear: synthesize, transcribe, compare to the input text.

Reports RTF and the transcript per step count so the trade is visible instead of
guessed.

.EXAMPLE
  .\Tools\Windows\Optimize-SupertonicLiteRt.ps1
  .\Tools\Windows\Optimize-SupertonicLiteRt.ps1 -Steps 2,4 -Text "귀환을 시작합니다."
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [int[]]$Steps = @(2, 3, 4, 6, 8),
    [string]$Text = "고도 백 미터로 상승합니다. 배터리 잔량 칠십 퍼센트.",
    [string]$Lang = "ko",
    [string]$TfliteDir = "",
    [string]$AssetsDir = "",
    [string]$Voice = "",
    [int]$Threads = 4,
    [string]$AsrModel = "Assets\StreamingAssets\ASR\whisper-turbo-acft-ko\acft_turbo_5s_drq.tflite"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$BenchDir = Join-Path $PSScriptRoot "TtsBench"
$VenvPython = Join-Path $BenchDir ".venv-convert\Scripts\python.exe"
if (!(Test-Path $VenvPython)) { throw "Conversion venv missing; run Convert-SupertonicToLiteRt.ps1 -Bootstrap." }

$env:PYTHONIOENCODING = "utf-8"
$env:TF_CPP_MIN_LOG_LEVEL = "2"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if (!$AssetsDir) { $AssetsDir = Join-Path $ProjectRoot "External\tts-work\supertonic-2-fp32" }
if (!$TfliteDir) { $TfliteDir = Join-Path $ProjectRoot "External\tts-work\supertonic-tflite" }
if (!$Voice) { $Voice = Join-Path $AssetsDir "F1.json" }

$logDir = Join-Path $ProjectRoot "Builds\Logs\TtsSupertonic"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$jsonl = Join-Path $logDir "litert-step-sweep.jsonl"
Set-Content -Path $jsonl -Value "" -NoNewline -Encoding utf8

Write-Host "text : $Text"
Write-Host ""

foreach ($step in $Steps) {
    $wav = Join-Path $logDir ("litert-steps{0:d2}.wav" -f $step)
    $line = & $VenvPython (Join-Path $BenchDir "supertonic_litert.py") `
        --tflite-dir $TfliteDir --assets-dir $AssetsDir --voice $Voice `
        --text $Text --out $wav --lang $Lang --steps $step --threads $Threads 2>$null
    if ($LASTEXITCODE -ne 0 -or !$line) {
        Write-Host ("steps {0,2}  FAILED" -f $step) -ForegroundColor Red
        continue
    }
    Add-Content -Path $jsonl -Value $line -Encoding utf8
    $r = $line | ConvertFrom-Json

    $asr = (& (Join-Path $PSScriptRoot "..\Whisper\Run-WhisperTfliteWindows.ps1") `
        -Model (Join-Path $ProjectRoot $AsrModel) -Audio $wav -Lang $Lang) *>&1 | Out-String
    $heard = ""
    foreach ($asrLine in ($asr -split "`r?`n")) {
        if ($asrLine -match "dec\s+[\d\.]+s\s+(.+)$") { $heard = $Matches[1].Trim() }
    }

    Write-Host ("steps {0,2}  RTF {1,6:N3}  synth {2,6:N2}s  audio {3,5:N2}s  ve/step {4,6:N3}s" -f `
        $step, $r.rtf, $r.synth_s, $r.audio_s, $r.vector_estimator_per_step_s)
    Write-Host ("           heard: {0}" -f $heard) -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Runs: $jsonl"
