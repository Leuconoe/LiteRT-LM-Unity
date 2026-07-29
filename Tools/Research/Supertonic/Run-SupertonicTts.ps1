<#
.SYNOPSIS
Synthesize Korean/English speech with Supertonic on Windows, and optionally check
the result by transcribing it back.

.DESCRIPTION
Drives Tools/Windows/TtsBench/supertonic_tts.py (sherpa-onnx + the Supertonic
ONNX package in Assets/StreamingAssets/TTS/supertonic-int8).

-RoundTrip feeds the synthesized WAV to Run-WhisperTfliteWindows.ps1 and prints
the transcript next to the input text. That is the objective quality gate for a
voice nobody on the team can evaluate by ear alone: if our own ASR reads it back
correctly, the audio is intelligible.

.EXAMPLE
  .\Tools\Windows\Run-SupertonicTts.ps1 -Text "고도 백 미터로 상승합니다." -RoundTrip

.EXAMPLE
  .\Tools\Windows\Run-SupertonicTts.ps1 -Preset -RoundTrip
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Text = "",
    [string]$ModelDir = "",
    [string]$Out = "",
    [ValidateSet("ko", "en")]
    [string]$Lang = "ko",
    [double]$Speed = 1.0,
    [int]$Threads = 4,
    [switch]$Preset,
    [switch]$RoundTrip,
    [switch]$Warmup,
    [string]$Python = "",
    # Judge model for -RoundTrip. Default is the accuracy-best tier on this
    # project's device ledger, so a misread points at the synthesis rather than
    # at a weak ASR. whisper-base i8 is faster but loses words on its own.
    [string]$AsrModel = "Assets\StreamingAssets\ASR\whisper-turbo-acft-ko\acft_turbo_5s_drq.tflite"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$Runner = Join-Path $PSScriptRoot "TtsBench\supertonic_tts.py"
if (!(Test-Path $Runner)) { throw "Runner not found: $Runner" }

$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if (!$ModelDir) { $ModelDir = Join-Path $ProjectRoot "Assets\StreamingAssets\TTS\supertonic-int8" }
if (!(Test-Path $ModelDir)) { throw "Model directory not found: $ModelDir" }

function Test-PythonDeps([string]$exe) {
    if ([string]::IsNullOrWhiteSpace($exe) -or !(Test-Path $exe)) { return $false }
    & $exe -c "import sherpa_onnx, soundfile, numpy" 2>&1 | Out-Null
    return $LASTEXITCODE -eq 0
}

$candidates = @()
if ($Python) { $candidates += $Python }
if ($env:LITERTLM_PYTHON) { $candidates += $env:LITERTLM_PYTHON }
$candidates += (Join-Path $PSScriptRoot "..\Whisper\WhisperTflite\.venv\Scripts\python.exe")
$candidates += (Join-Path $ProjectRoot "External\acft-training\.venv\Scripts\python.exe")
$py = $candidates | Where-Object { Test-PythonDeps $_ } | Select-Object -First 1
if (!$py) {
    Write-Host "No Python with sherpa-onnx found. Install it with:" -ForegroundColor Yellow
    Write-Host "  .\Tools\Windows\WhisperTflite\.venv\Scripts\python.exe -m pip install sherpa-onnx"
    exit 2
}
Write-Host "Python: $py"

# The Korean lines double as the sample-scene presets, so the bench and the
# scene speak the same sentences.
$PresetLines = if ($Lang -eq "ko") {
    @(
        "고도 백 미터로 상승합니다.",
        "배터리 잔량 칠십 퍼센트.",
        "임무 지점에 도착했습니다. 촬영을 시작합니다.",
        "경고. 강풍이 감지되었습니다. 고도를 낮춥니다.",
        "귀환을 시작합니다. 예상 소요 시간 삼 분."
    )
}
else {
    @(
        "Ascending to one hundred meters.",
        "Battery at seventy percent.",
        "Arrived at the waypoint. Starting capture.",
        "Warning. High wind detected. Reducing altitude.",
        "Returning to base. Three minutes remaining."
    )
}

$lines = if ($Preset) { $PresetLines } elseif ($Text) { @($Text) } else { @($PresetLines[0]) }

$logDir = Join-Path $ProjectRoot "Builds\Logs\TtsSupertonic"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$jsonl = Join-Path $logDir "supertonic-runs.jsonl"

$index = 0
$failures = 0
foreach ($line in $lines) {
    $index++
    $wav = if ($Out -and $lines.Count -eq 1) { $Out } else { Join-Path $logDir ("supertonic-$Lang-{0:d2}.wav" -f $index) }

    $arguments = @(
        $Runner, "--model-dir", $ModelDir, "--text", $line, "--out", $wav,
        "--lang", $Lang, "--speed", $Speed, "--threads", $Threads
    )
    if ($Warmup) { $arguments += "--warmup" }

    $output = & $py @arguments
    if ($LASTEXITCODE -ne 0 -or !$output) {
        Write-Host "FAIL  $line" -ForegroundColor Red
        $failures++
        continue
    }

    Add-Content -Path $jsonl -Value $output -Encoding utf8
    $r = $output | ConvertFrom-Json
    "{0,5:N2}s audio  {1,6:N3}s synth  RTF {2,6:N3}  {3}Hz  {4}" -f `
        $r.audio_s, $r.seconds, $r.rtf, $r.sample_rate, $line | Write-Host

    if ($RoundTrip) {
        # The ASR wrapper reports through Write-Host, so merge every stream before
        # parsing — piping stdout alone yields nothing.
        $asrModelPath = if ([IO.Path]::IsPathRooted($AsrModel)) { $AsrModel } else { Join-Path $ProjectRoot $AsrModel }
        $asr = (& (Join-Path $PSScriptRoot "..\Whisper\Run-WhisperTfliteWindows.ps1") `
            -Model $asrModelPath -Audio $wav -Lang $Lang) *>&1 | Out-String
        $heard = ""
        foreach ($asrLine in ($asr -split "`r?`n")) {
            if ($asrLine -match "dec\s+[\d\.]+s\s+(.+)$") { $heard = $Matches[1].Trim() }
        }
        Write-Host "      heard: $heard" -ForegroundColor Cyan
    }
}

Write-Host ""
Write-Host "Runs: $jsonl"
if ($failures -gt 0) { Write-Host "$failures of $($lines.Count) failed." -ForegroundColor Red; exit 1 }
Write-Host "$($lines.Count) synthesis run(s) OK." -ForegroundColor Green
