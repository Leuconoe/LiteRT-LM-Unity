<#
.SYNOPSIS
Transcribe audio with a Whisper tflite export on Windows CPU.

.DESCRIPTION
Drives Tools/Windows/WhisperTflite/whisper_tflite_runner.py, which runs the mel
frontend, encoder and greedy KV decode through ai_edge_litert — the same
pipeline the Android AAR runs natively. Proves and keeps proving that Whisper
tflite works on desktop; litert_lm_main cannot do this (it is an LLM runner).

Resolves a Python interpreter in this order:
  -Python argument, $env:LITERTLM_PYTHON, the repo-local venv
  (Tools/Windows/WhisperTflite/.venv), External/acft-training/.venv, py -3,
  python. Missing dependencies are reported with the exact install command;
  pass -Bootstrap to create the repo-local venv and install them.

.EXAMPLE
  .\Tools\Windows\Run-WhisperTfliteWindows.ps1 -Audio "Assets\StreamingAssets\TestAssets\Audio\volume-소리 키워줘.mp3"

.EXAMPLE
  .\Tools\Windows\Run-WhisperTfliteWindows.ps1 -Sweep
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Model = "",
    [string]$Tokenizer = "",
    [string]$Audio = "",
    [ValidateSet("ko", "en")]
    [string]$Lang = "ko",
    [int]$Runs = 3,
    [switch]$Sweep,
    [switch]$All,
    [string]$Python = "",
    [string]$LogPath = "",
    [switch]$Bootstrap
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$RunnerDir = Join-Path $PSScriptRoot "WhisperTflite"
$Runner = Join-Path $RunnerDir "whisper_tflite_runner.py"
$Requirements = Join-Path $RunnerDir "requirements.txt"
$LocalVenvPython = Join-Path $RunnerDir ".venv\Scripts\python.exe"

if (!(Test-Path $Runner)) { throw "Runner not found: $Runner" }

# Korean paths and transcripts must survive the pipe in both directions.
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Test-PythonDeps([string]$exe) {
    if ([string]::IsNullOrWhiteSpace($exe) -or !(Test-Path $exe)) { return $false }
    & $exe -c "import numpy, soundfile, tokenizers, ai_edge_litert" 2>&1 | Out-Null
    return $LASTEXITCODE -eq 0
}

function Resolve-Python {
    $candidates = @()
    if ($Python) { $candidates += $Python }
    if ($env:LITERTLM_PYTHON) { $candidates += $env:LITERTLM_PYTHON }
    $candidates += $LocalVenvPython
    $candidates += (Join-Path $ProjectRoot "External\acft-training\.venv\Scripts\python.exe")
    foreach ($c in $candidates) {
        if (Test-PythonDeps $c) { return (Resolve-Path $c).Path }
    }
    foreach ($name in @("python", "py")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd -and (Test-PythonDeps $cmd.Source)) { return $cmd.Source }
    }
    return $null
}

function Get-BootstrapBase {
    # numpy 2.2 has no cp314 wheel and building it from source fails, so an
    # interpreter the launcher defaults to is not necessarily usable. Ask for a
    # version that has wheels for all four packages, newest verified first.
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in @("3.12", "3.13", "3.11", "3.10")) {
            & py "-$v" -c "import sys" 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                return @{ Exe = "py"; Args = @("-$v"); Label = "py -$v" }
            }
        }
    }
    foreach ($name in @("python", "py")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) {
            Write-Host "No 3.10-3.13 interpreter found; trying $($cmd.Source)." -ForegroundColor Yellow
            return @{ Exe = $cmd.Source; Args = @(); Label = $cmd.Source }
        }
    }
    throw "No Python on PATH; install Python 3.12 (3.10-3.13 all work)."
}

function Invoke-Bootstrap {
    $venvDir = Join-Path $RunnerDir ".venv"
    if ((Test-Path $venvDir) -and !(Test-PythonDeps $LocalVenvPython)) {
        Write-Host "Removing unusable venv: $venvDir"
        Remove-Item -Recurse -Force $venvDir
    }
    if (!(Test-Path $venvDir)) {
        $base = Get-BootstrapBase
        Write-Host "Creating venv: $venvDir  (base: $($base.Label))"
        & $base.Exe @($base.Args + @("-m", "venv", $venvDir))
        if ($LASTEXITCODE -ne 0) { throw "venv creation failed." }
    }
    & $LocalVenvPython -m pip install --upgrade pip | Out-Null
    & $LocalVenvPython -m pip install -r $Requirements
    if ($LASTEXITCODE -ne 0) {
        throw "dependency install failed — see the pip output above. A wheel is " +
              "probably missing for $(& $LocalVenvPython --version); recreate " +
              "the venv on Python 3.12."
    }
}

if ($Bootstrap) { Invoke-Bootstrap }

$py = Resolve-Python
if (!$py) {
    Write-Host "No Python with the required packages was found." -ForegroundColor Yellow
    Write-Host "Fix it with either:"
    Write-Host "  .\Tools\Windows\Run-WhisperTfliteWindows.ps1 -Bootstrap    # creates $RunnerDir\.venv"
    Write-Host "  <your-python> -m pip install -r `"$Requirements`"          # then pass -Python <exe>"
    exit 2
}
Write-Host "Python: $py"

# Default sweep: the fast, representative tiers. -All adds the multi-GB stock
# medium/large exports (minutes per clip, ~5 GB of reads).
$SweepModels = @(
    "ASR\whisper-base-acft-ko\acft_base_5s_drq.tflite",
    "ASR\whisper-medium-acft-ko\acft_medium_5s_drq.tflite",
    "ASR\whisper-turbo-acft-ko\acft_turbo_5s_drq.tflite",
    "ASR\whisper-base\whisper_base_30s_i8.tflite",
    "ASR\whisper-tiny\whisper_tiny_30s_i8.tflite"
)
if ($All) {
    $SweepModels += @(
        "ASR\whisper-base\whisper_base_30s_i4.tflite",
        "ASR\whisper-tiny\whisper_tiny_30s_i4.tflite",
        "ASR\whisper-medium\whisper_medium_30s_i8.tflite",
        "ASR\whisper-large-v3-turbo\whisper_large_v3_turbo_30s_i8.tflite",
        "ASR\whisper-large-v3\whisper_large_v3_30s_i8.tflite"
    )
}
$SweepAudio = @(
    "volume-소리 키워줘.mp3",
    "volume-볼륨 업.mp3",
    "현재 서울의 날씨는 흐림 입니다.mp3",
    "2025년 3월 5일 전술평가 결과 보고.mp3"
)

$StreamingAssets = Join-Path $ProjectRoot "Assets\StreamingAssets"
$AudioRoot = Join-Path $StreamingAssets "TestAssets\Audio"

$jobs = @()
if ($Sweep) {
    foreach ($m in $SweepModels) {
        $modelPath = Join-Path $StreamingAssets $m
        if (!(Test-Path $modelPath)) {
            Write-Host "skip (absent): $m" -ForegroundColor DarkYellow
            continue
        }
        foreach ($a in $SweepAudio) {
            $audioPath = Join-Path $AudioRoot $a
            if (!(Test-Path $audioPath)) {
                Write-Host "skip (absent): $a" -ForegroundColor DarkYellow
                continue
            }
            $jobs += , @($modelPath, $audioPath)
        }
    }
}
else {
    if (!$Model) { $Model = Join-Path $StreamingAssets "ASR\whisper-base-acft-ko\acft_base_5s_drq.tflite" }
    if (!$Audio) { $Audio = Join-Path $AudioRoot "volume-소리 키워줘.mp3" }
    if (!(Test-Path $Model)) { throw "Model not found: $Model" }
    if (!(Test-Path $Audio)) { throw "Audio not found: $Audio" }
    $jobs += , @((Resolve-Path $Model).Path, (Resolve-Path $Audio).Path)
}

if ($jobs.Count -eq 0) { throw "Nothing to run — no model/audio pair resolved." }

if (!$LogPath) {
    $logDir = Join-Path $ProjectRoot "Builds\Logs"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $suffix = if ($Sweep) { "sweep" } else { "run" }
    $LogPath = Join-Path $logDir "whisper-windows-tflite-$suffix.jsonl"
}
Set-Content -Path $LogPath -Value "" -NoNewline -Encoding utf8

$fail = 0
foreach ($job in $jobs) {
    $modelPath, $audioPath = $job
    # Tokenizer lives next to the model unless one was named explicitly.
    $tok = if ($Tokenizer) { $Tokenizer } else { Join-Path (Split-Path $modelPath) "tokenizer.json" }
    if (!(Test-Path $tok)) {
        Write-Host "FAIL  $(Split-Path $modelPath -Leaf)  — tokenizer missing: $tok" -ForegroundColor Red
        $fail++
        continue
    }
    $line = & $py $Runner --model $modelPath --tokenizer $tok --audio $audioPath --lang $Lang --runs $Runs 2>$null
    if ($LASTEXITCODE -ne 0 -or !$line) {
        Write-Host "FAIL  $(Split-Path $modelPath -Leaf)  <-  $(Split-Path $audioPath -Leaf)" -ForegroundColor Red
        $fail++
        continue
    }
    Add-Content -Path $LogPath -Value $line -Encoding utf8
    $r = $line | ConvertFrom-Json
    "{0,-38} {1,5}mel {2,5}f  {3,5:N2}s audio  enc {4,6:N3}s  dec {5,6:N3}s  {6}" -f `
        $r.model, $r.n_mels, $r.frames, $r.audio_s, $r.encode_s, $r.decode_s, $r.text | Write-Host
}

Write-Host ""
Write-Host "Results: $LogPath"
if ($fail -gt 0) {
    Write-Host "$fail of $($jobs.Count) run(s) failed." -ForegroundColor Red
    exit 1
}
Write-Host "$($jobs.Count) run(s) OK." -ForegroundColor Green
