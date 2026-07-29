<#
.SYNOPSIS
Run the Supertonic-on-LiteRT TTS smoke test on the physical Android device and
bring the audio back for judging.

.DESCRIPTION
Installs the smoke APK, pushes the runtime config, starts the app, waits for the
runner's status file, then pulls the synthesized WAVs and transcribes them on the
desktop with the accuracy-best ASR tier. That last step matters: a device run that
merely does not crash proves nothing about the audio, and nobody can judge Korean
synthesis by reading a log.

Models come from the APK's StreamingAssets (staged by
Tools/Research/Supertonic/Deploy-SupertonicLiteRt.ps1), so unlike the ASR smoke test there is nothing to
push except the config — 202 MB over adb per run would be wasteful.

.EXAMPLE
  .\Tools\Windows\Run-LiteRtLmAndroidTtsSmokeTest.ps1
.EXAMPLE
  .\Tools\Windows\Run-LiteRtLmAndroidTtsSmokeTest.ps1 -Steps 8 -Backend CPU -ClearAppData
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$DeviceSerial = "46a880a0",
    [string]$PackageName = "com.Leuconoe.LiteRTLMUnity",
    [string]$ApkPath = "",
    [string]$Root = "TTS/supertonic-litert",
    [string]$Voice = "F1",
    [int]$Steps = 4,
    [double]$Speed = 1.05,
    [string]$Backend = "CPU",
    [int]$Seed = 1234,
    [string]$Language = "ko",
    [int]$RunsPerSentence = 2,
    [string[]]$Sentences = @(),
    [int]$TimeoutSeconds = 900,
    [switch]$ClearAppData,
    [switch]$SkipTranscribe,
    [string]$AsrModel = "Assets\StreamingAssets\ASR\whisper-base\whisper_base_30s_i8.tflite"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$logDirectory = Join-Path $ProjectRoot "Builds\Logs\AndroidTtsSmoke"
New-Item -ItemType Directory -Force $logDirectory | Out-Null
$runId = Get-Date -Format "yyyyMMdd-HHmmss"
$statusLog = Join-Path $logDirectory "$runId-status.txt"
$wavDirectory = Join-Path $logDirectory "$runId-wav"

$deviceFiles = "/sdcard/Android/data/$PackageName/files"
$deviceStatusPath = "$deviceFiles/LiteRtLmTtsSmokeTest.status.txt"
$deviceConfigPath = "$deviceFiles/LiteRtLmTtsSmokeTest.config.json"
$deviceWavDirectory = "$deviceFiles/LiteRTLM/TtsSmoke"

function Invoke-Adb([string[]]$Arguments) {
    & adb -s $DeviceSerial @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "adb failed with exit code ${LASTEXITCODE}: adb -s $DeviceSerial $($Arguments -join ' ')"
    }
}

function Invoke-AdbBestEffort([string[]]$Arguments) {
    & adb -s $DeviceSerial @Arguments 2>&1 | Out-Null
}

$devices = @(adb devices -l | Select-Object -Skip 1 | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
if (-not ($devices | Where-Object { $_ -match "^$([regex]::Escape($DeviceSerial))\s+device\s" })) {
    throw "Android device is not connected or not authorized: $DeviceSerial`nCurrent devices:`n$($devices -join [Environment]::NewLine)"
}

if ([string]::IsNullOrWhiteSpace($ApkPath)) {
    $candidate = Get-ChildItem (Join-Path $ProjectRoot "Builds\AndroidBuilds") -Filter "*.apk" -Recurse -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (!$candidate) {
        throw "No APK found under Builds\AndroidBuilds. Build one first (LiteRT-LM/Build/Android/...) or pass -ApkPath."
    }
    $ApkPath = $candidate.FullName
}
Write-Host "[LiteRT-LM] APK: $ApkPath"
Invoke-Adb @("install", "-r", "-d", "-t", $ApkPath)

if ($ClearAppData) {
    Write-Host "[LiteRT-LM] Clearing app data: $PackageName"
    Invoke-AdbBestEffort @("shell", "pm", "clear", $PackageName)
}

# Runtime config, so a sweep does not need an APK rebuild.
$config = [ordered]@{
    root = $Root
    voice = $Voice
    steps = [Math]::Max(1, $Steps)
    speed = $Speed
    backend = $Backend
    seed = $Seed
    language = $Language
    runsPerSentence = [Math]::Max(1, $RunsPerSentence)
}
if ($Sentences.Count -gt 0) { $config.sentences = $Sentences }

$configPath = Join-Path $logDirectory "$runId-config.json"
($config | ConvertTo-Json -Depth 4) | Set-Content -Path $configPath -Encoding utf8
Invoke-AdbBestEffort @("shell", "mkdir", "-p", $deviceFiles)
Invoke-Adb @("push", $configPath, $deviceConfigPath)
Write-Host "[LiteRT-LM] Config pushed: steps=$Steps, backend=$Backend, runsPerSentence=$RunsPerSentence"

Invoke-AdbBestEffort @("shell", "am", "force-stop", $PackageName)
Invoke-AdbBestEffort @("shell", "rm", "-f", $deviceStatusPath)
Invoke-AdbBestEffort @("shell", "rm", "-rf", $deviceWavDirectory)
Invoke-Adb @("logcat", "-c")

# Unity pauses when the activity loses focus, and the runner then produces no
# status file at all — indistinguishable from a hang. A pulled-down notification
# shade is enough to cause it, so the screen is woken, unlocked and the shade
# collapsed before launching.
Invoke-AdbBestEffort @("shell", "input", "keyevent", "KEYCODE_WAKEUP")
Invoke-AdbBestEffort @("shell", "wm", "dismiss-keyguard")
Invoke-AdbBestEffort @("shell", "cmd", "statusbar", "collapse")
Invoke-Adb @("shell", "monkey", "-p", $PackageName, "-c", "android.intent.category.LAUNCHER", "1")

# Confirm the app really has focus; without it the wait below would just time out.
Start-Sleep -Seconds 5
$focusLines = & adb -s $DeviceSerial shell dumpsys window 2>$null | Select-String -Pattern "mCurrentFocus"
$focus = ($focusLines | Select-Object -First 1).ToString()
if ($focus -notmatch [regex]::Escape($PackageName)) {
    Write-Host "[LiteRT-LM] App does not have focus yet ($($focus.Trim())); retrying." -ForegroundColor Yellow
    Invoke-AdbBestEffort @("shell", "cmd", "statusbar", "collapse")
    Invoke-AdbBestEffort @("shell", "am", "start", "-n", "$PackageName/com.unity3d.player.UnityPlayerActivity")
}

Write-Host "[LiteRT-LM] Waiting for the runner (timeout ${TimeoutSeconds}s)..."
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$statusText = ""
$verdict = "TIMEOUT"
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 3
    $lines = & adb -s $DeviceSerial shell cat $deviceStatusPath 2>$null
    if ($LASTEXITCODE -ne 0) { continue }
    $statusText = (@($lines) -join [Environment]::NewLine)
    if ([string]::IsNullOrWhiteSpace($statusText)) { continue }
    $statusText | Out-File -FilePath $statusLog -Encoding utf8
    if ($statusText -match "(?m)\]\s+(SUCCESS|FAILURE|SKIP):") {
        $verdict = $Matches[1]
        break
    }
}

if ([string]::IsNullOrWhiteSpace($statusText)) {
    $focusNow = (& adb -s $DeviceSerial shell dumpsys window 2>$null |
        Select-String -Pattern "mCurrentFocus" | Select-Object -First 1)
    Write-Host "[LiteRT-LM] No status file appeared. Focus: $($focusNow.ToString().Trim())" -ForegroundColor Yellow
    Write-Host "[LiteRT-LM] If that is not this app, Unity is paused — the runner never started." -ForegroundColor Yellow
    Write-Host "[LiteRT-LM] Recent logcat:" -ForegroundColor Yellow
    & adb -s $DeviceSerial logcat -d -t 200 2>$null |
        Select-String -Pattern "LiteRT-LM|Unity|tombstone|SIGSEGV|FATAL" |
        Select-Object -Last 30 | ForEach-Object { Write-Host "  $($_.Line)" }
    throw "TTS smoke test produced no status output."
}

Write-Host ""
Write-Host "=== device status ($verdict) ==="
foreach ($line in ($statusText -split "`r?`n")) {
    if ($line -match "\]\s+(RESULT|CHECKSUM|SENTENCE|SUCCESS|FAILURE|FAIL|SKIP|CONFIG):") { Write-Host "  $line" }
}
Write-Host "Full status: $statusLog"

# Pull the audio: the numbers say it ran, the audio says whether it worked.
New-Item -ItemType Directory -Force $wavDirectory | Out-Null
Invoke-AdbBestEffort @("pull", $deviceWavDirectory, $wavDirectory)
$wavs = @(Get-ChildItem $wavDirectory -Filter "*.wav" -Recurse -ErrorAction SilentlyContinue | Sort-Object Name)
Write-Host ""
Write-Host "Pulled $($wavs.Count) WAV file(s) into $wavDirectory"

if ($wavs.Count -gt 0 -and -not $SkipTranscribe) {
    Write-Host ""
    Write-Host "=== round-trip transcription (desktop ASR) ==="
    foreach ($wav in $wavs) {
        $asr = (& (Join-Path $PSScriptRoot "..\Research\Whisper\Run-WhisperTfliteWindows.ps1") `
            -Model (Join-Path $ProjectRoot $AsrModel) -Audio $wav.FullName -Lang $Language) *>&1 | Out-String
        $heard = ""
        foreach ($asrLine in ($asr -split "`r?`n")) {
            if ($asrLine -match "dec\s+[\d\.]+s\s+(.+)$") { $heard = $Matches[1].Trim() }
        }
        "{0,-24} {1}" -f $wav.Name, $heard | Write-Host
    }
}

if ($verdict -ne "SUCCESS") {
    Write-Host ""
    Write-Host "Verdict: $verdict" -ForegroundColor Red
    exit 1
}
Write-Host ""
Write-Host "Verdict: SUCCESS" -ForegroundColor Green
