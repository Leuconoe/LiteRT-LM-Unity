[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$DeviceSerial = "46a880a0",
    [string]$PackageName = "com.Leuconoe.LiteRTLMUnity",
    [string]$ApkPath = "",
    [int]$TimeoutSeconds = 300,
    [switch]$ClearAppData
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($ApkPath)) {
    $ApkPath = Join-Path $ProjectRoot "Builds\Android\LiteRtLmAndroidAsrSmokeTest-parakeet-tdt-0.6b-v3.apk"
}

if (!(Test-Path $ApkPath)) {
    throw "ASR smoke APK not found: $ApkPath"
}

$LogDirectory = Join-Path $ProjectRoot "Builds\Logs\AndroidDeviceRuns"
New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
$RunId = Get-Date -Format "yyyyMMdd-HHmmss"
$rawLog = Join-Path $LogDirectory "$RunId-asr-smoke.logcat.txt"
$summaryLog = Join-Path $LogDirectory "$RunId-asr-smoke.summary.txt"
$statusLog = Join-Path $LogDirectory "$RunId-asr-smoke.status.txt"
$deviceStatusPath = "/sdcard/Android/data/$PackageName/files/LiteRtLmAsrSmokeTest.status.txt"

function Invoke-Adb {
    param([string[]]$Arguments)
    & adb -s $DeviceSerial @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "adb failed with exit code ${LASTEXITCODE}: adb -s $DeviceSerial $($Arguments -join ' ')"
    }
}

function Invoke-AdbBestEffort {
    param([string[]]$Arguments)
    & adb -s $DeviceSerial @Arguments | Out-Null
}

function Get-DeviceStatusText {
    $lines = & adb -s $DeviceSerial shell cat $deviceStatusPath 2>$null
    if ($LASTEXITCODE -ne 0) {
        return ""
    }

    return (@($lines) -join [Environment]::NewLine)
}

$devices = @(adb devices -l | Select-Object -Skip 1 | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
if (-not ($devices | Where-Object { $_ -match "^$([regex]::Escape($DeviceSerial))\s+device\s" })) {
    throw "Android device is not connected or not authorized: $DeviceSerial`nCurrent devices:`n$($devices -join [Environment]::NewLine)"
}

Write-Host "[LiteRT-LM] Installing ASR smoke APK: $ApkPath"
Invoke-Adb @("install", "-r", "-d", "-t", $ApkPath)

if ($ClearAppData) {
    Write-Host "[LiteRT-LM] Clearing app data: $PackageName"
    Invoke-AdbBestEffort @("shell", "pm", "clear", $PackageName)
}

Invoke-AdbBestEffort @("shell", "am", "force-stop", $PackageName)
Invoke-AdbBestEffort @("shell", "rm", "-f", $deviceStatusPath)
Invoke-Adb @("logcat", "-c")
Invoke-Adb @("shell", "monkey", "-p", $PackageName, "-c", "android.intent.category.LAUNCHER", "1")

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$statusText = ""
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 3
    $statusText = Get-DeviceStatusText
    if (![string]::IsNullOrWhiteSpace($statusText)) {
        $statusText | Out-File -FilePath $statusLog -Encoding utf8
        if ($statusText -match "(?m)\]\s+(SUCCESS|FAILURE):") {
            break
        }
    }
}

& adb -s $DeviceSerial logcat -d -v threadtime | Out-File -FilePath $rawLog -Encoding utf8
$patterns = "LiteRT-LM ASRSmoke|INSPECT_RESULT|MODEL_READY|AUDIO_READY|SUCCESS|FAILURE|AndroidRuntime|FATAL|ERROR"
Select-String -Path $rawLog -Pattern $patterns | ForEach-Object { $_.Line } | Out-File -FilePath $summaryLog -Encoding utf8
if (Test-Path $statusLog) {
    Add-Content -Path $summaryLog -Encoding utf8 -Value ""
    Add-Content -Path $summaryLog -Encoding utf8 -Value "STATUS_FILE:"
    Get-Content $statusLog | Add-Content -Path $summaryLog -Encoding utf8
}

if ([string]::IsNullOrWhiteSpace($statusText)) {
    throw "ASR smoke test timed out without a status file. Log: $summaryLog"
}

if ($statusText -match "(?m)\]\s+SUCCESS:") {
    Write-Host "[LiteRT-LM] ASR smoke test passed. Summary: $summaryLog"
    exit 0
}

throw "ASR smoke test failed or timed out. Summary: $summaryLog"
