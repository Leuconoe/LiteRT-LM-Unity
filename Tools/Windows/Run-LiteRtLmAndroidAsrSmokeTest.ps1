[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$DeviceSerial = "46a880a0",
    [string]$PackageName = "com.Leuconoe.LiteRTLMUnity",
    [string]$ApkPath = "",
    [string]$ModelFileName = "ASR/whisper-tiny/whisper_tiny_30s_i8.tflite",
    [string]$AudioFileName = "TestAssets/Audio/2025년 3월 5일 전술평가 결과 보고.mp3",
    [string]$TokenizerJsonPath = "ASR/whisper-tiny/tokenizer.json",
    [ValidateSet("parakeet", "whisper")]
    [string]$AsrMode = "whisper",
    [string]$AsrLanguage = "ko",
    [ValidateSet("GPU_FP16", "GPU", "GPU_RELAXED", "GPU_NO_TEXTURE", "GPU_NO_CONVERT", "GPU_RELAXED_NO_CONVERT", "CPU")]
    [string]$Backend = "CPU",
    [int]$BenchmarkRuns = 1,
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
$deviceConfigPath = "/sdcard/Android/data/$PackageName/files/LiteRtLmAsrSmokeTest.config.json"

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

function Push-FileToAsrData {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourcePath,
        [Parameter(Mandatory = $true)]
        [string]$RelativeDevicePath
    )

    if (!(Test-Path $SourcePath)) {
        throw "ASR input file not found: $SourcePath"
    }

    $devicePath = "/sdcard/Android/data/$PackageName/files/LiteRTLM/ASR/$($RelativeDevicePath -replace '\\', '/')"
    $deviceDirectory = $devicePath.Substring(0, $devicePath.LastIndexOf('/'))
    Invoke-Adb @("shell", "mkdir", "-p", $deviceDirectory)
    Invoke-Adb @("push", $SourcePath, $devicePath)
}

function Get-WhisperEncoderCompanionFileName {
    param([string]$FileName)

    if ($FileName.EndsWith("_f32.tflite", [StringComparison]::OrdinalIgnoreCase)) {
        $preferred = $FileName.Substring(0, $FileName.Length - ".tflite".Length) + "_encoder.tflite"
        $legacy = $FileName.Substring(0, $FileName.Length - "_f32.tflite".Length) + "_encoder_f32.tflite"
        if (Test-Path (Join-Path $ProjectRoot "Assets\StreamingAssets\$preferred")) {
            return $preferred
        }
        if (Test-Path (Join-Path $ProjectRoot "Assets\StreamingAssets\$legacy")) {
            return $legacy
        }
        return $preferred
    }

    if ($FileName.EndsWith(".tflite", [StringComparison]::OrdinalIgnoreCase)) {
        return $FileName.Substring(0, $FileName.Length - ".tflite".Length) + "_encoder.tflite"
    }

    return ""
}

function Push-AsrRuntimeConfig {
    $configDirectory = Join-Path $ProjectRoot "temp\android-asr-device-configs"
    New-Item -ItemType Directory -Force -Path $configDirectory | Out-Null
    $configPath = Join-Path $configDirectory "$RunId-asr-smoke-config.json"

    $config = [ordered]@{
        modelPath = $ModelFileName
        audioPath = $AudioFileName
        tokenizerJsonPath = $TokenizerJsonPath
        backend = $Backend
        asrMode = $AsrMode
        asrLanguage = $AsrLanguage
        benchmarkRuns = [Math]::Max(1, $BenchmarkRuns)
    }

    ($config | ConvertTo-Json -Depth 4) | Set-Content -Path $configPath -Encoding utf8
    Invoke-Adb @("shell", "mkdir", "-p", "/sdcard/Android/data/$PackageName/files")
    Invoke-Adb @("push", $configPath, $deviceConfigPath)
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

Write-Host "[LiteRT-LM] Pushing ASR runtime inputs: model=$ModelFileName, backend=$Backend, mode=$AsrMode, language=$AsrLanguage"
Push-FileToAsrData -SourcePath (Join-Path $ProjectRoot "Assets\StreamingAssets\$ModelFileName") -RelativeDevicePath $ModelFileName
Push-FileToAsrData -SourcePath (Join-Path $ProjectRoot "Assets\StreamingAssets\$AudioFileName") -RelativeDevicePath $AudioFileName
Push-FileToAsrData -SourcePath (Join-Path $ProjectRoot "Assets\StreamingAssets\$TokenizerJsonPath") -RelativeDevicePath $TokenizerJsonPath
if ($AsrMode -eq "whisper" -and $Backend.StartsWith("GPU", [StringComparison]::OrdinalIgnoreCase)) {
    $encoderCompanion = Get-WhisperEncoderCompanionFileName -FileName $ModelFileName
    if (![string]::IsNullOrWhiteSpace($encoderCompanion)) {
        $encoderSource = Join-Path $ProjectRoot "Assets\StreamingAssets\$encoderCompanion"
        if (Test-Path $encoderSource) {
            Write-Host "[LiteRT-LM] Pushing Whisper encoder companion: $encoderCompanion"
            Push-FileToAsrData -SourcePath $encoderSource -RelativeDevicePath $encoderCompanion
        }
    }
}
Push-AsrRuntimeConfig

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
