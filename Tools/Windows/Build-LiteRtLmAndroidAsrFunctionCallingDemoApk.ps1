param(
    [string]$AsrModelFileName = "ASR/whisper-tiny/whisper_tiny_30s_i8.tflite",
    [string]$AudioFileName = "TestAssets/Audio/2025년 3월 5일 전술평가 결과 보고.mp3",
    [string]$TokenizerJsonPath = "ASR/whisper-tiny/tokenizer.json",
    [ValidateSet("GPU_FP16", "GPU", "GPU_RELAXED", "GPU_NO_TEXTURE", "GPU_NO_CONVERT", "GPU_RELAXED_NO_CONVERT", "CPU")]
    [string]$AsrBackend = "CPU",
    [string]$AsrLanguage = "ko",
    [string]$LlmModelFileName = "LLM/gemma3-1b/gemma3-1b-it-int4.litertlm",
    [ValidateSet("GPU", "CPU")]
    [string]$LlmBackend = "GPU",
    [int]$LlmMaxNumTokens = 512,
    [string]$OutputApk = "LiteRtLmAndroidAsrFunctionCallingDemo-gemma3-1b-whisper-tiny.apk",
    [string]$UnityPath = "",
    [string]$TempRoot = "",
    [switch]$KeepProjectCopy
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($TempRoot)) {
    $TempRoot = Join-Path $ProjectRoot "temp\unity-android-asr-function-calling-build"
}

function Invoke-CheckedRobocopy {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,
        [Parameter(Mandatory = $true)]
        [string]$Destination,
        [string[]]$Arguments = @()
    )

    & robocopy $Source $Destination @Arguments
    if ($LASTEXITCODE -gt 7) {
        throw "robocopy failed with exit code ${LASTEXITCODE}: $Source -> $Destination"
    }
}

function Resolve-UnityPath {
    if (![string]::IsNullOrWhiteSpace($UnityPath)) {
        return (Resolve-Path $UnityPath).Path
    }

    $projectVersionPath = Join-Path $ProjectRoot "ProjectSettings\ProjectVersion.txt"
    $versionLine = Get-Content $projectVersionPath | Where-Object { $_ -match "^m_EditorVersion:\s*(.+)$" } | Select-Object -First 1
    if (!$versionLine) {
        throw "Failed to resolve Unity editor version from $projectVersionPath"
    }

    $editorVersion = ($versionLine -replace "^m_EditorVersion:\s*", "").Trim()
    $candidate = "C:\Program Files\Unity\Hub\Editor\$editorVersion\Editor\Unity.exe"
    if (!(Test-Path $candidate)) {
        throw "Unity editor not found: $candidate"
    }

    return $candidate
}

function ConvertTo-ProcessArgument {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Argument
    )

    if ($Argument -notmatch '[\s"]') {
        return $Argument
    }

    return '"' + $Argument.Replace('"', '\"') + '"'
}

$asrModelSource = Join-Path $ProjectRoot "Assets\StreamingAssets\$AsrModelFileName"
if (!(Test-Path $asrModelSource)) {
    throw "ASR model not found: $asrModelSource"
}

$llmModelSource = Join-Path $ProjectRoot "Assets\StreamingAssets\$LlmModelFileName"
if (!(Test-Path $llmModelSource)) {
    throw "LLM model not found: $llmModelSource"
}

$audioSource = Join-Path $ProjectRoot "Assets\StreamingAssets\$AudioFileName"
if (!(Test-Path $audioSource)) {
    throw "ASR demo audio not found: $audioSource"
}

$tokenizerSource = Join-Path $ProjectRoot "Assets\StreamingAssets\$TokenizerJsonPath"
if (!(Test-Path $tokenizerSource)) {
    throw "ASR tokenizer not found: $tokenizerSource"
}

$TempRoot = [System.IO.Path]::GetFullPath($TempRoot)
if (!$TempRoot.StartsWith($ProjectRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "TempRoot must stay inside the Unity workspace. ProjectRoot=$ProjectRoot TempRoot=$TempRoot"
}

$runId = Get-Date -Format "yyyyMMdd-HHmmss"
$runRoot = Join-Path $TempRoot $runId
$buildProjectRoot = Join-Path $runRoot "p"
$buildLogs = Join-Path $ProjectRoot "Builds\Logs\AndroidBuilds"
$unityLogPath = Join-Path $buildLogs "$runId-asr-function-calling-unity.log"
$outputPath = Join-Path $ProjectRoot "Builds\Android\$OutputApk"

New-Item -ItemType Directory -Force -Path $runRoot, $buildLogs, (Split-Path -Parent $outputPath) | Out-Null

try {
    Write-Host "[LiteRT-LM] Preparing isolated Unity project copy: $buildProjectRoot"
    Invoke-CheckedRobocopy `
        -Source $ProjectRoot `
        -Destination $buildProjectRoot `
        -Arguments @(
            "/MIR",
            "/NFL",
            "/NDL",
            "/NJH",
            "/NJS",
            "/XD",
            "Library",
            "Temp",
            "Obj",
            "Logs",
            "Builds",
            "UserSettings",
            ".vs",
            "/XF",
            "*.csproj",
            "*.sln",
            "*.litertlm",
            "*.litertlm.meta",
            "*.task",
            "*.task.meta",
            "*.tflite",
            "*.tflite.meta",
            "*.mp3",
            "*.mp3.meta",
            "*.xnnpack_cache"
        )

    $streamingAssetsCopy = Join-Path $buildProjectRoot "Assets\StreamingAssets"
    New-Item -ItemType Directory -Force -Path $streamingAssetsCopy | Out-Null

    function Copy-StreamingAssetWithMeta {
        param(
            [string]$SourcePath,
            [string]$RelativeDestination
        )

        $destination = Join-Path $streamingAssetsCopy $RelativeDestination
        $destinationParent = Split-Path -Parent $destination
        if (![string]::IsNullOrWhiteSpace($destinationParent)) {
            New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null
        }
        Copy-Item -Force $SourcePath $destination
        if (Test-Path "$SourcePath.meta") {
            Copy-Item -Force "$SourcePath.meta" "$destination.meta"
        }
    }

    Copy-StreamingAssetWithMeta -SourcePath $asrModelSource -RelativeDestination $AsrModelFileName
    Copy-StreamingAssetWithMeta -SourcePath $llmModelSource -RelativeDestination $LlmModelFileName
    Copy-StreamingAssetWithMeta -SourcePath $audioSource -RelativeDestination $AudioFileName

    $resolvedUnityPath = Resolve-UnityPath
    $workspaceTemp = Join-Path $ProjectRoot "temp"
    New-Item -ItemType Directory -Force -Path $workspaceTemp | Out-Null
    $env:TEMP = $workspaceTemp
    $env:TMP = $workspaceTemp

    Write-Host "[LiteRT-LM] Building Android ASR function-calling demo APK"
    Write-Host "[LiteRT-LM] ASR model/backend/language: $AsrModelFileName / $AsrBackend / $AsrLanguage"
    Write-Host "[LiteRT-LM] LLM model/backend/max tokens: $LlmModelFileName / $LlmBackend / $LlmMaxNumTokens"
    Write-Host "[LiteRT-LM] Unity log: $unityLogPath"
    $unityArguments = @(
        "-batchmode",
        "-quit",
        "-projectPath",
        $buildProjectRoot,
        "-buildTarget",
        "Android",
        "-executeMethod",
        "LiteRTLM.Unity.Editor.LiteRtLmBuild.BuildAndroidAsrFunctionCallingDemoApkFromCommandLine",
        "-logFile",
        $unityLogPath,
        "-litertlmAsrModel",
        $AsrModelFileName,
        "-litertlmAsrAudio",
        $AudioFileName,
        "-litertlmAsrTokenizer",
        $TokenizerJsonPath,
        "-litertlmAsrBackend",
        $AsrBackend,
        "-litertlmAsrLanguage",
        $AsrLanguage,
        "-litertlmLlmModel",
        $LlmModelFileName,
        "-litertlmLlmBackend",
        $LlmBackend,
        "-litertlmLlmMaxNumTokens",
        $LlmMaxNumTokens,
        "-litertlmOutputApk",
        $outputPath
    )
    $unityArgumentLine = ($unityArguments | ForEach-Object { ConvertTo-ProcessArgument $_ }) -join " "
    $unityProcess = Start-Process -FilePath $resolvedUnityPath -ArgumentList $unityArgumentLine -PassThru -Wait
    $unityExitCode = $unityProcess.ExitCode
    if ($unityExitCode -ne 0) {
        throw "Unity batchmode failed with exit code $unityExitCode. Log: $unityLogPath"
    }

    if (!(Test-Path $outputPath)) {
        throw "Unity batchmode exited successfully but APK was not found: $outputPath"
    }

    Write-Host "[LiteRT-LM] APK written to: $outputPath"
}
finally {
    if (!$KeepProjectCopy -and (Test-Path $runRoot)) {
        Remove-Item -Recurse -Force $runRoot
    }
}
