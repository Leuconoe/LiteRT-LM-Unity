param(
    [string]$ModelFileName = "parakeet_tdt_0.6b_v3_5s_i8.tflite",
    [string]$AudioFileName = "Tactical Evaluation Results Report - March 5, 2025.mp3",
    [ValidateSet("GPU_FP16", "GPU", "GPU_RELAXED", "GPU_NO_TEXTURE", "GPU_NO_CONVERT", "CPU")]
    [string]$Backend = "GPU_FP16",
    [string]$OutputApk = "LiteRtLmAndroidAsrSmokeTest-parakeet-tdt-0.6b-v3.apk",
    [string]$UnityPath = "",
    [string]$TempRoot = "",
    [switch]$KeepProjectCopy
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($TempRoot)) {
    $TempRoot = Join-Path $ProjectRoot "temp\unity-android-asr-build"
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

$modelSource = Join-Path $ProjectRoot "Assets\StreamingAssets\$ModelFileName"
if (!(Test-Path $modelSource)) {
    throw "ASR model not found: $modelSource"
}

$audioSource = Join-Path $ProjectRoot "Assets\StreamingAssets\$AudioFileName"
if (!(Test-Path $audioSource)) {
    throw "ASR test audio not found: $audioSource"
}

$TempRoot = [System.IO.Path]::GetFullPath($TempRoot)
if (!$TempRoot.StartsWith($ProjectRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "TempRoot must stay inside the Unity workspace. ProjectRoot=$ProjectRoot TempRoot=$TempRoot"
}

$runId = Get-Date -Format "yyyyMMdd-HHmmss"
$runRoot = Join-Path $TempRoot $runId
$buildProjectRoot = Join-Path $runRoot "p"
$buildLogs = Join-Path $ProjectRoot "Builds\Logs\AndroidBuilds"
$unityLogPath = Join-Path $buildLogs "$runId-asr-smoke-unity.log"
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
    Copy-Item -Force $modelSource (Join-Path $streamingAssetsCopy $ModelFileName)
    Copy-Item -Force $audioSource (Join-Path $streamingAssetsCopy $AudioFileName)

    foreach ($source in @("$modelSource.meta", "$audioSource.meta")) {
        if (Test-Path $source) {
            Copy-Item -Force $source (Join-Path $streamingAssetsCopy (Split-Path -Leaf $source))
        }
    }

    $resolvedUnityPath = Resolve-UnityPath
    $workspaceTemp = Join-Path $ProjectRoot "temp"
    New-Item -ItemType Directory -Force -Path $workspaceTemp | Out-Null
    $env:TEMP = $workspaceTemp
    $env:TMP = $workspaceTemp

    Write-Host "[LiteRT-LM] Building Android ASR smoke APK"
    Write-Host "[LiteRT-LM] ASR backend: $Backend"
    Write-Host "[LiteRT-LM] Unity log: $unityLogPath"
    $unityArguments = @(
        "-batchmode",
        "-quit",
        "-projectPath",
        $buildProjectRoot,
        "-buildTarget",
        "Android",
        "-executeMethod",
        "LiteRTLM.Unity.Editor.LiteRtLmBuild.BuildAndroidAsrSmokeTestApkFromCommandLine",
        "-logFile",
        $unityLogPath,
        "-litertlmAsrModel",
        $ModelFileName,
        "-litertlmAsrAudio",
        $AudioFileName,
        "-litertlmBackend",
        $Backend,
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
