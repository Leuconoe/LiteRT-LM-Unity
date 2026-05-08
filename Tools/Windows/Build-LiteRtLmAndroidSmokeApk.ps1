param(
    [ValidateSet(
        "gemma-4-E2B-it-gpu",
        "gemma-4-E2B-it-gpu-nospec",
        "gemma-4-E2B-it-cpu",
        "gemma3-1b-it-gpu",
        "gemma3-1b-it-cpu",
        "gemma3-270m-it-gpu",
        "mobile-actions-gpu",
        "qwen3-0.6b-gpu",
        "qwen2.5-0.5b-gpu",
        "qwen2.5-0.5b-cpu",
        "qwen2.5-1.5b-gpu",
        "qwen2.5-1.5b-cpu"
    )]
    [string]$BenchmarkName = "gemma-4-E2B-it-gpu",
    [string]$UnityPath = "",
    [string]$TempRoot = "",
    [switch]$KeepProjectCopy
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$RepoRoot = (Resolve-Path (Join-Path $ProjectRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($TempRoot)) {
    $TempRoot = Join-Path $RepoRoot "temp\unity-android-build"
}

$benchmark = @{
    "gemma-4-E2B-it-gpu" = @{
        Method = "LiteRTLM.Unity.Editor.LiteRtLmBuild.BuildAndroidAvdSmokeTestApkGemma4"
        Model = "gemma-4-E2B-it.litertlm"
        Apk = "LiteRtLmAndroidSmokeTest-gemma-4-E2B-it.apk"
    }
    "gemma-4-E2B-it-gpu-nospec" = @{
        Method = "LiteRTLM.Unity.Editor.LiteRtLmBuild.BuildAndroidAvdSmokeTestApkGemma4NoSpeculative"
        Model = "gemma-4-E2B-it.litertlm"
        Apk = "LiteRtLmAndroidSmokeTest-gemma-4-E2B-it-nospec.apk"
    }
    "gemma-4-E2B-it-cpu" = @{
        Method = "LiteRTLM.Unity.Editor.LiteRtLmBuild.BuildAndroidAvdSmokeTestApkGemma4Cpu"
        Model = "gemma-4-E2B-it.litertlm"
        Apk = "LiteRtLmAndroidSmokeTest-gemma-4-E2B-it-CPU.apk"
    }
    "gemma3-1b-it-gpu" = @{
        Method = "LiteRTLM.Unity.Editor.LiteRtLmBuild.BuildAndroidAvdSmokeTestApkGemma1B"
        Model = "gemma3-1b-it-int4.litertlm"
        Apk = "LiteRtLmAndroidSmokeTest-gemma3-1b-it-int4.apk"
    }
    "gemma3-1b-it-cpu" = @{
        Method = "LiteRTLM.Unity.Editor.LiteRtLmBuild.BuildAndroidAvdSmokeTestApkGemma1BCpu"
        Model = "gemma3-1b-it-int4.litertlm"
        Apk = "LiteRtLmAndroidSmokeTest-gemma3-1b-it-int4-CPU.apk"
    }
    "gemma3-270m-it-gpu" = @{
        Method = "LiteRTLM.Unity.Editor.LiteRtLmBuild.BuildAndroidAvdSmokeTestApkGemma270M"
        Model = "gemma3-270m-it-q8.litertlm"
        Apk = "LiteRtLmAndroidSmokeTest-gemma3-270m-it-q8.apk"
    }
    "mobile-actions-gpu" = @{
        Method = "LiteRTLM.Unity.Editor.LiteRtLmBuild.BuildAndroidAvdSmokeTestApkMobileActions"
        Model = "mobile_actions_q8_ekv1024.litertlm"
        Apk = "LiteRtLmAndroidSmokeTest-mobile_actions_q8_ekv1024.apk"
    }
    "qwen3-0.6b-gpu" = @{
        Method = "LiteRTLM.Unity.Editor.LiteRtLmBuild.BuildAndroidAvdSmokeTestApkQwen3"
        Model = "Qwen3-0.6B.litertlm"
        Apk = "LiteRtLmAndroidSmokeTest-Qwen3-0.6B.apk"
    }
    "qwen2.5-0.5b-gpu" = @{
        Method = "LiteRTLM.Unity.Editor.LiteRtLmBuild.BuildAndroidAvdSmokeTestApkQwen25"
        Model = "Qwen2.5-0.5B-Instruct-q8.litertlm"
        Apk = "LiteRtLmAndroidSmokeTest-Qwen2.5-0.5B-Instruct.apk"
    }
    "qwen2.5-0.5b-cpu" = @{
        Method = "LiteRTLM.Unity.Editor.LiteRtLmBuild.BuildAndroidAvdSmokeTestApkQwen25Cpu"
        Model = "Qwen2.5-0.5B-Instruct-q8.litertlm"
        Apk = "LiteRtLmAndroidSmokeTest-Qwen2.5-0.5B-Instruct-CPU.apk"
    }
    "qwen2.5-1.5b-gpu" = @{
        Method = "LiteRTLM.Unity.Editor.LiteRtLmBuild.BuildAndroidAvdSmokeTestApkQwen25_1_5B"
        Model = "Qwen2.5-1.5B-Instruct-q8.litertlm"
        Apk = "LiteRtLmAndroidSmokeTest-Qwen2.5-1.5B-Instruct.apk"
    }
    "qwen2.5-1.5b-cpu" = @{
        Method = "LiteRTLM.Unity.Editor.LiteRtLmBuild.BuildAndroidAvdSmokeTestApkQwen25_1_5BCpu"
        Model = "Qwen2.5-1.5B-Instruct-q8.litertlm"
        Apk = "LiteRtLmAndroidSmokeTest-Qwen2.5-1.5B-Instruct-CPU.apk"
    }
}[$BenchmarkName]

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

$modelSource = Join-Path $ProjectRoot "Assets\StreamingAssets\$($benchmark.Model)"
if (!(Test-Path $modelSource)) {
    throw "Benchmark model not found: $modelSource"
}

$TempRoot = [System.IO.Path]::GetFullPath($TempRoot)
if (!$TempRoot.StartsWith($RepoRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "TempRoot must stay inside the repository workspace. RepoRoot=$RepoRoot TempRoot=$TempRoot"
}

$runId = Get-Date -Format "yyyyMMdd-HHmmss"
$runRoot = Join-Path $TempRoot $runId
$buildProjectRoot = Join-Path $runRoot "p"
$buildLogs = Join-Path $ProjectRoot "Builds\Logs\AndroidBuilds"
$unityLogPath = Join-Path $buildLogs "$runId-$BenchmarkName-unity.log"
$outputApkInCopy = Join-Path $buildProjectRoot "Builds\Android\$($benchmark.Apk)"
$outputApk = Join-Path $ProjectRoot "Builds\Android\$($benchmark.Apk)"

New-Item -ItemType Directory -Force -Path $runRoot, $buildLogs | Out-Null

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
            "*.xnnpack_cache"
        )

    $streamingAssetsCopy = Join-Path $buildProjectRoot "Assets\StreamingAssets"
    New-Item -ItemType Directory -Force -Path $streamingAssetsCopy | Out-Null
    Copy-Item -Force $modelSource (Join-Path $streamingAssetsCopy $benchmark.Model)

    $modelMetaSource = "$modelSource.meta"
    if (Test-Path $modelMetaSource) {
        Copy-Item -Force $modelMetaSource (Join-Path $streamingAssetsCopy "$($benchmark.Model).meta")
    }

    $resolvedUnityPath = Resolve-UnityPath
    $workspaceTemp = Join-Path $RepoRoot "temp"
    New-Item -ItemType Directory -Force -Path $workspaceTemp | Out-Null
    $env:TEMP = $workspaceTemp
    $env:TMP = $workspaceTemp

    Write-Host "[LiteRT-LM] Building $BenchmarkName via $($benchmark.Method)"
    Write-Host "[LiteRT-LM] Unity log: $unityLogPath"
    $unityArguments = @(
        "-batchmode",
        "-quit",
        "-projectPath",
        $buildProjectRoot,
        "-buildTarget",
        "Android",
        "-executeMethod",
        $benchmark.Method,
        "-logFile",
        $unityLogPath
    )
    $unityArgumentLine = ($unityArguments | ForEach-Object { ConvertTo-ProcessArgument $_ }) -join " "
    $unityProcess = Start-Process -FilePath $resolvedUnityPath -ArgumentList $unityArgumentLine -PassThru -Wait
    $unityExitCode = $unityProcess.ExitCode
    if ($unityExitCode -ne 0) {
        throw "Unity batchmode failed with exit code $unityExitCode. Log: $unityLogPath"
    }

    if (!(Test-Path $outputApkInCopy)) {
        throw "Unity batchmode exited successfully but APK was not found: $outputApkInCopy"
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outputApk) | Out-Null
    Copy-Item -Force $outputApkInCopy $outputApk
    Write-Host "[LiteRT-LM] APK copied to: $outputApk"
}
finally {
    if (!$KeepProjectCopy -and (Test-Path $runRoot)) {
        Remove-Item -Recurse -Force $runRoot
    }
}
