param(
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

. (Join-Path $PSScriptRoot "LiteRtLmAndroidBenchmarks.ps1")
$benchmark = Get-LiteRtLmAndroidBenchmark -Name $BenchmarkName

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
