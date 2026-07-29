param(
    [string]$UnityExe = "",
    [string]$LogFile = "",
    [string]$ExecuteMethod = "LiteRTLM.Unity.Editor.LiteRtLmBuild.RunWindowsEditorSelfTestBatchmode",
    [string]$StatusRelativePath = "Builds\Logs\LiteRtLmEditorSelfTest.status.txt",
    [string]$TestName = "Unity editor self-test",
    [string]$TempRoot = "",
    [int]$MaxAttempts = 3,
    [switch]$FullLog,
    [switch]$KeepProjectCopy
)

$ErrorActionPreference = "Stop"

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDirectory)
$RepoRoot = Split-Path -Parent $ProjectRoot
$ProjectVersionFile = Join-Path $ProjectRoot "ProjectSettings\ProjectVersion.txt"
$ExecutionProjectRoot = $ProjectRoot
$TemporaryCopyRoot = ""
$TemporaryRepoRoot = ""
$StatusFile = ""
$RunId = Get-Date -Format "yyyyMMdd-HHmmssfff"
$LogFileWasProvided = -not [string]::IsNullOrWhiteSpace($LogFile)
$LogDirectory = Join-Path $ProjectRoot "Builds\Logs"

if ([string]::IsNullOrWhiteSpace($TempRoot)) {
    $TempRoot = Join-Path $RepoRoot "temp\unity-editor-self-test"
}

New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
New-Item -ItemType Directory -Force -Path $TempRoot | Out-Null

if (-not $LogFileWasProvided) {
    $LogFile = Join-Path $LogDirectory ("LiteRtLmEditorSelfTest-" + $RunId + ".log")
}

function Resolve-UnityEditorPath {
    param([string]$ConfiguredPath)

    if (-not [string]::IsNullOrWhiteSpace($ConfiguredPath)) {
        return $ConfiguredPath
    }

    if (-not (Test-Path $ProjectVersionFile)) {
        throw "Unity ProjectVersion file not found: $ProjectVersionFile"
    }

    $projectVersionLine = Get-Content $ProjectVersionFile | Select-String "m_EditorVersion:" | Select-Object -First 1
    if ($null -eq $projectVersionLine) {
        throw "Failed to resolve Unity editor version from: $ProjectVersionFile"
    }

    $version = ($projectVersionLine.Line -split ":", 2)[1].Trim()
    $candidate = Join-Path "C:\Program Files\Unity\Hub\Editor" "$version\Editor\Unity.exe"
    if (-not (Test-Path $candidate)) {
        throw "Unity.exe not found at expected path: $candidate"
    }

    return $candidate
}

function New-SelfTestProjectCopy {
    $copyRootName = "lrtst-" + [guid]::NewGuid().ToString("N").Substring(0, 8)
    $candidateRoots = @()
    if (-not [string]::IsNullOrWhiteSpace($TempRoot)) {
        $candidateRoots += (Join-Path $TempRoot $copyRootName)
    }

    $copyRoot = ""
    foreach ($candidateRoot in $candidateRoots) {
        try {
            New-Item -ItemType Directory -Force -Path (Join-Path $candidateRoot "r\p") | Out-Null
            $copyRoot = $candidateRoot
            break
        } catch {
            Write-Warning "Failed to create self-test copy root '$candidateRoot': $($_.Exception.Message)"
        }
    }

    if ([string]::IsNullOrWhiteSpace($copyRoot)) {
        throw "Failed to create a Unity self-test project copy root."
    }

    $copyRepoRoot = Join-Path $copyRoot "r"
    $copyProjectRoot = Join-Path $copyRepoRoot "p"
    $copyRuntimeTestDataRoot = Join-Path $copyRepoRoot "runtime\testdata"

    New-Item -ItemType Directory -Force -Path $copyRuntimeTestDataRoot | Out-Null

    $robocopyArguments = @(
        $ProjectRoot,
        $copyProjectRoot,
        "/MIR",
        "/XD", "Library", "Temp", "Obj", "Builds", "Logs", "UserSettings", ".vs",
        "/XF", "*.csproj", "*.sln"
    )

    & robocopy @robocopyArguments | Out-Null
    $robocopyExitCode = $LASTEXITCODE
    if ($robocopyExitCode -ge 8) {
        throw "Failed to create Unity self-test project copy. robocopy exit code: $robocopyExitCode"
    }

    $sourceRuntimeModelPath = Join-Path $RepoRoot "runtime\testdata\test_lm.litertlm"
    if (-not (Test-Path $sourceRuntimeModelPath)) {
        throw "Runtime test model not found: $sourceRuntimeModelPath"
    }

    Copy-Item -Force $sourceRuntimeModelPath (Join-Path $copyRuntimeTestDataRoot "test_lm.litertlm")

    return @{
        CopyRoot = $copyRoot
        RepoRoot = $copyRepoRoot
        ProjectRoot = $copyProjectRoot
    }
}

$UnityEditorPath = Resolve-UnityEditorPath -ConfiguredPath $UnityExe

$projectLockFile = Join-Path $ProjectRoot "Temp\UnityLockfile"
if (Test-Path $projectLockFile) {
    Write-Host "[LiteRT-LM] Project appears to be open in another Unity instance. Using isolated self-test project copy."
    $projectCopyInfo = New-SelfTestProjectCopy
    $TemporaryCopyRoot = $projectCopyInfo.CopyRoot
    $TemporaryRepoRoot = $projectCopyInfo.RepoRoot
    $ExecutionProjectRoot = $projectCopyInfo.ProjectRoot
}

Write-Host "[LiteRT-LM] Unity Editor: $UnityEditorPath"
Write-Host "[LiteRT-LM] Project: $ExecutionProjectRoot"

$StatusFile = Join-Path $ExecutionProjectRoot $StatusRelativePath
$exitCode = 1
$statusContent = ""

for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    if ((-not $LogFileWasProvided) -and $MaxAttempts -gt 1) {
        $LogFile = Join-Path $LogDirectory ("LiteRtLmEditorSelfTest-" + $RunId + "-attempt" + $attempt + ".log")
    }

    if (Test-Path $StatusFile) {
        Remove-Item -Force $StatusFile
    }

    Write-Host "[LiteRT-LM] Attempt $attempt/$MaxAttempts"
    Write-Host "[LiteRT-LM] Log: $LogFile"

    $unityArguments = @(
        "-batchmode",
        "-projectPath", $ExecutionProjectRoot,
        "-executeMethod", $ExecuteMethod,
        "-logFile", $LogFile,
        "-quit"
    )

    try {
        $unityProcess = Start-Process -FilePath $UnityEditorPath -ArgumentList $unityArguments -Wait -PassThru -NoNewWindow
        $exitCode = $unityProcess.ExitCode
    } catch {
        $exitCode = $LASTEXITCODE
        if ($null -eq $exitCode) {
            $exitCode = 1
        }
        Write-Warning "Unity process raised an invocation error before wrapper post-processing: $($_.Exception.Message)"
    }

    Write-Host "[LiteRT-LM] Unity exit code: $exitCode"

    $statusContent = ""
    if (Test-Path $StatusFile) {
        $statusContent = Get-Content $StatusFile -Raw
    }

    if ($exitCode -ne 0 -and $statusContent -match "SUCCESS:") {
        Write-Warning "Unity returned exit code $exitCode, but the in-editor self-test recorded SUCCESS. Treating this run as passed."
        $exitCode = 0
    }

    if ($exitCode -eq 0) {
        break
    }

    $logContent = ""
    if (Test-Path $LogFile) {
        $logContent = Get-Content $LogFile -Raw
    }

    $reachedSelfTest = $statusContent -match "START:"
    $retryableUnityStartupFailure = (-not $reachedSelfTest) -and ($logContent -match "LicensingClient|IPC channel|Timed-out after|Connection to channel")
    if ($attempt -lt $MaxAttempts -and $retryableUnityStartupFailure) {
        Write-Warning "Unity failed before entering the LiteRT-LM test. Retrying after a short delay."
        Start-Sleep -Seconds 5
        continue
    }

    break
}

if (Test-Path $LogFile) {
    Write-Host "[LiteRT-LM] ---- $TestName Log ----"
    Write-Host "[LiteRT-LM] Full log: $LogFile"
    if ($FullLog) {
        Get-Content $LogFile
    } else {
        $logPattern = "LiteRT-LM|self-test|SUCCESS|FAIL|ExitCode|Invoking|ResponseLength|Exiting batchmode|return code|LicensingClient|IPC channel|Timed-out|batchmode"
        if ($exitCode -ne 0) {
            $logPattern = "LiteRT-LM|SUCCESS|FAIL|Exception|error|ExitCode|SelfTest|Invoking|ResponseLength|Quit|Aborting|LicensingClient|IPC channel|Timed-out|batchmode"
        }

        Select-String -Path $LogFile -Pattern $logPattern -CaseSensitive:$false |
            Select-Object -Last 120 |
            ForEach-Object { "$($_.LineNumber): $($_.Line)" }
    }
    Write-Host "[LiteRT-LM] -----------------------------"
}

if (Test-Path $StatusFile) {
    Write-Host "[LiteRT-LM] ---- $TestName Status ----"
    Get-Content $StatusFile |
        ForEach-Object {
            if ($_.Length -gt 300) {
                $_.Substring(0, 300) + "..."
            } else {
                $_
            }
        }
    Write-Host "[LiteRT-LM] --------------------------------"
}

if ($exitCode -ne 0) {
    if (-not [string]::IsNullOrWhiteSpace($TemporaryCopyRoot)) {
        Write-Host "[LiteRT-LM] Test project copy preserved for inspection: $TemporaryCopyRoot"
    }
    throw "$TestName failed with exit code $exitCode"
}

if ((-not [string]::IsNullOrWhiteSpace($TemporaryCopyRoot)) -and (-not $KeepProjectCopy)) {
    Remove-Item -Recurse -Force $TemporaryCopyRoot
}

Write-Host "[LiteRT-LM] $TestName passed."
