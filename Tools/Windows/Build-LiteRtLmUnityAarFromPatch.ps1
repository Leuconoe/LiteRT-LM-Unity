param(
    [string]$SourceRoot = "",
    [string]$PatchedRoot = "",
    [string]$ArtifactDir = "",
    [string]$UnityPluginsDir = "",
    [string]$ImageName = "litert-lm-android-builder",
    [switch]$SkipImageBuild,
    [string]$ReuseContainer = "",
    [switch]$SyncWorktree,
    [switch]$SyncOnly,
    [switch]$PackageOnly,
    [string]$ReuseAar = "",
    [string]$BazelJobs = "",
    [switch]$PrepareOnly,
    [switch]$KeepPatchedSource,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDirectory "..\..")).Path
$PatchPath = Join-Path $ProjectRoot "Tools\UnityAar\litert-lm-unity-aar.patch"

function Test-LiteRtLmSourceRoot {
    param([Parameter(Mandatory = $true)][string]$CandidatePath)

    return (Test-Path (Join-Path $CandidatePath "WORKSPACE")) -and
        (Test-Path (Join-Path $CandidatePath "kotlin")) -and
        (Test-Path (Join-Path $CandidatePath "runtime")) -and
        (Test-Path (Join-Path $CandidatePath "tools"))
}

function Resolve-DefaultLiteRtLmSourceRoot {
    $candidates = @(
        (Join-Path $ProjectRoot "External\LiteRT-LM"),
        (Join-Path $ProjectRoot "LiteRT-LM"),
        (Join-Path $ProjectRoot "ThirdParty\LiteRT-LM")
    )

    foreach ($candidate in $candidates) {
        if ((Test-Path $candidate) -and (Test-LiteRtLmSourceRoot $candidate)) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }

    throw "SourceRoot was not provided and no LiteRT-LM source checkout was found. Expected a Unity-local submodule such as External\LiteRT-LM, or pass -SourceRoot explicitly."
}

if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
    $SourceRoot = Resolve-DefaultLiteRtLmSourceRoot
}
$SourceRoot = [System.IO.Path]::GetFullPath($SourceRoot)

if ([string]::IsNullOrWhiteSpace($PatchedRoot)) {
    $PatchedRoot = Join-Path $ProjectRoot "temp\unity-aar-patched\litert-lm"
}
$PatchedRoot = [System.IO.Path]::GetFullPath($PatchedRoot)

if ([string]::IsNullOrWhiteSpace($ArtifactDir)) {
    $ArtifactDir = Join-Path $ProjectRoot "Builds\AndroidAar"
}
$ArtifactDir = [System.IO.Path]::GetFullPath($ArtifactDir)

if ([string]::IsNullOrWhiteSpace($UnityPluginsDir)) {
    $UnityPluginsDir = Join-Path $ProjectRoot "Assets\Plugins\Android"
}
$UnityPluginsDir = [System.IO.Path]::GetFullPath($UnityPluginsDir)

function Assert-UnderPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Parent,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $fullParent = [System.IO.Path]::GetFullPath($Parent)
    if (!$fullParent.EndsWith([System.IO.Path]::DirectorySeparatorChar)) {
        $fullParent += [System.IO.Path]::DirectorySeparatorChar
    }

    if (!$fullPath.StartsWith($fullParent, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label must stay under $fullParent. Actual path: $fullPath"
    }
}

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Program,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [string]$WorkingDirectory = ""
    )

    if ([string]::IsNullOrWhiteSpace($WorkingDirectory)) {
        & $Program @Arguments
    }
    else {
        Push-Location $WorkingDirectory
        try {
            & $Program @Arguments
        }
        finally {
            Pop-Location
        }
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Program $($Arguments -join ' ')"
    }
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

function Copy-SourceInput {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourceRoot,
        [Parameter(Mandatory = $true)]
        [string]$DestinationRoot,
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    $sourcePath = Join-Path $SourceRoot $RelativePath
    if (!(Test-Path $sourcePath)) {
        return
    }

    $destinationPath = Join-Path $DestinationRoot $RelativePath
    $destinationParent = Split-Path -Parent $destinationPath
    if (![string]::IsNullOrWhiteSpace($destinationParent)) {
        New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null
    }

    if (Test-Path $sourcePath -PathType Container) {
        Invoke-CheckedRobocopy `
            -Source $sourcePath `
            -Destination $destinationPath `
            -Arguments @(
                "/MIR",
                "/NFL",
                "/NDL",
                "/NJH",
                "/NJS",
                "/XD",
                ".git",
                "bazel-bin",
                "bazel-out",
                "bazel-testlogs",
                "/XF",
                "*.xnnpack_cache"
            )
    }
    else {
        Copy-Item -LiteralPath $sourcePath -Destination $destinationPath -Force
    }
}

function Copy-AarBuildSourceInputs {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourceRoot,
        [Parameter(Mandatory = $true)]
        [string]$DestinationRoot
    )

    $paths = @(
        ".bazelrc",
        ".bazelversion",
        "WORKSPACE",
        "BUILD",
        "BUILD.antlr4",
        "BUILD.directx_shader_compiler",
        "BUILD.llguidance",
        "BUILD.miniaudio",
        "BUILD.minizip",
        "BUILD.minja",
        "BUILD.nanobind_json",
        "BUILD.sentencepiece",
        "BUILD.stb",
        "BUILD.tokenizers_cpp",
        "Cargo.lock",
        "Cargo.toml",
        "cargo-bazel-lock.json",
        "requirements.txt",
        "android_ndk_env.bzl",
        "rust_cxx_bridge.bzl",
        "version.bzl",
        "build_config",
        "c",
        "cmake",
        "cxxbridge_cmd",
        "kotlin",
        "prebuilt",
        "runtime",
        "rust",
        "schema",
        "src",
        "tools"
    )

    foreach ($relativePath in $paths) {
        Copy-SourceInput -SourceRoot $SourceRoot -DestinationRoot $DestinationRoot -RelativePath $relativePath
    }

    Get-ChildItem -LiteralPath $SourceRoot -Filter "PATCH.*" -File |
        ForEach-Object {
            Copy-SourceInput -SourceRoot $SourceRoot -DestinationRoot $DestinationRoot -RelativePath $_.Name
        }
}

function ConvertTo-RelativeGitPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BasePath,
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $baseFullPath = [System.IO.Path]::GetFullPath($BasePath)
    if (!$baseFullPath.EndsWith([System.IO.Path]::DirectorySeparatorChar)) {
        $baseFullPath += [System.IO.Path]::DirectorySeparatorChar
    }

    $baseUri = [System.Uri]::new($baseFullPath)
    $pathUri = [System.Uri]::new([System.IO.Path]::GetFullPath($Path))
    return [System.Uri]::UnescapeDataString($baseUri.MakeRelativeUri($pathUri).ToString()).TrimEnd("/")
}

function Resolve-BashPath {
    $candidates = @(
        "C:\Program Files\Git\bin\bash.exe",
        "C:\Program Files\Git\usr\bin\bash.exe"
    )

    $bashPath = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (![string]::IsNullOrWhiteSpace($bashPath)) {
        return $bashPath
    }

    $command = Get-Command bash -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    throw "bash was not found. Install Git for Windows or run the patched build script from an existing bash shell."
}

function Test-GitApply {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepositoryRoot,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $quotedArguments = $Arguments | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' }
    $command = 'git -C "' + $RepositoryRoot.Replace('"', '\"') + '" ' + ($quotedArguments -join " ") + " >NUL 2>NUL"
    & cmd.exe /d /c $command
    return $LASTEXITCODE -eq 0
}

function Apply-UnityAarPatch {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TargetRoot,
        [Parameter(Mandatory = $true)]
        [string]$PatchFile,
        [Parameter(Mandatory = $true)]
        [string]$RepositoryRoot
    )

    $targetDirectory = ConvertTo-RelativeGitPath -BasePath $RepositoryRoot -Path $TargetRoot
    if (Test-GitApply -RepositoryRoot $RepositoryRoot -Arguments @("apply", "--check", "--directory=$targetDirectory", $PatchFile)) {
        Invoke-Native "git" @("-C", $RepositoryRoot, "apply", "--directory=$targetDirectory", $PatchFile)
        Write-Host "[LiteRT-LM] Applied Unity AAR patch: $PatchFile"
        return
    }

    if (Test-GitApply -RepositoryRoot $RepositoryRoot -Arguments @("apply", "--reverse", "--check", "--directory=$targetDirectory", $PatchFile)) {
        Write-Host "[LiteRT-LM] Unity AAR patch is already present in patched source."
        return
    }

    $checkOutput = & git -C $RepositoryRoot apply --check "--directory=$targetDirectory" $PatchFile 2>&1
    $reverseCheckOutput = & git -C $RepositoryRoot apply --reverse --check "--directory=$targetDirectory" $PatchFile 2>&1
    Write-Host $checkOutput
    Write-Host $reverseCheckOutput
    throw "Failed to apply Unity AAR patch to $TargetRoot"
}

if (!(Test-Path $PatchPath)) {
    throw "Unity AAR patch not found: $PatchPath"
}
if (!(Test-Path $SourceRoot)) {
    throw "SourceRoot not found: $SourceRoot"
}

$WorkspaceTempRoot = Join-Path $ProjectRoot "temp"
Assert-UnderPath -Path $PatchedRoot -Parent $WorkspaceTempRoot -Label "PatchedRoot"
Assert-UnderPath -Path $ArtifactDir -Parent $ProjectRoot -Label "ArtifactDir"

if (Test-Path $PatchedRoot) {
    Remove-Item -LiteralPath $PatchedRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $PatchedRoot | Out-Null

Write-Host "[LiteRT-LM] Copying LiteRT-LM source to patched workspace: $PatchedRoot"
Copy-AarBuildSourceInputs -SourceRoot $SourceRoot -DestinationRoot $PatchedRoot

Apply-UnityAarPatch -TargetRoot $PatchedRoot -PatchFile $PatchPath -RepositoryRoot $ProjectRoot

if ($PrepareOnly) {
    Write-Host "[LiteRT-LM] Prepare-only mode complete: $PatchedRoot"
    exit 0
}

$bashScript = Join-Path $PatchedRoot "tools\docker\build_unity_aar.sh"
if (!(Test-Path $bashScript)) {
    throw "Patched build script was not created: $bashScript"
}

$arguments = @(
    $bashScript,
    "--image-name", $ImageName,
    "--artifact-dir", $ArtifactDir,
    "--unity-plugins-dir", $UnityPluginsDir
)
if ($SkipImageBuild) {
    $arguments += "--skip-image-build"
}
if (![string]::IsNullOrWhiteSpace($ReuseContainer)) {
    $arguments += @("--reuse-container", $ReuseContainer)
}
if ($SyncWorktree) {
    $arguments += "--sync-worktree"
}
if ($SyncOnly) {
    $arguments += "--sync-only"
}
if ($PackageOnly) {
    $arguments += "--package-only"
}
if (![string]::IsNullOrWhiteSpace($ReuseAar)) {
    $arguments += @("--reuse-aar", $ReuseAar)
}
if (![string]::IsNullOrWhiteSpace($BazelJobs)) {
    $arguments += @("--bazel-jobs", $BazelJobs)
}
if ($ExtraArgs -and $ExtraArgs.Count -gt 0) {
    $arguments += $ExtraArgs
}

$bashPath = Resolve-BashPath
Write-Host "[LiteRT-LM] Running patched Unity AAR build script."
Invoke-Native $bashPath $arguments

if (!$KeepPatchedSource -and (Test-Path $PatchedRoot)) {
    Remove-Item -LiteralPath $PatchedRoot -Recurse -Force
}
