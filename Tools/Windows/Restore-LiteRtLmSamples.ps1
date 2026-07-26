<#
.SYNOPSIS
    Copies the package sample into Assets/ the same way Package Manager would.

.DESCRIPTION
    Samples live under Packages/com.leuconoe.litert-lm-unity/Samples~/, and the
    trailing '~' keeps Unity from compiling them. Importing through the Package
    Manager UI copies them to Assets/Samples/<displayName>/<version>/<sample>.

    This script does the same thing without the UI, so batchmode builds and the
    device test workflow can run on a fresh clone. It is idempotent.

.PARAMETER Remove
    Delete the imported copy instead of creating it.

.PARAMETER Force
    Overwrite an existing import.

.EXAMPLE
    .\Tools\Windows\Restore-LiteRtLmSamples.ps1
.EXAMPLE
    .\Tools\Windows\Restore-LiteRtLmSamples.ps1 -Force
.EXAMPLE
    .\Tools\Windows\Restore-LiteRtLmSamples.ps1 -Remove
#>
[CmdletBinding()]
param(
    [switch]$Remove,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$packageDir = Join-Path $repoRoot 'Packages/com.leuconoe.litert-lm-unity'
$packageJsonPath = Join-Path $packageDir 'package.json'

if (-not (Test-Path $packageJsonPath)) {
    throw "package.json not found at $packageJsonPath"
}

$package = Get-Content $packageJsonPath -Raw | ConvertFrom-Json
$displayName = $package.displayName
$version = $package.version

foreach ($sample in $package.samples) {
    $sourceDir = Join-Path $packageDir $sample.path
    $targetDir = Join-Path $repoRoot (Join-Path 'Assets/Samples' (Join-Path $displayName (Join-Path $version $sample.displayName)))

    if ($Remove) {
        if (Test-Path $targetDir) {
            Remove-Item -Recurse -Force $targetDir
            $meta = "$targetDir.meta"
            if (Test-Path $meta) { Remove-Item -Force $meta }
            Write-Host "Removed $targetDir"
        }
        else {
            Write-Host "Nothing to remove at $targetDir"
        }
        continue
    }

    if (-not (Test-Path $sourceDir)) {
        throw "Sample source not found: $sourceDir"
    }

    if ((Test-Path $targetDir) -and -not $Force) {
        Write-Host "Already imported: $targetDir (use -Force to overwrite)"
        continue
    }

    if (Test-Path $targetDir) {
        Remove-Item -Recurse -Force $targetDir
    }

    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    Copy-Item -Recurse -Force (Join-Path $sourceDir '*') $targetDir
    Write-Host "Imported '$($sample.displayName)' -> $targetDir"
}

Write-Host ''
Write-Host 'Open the project (or use Assets > Refresh) so Unity picks up the change.'
