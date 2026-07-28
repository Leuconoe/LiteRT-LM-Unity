<#
.SYNOPSIS
Type-check the package Runtime sources without opening Unity.

.DESCRIPTION
Compiles Packages/com.leuconoe.litert-lm-unity/Runtime/*.cs with the VS2022
Roslyn compiler against Unity's reference assemblies, once per platform define
set (editor/Windows and Android). Catches the class of mistake that only shows
up on the *other* platform — a field whose type lives inside a `#if UNITY_ANDROID`
block compiles fine on device and breaks the editor, and vice versa.

Opening Unity in batch mode also finds these, but takes minutes and reimports
assets; this takes seconds. It is a syntax and type gate, not a substitute for
running the editor.
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$UnityVersion = "2022.3.62f3",
    [string]$Csc = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$RuntimeDir = Join-Path $ProjectRoot "Packages\com.leuconoe.litert-lm-unity\Runtime"

if (!$Csc) {
    $candidates = @(
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\Roslyn\csc.exe"
    ) + (Get-ChildItem "C:\Program Files*\Microsoft Visual Studio\*\*\MSBuild\Current\Bin\Roslyn\csc.exe" -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
    $Csc = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (!$Csc) { throw "No Roslyn csc.exe found. Install VS2022 Build Tools or pass -Csc." }

$EditorData = "C:\Program Files\Unity\Hub\Editor\$UnityVersion\Editor\Data"
if (!(Test-Path $EditorData)) { throw "Unity $UnityVersion not found at $EditorData" }

$netstandard = Join-Path $EditorData "NetStandard\ref\2.1.0\netstandard.dll"
if (!(Test-Path $netstandard)) { throw "netstandard reference assembly not found: $netstandard" }

$managed = Join-Path $EditorData "Managed\UnityEngine"
$modules = @(
    "UnityEngine.CoreModule",
    "UnityEngine.AudioModule",
    "UnityEngine.IMGUIModule",
    "UnityEngine.TextRenderingModule",
    "UnityEngine.UnityWebRequestModule",
    "UnityEngine.UnityWebRequestAudioModule",
    "UnityEngine.AndroidJNIModule"
)
$refs = @("-r:$netstandard") + ($modules | ForEach-Object { "-r:" + (Join-Path $managed "$_.dll") })

$sources = Get-ChildItem (Join-Path $RuntimeDir "*.cs") | ForEach-Object { $_.FullName }
if (!$sources) { throw "No sources under $RuntimeDir" }

$passes = @(
    @{ Name = "editor / Windows"; Defines = "UNITY_EDITOR;UNITY_EDITOR_WIN;UNITY_STANDALONE_WIN;UNITY_2020_1_OR_NEWER" },
    @{ Name = "Android player";   Defines = "UNITY_ANDROID;UNITY_2020_1_OR_NEWER" }
)

$failed = 0
foreach ($pass in $passes) {
    $out = Join-Path $env:TEMP ("litertlm-runtime-" + ($pass.Name -replace "[^a-zA-Z]", "") + ".dll")
    $arguments = @(
        "-target:library", "-nostdlib+", "-noconfig", "-langversion:9.0",
        "-define:$($pass.Defines)", "-out:$out"
    ) + $refs + $sources

    $output = & $Csc @arguments 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "PASS  $($pass.Name)" -ForegroundColor Green
    }
    else {
        $failed++
        Write-Host "FAIL  $($pass.Name)" -ForegroundColor Red
        $output | Where-Object { $_ -match "error|warning" } | ForEach-Object { Write-Host "  $_" }
    }
}

if ($failed -gt 0) {
    Write-Host "$failed of $($passes.Count) pass(es) failed." -ForegroundColor Red
    exit 1
}
Write-Host "Runtime compiles for both define sets." -ForegroundColor Green
