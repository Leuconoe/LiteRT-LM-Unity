<#
.SYNOPSIS
Measure Supertonic TTS on the device across CPU and GPU backends and print the
comparison.

.DESCRIPTION
Runs the headless smoke test once per backend, parses the RESULT and CHECKSUM
lines, and tabulates RTF plus the per-graph timings side by side.

Only the *bucketed* graphs (vector_estimator, vocoder) change accelerator:
duration_predictor and text_encoder are resized per utterance and must stay on
CPU, since a GPU delegate cannot re-prepare after a resize any more than XNNPACK
can. They are also the cheap half, so there is little to win there.

Checksums are compared as well: a GPU path that is faster but numerically
different (fp16, for instance) is a different trade, not a free win, and the
tensor checksums make that visible without listening to anything.

.EXAMPLE
  .\Tools\Windows\Compare-SupertonicDeviceBackends.ps1
.EXAMPLE
  .\Tools\Windows\Compare-SupertonicDeviceBackends.ps1 -Backends CPU,GPU,GPU_FP16
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string[]]$Backends = @("CPU", "GPU", "GPU_FP16"),
    [string]$DeviceSerial = "46a880a0",
    [string]$ApkPath = "Builds\AndroidBuilds\LiteRtLmAndroidTtsSmokeTest.apk",
    [int]$Steps = 4,
    [int]$RunsPerSentence = 2,
    [int]$TimeoutSeconds = 900
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$logDirectory = Join-Path $ProjectRoot "Builds\Logs\AndroidTtsSmoke"
New-Item -ItemType Directory -Force $logDirectory | Out-Null
$reportPath = Join-Path $logDirectory ("backend-comparison-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".json")

$all = @{}
foreach ($backend in $Backends) {
    Write-Host ""
    Write-Host "=== $backend ===" -ForegroundColor Cyan

    # -ClearAppData only on the first run: re-staging 200 MB out of the APK each
    # time would dominate the measurement.
    $clear = ($backend -eq $Backends[0])
    $output = & (Join-Path $PSScriptRoot "Run-LiteRtLmAndroidTtsSmokeTest.ps1") `
        -DeviceSerial $DeviceSerial -ApkPath $ApkPath -Backend $backend `
        -Steps $Steps -RunsPerSentence $RunsPerSentence `
        -TimeoutSeconds $TimeoutSeconds -SkipTranscribe:$true `
        -ClearAppData:$clear *>&1 | Out-String

    $runs = @()
    foreach ($line in ($output -split "`r?`n")) {
        if ($line -match "RESULT: \[(\d+)\] run (\d+): rtf=([\d\.]+), audioSeconds=([\d\.]+), wallSeconds=([\d\.]+), vePerStep=([\d\.]+), vocoder=([\d\.]+), compile=([\d\.]+), cache=(\w+)") {
            $runs += [ordered]@{
                sentence = [int]$Matches[1]; run = [int]$Matches[2]
                rtf = [double]$Matches[3]; audio = [double]$Matches[4]
                wall = [double]$Matches[5]; vePerStep = [double]$Matches[6]
                vocoder = [double]$Matches[7]; compile = [double]$Matches[8]
                cache = $Matches[9]
            }
        }
        elseif ($line -match "CHECKSUM: \[(\d+)\] duration=([\d\.]+).*?textEmb=(\w+)") {
            $runs += [ordered]@{ sentence = [int]$Matches[1]; checksum = $true
                                 duration = [double]$Matches[2]; textEmb = $Matches[3] }
        }
    }

    $verdict = if ($output -match "Verdict: SUCCESS") { "SUCCESS" }
               elseif ($output -match "Verdict: (\w+)") { $Matches[1] } else { "UNKNOWN" }
    $all[$backend] = [ordered]@{ verdict = $verdict; runs = $runs }
    Write-Host "  verdict=$verdict, parsed $($runs.Count) line(s)"
}

Write-Host ""
Write-Host "=== warm RTF by sentence (lower is better) ===" -ForegroundColor Green
$header = "{0,-10}" -f "sentence"
foreach ($backend in $Backends) { $header += "{0,14}" -f $backend }
Write-Host $header

foreach ($sentence in 1..3) {
    $row = "{0,-10}" -f $sentence
    foreach ($backend in $Backends) {
        $warm = $all[$backend].runs |
            Where-Object { $_.sentence -eq $sentence -and $_.cache -eq "hit" -and $_.rtf } |
            Select-Object -First 1
        $row += "{0,14}" -f ($(if ($warm) { "{0:N3}" -f $warm.rtf } else { "-" }))
    }
    Write-Host $row
}

foreach ($metric in @("vePerStep", "vocoder", "compile")) {
    Write-Host ""
    Write-Host "=== $metric seconds ===" -ForegroundColor Green
    Write-Host $header
    foreach ($sentence in 1..3) {
        $row = "{0,-10}" -f $sentence
        foreach ($backend in $Backends) {
            $entry = $all[$backend].runs |
                Where-Object { $_.sentence -eq $sentence -and $_.PSObject.Properties.Name -contains $metric } |
                Select-Object -Last 1
            $row += "{0,14}" -f ($(if ($entry) { "{0:N3}" -f $entry[$metric] } else { "-" }))
        }
        Write-Host $row
    }
}

Write-Host ""
Write-Host "=== numerics (duration / textEmb md5) ===" -ForegroundColor Green
foreach ($backend in $Backends) {
    $checks = $all[$backend].runs | Where-Object { $_.checksum }
    foreach ($check in $checks) {
        "{0,-10} sentence {1}  duration={2:N4}  textEmb={3}" -f `
            $backend, $check.sentence, $check.duration, $check.textEmb | Write-Host
    }
}

($all | ConvertTo-Json -Depth 6) | Set-Content -Path $reportPath -Encoding utf8
Write-Host ""
Write-Host "Report: $reportPath"
