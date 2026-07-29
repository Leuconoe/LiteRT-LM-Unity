[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$ModelPath = "",
    [string]$AudioPath = "",
    [string]$Prompt = "Transcribe the audio: {audio}",
    [ValidateSet("cpu", "gpu")]
    [string]$Backend = "cpu",
    [ValidateSet("cpu", "gpu")]
    [string]$AudioBackend = "cpu",
    [int]$TimeoutSeconds = 600,
    [switch]$Benchmark
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
if ([string]::IsNullOrWhiteSpace($ModelPath)) {
    $ModelPath = Join-Path $ProjectRoot "Assets\StreamingAssets\Multimodal\gemma-4-e2b\gemma-4-E2B-it.litertlm"
}
if ([string]::IsNullOrWhiteSpace($AudioPath)) {
    $AudioPath = Join-Path $ProjectRoot "Assets\StreamingAssets\TestAssets\Audio\2025년 3월 5일 전술평가 결과 보고.mp3"
}

$ExecutablePath = Join-Path $PSScriptRoot "..\Bin\litert_lm_advanced_main.windows_x86_64.exe"
if (!(Test-Path $ExecutablePath)) {
    throw "Advanced CLI not found: $ExecutablePath"
}
if (!(Test-Path $ModelPath)) {
    throw "Model bundle not found: $ModelPath"
}
if (!(Test-Path $AudioPath)) {
    throw "Audio file not found: $AudioPath"
}

$LogDirectory = Join-Path $ProjectRoot "Builds\Logs\WindowsAsrSmoke"
New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
$RunId = Get-Date -Format "yyyyMMdd-HHmmss"
$rawLog = Join-Path $LogDirectory "$RunId-windows-asr-smoke.log"
$summaryLog = Join-Path $LogDirectory "$RunId-windows-asr-smoke.summary.txt"

# The CLI media tag regex is \[(image|audio):([^\s\]]+)\] — whitespace in the
# audio path breaks tag detection, so stage the audio under a space-free path.
$resolvedAudioPath = (Resolve-Path $AudioPath).Path
if ($resolvedAudioPath -match "\s") {
    $extension = [System.IO.Path]::GetExtension($resolvedAudioPath)
    $stagedAudioPath = Join-Path ([System.IO.Path]::GetTempPath()) "litertlm-asr-smoke-$RunId$extension"
    Copy-Item -Path $resolvedAudioPath -Destination $stagedAudioPath -Force
    Write-Host "Audio path contains whitespace; staged copy: $stagedAudioPath"
    $resolvedAudioPath = $stagedAudioPath
}

$audioTag = "[audio:" + ($resolvedAudioPath -replace "\\", "/") + "]"
if ($Prompt.Contains("{audio}")) {
    $resolvedPrompt = $Prompt.Replace("{audio}", $audioTag)
}
else {
    $resolvedPrompt = "$Prompt $audioTag"
}

# Pass the prompt via UTF-8 file so Korean text and the media tag survive
# console encoding.
$promptFile = Join-Path ([System.IO.Path]::GetTempPath()) "litertlm-asr-smoke-$RunId-prompt.txt"
[System.IO.File]::WriteAllText($promptFile, $resolvedPrompt, [System.Text.UTF8Encoding]::new($false))

Write-Host "Model     : $ModelPath"
Write-Host "Audio     : $resolvedAudioPath"
Write-Host "Backend   : $Backend (audio: $AudioBackend)"
Write-Host "Prompt    : $resolvedPrompt"
Write-Host "Raw log   : $rawLog"

$arguments = @(
    "--backend=$Backend",
    "--audio_backend=$AudioBackend",
    "--model_path=$ModelPath",
    "--input_prompt_file=$promptFile"
)
if ($Benchmark) {
    $arguments += "--benchmark"
}

$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName = $ExecutablePath
$startInfo.Arguments = ($arguments | ForEach-Object { '"' + $_ + '"' }) -join ' '
$startInfo.WorkingDirectory = $PSScriptRoot
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
$startInfo.StandardOutputEncoding = [System.Text.Encoding]::UTF8
$startInfo.StandardErrorEncoding = [System.Text.Encoding]::UTF8
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true

$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$process = [System.Diagnostics.Process]::Start($startInfo)
$stdoutTask = $process.StandardOutput.ReadToEndAsync()
$stderrTask = $process.StandardError.ReadToEndAsync()
if (!$process.WaitForExit($TimeoutSeconds * 1000)) {
    try { $process.Kill($true) } catch {}
    Remove-Item -Path $promptFile -Force -ErrorAction SilentlyContinue
    throw "ASR smoke run timed out after $TimeoutSeconds seconds."
}
$stopwatch.Stop()
$stdout = $stdoutTask.Result
$stderr = $stderrTask.Result
$exitCode = $process.ExitCode
Remove-Item -Path $promptFile -Force -ErrorAction SilentlyContinue

Set-Content -Path $rawLog -Value ($stdout + "`n--- STDERR ---`n" + $stderr) -Encoding utf8

# Extract the transcript: drop runtime log lines and the metadata dump, keep
# text between the settings dump and the BenchmarkInfo block.
$transcriptLines = @()
$inBenchmark = $false
foreach ($line in ($stdout -split "`r?`n")) {
    if ($line.StartsWith("BenchmarkInfo:")) { $inBenchmark = $true }
    if ($inBenchmark) { continue }
    if ($line -match "^(INFO|WARNING|ERROR):" -or $line -match "^[IWE]\d{4}\s") { continue }
    $transcriptLines += $line
}
$transcript = (($transcriptLines -join "`n").Trim() -split "`n" | Select-Object -Last 8) -join "`n"

$initTotal = if ($stdout -match "Init Total:\s*([\d\.]+)\s*ms") { [double]$Matches[1] / 1000 } else { $null }
$prefillTps = if ($stdout -match "Prefill Speed:\s*([\d\.]+)\s*tokens/sec") { [double]$Matches[1] } else { $null }
$decodeTps = if ($stdout -match "Decode Speed:\s*([\d\.]+)\s*tokens/sec") { [double]$Matches[1] } else { $null }
$hasAudioSections = $stdout -match "tf_lite_audio_encoder" -or $stderr -match "tf_lite_audio_encoder"

$verdict = if ($exitCode -eq 0 -and $transcript.Length -gt 0) { "PASS" } else { "FAIL" }

$summary = @(
    "verdict=$verdict",
    "exitCode=$exitCode",
    "elapsedSeconds=$([Math]::Round($stopwatch.Elapsed.TotalSeconds, 1))",
    "model=$ModelPath",
    "audio=$resolvedAudioPath",
    "backend=$Backend audioBackend=$AudioBackend",
    "audioSectionsInBundle=$hasAudioSections",
    "initTotalSeconds=$initTotal",
    "prefillTokensPerSec=$prefillTps",
    "decodeTokensPerSec=$decodeTps",
    "transcript=$transcript"
)
Set-Content -Path $summaryLog -Value ($summary -join "`n") -Encoding utf8

Write-Host ""
Write-Host "=== Windows ASR smoke summary ==="
$summary | ForEach-Object { Write-Host $_ }
Write-Host "Summary log: $summaryLog"

if ($verdict -ne "PASS") {
    Write-Host "--- stderr tail ---"
    ($stderr -split "`r?`n" | Select-Object -Last 20) | ForEach-Object { Write-Host $_ }
    exit 1
}
