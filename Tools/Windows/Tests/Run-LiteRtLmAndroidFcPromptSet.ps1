<#
.SYNOPSIS
Run a list of function-calling prompts through a .litertlm bundle on the physical
Android device and bring back every raw response with its wall time.

.DESCRIPTION
Drives the Android smoke runner's prompt-set mode (`promptsFile` in its runtime
config): pushes the bundle (skipped when the device copy already has the same
size), the system instruction and the prompts, launches the app, waits for the
status file, pulls `LiteRtLmAndroidSmokeTest.results.jsonl` and prints mean / p50 /
p95 wall ms. Grading is done on the desktop afterwards (fg_common), so the raw
text is kept untouched.

There is no thermal gate (the device idles above the 45 °C the benchmark script
waits for); the hottest thermal zone is recorded before and after instead.

Needs pwsh 7. The installed APK must contain the prompt-set runner — build it from
LiteRtLmBuild (smoke APK with packageModel=false) and pass -ApkPath once.

.EXAMPLE
  .\Tools\Windows\Tests\Run-LiteRtLmAndroidFcPromptSet.ps1 `
    -Bundle External\functiongemma-work\convert\stock\functiongemma-stock-ekv1024.litertlm `
    -SystemInstructionFile sys.txt -PromptsFile Tools\Research\FunctionGemma\FunctionGemmaBench\sample_eval_cases.json -Limit 10
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$DeviceSerial = "46a880a0",
    [string]$PackageName = "com.Leuconoe.LiteRTLMUnity",
    [Parameter(Mandatory = $true)][string]$Bundle,
    [string]$SystemInstructionFile = "",
    [Parameter(Mandatory = $true)][string]$PromptsFile,
    [int]$Limit = 0,
    [int]$MaxNumTokens = 1024,
    [string]$Backend = "CPU",
    [int]$BenchmarkPrefillTokens = 0,
    [int]$BenchmarkDecodeTokens = 32,
    [string]$ApkPath = "",
    [string]$OutDir = "",
    [int]$TimeoutSeconds = 1800
)

$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -lt 7) { throw "Run with pwsh 7 (Windows PowerShell turns adb stderr into terminating errors)." }
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$utf8 = [System.Text.UTF8Encoding]::new($false)

$runId = Get-Date -Format "yyyyMMdd-HHmmss"
if (-not $OutDir) { $OutDir = Join-Path $ProjectRoot "Builds\Logs\AndroidFcPromptSet\$runId" }
$OutDir = (New-Item -ItemType Directory -Force $OutDir).FullName  # absolute: [IO.File] resolves against the process cwd, not $PWD

$deviceFiles = "/sdcard/Android/data/$PackageName/files"
$deviceStatusPath = "$deviceFiles/LiteRtLmAndroidSmokeTest.status.txt"
$deviceResultsPath = "$deviceFiles/LiteRtLmAndroidSmokeTest.results.jsonl"
$bundleName = Split-Path $Bundle -Leaf
$deviceBundlePath = "$deviceFiles/LiteRTLM/$bundleName"

function Invoke-Adb([string[]]$Arguments) {
    & adb -s $DeviceSerial @Arguments
    if ($LASTEXITCODE -ne 0) { throw "adb failed (${LASTEXITCODE}): adb -s $DeviceSerial $($Arguments -join ' ')" }
}
function Invoke-AdbBestEffort([string[]]$Arguments) { & adb -s $DeviceSerial @Arguments 2>&1 | Out-Null }
function Get-MaxTemperature {
    # ponytail: hottest plausible zone only; per-zone breakdown lives in Run-LiteRtLmAndroidDeviceBenchmarks.ps1
    $lines = & adb -s $DeviceSerial shell 'for z in /sys/class/thermal/thermal_zone*; do cat $z/temp; done' 2>$null
    $c = @($lines | Where-Object { "$_".Trim() -match "^[0-9]+$" } | ForEach-Object { $v = [double]"$_".Trim(); if ($v -gt 1000) { $v / 1000 } else { $v } } |
        Where-Object { $_ -gt 1 -and $_ -lt 120 })
    if ($c.Count -eq 0) { return $null }
    return [math]::Round(($c | Measure-Object -Maximum).Maximum, 1)
}

$devices = @(adb devices | Select-Object -Skip 1)
if (-not ($devices | Where-Object { $_ -match "^$([regex]::Escape($DeviceSerial))\s+device" })) { throw "Device not connected/authorized: $DeviceSerial" }
if ($ApkPath) { Write-Host "[FC] installing $ApkPath"; Invoke-Adb @("install", "-r", "-d", "-t", (Resolve-Path $ApkPath).Path) }

# Prompts: plain text one per line, or a cases JSON whose `cases[].text` are the prompts.
$promptsPath = (Resolve-Path $PromptsFile).Path
$prompts = if ($promptsPath -like "*.json") {
    @((Get-Content $promptsPath -Raw -Encoding utf8 | ConvertFrom-Json).cases | ForEach-Object { $_.text })
} else {
    @(Get-Content $promptsPath -Encoding utf8 | Where-Object { $_.Trim() })
}
if ($Limit -gt 0) { $prompts = @($prompts | Select-Object -First $Limit) }
$localPrompts = Join-Path $OutDir "prompts.txt"
[System.IO.File]::WriteAllText($localPrompts, ($prompts -join "`n") + "`n", $utf8)

$config = [ordered]@{
    modelPath = $bundleName
    backend = $Backend
    maxNumTokens = $MaxNumTokens
    maxNumImages = 0
    enableSpeculativeDecoding = $false
    runStandaloneBenchmark = ($BenchmarkPrefillTokens -gt 0)
    benchmarkPrefillTokens = [Math]::Max(1, $BenchmarkPrefillTokens)
    benchmarkDecodeTokens = $BenchmarkDecodeTokens
    benchmarkRuns = 3
    resetConversationBeforeEachPrompt = $true
    promptsFile = "LiteRtLmAndroidSmokeTest.prompts.txt"
}
if ($SystemInstructionFile) { $config.systemInstruction = [System.IO.File]::ReadAllText((Resolve-Path $SystemInstructionFile).Path, $utf8) }
$localConfig = Join-Path $OutDir "config.json"
[System.IO.File]::WriteAllText($localConfig, ($config | ConvertTo-Json -Depth 3), $utf8)

Invoke-AdbBestEffort @("shell", "mkdir", "-p", "$deviceFiles/LiteRTLM")
# md5, not size: a fine-tuned and a stock int8 bundle are byte-for-byte the same length.
$localMd5 = (Get-FileHash $Bundle -Algorithm MD5).Hash.ToLowerInvariant()
$deviceMd5 = "$(& adb -s $DeviceSerial shell md5sum $deviceBundlePath 2>$null)".Trim().Split(" ")[0]
if ($deviceMd5 -ne $localMd5) { Write-Host "[FC] pushing bundle ($([math]::Round((Get-Item $Bundle).Length / 1MB)) MB)"; Invoke-Adb @("push", (Resolve-Path $Bundle).Path, $deviceBundlePath) }
else { Write-Host "[FC] bundle already on device ($bundleName, md5 match)" }
Invoke-Adb @("push", $localConfig, "$deviceFiles/LiteRtLmAndroidSmokeTest.config.json")
Invoke-Adb @("push", $localPrompts, "$deviceFiles/LiteRtLmAndroidSmokeTest.prompts.txt")

Invoke-AdbBestEffort @("shell", "am", "force-stop", $PackageName)
Invoke-AdbBestEffort @("shell", "rm", "-f", $deviceStatusPath, $deviceResultsPath)
$tempBefore = Get-MaxTemperature

# Unity pauses without window focus and then writes no status file at all.
Invoke-AdbBestEffort @("shell", "input", "keyevent", "KEYCODE_WAKEUP")
Invoke-AdbBestEffort @("shell", "wm", "dismiss-keyguard")
Invoke-AdbBestEffort @("shell", "cmd", "statusbar", "collapse")
Invoke-AdbBestEffort @("shell", "monkey", "-p", $PackageName, "-c", "android.intent.category.LAUNCHER", "1")  # failure shows up in the focus check
Start-Sleep -Seconds 5
$focus = "$(& adb -s $DeviceSerial shell dumpsys window 2>$null | Select-String -Pattern 'mCurrentFocus' | Select-Object -First 1)"
if ($focus -notmatch [regex]::Escape($PackageName)) {
    Write-Host "[FC] no focus yet ($($focus.Trim())), retrying" -ForegroundColor Yellow
    Invoke-AdbBestEffort @("shell", "cmd", "statusbar", "collapse")
    Invoke-AdbBestEffort @("shell", "am", "start", "-n", "$PackageName/com.unity3d.player.UnityPlayerActivity")
}

Write-Host "[FC] $($prompts.Count) prompts, waiting (timeout ${TimeoutSeconds}s)"
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$statusText = ""; $verdict = "TIMEOUT"
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 3
    $statusText = (@(& adb -s $DeviceSerial shell cat $deviceStatusPath 2>$null) -join "`n")
    if ($statusText -match "(?m)\]\s+(SUCCESS|FAILURE|SKIP):") { $verdict = $Matches[1]; break }
}
$tempAfter = Get-MaxTemperature
[System.IO.File]::WriteAllText((Join-Path $OutDir "status.txt"), $statusText, $utf8)
$localResults = Join-Path $OutDir "results.jsonl"
Invoke-AdbBestEffort @("pull", $deviceResultsPath, $localResults)
Invoke-AdbBestEffort @("shell", "am", "force-stop", $PackageName)
# The headset's home launcher holds KEEP_SCREEN_ON, so the screen never times out on its own: sleep it explicitly.
Invoke-AdbBestEffort @("shell", "input", "keyevent", "KEYCODE_SLEEP")

if (-not $statusText) {
    Write-Host "[FC] no status file. focus: $("$(& adb -s $DeviceSerial shell dumpsys window 2>$null | Select-String -Pattern 'mCurrentFocus' | Select-Object -First 1)".Trim())" -ForegroundColor Yellow
}
$rows = if (Test-Path $localResults) { @(Get-Content $localResults -Encoding utf8 | Where-Object { $_ } | ForEach-Object { $_ | ConvertFrom-Json }) } else { @() }
$ok = @($rows | Where-Object { -not $_.error })
$ms = @($ok | ForEach-Object { [double]$_.wallMs } | Sort-Object)
function Get-Pct([double[]]$Sorted, [double]$P) { if ($Sorted.Count -eq 0) { return 0 }; $Sorted[[math]::Min($Sorted.Count - 1, [math]::Floor($P * $Sorted.Count))] }

Write-Host ""
Write-Host "=== FC prompt set ($verdict) ==="
$statusText -split "`n" | Where-Object { $_ -match "\]\s+(INITIALIZED|BENCHMARK_RESULT|PROMPT_SET_DONE|FAILURE):" } | ForEach-Object { Write-Host "  $_" }
"results {0}/{1}, errors {2}, function-call text in {3}" -f $rows.Count, $prompts.Count, ($rows.Count - $ok.Count), @($ok | Where-Object { $_.response -match "<start_function_call>" }).Count | Write-Host
if ($ms.Count) { "wall ms: mean {0:0} | p50 {1:0} | p95 {2:0} | max {3:0}" -f ($ms | Measure-Object -Average).Average, (Get-Pct $ms 0.5), (Get-Pct $ms 0.95), $ms[-1] | Write-Host }
"max thermal zone: before {0} C, after {1} C" -f $tempBefore, $tempAfter | Write-Host
Write-Host "out: $OutDir"
if ($verdict -ne "SUCCESS") { exit 1 }
