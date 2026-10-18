<#
.SYNOPSIS
Run the Jev sample ("jevlike offline": gemma-3 trunk + decision head) on the physical
Android device and bring back per-prompt probabilities and trunk / head timings.

.DESCRIPTION
Pushes the exported trunk .tflite, tokenizer.model and jev_head.json/.bin into the Jev APK's
files/LiteRTLM/jev (the Jev APK has its own application id, so the FC smoke APK's
files/LiteRTLM/model.litertlm is never touched), pushes the prompts, launches, waits for the
status file, pulls LiteRtLmJevSmokeTest.results.jsonl and prints mean / p50 / p95 trunk, head
and wall ms. Without -PromptsFile the runner scores its five demo commands (status file).

No thermal gate (the device idles above 45 °C); the hottest zone is recorded before and after.
Needs pwsh 7. Build the APK with LiteRT-LM/Build/Android/Build Jev Smoke Test APK.

.EXAMPLE
  .\Tools\Windows\Tests\Run-LiteRtLmAndroidJevSmokeTest.ps1 -ApkPath Builds\Android\LiteRtLmAndroidJevSmokeTest.apk `
    -PromptsFile Tools\Research\FunctionGemma\FunctionGemmaBench\heldout3_cases.json
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$DeviceSerial = "46a880a0",
    [string]$PackageName = "com.Leuconoe.LiteRTLMUnity.jev",
    [string]$Trunk = "External\jev-work\trunk\gemma3_270m_l12_t48_w8.tflite",
    [string]$Tokenizer = "External\jev-work\head\tokenizer.model",
    [string]$Head = "External\jev-work\head-w8\jev_head.json",
    [string]$PromptsFile = "",
    [int]$Limit = 0,
    [string]$ApkPath = "",
    [string]$OutDir = "",
    [int]$TimeoutSeconds = 1800
)

$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -lt 7) { throw "Run with pwsh 7 (Windows PowerShell turns adb stderr into terminating errors)." }
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Push-Location $ProjectRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$utf8 = [System.Text.UTF8Encoding]::new($false)

if (-not $OutDir) { $OutDir = Join-Path $ProjectRoot "Builds\Logs\AndroidJevSmoke\$(Get-Date -Format yyyyMMdd-HHmmss)" }
$OutDir = (New-Item -ItemType Directory -Force $OutDir).FullName

$deviceFiles = "/sdcard/Android/data/$PackageName/files"
$deviceJev = "$deviceFiles/LiteRTLM/jev"
$deviceStatusPath = "$deviceFiles/LiteRtLmJevSmokeTest.status.txt"
$deviceResultsPath = "$deviceFiles/LiteRtLmJevSmokeTest.results.jsonl"

function Invoke-Adb([string[]]$Arguments) {
    & adb -s $DeviceSerial @Arguments
    if ($LASTEXITCODE -ne 0) { throw "adb failed (${LASTEXITCODE}): adb -s $DeviceSerial $($Arguments -join ' ')" }
}
function Invoke-AdbBestEffort([string[]]$Arguments) { & adb -s $DeviceSerial @Arguments 2>&1 | Out-Null }
function Get-MaxTemperature {
    $lines = & adb -s $DeviceSerial shell 'for z in /sys/class/thermal/thermal_zone*; do cat $z/temp; done' 2>$null
    $c = @($lines | Where-Object { "$_".Trim() -match "^[0-9]+$" } | ForEach-Object { $v = [double]"$_".Trim(); if ($v -gt 1000) { $v / 1000 } else { $v } } |
        Where-Object { $_ -gt 1 -and $_ -lt 120 })
    if ($c.Count -eq 0) { return $null }
    return [math]::Round(($c | Measure-Object -Maximum).Maximum, 1)
}
function Push-IfChanged([string]$Local, [string]$Remote) {
    $localMd5 = (Get-FileHash $Local -Algorithm MD5).Hash.ToLowerInvariant()
    $deviceMd5 = "$(& adb -s $DeviceSerial shell md5sum $Remote 2>$null)".Trim().Split(" ")[0]
    if ($deviceMd5 -ne $localMd5) { Invoke-Adb @("push", (Resolve-Path $Local).Path, $Remote) | Out-Null; Write-Host "[Jev] pushed $(Split-Path $Local -Leaf)" }
}

try {
    $devices = @(adb devices | Select-Object -Skip 1)
    if (-not ($devices | Where-Object { $_ -match "^$([regex]::Escape($DeviceSerial))\s+device" })) { throw "Device not connected/authorized: $DeviceSerial" }
    $busy = "$(& adb -s $DeviceSerial shell pidof com.Leuconoe.LiteRTLMUnity 2>$null)".Trim()
    if ($busy) { throw "The FC smoke app (com.Leuconoe.LiteRTLMUnity, pid $busy) is running — not starting a parallel device run." }
    if ($ApkPath) { Write-Host "[Jev] installing $ApkPath"; Invoke-Adb @("install", "-r", "-d", "-t", (Resolve-Path $ApkPath).Path) }

    Invoke-AdbBestEffort @("shell", "mkdir", "-p", $deviceJev)
    Push-IfChanged $Trunk "$deviceJev/$(Split-Path $Trunk -Leaf)"
    Push-IfChanged $Tokenizer "$deviceJev/tokenizer.model"
    Push-IfChanged $Head "$deviceJev/jev_head.json"
    Push-IfChanged ([IO.Path]::ChangeExtension($Head, ".bin")) "$deviceJev/jev_head.bin"

    $config = [ordered]@{
        trunkFile = "LiteRTLM/jev/$(Split-Path $Trunk -Leaf)"
        tokenizerFile = "LiteRTLM/jev/tokenizer.model"
        headFile = "LiteRTLM/jev/jev_head.json"
        promptsFile = ""
        warmupRuns = 1
    }
    $prompts = @()
    if ($PromptsFile) {
        $path = (Resolve-Path $PromptsFile).Path
        $prompts = if ($path -like "*.json") { @((Get-Content $path -Raw -Encoding utf8 | ConvertFrom-Json).cases | ForEach-Object { $_.text }) }
                   else { @(Get-Content $path -Encoding utf8 | Where-Object { $_.Trim() }) }
        if ($Limit -gt 0) { $prompts = @($prompts | Select-Object -First $Limit) }
        $localPrompts = Join-Path $OutDir "prompts.txt"
        [IO.File]::WriteAllText($localPrompts, ($prompts -join "`n") + "`n", $utf8)
        Invoke-Adb @("push", $localPrompts, "$deviceFiles/LiteRtLmJevSmokeTest.prompts.txt") | Out-Null
        $config.promptsFile = "LiteRtLmJevSmokeTest.prompts.txt"
    }
    $localConfig = Join-Path $OutDir "config.json"
    [IO.File]::WriteAllText($localConfig, ($config | ConvertTo-Json), $utf8)
    Invoke-Adb @("push", $localConfig, "$deviceFiles/LiteRtLmJevSmokeTest.config.json") | Out-Null

    Invoke-AdbBestEffort @("shell", "am", "force-stop", $PackageName)
    Invoke-AdbBestEffort @("shell", "rm", "-f", $deviceStatusPath, $deviceResultsPath)
    $tempBefore = Get-MaxTemperature

    # Unity pauses without window focus and then writes no status file at all.
    Invoke-AdbBestEffort @("shell", "input", "keyevent", "KEYCODE_WAKEUP")
    Invoke-AdbBestEffort @("shell", "wm", "dismiss-keyguard")
    Invoke-AdbBestEffort @("shell", "cmd", "statusbar", "collapse")
    Invoke-AdbBestEffort @("shell", "monkey", "-p", $PackageName, "-c", "android.intent.category.LAUNCHER", "1")
    Start-Sleep -Seconds 5
    $focus = "$(& adb -s $DeviceSerial shell dumpsys window 2>$null | Select-String -Pattern 'mCurrentFocus' | Select-Object -First 1)"
    if ($focus -notmatch [regex]::Escape($PackageName)) {
        Write-Host "[Jev] no focus yet ($($focus.Trim())), retrying" -ForegroundColor Yellow
        Invoke-AdbBestEffort @("shell", "cmd", "statusbar", "collapse")
        Invoke-AdbBestEffort @("shell", "am", "start", "-n", "$PackageName/com.unity3d.player.UnityPlayerActivity")
    }

    Write-Host "[Jev] $($prompts.Count) prompts, waiting (timeout ${TimeoutSeconds}s)"
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $statusText = ""; $verdict = "TIMEOUT"
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 3
        $statusText = (@(& adb -s $DeviceSerial shell cat $deviceStatusPath 2>$null) -join "`n")
        if ($statusText -match "(?m)\]\s+(SUCCESS|FAILURE|SKIP):") { $verdict = $Matches[1]; break }
    }
    $tempAfter = Get-MaxTemperature
    [IO.File]::WriteAllText((Join-Path $OutDir "status.txt"), $statusText, $utf8)
    $localResults = Join-Path $OutDir "results.jsonl"
    if ($prompts.Count) { Invoke-AdbBestEffort @("pull", $deviceResultsPath, $localResults) }
    Invoke-AdbBestEffort @("shell", "am", "force-stop", $PackageName)
    # The headset's home launcher holds KEEP_SCREEN_ON, so the screen never times out on its own: sleep it explicitly.
    Invoke-AdbBestEffort @("shell", "input", "keyevent", "KEYCODE_SLEEP")

    Write-Host ""
    Write-Host "=== Jev smoke ($verdict) ==="
    $statusText -split "`n" | Where-Object { $_ -match "\]\s+(INITIALIZED|DEMO|PROMPT_SET_DONE|FAILURE):" } | ForEach-Object { Write-Host "  $_" }
    if ($prompts.Count -and (Test-Path $localResults)) {
        $rows = @(Get-Content $localResults -Encoding utf8 | Where-Object { $_ } | ForEach-Object { $_ | ConvertFrom-Json })
        $ok = @($rows | Where-Object { -not $_.error })
        function Get-Stats([double[]]$v) { $s = @($v | Sort-Object); if (-not $s.Count) { return "n/a" }
            "mean {0:0.0} | p50 {1:0.0} | p95 {2:0.0}" -f ($s | Measure-Object -Average).Average, $s[[math]::Floor(0.5 * $s.Count)], $s[[math]::Min($s.Count - 1, [math]::Floor(0.95 * $s.Count))] }
        "results {0}/{1}, errors {2}" -f $rows.Count, $prompts.Count, ($rows.Count - $ok.Count) | Write-Host
        "trunk ms: $(Get-Stats @($ok | ForEach-Object { [double]$_.trunkMs }))" | Write-Host
        "head ms:  $(Get-Stats @($ok | ForEach-Object { [double]$_.headMs }))" | Write-Host
        "wall ms:  $(Get-Stats @($ok | ForEach-Object { [double]$_.wallMs }))" | Write-Host
    }
    "max thermal zone: before {0} C, after {1} C" -f $tempBefore, $tempAfter | Write-Host
    Write-Host "out: $OutDir"
    if ($verdict -ne "SUCCESS") { exit 1 }
}
finally { Pop-Location }
