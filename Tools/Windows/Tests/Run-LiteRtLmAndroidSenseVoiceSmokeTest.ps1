<#
.SYNOPSIS
Run SenseVoice-Small (fixed-shape LiteRT buckets) on the physical Android device and bring back per-clip
transcripts, CTC confidence and feat / infer / decode timings.

.DESCRIPTION
Pushes the model folder (am.mvn, tokens.txt, sensevoice_small_<bucket>_<variant>.tflite) into the SenseVoice
APK's files/LiteRTLM/sensevoice (its own application id, so the other sample apps' files are never touched),
pushes the clips listed in -ClipsFile (`localAudio<TAB>reference` per line; make one with
`sv_device.py clips`), launches, waits for the status file, pulls LiteRtLmSenseVoiceSmokeTest.results.jsonl
(and the feature/logit dumps of the first -DumpCount clips), records the app's total PSS, force-stops the app
and puts the headset to sleep. Score with `sv_device.py score <OutDir>`.

No thermal gate (the device idles above 45 °C); the hottest zone is recorded before and after.
Needs pwsh 7. Build the APK with LiteRT-LM/Build/Android/Build SenseVoice Smoke Test APK.

.EXAMPLE
  .\Tools\Windows\Tests\Run-LiteRtLmAndroidSenseVoiceSmokeTest.ps1 -ApkPath Builds\Android\LiteRtLmAndroidSenseVoiceSmokeTest.apk -ClipsFile External\sensevoice-work\device-clips\clips.tsv -DumpCount 10
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$DeviceSerial = "46a880a0",
    [string]$PackageName = "com.Leuconoe.LiteRTLMUnity.sensevoice",
    [string]$ModelDir = "External\sensevoice-work\hf-release\sensevoice-small-litert",
    [ValidateSet("w8", "i8")][string]$Variant = "w8",
    [string[]]$Buckets = @("3s", "5s", "10s", "30s"),
    [string]$Language = "ko",
    [Parameter(Mandatory = $true)][string]$ClipsFile,
    [int]$DumpCount = 0,
    [string]$ApkPath = "",
    [string]$OutDir = "",
    [int]$TimeoutSeconds = 3600
)

$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -lt 7) { throw "Run with pwsh 7 (Windows PowerShell turns adb stderr into terminating errors)." }
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Push-Location $ProjectRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$utf8 = [System.Text.UTF8Encoding]::new($false)

if (-not $OutDir) { $OutDir = Join-Path $ProjectRoot "Builds\Logs\AndroidSenseVoiceSmoke\$(Get-Date -Format yyyyMMdd-HHmmss)-$Variant" }
$OutDir = (New-Item -ItemType Directory -Force $OutDir).FullName

$deviceFiles = "/sdcard/Android/data/$PackageName/files"
$deviceModel = "$deviceFiles/LiteRTLM/sensevoice"
$deviceClips = "$deviceFiles/LiteRTLM/sensevoice-clips"
$deviceDump = "$deviceFiles/LiteRTLM/sensevoice-dump"
$deviceStatusPath = "$deviceFiles/LiteRtLmSenseVoiceSmokeTest.status.txt"
$deviceResultsPath = "$deviceFiles/LiteRtLmSenseVoiceSmokeTest.results.jsonl"

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
    if ($deviceMd5 -eq $localMd5) { return }
    # One retry: a push among dozens of small clips occasionally fails transiently.
    try { Invoke-Adb @("push", (Resolve-Path $Local).Path, $Remote) 2>&1 | Out-Null }
    catch { Start-Sleep -Seconds 2; Invoke-Adb @("push", (Resolve-Path $Local).Path, $Remote) 2>&1 | Out-Null }
    Write-Host "[SV] pushed $Local"
}

try {
    $devices = @(adb devices | Select-Object -Skip 1)
    if (-not ($devices | Where-Object { $_ -match "^$([regex]::Escape($DeviceSerial))\s+device" })) { throw "Device not connected/authorized: $DeviceSerial" }
    foreach ($other in @("com.Leuconoe.LiteRTLMUnity", "com.Leuconoe.LiteRTLMUnity.rag")) {
        $busy = "$(& adb -s $DeviceSerial shell pidof $other 2>$null)".Trim()
        if ($busy) { throw "$other (pid $busy) is running — not starting a parallel device run." }
    }
    if ($ApkPath) { Write-Host "[SV] installing $ApkPath"; Invoke-Adb @("install", "-r", "-d", "-t", (Resolve-Path $ApkPath).Path) }

    Invoke-AdbBestEffort @("shell", "mkdir", "-p", $deviceModel, $deviceClips)
    $modelFiles = @("am.mvn", "tokens.txt") + @($Buckets | ForEach-Object { "sensevoice_small_${_}_$Variant.tflite" })
    foreach ($name in $modelFiles) { Push-IfChanged (Join-Path $ModelDir $name) "$deviceModel/$name" }

    # Clips: pushed as <index><ext>; the device list keeps the reference for scoring.
    $clips = @(Get-Content $ClipsFile -Encoding utf8 | Where-Object { $_.Trim() } | ForEach-Object { , ($_ -split "`t", 2) })
    $deviceList = [System.Collections.Generic.List[string]]::new()
    for ($i = 0; $i -lt $clips.Count; $i++) {
        $local = $clips[$i][0]
        $name = "$i$([IO.Path]::GetExtension($local))"
        Push-IfChanged $local "$deviceClips/$name"
        $deviceList.Add("LiteRTLM/sensevoice-clips/$name`t$(if ($clips[$i].Count -gt 1) { $clips[$i][1] } else { '' })")
    }
    $localList = Join-Path $OutDir "clips.tsv"
    [IO.File]::WriteAllText($localList, ($deviceList -join "`n") + "`n", $utf8)
    Invoke-Adb @("push", $localList, "$deviceFiles/LiteRtLmSenseVoiceSmokeTest.clips.tsv") | Out-Null
    Copy-Item $ClipsFile (Join-Path $OutDir "clips-local.tsv")

    $config = [ordered]@{
        modelDir = "LiteRTLM/sensevoice"
        variant = $Variant
        language = $Language
        clipsFile = "LiteRtLmSenseVoiceSmokeTest.clips.tsv"
        dumpCount = $DumpCount
    }
    $localConfig = Join-Path $OutDir "config.json"
    [IO.File]::WriteAllText($localConfig, ($config | ConvertTo-Json), $utf8)
    Invoke-Adb @("push", $localConfig, "$deviceFiles/LiteRtLmSenseVoiceSmokeTest.config.json") | Out-Null

    Invoke-AdbBestEffort @("shell", "am", "force-stop", $PackageName)
    Invoke-AdbBestEffort @("shell", "rm", "-rf", $deviceStatusPath, $deviceResultsPath, $deviceDump)
    $tempBefore = Get-MaxTemperature

    # Unity pauses without window focus and then writes no status file at all.
    Invoke-AdbBestEffort @("shell", "input", "keyevent", "KEYCODE_WAKEUP")
    Invoke-AdbBestEffort @("shell", "wm", "dismiss-keyguard")
    Invoke-AdbBestEffort @("shell", "cmd", "statusbar", "collapse")
    Invoke-AdbBestEffort @("shell", "monkey", "-p", $PackageName, "-c", "android.intent.category.LAUNCHER", "1")
    Start-Sleep -Seconds 5
    $focus = "$(& adb -s $DeviceSerial shell dumpsys window 2>$null | Select-String -Pattern 'mCurrentFocus' | Select-Object -First 1)"
    if ($focus -notmatch [regex]::Escape($PackageName)) {
        Write-Host "[SV] no focus yet ($($focus.Trim())), retrying" -ForegroundColor Yellow
        Invoke-AdbBestEffort @("shell", "cmd", "statusbar", "collapse")
        Invoke-AdbBestEffort @("shell", "am", "start", "-n", "$PackageName/com.unity3d.player.UnityPlayerActivity")
    }

    Write-Host "[SV] $($clips.Count) clips, $Variant, waiting (timeout ${TimeoutSeconds}s)"
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $statusText = ""; $verdict = "TIMEOUT"
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 3
        $statusText = (@(& adb -s $DeviceSerial shell cat $deviceStatusPath 2>$null) -join "`n")
        if ($statusText -match "(?m)\]\s+(SUCCESS|FAILURE|SKIP):") { $verdict = $Matches[1]; break }
    }
    $tempAfter = Get-MaxTemperature
    $pss = "$(& adb -s $DeviceSerial shell dumpsys meminfo $PackageName 2>$null | Select-String -CaseSensitive -Pattern 'TOTAL PSS:|^\s+TOTAL\s+\d' | Select-Object -First 1)".Trim()
    [IO.File]::WriteAllText((Join-Path $OutDir "status.txt"), $statusText, $utf8)
    [IO.File]::WriteAllText((Join-Path $OutDir "meminfo.txt"), $pss, $utf8)
    $localResults = Join-Path $OutDir "results.jsonl"
    Invoke-AdbBestEffort @("pull", $deviceResultsPath, $localResults)
    if ($DumpCount -gt 0) { Invoke-AdbBestEffort @("pull", $deviceDump, (Join-Path $OutDir "dump")) }
    Invoke-AdbBestEffort @("shell", "am", "force-stop", $PackageName)
    # The headset's home launcher holds KEEP_SCREEN_ON, so the screen never times out on its own: sleep it explicitly.
    Invoke-AdbBestEffort @("shell", "input", "keyevent", "KEYCODE_SLEEP")

    Write-Host ""
    Write-Host "=== SenseVoice smoke ($verdict) ==="
    $statusText -split "`n" | Where-Object { $_ -match "\]\s+(INITIALIZED|PROMPT_SET_DONE|FAILURE):" } | ForEach-Object { Write-Host "  $_" }
    if (Test-Path $localResults) {
        $rows = @(Get-Content $localResults -Encoding utf8 | Where-Object { $_ } | ForEach-Object { $_ | ConvertFrom-Json })
        $ok = @($rows | Where-Object { -not $_.error })
        function Get-Stats([double[]]$v) { $s = @($v | Sort-Object); if (-not $s.Count) { return "n/a" }
            "mean {0:0.0} | p50 {1:0.0} | p95 {2:0.0}" -f ($s | Measure-Object -Average).Average, $s[[math]::Floor(0.5 * $s.Count)], $s[[math]::Min($s.Count - 1, [math]::Floor(0.95 * $s.Count))] }
        "results {0}/{1}, errors {2}" -f $rows.Count, $clips.Count, ($rows.Count - $ok.Count) | Write-Host
        $warm = @($ok | Where-Object { $_.compileMs -eq 0 })
        "feat ms:  $(Get-Stats @($ok | ForEach-Object { [double]$_.featMs }))" | Write-Host
        "infer ms: $(Get-Stats @($warm | ForEach-Object { [double]$_.inferMs })) (warm, compile excluded)" | Write-Host
        "wall ms:  $(Get-Stats @($warm | ForEach-Object { [double]$_.wallMs }))" | Write-Host
    }
    "PSS: $pss" | Write-Host
    "max thermal zone: before {0} C, after {1} C" -f $tempBefore, $tempAfter | Write-Host
    Write-Host "out: $OutDir"
    if ($verdict -ne "SUCCESS") { exit 1 }
}
finally { Pop-Location }
