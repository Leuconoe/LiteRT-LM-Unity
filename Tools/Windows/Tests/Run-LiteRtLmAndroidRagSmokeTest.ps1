<#
.SYNOPSIS
Run the RAG sample (query embedder + KorQuAD-dev index + gemma-4-E2B) on the physical Android device and
bring back per-question retrieval hits, embed / retrieve / LLM timings (TTFT from the turn's BenchmarkInfo)
and answers.

.DESCRIPTION
Pushes one embedder's files from the index folder built by `Run-RagBench.ps1 index` (<embedder>.json,
.query.tflite, tokenizer, .f16) plus passages.json and the LLM into the RAG APK's files/LiteRTLM/rag (its
own application id, so the other sample apps' files are never touched), pushes the questions, launches, waits
for the status file and pulls LiteRtLmRagSmokeTest.results.jsonl. Score answers with
`Run-RagBench.ps1 grade-device <results.jsonl> …`.

No thermal gate (the device idles above 45 °C); the hottest zone is recorded before and after.
Needs pwsh 7. Build the APK with LiteRT-LM/Build/Android/Build RAG Smoke Test APK.

.EXAMPLE
  .\Tools\Windows\Tests\Run-LiteRtLmAndroidRagSmokeTest.ps1 -ApkPath Builds\Android\LiteRtLmAndroidRagSmokeTest.apk -Embedder e5-small -LlmBackend GPU
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$DeviceSerial = "46a880a0",
    [string]$PackageName = "com.Leuconoe.LiteRTLMUnity.rag",
    [ValidateSet("e5-small", "embeddinggemma")][string]$Embedder = "e5-small",
    [string]$IndexDir = "External\rag-work\index",
    [string]$LlmModel = "Assets\StreamingAssets\Multimodal\gemma-4-e2b\gemma-4-E2B-it.litertlm",
    [ValidateSet("GPU", "CPU")][string]$LlmBackend = "GPU",
    [string]$QuestionsFile = "",
    [int]$Limit = 0,
    [int]$TopK = 3,
    [int]$MaxPassageChars = 700,
    [switch]$SkipLlm,
    [string]$ApkPath = "",
    [string]$OutDir = "",
    [int]$TimeoutSeconds = 7200
)

$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -lt 7) { throw "Run with pwsh 7 (Windows PowerShell turns adb stderr into terminating errors)." }
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Push-Location $ProjectRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$utf8 = [System.Text.UTF8Encoding]::new($false)

if (-not $OutDir) { $OutDir = Join-Path $ProjectRoot "Builds\Logs\AndroidRagSmoke\$(Get-Date -Format yyyyMMdd-HHmmss)-$Embedder-$LlmBackend" }
$OutDir = (New-Item -ItemType Directory -Force $OutDir).FullName
if (-not $QuestionsFile) { $QuestionsFile = Join-Path $IndexDir "questions.jsonl" }

$deviceFiles = "/sdcard/Android/data/$PackageName/files"
$deviceRag = "$deviceFiles/LiteRTLM/rag"
$deviceStatusPath = "$deviceFiles/LiteRtLmRagSmokeTest.status.txt"
$deviceResultsPath = "$deviceFiles/LiteRtLmRagSmokeTest.results.jsonl"

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
    if ($deviceMd5 -ne $localMd5) { Invoke-Adb @("push", (Resolve-Path $Local).Path, $Remote) | Out-Null; Write-Host "[RAG] pushed $(Split-Path $Local -Leaf)" }
}

try {
    $devices = @(adb devices | Select-Object -Skip 1)
    if (-not ($devices | Where-Object { $_ -match "^$([regex]::Escape($DeviceSerial))\s+device" })) { throw "Device not connected/authorized: $DeviceSerial" }
    foreach ($other in @("com.Leuconoe.LiteRTLMUnity", "com.Leuconoe.LiteRTLMUnity.sensevoice")) {
        $busy = "$(& adb -s $DeviceSerial shell pidof $other 2>$null)".Trim()
        if ($busy) { throw "$other (pid $busy) is running — not starting a parallel device run." }
    }
    if ($ApkPath) { Write-Host "[RAG] installing $ApkPath"; Invoke-Adb @("install", "-r", "-d", "-t", (Resolve-Path $ApkPath).Path) }

    $meta = Get-Content (Join-Path $IndexDir "$Embedder.json") -Raw -Encoding utf8 | ConvertFrom-Json
    Invoke-AdbBestEffort @("shell", "mkdir", "-p", $deviceRag)
    foreach ($name in @("$Embedder.json", $meta.queryGraph, $meta.tokenizer, $meta.index, "passages.json")) {
        Push-IfChanged (Join-Path $IndexDir $name) "$deviceRag/$name"
    }
    if (-not $SkipLlm) { Push-IfChanged $LlmModel "$deviceRag/$(Split-Path $LlmModel -Leaf)" }

    $questions = @(Get-Content $QuestionsFile -Encoding utf8 | Where-Object { $_.Trim() })
    if ($Limit -gt 0) { $questions = @($questions | Select-Object -First $Limit) }
    $localQuestions = Join-Path $OutDir "questions.jsonl"
    [IO.File]::WriteAllText($localQuestions, ($questions -join "`n") + "`n", $utf8)
    Invoke-Adb @("push", $localQuestions, "$deviceFiles/LiteRtLmRagSmokeTest.questions.jsonl") | Out-Null

    $config = [ordered]@{
        embedderFile = "LiteRTLM/rag/$Embedder.json"
        passagesFile = "LiteRTLM/rag/passages.json"
        questionsFile = "LiteRtLmRagSmokeTest.questions.jsonl"
        llmModelFile = "LiteRTLM/rag/$(Split-Path $LlmModel -Leaf)"
        llmBackend = $LlmBackend
        maxNumTokens = 2048
        topK = $TopK
        maxPassageChars = $MaxPassageChars
        skipLlm = [bool]$SkipLlm
    }
    $localConfig = Join-Path $OutDir "config.json"
    [IO.File]::WriteAllText($localConfig, ($config | ConvertTo-Json), $utf8)
    Invoke-Adb @("push", $localConfig, "$deviceFiles/LiteRtLmRagSmokeTest.config.json") | Out-Null

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
        Write-Host "[RAG] no focus yet ($($focus.Trim())), retrying" -ForegroundColor Yellow
        Invoke-AdbBestEffort @("shell", "cmd", "statusbar", "collapse")
        Invoke-AdbBestEffort @("shell", "am", "start", "-n", "$PackageName/com.unity3d.player.UnityPlayerActivity")
    }

    Write-Host "[RAG] $($questions.Count) questions, $Embedder + gemma-4-E2B $LlmBackend, waiting (timeout ${TimeoutSeconds}s)"
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $statusText = ""; $verdict = "TIMEOUT"
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 5
        $statusText = (@(& adb -s $DeviceSerial shell cat $deviceStatusPath 2>$null) -join "`n")
        if ($statusText -match "(?m)\]\s+(SUCCESS|FAILURE|SKIP):") { $verdict = $Matches[1]; break }
    }
    $tempAfter = Get-MaxTemperature
    [IO.File]::WriteAllText((Join-Path $OutDir "status.txt"), $statusText, $utf8)
    $localResults = Join-Path $OutDir "results.jsonl"
    Invoke-AdbBestEffort @("pull", $deviceResultsPath, $localResults)
    Invoke-AdbBestEffort @("shell", "am", "force-stop", $PackageName)
    # The headset's home launcher holds KEEP_SCREEN_ON, so the screen never times out on its own: sleep it explicitly.
    Invoke-AdbBestEffort @("shell", "input", "keyevent", "KEYCODE_SLEEP")

    Write-Host ""
    Write-Host "=== RAG smoke ($verdict) ==="
    $statusText -split "`n" | Where-Object { $_ -match "\]\s+(INITIALIZED|PROMPT_SET_DONE|FAILURE):" } | ForEach-Object { Write-Host "  $_" }
    if (Test-Path $localResults) {
        $rows = @(Get-Content $localResults -Encoding utf8 | Where-Object { $_ } | ForEach-Object { $_ | ConvertFrom-Json })
        $ok = @($rows | Where-Object { -not $_.error })
        function Get-Stats([double[]]$v) { $s = @($v | Sort-Object); if (-not $s.Count) { return "n/a" }
            "mean {0:0.0} | p50 {1:0.0} | p95 {2:0.0}" -f ($s | Measure-Object -Average).Average, $s[[math]::Floor(0.5 * $s.Count)], $s[[math]::Min($s.Count - 1, [math]::Floor(0.95 * $s.Count))] }
        "results {0}/{1}, errors {2}" -f $rows.Count, $questions.Count, ($rows.Count - $ok.Count) | Write-Host
        if ($ok.Count) {
            "hit@1 {0:0.00}  hit@3 {1:0.00}  (desktop {2:0.00} / {3:0.00})" -f ($ok | Measure-Object hit1 -Average).Average, ($ok | Measure-Object hit3 -Average).Average, $meta.desktopHit1, $meta.desktopHit3 | Write-Host
        }
        "embed ms:    $(Get-Stats @($ok | ForEach-Object { [double]$_.embedMs }))" | Write-Host
        "retrieve ms: $(Get-Stats @($ok | ForEach-Object { [double]$_.retrieveMs }))" | Write-Host
        if (-not $SkipLlm) {
            "TTFT s:      $(Get-Stats @($ok | ForEach-Object { [double]$_.ttftSeconds }))" | Write-Host
            "LLM ms:      $(Get-Stats @($ok | ForEach-Object { [double]$_.llmMs }))" | Write-Host
        }
    }
    "max thermal zone: before {0} C, after {1} C" -f $tempBefore, $tempAfter | Write-Host
    Write-Host "out: $OutDir"
    if ($verdict -ne "SUCCESS") { exit 1 }
}
finally { Pop-Location }
