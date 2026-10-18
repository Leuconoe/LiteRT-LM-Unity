<#
.SYNOPSIS
  One command from a changed ToolsPrompt.txt to a validated, converted, staged
  FunctionGemma router (decl-mode none: the contract lives in the weights).
.DESCRIPTION
  Every stage logs to Builds\Logs\fg-rebuild-<label>\ and report.md is written at the end,
  also when a stage stops the run.

    1 contract  ToolsPrompt vs FunctionGemmaBench\contract_snapshot.json (written on first run).
                STOPS with a to-do list when make_dataset.py has no generator / SCHEMA entry or
                enum value for the contract, or when fg_common.py / LiteRtLmFunctionGemmaRouter.cs
                miss a callsign or filter name. Eval cases whose expected call no longer
                validates are listed and left out of every score (relabel by hand).
    2 data      make_dataset v4 templates, + the old two-teacher paraphrases that still validate
                (-RefreshParaphrases: regenerate them with vLLM, GPU), + the v6 classes, exact-text
                filtered against every *_cases.json incl. locked\ -> FunctionGemmaBench\data_rebuild_<label>\.
    3 train     GPU guard (>= 12 GB free, no other training; -WaitForGpu polls every 2 min), then the
                v6e recipe through Run-FunctionGemmaEval.ps1 -Train; after a crash it reruns with
                --resume, 3 attempts in all.
    4 gate      fg_eval on eval_cases, batch 1 + clause split: 0 wrong actions and >= -GatePass.
                heldout1-4 and the locked sets are scored for the report (aggregates only).
    5 convert   cache 256 / prefill 64,128; the Windows CLI must reproduce the HF calls.
    6 export    functiongemma_tools.json (Samples~ + Assets/Samples); -Stage also copies the bundle
                to StreamingAssets and repoints FunctionGemmaRouterModel; contract_snapshot.json
                is rewritten after a passed gate only.
    7 device    -Device: eval_cases on 46a880a0, graded with the router logic.

  -Smoke is the CPU plumbing test (CUDA_VISIBLE_DEVICES=-1): ~200 records, 0.05 epoch, 8 cases per
  set, gate reported but not enforced, no staging, no snapshot, no device. Needs pwsh 7.
.EXAMPLE
  .\Tools\Research\FunctionGemma\Rebuild-FunctionGemmaRouter.ps1 -DryRun
  .\Tools\Research\FunctionGemma\Rebuild-FunctionGemmaRouter.ps1 -Smoke
  .\Tools\Research\FunctionGemma\Rebuild-FunctionGemmaRouter.ps1 -WaitForGpu -Stage -Device
#>
[CmdletBinding()]
param(
    [string]$Label = "",
    [string]$ToolsPrompt = "",  # default: ..\VoiceCommand-WebServer\assets\ToolsPrompt.txt
    [switch]$RefreshParaphrases,
    [switch]$WaitForGpu,
    [double]$GatePass = 0.98,
    [switch]$Force,
    [switch]$Stage,
    [switch]$Device,
    [switch]$Smoke,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -lt 7) { throw "Run with pwsh 7 (Windows PowerShell turns native stderr into terminating errors)." }
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$Bench = Join-Path $PSScriptRoot "FunctionGemmaBench"
$Py = Join-Path $Bench ".venv\Scripts\python.exe"
$Check = Join-Path $Bench "rebuild_check.py"
if (-not $Label) { $Label = "$(if ($Smoke) { 'smoke-' } else { 'r' })$(Get-Date -Format 'yyyyMMdd-HHmm')" }
if (-not $ToolsPrompt) { $ToolsPrompt = Join-Path (Split-Path $ProjectRoot) "VoiceCommand-WebServer\assets\ToolsPrompt.txt" }
$ToolsPrompt = (Resolve-Path $ToolsPrompt).Path
$Log = (New-Item -ItemType Directory -Force (Join-Path $ProjectRoot "Builds\Logs\fg-rebuild-$Label")).FullName
$Cases = Join-Path $Log "cases"   # non-stale copies of every case file, written by stage 1
$Data = Join-Path $Bench "data_rebuild_$Label"
$Work = Join-Path $ProjectRoot "External\functiongemma-work\ft-functiongemma-none-rebuild-$Label"
$Merged = Join-Path $Work "merged"
$BundleDir = Join-Path $Work "litertlm-c256-p64"
$BundleName = "functiongemma_270m_ko_${Label}_c256_p64.litertlm"
$Serial = "46a880a0"
$SampleRuntime = @("Packages\com.leuconoe.litert-lm-unity\Samples~\TestScenes\Runtime",
                   "Assets\Samples\LiteRT-LM for Unity\0.14.0\Test Scenes\Runtime") | ForEach-Object { Join-Path $ProjectRoot $_ }
# Paraphrase stage models (docs/benchmarks/functiongemma-korean-fc.md §3): EXAONE writes, Qwen and EXAONE judge.
$ParaphraseGen, $TeacherA, $TeacherB = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-AWQ", "Qwen/Qwen2.5-14B-Instruct-AWQ", "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-AWQ"

[Console]::OutputEncoding = [Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$savedCuda = $env:CUDA_VISIBLE_DEVICES
if ($Smoke -or $DryRun) { $env:CUDA_VISIBLE_DEVICES = "-1" }   # "" still reports CUDA on Windows
$script:Status = "running"

# Runs a command, tees its output to <log>\<name>.log, returns the exit code. -Quiet keeps it off the
# console (the locked sets: their failures are never looked at).
# The command block resolves variables through this function's scope first, so its own names are unusual.
function Invoke-Logged([string]$LoggedStep, [scriptblock]$LoggedCommand, [switch]$LoggedQuiet) {
    $loggedFile = Join-Path $Log "$LoggedStep.log"
    $global:LASTEXITCODE = 0
    if ($LoggedQuiet) { & $LoggedCommand 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 $loggedFile }
    else { & $LoggedCommand 2>&1 | ForEach-Object { "$_" } | Tee-Object -Append -FilePath $loggedFile | Out-Host }
    return $global:LASTEXITCODE
}

function Get-GpuBlocker {
    $free = [int](& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | Select-Object -First 1)
    $onGpu = @(& nvidia-smi --query-compute-apps=pid --format=csv,noheader | ForEach-Object { [int]"$_".Trim() })
    $trainers = @(Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
        Where-Object { $onGpu -contains $_.ProcessId -and $_.CommandLine -match '(?i)train|vllm' })
    if ($trainers) { return "training on the GPU: " + (($trainers | ForEach-Object { "pid $($_.ProcessId) $($_.CommandLine)" }) -join "; ") }
    if ($free -lt 12288) { return "free VRAM $free MiB < 12 GB" }
    return $null
}

function Wait-Gpu {
    while ($blocker = Get-GpuBlocker) {
        if (-not $WaitForGpu) { throw "GPU busy — $blocker. Rerun with -WaitForGpu to poll every 2 min." }
        Write-Host "[GPU] $blocker — next check in 2 min" -ForegroundColor Yellow
        Start-Sleep -Seconds 120
    }
}

function Invoke-Rebuild {
    # ---------------------------------------------------------------- 1 contract
    $code = Invoke-Logged "01-contract" { & $Py $Check contract --tools-prompt $ToolsPrompt --log $Log }
    if ($code -eq 2) { $script:Status = "STOPPED at stage 1: hand edits needed (to-do list above)"; return }
    if ($code -ne 0) { throw "contract check failed ($code)" }
    if ($DryRun) {
        $plan = @("## Plan (dry run — nothing below ran)", "",
            "- 2 data: make_dataset v4 -> $Data\v4; paraphrases $(if ($RefreshParaphrases) { "regenerated ($ParaphraseGen; teachers $TeacherA + $TeacherB; GPU)" } else { 'carried over from data_v5_empty where they still validate' }) -> $Data\v5; v6 composition -> $Data",
            "- 3 train: v6e recipe (3 epochs, lr 2e-5, batch 8, decl none) -> $Merged; GPU guard$(if ($WaitForGpu) { ', waiting' }); --resume retries (3 attempts)",
            "- 4 gate: eval_cases minus stale, batch 1 + clause split: 0 wrong actions and >= $($GatePass.ToString('P0')) pass$(if ($Force) { ' (-Force: not enforced)' })",
            "- 5 convert: cache 256 / prefill 64,128 -> $BundleDir; Windows CLI parity on eval_cases",
            "- 6 export: functiongemma_tools.json$(if ($Stage) { "; stage Assets\StreamingAssets\LLM\functiongemma-270m-ko\$BundleName + FunctionGemmaRouterModel" }); contract_snapshot.json after a passed gate",
            "- 7 device: $(if ($Device) { "eval_cases on $Serial, router-graded" } else { 'skipped (no -Device)' })")
        $plan -join "`n" | Out-File -Encoding utf8 (Join-Path $Log "01-plan.md")
        $plan | Out-Host
        $script:Status = "dry run: stage 1 only"
        return
    }

    # ---------------------------------------------------------------- 2 data
    $md = Join-Path $Bench "make_dataset.py"
    if ((Invoke-Logged "02-data-v4" { & $Py $md --version 4 --filter-all --tools-prompt $ToolsPrompt --out "$Data\v4" }) -ne 0) { throw "make_dataset v4 failed" }
    $oldV4, $oldV5 = (Join-Path $Bench "data_v4_empty"), (Join-Path $Bench "data_v5_empty")
    if ($RefreshParaphrases) {
        Wait-Gpu
        $env:FG_PARAPHRASE_V4, $env:FG_PARAPHRASE_V5 = "$Data\v4", "$Data\v5-refreshed"
        $pp = Join-Path $Bench "paraphrase\paraphrase.py"
        try {
            foreach ($step in @(@("serve", "--model", $ParaphraseGen), @("seeds"), @("gen"), @("serve", "--model", $TeacherA), @("teach", "--label", "A"),
                                @("serve", "--model", $TeacherB), @("teach", "--label", "B"), @("build"))) {
                if ((Invoke-Logged "02-data-paraphrase" { & $Py $pp @step }) -ne 0) { throw "paraphrase.py $($step -join ' ') failed" }
            }
        }
        finally {
            & $Py $pp stop
            Remove-Item Env:FG_PARAPHRASE_V4, Env:FG_PARAPHRASE_V5 -ErrorAction SilentlyContinue
        }
        $oldV4, $oldV5 = "$Data\v4", "$Data\v5-refreshed"
    }
    if ((Invoke-Logged "02-data-carry" { & $Py $Check carry --tools-prompt $ToolsPrompt --old-v4 $oldV4 --old-v5 $oldV5 --new-v4 "$Data\v4" --out "$Data\v5" }) -ne 0) { throw "paraphrase carry-over failed" }
    if ((Invoke-Logged "02-data-v6" { & $Py $md --version 6 --filter-all --tools-prompt $ToolsPrompt --v5 "$Data\v5" --v4 "$Data\v4" --out $Data }) -ne 0) { throw "make_dataset v6 failed" }
    $trainDir = $Data
    if ($Smoke) {
        if ((Invoke-Logged "02-data-subset" { & $Py $Check subset --src $Data --out "$Data\smoke" --n 200 }) -ne 0) { throw "subset failed" }
        $trainDir = "$Data\smoke"
    }
    $summary = Get-Content (Get-ChildItem $Log -Filter "02-data-*.log").FullName -Encoding utf8 |
        Select-String -Pattern '^(exact-text filter|collisions|base v5|train \d+|DefaultResponse share|carry:|subset:|label conflicts|shortfall)' | ForEach-Object { $_.Line }
    (@("## 2. Data", "", "Train set: ``$trainDir``", "", '```') + $summary + '```') -join "`n" | Out-File -Encoding utf8 (Join-Path $Log "02-data.md")

    # ---------------------------------------------------------------- 3 train
    if (-not $Smoke) { Wait-Gpu }
    $epochs = if ($Smoke) { "0.05" } else { "3" }
    $trainArgs = @("--train", "$trainDir\train.jsonl", "--val", "$trainDir\val.jsonl", "--format", "functiongemma", "--decl-mode", "none",
                   "--epochs", $epochs, "--lr", "2e-5", "--batch-size", "8", "--label", "ft-functiongemma-none-rebuild-$Label",
                   "--out", $Work, "--tools-prompt", $ToolsPrompt)
    $evalPs1 = Join-Path $PSScriptRoot "Run-FunctionGemmaEval.ps1"
    $code, $attempt = -1, 0
    while ($code -ne 0 -and $attempt -lt 3) {
        $attempt++
        # fg_train refuses --resume without a checkpoint and a fresh start with one: follow what is on disk.
        $resume = @(Get-ChildItem $Work -Directory -Filter "checkpoint-*" -ErrorAction SilentlyContinue).Count -gt 0
        if ($attempt -gt 1 -and -not $Smoke) { Wait-Gpu }   # a crash is usually a neighbour on the GPU
        Write-Host "[train] attempt $attempt/3$(if ($resume) { ' --resume' })" -ForegroundColor Cyan
        $passThru = $trainArgs + @(if ($resume) { "--resume" })
        $code = Invoke-Logged "03-train" { & $evalPs1 -Train -PassThru $passThru }
    }
    $trainNote = "attempts $attempt, exit $code, epochs $epochs$(if ($Smoke) { ' (CPU smoke)' })"
    $best = (Get-Content (Join-Path $Log "03-train.log") -Encoding utf8 | Select-String -Pattern '^saved .*best' | Select-Object -Last 1).Line
    @("## 3. Train", "", "v6e recipe (lr 2e-5, batch 8, decl none) -> ``$Merged``; $trainNote.", "", "$best") -join "`n" | Out-File -Encoding utf8 (Join-Path $Log "03-train.md")
    if ($code -ne 0) { throw "training failed after $attempt attempts — see 03-train.log" }

    # ---------------------------------------------------------------- 4 gate
    $limit = if ($Smoke) { @("--limit", "8") } else { @() }
    $evalDevice = if ($Smoke) { @("--device", "cpu") } else { @() }   # not $device: names are case-insensitive, -Device lives here
    $specs = @()
    foreach ($caseFile in Get-ChildItem $Cases -Filter "*_cases.json" | Sort-Object { $_.Name -ne "eval_cases.json" }, Name) {
        $setName = $caseFile.BaseName -replace '_cases$', ''
        $setOut = Join-Path $Log "04-hf-$setName.jsonl"
        $locked = Test-Path (Join-Path $Bench "locked\$($caseFile.Name)")
        $code = Invoke-Logged "04-eval-$setName" -LoggedQuiet:$locked {
            & $Py (Join-Path $Bench "fg_eval.py") --model $Merged --format functiongemma --decl-mode none --cases $caseFile.FullName `
                --batch-size 1 --segment --label "rebuild-$Label-$setName" --out $setOut --tools-prompt $ToolsPrompt @evalDevice @limit
        }
        if ($code -ne 0) { throw "fg_eval on $($caseFile.Name) failed ($code)" }
        $specs += "$setName=$setOut"
    }
    $gate = Invoke-Logged "04-score" { & $Py $Check score @specs --gate eval --pass-rate $GatePass --log $Log }
    if ($gate -notin 0, 3) { throw "scoring failed ($gate)" }
    $gatePassed = $gate -eq 0
    if (-not $gatePassed -and -not ($Force -or $Smoke)) { $script:Status = "STOPPED at stage 4: gate failed (-Force to convert anyway)"; return }

    # ---------------------------------------------------------------- 5 convert + parity
    $convert = Join-Path $PSScriptRoot "Convert-FunctionGemmaToLiteRtLm.ps1"
    $null = Invoke-Logged "05-convert" { & $convert -Model $Merged -OutDir $BundleDir -CacheLength 256 -PrefillLengths @(64, 128) }
    $bundle = Join-Path $BundleDir $BundleName   # a unique name: the device runner skips a push when the size matches
    Move-Item (Join-Path $BundleDir "model.litertlm") $bundle -Force
    $cliOut = Join-Path $Log "05-cli-eval.jsonl"
    $code = Invoke-Logged "05-cli-eval" {
        & $Py (Join-Path $Bench "fg_eval_litertlm.py") --bundle $bundle --decl-mode none --cases (Join-Path $Cases "eval_cases.json") `
            --tokenizer $Merged --label "rebuild-$Label" --out $cliOut @limit
    }
    if ($code -ne 0) { throw "fg_eval_litertlm failed ($code)" }
    $parity = Invoke-Logged "05-parity" { & $Py $Check parity --hf (Join-Path $Log "04-hf-eval.jsonl") --cli $cliOut --bundle $bundle --log $Log }
    if ($parity -notin 0, 4) { throw "parity check failed ($parity)" }

    # ---------------------------------------------------------------- 6 export / stage
    $toolsJson = $SampleRuntime | ForEach-Object { Join-Path $_ "AsrFunctionCalling\Resources\functiongemma_tools.json" }
    $exportArgs = @($toolsJson | ForEach-Object { "--dest", $_ }) + @(if ($Smoke) { "--dry" })
    if ((Invoke-Logged "06-export" { & $Py $Check export --tools-prompt $ToolsPrompt --log $Log @exportArgs }) -ne 0) { throw "export failed" }
    if ($Stage -and -not $Smoke) {
        $rel = "LLM/functiongemma-270m-ko/$BundleName"
        $dest = Join-Path $ProjectRoot "Assets\StreamingAssets\$rel"
        New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
        Copy-Item $bundle $dest -Force
        $csArgs = @($SampleRuntime | ForEach-Object { "--file", (Join-Path $_ "Shared\LiteRtLmSampleAssets.cs") })
        if ((Invoke-Logged "06-stage" { & $Py $Check stage --path $rel --log $Log @csArgs }) -ne 0) { throw "staging failed" }
        "- bundle copied to ``Assets/StreamingAssets/$rel``" | Out-File -Append -Encoding utf8 (Join-Path $Log "06-stage.md")
    }
    if ($gatePassed -and -not $Smoke) {
        if ((Invoke-Logged "06-snapshot" { & $Py $Check snapshot --tools-prompt $ToolsPrompt }) -ne 0) { throw "snapshot update failed" }
        "- ``contract_snapshot.json`` updated to this ToolsPrompt" | Out-File -Append -Encoding utf8 (Join-Path $Log "06-export.md")
    }

    # ---------------------------------------------------------------- 7 device
    if ($Device -and -not $Smoke) {
        $packages = @(& adb -s $Serial shell pm list packages com.Leuconoe 2>$null | ForEach-Object { "$_".Trim() -replace '^package:', '' } | Where-Object { $_ })
        foreach ($p in $packages) {
            $busy = "$(& adb -s $Serial shell pidof $p 2>$null)".Trim()
            if ($busy) { throw "$p (pid $busy) is running on $Serial — not starting a parallel device run." }
        }
        $devDir = Join-Path $Log "device"
        if ((Invoke-Logged "07-device-prep" { & $Py $Check device-prep --cases (Join-Path $Cases "eval_cases.json") --out $devDir }) -ne 0) { throw "device prep failed" }
        & adb -s $Serial shell "rm -f /sdcard/Android/data/com.Leuconoe.LiteRTLMUnity/files/LiteRTLM/$BundleName*" 2>$null
        $fc = Join-Path $ProjectRoot "Tools\Windows\Tests\Run-LiteRtLmAndroidFcPromptSet.ps1"
        $null = Invoke-Logged "07-device-run" { & $fc -DeviceSerial $Serial -Bundle $bundle -SystemInstructionFile "$devDir\system.txt" -PromptsFile "$devDir\prompts.txt" -MaxNumTokens 256 -OutDir $devDir }
        if ((Invoke-Logged "07-device-grade" { & $Py $Check device-grade --cases (Join-Path $Cases "eval_cases.json") --results "$devDir\results.jsonl" --tools-prompt $ToolsPrompt --log $Log }) -ne 0) { throw "device grading failed" }
    }
    $script:Status = if ($Smoke) { "smoke OK: every stage ran on CPU (gate not enforced, nothing staged)" }
                     elseif ($gatePassed) { "OK: gate passed$(if ($Stage) { ', staged' })" } else { "DONE with -Force: gate FAILED, snapshot not updated" }
}

$exitCode = 0
try { Invoke-Rebuild }
catch { $script:Status = "FAILED: $($_.Exception.Message)"; $exitCode = 1 }
finally {
    $env:CUDA_VISIBLE_DEVICES = $savedCuda
    $switches = @("RefreshParaphrases", "WaitForGpu", "Force", "Stage", "Device", "Smoke", "DryRun") | Where-Object { (Get-Variable $_ -ValueOnly).IsPresent }
    $head = @("# FunctionGemma router rebuild — $Label", "", "$(Get-Date -Format 'yyyy-MM-dd HH:mm')  ·  ToolsPrompt ``$ToolsPrompt``  ·  " +
              "switches: $(if ($switches) { $switches -join ', ' } else { 'none' })  ·  gate >= $($GatePass.ToString('P0'))", "", "**Status: $($script:Status)**", "")
    $parts = Get-ChildItem $Log -Filter "*.md" | Where-Object Name -ne "report.md" | Sort-Object Name | ForEach-Object { Get-Content $_.FullName -Raw -Encoding utf8 }
    ($head + $parts) -join "`n" | Out-File -Encoding utf8 (Join-Path $Log "report.md")
    Write-Host "`n$($script:Status)`nreport: $(Join-Path $Log 'report.md')" -ForegroundColor $(if ($exitCode -or $script:Status -like "STOPPED*") { "Red" } else { "Green" })
}
if ($script:Status -like "STOPPED*") { $exitCode = 2 }
exit $exitCode
