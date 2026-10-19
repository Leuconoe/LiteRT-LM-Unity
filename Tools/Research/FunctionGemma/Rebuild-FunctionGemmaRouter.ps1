<#
.SYNOPSIS
  One command from a tool contract to a validated, converted, staged FunctionGemma router
  (decl-mode none: the contract lives in the weights, so every contract change is a retrain).
.DESCRIPTION
  Every stage logs to Builds\Logs\fg-rebuild-<label>\ and report.md is written at the end,
  also when a stage stops the run.

    1 contract  -Contract vs FunctionGemmaBench\contract_snapshot.json (written after the first passed
                gate). STOPS with a to-do list when a tool has no template, a template misses a required
                parameter, or a parameter cannot be generated. Eval cases whose expected call no longer
                validates are listed and left out of every score (relabel by hand).
    2 data      make_dataset.py: templates filled from the contract, ~20 % refusals, exact-text filtered
                against the eval cases -> FunctionGemmaBench\data_rebuild_<label>\.
    3 train     GPU guard (>= 12 GB free, no other training; -WaitForGpu polls every 2 min), then the
                recipe (full fine-tune, 3 epochs, lr 2e-5, batch 8, decl none) through
                Run-FunctionGemmaEval.ps1 -Train; after a crash it reruns with --resume, 3 attempts in all.
    4 gate      fg_eval on the eval cases, batch 1 + clause split: 0 wrong actions and >= -GatePass.
    5 convert   cache 256 / prefill 64,128; the Windows CLI must reproduce the HF calls.
    6 export    the contract as Resources\functiongemma_tools.json (Samples~ + Assets/Samples copy);
                -Stage also copies the bundle to StreamingAssets and repoints FunctionGemmaRouterModel;
                contract_snapshot.json is rewritten after a passed gate only.
    7 device    -Device: the eval cases on the Android device (FC prompt-set runner), graded with the router.

  -Smoke is the CPU plumbing test (CUDA_VISIBLE_DEVICES=-1): ~200 records, 0.05 epoch, 8 cases,
  gate reported but not enforced, no staging, no snapshot, no device. Needs pwsh 7.
.EXAMPLE
  .\Tools\Research\FunctionGemma\Rebuild-FunctionGemmaRouter.ps1 -DryRun
  .\Tools\Research\FunctionGemma\Rebuild-FunctionGemmaRouter.ps1 -Smoke
  .\Tools\Research\FunctionGemma\Rebuild-FunctionGemmaRouter.ps1 -Label sample -WaitForGpu -Stage -Device
  .\Tools\Research\FunctionGemma\Rebuild-FunctionGemmaRouter.ps1 -Contract my_tools.json -Templates my_templates.json -Cases my_cases.json -Label myapp
#>
[CmdletBinding()]
param(
    [string]$Label = "",
    [string]$Contract = "",   # default: FunctionGemmaBench\sample_contract.json
    [string]$Templates = "",  # default: FunctionGemmaBench\sample_templates.json
    [string]$Cases = "",      # default: FunctionGemmaBench\sample_eval_cases.json
    [switch]$WaitForGpu,
    [double]$GatePass = 0.95,
    [switch]$Force,
    [switch]$Stage,
    [switch]$Device,
    [string]$DeviceSerial = "",
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
$Contract = (Resolve-Path $(if ($Contract) { $Contract } else { Join-Path $Bench "sample_contract.json" })).Path
$Templates = (Resolve-Path $(if ($Templates) { $Templates } else { Join-Path $Bench "sample_templates.json" })).Path
$Cases = (Resolve-Path $(if ($Cases) { $Cases } else { Join-Path $Bench "sample_eval_cases.json" })).Path
$Log = (New-Item -ItemType Directory -Force (Join-Path $ProjectRoot "Builds\Logs\fg-rebuild-$Label")).FullName
$GateCases = Join-Path $Log "cases\$(Split-Path $Cases -Leaf)"   # non-stale copy, written by stage 1
$Data = Join-Path $Bench "data_rebuild_$Label"
$Work = Join-Path $ProjectRoot "External\functiongemma-work\ft-functiongemma-none-rebuild-$Label"
$Merged = Join-Path $Work "merged"
$BundleDir = Join-Path $Work "litertlm-c256-p64"
$BundleName = "functiongemma_270m_${Label}_c256_p64.litertlm"
$SampleRuntime = @("Packages\com.leuconoe.litert-lm-unity\Samples~\TestScenes\Runtime") +
                 @(Get-ChildItem (Join-Path $ProjectRoot "Assets\Samples\*\*\Test Scenes\Runtime") -Directory -ErrorAction SilentlyContinue |
                   ForEach-Object { $_.FullName.Substring($ProjectRoot.Length + 1) }) | ForEach-Object { Join-Path $ProjectRoot $_ }
$ContractArgs = @("--contract", $Contract)

[Console]::OutputEncoding = [Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$savedCuda = $env:CUDA_VISIBLE_DEVICES
if ($Smoke -or $DryRun) { $env:CUDA_VISIBLE_DEVICES = "-1" }   # "" still reports CUDA on Windows
$script:Status = "running"

# Runs a command, tees its output to <log>\<name>.log, returns the exit code.
# The command block resolves variables through this function's scope first, so its own names are unusual.
function Invoke-Logged([string]$LoggedStep, [scriptblock]$LoggedCommand) {
    $loggedFile = Join-Path $Log "$LoggedStep.log"
    $global:LASTEXITCODE = 0
    & $LoggedCommand 2>&1 | ForEach-Object { "$_" } | Tee-Object -Append -FilePath $loggedFile | Out-Host
    return $global:LASTEXITCODE
}

function Get-GpuBlocker {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return $null }   # CPU-only machine: nothing to guard
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
    $code = Invoke-Logged "01-contract" { & $Py $Check contract @ContractArgs --templates $Templates --cases $Cases --log $Log }
    if ($code -eq 2) { $script:Status = "STOPPED at stage 1: hand edits needed (to-do list above)"; return }
    if ($code -ne 0) { throw "contract check failed ($code)" }
    if ($DryRun) {
        $plan = @("## Plan (dry run — nothing below ran)", "",
            "- 2 data: make_dataset.py ($(Split-Path $Templates -Leaf)) -> $Data",
            "- 3 train: full FT, 3 epochs, lr 2e-5, batch 8, decl none -> $Merged; GPU guard$(if ($WaitForGpu) { ', waiting' }); --resume retries (3 attempts)",
            "- 4 gate: $(Split-Path $Cases -Leaf) minus stale, batch 1 + clause split: 0 wrong actions and >= $($GatePass.ToString('P0')) pass$(if ($Force) { ' (-Force: not enforced)' })",
            "- 5 convert: cache 256 / prefill 64,128 -> $BundleDir; Windows CLI parity",
            "- 6 export: functiongemma_tools.json$(if ($Stage) { "; stage Assets\StreamingAssets\LLM\functiongemma-270m-sample\$BundleName + FunctionGemmaRouterModel" }); contract_snapshot.json after a passed gate",
            "- 7 device: $(if ($Device) { "eval cases on $(if ($DeviceSerial) { $DeviceSerial } else { 'the adb default device' }), router-graded" } else { 'skipped (no -Device)' })")
        $plan -join "`n" | Out-File -Encoding utf8 (Join-Path $Log "01-plan.md")
        $plan | Out-Host
        $script:Status = "dry run: stage 1 only"
        return
    }

    # ---------------------------------------------------------------- 2 data
    $md = Join-Path $Bench "make_dataset.py"
    if ((Invoke-Logged "02-data" { & $Py $md @ContractArgs --templates $Templates --cases $Cases --out $Data }) -ne 0) { throw "make_dataset failed" }
    $trainDir = $Data
    if ($Smoke) {
        if ((Invoke-Logged "02-data-subset" { & $Py $Check subset --src $Data --out "$Data\smoke" --n 200 }) -ne 0) { throw "subset failed" }
        $trainDir = "$Data\smoke"
    }
    $summary = Get-Content (Get-ChildItem $Log -Filter "02-data*.log").FullName -Encoding utf8 |
        Select-String -Pattern '^(exact-text filter|train \d+|subset:)' | ForEach-Object { $_.Line }
    (@("## 2. Data", "", "Train set: ``$trainDir``", "", '```') + $summary + '```') -join "`n" | Out-File -Encoding utf8 (Join-Path $Log "02-data.md")

    # ---------------------------------------------------------------- 3 train
    if (-not $Smoke) { Wait-Gpu }
    $epochs = if ($Smoke) { "0.05" } else { "3" }
    $trainArgs = @("--train", "$trainDir\train.jsonl", "--val", "$trainDir\val.jsonl", "--decl-mode", "none",
                   "--epochs", $epochs, "--lr", "2e-5", "--batch-size", "8", "--label", "ft-functiongemma-none-rebuild-$Label",
                   "--out", $Work) + $ContractArgs
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
    @("## 3. Train", "", "full FT (lr 2e-5, batch 8, decl none) -> ``$Merged``; $trainNote.", "", "$best") -join "`n" | Out-File -Encoding utf8 (Join-Path $Log "03-train.md")
    if ($code -ne 0) { throw "training failed after $attempt attempts — see 03-train.log" }

    # ---------------------------------------------------------------- 4 gate
    $limit = if ($Smoke) { @("--limit", "8") } else { @() }
    $evalDevice = if ($Smoke) { @("--device", "cpu") } else { @() }   # not $device: names are case-insensitive, -Device lives here
    $hfOut = Join-Path $Log "04-hf-eval.jsonl"
    $code = Invoke-Logged "04-eval" {
        & $Py (Join-Path $Bench "fg_eval.py") --model $Merged --decl-mode none --cases $GateCases --batch-size 1 --segment `
            --label "rebuild-$Label" --out $hfOut @ContractArgs @evalDevice @limit
    }
    if ($code -ne 0) { throw "fg_eval failed ($code)" }
    $gate = Invoke-Logged "04-score" { & $Py $Check score "eval=$hfOut" --gate eval --pass-rate $GatePass --log $Log }
    if ($gate -notin 0, 3) { throw "scoring failed ($gate)" }
    $gatePassed = $gate -eq 0
    if (-not $gatePassed -and -not ($Force -or $Smoke)) { $script:Status = "STOPPED at stage 4: gate failed (-Force to convert anyway)"; return }

    # ---------------------------------------------------------------- 5 convert + parity
    $convert = Join-Path $PSScriptRoot "Convert-FunctionGemmaToLiteRtLm.ps1"
    $null = Invoke-Logged "05-convert" { & $convert -Model $Merged -OutDir $BundleDir -CacheLength 256 -PrefillLengths @(64, 128) }
    $bundle = Join-Path $BundleDir $BundleName   # a unique name per label
    Move-Item (Join-Path $BundleDir "model.litertlm") $bundle -Force
    $cliOut = Join-Path $Log "05-cli-eval.jsonl"
    $code = Invoke-Logged "05-cli-eval" {
        & $Py (Join-Path $Bench "fg_eval_litertlm.py") --bundle $bundle --decl-mode none --cases $GateCases `
            --tokenizer $Merged --label "rebuild-$Label" --out $cliOut @ContractArgs @limit
    }
    if ($code -ne 0) { throw "fg_eval_litertlm failed ($code)" }
    $parity = Invoke-Logged "05-parity" { & $Py $Check parity --hf $hfOut --cli $cliOut --bundle $bundle --log $Log }
    if ($parity -notin 0, 4) { throw "parity check failed ($parity)" }

    # ---------------------------------------------------------------- 6 export / stage
    $toolsJson = $SampleRuntime | ForEach-Object { Join-Path $_ "AsrFunctionCalling\Resources\functiongemma_tools.json" }
    $exportArgs = @($toolsJson | ForEach-Object { "--dest", $_ }) + @(if ($Smoke) { "--dry" })
    if ((Invoke-Logged "06-export" { & $Py $Check export @ContractArgs --log $Log @exportArgs }) -ne 0) { throw "export failed" }
    if ($Stage -and -not $Smoke) {
        $rel = "LLM/functiongemma-270m-sample/$BundleName"
        $dest = Join-Path $ProjectRoot "Assets\StreamingAssets\$rel"
        New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
        Copy-Item $bundle $dest -Force
        $csArgs = @($SampleRuntime | ForEach-Object { "--file", (Join-Path $_ "Shared\LiteRtLmSampleAssets.cs") })
        if ((Invoke-Logged "06-stage" { & $Py $Check stage --path $rel --log $Log @csArgs }) -ne 0) { throw "staging failed" }
        "- bundle copied to ``Assets/StreamingAssets/$rel``" | Out-File -Append -Encoding utf8 (Join-Path $Log "06-stage.md")
    }
    if ($gatePassed -and -not $Smoke) {
        if ((Invoke-Logged "06-snapshot" { & $Py $Check snapshot @ContractArgs }) -ne 0) { throw "snapshot update failed" }
        "- ``contract_snapshot.json`` updated to this contract" | Out-File -Append -Encoding utf8 (Join-Path $Log "06-export.md")
    }

    # ---------------------------------------------------------------- 7 device
    if ($Device -and -not $Smoke) {
        $serial = if ($DeviceSerial) { $DeviceSerial } else { "$(& adb get-serialno 2>$null)".Trim() }
        if (-not $serial -or $serial -eq "unknown") { throw "no Android device: connect one or pass -DeviceSerial" }
        $devDir = Join-Path $Log "device"
        if ((Invoke-Logged "07-device-prep" { & $Py $Check device-prep --cases $GateCases --out $devDir }) -ne 0) { throw "device prep failed" }
        $fc = Join-Path $ProjectRoot "Tools\Windows\Tests\Run-LiteRtLmAndroidFcPromptSet.ps1"
        $null = Invoke-Logged "07-device-run" { & $fc -DeviceSerial $serial -Bundle $bundle -SystemInstructionFile "$devDir\system.txt" -PromptsFile "$devDir\prompts.txt" -MaxNumTokens 256 -OutDir $devDir }
        if ((Invoke-Logged "07-device-grade" { & $Py $Check device-grade --cases $GateCases --results "$devDir\results.jsonl" @ContractArgs --log $Log }) -ne 0) { throw "device grading failed" }
    }
    $script:Status = if ($Smoke) { "smoke OK: every stage ran on CPU (gate not enforced, nothing staged)" }
                     elseif ($gatePassed) { "OK: gate passed$(if ($Stage) { ', staged' })" } else { "DONE with -Force: gate FAILED, snapshot not updated" }
}

$exitCode = 0
try { Invoke-Rebuild }
catch { $script:Status = "FAILED: $($_.Exception.Message)"; $exitCode = 1 }
finally {
    $env:CUDA_VISIBLE_DEVICES = $savedCuda
    $switches = @("WaitForGpu", "Force", "Stage", "Device", "Smoke", "DryRun") | Where-Object { (Get-Variable $_ -ValueOnly).IsPresent }
    $head = @("# FunctionGemma router rebuild — $Label", "", "$(Get-Date -Format 'yyyy-MM-dd HH:mm')  ·  contract ``$Contract``  ·  " +
              "switches: $(if ($switches) { $switches -join ', ' } else { 'none' })  ·  gate >= $($GatePass.ToString('P0'))", "", "**Status: $($script:Status)**", "")
    $parts = Get-ChildItem $Log -Filter "*.md" | Where-Object Name -ne "report.md" | Sort-Object Name | ForEach-Object { Get-Content $_.FullName -Raw -Encoding utf8 }
    ($head + $parts) -join "`n" | Out-File -Encoding utf8 (Join-Path $Log "report.md")
    Write-Host "`n$($script:Status)`nreport: $(Join-Path $Log 'report.md')" -ForegroundColor $(if ($exitCode -or $script:Status -like "STOPPED*") { "Red" } else { "Green" })
}
if ($script:Status -like "STOPPED*") { $exitCode = 2 }
exit $exitCode
