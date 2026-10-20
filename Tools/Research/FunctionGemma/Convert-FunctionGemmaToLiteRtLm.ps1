<#
.SYNOPSIS
  Converts a FunctionGemma checkpoint (HF id or local merged dir) to a .litertlm
  bundle the v0.14 runtime loads, in a Linux Docker container.

.DESCRIPTION
  litert-torch has no Windows wheels, so the export runs in Docker. Versions are
  pinned to what the v0.14 runtime accepts — every one of these was learned the
  hard way (2026-10-16):

  - litert-torch 0.9.4 (stable). The nightly imports tensorflow it does not
    declare and dies at startup.
  - transformers 5.15.1. 5.20 removed Cache.get_max_cache_shape, which 0.9.4 calls.
  - --tokenizer_path_override=tokenizer.model. The default packs the HF
    tokenizer.json, and the runtime refuses constrained decoding without a
    SentencePiece tokenizer.
  - --use_jinja_template=False. The HF chat template fails to render in the v0.14
    runtime ("undefined value (in template:69)"); without it the runtime uses its
    built-in FunctionGemma template, chosen by --litert_lm_model_type_override.
  - No --externalize_embedder. Gemma ties lm_head to the embedding, so it only
    duplicates the 168 MB table (462 MB bundle instead of 290 MB).

  -CacheLength is the speed knob: 4096 prefills 2.8x slower than 1024 on the same
  598-token prompt (658 vs 1865 tok/s, Windows CPU). Size it to prompt + output.
  -PrefillLengths must cover the real prompt: a prompt longer than the smallest
  bucket runs in the next one up. Decl-mode none prompts are 28-44 tokens, and on
  kona (46a880a0) the default 32/272 buckets gave TTFT 0.45 s / 569 ms per
  command; -CacheLength 256 -PrefillLengths 64,128 gave TTFT 0.10 s / 336 ms at
  the same accuracy. Call this script directly with an array (`@(64,128)`):
  through `pwsh -File`, `64,128` arrives as the single integer 64128.

.EXAMPLE
  .\Tools\Research\FunctionGemma\Convert-FunctionGemmaToLiteRtLm.ps1 -Bootstrap
  .\Tools\Research\FunctionGemma\Convert-FunctionGemmaToLiteRtLm.ps1 -Model google/functiongemma-270m-it -OutDir External\functiongemma-work\convert\stock
  .\Tools\Research\FunctionGemma\Convert-FunctionGemmaToLiteRtLm.ps1 -Model External\functiongemma-work\ft1\merged -OutDir External\functiongemma-work\ft1\litertlm -CacheLength 2048 -PrefillLengths 128,1024
#>
[CmdletBinding()]
param(
    [string]$Model = "google/functiongemma-270m-it",
    [Parameter(Mandatory = $true)][string]$OutDir,
    [string]$Quantization = "dynamic_wi8_afp32",
    [int]$CacheLength = 1024,
    [int[]]$PrefillLengths = @(32, 272),
    [string]$Image = "litert-torch-converter:0.9.4",
    # "" = let litert-torch infer it (plain gemma-3 -> its Gemma3 template).
    [string]$ModelType = "function_gemma",
    # Source of tokenizer.model when the merged dir has none: the base the checkpoint was tuned from.
    [string]$BaseTokenizer = "google/functiongemma-270m-it",
    [switch]$Bootstrap
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path

if ($Bootstrap) {
    @"
FROM python:3.12-slim
RUN pip install --no-cache-dir --root-user-action=ignore litert-torch==0.9.4 litert-lm-builder==0.14.0 transformers==5.15.1
"@ | docker build -t $Image -
    if ($LASTEXITCODE -ne 0) { throw "docker build failed." }
}

# The token is gated-model access; never print it.
$tokenLine = Get-Content (Join-Path $ProjectRoot ".env") -ErrorAction SilentlyContinue |
    Where-Object { $_ -match '^\s*HF_TOKEN\s*=' } | Select-Object -First 1
if ($tokenLine) { $env:HF_TOKEN = ($tokenLine -replace '^\s*HF_TOKEN\s*=\s*', '').Trim().Trim('"') }

if (-not [IO.Path]::IsPathRooted($OutDir)) { $OutDir = Join-Path $ProjectRoot $OutDir }
New-Item -ItemType Directory -Force $OutDir | Out-Null

$mounts = @("-v", "fg-hf-cache:/root/.cache/huggingface", "-v", "${OutDir}:/out")
$localModel = if ([IO.Path]::IsPathRooted($Model)) { $Model } else { Join-Path $ProjectRoot $Model }
if (Test-Path $localModel) {
    $mounts += @("-v", "$((Resolve-Path $localModel).Path):/model:ro")
    $modelArg = "/model"
}
else {
    $modelArg = $Model
}

# Fine-tuning does not touch the tokenizer, so a merged dir without
# tokenizer.model borrows the base model's.
$script = @"
set -e
if [ -f /model/tokenizer.model ]; then TOK=/model/tokenizer.model; else
  TOK=`$(python -c "from huggingface_hub import hf_hub_download as d; print(d('$BaseTokenizer','tokenizer.model'))"); fi
litert-torch export_hf $modelArg /out \
  --cache_length=$CacheLength --prefill_lengths=$($PrefillLengths -join ',') \
  --quantization_recipe=$Quantization \
  $(if ($ModelType) { "--litert_lm_model_type_override=$ModelType" }) \
  --tokenizer_path_override=`$TOK --use_jinja_template=False > /out/export.log 2>&1
litert-lm-peek --litertlm_file=/out/model.litertlm | grep -i -E 'Data Type|max_num_tokens|model_type|gemma'
"@ -replace "`r", ""

docker run --rm -e HF_TOKEN @mounts $Image bash -c $script
if ($LASTEXITCODE -ne 0) { throw "Export failed — see $OutDir\export.log" }

$bundle = Get-Item (Join-Path $OutDir "model.litertlm")
Write-Host ("{0} ({1:N1} MB)" -f $bundle.FullName, ($bundle.Length / 1MB)) -ForegroundColor Green
