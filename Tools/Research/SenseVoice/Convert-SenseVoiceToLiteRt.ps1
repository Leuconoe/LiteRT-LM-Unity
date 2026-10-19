<#
.SYNOPSIS
  SenseVoice-Small (stock or fine-tuned) -> fixed-shape LiteRT bucket graphs, fp32 + int8.

.DESCRIPTION
  1. export   (.venv)          funasr model -> one STATIC ONNX per bucket (sv_t{N}.onnx).
                               Static shapes on purpose: a dynamic graph keeps XNNPACK from
                               attaching (Supertonic: 13x slower). Lengths stay a graph input,
                               so padding to the bucket is masked, not heard.
  2. convert  (.venv-convert)  onnxsim -> onnx2tf -> sv_t{N}.tflite, then ai-edge-quantizer
                               w8 (weight_only_wi8_afp32) and i8 (dynamic_wi8_afp32).
  3. parity   (.venv)          each tflite vs onnxruntime on real clips via sv_eval --ref-onnx.
  Side files (am.mvn, *.bpe.model, config.yaml) are copied next to every variant.

.EXAMPLE
  .\Tools\Research\SenseVoice\Convert-SenseVoiceToLiteRt.ps1 -Bootstrap
  .\Tools\Research\SenseVoice\Convert-SenseVoiceToLiteRt.ps1 -Model FunAudioLLM/SenseVoiceSmall -Name stock
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Model = "FunAudioLLM/SenseVoiceSmall",
    [string]$Name = "stock",
    [int[]]$Buckets = @(50, 84, 167, 500),   # LFR frames (60 ms each): 3 s / 5 s / 10 s / 30 s
    [string[]]$Quant = @("w8", "i8"),
    [string]$WorkDir = "",
    [switch]$Bootstrap,
    [switch]$SkipExport,
    [switch]$SkipParity
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$BenchDir = Join-Path $PSScriptRoot "SenseVoiceBench"
$EvalPython = Join-Path $BenchDir ".venv\Scripts\python.exe"
$ConvPython = Join-Path $BenchDir ".venv-convert\Scripts\python.exe"
if (!$WorkDir) { $WorkDir = Join-Path $ProjectRoot "External\sensevoice-work\tflite\$Name" }

$env:PYTHONIOENCODING = "utf-8"
$env:TF_ENABLE_ONEDNN_OPTS = "0"
$env:TF_CPP_MIN_LOG_LEVEL = "2"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if ($Bootstrap) {
    if (!(Test-Path $ConvPython)) {
        $base = Join-Path $env:APPDATA "uv\python\cpython-3.12.13-windows-x86_64-none\python.exe"
        if (!(Test-Path $base)) { throw "Python 3.12 not found (TensorFlow has no cp314 wheel)." }
        & $base -m venv (Join-Path $BenchDir ".venv-convert")
        if ($LASTEXITCODE -ne 0) { throw "venv creation failed." }
    }
    & $ConvPython -m pip install --upgrade pip | Out-Null
    & $ConvPython -m pip install -r (Join-Path $BenchDir "requirements-convert.txt")
    if ($LASTEXITCODE -ne 0) { throw "conversion toolchain install failed." }
    # --no-deps: its ai-edge-litert/onnxsim pins conflict with ai-edge-quantizer (see requirements-convert.txt).
    & $ConvPython -m pip install --no-deps onnx2tf==2.6.7
    if ($LASTEXITCODE -ne 0) { throw "onnx2tf install failed." }
    exit 0
}

foreach ($p in @($EvalPython, $ConvPython)) { if (!(Test-Path $p)) { throw "Missing $p — run -Bootstrap here and in Run-SenseVoiceEval.ps1." } }
$onnxDir = Join-Path $WorkDir "onnx"
$bucketArg = ($Buckets -join ",")

if (!$SkipExport) {
    & $EvalPython (Join-Path $BenchDir "export_sensevoice_onnx.py") --model $Model --out-dir $onnxDir --buckets $bucketArg
    if ($LASTEXITCODE -ne 0) { throw "ONNX export failed." }
}
& $ConvPython (Join-Path $BenchDir "convert_sensevoice_to_tflite.py") --onnx-dir $onnxDir --out-dir $WorkDir --buckets $bucketArg --quant ($Quant -join ",")
if ($LASTEXITCODE -ne 0) { throw "tflite conversion failed." }

if (!$SkipParity) {
    foreach ($v in @("fp32") + $Quant) {
        & $EvalPython (Join-Path $BenchDir "sv_eval.py") --backend tflite --model (Join-Path $WorkDir $v) `
            --ref-onnx (Join-Path $onnxDir "model.onnx") --sets real,gate40 --label "$Name-$v"
        if ($LASTEXITCODE -ne 0) { throw "parity run failed for $v." }
    }
}
