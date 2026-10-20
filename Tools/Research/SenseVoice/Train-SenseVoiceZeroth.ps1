<#
.SYNOPSIS
  Fine-tunes FunAudioLLM/SenseVoiceSmall on Zeroth-Korean (OpenSLR 40) with FunASR's
  train_ds.py — the official SenseVoice finetune.sh, single GPU, conservative lr.

.DESCRIPTION
  -Bootstrap  creates SenseVoiceBench/.venv-train (Python 3.12/3.13, torch+torchaudio
              cu128, then requirements-train.txt). Kept apart from S1's eval .venv.
  -Prepare    exports wav + FunASR SenseVoice jsonl manifests from the ACFT data
              (External/acft-training/data/acft_mix: zeroth train/test + FLEURS-en)
              via SenseVoiceBench/sv_zeroth_prep.py.
  default     trains. Every extra argument is passed to train_ds.py as a ++override,
              e.g. '++optim_conf.lr=0.00003'.

  Evaluation is Run-SenseVoiceEval.ps1 (sv_eval.py) against the best checkpoint dir.

.EXAMPLE
  .\Tools\Research\SenseVoice\Train-SenseVoiceZeroth.ps1 -Bootstrap
  .\Tools\Research\SenseVoice\Train-SenseVoiceZeroth.ps1 -Prepare
  .\Tools\Research\SenseVoice\Train-SenseVoiceZeroth.ps1 -MaxEpoch 4 -Lr 5e-5
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [switch]$Bootstrap,
    [switch]$Prepare,
    [string]$WorkDir = "External\sensevoice-work",
    [string]$OutDir = "External\sensevoice-work\zeroth-ft",
    [double]$Lr = 5e-5,
    [int]$MaxEpoch = 4,
    [int]$BatchFrames = 8000,   # token batch = sum of fbank frames (10 ms) per step ≈ 80 s audio
    [int]$WarmupSteps = 1000,   # base config ships 25000, which a 4-epoch run (~10k steps) never leaves
    [double]$EnShare = 0.15,    # FLEURS-en share of the train manifest (forgetting guard)
    [int]$SaveSteps = 1000000,  # extra mid-epoch checkpoints (model.pt.ep{E}.{step}); ~2690 steps/epoch at 8000 frames
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Overrides = @()
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$BenchDir = Join-Path $PSScriptRoot "SenseVoiceBench"
$Py = Join-Path $BenchDir ".venv-train\Scripts\python.exe"
Set-Location $ProjectRoot
$env:PYTHONIOENCODING = "utf-8"

if ($Bootstrap) {
    if (!(Test-Path $Py)) {
        $base = Join-Path $env:APPDATA "uv\python\cpython-3.12.13-windows-x86_64-none\python.exe"
        if (!(Test-Path $base)) { $base = (& py -3.13 -c "import sys; print(sys.executable)").Trim() }
        & $base -m venv (Join-Path $BenchDir ".venv-train")
        if ($LASTEXITCODE -ne 0) { throw "venv creation failed." }
    }
    & $Py -m pip install --upgrade pip | Out-Null
    # 2.8 pin: torchaudio >= 2.9 routes torchaudio.load through torchcodec (needs FFmpeg
    # DLLs on Windows), and FunASR's data loader calls torchaudio.load on every wav.
    & $Py -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchaudio==2.8.0
    if ($LASTEXITCODE -ne 0) { throw "torch install failed." }
    & $Py -m pip install -r (Join-Path $BenchDir "requirements-train.txt")
    if ($LASTEXITCODE -ne 0) { throw "requirements install failed." }
    & $Py -c "import torch, funasr; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'funasr', funasr.__version__)"
    exit $LASTEXITCODE
}
if (!(Test-Path $Py)) { throw "No venv. Run with -Bootstrap first." }

if ($Prepare) {
    & $Py (Join-Path $BenchDir "sv_zeroth_prep.py") --work $WorkDir --en-share $EnShare
    exit $LASTEXITCODE
}

$base = Join-Path $WorkDir "base\SenseVoiceSmall"
if (!(Test-Path (Join-Path $base "model.pt"))) {
    & $Py -c "from huggingface_hub import snapshot_download as s; s('FunAudioLLM/SenseVoiceSmall', local_dir=r'$base')"
    if ($LASTEXITCODE -ne 0) { throw "base model download failed." }
}
# funasr prints an ffmpeg notice on import, so keep only the last stdout line.
$trainTool = (& $Py -c "import funasr, os; print(os.path.join(os.path.dirname(funasr.__file__), 'bin', 'train_ds.py'))" 2>$null | Select-Object -Last 1).Trim()
New-Item -ItemType Directory -Force $OutDir | Out-Null

# Mirrors SenseVoice finetune.sh; single GPU so no torchrun.
$trainArgs = @(
    $trainTool,
    "++model=$((Resolve-Path $base).Path)",
    "++trust_remote_code=false",
    "++train_data_set_list=$WorkDir\data\train.jsonl",
    "++valid_data_set_list=$WorkDir\data\dev.jsonl",
    "++dataset_conf.data_split_num=1",
    "++dataset_conf.batch_sampler=BatchSampler",
    "++dataset_conf.batch_size=$BatchFrames",
    "++dataset_conf.sort_size=1024",
    "++dataset_conf.batch_type=token",
    "++dataset_conf.num_workers=0",
    "++train_conf.max_epoch=$MaxEpoch",
    "++train_conf.log_interval=50",
    "++train_conf.resume=false",
    # Must equal the save interval: a mid-epoch save looks up that step's val result and
    # raises KeyError ('model.pt.ep0.1345') when validation did not run on the same step.
    "++train_conf.validate_interval=$SaveSteps",
    # Checkpoints are 2.8 GB each (optimizer included) and all kept: FunASR ranks "best" by
    # acc, which this CTC-only model reports as 0 — pick by CER with sv_eval.py instead.
    "++train_conf.save_checkpoint_interval=$SaveSteps",
    "++train_conf.keep_nbest_models=20",
    "++train_conf.avg_nbest_model=0",
    "++train_conf.use_deepspeed=false",
    "++optim_conf.lr=$Lr",
    "++scheduler_conf.warmup_steps=$WarmupSteps",
    "++output_dir=$((Resolve-Path $OutDir).Path)"
) + $Overrides
& $Py @trainArgs
exit $LASTEXITCODE
