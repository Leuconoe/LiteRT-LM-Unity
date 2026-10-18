<#
.SYNOPSIS
  Builds the per-engine venvs for the open-source Korean TTS voices used to widen
  the locked heldout5 voice evaluation (CosyVoice3, MeloTTS-Korean, Qwen3-TTS).

.DESCRIPTION
  Each engine pins conflicting versions, so each gets its own venv next to this
  script. Engine sources are cloned into External/oss-tts (untracked scratch).
  Python 3.10 for CosyVoice/MeloTTS (their pins predate 3.12 wheels), 3.12 for Qwen3-TTS.

.EXAMPLE
  .\Setup-OssTts.ps1 -Engine cosyvoice
#>
[CmdletBinding()]
param([ValidateSet("cosyvoice", "melo", "qwen3")][string]$Engine)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..\..")).Path
$Src = Join-Path $Root "External\oss-tts"
$Venv = Join-Path $PSScriptRoot ".venv-$Engine"
$Py = Join-Path $Venv "Scripts\python.exe"
$cu = "https://download.pytorch.org/whl/cu121"

switch ($Engine) {
    "cosyvoice" {
        if (-not (Test-Path "$Src\CosyVoice")) { git clone --depth 1 --recursive https://github.com/FunAudioLLM/CosyVoice.git "$Src\CosyVoice" }
        uv venv --python 3.10 $Venv
        uv pip install --python $Py torch==2.3.1 torchaudio==2.3.1 --index-url $cu
        # requirements.txt minus the serving/UI/linux-only lines
        $req = Get-Content "$Src\CosyVoice\requirements.txt" | Where-Object { $_ -notmatch "^(--|deepspeed|tensorrt|gradio|fastapi|uvicorn|grpcio|tensorboard|torch==|torchaudio==)" }
        $tmp = New-TemporaryFile; $req | Set-Content $tmp
        # openai-whisper's sdist needs pkg_resources at build time; uv's isolated build has none
        uv pip install --python $Py "setuptools<81" wheel
        uv pip install --python $Py --no-build-isolation openai-whisper==20231117
        uv pip install --python $Py -r $tmp huggingface_hub
    }
    "melo" {
        if (-not (Test-Path "$Src\MeloTTS")) { git clone --depth 1 https://github.com/myshell-ai/MeloTTS.git "$Src\MeloTTS" }
        uv venv --python 3.10 $Venv
        uv pip install --python $Py torch==2.3.1 torchaudio==2.3.1 --index-url $cu
        # librosa 0.9 imports pkg_resources; g2pkk needs eunjeon (Korean MeCab) on Windows
        uv pip install --python $Py -e "$Src\MeloTTS" "setuptools<81" eunjeon
        & $Py -m unidic download
    }
    "qwen3" {
        uv venv --python 3.12 $Venv
        uv pip install --python $Py torch torchaudio --index-url https://download.pytorch.org/whl/cu128
        uv pip install --python $Py qwen-tts soundfile
    }
}
& $Py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
