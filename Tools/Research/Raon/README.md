# RaonBench — Raon-Speech on the workstation

KRAFTON's Raon-Speech is the only model this project has evaluated that does
**speech recognition and speech synthesis in one network**. That makes it worth
keeping a working driver for, even though it cannot ship.

```powershell
.\Tools\Research\Raon\Run-RaonDesktop.ps1 -Bootstrap                      # first run
.\Tools\Research\Raon\Run-RaonDesktop.ps1 -Model KRAFTON/Raon-Speech-9B   # BF16 — works on Windows
.\Tools\Research\Raon\Run-RaonDesktop.ps1                                 # AWQ-INT4 — see below
```

STT runs the clips the ASR matrix uses; TTS speaks the sentences the Supertonic
device test speaks, so the output sits beside numbers we already have. WAVs and
a JSONL record land in `Builds/Logs/RaonDesktop/`.

## Use the BF16 build on Windows, not AWQ-INT4

The AWQ-INT4 build is the obvious choice on paper — 7.3 GB against 16.9 GB, both
of which fit an RTX 4090. **It does not load on Windows**, and the reason is
packaging, not the GPU:

`transformers` 5.x routes AWQ through `gptqmodel`, which imports `pcre`, which
comes from `pypcre` — a source-only package on Windows whose build script probes
for a Visual Studio generator and fails even with the VS2022 Build Tools
(MSVC 14.44) on `PATH` via `vcvarsall`:

```
RuntimeError: vswhere cannot find a valid VS installation: None
ERROR: Failed to build 'pypcre' when getting requirements to build wheel
```

`pip install --no-deps gptqmodel` installs cleanly but then dies on the same
import at runtime. The upstream serving path is Linux + Docker
(`krafton-ai/vllm-omni`), which is where the published numbers come from.

So on this workstation, two routes:

| | BF16 via `transformers` | AWQ-INT4 via Docker |
| --- | --- | --- |
| Command | `Run-RaonDesktop.ps1 -Model KRAFTON/Raon-Speech-9B` | `Build-RaonVllmOmni.ps1 -Smoke` |
| Weights | 16.9 GB, 18.11 GB allocated | 7.3 GB |
| Works here | **verified** — CER 0.000 on all four STT clips | **not yet run** — see below |
| Speed | RTF 2.1–5.0, i.e. slower than real time | the configuration KRAFTON's 0.27–0.45 RTF comes from |
| Why the gap | naive `generate()` loop, no streaming, no batching | vLLM continuous batching + streaming |

Use BF16 for a functional check, Docker for anything you intend to quote as a
performance number.

**The Docker route is researched, not proven.** GPU passthrough is verified
(Docker Desktop, WSL2 backend, driver 596.49, the 4090 visible inside
`nvidia/cuda:12.8.0-base`), and `Dockerfile.ci` is `FROM vllm/vllm-openai` so
the build is a pull rather than a compile — but the image build was stopped
during that pull and the server has never been started from here. Treat
`Build-RaonVllmOmni.ps1` as a recipe to debug on first use, not a working
command.

One caveat the loader prints and it is worth heeding: `flash_attn` is not
installed, so Mimi falls back to SDPA and **ignores its sliding window — audio
longer than about 20 s may contain artifacts**.

## Two things to be clear about before quoting any of this

**It cannot run on the device, and it cannot run through LiteRT-LM.** Not a
matter of effort:

| Obstacle | Why it is not a workaround away |
| --- | --- |
| `RaonModel` is a custom architecture shipped as remote code | litert-torch converts a fixed set of architectures; this is not one, and it is not one model but seven — Qwen3 backbone (36 layers), Qwen3OmniMoe audio encoder (24), Mimi codec (32 quantizers), talker (4), code predictor (5), ECAPA-TDNN speaker encoder, and the input/output adaptors |
| AWQ is a CUDA GEMM kernel format | LiteRT has no AWQ path. Converting means dequantizing to fp16 and requantizing with `ai_edge_quantizer`, so the AWQ build buys nothing — you would start from the 16.9 GB BF16 weights |
| Only the LLM backbone is quantized | `modules_to_not_convert` keeps the audio encoder, tokenizer, talker, code predictor, speaker encoder, adaptors, embeddings and LM head in full precision, so the 7.3 GB file is not a 4-bit 9 B model |
| LiteRT-LM is a text/vision/audio-**input** LLM runtime | It has no speech-**generation** stack — no codec decoder, no talker loop. That part would be a new native driver, the way Whisper needed one |
| kona has 7.5 GB RAM, no CUDA, no NPU in the litert-community lists | Even a converted model would not fit or accelerate |

**Licence: CC-BY-NC-4.0 across the whole Raon family.** Non-commercial. This is
an evaluation harness, not a path to a delivered product, unless KRAFTON grants
separate commercial terms.

## What it would mean to adopt it anyway

Moving speech off the headset and onto a GPU-equipped ground station reached
over the network. That is a legitimate architecture for a drone system, and it
would let ASR and TTS collapse into one model — but it adds a network
dependency, link latency on top of the model's own first-audio time, and a
second machine to power and maintain. It is a system-architecture decision, not
an engine swap. Written up in
[`docs/tts-details.md`](../../../docs/tts-details.md).
