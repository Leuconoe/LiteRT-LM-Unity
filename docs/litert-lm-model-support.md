# What LiteRT-LM can actually run

Every candidate model in this project has to clear the same gate, and the same
question keeps getting asked from scratch. This page is the answer, so a new
model can be screened in minutes instead of days.

**The criterion is on-device execution.** PC operation is a secondary goal —
large models are expected to work on the workstation — but the deployment
target is the kona / SM8250 headset, and a model that only runs on the PC
is a companion-machine architecture, not a drop-in.

## Two runtimes, not one

The distinction matters more than anything else on this page, and mixing them up
is what makes "can it run on LiteRT?" sound unanswerable.

| | **LiteRT-LM** | **LiteRT** |
| --- | --- | --- |
| What it is | A packaged **LLM runtime**: tokenizer, KV cache, sampler, chat session, constrained decoding | A **tensor runtime** — it executes `.tflite` graphs and nothing else |
| Input | `.litertlm` bundle | `.tflite` file per graph |
| You write | A prompt | The whole pipeline around the graphs |
| Ships here as | `litert_lm_main.exe`, the Android AAR's LLM path | The same AAR's ASR and TTS paths |
| Runs | gemma-4, Gemma3, Qwen2.5/3, LFM2.5 | Whisper, Silero VAD, Supertonic |

So "it must run on LiteRT-LM" is achievable only for **text LLMs of a supported
architecture**. Speech models reach the device through the second column: raw
graphs plus a driver we write. That is exactly how TTS shipped — Supertonic is
four `.tflite` graphs driven by our own JNI code, not a `.litertlm` bundle.

## The gate, in order

Check these before downloading anything. Each one has killed a candidate.

1. **Is the architecture supported by the converter?**
   `.litertlm` bundles are produced by litert-torch / ai-edge-torch, which
   supports a fixed model zoo. A `config.json` with an `auto_map` pointing at
   `modeling_*.py` — that is, `trust_remote_code` — means **no**, unless the
   architecture happens to be a supported one under a new name. Authoring a new
   architecture is a porting project, not a conversion.
2. **Is the licence deliverable?** Non-commercial (CC-BY-NC) is fatal.
   Bespoke licences that carve out on-premise or SI delivery need a negotiated
   agreement before engineering time is worth spending.
3. **Does it fit?** kona has 7.5 GB of system RAM, no CUDA, and no NPU in the
   litert-community per-SoC lists (those start at SM8450). Practical ceiling for
   a resident model is well under 1 GB; the 2.6 GB flagship is loaded on demand
   and peaks at 3.6 GB on image turns.
4. **Only then**: convert, quantize to int4 minimum tier, and measure.

Note that conversion itself needs Linux — the `litert-lm convert` CLI is a stub
in v0.14 and there are no Windows converter wheels.

## Verdicts on file

| Model | Gate it failed | Detail |
| --- | --- | --- |
| gemma-4-E2B, Gemma3-1B/270M, Qwen2.5-0.5B/1.5B, Qwen3-0.6B, LFM2.5-1.2B, FunctionGemma-270M | — passes | [`llm-details.md`](llm-details.md) |
| Whisper, Silero VAD, Supertonic | not LiteRT-LM — runs on **LiteRT** with our driver | [`asr-details.md`](asr-details.md), [`tts-model-research.md`](tts-model-research.md) |
| **Kanana-2-1.3B** | 1 (custom `kanana2_tiny`, hybrid SWA + per-layer-type RoPE) and 2 (Kanana licence §4.1(ii) covers on-premise/SI delivery) | [`llm-details.md` §6](llm-details.md#6-evaluated-and-rejected) |
| **Raon-Speech-9B(-AWQ-INT4)** | 1, 2 and 3 — all three | below |
| Qwen3.5-0.8B-MTP | 1 — architecture unsupported; MTP is a llama.cpp feature | [`llm-details.md` §6](llm-details.md#6-evaluated-and-rejected) |
| Bonsai-1.7B ternary | 1 — LiteRT requantizes to int4/8, erasing the 1-bit advantage | same |
| Hammer2.1-0.5b | 2 — CC-BY-NC | same |
| VibeVoice-ASR | 3 — 8.7 B | same |

## Raon-Speech, specifically

Asked for repeatedly, so the reasoning is recorded in full rather than
summarised as "no".

**On LiteRT-LM: not possible, and not by a margin that effort closes.**

- It is not one model. `config.json` describes a Qwen3 backbone (36 layers,
  4096 hidden), a Qwen3OmniMoe audio encoder (24 layers), a Mimi codec with 32
  quantizers, a 4-layer talker, a 5-layer code predictor
  (`qwen3_omni_moe_talker_code_predictor`), an ECAPA-TDNN speaker encoder, and
  input/output adaptors. Each would need authoring in ai-edge-torch separately.
- `architectures: ["RaonModel"]`, `model_type: "raon"`, loaded through
  `auto_map` → remote code. Gate 1, unambiguously.
- **The AWQ build does not help.** `quant_method: "awq"`, GEMM kernels, CUDA
  only. LiteRT has no AWQ path, so conversion means dequantizing to fp16 and
  requantizing with `ai_edge_quantizer` — i.e. starting from the 16.9 GB BF16
  weights and gaining nothing from the 7.3 GB file. Worse,
  `modules_to_not_convert` keeps the audio encoder, audio tokenizer, talker,
  code predictor, speaker encoder, adaptors, embeddings and LM head in full
  precision, so the file is not a 4-bit 9 B model to begin with.
- **LiteRT-LM has no speech-generation stack.** It accepts audio *input* on
  gemma-4. Producing audio needs a codec decoder and a talker loop driven from
  outside — a new native driver, the way Whisper needed one.
- Gate 3 is decided regardless: 9 B against 7.5 GB and no CUDA.

**On the PC: yes, and there is a driver for it.**
`Tools/Research/Raon/Run-RaonDesktop.ps1` runs it on the RTX 4090 through the
model's own `RaonPipeline`, measuring STT on the ASR test clips and TTS on the
sentences the Supertonic device test uses. Results in `Builds/Logs/RaonDesktop/`.

**Use the BF16 build, not AWQ-INT4** — on Windows the smaller file is the one
that does not work. `transformers` 5.x routes AWQ through `gptqmodel`, which
needs `pcre`; `pypcre` is source-only on Windows and its build fails to find a
Visual Studio generator even with the VS2022 Build Tools on `PATH`
(`vswhere cannot find a valid VS installation`). Upstream serves AWQ through
Linux + Docker (`krafton-ai/vllm-omni`). BF16 needs no AWQ kernels and fits in
24 GB, so nothing is lost here except disk.

**Gate 2 still stands.** The whole Raon family is **CC-BY-NC-4.0**. The desktop
driver is an evaluation harness; it is not a route to a delivered product
without separate terms from KRAFTON.

## If a candidate fails gate 1 but you still want a number

Score it on the desktop before paying for a port.
`Tools/Research/Kanana/Run-KananaEval.ps1` runs the 20-case Korean routing
benchmark against any Hugging Face causal LM with the same cases, tools and
grading, so the pass rate is comparable to
[`benchmarks/fc-model-benchmark.md`](benchmarks/fc-model-benchmark.md). The
latency is not comparable — it is a desktop GPU in bf16.
