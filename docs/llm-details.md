# LLM Details

LLM setup and benchmark records for the Unity LiteRT-LM bridge.
**The criterion throughout is Android on-device execution** (Snapdragon 865 /
kona / 7.5 GiB RAM, device `46a880a0`); Windows numbers are development aids
only. The README keeps a summary; the evidence is here.

## Model inventory — everything measured, with its source

The sections below quote whichever models are relevant to a decision. This table
is the complete list, so that "did we ever try X?" has an answer that does not
depend on reading all of them.

**Ran on LiteRT** — `.litertlm` bundles, measured on device or on the Windows CLI.
FC is the 20-case Korean routing score ([§4](#4-function-calling-tiers)); decode
is Android CPU unless noted.

| Model | Build measured | Size | FC | Decode | Verdict | Source |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Qwen2.5-0.5B-Instruct | project `wi4b64_ekv1280` | 265 MB | 2/20 | 35.5 | **resident** — fastest chat, not a router | [project i4](https://huggingface.co/leuconoe/litert-lm-unity-quantized) · [upstream](https://huggingface.co/litert-community/Qwen2.5-0.5B-Instruct) |
| Qwen2.5-0.5B-Instruct | official `q8` | 521 MB | — | 25.7 | superseded by the i4 build; **fails engine creation on GPU** | [litert-community](https://huggingface.co/litert-community/Qwen2.5-0.5B-Instruct) |
| Qwen2.5-1.5B-Instruct | official `q8_ekv4096` | 1.5 GB | — | 8.5 | not deployed — the size does not pay off | [litert-community](https://huggingface.co/litert-community/Qwen2.5-1.5B-Instruct) |
| Qwen2.5-1.5B-Instruct | project `wi4b64` prototype | 790 MB | 8/20 | 18.2† | not deployed | [project i4](https://huggingface.co/leuconoe/litert-lm-unity-quantized) |
| Qwen3-0.6B | stock | 586 MB | **20/20** | 6.1 | best accuracy on the set, but 22.5 s per command | [litert-community](https://huggingface.co/litert-community/Qwen3-0.6B) |
| Qwen3-0.6B | project `mixed_int4` | 475 MB | **18/20** | 20.9 | **resident — small-tier FC pick** | [litert-community](https://huggingface.co/litert-community/Qwen3-0.6B) |
| Gemma3-270M-IT | official `q8` | 290 MB | — | 27.7 | not deployed — answers missed our smoke bar. Cannot be i4'd (no f32 source) | [litert-community](https://huggingface.co/litert-community/gemma-3-270m-it) |
| Gemma3-1B-IT | official `int4` | 557 MB | 3/20 | 16.0 | chat fallback only — **not a router** | [litert-community](https://huggingface.co/litert-community/Gemma3-1B-IT) |
| FunctionGemma-270M | `ft-mobile-actions` | 276 MB | 6/7‡ | 38.2† | comparison point — built around fine-tuning, so excluded from selection | [litert-community](https://huggingface.co/litert-community/functiongemma-270m-ft-mobile-actions) |
| LFM2.5-1.2B-Instruct | `int4` | 702 MB | **17/20** | 16.8 | **mid-tier FC pick**. Requires the v0.14 runtime | [LiquidAI](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) |
| gemma-4-E2B-it | official QAT `wNa8o8` | 2.6 GB | **19/20** | 5.2 | **on-demand flagship** — chat, image, audio and FC in one model | [litert-community](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) |

† Windows CLI decode — this model was never deployed to the device.
‡ Scored on the separate 7-case English mobile-actions set, not the Korean 20.

**Screened and rejected** — none of these reached a device measurement, and the
reason is recorded so nobody re-runs the search.

| Model | Why not | Source |
| --- | --- | --- |
| Kanana-2-1.3B-Instruct | **No LiteRT path** (custom `kanana2_tiny` architecture) and the licence requires a separate Kakao agreement for on-premise delivery. Scored 17/20 on desktop — see [§6](#6-evaluated-and-rejected) | [kakaocorp](https://huggingface.co/kakaocorp/kanana-2-1.3b-instruct) |
| Qwen3.5-0.8B-MTP | MTP is a llama.cpp feature; litert-torch does not support the architecture, and the community port produced incoherent output on v0.14 | [Qwen](https://huggingface.co/Qwen/Qwen3.5-0.8B) |
| Bonsai-1.7B (ternary) | The 1-bit size advantage does not survive LiteRT's int4/8 requantization, so the format's whole benefit is lost | [prism-ml](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-gguf) |
| Hammer2.1-0.5b | **CC-BY-NC** — non-commercial, unusable in a delivered product | [MadeAgents](https://huggingface.co/MadeAgents/Hammer2.1-0.5b) |
| VibeVoice-ASR | 8.7 B — outside the on-device size budget | [microsoft](https://huggingface.co/microsoft/VibeVoice-ASR-BitNet) |
| gemma-4-E2B GGUF | ~16× faster decode on desktop CUDA, but **no Android path**. Desktop experimentation only — [comparison](benchmarks/gemma4-gguf-vs-litertlm.md) | [unsloth](https://huggingface.co/unsloth/gemma-4-E2B-it-qat-mobile-GGUF) |

ASR and TTS models are inventoried separately in
[`asr-details.md`](asr-details.md) and
[`tts-model-research.md`](tts-model-research.md).

## 1. Recommended combinations by device RAM

`Mobile` verdicts: **resident** = fine to keep loaded at all times ·
**on-demand** = load when needed, then release · **not deployed** = does not
fit this project's on-device targets.

| Model | Size | Idle PSS | Peak PSS | Mobile | Role |
| --- | ---: | ---: | ---: | --- | --- |
| `LLM/qwen2.5-0.5b/…_wi4b64_ekv1280.litertlm` (project i4) | 265 MB | ~0.38 GB | — | **resident** | Fastest chat at 35.5 tok/s. Not an FC router |
| `LLM/qwen3-0.6b/qwen3_0_6b_mixed_int4.litertlm` | 475 MB | ~0.36 GB | — | **resident** | Small-tier FC pick 18/20, plus 20.9 tok/s chat |
| `LLM/gemma3-1b/gemma3-1b-it-int4.litertlm` | 557 MB | 0.40 GB | — | **resident** | Chat fallback 16.0 tok/s. 3/20 as a router — do not use for FC |
| `LLM/lfm2.5-1.2b/LFM2.5-1.2B-Instruct_int4.litertlm` | 702 MB | — | — | resident (6 GB+) | Mid-tier FC 17/20, 16.8 tok/s. Requires the v0.14 runtime |
| `Multimodal/gemma-4-e2b/gemma-4-E2B-it.litertlm` | 2.6 GB | 0.48 GB | **3.6 GB** | **on-demand (8 GB+)** | Chat, image, audio and FC in one model. See the warning below |
| `LLM/qwen2.5-1.5b/…_q8_ekv4096.litertlm` | 1.5 GB | 0.35 GB | — | **not deployed** | 8.5 tok/s CPU, 3.6–3.8 s per turn — the size does not pay off for our targets |
| `LLM/gemma3-270m/gemma3-270m-it-q8.litertlm` | 290 MB | 0.35 GB | — | **not deployed** | Fast at 27.7 tok/s, but its answers did not meet our smoke-test bar. Cannot be i4'd (no f32 source) |

Recommended stacks (same as the README). Add **one** ASR model sized to the
utterance lengths you handle — see [`asr-details.md`](asr-details.md).

| Device RAM | LLM | + one ASR | Resident total |
| --- | --- | --- | ---: |
| 4–6 GB | Qwen2.5-0.5B i4 (265 MB) | base-acft-ko 5s (101 MB) | ~370 MB |
| 6–8 GB | Qwen3-0.6B i4 (475 MB) | whichever fits the use case (77–101 MB) | ~570 MB |
| 8 GB+ | gemma-4-E2B QAT (2.6 GB) | none — audio input is built in (4.1 s, content-exact) | 0.48 GB idle / 3.6 GB on image turns |

⚠️ **gemma-4-E2B's real constraint is the memory spike on image turns, not the
2.6 GB file.** Text and FC turns sit at 0.48 GB PSS, but processing one
700×467 image (2340 vision patches) climbs to **3.6 GB** (measured with
MemAvailable holding at 3.4 GB, zero lowmemorykiller events or crashes). On a
6 GB device it is unsafe to keep it resident alongside anything else — **load
it only for the multimodal stretch**.

⚠️ **Do not use a chat model for function calling.** gemma3-1b scores 3/20 and
the qwen2.5 family 2–8/20 as routers. Pick from Qwen3-0.6B (small),
LFM2.5-1.2B (mid) or gemma-4-E2B (flagship).

## 2. Device measurements (v0.14, 2026-07-23)

Smoke (2 chat turns) plus benchmark (3 runs × 64 prefill / 32 decode tokens),
warm device at 52–57 °C, take3 AAR. Full logs:
[`benchmarks/device-cycle1-baseline.md`](benchmarks/device-cycle1-baseline.md).

**Hit rate** is the 20-case Korean FC routing score (§4); chat quality is
reported separately as a PASS/FAIL on Korean/English response coherence.

| Model | Backend | Init s | Prefill tok/s | Decode tok/s | FC hit rate | Chat | Note |
| --- | --- | ---: | ---: | ---: | ---: | :-: | --- |
| `qwen2.5-0.5b` project i4 (264 MB) | CPU | **1.37** | **218.3** | **35.5** | 2–8/20 ✗ | PASS | Fastest. Half the size of the official q8 with +38 % decode |
| `qwen3-0.6b` mixed_int4 (475 MB) | CPU | 1.66 | 31.8 | 20.9 | **18/20** | PASS | `/think` on by default → longer turns. Prefill is low for its size |
| `lfm2.5-1.2b` int4 (702 MB) | CPU | 7.73 | 57.1 | 16.8 | 17/20 | PASS | Loads only on v0.14 |
| `gemma3-1b` int4 (557 MB) | CPU | 4.14 | 100.9 | 16.0 | 3/20 ✗ | PASS | ~6 % below the cool-device record (warm run) |
| `gemma3-1b` int4 | GPU | 9.73 | **184.2** | 13.7 | 3/20 ✗ | PASS | OpenCL delegate + GPU TopK sampler |
| `gemma-4-E2B` QAT (2.6 GB) | CPU/GPU | 1.2–18.5 | 141 / 431 | 5.2 / 7.0 | **19/20** | PASS | Multimodal duty. Slow decode, but covers everything in one model |

All configurations PASS (coherent Korean/English output, zero crashes or OOM
across the cycle).

**Speed does not predict FC accuracy** — the fastest model (35.5 tok/s) is the
lowest-scoring router on our set, and the small FC pick (20.9 tok/s) is within one point of the
flagship.

### Multimodal turns (gemma-4-E2B, cycles 2–3)

| Input | Backends (llm/vision/audio) | Latency | Result |
| --- | --- | ---: | --- |
| 700×467 image | CPU/CPU/– | 23.4 s | Accurate (PSS 3.6 GB) |
| 700×467 image | GPU/GPU/– | **7.6 s** | Same content — GPU is 3.1× faster |
| 3.8 s Korean audio | CPU/–/CPU | 4.1 s | Content-exact including the year |
| Image + utterance → tool call | CPU | 40.7 s E2E | Bare tool JSON, no constrained decoding |

`maxNumTokens 4000` is required (2340 image patches plus a ~1.3k-token tools
prompt). Audio transcription at 4.1 s beat every dedicated ASR model on that
clip and got the spacing right — **if the LLM is already resident, you can do
transcription and function calling in a single turn without a separate ASR
model**.

## 3. Backend choice (CPU vs GPU)

- **Android (Adreno 650 / kona)**: GPU decode is **slower** than CPU
  (13.7 vs 16.0 tok/s). This is neither a sampler fallback nor thermal
  throttling — single-token decode is bandwidth-bound, so per-step OpenCL
  dispatch and synchronization overhead exceeds the compute win. GPU does win
  prefill by ~1.8× (184 vs 101 tok/s) and multimodal image turns by ~3.1×.
  → **CPU for chat, GPU for long-prompt prefill and multimodal.**
- **Windows (RTX 4090, WebGPU/Dawn over D3D12, reference only)**: GPU wins on
  decode but loses everywhere else, and the sample scenes default to **CPU**
  because of it. Measured with `litert_lm_main`, one process per request:

  | gemma-4-E2B | CPU | GPU |
  | --- | ---: | ---: |
  | Init executor | 305 ms | 5,097 ms |
  | Time to first token | 0.49 s | 2.38 s |
  | Prefill | 43.9 tok/s | 7.6 tok/s |
  | Decode | 13.2 tok/s | 53.3 tok/s |

  The CLI is stateless, so every request pays executor init again: about 4.8 s
  extra on GPU, which only 84-odd output tokens of faster decode would repay.
  Chat turns and function-call routing are far shorter than that, and prefill —
  which dominates long prompts and image turns — is 6× *slower* on GPU here.
  OpenCL context creation fails on this build, so GPU runs through WebGPU
  (Dawn → D3D12) rather than the native path Android uses, and there is no
  shader cache equivalent to the CPU `xnnpack_cache_*` file.
  `LiteRtLmWindowsCliClient` also falls back to CPU once per session on failure.
  **The profile is the opposite of Android's, so never pick a device backend
  from desktop results.**

## 4. Function-calling tiers

The 20-case Korean routing score is measured on Windows; the top tiers were
re-confirmed end-to-end on the device. Per-case detail:
[`benchmarks/fc-model-benchmark.md`](benchmarks/fc-model-benchmark.md).

| Tier | Model | Score | Device confirmation |
| --- | --- | --- | --- |
| Flagship (~2.5 GB) | gemma-4-E2B (official QAT wNa8o8) | **19/20** | Image + utterance → tool call, 40.7 s PASS |
| Small (475 MB) | qwen3_0_6b_mixed_int4 + Hermes-style prompt | 18/20 | Chat path PASS |
| Mid (702 MB) | LFM2.5-1.2B int4 + Hermes-style prompt | 17/20 | Load and chat PASS (fastest FC at 23.6 tok/s) |
| Not a router | gemma3-1b (3/20), qwen2.5 family (2–8/20) | — | gemma3-1b does PASS on device as the LLM inside the single-tool voice FC pipeline (15.5 s E2E) |

## 5. Self-quantization pipeline (i4)

Under the int4-minimum-tier policy, `.litertlm` bundles go through
unpack → quantize (`ai_edge_quantizer`) → repack (`litert-lm-builder`).

- Recipe: `dynamic_wi4b64_afp32` (4-bit weights, fp16 scale per 64-value block,
  fp32 activations) with sensitive scopes (embeddings/logits, encoders) kept at i8
- Channelwise `wi4c` is **never used** — accuracy degrades sharply in our tests
- Products: qwen2.5-0.5b i4 (265 MB, device-validated at +38 % decode) and
  qwen2.5-1.5b i4 (790 MB)
- gemma3-270m / FunctionGemma cannot be i4'd (no f32 source; i8→i4 is a no-op)
- Detail: [`External/ModelWork/README-i4-prototypes.md`](../External/ModelWork/README-i4-prototypes.md)

**Why this matters on mobile**: int4 buys size and speed at once. Decode is
memory-bandwidth-bound, so smaller weights translate directly into throughput
(q8 → i4 gives half the size and +38 % decode). int2, Q5 and 1.58-bit are
impossible — LiteRT has no kernels for them.

## 6. Evaluated and rejected

### Kanana-2-1.3B-Instruct (2026-07-29) — good model, two hard blockers

Kakao's Korean SLM, released 2026-07-27. Measured because its published Korean
scores are strong for the size: HAE-RAE 75.34 against Qwen3-1.7B-Base's 55.54,
KoMT-Bench 6.54 against 5.29, BFCL-v3 Live 69.64 against 65.48.

**It scores 17/20 on our routing set** — level with LFM2.5-1.2B, one below the
Qwen3-0.6B int4 we ship. Reproduce with
`.\Tools\Research\Kanana\Run-KananaEval.ps1`; raw records in
`Builds/Logs/kanana-2-1.3b-instruct-fc-bench.jsonl`.

| | |
| --- | --- |
| Parameters | 1,291,478,272 (bf16, 2.58 GB; an fp32 copy also ships) |
| Architecture | `Kanana2TinyForCausalLM` — 32 layers, 32 heads, 8 KV heads, **3:1 hybrid sliding-window / full attention**, per-layer-type RoPE (YaRN on full-attention layers only) |
| Context | 32,768 |
| Tool format | qwen3-coder style (`<tool_call><function=Name><parameter=…>`) |
| Measured on | RTX 4090, bf16, greedy, `transformers` 5.14.1 |

The three failures are worth reading, because two of them are the same bug:

| Case | Asked | Answered |
| --- | --- | --- |
| B11 | 어제 (from 2026-04-24) | `2025-04-23` — right day, **wrong year** |
| B19 | 지난달 | `2025-03-01` — right month, **wrong year** |
| B17 | 내일 날씨는 어때? | Prose refusal instead of the `DefaultResponse` tool |

Relative *past* dates slip a year while absolute and same-day ones are correct
(B10, B12, B13, B14, B20 all pass), so this is date arithmetic, not tool
selection. A Korean chat check also mistranslated 백 as "one thousand", which is
a numeral error in the direction that matters for a command UI.

**Blocker 1 — no LiteRT path.** The architecture is custom and shipped as remote
code (`trust_remote_code`). litert-torch converts a fixed set of architectures,
and a 3:1 hybrid SWA layout with two different RoPE configurations per layer type
is not one of them; supporting it means authoring the model in ai-edge-torch, not
running a converter. Nothing measured above runs on the device today.

**Blocker 2 — licence.** Weights are under the
[Kanana Open License Agreement](https://huggingface.co/kakaocorp/kanana-2-1.3b-instruct/blob/main/LICENSE),
not a permissive licence. §4.1(ii) requires a **separate commercial licence from
Kakao**, at Kakao's sole discretion, to offer the model to third parties "as part
of a system integration (SI) or on-premise deployment solution" — which is
exactly this product's shape. §4.1(i) covers API/cloud resale. Evaluation is
unaffected; delivery is a negotiation.

Neither blocker is worth clearing for 17/20 when Qwen3-0.6B int4 already gives
18/20 at 475 MB, on device, under Apache-2.0.

### Earlier screening (2026-07-23)

- **Qwen3.5-0.8B-MTP** — not feasible here. MTP is a llama.cpp feature,
  litert-torch does not support the architecture, and the community litertlm
  port did not produce coherent output on v0.14
- **Bonsai-1.7B** — not adopted. Its 1-bit size advantage does not survive the
  int4/8 requantization that LiteRT conversion applies, so the benefit that
  motivates the format is lost on this runtime
- **VibeVoice-ASR** — 8.7B, outside the on-device size budget for this project
- **GGUF / llama.cpp** — Windows CUDA decode ~243 tok/s (~16× litertlm CPU) but
  **no Android path**. llama.cpp is the desktop experimentation stack; litertlm
  is the product runtime.
  [`benchmarks/gemma4-gguf-vs-litertlm.md`](benchmarks/gemma4-gguf-vs-litertlm.md)

## 7. Smoke tests

The benchmark runner reuses a single APK. Push the model and runtime JSON config
into app storage, then launch the same build:

```powershell
.\Tools\Windows\Tests\Run-LiteRtLmAndroidDeviceBenchmarks.ps1 `
  -DeviceSerial <device-serial> `
  -BenchmarkName gemma-4-e2b-it-gpu,gemma-4-e2b-it-cpu `
  -SingleApkPath Builds\Android\LiteRtLmAndroidSmokeTest-gemma3-270m-it-q8-CPU.apk `
  -TimeoutSeconds 900
```

Model transfer is skipped on later runs when the device-side file size already
matches the local file.

---

## Appendix — 2026-05-16 cool-device record (pre-v0.14, superseded)

See §2 for current numbers. This table is kept as the **cool-device reference**
(§2 was measured warm at 52–57 °C and runs ~6 % lower). The q8 models here have
since been replaced by project i4 builds, so **do not use this table to choose a
model.**

| Model | Backend | Status | File MB | PSS MB | Init s | Prefill tok/s | Decode tok/s |
| --- | --- | :-: | ---: | ---: | ---: | ---: | ---: |
| gemma-4-E2B-it | GPU | PASS | 2468.3 | 470.2 | 10.06 | 431.1 | 7.0 |
| gemma-4-E2B-it | CPU | PASS | 2468.3 | 386.4 | 4.78 | 141.0 | 5.2 |
| gemma3-270m-it-q8 | GPU | PASS | 289.9 | 425.2 | 6.49 | 377.6 | 26.0 |
| gemma3-270m-it-q8 | CPU | PASS | 289.9 | 346.3 | 0.87 | 101.3 | 27.7 |
| gemma3-1b-it-int4 | GPU | PASS | 557.3 | 471.4 | 7.91 | 197.0 | 16.4 |
| gemma3-1b-it-int4 | CPU | PASS | 557.3 | 404.1 | 3.44 | 108.1 | 17.5 |
| Qwen2.5-0.5B-Instruct-q8 | GPU | **FAIL** | 520.7 | 458.7 | — | — | — |
| Qwen2.5-0.5B-Instruct-q8 | CPU | PASS | 520.7 | 384.8 | 1.31 | 206.8 | 25.7 |
| Qwen2.5-1.5B q8 ekv4096 | GPU | PASS | 1523.9 | 365.2 | 9.41 | 88.9 | 9.9 |
| Qwen2.5-1.5B q8 ekv4096 | CPU | PASS | 1523.9 | 347.3 | 4.29 | 30.7 | 8.5 |
| Qwen3-0.6B (stock) | GPU | PASS | 585.8 | 395.5 | 5.72 | 96.5 | 9.1 |
| Qwen3-0.6B (stock) | CPU | PASS | 585.8 | 359.4 | 1.74 | 30.1 | 6.1 |

Worth recording:

- **Qwen2.5-0.5B q8 fails engine creation on GPU**
  (`llm_litert_compiled_model_executor.cc:1546`, after partial delegation of
  1272 ops to GPU / 54 to CPU). The HF model card only publishes Android CPU
  results — treat that file as CPU-only. It has since been replaced by the
  project i4 build.
- The 2026-05 AAR logged `GPU sampler unavailable. Falling back to CPU
  sampling.` The v0.14 AAR loads a native OpenCL TopK sampler
  (`LiteRtTopKOpenClSampler`), so that fallback no longer occurs — yet GPU
  decode is still slower on Adreno 650 for the structural reason in §3.
- Stock Qwen3-0.6B decoded at 6–9 tok/s; after mixed_int4 conversion it reaches
  20.9 tok/s, which is what made it the small-tier FC pick.
