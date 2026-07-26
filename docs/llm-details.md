# LLM Details

LLM setup and benchmark records for the Unity LiteRT-LM bridge.
**The criterion throughout is Android on-device execution** (Snapdragon 865 /
kona / 7.5 GiB RAM, device `46a880a0`); Windows numbers are development aids
only. The README keeps a summary; the evidence is here.

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
- **Windows (RTX 4090, WebGPU/Dawn over D3D12, reference only)**: v0.14 fixes
  the GPU backend — decode 49.3 tok/s vs 17.8–26.8 on CPU (~2×). GPU is the
  Windows default and `LiteRtLmWindowsCliClient` falls back to CPU once per
  session on failure. **The profile is the opposite of Android's, so never pick
  a device backend from desktop results.**

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

## 6. Evaluated and rejected (2026-07-23)

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
.\Tools\Windows\Run-LiteRtLmAndroidDeviceBenchmarks.ps1 `
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
