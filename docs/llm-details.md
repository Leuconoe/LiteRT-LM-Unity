# LLM Details

This document records Android LLM setup and benchmark results for the Unity
LiteRT-LM bridge. The README keeps only requirements and recommended models.

## 2026-07-23 Update — LiteRT-LM v0.14

Framework upgraded to v0.14.0 (`unity-v0.14.0` fork branch, patch-based
customization). All results below are from the 2026-07-23 session; source
benchmark docs live in `docs/benchmarks/`.

### Device benchmark (46a880a0, Qualcomm kona / SM8250, Android 12)

Smoke (2 chat turns) + standalone benchmark (3 runs × 64 prefill / 32 decode
tokens), warm device (52–57 °C), take3 AAR. Full detail incl. logs:
[`docs/benchmarks/device-cycle1-baseline.md`](benchmarks/device-cycle1-baseline.md).

| Model | Backend | Init s | Prefill tok/s | Decode tok/s | Note |
| --- | --- | ---: | ---: | ---: | --- |
| `LLM/gemma3-1b/gemma3-1b-it-int4.litertlm` (557 MB) | GPU | 9.73 | 184.2 | 13.7 | OpenCL delegate + GPU TopK sampler |
| same | CPU | 4.14 | 100.9 | 16.0 | ~6 % below cool-device history (warm run) |
| `LLM/qwen3-0.6b/qwen3_0_6b_mixed_int4.litertlm` (475 MB) | CPU | 1.66 | 31.8 | 20.9 | `/think` mode on by default → long turns; prefill anomaly under investigation |
| `LLM/lfm2.5-1.2b/LFM2.5-1.2B-Instruct_int4.litertlm` (702 MB) | CPU | 7.73 | 57.1 | 16.8 | Architecture supported on device (v0.14 requirement) |
| `LLM/qwen2.5-0.5b/Qwen2.5-0.5B-Instruct_wi4b64_ekv1280.litertlm` (264 MB, self-made i4) | CPU | 1.37 | 218.3 | 35.5 | Fastest of the set; +38 % decode vs stock q8 at half the size |

All five configurations PASS (coherent Korean/English output, zero
crashes/OOM across the cycle).

### GPU vs CPU decode guidance

- **Android (Adreno 650 / kona)**: GPU decode is *slower* than CPU
  (13.7 vs 16.0 tok/s) — verified not a sampler fallback and not thermal;
  single-token decode is bandwidth-bound and per-step OpenCL dispatch overhead
  exceeds the compute win. GPU wins prefill ~1.8× (184 vs 101 tok/s) and
  multimodal image turns ~3.1×. Use **CPU for decode-heavy chat**, **GPU for
  long-prompt prefill and multimodal** on this SoC class.
- **Windows (RTX 4090, WebGPU/Dawn over D3D12)**: v0.14 fixes the previously
  broken GPU backend — decode 49.3 tok/s vs 17.8–26.8 CPU (~2×). GPU is now
  the default Windows backend; `LiteRtLmWindowsCliClient` falls back to CPU
  once per session on GPU failure (covers the ~5.7 s GPU init and
  non-D3D12 machines).

### Windows function-calling benchmark (20-case Korean routing)

Full results and per-model notes:
[`docs/benchmarks/fc-model-benchmark.md`](benchmarks/fc-model-benchmark.md).

| Tier | Pick | Score |
| --- | --- | --- |
| Flagship (~2.5 GB) | gemma-4-E2B (QAT wNa8o8 official build) | 19/20, 15.5 tok/s, constrained decoding |
| Mid (~0.7 GB) | LFM2.5-1.2B int4 + Hermes-style prompt | 17/20, 23.6 tok/s (fastest) |
| Small (~0.5 GB) | qwen3_0_6b_mixed_int4 + Hermes-style prompt | 18/20, 475 MB |
| Not usable as router | gemma3-1b (3/20), qwen2.5 family (2–8/20) | keep for chat/prototype roles |

Both function-calling pipelines (ASR→LLM voice routing and multimodal
image+utterance) were also validated end-to-end on the device — see the
cycle-3 ledger in `device-cycle1-baseline.md`.

### i4 self-quantization pipeline

Per the int4-minimum-tier policy, an unpack → quantize (`ai_edge_quantizer`)
→ repack pipeline for `.litertlm` bundles is established (litert-lm-builder).
Recipe: `dynamic_wi4b64_afp32` (4-bit weights, fp16 scale per 64-value block,
fp32 activations) with sensitive scopes (embeddings/logits, encoders) kept at
i8 — channelwise `wi4c` is never used (quality collapse). Validated products:
qwen2.5-0.5b i4 (265 MB) and qwen2.5-1.5b i4 (790 MB); the 0.5b i4 is
device-validated (+38 % decode vs q8). gemma3-270m/FunctionGemma cannot be
i4'd (no f32 source; i8→i4 is a no-op). Details:
[`External/ModelWork/README-i4-prototypes.md`](../External/ModelWork/README-i4-prototypes.md).

### Evaluated and rejected (2026-07-23)

- **Qwen3.5-0.8B-MTP**: not feasible — MTP is llama.cpp-only, the architecture
  is unsupported by litert-torch, and the community litertlm port produces
  collapsed output on v0.14.
- **Bonsai-1.7B**: skip recommended — its 1-bit size advantage is destroyed by
  int4/8 requantization in LiteRT conversion (awaiting final user decision).
- **VibeVoice-ASR**: 8.7B — not viable on-device; dropped.
- **GGUF/llama.cpp on Windows**: CUDA decode ~243 tok/s (~16× litertlm CPU),
  but no Android path — llama.cpp is the Windows experimentation stack,
  litertlm is the product runtime. See
  [`docs/benchmarks/gemma4-gguf-vs-litertlm.md`](benchmarks/gemma4-gguf-vs-litertlm.md).

## Setup (historical baseline, 2026-05-16)

- Report date: 2026-05-16
- Unity package: `com.Leuconoe.LiteRTLMUnity`
- Device class: Qualcomm Android 12 physical device, about 7.52 GiB RAM
- Test flow: initialize model, run two short chat turns, then run three
  standalone benchmark iterations with 64 prefill tokens and 32 decode tokens.
- Runtime inputs: one smoke-test APK plus model/config files pushed to app
  device storage.

Device serials and local absolute paths are intentionally omitted.

## Benchmarks (historical, 2026-05-16 — pre-v0.14)

Superseded by the 2026-07-23 device benchmark above for current numbers;
kept as the cool-device reference (the 2026-07-23 run was warm).

### Results

The run tested the `.litertlm` files that were present in
`Assets/StreamingAssets` (now organized under `LLM/<model>/` and
`Multimodal/<model>/` subfolders). GPU was tried first for each model, and CPU
was tested as the comparison path.

| Model | Source | Backend | Status | GPU evidence | File size | Memory PSS | Init s | First chat s | Second chat s | Benchmark avg s | Prefill tok/s | Decode tok/s | Note |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `Multimodal/gemma-4-e2b/gemma-4-E2B-it.litertlm` | [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) | GPU | PASS | Native OpenCL + CPU sampler fallback | 2468.25 MB | 470.15 MB | 10.055 | 0.974 | 0.712 | 8.189 | 431.121 | 7.049 | Best quality recommendation when memory allows. |
| `Multimodal/gemma-4-e2b/gemma-4-E2B-it.litertlm` | same | CPU | PASS | N/A | 2468.25 MB | 386.39 MB | 4.781 | 1.734 | 1.311 | 7.540 | 141.014 | 5.206 | CPU is usable, but GPU gives much faster prefill. |
| `LLM/gemma3-270m/gemma3-270m-it-q8.litertlm` | [litert-community/gemma-3-270m-it](https://huggingface.co/litert-community/gemma-3-270m-it) | GPU | PASS | Native OpenCL + CPU sampler fallback | 289.92 MB | 425.18 MB | 6.493 | 1.302 | 1.299 | 3.507 | 377.555 | 25.984 | Very fast, but generic smoke output quality is weak. |
| `LLM/gemma3-270m/gemma3-270m-it-q8.litertlm` | same | CPU | PASS | N/A | 289.92 MB | 346.32 MB | 0.873 | 0.950 | 0.894 | 2.027 | 101.305 | 27.677 | Fastest startup among tested models. |
| `LLM/gemma3-1b/gemma3-1b-it-int4.litertlm` | [litert-community/Gemma3-1B-IT](https://huggingface.co/litert-community/Gemma3-1B-IT) | GPU | PASS | Native OpenCL + CPU sampler fallback | 557.34 MB | 471.42 MB | 7.913 | 1.014 | 2.039 | 6.285 | 197.027 | 16.382 | Compact GPU-capable fallback. |
| `LLM/gemma3-1b/gemma3-1b-it-int4.litertlm` | same | CPU | PASS | N/A | 557.34 MB | 404.05 MB | 3.441 | 1.320 | 2.261 | 3.058 | 108.090 | 17.523 | CPU benchmark was faster overall for this short test. |
| `LLM/qwen2.5-0.5b/Qwen2.5-0.5B-Instruct-q8.litertlm` | [litert-community/Qwen2.5-0.5B-Instruct](https://huggingface.co/litert-community/Qwen2.5-0.5B-Instruct) | GPU | FAIL | Native OpenCL attempted | 520.73 MB | 458.68 MB | N/A | N/A | N/A | N/A | N/A | N/A | Engine creation failed after partial GPU delegation; the model card publishes Android CPU results, not GPU results. |
| `LLM/qwen2.5-0.5b/Qwen2.5-0.5B-Instruct-q8.litertlm` | same | CPU | PASS | N/A | 520.73 MB | 384.80 MB | 1.305 | 0.698 | 0.703 | 2.171 | 206.757 | 25.742 | Recommended fast CPU alternative. |
| `LLM/qwen2.5-1.5b/Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm` | [litert-community/Qwen2.5-1.5B-Instruct](https://huggingface.co/litert-community/Qwen2.5-1.5B-Instruct) | GPU | PASS | Native OpenCL + CPU sampler fallback | 1523.91 MB | 365.20 MB | 9.412 | 1.969 | 2.047 | 9.904 | 88.850 | 9.946 | Larger Qwen option; useful for compatibility comparison. |
| `LLM/qwen2.5-1.5b/Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm` | same | CPU | PASS | N/A | 1523.91 MB | 347.30 MB | 4.290 | 3.845 | 3.584 | 6.597 | 30.735 | 8.537 | CPU passed, but latency is high. |
| `LLM/qwen3-0.6b/Qwen3-0.6B.litertlm` | [litert-community/Qwen3-0.6B](https://huggingface.co/litert-community/Qwen3-0.6B) | GPU | PASS | Native OpenCL + CPU sampler fallback | 585.78 MB | 395.54 MB | 5.721 | 3.342 | 3.340 | 6.983 | 96.503 | 9.104 | Runtime passed; prompt quality still needs model-specific tuning. |
| `LLM/qwen3-0.6b/Qwen3-0.6B.litertlm` | same | CPU | PASS | N/A | 585.78 MB | 359.39 MB | 1.741 | 6.455 | 6.182 | 7.721 | 30.141 | 6.143 | CPU passed, but chat turns were slow. |

### GPU Notes

Note (2026-07-23): the v0.14 AAR loads a native OpenCL TopK sampler
(`LiteRtTopKOpenClSampler`) — the sampler fallback below no longer occurs;
GPU decode being slower than CPU on Adreno 650 persists and is structural
(see the guidance section above).

Native OpenCL model execution is working for most GPU runs. The May AAR
still reported sampler fallback:

```text
GPU sampler unavailable. Falling back to CPU sampling.
```

This means model graph execution can use OpenCL while top-k/top-p sampling falls
back to CPU. That is why the GPU evidence column uses `Native OpenCL + CPU
sampler fallback`.

`Qwen2.5-0.5B-Instruct-q8.litertlm` failed on the GPU path with:

```text
Failed to create engine: INTERNAL:
[runtime/executor/llm_litert_compiled_model_executor.cc:1546]
```

The log showed OpenCL initialization and partial GPU delegation before failure:

```text
1272 operations will run on the GPU, and the remaining 54 operations will run on the CPU.
```

The Hugging Face model card for this Qwen2.5 0.5B LiteRT model lists Android CPU
benchmark results, but does not advertise a tested GPU result. Treat CPU as the
supported path for this file until a GPU-specific variant is available.

### Recommendations

1. Use `gemma-4-E2B-it.litertlm` as the primary quality model when memory
   allows.
2. Use `Qwen2.5-0.5B-Instruct-q8.litertlm` as the fast CPU fallback.
3. Use `gemma3-1b-it-int4.litertlm` when a smaller Gemma-family model is needed
   and native GPU execution is preferred.
4. Treat `gemma3-270m-it-q8.litertlm` and `Qwen3-0.6B.litertlm` as experimental
   candidates until prompt quality is tuned and checked with a task-specific
   benchmark.

## Smoke Tests

The current benchmark runner can reuse a single APK. Push the selected model and
runtime JSON config into app storage, then launch the same build:

```powershell
.\Tools\Windows\Run-LiteRtLmAndroidDeviceBenchmarks.ps1 `
  -DeviceSerial <device-serial> `
  -BenchmarkName gemma-4-e2b-it-gpu,gemma-4-e2b-it-cpu `
  -SingleApkPath Builds\Android\LiteRtLmAndroidSmokeTest-gemma3-270m-it-q8-CPU.apk `
  -TimeoutSeconds 900
```

Model files are skipped on later runs when the device-side file size already
matches the local file.
