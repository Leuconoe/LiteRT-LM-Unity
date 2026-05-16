# LLM Details

This document records Android LLM setup and benchmark results for the Unity
LiteRT-LM bridge. The README keeps only requirements and recommended models.

## Setup

- Report date: 2026-05-16
- Unity package: `com.Leuconoe.LiteRTLMUnity`
- Device class: Qualcomm Android 12 physical device, about 7.52 GiB RAM
- Test flow: initialize model, run two short chat turns, then run three
  standalone benchmark iterations with 64 prefill tokens and 32 decode tokens.
- Runtime inputs: one smoke-test APK plus model/config files pushed to app
  device storage.

Device serials and local absolute paths are intentionally omitted.

## Benchmarks

### Latest Results

The current run tested the `.litertlm` files that were present in
`Assets/StreamingAssets`. GPU was tried first for each model, and CPU was tested
as the comparison path.

| Model | Source | Backend | Status | GPU evidence | File size | Memory PSS | Init s | First chat s | Second chat s | Benchmark avg s | Prefill tok/s | Decode tok/s | Note |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `gemma-4-E2B-it.litertlm` | [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) | GPU | PASS | Native OpenCL + CPU sampler fallback | 2468.25 MB | 470.15 MB | 10.055 | 0.974 | 0.712 | 8.189 | 431.121 | 7.049 | Best quality recommendation when memory allows. |
| `gemma-4-E2B-it.litertlm` | same | CPU | PASS | N/A | 2468.25 MB | 386.39 MB | 4.781 | 1.734 | 1.311 | 7.540 | 141.014 | 5.206 | CPU is usable, but GPU gives much faster prefill. |
| `gemma3-270m-it-q8.litertlm` | [litert-community/gemma-3-270m-it](https://huggingface.co/litert-community/gemma-3-270m-it) | GPU | PASS | Native OpenCL + CPU sampler fallback | 289.92 MB | 425.18 MB | 6.493 | 1.302 | 1.299 | 3.507 | 377.555 | 25.984 | Very fast, but generic smoke output quality is weak. |
| `gemma3-270m-it-q8.litertlm` | same | CPU | PASS | N/A | 289.92 MB | 346.32 MB | 0.873 | 0.950 | 0.894 | 2.027 | 101.305 | 27.677 | Fastest startup among tested models. |
| `gemma3-1b-it-int4.litertlm` | [litert-community/Gemma3-1B-IT](https://huggingface.co/litert-community/Gemma3-1B-IT) | GPU | PASS | Native OpenCL + CPU sampler fallback | 557.34 MB | 471.42 MB | 7.913 | 1.014 | 2.039 | 6.285 | 197.027 | 16.382 | Compact GPU-capable fallback. |
| `gemma3-1b-it-int4.litertlm` | same | CPU | PASS | N/A | 557.34 MB | 404.05 MB | 3.441 | 1.320 | 2.261 | 3.058 | 108.090 | 17.523 | CPU benchmark was faster overall for this short test. |
| `Qwen2.5-0.5B-Instruct-q8.litertlm` | [litert-community/Qwen2.5-0.5B-Instruct](https://huggingface.co/litert-community/Qwen2.5-0.5B-Instruct) | GPU | FAIL | Native OpenCL attempted | 520.73 MB | 458.68 MB | N/A | N/A | N/A | N/A | N/A | N/A | Engine creation failed after partial GPU delegation; the model card publishes Android CPU results, not GPU results. |
| `Qwen2.5-0.5B-Instruct-q8.litertlm` | same | CPU | PASS | N/A | 520.73 MB | 384.80 MB | 1.305 | 0.698 | 0.703 | 2.171 | 206.757 | 25.742 | Recommended fast CPU alternative. |
| `Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm` | [litert-community/Qwen2.5-1.5B-Instruct](https://huggingface.co/litert-community/Qwen2.5-1.5B-Instruct) | GPU | PASS | Native OpenCL + CPU sampler fallback | 1523.91 MB | 365.20 MB | 9.412 | 1.969 | 2.047 | 9.904 | 88.850 | 9.946 | Larger Qwen option; useful for compatibility comparison. |
| `Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm` | same | CPU | PASS | N/A | 1523.91 MB | 347.30 MB | 4.290 | 3.845 | 3.584 | 6.597 | 30.735 | 8.537 | CPU passed, but latency is high. |
| `Qwen3-0.6B.litertlm` | [litert-community/Qwen3-0.6B](https://huggingface.co/litert-community/Qwen3-0.6B) | GPU | PASS | Native OpenCL + CPU sampler fallback | 585.78 MB | 395.54 MB | 5.721 | 3.342 | 3.340 | 6.983 | 96.503 | 9.104 | Runtime passed; prompt quality still needs model-specific tuning. |
| `Qwen3-0.6B.litertlm` | same | CPU | PASS | N/A | 585.78 MB | 359.39 MB | 1.741 | 6.455 | 6.182 | 7.721 | 30.141 | 6.143 | CPU passed, but chat turns were slow. |

### GPU Notes

Native OpenCL model execution is working for most GPU runs. The current AAR
still reports sampler fallback:

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
