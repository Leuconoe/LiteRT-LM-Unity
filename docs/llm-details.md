# LLM Details

This document records LLM setup, benchmark results, and smoke-test notes for the
Unity LiteRT-LM Android builds. It combines all currently available device-run
CSVs and selected summary logs under `Builds/Logs/AndroidDeviceRuns`.

## Setup

- Report date: 2026-04-28
- Unity project: `LiteRT-LM-Unity`
- Package: `com.Leuconoe.LiteRTLMUnity`
- Test type: Android runtime smoke plus low-level benchmark
- Smoke flow: initialize model, run two chat turns, optionally run three
  standalone benchmark iterations
- Device class: Qualcomm Android 12 physical devices

Device serials and local absolute paths are intentionally omitted. Use the
relative run IDs and log paths in this report to trace the source artifacts.

## Benchmarks

### Executive Summary

The current best physical-device result is `gemma3-1b-it-int4` on the native
OpenCL GPU path. It passed on two physical devices with clear OpenCL execution
evidence. The Top-K GPU sampler still falls back to CPU.

The older 2026-04-27 runs used a WebGPU fallback path and show why several
models failed: the device WebGPU max storage buffer binding size was
`134217728` bytes. Qwen2.5 0.5B and 1.5B exceeded that limit on GPU. Qwen2.5
0.5B worked on CPU and is the recommended fast CPU alternative. Qwen3 0.6B and
smaller Gemma/mobile-action models could start, but smoke output quality was
poor for the generic smoke prompt.

### Consolidated Result Table

| Model | APK / model file | Backend | Best status | GPU path | Warmup / init | First chat | Second chat | Benchmark avg | Benchmark init | TTFT | Prefill | Decode | Function hit rate | Quality note |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `gemma3-1b-it-int4` | `LiteRtLmAndroidSmokeTest-gemma3-1b-it-int4.apk` | GPU | PASS | Native OpenCL + CPU sampler fallback | 11.518s to 13.998s | 1.059s to 1.414s | 0.372s to 0.483s | 10.823s to 13.436s | 11443.1ms to 14361ms | 0.39985s to 0.515921s | 146.943 to 187.426 tok/s | 12.4414 to 17.1284 tok/s | Not measured on Android smoke | Best verified native GPU path. |
| `gemma3-270m-it-q8` | `LiteRtLmAndroidSmokeTest-gemma3-270m-it-q8.apk` | GPU | PASS | WebGPU + CPU sampler fallback | 2.417s | 1.850s | 0.175s | 3.071s | 2271.46ms to 2388.69ms | 0.443379s to 0.468635s | 150.6 to 159.944 tok/s | 22.8996 to 24.2037 tok/s | Not measured on Android smoke | Very fast, but generic smoke output was repeated `<pad>`. |
| `mobile_actions_q8_ekv1024` | `LiteRtLmAndroidSmokeTest-mobile_actions_q8_ekv1024.apk` | GPU | PASS | WebGPU + CPU sampler fallback | 5.266s | 1.782s | 0.115s | 2.247s | 1429.53ms to 1519.1ms | 0.372420s to 0.375392s | 189.076 to 190.22 tok/s | 27.0983 to 28.7215 tok/s | Not measured on Android smoke | Built for mobile-actions function calling; generic smoke prompt produced repeated `<pad>`. |
| `Qwen3-0.6B` | `LiteRtLmAndroidSmokeTest-Qwen3-0.6B.apk` | GPU | PASS | GPU requested, older run did not include final native evidence field | 2.640s | 4.289s | 1.183s | N/A | N/A | N/A | N/A | N/A | Not measured on Android smoke | Runtime passed, but output was repeated `!`, so quality was unusable for smoke. |
| `Qwen2.5-0.5B-Instruct-q8` | `LiteRtLmAndroidSmokeTest-Qwen2.5-0.5B-Instruct-CPU.apk` | CPU | PASS | CPU | 2.010s | 0.848s | 0.279s | N/A | N/A | N/A | N/A | N/A | Not measured on Android smoke | Recommended fast CPU alternative when GPU is unavailable or unstable. |
| `Qwen2.5-0.5B-Instruct-q8` | `LiteRtLmAndroidSmokeTest-Qwen2.5-0.5B-Instruct.apk` | GPU | FAIL | WebGPU fallback | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Not measured | Failed engine creation: WebGPU binding `136134656` > limit `134217728`. |
| `Qwen2.5-1.5B-Instruct-q8` | `LiteRtLmAndroidSmokeTest-Qwen2.5-1.5B-Instruct.apk` | GPU | FAIL | WebGPU fallback | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Not measured | Failed WebGPU binding `233373696` > limit `134217728`, then lowmemorykiller killed the app. |
| `gemma-4-E2B-it` | `LiteRtLmAndroidSmokeTest-gemma-4-E2B-it.apk` | GPU | FAIL | WebGPU fallback | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Windows FC: 20/20 | Failed engine creation on the older WebGPU fallback path. |

### Representative Run Artifacts

| Model | Run ID | CSV | Summary log |
| --- | --- | --- | --- |
| `gemma3-1b-it-int4` | `20260428-135910` | `Builds/Logs/AndroidDeviceRuns/20260428-135910-results.csv` | `Builds/Logs/AndroidDeviceRuns/20260428-135910-gemma3-1b-it-gpu.summary.txt` |
| `gemma3-1b-it-int4` | `20260428-141846` | `Builds/Logs/AndroidDeviceRuns/20260428-141846-results.csv` | `Builds/Logs/AndroidDeviceRuns/20260428-141846-gemma3-1b-it-gpu.summary.txt` |
| `gemma3-270m-it-q8` | `20260428-104008` | `Builds/Logs/AndroidDeviceRuns/20260428-104008-results.csv` | `Builds/Logs/AndroidDeviceRuns/20260428-104008-gemma3-270m-it-gpu.summary.txt` |
| `mobile_actions_q8_ekv1024` | `20260428-110048` | `Builds/Logs/AndroidDeviceRuns/20260428-110048-results.csv` | `Builds/Logs/AndroidDeviceRuns/20260428-110048-mobile-actions-gpu.summary.txt` |
| `Qwen3-0.6B` | `20260427-144049` | `Builds/Logs/AndroidDeviceRuns/20260427-144049-results.csv` | `Builds/Logs/AndroidDeviceRuns/20260427-144049-qwen3-0.6b-gpu.summary.txt` |
| `Qwen2.5-0.5B-Instruct-q8` CPU | `20260427-144540` | `Builds/Logs/AndroidDeviceRuns/20260427-144540-results.csv` | `Builds/Logs/AndroidDeviceRuns/20260427-144540-qwen2.5-0.5b-cpu.summary.txt` |
| `Qwen2.5-0.5B-Instruct-q8` GPU | `20260427-144540` | `Builds/Logs/AndroidDeviceRuns/20260427-144540-results.csv` | `Builds/Logs/AndroidDeviceRuns/20260427-144540-qwen2.5-0.5b-gpu.summary.txt` |
| `Qwen2.5-1.5B-Instruct-q8` GPU | `20260427-145901` | N/A, manual confirmation | `Builds/Logs/AndroidDeviceRuns/20260427-145901-qwen2.5-1.5b-gpu-manual.summary.txt` |
| `gemma-4-E2B-it` | `20260427-144049` | `Builds/Logs/AndroidDeviceRuns/20260427-144049-results.csv` | `Builds/Logs/AndroidDeviceRuns/20260427-144049-gemma-4-E2B-it-gpu.summary.txt` |

### GPU Evidence

The final `gemma3-1b-it-int4` runs show native OpenCL execution:

```text
Loaded OpenCL library with dlopen.
Created OpenCL device from provided device id and platform id.
Initializing OpenCL-based API from serialized data.
```

The same runs also show sampler fallback:

```text
OpenCL sampler not available, falling back to other sampler options.
WebGPU sampler not available, falling back to other sampler options.
GPU sampler unavailable. Falling back to CPU sampling.
```

Interpretation:

- Native GPU model execution is working for `gemma3-1b-it-int4`.
- Token sampling still uses CPU fallback.
- This is materially better than the older WebGPU-only fallback path.

### Failure Details

#### WebGPU Storage Buffer Limits

The older GPU runs for Qwen2.5 failed because one WebGPU binding exceeded the
device limit:

```text
Qwen2.5-0.5B: Binding size (136134656) ... larger than ... (134217728)
Qwen2.5-1.5B: Binding size (233373696) ... larger than ... (134217728)
```

The Qwen2.5 1.5B run also triggered Android memory pressure:

```text
lowmemorykiller: Kill 'com.Leuconoe.LiteRTLMUnity' ... to free 386868kB rss
```

#### Gemma4 Android GPU

`gemma-4-E2B-it` failed during engine creation on the older WebGPU fallback
path. It remains a strong Windows function-calling model, but Android GPU has
not been verified for it in the current native OpenCL build path.

### Function-Calling Accuracy

The Android device smoke benchmark does not execute the 20-case function-calling
accuracy suite, so Android `FunctionCallingHitRate` is not available in these
CSV files.

Available non-Android function-calling evidence:

| Model | Environment | Prompt profile | Result |
| --- | --- | --- | --- |
| `gemma-4-E2B-it` | Windows CLI / Unity Editor benchmark | `CurrentTuned` | 20/20, 100% accuracy |

The Android function-calling hit rate still needs a dedicated on-device runner
that uses the same 20 cases as `LiteRtLmFunctionCallingBenchmarkRunner`.

### Recommendations

1. Use `gemma3-1b-it-int4` as the current Android native GPU baseline.
2. Keep `Qwen2.5-0.5B-Instruct-q8` as the fast CPU alternative when GPU is
   unavailable, unstable, or not worth the initialization risk.
3. Do not treat `gemma3-270m-it-q8`, `mobile_actions_q8_ekv1024`, or
   `Qwen3-0.6B` generic smoke PASS as quality PASS until function-specific
   prompts are benchmarked on device.
4. Re-test `gemma-4-E2B-it` on the current native OpenCL AAR before making a
   final Android recommendation for it.
5. Add an Android function-calling benchmark runner to fill the missing
   `FunctionCallingHitRate` column for all candidate models.

## Smoke Tests

The Android LLM smoke flow initializes the selected model, runs two chat turns,
and can run three standalone benchmark iterations. Use
`Tools/Windows/LiteRtLmAndroidBenchmarks.ps1` to inspect available model
definitions and `Tools/Windows/Run-LiteRtLmAndroidDeviceBenchmarks.ps1` to run
selected device benchmarks.

## Notes

- Some summary logs include unrelated device/system noise because they are
  filtered from raw logcat. This report only records LiteRT-LM relevant lines.
- Run `20260428-141846` has incorrect device-side logcat timestamps due to the
  device clock. The host run ID and CSV timestamp are authoritative.
- Future runs should include `RESET_CONVERSATION` in the smoke logs because the
  Unity client now resets conversation state before each prompt.
