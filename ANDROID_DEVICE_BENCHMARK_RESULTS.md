# Android Physical Device Benchmark Results

This document records the latest verified physical-device benchmark results for
the Unity LiteRT-LM Android build.

## Test Scope

- Date: 2026-04-28
- Unity project: `LiteRT-LM-Unity`
- Package: `com.Leuconoe.LiteRTLMUnity`
- APK: `Builds/Android/LiteRtLmAndroidSmokeTest-gemma3-1b-it-int4.apk`
- Model: `gemma3-1b-it-int4.litertlm`
- Requested backend: `GPU`
- Devices: two Qualcomm Android 12 physical devices
- Command shape:

```powershell
.\Tools\Windows\Run-LiteRtLmAndroidDeviceBenchmarks.ps1 `
  -DeviceSerial <physical-device-serial> `
  -BenchmarkName gemma3-1b-it-gpu `
  -TimeoutSeconds 900 `
  -ClearAppData
```

Device serials and local absolute paths are intentionally omitted from this
report. Use the run IDs and relative log paths below to trace the exact source
artifacts.

## Summary

Both physical-device runs passed with native OpenCL model execution:

- `NativeOpenCL` evidence was present in logcat.
- The Top-K GPU sampler was not usable and fell back to CPU sampling.
- This means model inference used native OpenCL GPU, while token sampling used
  CPU fallback.

| Run ID | Device | Status | GPU evidence | Init | Turn 1 | Turn 2 | Benchmark avg | Benchmark init | TTFT | Prefill | Decode | Total |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20260428-135910` | Physical device A | PASS | `NativeOpenCL+CpuSamplerFallback` | 13.998s | 1.414s | 0.483s | 10.823s | 14361ms | 0.515921s | 146.943 tok/s | 12.4414 tok/s | 49.064s |
| `20260428-141846` | Physical device B | PASS | `NativeOpenCL+CpuSamplerFallback` | 11.518s | 1.059s | 0.372s | 13.436s | 11443.1ms | 0.39985s | 187.426 tok/s | 17.1284 tok/s | 53.686s |

## Source Artifacts

| Run ID | CSV | Summary log | Raw logcat |
| --- | --- | --- | --- |
| `20260428-135910` | `Builds/Logs/AndroidDeviceRuns/20260428-135910-results.csv` | `Builds/Logs/AndroidDeviceRuns/20260428-135910-gemma3-1b-it-gpu.summary.txt` | `Builds/Logs/AndroidDeviceRuns/20260428-135910-gemma3-1b-it-gpu.logcat.txt` |
| `20260428-141846` | `Builds/Logs/AndroidDeviceRuns/20260428-141846-results.csv` | `Builds/Logs/AndroidDeviceRuns/20260428-141846-gemma3-1b-it-gpu.summary.txt` | `Builds/Logs/AndroidDeviceRuns/20260428-141846-gemma3-1b-it-gpu.logcat.txt` |

## GPU Evidence

The logs contain native OpenCL initialization lines:

```text
Loaded OpenCL library with dlopen.
Created OpenCL device from provided device id and platform id.
Initializing OpenCL-based API from serialized data.
```

The logs also show sampler fallback:

```text
OpenCL sampler not available, falling back to other sampler options.
WebGPU sampler not available, falling back to other sampler options.
GPU sampler unavailable. Falling back to CPU sampling.
```

Interpretation:

- GPU model execution is working through native OpenCL.
- The sampler is still CPU fallback because `libLiteRtTopKOpenClSampler.so` and
  `libLiteRtTopKWebGpuSampler.so` could not resolve `LiteRtCreateEnvironment`.
- This is materially better than the earlier WebGPU-only path, where larger
  graphs failed at engine creation or storage-buffer limits.

## Smoke Output

Both runs completed two real chat turns:

| Run ID | Turn 1 preview | Turn 2 preview |
| --- | --- | --- |
| `20260428-135910` | `Android LiteRT-LM smoke test turn one.` | `Android` |
| `20260428-141846` | `Android LiteRT-LM smoke test turn one.` | `Android` |

The second turn is short but non-empty, so it passed the smoke criteria. It is
not a quality benchmark.

## Notes

- Run `20260428-141846` has incorrect device-side logcat timestamps
  (`01-28 ...`) due to the device clock. The host run ID and CSV timestamp are
  the authoritative timestamps.
- These runs were collected before the later Unity change that resets the
  conversation before each prompt. Future runs should show
  `RESET_CONVERSATION` in the Android smoke logs.
- `FunctionCallingHitRate` is empty in these CSVs because this APK path measures
  Android runtime smoke and low-level benchmark speed, not the function-calling
  benchmark accuracy suite.
