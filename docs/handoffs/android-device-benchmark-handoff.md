# HANDOFF: Android Device Model Benchmarks

## User Goal

Create separate Android builds per LiteRT-LM model and collect benchmark results on the connected physical Android device.

The user explicitly asked:

- Create a separate build per model, then benchmark on the currently connected
  physical device and collect the test results.
- Write it up and produce a handoff.

## Environment

- Repo root: parent `LiteRT-LM` checkout.
- Unity project: `LiteRT-LM-Unity`.
- Unity version: `6000.4.4f1`
- Physical test device:
  - Android `12`
  - ABI includes `arm64-v8a`
  - Exact serial/model intentionally omitted from this handoff.
- Other connected devices may include Android emulators.
- Do not install benchmark APKs to emulators unless explicitly requested.

## Important MCP Warning

Unity MCP was checked and was attached to the wrong Unity instance:

```text
project=DroneARV2
path=<different Unity project>/Assets
unity=2022.3.62f3
```

Do not use Unity MCP build/menu actions until the active instance is verified as `LiteRT-LM-Unity` on Unity `6000.4.4f1`.

## Work Completed

### Smoke Runner Improvement

`Assets/Scripts/LiteRTLM/LiteRtLmAndroidSmokeTestRunner.cs`

- Added default behavior that skips standalone `RunBenchmark`.
- Reason: `RunBenchmark` creates a second native LiteRT-LM engine and duplicates GPU initialization during smoke tests.
- The smoke runner now:
  - resolves/copies the model,
  - initializes one `LiteRtLmUnityClient`,
  - sends two real chat turns,
  - logs `INITIALIZED`, `RESPONSE`, `BENCHMARK_SKIPPED`, and `SUCCESS`.

### Device Benchmark Script

Added:

`Tools/Windows/Run-LiteRtLmAndroidDeviceBenchmarks.ps1`

Key behavior:

- Selects only a physical device by default.
- Refuses emulator serials for physical-device benchmark runs.
- Supports `-DeviceSerial`.
- Supports `-BenchmarkName` filtering.
- Uses `adb install -r -d -t` because Unity Development APKs require `-t` on this device.
- Collects raw logcat and summary logs under:

```text
Builds/Logs/AndroidDeviceRuns
```

- Exports CSV result summaries.
- Was patched to classify low-memory process death as failure, not timeout.

### Separate APK Builds

Separate APKs were built and copied to:

```text
LiteRT-LM-Unity/Builds/Android
```

Current APKs:

| APK | Size | Embedded model |
| --- | ---: | --- |
| `LiteRtLmAndroidSmokeTest-gemma-4-E2B-it.apk` | `4240404009` | `assets/gemma-4-E2B-it.litertlm` |
| `LiteRtLmAndroidSmokeTest-Qwen3-0.6B.apk` | `1657318888` | `assets/Qwen3-0.6B.litertlm` |
| `LiteRtLmAndroidSmokeTest-Qwen2.5-0.5B-Instruct.apk` | `1657318902` | `assets/Qwen2.5-0.5B-Instruct-q8.litertlm` |
| `LiteRtLmAndroidSmokeTest-Qwen2.5-0.5B-Instruct-CPU.apk` | `1657318902` | `assets/Qwen2.5-0.5B-Instruct-q8.litertlm` |
| `LiteRtLmAndroidSmokeTest-Qwen2.5-1.5B-Instruct.apk` | `1657318902` | `assets/Qwen2.5-1.5B-Instruct-q8.litertlm` |

The APK model contents were verified by inspecting ZIP entries.

## Physical Device Results

### Main CSVs

```text
Builds/Logs/AndroidDeviceRuns/20260427-144049-results.csv
Builds/Logs/AndroidDeviceRuns/20260427-144540-results.csv
```

### Summary Table

| Model | Backend | Result | Init | Turn 1 | Turn 2 | Total | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `gemma-4-E2B-it` | GPU | FAIL | N/A | N/A | N/A | N/A | Fails during LiteRT-LM engine creation on GPU/WebGPU path. |
| `Qwen3-0.6B` | GPU | PASS | `2.64s` | `4.289s` | `1.183s` | `8.306s` | Runs, but response quality is invalid for smoke prompt: output was repeated `!`. |
| `Qwen2.5-0.5B-Instruct` | GPU | FAIL | N/A | N/A | N/A | N/A | WebGPU max storage buffer binding size exceeded. |
| `Qwen2.5-0.5B-Instruct` | CPU | PASS | `2.01s` | `0.848s` | `0.279s` | `3.163s` | Best verified path on this physical device. |
| `Qwen2.5-1.5B-Instruct` | GPU | FAIL | N/A | N/A | N/A | N/A | WebGPU binding size exceeded, then Android lowmemorykiller killed the app. |

## Key Evidence

### Qwen3-0.6B GPU PASS But Bad Output

Log:

```text
Builds/Logs/AndroidDeviceRuns/20260427-144049-qwen3-0.6b-gpu.summary.txt
```

Important lines:

```text
[LiteRT-LM AndroidSmoke] MODEL_READY: ... Qwen3-0.6B.litertlm, bytes=614236160, copied=True
[LiteRT-LM AndroidSmoke] INITIALIZED: isInitialized=True, elapsedSeconds=2.64
[LiteRT-LM AndroidSmoke] RESPONSE: 1/2: elapsedSeconds=4.289, length=26, preview=!!!!!!!!!!!!!!!!!!!!!!!!!!
[LiteRT-LM AndroidSmoke] RESPONSE: 2/2: elapsedSeconds=1.183, length=1, preview=!
[LiteRT-LM AndroidSmoke] SUCCESS: backend=GPU, turns=2, totalElapsedSeconds=8.306
```

Interpretation: runtime smoke passes mechanically, but generated output is not semantically usable.

### Qwen2.5-0.5B CPU PASS

Log:

```text
Builds/Logs/AndroidDeviceRuns/20260427-144540-qwen2.5-0.5b-cpu.summary.txt
```

Important lines:

```text
[LiteRT-LM AndroidSmoke] MODEL_READY: ... Qwen2.5-0.5B-Instruct-q8.litertlm, bytes=546029568, copied=False
[LiteRT-LM AndroidSmoke] INITIALIZED: isInitialized=True, elapsedSeconds=2.01
[LiteRT-LM AndroidSmoke] RESPONSE: 1/2: elapsedSeconds=0.848, length=38, preview=Android LiteRT-LM smoke test turn one.
[LiteRT-LM AndroidSmoke] RESPONSE: 2/2: elapsedSeconds=0.279, length=7, preview=Android
[LiteRT-LM AndroidSmoke] SUCCESS: backend=CPU, turns=2, totalElapsedSeconds=3.163
```

Interpretation: this is the only currently verified stable and semantically reasonable path on the physical device.

### Qwen2.5-0.5B GPU FAIL

Log:

```text
Builds/Logs/AndroidDeviceRuns/20260427-144540-qwen2.5-0.5b-gpu.summary.txt
```

Important lines:

```text
OpenCL not supported on this platform. Using WebGPU instead.
Validation error: Binding size (136134656) of [Buffer (unlabeled)] is larger than the maximum storage buffer binding size (134217728).
[LiteRT-LM AndroidSmoke] FAILURE: UnityEngine.AndroidJavaException: com.google.ai.edge.litertlm.LiteRtLmJniException: Failed to create engine
```

Interpretation: this device's WebGPU path cannot create the engine for this graph because one buffer binding exceeds the device limit.

### Qwen2.5-1.5B GPU FAIL

Manual confirmation log:

```text
Builds/Logs/AndroidDeviceRuns/20260427-145901-qwen2.5-1.5b-gpu-manual.summary.txt
```

Important lines:

```text
[LiteRT-LM AndroidSmoke] MODEL_READY: ... Qwen2.5-1.5B-Instruct-q8.litertlm, bytes=1597931520, copied=False
OpenCL not supported on this platform. Using WebGPU instead.
Validation error: Binding size (233373696) of [Buffer (unlabeled)] is larger than the maximum storage buffer binding size (134217728).
lowmemorykiller: Kill 'com.Leuconoe.LiteRTLMUnity' ... to free 386868kB rss
ActivityManager: Process com.Leuconoe.LiteRTLMUnity ... has died
```

Interpretation: this model is not viable on the current physical device GPU/WebGPU path and also causes memory pressure.

### Gemma4 GPU FAIL

Log:

```text
Builds/Logs/AndroidDeviceRuns/20260427-144049-gemma-4-E2B-it-gpu.summary.txt
```

Important lines:

```text
[LiteRT-LM AndroidSmoke] MODEL_READY: ... gemma-4-E2B-it.litertlm, bytes=2583085056, copied=True
OpenCL not supported on this platform. Using WebGPU instead.
[LiteRT-LM AndroidSmoke] FAILURE: UnityEngine.AndroidJavaException: com.google.ai.edge.litertlm.LiteRtLmJniException: Failed to create engine
```

Interpretation: Gemma4 is too large/heavy for the current device GPU/WebGPU path.

## Current Technical Conclusion

On the tested physical Android device:

- OpenCL is not available to LiteRT-LM.
- LiteRT falls back to WebGPU.
- WebGPU max storage buffer binding size is `134217728` bytes.
- Qwen2.5 0.5B GPU requires at least `136134656` bytes for one binding, so it fails.
- Qwen2.5 1.5B GPU requires at least `233373696` bytes for one binding, then gets killed by lowmemorykiller.
- Gemma4 GPU also fails engine creation.
- Qwen3 0.6B GPU starts, but generated text is unusable for the smoke prompt.
- Qwen2.5 0.5B CPU is the best verified path right now.

## Files Changed In This Work Segment

Likely relevant changes:

```text
README.md
Assets/Scripts/LiteRTLM/LiteRtLmAndroidSmokeTestRunner.cs
Tools/Windows/Run-LiteRtLmAndroidDeviceBenchmarks.ps1
```

Note: the Unity project working tree has many unrelated changes. Do not assume `git status` is clean.

## Commands To Continue

Verify physical device:

```powershell
adb devices -l
```

Run all physical-device benchmarks:

```powershell
.\Tools\Windows\Run-LiteRtLmAndroidDeviceBenchmarks.ps1
```

Run only selected benchmarks:

```powershell
.\Tools\Windows\Run-LiteRtLmAndroidDeviceBenchmarks.ps1 `
  -BenchmarkName 'qwen2.5-0.5b-cpu','qwen3-0.6b-gpu'
```

## Recommended Next Steps

1. Treat `Qwen2.5-0.5B-Instruct` CPU as the current physical-device baseline.
2. Do not pursue Qwen2.5 GPU on this device unless the graph can be converted/split to stay under the WebGPU storage buffer binding limit.
3. Investigate why Qwen3 GPU outputs repeated punctuation before considering it for function calling.
4. If GPU is mandatory, next work should focus on graph conversion constraints, smaller per-buffer partitioning, or a device/runtime path with OpenCL support.
5. Update README with the physical-device table after the user confirms the conclusion.
