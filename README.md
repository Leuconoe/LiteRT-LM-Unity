# LiteRT-LM-Unity

Unity integration sample for running LiteRT-LM from a Unity project.

This project includes a Unity Editor sample flow, a Windows CLI fallback path,
and batchmode self-tests that verify the Editor integration without relying on
manual UI steps.

## Requirements

- Unity `6000.4.6f1`
- Windows for the included Editor CLI fallback scripts
- PowerShell
- Docker Desktop and Git for Windows Bash for rebuilding the Android bridge AAR

## Included

- `Assets/Scenes/LiteRtLmSampleScene.unity`
  - Manual Editor sample scene.
- `Assets/Scenes/LiteRtLmConversationTestScene.unity`
  - Automated 10-turn conversation test scene.
- `Assets/Scripts/LiteRTLM/LiteRtLmSampleController.cs`
  - IMGUI sample UI with IME-aware prompt input.
- `Assets/Scripts/LiteRTLM/LiteRtLmWindowsCliClient.cs`
  - Windows Editor CLI fallback client.
- `Tools/Windows/Run-LiteRtLmSample.ps1`
  - Stable wrapper around `litert_lm_main.windows_x86_64.exe`.
- `Tools/Windows/Run-LiteRtLmEditorSelfTest.ps1`
  - Unity batchmode self-test runner.
- `Tools/Windows/Build-LiteRtLmAndroidAsrSmokeApk.ps1`
  - Builds the Android Parakeet ASR smoke-test APK.
- `Tools/Windows/Run-LiteRtLmAndroidAsrSmokeTest.ps1`
  - Installs and runs the Parakeet ASR smoke test on an Android device.

## Model Files

`Assets/StreamingAssets/model.litertlm` is committed as the small test model so
the Editor self-test can run after checkout.

Other `Assets/StreamingAssets` files are local artifacts and are ignored by
default. This includes downloaded models, generated `.xnnpack_cache` files, and
runtime cache metadata.

For manual testing with a larger model, place it under `Assets/StreamingAssets`
locally and select it in the sample scene. Large model files are intentionally
not committed.

## Recommended Models

The current function-calling benchmark uses 20 Unity command prompts covering
display controls, volume controls, visualization commands, date-range queries,
and fallback/default responses. The Android table below measures runtime smoke
and throughput, not the full function-calling accuracy suite.

Current Android recommendation:

1. `gemma-4-E2B-it.litertlm` - default model. It is the best current quality
   choice and passes native OpenCL GPU execution plus OpenCL Top-K sampling.
2. `Qwen2.5-0.5B-Instruct-q8.litertlm` - fast CPU fallback. It is the fastest
   practical alternative in this Unity build, but its GPU graph fails engine
   creation on the tested SM8250 device.
3. `gemma3-1b-it-int4.litertlm` - compact GPU-capable fallback when model size
   matters more than Gemma 4 output quality.

| Model | Recommended use | Latest result | Links |
| --- | --- | --- | --- |
| `gemma-4-E2B-it.litertlm` | Primary Android model | GPU PASS, native OpenCL execution and OpenCL sampler. Chat turns: 1.689s, 0.639s. | [LiteRT-LM](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) |
| `Qwen2.5-0.5B-Instruct-q8.litertlm` | Fast CPU fallback | CPU PASS. Chat turns: 0.688s, 0.616s. GPU failed engine creation on the tested SM8250 device. | [LiteRT-LM](https://huggingface.co/litert-community/Qwen2.5-0.5B-Instruct) |
| `gemma3-1b-it-int4.litertlm` | Compact GPU fallback | GPU PASS, native OpenCL execution and OpenCL sampler. File size is 557.34 MB. | [LiteRT-LM](https://huggingface.co/litert-community/Gemma3-1B-IT) |

## Android Physical Device Benchmarks

Results were collected on 2026-05-11 with package
`com.Leuconoe.LiteRTLMUnity`. Public device details are limited to chipset and
memory: Qualcomm QTI `SM8250` (`kona`), 7.52 GiB RAM.

The benchmark wrapper builds one APK per model/backend, installs it on the
physical device, launches the Unity smoke runner, waits for two real chat turns,
then runs the native benchmark loop three times. GPU was always tested before
CPU. Before each run, the wrapper sampled device thermal zones and waited until
the hottest readable sensor was at or below 45 C. Gemma 4 rows were built with
speculative decoding enabled for LiteRT-LM multi-token prediction. The Android
smoke runner records throughput and memory; function-calling hit rate is not
scored in this APK path and remains covered by the Editor function-calling
benchmark.

Coverage status:

- All configured benchmark cases completed an attempted run: 28 cases covering
  14 model files across GPU and CPU backends.
- 17 cases passed and 11 cases failed with captured status/log evidence.
- Failures in the table are confirmed runtime failures, not skipped or missing
  runs.
- `Assets/StreamingAssets/model.litertlm` is the small Editor smoke-test model
  and is intentionally excluded from the Android model comparison.
- `Gemma3-1B-IT_multi-prefill-seq_q4_ekv4096.litertlm` is a local duplicate of
  the measured `gemma3-1b-it-int4.litertlm` payload size, so the benchmark table
  reports the `gemma3-1b-it-int4.litertlm` row only.

| Model file | Backend | Result | GPU evidence | File MB | PSS MB | Init s | Chat1 s | Chat2 s | Bench avg s | TTFT s | Prefill tok/s | Decode tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gemma-4-E2B-it.litertlm` | GPU | PASS | NativeOpenCL+OpenCLSampler | 2468.25 | 460.68 | 14.083 | 1.689 | 0.639 | 19.125 | 0.563 | 278.088 | 9.728 |
| `gemma-4-E2B-it.litertlm` | CPU | PASS | N/A | 2468.25 | 460.05 | 5.733 | 1.876 | 1.463 | 10.166 | 1.081 | 145.379 | 4.996 |
| `gemma-4-E4B-it.litertlm` | GPU | FAIL | NativeOpenCL+OpenCLSampler | 3490.00 | 485.97 | 22.331 | 2.652 | 1.590 | N/A | N/A | N/A | N/A |
| `gemma-4-E4B-it.litertlm` | CPU | PASS | N/A | 3490.00 | 461.45 | 11.223 | 4.382 | 3.051 | 33.547 | 2.444 | 60.883 | 2.927 |
| `gemma3-270m-it-q8.litertlm` | GPU | PASS | NativeOpenCL+OpenCLSampler | 289.92 | 434.76 | 6.573 | 1.665 | 1.018 | 3.927 | 0.194 | 394.726 | 31.033 |
| `gemma3-270m-it-q8.litertlm` | CPU | PASS | N/A | 289.92 | 368.57 | 0.867 | 0.895 | 0.887 | 1.936 | 0.587 | 115.520 | 30.213 |
| `gemma-3n-E2B-it-int4.litertlm` | GPU | PASS | NativeOpenCL+OpenCLSampler | 3486.47 | 407.60 | 14.029 | 3.470 | 2.678 | 20.788 | 1.438 | 98.043 | 7.561 |
| `gemma-3n-E2B-it-int4.litertlm` | CPU | PASS | N/A | 3486.47 | 462.34 | 6.354 | 6.190 | 5.607 | 28.362 | 4.134 | 32.158 | 6.491 |
| `gemma-3n-E4B-it-int4.litertlm` | GPU | PASS | NativeOpenCL+OpenCLSampler | 4691.64 | 406.71 | 20.546 | 4.682 | 3.883 | 33.185 | 2.144 | 65.352 | 5.403 |
| `gemma-3n-E4B-it-int4.litertlm` | CPU | PASS | N/A | 4691.64 | 409.18 | 11.476 | 9.485 | 7.821 | 55.446 | 30.819 | 4.247 | 1.469 |
| `gemma3-1b-it-int4.litertlm` | GPU | PASS | NativeOpenCL+OpenCLSampler | 557.34 | 492.49 | 7.963 | 1.479 | 1.672 | 7.036 | 0.373 | 198.129 | 19.969 |
| `gemma3-1b-it-int4.litertlm` | CPU | PASS | N/A | 557.34 | 415.00 | 3.422 | 1.240 | 2.033 | 3.013 | 0.650 | 107.298 | 18.758 |
| `Phi-4-mini-instruct_multi-prefill-seq_q8_ekv4096.litertlm` | GPU | FAIL | NativeOpenCL | 3728.95 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `Phi-4-mini-instruct_multi-prefill-seq_q8_ekv4096.litertlm` | CPU | PASS | N/A | 3728.95 | 369.07 | 10.697 | 75.207 | 115.644 | 110.536 | 34.385 | 1.997 | 0.428 |
| `Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm` | GPU | PASS | NativeOpenCL+OpenCLSampler | 1523.91 | 439.52 | 9.216 | 2.500 | 1.878 | 14.848 | 0.827 | 87.360 | 10.544 |
| `Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm` | CPU | PASS | N/A | 1523.91 | 410.32 | 3.737 | 3.554 | 3.746 | 7.174 | 2.244 | 30.080 | 8.567 |
| `DeepSeek-R1-Distill-Qwen-1.5B_multi-prefill-seq_q8_ekv4096.litertlm` | GPU | PASS | NativeOpenCL+OpenCLSampler | 1748.52 | 459.78 | 9.989 | 4.441 | 3.879 | 19.657 | 0.856 | 84.309 | 10.342 |
| `DeepSeek-R1-Distill-Qwen-1.5B_multi-prefill-seq_q8_ekv4096.litertlm` | CPU | PASS | N/A | 1748.52 | 421.81 | 4.074 | 5.578 | 4.803 | 10.152 | 2.206 | 30.678 | 8.334 |
| `SmolLM-135M-Instruct_multi-prefill-seq_q8_ekv1280.task` | GPU | FAIL | RequestedGPU | 159.03 | 358.80 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `SmolLM-135M-Instruct_multi-prefill-seq_q8_ekv1280.task` | CPU | FAIL | N/A | 159.03 | 362.93 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `TinyLlama-1.1B-Chat-v1.0_multi-prefill-seq_q8_ekv1280.task` | GPU | FAIL | RequestedGPU | 1095.13 | 358.52 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `TinyLlama-1.1B-Chat-v1.0_multi-prefill-seq_q8_ekv1280.task` | CPU | FAIL | N/A | 1095.13 | 355.76 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `Gemma2-2B-IT_multi-prefill-seq_q8_ekv1280.task` | GPU | FAIL | RequestedGPU | 2587.58 | 358.27 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `Gemma2-2B-IT_multi-prefill-seq_q8_ekv1280.task` | CPU | FAIL | N/A | 2587.58 | 358.67 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `Qwen2.5-0.5B-Instruct-q8.litertlm` | GPU | FAIL | NativeOpenCL | 520.73 | 477.31 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `Qwen2.5-0.5B-Instruct-q8.litertlm` | CPU | PASS | N/A | 520.73 | 383.37 | 1.255 | 0.688 | 0.616 | 2.108 | 0.260 | 288.899 | 26.181 |
| `Qwen2.5-0.5B-Instruct_multi-prefill-seq_q8_ekv1280.task` | GPU | FAIL | RequestedGPU | 521.34 | 358.78 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `Qwen2.5-0.5B-Instruct_multi-prefill-seq_q8_ekv1280.task` | CPU | FAIL | N/A | 521.34 | 358.90 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

Failure and coverage notes:

| Model | Status | Detail |
| --- | --- | --- |
| `gemma-4-E4B-it.litertlm` GPU | Partial smoke, benchmark fail | The two chat turns completed, but the benchmark engine failed allocating 423,395,520 bytes of OpenCL device memory with `clCreateBuffer: Out of resources`. |
| `Phi-4-mini-instruct_multi-prefill-seq_q8_ekv4096.litertlm` GPU | Fail | Native OpenCL started, then Adreno allocation failed and lowmemorykiller killed the app process. CPU completes, but the verified run took 534.566 seconds end to end, so it is not practical for this sample app. |
| `Qwen2.5-0.5B-Instruct-q8.litertlm` GPU | Fail | OpenCL initialized, then LiteRT-LM failed compiled model executor creation. CPU is the recommended mode for this file on SM8250. |
| `.task` files in this table | Fail | Current Unity bridge expects LiteRT-LM metadata and reports `INVALID_ARGUMENT: Failed to parse LlmMetadata`. The same bundles may require the Google AI Edge Gallery task loader path rather than the LiteRT-LM Unity bridge path. |

## Parakeet ASR

The Android bridge includes an experimental Parakeet ASR smoke path for
`parakeet_tdt_0.6b_v3_5s_i8.tflite`. It loads the TFLite model directly through
LiteRT, runs the `encode` and `decode` signatures, decodes token IDs with
`parakeet-tdt-0.6b-v3/tokenizer.json`, and writes status logs for each stage.

Parakeet ASR assets are local `Assets/StreamingAssets` artifacts and are not
committed by default. To run the smoke test, place these files locally:

- `Assets/StreamingAssets/parakeet_tdt_0.6b_v3_5s_i8.tflite`
- `Assets/StreamingAssets/parakeet-tdt-0.6b-v3/tokenizer.json`
- an English test audio file such as
  `Assets/StreamingAssets/Tactical Evaluation Results Report - March 5, 2025.mp3`

Build and run on a connected Android device:

```powershell
.\Tools\Windows\Build-LiteRtLmAndroidAsrSmokeApk.ps1 `
  -Backend GPU_FP16

.\Tools\Windows\Run-LiteRtLmAndroidAsrSmokeTest.ps1 `
  -DeviceSerial 90e3c875 `
  -ClearAppData
```

Verified result on the tested SM8250 device:

| Backend | Result | GPU evidence | Compile s | Encode s | Decode s | Total s | Tokens | Transcript |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `GPU_FP16` | PASS | LiteRT GPU delegate, fp16 | 4.167 | 0.344 | 6.347 | 11.046 | 18 | `Evaluation Res Report, March 5 2025` |
| `CPU` | PASS | N/A | 1.507 | 0.326 | 6.894 | 8.911 | 18 | `Evaluation Res Report, March 5 2025` |

`GPU_FP16` is the working GPU mode for this Parakeet smoke path. The plain
`GPU`/FP32 variants initialized but produced zero decoded tokens on the tested
device, so the runner treats empty-token ASR output as a failure. For a short
cold run CPU can still finish sooner because GPU compilation costs about four
seconds; GPU decode is slightly faster once compiled.

The tested Parakeet model does not support Korean. This path is useful for
validating Unity-side LiteRT ASR plumbing with English audio now, and can be
reused for Korean ASR when a compatible LiteRT model and tokenizer are available.

## Custom LiteRT-LM Android Bridge Build

Unity can use the committed `Assets/Plugins/Android/litertlm-unity-bridge.aar`
without modifying a LiteRT-LM checkout. This Unity repository is the root
project; LiteRT-LM is kept as a Unity-local submodule under `External/LiteRT-LM`
and the Unity AAR patch is applied at build time.

The intended repository layout is:

```text
LiteRT-LM-Unity/
  Assets/
  Tools/
    UnityAar/
      litert-lm-unity-aar.patch
  External/
    LiteRT-LM/              # git submodule
```

Initialize or refresh the submodule from the Unity project root:

```powershell
git submodule update --init --recursive
git -C External\LiteRT-LM checkout c87189528a758db32ead241f4fc9c64836398ee7
```

The current patch is validated against LiteRT-LM `c87189528a758db32ead241f4fc9c64836398ee7`
(`v0.11.0`). Update the patch when moving the submodule to a newer LiteRT-LM
revision.

Then build the patched AAR:

```powershell
.\Tools\Windows\Build-LiteRtLmUnityAarFromPatch.ps1 `
  -BazelJobs 8
```

The wrapper resolves LiteRT-LM from `External\LiteRT-LM` by default, copies the
source into `.\temp\unity-aar-patched`, applies
`Tools\UnityAar\litert-lm-unity-aar.patch` there, then runs the patched
Docker/Bazel AAR build through Bash. The submodule checkout is left untouched.
The generated AAR is exported to `Builds\AndroidAar` and copied into
`Assets\Plugins\Android\litertlm-unity-bridge.aar`.

During the transition period where this Unity project is still checked out as a
submodule inside a LiteRT-LM source tree, use `-SourceRoot` to point at a clean
LiteRT-LM checkout pinned to the patch revision. Avoid pointing at a newer or
dirty parent worktree unless the patch has already been refreshed for that
revision.

```powershell
.\Tools\Windows\Build-LiteRtLmUnityAarFromPatch.ps1 `
  -SourceRoot ..\LiteRT-LM-v0.11.0 `
  -BazelJobs 8
```

For a quick patch-only check without Docker:

```powershell
.\Tools\Windows\Build-LiteRtLmUnityAarFromPatch.ps1 -PrepareOnly
```

## Run The Editor Self-Test

From the Unity project root:

```powershell
.\Tools\Windows\Run-LiteRtLmEditorSelfTest.ps1 `
  -MaxAttempts 1 `
  -ExecuteMethod 'LiteRTLM.Unity.Editor.LiteRtLmBuild.RunWindowsConversationSceneTestBatchmode' `
  -StatusRelativePath 'Builds\Logs\LiteRtLmConversationTest.status.txt' `
  -TestName 'Unity conversation scene test'
```

The test performs a Unity domain reload, opens the conversation test scene, and
runs 10 prompts covering short prompts, Korean input, mixed-language prompts,
longer diagnostic prompts, and context recall.

Expected result:

- Unity process exits with code `0`.
- `Builds/Logs/LiteRtLmConversationTest.status.txt` ends with `SUCCESS`.
- The final context recall response includes `LRT-CTX-042`.

## Notes

- The Windows Editor path starts the CLI process through PowerShell for process
  and encoding stability.
- Korean prompt input in the sample UI uses IME-aware text fields.
- UTF-8 stdout and stderr handling is enabled for Korean text and emoji output.
