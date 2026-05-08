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
   verified option on the tested device, but its GPU graph fails engine
   creation on this chipset.
3. `gemma3-1b-it-int4.litertlm` - smaller GPU-capable fallback when model size
   matters more than output quality.

| Model | Recommended use | Latest result | Links |
| --- | --- | --- | --- |
| `gemma-4-E2B-it.litertlm` | Primary Android model | GPU PASS, native OpenCL execution and OpenCL sampler. Chat turns: 1.561s, 0.582s. | [LiteRT-LM](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) |
| `Qwen2.5-0.5B-Instruct-q8.litertlm` | Fast CPU fallback | CPU PASS. Chat turns: 0.777s, 0.630s. GPU failed engine creation on the tested SM8250 device. | [LiteRT-LM](https://huggingface.co/litert-community/Qwen2.5-0.5B-Instruct) |
| `gemma3-1b-it-int4.litertlm` | Compact GPU fallback | GPU PASS, native OpenCL execution and OpenCL sampler. File size is 557.34 MB. | [LiteRT-LM](https://huggingface.co/litert-community/Gemma3-1B-IT) |

## Android Physical Device Benchmarks

Results were collected on 2026-05-08 with package
`com.Leuconoe.LiteRTLMUnity`. Public device details are limited to chipset and
memory: Qualcomm QTI `SM8250` (`kona`), 7.52 GiB RAM.

The benchmark wrapper builds one APK per model/backend, installs it on the
physical device, launches the Unity smoke runner, waits for two real chat turns,
then runs the native benchmark loop three times. GPU was always tested before
CPU. Before each run, the wrapper sampled device thermal zones and waited until
the hottest readable sensor was at or below 43 C. Gemma 4 rows were built with
speculative decoding enabled for LiteRT-LM multi-token prediction.

| Model file | Backend | Result | GPU evidence | File MB | PSS MB | Init s | Chat1 s | Chat2 s | Bench avg s | TTFT s | Prefill tok/s | Decode tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gemma-4-E2B-it.litertlm` | GPU | PASS | NativeOpenCL+OpenCLSampler | 2468.25 | 459.42 | 12.626 | 1.561 | 0.582 | 16.415 | 0.423 | 396.12 | 9.98 |
| `gemma-4-E2B-it.litertlm` | CPU | PASS | N/A | 2468.25 | 404.26 | 4.993 | 1.811 | 1.373 | 9.434 | 1.013 | 155.26 | 5.29 |
| `gemma-4-E4B-it.litertlm` | GPU | FAIL | NativeOpenCL+OpenCLSampler | 3490.00 | 437.05 | 19.688 | 2.610 | 1.583 | N/A | N/A | N/A | N/A |
| `gemma-4-E4B-it.litertlm` | CPU | PASS | N/A | 3490.00 | 459.19 | 14.137 | 5.565 | 2.946 | 41.131 | 9.832 | 14.30 | 1.13 |
| `gemma3-1b-it-int4.litertlm` | GPU | PASS | NativeOpenCL+OpenCLSampler | 557.34 | 493.21 | 7.849 | 1.481 | 1.662 | 6.864 | 0.373 | 197.53 | 20.40 |
| `gemma3-1b-it-int4.litertlm` | CPU | PASS | N/A | 557.34 | 416.67 | 3.430 | 1.219 | 1.976 | 2.891 | 0.605 | 116.08 | 18.50 |
| `Phi-4-mini-instruct_multi-prefill-seq_q8_ekv4096.litertlm` | GPU | FAIL | NativeOpenCL | 3728.95 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `Phi-4-mini-instruct_multi-prefill-seq_q8_ekv4096.litertlm` | CPU | PASS | N/A | 3728.95 | 370.30 | 19.811 | 97.691 | 98.737 | 69.891 | 34.921 | 1.89 | 0.98 |
| `Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm` | GPU | PASS | NativeOpenCL+OpenCLSampler | 1523.91 | 448.08 | 9.184 | 2.536 | 1.902 | 14.864 | 0.822 | 87.88 | 10.68 |
| `Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm` | CPU | PASS | N/A | 1523.91 | 411.38 | 4.077 | 4.145 | 3.322 | 7.940 | 2.258 | 29.90 | 8.51 |
| `DeepSeek-R1-Distill-Qwen-1.5B_multi-prefill-seq_q8_ekv4096.litertlm` | GPU | PASS | NativeOpenCL+OpenCLSampler | 1748.52 | 456.66 | 10.146 | 4.347 | 3.695 | 19.365 | 0.831 | 87.45 | 10.10 |
| `DeepSeek-R1-Distill-Qwen-1.5B_multi-prefill-seq_q8_ekv4096.litertlm` | CPU | PASS | N/A | 1748.52 | 419.12 | 4.575 | 5.137 | 4.671 | 12.219 | 2.386 | 28.18 | 8.74 |
| `SmolLM-135M-Instruct_multi-prefill-seq_q8_ekv1280.task` | GPU | FAIL | RequestedGPU | 159.03 | 372.46 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `SmolLM-135M-Instruct_multi-prefill-seq_q8_ekv1280.task` | CPU | FAIL | N/A | 159.03 | 372.66 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `TinyLlama-1.1B-Chat-v1.0_multi-prefill-seq_q8_ekv1280.task` | GPU | FAIL | RequestedGPU | 1095.13 | 362.11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `TinyLlama-1.1B-Chat-v1.0_multi-prefill-seq_q8_ekv1280.task` | CPU | FAIL | N/A | 1095.13 | 361.86 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `Qwen2.5-0.5B-Instruct-q8.litertlm` | GPU | FAIL | NativeOpenCL | 520.73 | 473.51 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `Qwen2.5-0.5B-Instruct-q8.litertlm` | CPU | PASS | N/A | 520.73 | 391.74 | 1.245 | 0.777 | 0.630 | 2.125 | 0.333 | 217.03 | 26.55 |

Failure and coverage notes:

| Model | Status | Detail |
| --- | --- | --- |
| `gemma-4-E4B-it.litertlm` GPU | Partial smoke, benchmark fail | The two chat turns completed, but the benchmark engine failed allocating 423,395,520 bytes of OpenCL device memory with `clCreateBuffer: Out of resources`. |
| `Phi-4-mini-instruct_multi-prefill-seq_q8_ekv4096.litertlm` GPU | Fail | Native OpenCL started, then the device hit memory pressure and thermal rose to 57.6 C. CPU works but is too slow for the sample app. |
| `Qwen2.5-0.5B-Instruct-q8.litertlm` GPU | Fail | OpenCL initialized, then LiteRT-LM failed compiled model executor creation. CPU is the recommended mode for this file on SM8250. |
| `SmolLM-135M-Instruct` and `TinyLlama-1.1B-Chat-v1.0` `.task` files | Fail | Current Unity bridge expects LiteRT-LM metadata and reports `INVALID_ARGUMENT: Failed to parse LlmMetadata`. |
| `gemma3-270m-it-q8.litertlm` | Not benchmarked | Requested file was not present locally, and the Hugging Face manifest scan returned gated/unauthorized access. |
| `gemma-3n-E2B-it-int4.litertlm` / `gemma-3n-E4B-it-int4.litertlm` | Not benchmarked | Requested files were not present locally, and the Google Gemma 3n LiteRT-LM repos were gated for this run. |
| `SmolVLM-256M-Instruct` | Not benchmarked | No LiteRT-LM-compatible `.litertlm` or `.task` package was available from the scanned repository listing. |
| `Gemma2-2B-IT_multi-prefill-seq_q8_ekv1280.task` | Not benchmarked | Requested file was gated or missing locally during APK generation. |

## Custom LiteRT-LM Android Bridge Build

Unity can use the committed `Assets/Plugins/Android/litertlm-unity-bridge.aar`
without modifying a LiteRT-LM checkout. To rebuild that AAR from source, keep
LiteRT-LM as a Unity-local submodule and apply the Unity AAR patch at build
time.

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

Add or refresh the submodule from the Unity project root:

```powershell
git submodule add https://github.com/Leuconoe/LiteRT-LM External/LiteRT-LM
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
