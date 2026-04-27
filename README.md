# LiteRT-LM-Unity

Unity integration sample for running LiteRT-LM from a Unity project.

This project includes a Unity Editor sample flow, a Windows CLI fallback path,
and batchmode self-tests that verify the Editor integration without relying on
manual UI steps.

## Requirements

- Unity `6000.4.4f1`
- Windows for the included Editor CLI fallback scripts
- PowerShell

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
and fallback/default responses.

| Model | Recommended use | Benchmark result | Notes | Links |
| --- | --- | --- | --- | --- |
| `gemma-4-E2B-it.litertlm` | Default recommendation for reliability | 20/20, 100% accuracy, 8.01s average per turn | Best current balance for this Unity function-calling path. Uses the LiteRT `tools_json` constrained path. Higher memory footprint than Qwen. | [Official model](https://huggingface.co/google/gemma-4-e2b-it), [LiteRT-LM conversion](https://huggingface.co/DEEPBULE/gemma-4-E2B-it-litert-lm) |
| `Qwen3-0.6B.litertlm` | Smaller-memory alternative | 20/20, 100% accuracy, 14.18s average per turn | Much smaller model. Requires the Qwen Hermes/ChatML prompt profile and deterministic routing guards; do not use the Gemma `tools_json` prompt path for this model. | [Official model](https://huggingface.co/Qwen/Qwen3-0.6B) |
| `Qwen2.5-1.5B-Instruct-q8.litertlm` | Android GPU candidate for Qwen2.5 | Android AVD smoke passed on GPU backend | Larger than Qwen3-0.6B, but the available q8 LiteRT-LM graph initializes on the current AVD GPU path. | [LiteRT model](https://huggingface.co/litert-community/Qwen2.5-1.5B-Instruct) |

`gemma-4-E2B-it` is the safest default when memory is available. Use
`Qwen3-0.6B` when model size is the primary constraint and the extra prompt
profile/guard logic is acceptable.

## Android AVD Smoke Results

These results were collected on `Medium_Phone_API_36.0` with package
`com.Leuconoe.LiteRTLMUnity`.

The Android smoke runner initializes one Unity client and sends two real chat
turns. Standalone `RunBenchmark` is disabled by default because it creates a
second native LiteRT-LM engine and duplicates the expensive GPU initialization.

| Model | Backend | Result | Init total | TTFT | Prefill | Decode | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `Qwen2.5-1.5B-Instruct-q8.litertlm` | GPU | PASS | 161.031s | 0.905s | 115.52 tokens/sec | 2.85 tokens/sec | OpenCL was unavailable on the AVD, so LiteRT used WebGPU for model execution. Top-K GPU sampler libraries fell back to CPU sampling. |
| `Qwen2.5-0.5B-Instruct-q8.litertlm` | CPU | PASS | N/A | 1.590s | 41.64 tokens/sec | 18.95 tokens/sec | The 0.5B q8 graph failed GPU compiled-model creation on this AVD. |
| `Qwen3-0.6B.litertlm` | GPU | PASS | N/A | 0.381s | 228.59 tokens/sec | 9.92 tokens/sec | Fastest verified Android AVD GPU option so far, but function-calling requires the Qwen prompt profile and routing guards. |

Latest runtime smoke results for `Qwen2.5-1.5B-Instruct-q8.litertlm` on GPU:
model reuse `copied=False`, client initialization `56.411s` to `57.757s`,
turn 1 `2.470s` to `2.508s`, turn 2 `0.457s` to `0.480s`, total smoke
runtime `59.507s` to `60.848s`.

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
