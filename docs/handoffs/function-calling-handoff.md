# Function Calling Handoff

> **Update 2026-07-23**: the custom CLI flags described below do **not** live
> in a committed `runtime/engine/litert_lm_main.cc` — they are maintained in
> `Tools/UnityAar/litert-lm-unity-aar.patch`, which is applied to a pristine
> LiteRT-LM checkout (`External/LiteRT-LM`, branch `unity-v0.14.0`) when
> building the Windows exe or the Android AAR. The framework has since been
> upgraded to v0.14.0 and the patch regenerated; see
> [`v0.14-upgrade-handoff.md`](v0.14-upgrade-handoff.md) for the current
> state, and `docs/benchmarks/fc-model-benchmark.md` for the shipped
> function-calling benchmark results.

## Goal

Implement and stabilize LiteRT-LM Unity function calling by exposing grammar-style constrained decoding in the Windows CLI, wiring it through the Unity Windows client, and validating it with a 20-case Unity benchmark scene.

## User Instructions

- Continue the prior work using a PDCA loop.
- Do not stop at a proposal. Build, test, inspect failures, improve, and repeat until the benchmark is working or a hard blocker is proven.
- Core target: function calling support through grammar/constrained decoding plus prompt tuning.
- Benchmark scene must use 20 varied chat cases:
  - short and long utterances
  - Korean and English
  - date range extraction
  - previous-context style checks
  - irrelevant/default-response cases
- Emit debug logs/status so test state is visible.
- Before Unity tests, account for domain reload/editor compilation.
- Preserve the existing Windows exe as a baseline before replacing it.

## Current Proven State

- Unity project compiles with the new benchmark runner and scene.
- The benchmark scene opens in Unity batchmode.
- The current checked baseline exe does not expose constrained/function-calling flags.
- Baseline exe was committed as:
  - `e321987 chore: add baseline Windows LiteRT LM executable`
- Local safety backup exists but must not be committed:
  - `Tools/Windows/litert_lm_main.windows_x86_64.exe.backup-*`

## Implemented Work In Progress

- `Tools/UnityAar/litert-lm-unity-aar.patch` (applied to
  `External/LiteRT-LM` `runtime/engine/litert_lm_main.cc` at build time — not
  committed upstream source)
  - Added CLI flags for system message, tools JSON, messages JSON, constrained decoding, and JSON message output.
- `Tools/Windows/Run-LiteRtLmSample.ps1`
  - Added forwarding for the new CLI flags.
- `Packages/com.leuconoe.litert-lm-unity/Runtime/LiteRtLmWindowsCliClient.cs`
  - Added optional system/tools/messages/constrained/json-output arguments.
- `Packages/com.leuconoe.litert-lm-unity/Samples~/TestScenes/Runtime/LiteRtLmFunctionCallingBenchmarkRunner.cs`
  - Added 20-case benchmark runner.
  - Writes status to `Builds/Logs/LiteRtLmFunctionCallingBenchmark.status.txt`.
  - Fails early if the exe lacks constrained CLI flags.
- `Assets/Scenes/LiteRtLmFunctionCallingBenchmarkScene.unity`
  - Added benchmark scene.
- `Packages/com.leuconoe.litert-lm-unity/Samples~/TestScenes/Editor/LiteRtLmBuild.cs`
  - Added batchmode entrypoint for the function-calling benchmark.

## Next PDCA Loop

### Plan

Build the updated Windows CLI, replace the Unity exe, verify flags, run the Unity benchmark, then tune prompt/grammar behavior from observed failures.

### Do

1. Build `//runtime/engine:litert_lm_main --config=windows`.
2. Copy the built exe into:
   - `LiteRT-LM-Unity/Tools/Windows/litert_lm_main.windows_x86_64.exe`
3. Verify `--helpfull` includes:
   - `--tools_json_file`
   - `--enable_constrained_decoding`
   - `--output_message_json`
   - `--system_message_file`
   - `--messages_json_file`

### Check

Run:

```powershell
& 'LiteRT-LM-Unity\Tools\Windows\Run-LiteRtLmEditorSelfTest.ps1' `
  -MaxAttempts 1 `
  -ExecuteMethod 'LiteRTLM.Unity.Editor.LiteRtLmBuild.RunWindowsFunctionCallingBenchmarkBatchmode' `
  -StatusRelativePath 'Builds\Logs\LiteRtLmFunctionCallingBenchmark.status.txt' `
  -TestName 'Unity function-calling benchmark'
```

Inspect:

- `LiteRT-LM-Unity/Builds/Logs/LiteRtLmFunctionCallingBenchmark.status.txt`
- latest `LiteRtLmEditorSelfTest-*.log`

### Act

- If the binary lacks flags, fix the C++ CLI build.
- If tool-call JSON shape differs, fix Unity parser.
- If cases fail semantically, tune system prompt/tool schema/case expectations.
- If performance is poor, shorten prompt and reduce redundant tool descriptions while keeping pass rate.

## Important Constraints

- Do not revert unrelated dirty files in the parent repository.
- Do not commit local backup binaries.
- `StreamingAssets` should stay controlled. Test model `model.litertlm` may be committed if required, cache files must not.
- Prefer a clean final commit after benchmark pass.
