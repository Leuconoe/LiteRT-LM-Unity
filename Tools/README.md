# Tools

Split by **how often you should need it**, because the two kinds were mixed and
it was not obvious which scripts were part of the normal workflow and which
existed to answer one question in one session.

| Directory | What lives there |
| --- | --- |
| `Windows/` | The standing workflow: build the AAR, restore samples, compile-check the runtime. Nothing else at this level. |
| `Windows/Bin/` | Prebuilt Windows runtime — the CLI executables and their DLLs. Binaries only, no scripts. |
| `Windows/Tests/` | Everything whose purpose is to test: device gates, smoke runners, benchmarks, and the APK builders that exist to feed them. |
| `Research/` | One-off investigations: model conversion, quantization, candidate evaluation. Kept because the numbers in `docs/` are only trustworthy if the driver that produced them still exists. |
| `UnityAar/` | The native patch applied to `External/LiteRT-LM` when building the AAR. |

Three questions decide where a new script goes. Would you run it as part of
building the product? `Windows/`. Does it exist to test or measure something
that already works? `Windows/Tests/`. Does it exist to answer an open question
and produce a number for `docs/`? `Research/`.

## Windows — the standing workflow

| Script | What it does |
| --- | --- |
| `Build-LiteRtLmUnityAarFromPatch.ps1` | Builds `litertlm-unity-bridge.aar` in Docker from `External/LiteRT-LM` + `UnityAar/litert-lm-unity-aar.patch` |
| `Restore-LiteRtLmSamples.ps1` | Imports `Samples~` into `Assets/Samples/…`. **Required after editing anything under `Samples~`** — Unity does not compile that folder |
| `Sync-LiteRtLmGeneratedScenes.ps1` | Copies generated `.unity` scenes back into `Samples~`, so the next restore does not delete them |
| `Invoke-LiteRtLmRuntimeCompileCheck.ps1` | Roslyn-compiles `Runtime/` for the editor and Android define sets in seconds, without launching Unity |
| `Run-LiteRtLmSample.ps1` | Runs a sample scene against `Bin/litert_lm_main` |

## Windows/Bin — the prebuilt runtime

`litert_lm_main.windows_x86_64.exe`, `litert_lm_advanced_main.windows_x86_64.exe`
and the DLLs they load (`libLiteRt`, the WebGPU accelerator and sampler, Dawn,
DirectX shader compilers, the Gemma constraint provider).

The executables carry custom flags from `UnityAar/litert-lm-unity-aar.patch` and
their source is **not committed** — read `CLAUDE.md` before attempting a rebuild,
particularly the part about pinning VS2022 through `BAZEL_VC`.

Sample scenes reference `Tools/Windows/Bin/litert_lm_main.windows_x86_64.exe` as
a **serialized string**, so moving these files means editing scene YAML as well
as code.

## Windows/Tests — device gates and smoke runners

| Script | What it does |
| --- | --- |
| `Build-LiteRtLmAndroid*Apk.ps1` | APK builders for the gates (smoke, ASR smoke, FC demo) |
| `Run-LiteRtLmAndroidAsrSmokeTest.ps1`, `Run-LiteRtLmAndroidTtsSmokeTest.ps1` | On-device ASR and TTS gates |
| `Run-LiteRtLmAndroidAsrFunctionCallingDemo.ps1` | Voice → routing demo on the device |
| `Run-LiteRtLmAndroidDeviceBenchmarks.ps1`, `LiteRtLmAndroidBenchmarks.ps1` | Device throughput benchmarks and the shared helper they dot-source |
| `Run-LiteRtLmWindowsAsrSmokeTest.ps1` | Desktop ASR smoke through the CLI |
| `Run-LiteRtLmEditorSelfTest.ps1` | Editor-side self test |

Device runs need `-DeviceSerial 46a880a0` — an emulator is usually attached too.

## Research — how each number in `docs/` was produced

| Directory | Question it answered | Written up in |
| --- | --- | --- |
| `Supertonic/` | Can Supertonic TTS run on LiteRT, and how fast? Conversion, quantization, bucketing, desktop/device comparison | `docs/tts-details.md` |
| `Whisper/` | Can the Whisper tflite exports run on Windows? | `docs/benchmarks/whisper-windows-tflite.md` |
| `Kanana/` | How does a candidate LLM score on our 20-case Korean routing set when LiteRT cannot run it yet? | `docs/llm-details.md` |
| `Moonshine/` | Are the Moonshine Korean ASR models more accurate than the Whisper tiers we ship, before any conversion is spent? | `docs/benchmarks/moonshine-korean-eval.md` |
| `FunctionGemma/` | Can a fine-tuned FunctionGemma-270M route the WebServer tool contract on the headset? Data generator, training, eval sets, conversion, voice E2E. `Rebuild-FunctionGemmaRouter.ps1` retrains after a ToolsPrompt change (`-DryRun` / `-Smoke` / `-Stage`) | `docs/benchmarks/functiongemma-korean-fc.md` §9 |
| `SenseVoice/` | Is SenseVoice-Small a better on-device Korean ASR, and can it run on LiteRT? Eval, conversion, Zeroth fine-tune | `docs/benchmarks/sensevoice-korean-eval.md` |
| `Jev/` | Can a Jev-style decision model run offline (frozen gemma-3-270m trunk + jevlike head)? | `Tools/Research/Jev/README.md` (in progress) |
| `Rag/` | Which embedder and Korean corpus for the RAG sample, and does multilingual-e5 run on LiteRT? | `Tools/Research/Rag/README.md` (stage 1) |

Each has its own venv (gitignored) and its own `-Bootstrap` switch. They are
independent — installing one does not disturb another, which matters because
they pin conflicting versions.

## Running any of this

The `.ps1` files are unsigned, so `powershell.exe` launched from bash refuses
them with `PSSecurityException`. Run them from PowerShell directly, or add
`-ExecutionPolicy Bypass` when shelling out.
