# Tools

Split by **how often you should need it**, because the two kinds were mixed and
it was not obvious which scripts were part of the normal workflow and which
existed to answer one question in one session.

| Directory | What lives there |
| --- | --- |
| `Windows/` | The standing workflow: build the AAR, restore samples, compile-check the runtime, build and run the device tests. Plus the prebuilt Windows binaries and DLLs. |
| `Research/` | One-off investigations: model conversion, quantization, benchmarks, candidate evaluation. Kept because the numbers in `docs/` are only trustworthy if the driver that produced them still exists. |
| `UnityAar/` | The native patch applied to `External/LiteRT-LM` when building the AAR. |

The rule for adding a script: if you would run it again next week as part of
building or testing the product, it belongs in `Windows/`. If it exists to
answer a question and produce a number, it belongs in `Research/`.

## Windows — the standing workflow

| Script | What it does |
| --- | --- |
| `Build-LiteRtLmUnityAarFromPatch.ps1` | Builds `litertlm-unity-bridge.aar` in Docker from `External/LiteRT-LM` + `UnityAar/litert-lm-unity-aar.patch` |
| `Restore-LiteRtLmSamples.ps1` | Imports `Samples~` into `Assets/Samples/…`. **Required after editing anything under `Samples~`** — Unity does not compile that folder |
| `Sync-LiteRtLmGeneratedScenes.ps1` | Copies generated `.unity` scenes back into `Samples~`, so the next restore does not delete them |
| `Invoke-LiteRtLmRuntimeCompileCheck.ps1` | Roslyn-compiles `Runtime/` for the editor and Android define sets in seconds, without launching Unity |
| `Run-LiteRtLmEditorSelfTest.ps1` | Editor-side self test |
| `Run-LiteRtLmSample.ps1` | Runs a sample scene |
| `Build-LiteRtLmAndroid*Apk.ps1` | APK builders for the device gates (smoke, ASR smoke, FC demo) |
| `Run-LiteRtLmAndroid*.ps1` | Device runners for those gates, plus `Run-LiteRtLmAndroidDeviceBenchmarks.ps1` |
| `Run-LiteRtLmWindowsAsrSmokeTest.ps1` | Desktop ASR smoke through the CLI |
| `litert_lm_main.windows_x86_64.exe`, `*.dll` | Prebuilt Windows runtime. Source is not committed; see `CLAUDE.md` before rebuilding |

Device runs need `-DeviceSerial 46a880a0` — an emulator is usually attached too.

## Research — how each number in `docs/` was produced

| Directory | Question it answered | Written up in |
| --- | --- | --- |
| `Supertonic/` | Can Supertonic TTS run on LiteRT, and how fast? Conversion, quantization, bucketing, desktop/device comparison | `docs/tts-model-research.md` |
| `Whisper/` | Can the Whisper tflite exports run on Windows? | `docs/benchmarks/whisper-windows-tflite.md` |
| `Kanana/` | How does a candidate LLM score on our 20-case Korean routing set when LiteRT cannot run it yet? | `docs/llm-details.md` |

Each has its own venv (gitignored) and its own `-Bootstrap` switch. They are
independent — installing one does not disturb another, which matters because
they pin conflicting versions.

## Running any of this

The `.ps1` files are unsigned, so `powershell.exe` launched from bash refuses
them with `PSSecurityException`. Run them from PowerShell directly, or add
`-ExecutionPolicy Bypass` when shelling out.
