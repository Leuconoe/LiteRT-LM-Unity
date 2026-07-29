# Porting the Whisper driver to Windows — cost estimate

Written 2026-07-27, after [Whisper tflite on Windows](../benchmarks/whisper-windows-tflite.md)
proved the models run on desktop through a Python harness. The question here is
what it costs to make Unity do it on Windows, without Python.

**Short answer: ~2–3 focused days** for the recommended route (extract the
existing C++ out of the AAR into a Windows bazel target). The algorithm work is
already done and paid for; almost all remaining risk is build/link plumbing.
Nothing needs to be invented, and nothing needs new hardware or licences.

## What actually has to move

The Whisper pipeline lives in one file,
`External/LiteRT-LM/kotlin/java/com/google/ai/edge/litertlm/jni/litertlm.cc`
(4,823 lines total, of which the Unity patch contributes most). It is **not**
Android-specific: Android appears only in `#if defined(__ANDROID__)` logging and
`dlopen` blocks plus the JNI entry points at the bottom.

| Piece | Lines | Portable as-is? |
| --- | ---: | --- |
| `ReadBinaryFile`, `HannWindow`, tensor helpers | ~55 | yes |
| VAD stack (energy + Silero tflite, trim/normalize, report) | ~396 | yes |
| Slaney mel filterbank + `CreateWhisperInputFeatures` (kissfft STFT) | ~132 | yes |
| Whisper BPE detokenizer (GPT-2 byte decoder, UTF-8 walk, suppression) | ~152 | yes |
| `ResolveWhisperDecodeBinding` (binds decode inputs by shape) | ~155 | yes |
| `RunWhisperAsrSmokeToJson` (model cache, encode/decode loop, JSON result) | ~630 | yes, minus the GPU options branch |
| **Total to extract** | **~1,520** | |
| New: `main()` / flag parsing / optional stdin server loop | ~150–300 | new code |

The dependency list from the JNI BUILD target is all cross-platform:
`@litert//litert/cc:litert_api_with_dynamic_runtime`, `@kissfft//:kissfftr`,
`//runtime/components/preprocessor:audio_preprocessor_miniaudio`, absl,
nlohmann. It already uses the generic LiteRT C++ API (`litert/cc/litert_compiled_model.h`,
`litert_environment.h`, `litert_tensor_buffer.h`) — the same API the Windows
`litert_lm_main` links against.

## Environment readiness (checked, not assumed)

| Item | State |
| --- | --- |
| VS2022 Build Tools (required — VS2026/MSVC 14.51 miscompiles litert) | present: `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools`, MSVC 14.44.35207. Must be pinned via `BAZEL_VC`; `BAZEL_VC` is currently unset and VS 18.8 would be picked by default |
| bazelisk | present on PATH; four bazel output bases already exist (warm-ish cache) |
| `libLiteRt.dll` | ships in `Tools/Windows/`; exports `LiteRtCreateCompiledModel`, `LiteRtCreateEnvironment`, `LiteRtCreateTensorBuffer`, `LiteRtGetModelSignature`, `LiteRtCreateOptions` |
| `@kissfft` | comes transitively from the TensorFlow workspace — built on Windows by TF itself, no new repo rule needed |
| miniaudio decoder | in-repo, header-only, already used by the Windows audio path |
| Windows link recipe | precedent exists: `litert_lm_advanced_main` uses `@litert//litert/c:windows_exported_symbols.def` and a `/DEF:` linkopt |
| Prior Windows bazel build cost | ~16 min for the heavy step, ~6 min for the next, on the 7950X |

Nothing on this list has to be acquired. The one landmine is `BAZEL_VC` — an
unpinned build silently uses MSVC 14.51 and produces a miscompiled binary.

## Options

### A. Extract into a Windows bazel target (recommended)

Move the ~1,520 lines into a new `cc_binary` (e.g.
`runtime/engine/whisper_asr_main.cc`) alongside `litert_lm_main`, add flags,
drop the JNI and GPU branches, emit the same JSON the AAR already emits.

- **A1, one-shot CLI** — matches the existing `LiteRtLmWindowsCliClient`
  subprocess pattern (544 lines, already parses CLI JSON). Cheapest, but repays
  model load per utterance: fine for base (100 MB), poor for medium/turbo
  (830–880 MB).
- **A2, add a stdin/stdout server mode** — keeps the compiled model warm across
  utterances, which continuous ASR in the sample scenes needs. +~150 lines over
  A1 and no DLL-lock problem in the editor.

| Phase | Effort | Risk |
| --- | --- | --- |
| Extract + new BUILD target + flags | 0.5–1 day | low — copy-paste, the code is already parameterised |
| Windows build/link (BAZEL_VC pin, def file, symbol export) | 0.5–1 day | **medium** — the only real unknown |
| C# client + scene wiring (reuse `LiteRtLmWindowsCliClient`) | 0.5 day | low |
| Validation against the 20-run Python baseline | 0.5 day | low — the baseline exists and is one command |
| **Total** | **2–3 days** | |

### B. Pure C# against `libLiteRt.dll`

Reimplement mel/STFT, the byte-level BPE decoder and the KV loop in C#, plus a
P/Invoke layer over the LiteRT C API. ~1,500–2,000 lines of new C#, no bazel and
no MSVC ever again, and it runs in-process in the editor.

**4–6 days, higher risk.** Everything gets rewritten rather than reused; the
compiled-model/signature C API surface would have to be verified export by
export (`LiteRtCompiledModelRun` did not appear in the DLL string scan under that
name); marshalling mistakes fail as access violations, not compile errors. Only
worth it if the goal is to delete the C++ toolchain dependency outright.

### C. Do nothing in Unity

Keep Python as a bench harness and leave the desktop ASR path on gemma-4 audio,
which works today (1.4–3.7 s per clip). **0 days.**

What C forgoes: gemma-4 is a 2.41 GB bundle answering in seconds, where
`whisper_base_30s_i8` is 77 MB answering in ~0.7 s and `acft_base_5s_drq` in
~0.5 s. The win from A is editor iteration speed, desktop/device parity for A/B
work on the same exports, and a desktop translate path that does not route
through a 2.4 GB LLM — not new capability on the shipping target, which is
Android and already has all of this.

## Recommendation

Take **A2**, but gate it: spend the first half-day on the build only — add the
target, pin `BAZEL_VC` to the 2022 Build Tools, and get an empty `main()` that
links against litert + kissfft + miniaudio on Windows. That step contains
essentially all the risk. If it links, the rest is mechanical and the estimate
holds. If it does not link within that half-day, stop and reconsider B, because
the same wall would be hit on every future rebuild.

Do not start this if Android throughput is the priority — none of it improves
the deployment target.
