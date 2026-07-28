# Documentation

The project README stays short; the detail lives here.
**Every document is written against Android on-device execution** — desktop
numbers are reference only.

> **Code moved on 2026-07-27.** The integration is now a UPM package. Documents
> written before that date refer to the old locations; map them like this:
>
> | Old path | Now |
> | --- | --- |
> | `Assets/Scripts/LiteRTLM/LiteRtLm{UnityClient,WindowsCliClient,MicVadCapture,StatusHudOverlay}.cs` | `Packages/com.leuconoe.litert-lm-unity/Runtime/` |
> | `Assets/Scripts/LiteRTLM/*Runner.cs`, `Editor/` | `Packages/com.leuconoe.litert-lm-unity/Samples~/TestScenes/` |
> | `Assets/Scenes/Tests/*.unity` | same `Samples~/TestScenes/Scenes/` |
> | `Assets/Plugins/Android/litertlm-unity-bridge.aar` | `…/Runtime/Plugins/Android/` |

## Details

- [LLM details](llm-details.md) — tiers, backend choice, device measurements
- [ASR details](asr-details.md) — full lineup, VAD, ACFT-KO training background,
  smoke-test commands
- [TTS model research and plan](tts-model-research.md) — engine choice for the
  missing half of the voice loop, licences, and the phased build

## Benchmarks (source data)

- [ASR model matrix](benchmarks/asr-model-matrix.md) — every tier × 10 clips,
  CER/WER/RTF
- [FC model benchmark](benchmarks/fc-model-benchmark.md) — 20-case scoring
- [Device PDCA ledger](benchmarks/device-cycle1-baseline.md) — cycles 1–6 in full
- [Short-utterance ASR research](benchmarks/short-utterance-asr-research.md)
- [Whisper tflite on Windows](benchmarks/whisper-windows-tflite.md) — desktop
  runs, and why the 128-mel device failure is a JNI bug
- [gemma-4 GGUF comparison](benchmarks/gemma4-gguf-vs-litertlm.md)
- [v0.14 session report](benchmarks/session-final-report-20260723.md)

## Handoffs

Kept for traceability and continuation.

- [ASR training program](handoffs/asr-training-program-handoff.md) — clean ACFT
  recipe, the abandoned kspon program (artifacts deleted; this is the only
  surviving record), and the rules for any future retraining
- [v0.14 upgrade](handoffs/v0.14-upgrade-handoff.md)
- [Android device benchmark](handoffs/android-device-benchmark-handoff.md)
- [Function calling](handoffs/function-calling-handoff.md)
- [Sample scene rework](handoffs/sample-scene-rework-verification.md) — the
  eleven scene defects, what caused each one and the run that proves the fix
- [Whisper Windows port estimate](handoffs/whisper-windows-port-estimate.md) —
  what it costs to run Whisper tflite from Unity on Windows, and whether it is
  worth doing
- [Sample scene session](handoffs/sample-scene-session-handoff.md) — where the
  work stands, the late Windows ASR and backend changes, and the open question
  about running Whisper tflite on Windows
