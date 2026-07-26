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

## Benchmarks (source data)

- [ASR model matrix](benchmarks/asr-model-matrix.md) — every tier × 10 clips,
  CER/WER/RTF
- [FC model benchmark](benchmarks/fc-model-benchmark.md) — 20-case scoring
- [Device PDCA ledger](benchmarks/device-cycle1-baseline.md) — cycles 1–6 in full
- [Short-utterance ASR research](benchmarks/short-utterance-asr-research.md)
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
