# Changelog

All notable changes to this package are documented here. The version tracks the
LiteRT-LM runtime it wraps.

## [0.14.0] — 2026-07-27

First release as a UPM package. Previously the same code shipped as loose files
under `Assets/` in the sample project.

### Added

- `LiteRtLmUnityClient` — Android bridge: `Initialize`, `SendMessage`,
  `SendMessageWithMedia` (image + audio), `RunWhisperAsrSmoke`,
  `RunQwen3AsrSmoke`.
- Signature-driven model detection in the native bridge: mel bins (80/128),
  vocab (51865/51866) and window frames (100–3000) are read from the model, so
  5 s ACFT exports and 30 s stock graphs share one code path. Whisper decode
  inputs bind by shape rather than by position.
- Dual-mode VAD on every ASR entry point — `vadMode` = `energy` (adaptive
  threshold, no extra model) / `ai` (Silero v5 tflite, 1.25 MB) / `off`.
- `LiteRtLmMicVadCapture` — 16 kHz microphone capture with streaming energy VAD
  endpointing that mirrors the native parameters, WAV output, continuous mode
  and runtime `RECORD_AUDIO` permission handling.
- Whisper `task` parameter (`transcribe` / `translate`) with the task token
  resolved per tokenizer family.
- `LiteRtLmWindowsCliClient` — Windows editor fallback driving the LiteRT-LM CLI,
  with automatic GPU → CPU fallback.
- `LiteRtLmStatusHudOverlay` — on-screen status/telemetry overlay.
- **Test Scenes** sample: LLM chat, ASR (file / microphone / always-listening),
  multimodal image+audio, voice and multimodal function calling, translation,
  plus the Android APK build menu and the scene generator.

- `LiteRtLmSceneNavigator` — a prev/next bar on the sample scenes so the whole
  set can be walked through on a device without rebuilding. It appears
  automatically on the six sample scenes (set `AutoSpawnEnabled = false` to
  suppress it) and skips scenes missing from Build Settings.
- `ILiteRtLmModelHost` / `LiteRtLmModelMemory.ReleaseAll()` — explicit model
  release. A LiteRT-LM engine holds native memory the GC does not track, so
  every sample component now releases its engine in `OnDestroy`, and the
  navigator releases all of them *before* loading the next scene rather than
  letting two models be resident at once.
  - `LiteRtLmAsrTestRunner` additionally stops the continuous session, the
    microphone and any in-flight transcription (bounded 10 s wait) before
    disposing — the worker thread calls into the engine, so disposing under it
    would be a use-after-free.

### Notes

- Minimum Unity 2022.3. Developed and device-verified on 6000.4.6f1.
- Android is the target platform; the Windows path exists for editor
  verification only and has the opposite backend profile (GPU wins there).
- Model files are not included — see the repository README for the download
  tables.
