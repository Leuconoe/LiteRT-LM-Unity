# Changelog

All notable changes to this package are documented here. The version tracks the
LiteRT-LM runtime it wraps.

## [0.14.0b] — 2026-07-29

Text to speech. The package version still tracks the LiteRT-LM runtime it wraps,
which is unchanged at v0.14.0; the suffix marks the package revision.

### Added

- `ILiteRtLmTts` — one interface, three backends, so callers do not change when
  the engine does.
- `LiteRtLmSystemTts` — the platform voice: Windows SAPI (a Korean voice ships
  with the OS) and Android `TextToSpeech`. No model files, no model licence.
  **Not available on the Android target device**, which is an AOSP build with no TTS
  engine and no way to install one — the reason the neural backend exists.
- `LiteRtLmSupertonicTts` — Supertonic running on LiteRT, the runtime the LLM and
  ASR paths already use. Verified on device: **RTF 0.15–0.27, four to seven times
  faster than real time** on kona (CPU), audio confirmed by round-trip ASR.
- `LiteRtLmSupertonicText` — the reference text front end ported to C#. Unicode
  NFKD runs here so that ICU stays out of the AAR; byte-identical to the Python
  reference across 8 cases, Hangul jamo decomposition included.
- `LiteRtLmUnityClient.RunSupertonicTts` and the matching
  `UnityLiteRtLmBridge.runSupertonicTts` / native runner. One bridge call per
  utterance: the flow-matching loop stays native, so neither the latent nor the
  PCM crosses JNI.
- **TTS test scene** (interactive, backend toggle and flow-step slider) and a
  headless **TTS smoke scene** with an APK build entry that packages only the
  TTS model set.
- `LiteRtLmTtsDisclosure` — the "this voice is synthesized" notice in Korean and
  English, in one place. OpenRAIL-M use restriction (e) requires machine-generated
  audio to be disclosed; the test scene shows it and the smoke runner records it
  in the status file next to the WAVs. Shown for every backend, not only the one
  that carries the term — a notice that appears sometimes is worse than one that
  always does.

### Notes

- Model weights are **OpenRAIL-M** and are not redistributed with the package.
  Commercial use is permitted — the restrictions are on use cases, not commerce —
  and was accepted for this project on 2026-07-29. Anyone shipping the weights
  still has to include the licence with them, pass the use restrictions
  downstream, and disclose that the audio is machine-generated (clause (e)).
  See `docs/tts-details.md`.
- CPU is the shipping backend. The OpenCL delegate rejects the converted graphs
  on a BHWC shape mismatch; that is a conversion-layout limitation, not a tuning
  knob.

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
