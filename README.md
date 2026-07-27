# LiteRT-LM-Unity

Unity integration for **on-device AI on Android**. Runs LLM chat, speech
recognition (ASR), image understanding and function calling entirely on the
device, with no network.

- **LiteRT-LM v0.14.0** (`.litertlm` 1.5.0); customizations live in
  `Tools/UnityAar/litert-lm-unity-aar.patch`
- Device-verified on **Snapdragon 865 / 7.5 GB RAM / Android 12** — all four
  capabilities PASS across 6 PDCA cycles, 80+ runs, zero crashes
  ([ledger](docs/benchmarks/device-cycle1-baseline.md))
- Published models:
  [whisper-acft](https://huggingface.co/litert-community/whisper-acft) ·
  [whisper-acft-ko](https://huggingface.co/leuconoe/whisper-acft-ko) ·
  [litert-lm-unity-quantized](https://huggingface.co/leuconoe/litert-lm-unity-quantized)

## Capabilities (measured on device)

| Capability | Speed | Hit rate | Model |
| --- | --- | --- | --- |
| LLM chat | 35.5 tok/s | — | Qwen2.5-0.5B i4 |
| Speech recognition | 0.7–0.8 s | 4/5 | whisper-base-acft-ko 5s |
| Image understanding | 7.6 s (GPU) | accurate | gemma-4-E2B QAT |
| Function calling | 15.5 s E2E (voice → tool) | 19/20 | gemma-4-E2B / Qwen3-0.6B |

## Requirements

Unity **2022.3 or newer** (developed and device-verified on `6000.4.6f1`) +
Android Build Support · Android device (`adb`, Snapdragon 865 class or better,
4 GB+ RAM) · Windows PowerShell (build scripts) · Docker (only to rebuild the AAR)

## Install

The runtime ships as a UPM package,
`com.leuconoe.litert-lm-unity`. In Package Manager choose
**Add package from git URL…** and paste:

```
https://github.com/Leuconoe/LiteRT-LM-Unity.git?path=/Packages/com.leuconoe.litert-lm-unity
```

Or add the line yourself to `Packages/manifest.json`:

```json
{
  "dependencies": {
    "com.leuconoe.litert-lm-unity": "https://github.com/Leuconoe/LiteRT-LM-Unity.git?path=/Packages/com.leuconoe.litert-lm-unity"
  }
}
```

Pin a release by appending a tag: `…litert-lm-unity#v0.14.0a-unity`.
Git URL installs need [git](https://git-scm.com/) on `PATH`; the package carries
a 31 MB Android AAR, so the first resolve takes a moment.

**Samples** — in Package Manager select *LiteRT-LM for Unity* → **Samples** →
*Test Scenes* → **Import**. That brings in six scenes, the Android build menu
and the scene generator.

Working in this repository instead of consuming the package? The samples live in
`Samples~/`, which Unity does not compile. Import them once with:

```powershell
.\Tools\Windows\Restore-LiteRtLmSamples.ps1
```

## Quick Start

1. **Install the package** and import the *Test Scenes* sample (above)
2. **Place models** — pick from the tables below and put them under
   `Assets/StreamingAssets/` (model files are not in the repository)
3. **Build the APK** — Unity menu `LiteRT-LM/Android/...` or
   `Tools/Windows/Build-LiteRtLmAndroid*.ps1`
4. **Smoke test** — `Run-LiteRtLmAndroidAsrSmokeTest.ps1 -DeviceSerial <serial>`;
   results land in `Builds/Logs/AndroidDeviceRuns/`

## Package layout

| Path | Contents |
| --- | --- |
| `Packages/com.leuconoe.litert-lm-unity/Runtime/` | `LiteRtLmUnityClient` (Android bridge), `LiteRtLmMicVadCapture`, `LiteRtLmStatusHudOverlay`, `LiteRtLmWindowsCliClient`, and the native AAR |
| `Packages/com.leuconoe.litert-lm-unity/Samples~/TestScenes/` | Scene runners, the six test scenes, the APK build menu and the scene generator |
| `Assets/StreamingAssets/` | Where you place models (not in the repository) |
| `Tools/Windows/` | Build, AAR and device scripts |
| `docs/` | Benchmarks and handoffs |

## Recommended models

### LLM — pick by device RAM

| Device RAM | Model | Size | Measured on device | Download |
| --- | --- | ---: | --- | --- |
| 4–6 GB | `LLM/qwen2.5-0.5b/…_wi4b64_ekv1280.litertlm` | 265 MB | 35.5 tok/s — chat only (not usable as an FC router) | [project int4](https://huggingface.co/leuconoe/litert-lm-unity-quantized) (upstream [f32](https://huggingface.co/litert-community/Qwen2.5-0.5B-Instruct)) |
| 6–8 GB | `LLM/qwen3-0.6b/qwen3_0_6b_mixed_int4.litertlm` | 475 MB | 20.9 tok/s, FC 18/20 | [litert-community/Qwen3-0.6B](https://huggingface.co/litert-community/Qwen3-0.6B) |
| 8 GB+ | `Multimodal/gemma-4-e2b/gemma-4-E2B-it.litertlm` | 2.6 GB | FC 19/20, image 7.6 s, audio 4.1 s; image turns peak at 3.6 GB PSS | [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) |

`LFM2.5-1.2B int4` (702 MB, 16.8 tok/s, FC 17/20) is also available as a
mid-size FC router. Use **CPU** for chat (decode) and **GPU** for long prompts
and images. [LLM details →](docs/llm-details.md)

### ASR — pick by utterance length (one model is usually enough)

| Utterance length | Model | Size | Measured on device | Download |
| --- | --- | ---: | --- | --- |
| ≤5 s (commands, short sentences) | `ASR/whisper-base-acft-ko/acft_base_5s_drq.tflite` | 101 MB | 0.7–0.8 s, 4/5 exact | [leuconoe/whisper-acft-ko](https://huggingface.co/leuconoe/whisper-acft-ko) |
| 5–30 s (dictation) | `ASR/whisper-base/whisper_base_30s_i8.tflite` | 77 MB | 2.7 s, sentence CER 0.000 | [project i8](https://huggingface.co/leuconoe/litert-lm-unity-quantized) (upstream [f32](https://huggingface.co/litert-community/whisper-base)) |
| >30 s (batch) | `ASR/qwen3-asr-0.6b/qwen3_asr_0.6b_5s_i8.tflite` | 794 MB | chunk loop, RTF ≈2.6 | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) |

- Ship the matching per-tier `tokenizer.json` in the same folder
  ([openai/whisper-*](https://huggingface.co/openai/whisper-base))
- **VAD** is on by default — `energy` (free) /
  `ai` ([Silero](https://huggingface.co/pat229988/silero-vad-16k-tflite), 1.25 MB) / `off`
- For quiet recordings, load turbo-acft-ko 5s (883 MB, **5/5**) on demand
  ([leuconoe/whisper-acft-ko](https://huggingface.co/leuconoe/whisper-acft-ko))
- English-only short speech can use the original futo ACFT models
  ([litert-community/whisper-acft](https://huggingface.co/litert-community/whisper-acft))

[All 10 tiers, selection rationale, ACFT training background →](docs/asr-details.md)

## Test scenes

Shipped as the package's *Test Scenes* sample; after import they land under
`Assets/Samples/LiteRT-LM for Unity/<version>/Test Scenes/Scenes/`. Regenerate
them with the menu `LiteRT-LM/Test Scenes/Generate All` — scene paths resolve
from the import location, so no path editing is needed.

Every scene shows a **◀ Prev / Next ▶** bar so the set can be walked through on
a device. Each switch releases the loaded model first — engines hold native
memory the GC does not track, so the outgoing one is disposed before the next
loads rather than both being resident.

| Scene | Purpose |
| --- | --- |
| `LiteRtLmLlmChatTestScene` | Multi-turn chat, think/no_think toggle |
| `LiteRtLmAsrTestScene` | ASR — file / microphone / always-listening (Continuous) |
| `LiteRtLmMultimodalTestScene` | Image + audio input |
| `LiteRtLmAsrFunctionCallingTestScene` | Voice → tool call (15.5 s) |
| `LiteRtLmMultimodalFunctionCallingTestScene` | Image + utterance → tool call (40.7 s) |
| `LiteRtLmTranslateTestScene` | Translation — Whisper Direct / ASR+LLM |

## Rebuilding the AAR (after native changes)

`Tools/Windows/Build-LiteRtLmUnityAarFromPatch.ps1 -SourceRoot <pristine v0.14.0>`
applies the patch, builds in Docker and deploys to
`Packages/com.leuconoe.litert-lm-unity/Runtime/Plugins/Android/`.

⚠️ `-SkipImageBuild` builds the sources baked into the Docker image, so the
image must be rebuilt after any patch change.

## Docs

| Document | Contents |
| --- | --- |
| [`docs/llm-details.md`](docs/llm-details.md) | LLM tiers, backend choice, device measurements |
| [`docs/asr-details.md`](docs/asr-details.md) | Every ASR tier, VAD, ACFT-KO training background |
| [`docs/README.md`](docs/README.md) | Full benchmark and handoff index |

The Windows editor path exists only to validate logic before deploying to a
device. Its performance profile is the opposite of Android's (GPU wins there),
so never use desktop numbers to make device decisions.
