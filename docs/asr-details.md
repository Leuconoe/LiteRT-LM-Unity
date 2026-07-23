# ASR Details

This document records Android ASR setup and benchmark results for the Unity
LiteRT-LM bridge. The README keeps only requirements and recommended models.

## 2026-07-23 Update — v0.14 tier lineup

### Deployed tiers (`Assets/StreamingAssets/ASR/<model>/`)

Every folder carries its own matching `tokenizer.json` (medium and
large-v3/turbo tokenizers differ from tiny/base). i8/i4 tiers are
project-quantized (int4-minimum-tier policy: `dynamic_wi4b64_afp32` blocks +
i8 sensitive scopes; channelwise `wi4c` is never used).

| Model | f32 MB | i8 MB | i4 MB | Mel bins / vocab |
| --- | ---: | ---: | ---: | --- |
| whisper-tiny | 151.0 | 41.1 | 36.5 | 80 / 51865 |
| whisper-base | 290.1 | 77.0 | 45.3 | 80 / 51865 |
| whisper-medium | — | 831.6 | 664.3 | 80 / 51865 |
| whisper-large-v3 | — | 1631.7 | 1148.1 | 128 / 51866 |
| whisper-large-v3-turbo | — | 1088.3 | 755.3 | 128 / 51866 |
| qwen3-asr-0.6b | — | 793.9 | — (i4 removed: silent failures) | 128-mel, 5 s chunks |

Full CER/WER/RTF matrix over the 9-clip re-recorded test set:
[`docs/benchmarks/asr-model-matrix.md`](benchmarks/asr-model-matrix.md).
Desktop highlights: **turbo i4 is the best tier in the matrix** (8/9 exact,
CER 0.000 ko+en, 72 ms/step); base i8 matches base f32 transcripts at 27 % of
the size; tiny is English-lean (Korean year errors).

### Device recommendations (46a880a0, kona / SM8250)

- **Accuracy: whisper-large-v3-turbo i4** — the only whisper tier that passes
  all 3 device gate clips **including both 볼륨 업 takes** (cycle 3, take5
  AAR). Cost: ~17 s CPU encode + ~0.42 s/decode-step ⇒ 21–24 s per clip —
  accuracy king, not the latency king.
- **Voice commands: qwen3-asr-0.6b i8** — recognizes all short Korean FC
  commands on device including both 볼륨 업 takes (EOS guard prevents empty
  output). ~0.5 s/decode-step, ~1.9 s compile per fresh process; spells out
  numbers (`이천이십오년`) — display-format gap, not a recognition failure.
- **Latency on long clips: whisper-base i8** — clip1 exact in ~2.7 s, but both
  볼륨 업 takes fail on device (see numerics note below).
- gemma-4 E2B audio (multimodal path) transcribes the Korean gate clip
  content-exact in ~4.1 s when the 2.6 GB model is already resident.

**Known device-vs-desktop numerics note (tiny/base short clips)**: device PCM
is bit-identical to desktop, but the device C++ mel/STFT differs from the
Python reference by ~0.1 % in energy (`featureMd5` A/B, cycle 3). Small
whisper models sit near decision boundaries on <1.2 s clips and flip
transcripts; long clips agree, and turbo/qwen3-asr absorb the delta. This is
an implementation-level numeric difference, not a preprocessing bug — solved
in practice by routing short utterances to turbo i4 or qwen3-asr i8.

### Short-utterance improvements (shipped in the AAR)

Implemented after the same-day research
([`docs/benchmarks/short-utterance-asr-research.md`](benchmarks/short-utterance-asr-research.md)):

- **Boost-only RMS loudness normalization** (gain ≥ 1 only, peak-clamped) and
  **energy-gate VAD trim** (0.1 s lead-in / 0.3 s tail retained) in the shared
  ASR PCM path — the VAD trim fixed turbo's `볼륨 억` miss; zero shipped-tier
  regressions measured.
- **Qwen3 EOS min-length guard** (no EOS for the first 3 generated tokens) —
  removes the immediate-EOS empty-output failure mode.
- **Hotword / system-prompt biasing was prototyped but is excluded by user
  policy** — ASR must not be pre-biased toward specific expected values; no
  hotword code ships (native or C#).
- FC keyword matching is space/comma-insensitive in the C# runners
  (display/matching only — model input untouched).

### 128-mel support (take5 AAR)

The whisper JNI path is now signature-driven instead of hardcoded:

- Mel-bin count and vocab size are read from the model (80/51865 tiny–medium,
  128/51866 large-v3/turbo).
- Decode inputs are bound by signature name → tensor shape → positional
  fallback, fixing turbo's reversed `(mask, encoder, tokens)` decode input
  order that broke takes 3–4.
- The whisper smoke JSON reports `melBins`, `vocabSize`,
  `decodeBindingStrategy`, and `featureMd5`/`featureSum` (mel-frontend
  diagnostics).

### Windows ASR (gemma-4 audio path)

`Tools/Windows/litert_lm_advanced_main.windows_x86_64.exe` supports
`[audio:<path>]` prompt tags with `--audio_backend`; the gemma-4-E2B bundle
contains the audio encoder sections and transcribes the Korean gate clip
exactly (mp3 supported, 3.9–5.3 s warm). Scripted entry point:

```powershell
.\Tools\Windows\Run-LiteRtLmWindowsAsrSmokeTest.ps1 `
  -AudioPath "Assets\StreamingAssets\TestAssets\Audio\2025년 3월 5일 전술평가 결과 보고.mp3"
```

Pitfall: the CLI media-tag regex rejects paths containing whitespace — the
script auto-stages such files under a space-free path. Logs land in
`Builds/Logs/WindowsAsrSmoke/`. Details:
[`docs/benchmarks/fc-model-benchmark.md`](benchmarks/fc-model-benchmark.md)
(Windows ASR smoke section).

## Setup (historical baseline, 2026-05-17)

- Report date: 2026-05-17
- Unity package: `com.Leuconoe.LiteRTLMUnity`
- Device class: Qualcomm Android 12 physical device, about 7.52 GiB RAM
- Test audio: Korean `TestAssets/Audio/2025년 3월 5일 전술평가 결과 보고.mp3`
- Runtime inputs: one ASR smoke-test APK plus model/audio/tokenizer/config files
  pushed to app device storage.

ASR requires a tokenizer that matches the model family. ASR assets live under
`Assets/StreamingAssets/ASR/<model>/`. Whisper Tiny uses
`ASR/whisper-tiny/tokenizer.json`; Whisper Base uses
`ASR/whisper-base/tokenizer.json`.

## Required Assets

| Model | Required files (under `Assets/StreamingAssets/`) | Model source | Tokenizer source |
| --- | --- | --- | --- |
| Whisper Tiny i8 CPU | `ASR/whisper-tiny/whisper_tiny_30s_i8.tflite`, `ASR/whisper-tiny/tokenizer.json` | [litert-community/whisper-tiny](https://huggingface.co/litert-community/whisper-tiny) | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) |
| Whisper Tiny i8 GPU attempt | `ASR/whisper-tiny/whisper_tiny_30s_i8.tflite`, `ASR/whisper-tiny/whisper_tiny_30s_i8_encoder.tflite`, `ASR/whisper-tiny/tokenizer.json` | [litert-community/whisper-tiny](https://huggingface.co/litert-community/whisper-tiny) | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) |
| Whisper Tiny f32 GPU split | `ASR/whisper-tiny/whisper_tiny_30s_f32.tflite`, `ASR/whisper-tiny/whisper_tiny_30s_f32_encoder.tflite`, `ASR/whisper-tiny/tokenizer.json` | [litert-community/whisper-tiny](https://huggingface.co/litert-community/whisper-tiny) | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) |
| Whisper Base f32 CPU | `ASR/whisper-base/whisper_base_30s_f32.tflite`, `ASR/whisper-base/tokenizer.json` | [litert-community/whisper-base](https://huggingface.co/litert-community/whisper-base) | [openai/whisper-base](https://huggingface.co/openai/whisper-base) |

## Benchmarks (historical, 2026-05-17 — pre-v0.14, old recordings)

Superseded by `docs/benchmarks/asr-model-matrix.md` (re-recorded clip set)
and the device cycles in `docs/benchmarks/device-cycle1-baseline.md`.

### Results

The run tested the `.tflite` ASR files that were present in
`Assets/StreamingAssets`. CPU and GPU were both attempted for each full ASR
model.

| Model | Backend requested | Backend used | Status | File size | Compile s | Encode s | Decode s | Elapsed s | Transcript / failure |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `whisper_tiny_30s_i8.tflite` | CPU | CPU | PASS | 41.12 MB | 0.079 | 0.333 | 1.028 | 1.629 | `2025년 3월 5일 전술 평가 결과보고` |
| `whisper_tiny_30s_i8.tflite` | GPU | GPU encoder + CPU decoder | FAIL | 41.12 MB + 33.15 MB encoder | N/A | N/A | N/A | N/A | GPU encoder companion was found, but encoder compilation failed: `Failed to compile model`. |
| `whisper_tiny_30s_f32.tflite` | CPU | CPU | PASS | 150.98 MB | 0.092 | 0.462 | 2.147 | 2.899 | `2025년 3월 5일 전술 평가 결과보고` |
| `whisper_tiny_30s_f32.tflite` | GPU_FP16 | GPU encoder + CPU decoder | PASS | 150.98 MB + 32.94 MB encoder | 1.535 first run, 0.000 cached | 0.097 | 1.959 | 2.372 10-run avg | `2025년 3월 5일 전술 평가 결과 보고` |
| `whisper_base_30s_f32.tflite` | CPU | CPU | PASS | 290.08 MB | 0.242 | 1.082 | 4.205 | 5.739 | `2025년 3월 5일 전술평가 결과 보고` |
| `whisper_base_30s_f32.tflite` | GPU | N/A | FAIL | 290.08 MB | N/A | N/A | N/A | N/A | Current GPU split path requires `whisper_base_30s_encoder_f32.tflite`, but that companion is not present in `StreamingAssets`. |

### Repeat Stability

The same Korean audio was repeated 10 times on the physical test device with
`whisper_tiny_30s_f32.tflite`.

| Backend requested | Backend used | Runs | Avg compile s | Avg encode s | Avg decode s | Avg elapsed s | Transcript |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| CPU | CPU | 10 | 0.095 | 0.491 | 2.145 | 2.919 | `2025년 3월 5일 전술 평가 결과보고` |
| GPU_FP16 | GPU encoder + CPU decoder | 10 | 0.153 | 0.097 | 1.959 | 2.372 | `2025년 3월 5일 전술 평가 결과 보고` |

The GPU encoder is about 5x faster than CPU encoding for this clip, but the
native smoke path now keeps compiled encoder and decoder models alive while the
process is running. In the 10-run GPU_FP16 test, run 1 compiled the models
(`compiledModelCache=miss`, 1.535 s compile) and runs 2-10 reused them
(`compiledModelCache=hit`, 0.000 s compile). This makes the cached GPU split
path faster than the CPU path for repeated utterances on the same process.

### GPU Notes

The Unity ASR GPU path is split execution: the encoder is compiled for GPU and
the decoder runs on CPU. The full Whisper model alone is sufficient for CPU, but
GPU currently needs a matching encoder companion next to the full model.

The `whisper_tiny_30s_f32_encoder.tflite` companion used for the successful
Whisper Tiny f32 GPU test is a project-generated split encoder artifact. It is
not part of the upstream `litert-community/whisper-tiny` model files. Keep it
with the full f32 model when testing Unity GPU ASR. The Unity runner creates a
legacy data-file alias named `whisper_tiny_30s_encoder_f32.tflite` because the
current native AAR still derives that internal name.

`whisper_tiny_30s_i8.tflite` has an i8 encoder companion in the project, but the
Qualcomm device rejected it during GPU compilation. The `litert-community`
Whisper Tiny model card exposes multiple hardware-specific f32 variants and an
i8 full model, but does not document this i8 encoder companion as a guaranteed
GPU path.

`whisper_base_30s_f32.tflite` passed on CPU. The `litert-community/whisper-base`
model card currently provides the full f32 model; no matching encoder companion
was present in the project, so the GPU split path could not start.

### Recommendations

1. Use `whisper_tiny_30s_i8.tflite` on CPU as the smallest default
   Korean/English ASR model.
2. Use `whisper_tiny_30s_f32.tflite` plus
   `whisper_tiny_30s_f32_encoder.tflite` when validating the GPU split path.
   The encoder step is much faster on GPU. The first utterance still pays the
   compile cost, but repeated utterances reuse the compiled encoder and decoder.
3. Use `whisper_base_30s_f32.tflite` only for quality comparison; it is much
   larger and slower than Tiny.

## Smoke Tests

The current ASR runner can reuse a single APK. Push the selected model, audio,
tokenizer, and runtime JSON config into app storage, then launch the same build:

```powershell
.\Tools\Windows\Run-LiteRtLmAndroidAsrSmokeTest.ps1 `
  -DeviceSerial <device-serial> `
  -ApkPath Builds\Android\LiteRtLmAndroidAsrSmokeTest-generic.apk `
  -ModelFileName "ASR/whisper-tiny/whisper_tiny_30s_i8.tflite" `
  -AudioFileName "TestAssets/Audio/2025년 3월 5일 전술평가 결과 보고.mp3" `
  -TokenizerJsonPath "ASR/whisper-tiny/tokenizer.json" `
  -AsrMode whisper `
  -AsrLanguage ko `
  -Backend CPU `
  -TimeoutSeconds 300
```

Use `-Backend GPU` only when the matching encoder companion model is present and
known to compile on the target device.

Qwen3-ASR uses `-AsrMode qwen3` (CPU only, language auto-detect):

```powershell
.\Tools\Windows\Run-LiteRtLmAndroidAsrSmokeTest.ps1 `
  -DeviceSerial <device-serial> `
  -ApkPath Builds\Android\LiteRtLmAndroidAsrSmokeTest-generic.apk `
  -ModelFileName "ASR/qwen3-asr-0.6b/qwen3_asr_0.6b_5s_i8.tflite" `
  -AudioFileName "TestAssets/Audio/volume-볼륨, 업.mp3" `
  -TokenizerJsonPath "ASR/qwen3-asr-0.6b/tokenizer.json" `
  -AsrMode qwen3 `
  -AsrLanguage auto `
  -Backend CPU `
  -TimeoutSeconds 300
```
