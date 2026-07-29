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

### Whisper translate task (take8 AAR, task #26)

`runWhisperAsrSmoke` accepts a `task` parameter (`"transcribe"` default /
`"translate"`): translate swaps the decoder-prompt task token to Whisper's
native X→English translation. Token ids per vocab family (verified against
each `tokenizer.json` `added_tokens` — the 51866 family inserts `<|yue|>` at
50358, shifting the task tokens up by one):

| Token | 51865 family (tiny–medium) | 51866 family (large-v3/turbo) |
| --- | ---: | ---: |
| `<|startoftranscript|>` | 50258 | 50258 |
| `<|ko|>` | 50264 | 50264 |
| `<|translate|>` | **50358** | **50359** |
| `<|transcribe|>` | 50359 | 50360 |
| `<|notimestamps|>` | 50363 | 50364 |

The result JSON reports `task` and `taskTokenId`. Output language is always
English (that is all Whisper's translate task supports). Notes:

- ACFT-KO distilled tiers were trained on the transcribe task only —
  translate quality through them is unvalidated; use stock tiers for
  direct translation.
- The `LiteRtLmTranslateTestScene` exposes both this path (engine
  "Whisper Direct") and an ASR→LLM pipeline (any ASR tier → Qwen3-0.6B
  int4 with a translation prompt + `/no_think`, target
  English/Japanese/Chinese).

### Windows ASR (gemma-4 audio path)

`Tools/Windows/Bin/litert_lm_advanced_main.windows_x86_64.exe` supports
`[audio:<path>]` prompt tags with `--audio_backend`; the gemma-4-E2B bundle
contains the audio encoder sections and transcribes the Korean gate clip
exactly (mp3 supported, 3.9–5.3 s warm). Scripted entry point:

```powershell
.\Tools\Windows\Tests\Run-LiteRtLmWindowsAsrSmokeTest.ps1 `
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
.\Tools\Windows\Tests\Run-LiteRtLmAndroidAsrSmokeTest.ps1 `
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
.\Tools\Windows\Tests\Run-LiteRtLmAndroidAsrSmokeTest.ps1 `
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


---

## Full deployed lineup — Android criteria (2026-07-26, moved out of the README)

Ranked by measurements on device 46a880a0 (Snapdragon 865 / 7.5 GB /
Android 12). Every folder needs its own tier-specific `tokenizer.json` (medium
and large-v3/turbo each use a different one). Full CER/WER/RTF matrix:
[`benchmarks/asr-model-matrix.md`](benchmarks/asr-model-matrix.md).

### First — you usually need only **one** ASR model

Choose by the utterance lengths you handle. The only reason to keep two
resident is when input straddles the 5 s boundary and both sides must be
handled at their best.

| Utterances you handle | One model | Why |
| --- | --- | --- |
| **≤5 s** (voice commands, short sentences) | base-acft-ko 5s (101 MB) | Even a 3.8 s sentence clip comes out exact — inside 5 s this model covers sentences too |
| **5–30 s** (dictation, notes) | whisper-base 30s i8 (77 MB) | Sentence CER 0.000, but it fails on short commands (below) |
| **>30 s** (meetings, lectures) | qwen3-asr-0.6b (794 MB) | Chunk loop. Batch processing only |

**They do not substitute for each other.** On this device stock base-30s reads
`볼륨, 업` as `뽈림` and `볼륨 업` as `보여요.`, so it cannot serve commands
(cycle 2). Conversely base-acft-ko 5s has a 500-frame window, so anything past
5 s is truncated. The ACFT models do ship 10s/30s exports, but **command
accuracy degrades as the window grows** (turbo 30s turns `볼륨, 업` into
`볼륨`; medium 10s/30s turns `음량 증가` into `음향증가`), so they do not
replace stock 30s for long-form.

### All tiers compared

`Mobile` verdicts: **resident** = fine to keep loaded · **on-demand** = load
when needed, then release · **batch** = cannot meet interactive latency,
background only · **not deployed** = does not fit our on-device targets.
`Hit rate` is the number of exact matches on the device command/gate clips.

| Use case (device) | Model (under `Assets/StreamingAssets/`) | Size | Mobile | Hit rate | E2E | Device result |
| --- | --- | ---: | --- | ---: | ---: | --- |
| Voice commands, 1st pick | `ASR/whisper-base-acft-ko/acft_base_5s_drq.tflite` | 101 MB | **resident** | **4/5** | 0.7–0.8 s | Every normal-loudness command and the 3.8 s sentence exact. Encoder is ~12× faster than stock base-30s (0.05 vs 0.62 s) and decode is 1.8× faster per step. The one miss is a quiet 0.79 s legacy recording |
| Dictation / sentence transcription | `ASR/whisper-base/whisper_base_30s_i8.tflite` | 77 MB | **resident** | 6/9 | 2.7 s | Korean sentence CER 0.000. Fails short commands (`뽈림` / `보여요.`); clips under 1.2 s are unstable (mel numerics) |
| Command accuracy fallback | `ASR/whisper-turbo-acft-ko/acft_turbo_5s_drq.tflite` | 883 MB | **on-demand** | **5/5** | 1.9 s warm / 4.0 s cold | The **only** model that reads even the quiet `볼륨 업` legacy take. Its 4-layer decoder keeps it light for 883 MB (≈0.15 s/step) — load it only to retry low-confidence results |
| Long-form (>30 s) | `ASR/qwen3-asr-0.6b/qwen3_asr_0.6b_5s_i8.tflite` | 794 MB | **batch** | 4/9* | RTF ≈2.6 | Unlimited length via a 5 s chunk loop — 98 s of audio transcribed in full across 20 chunks (4.2 min, flat RAM). *The misses are only digits spelled out in Hangul |
| Tiny (English) | `ASR/whisper-tiny/whisper_tiny_30s_i8.tflite` | 41 MB | resident (English) | 3/9 | ~1 s | English CER 0.000. Misreads Korean years, unstable on very short clips |
| Desktop accuracy leader | `ASR/whisper-large-v3-turbo/whisper_large_v3_turbo_30s_i4.tflite` | 755 MB | **batch** | **8/9** | **21–24 s** | Best in the matrix (CER 0.000), device gate 3/3. Encodes the full 30 s window every time, so interactive use is out |
| Accuracy reference | `ASR/whisper-large-v3/whisper_large_v3_30s_i4.tflite` | 1148 MB | **not deployed** | 7/9 | 60–170 s | Character-perfect but 3–7× slower than turbo-30s — desktop comparison only |
| Mid tier | `ASR/whisper-medium/whisper_medium_30s_i8.tflite` (i4: 664 MB) | 832 MB | **not deployed** | 7/9 | — | 24-layer decoder, ≈0.46 s/step on device — larger and slower than turbo (4 layers, ≈0.15) at a lower score on our set |
| medium-acft-ko | `ASR/whisper-medium-acft-ko/` | 826 MB | **not deployed** | 4/5 | 4.8–5.3 s | Kept for the test scene only — turbo-acft scores higher and runs faster on this device |
| tiny-acft-ko | `ASR/whisper-tiny-acft-ko/` | 46 MB | **not deployed (Korean)** | **1/4** | ~0.5 s | Cycle-4 reject. Real-recording command CER 0.896 — the 0.3 s saved over base-5s does not pay for the accuracy loss |

### Why 883 MB turbo-acft works and 755 MB turbo-30s does not

Same large-v3-turbo weights, yet device latency differs by more than 10×
(**1.9 s vs 21–24 s**). The cause is not size but **how long a window gets
encoded**.

- turbo-**30s**: input is always 3000 frames, so a one-second command still
  pays for a 30-second encode
- turbo-**acft 5s**: ACFT self-distillation retrained it for a 500-frame (5 s)
  window — 1/6 the encoder work, and the decoder was already 4 layers at
  ≈0.15 s/step
- In other words, the mobile ASR bottleneck is **fixed window length**, not
  parameter count. As long as you handle short utterances, dropping to a
  smaller 30 s-window tier will not recover this loss.

⚠️ **Never feed audio longer than 30 s straight into a whisper 30 s model** —
it fails three ways at once (truncation, token cap, early stop). Use the qwen3
chunk path for long-form.

**Alternative path**: gemma-4's audio input (multimodal LLM) also transcribes —
when the LLM is already resident it handles **transcription and function
calling in a single turn** with no extra model (4.1 s on device).

### Model provenance

whisper tiny/base come from
[litert-community](https://huggingface.co/litert-community/whisper-tiny)
([base](https://huggingface.co/litert-community/whisper-base)); medium,
large-v3, turbo and every i8/i4 tier are **quantized by this project**
(int4-minimum-tier policy: `dynamic_wi4b64_afp32` plus i8 on sensitive scopes —
`External/community-release/` holds the community copies and manifest).
Tokenizers are the per-tier
[openai/whisper-*](https://huggingface.co/openai/whisper-tiny) files.
Qwen3-ASR is the official tflite with a project JNI port.

## VAD (voice activity detection) — Android behavior

Every ASR path supports `vadMode`.

| Mode | What it does | Cost |
| --- | --- | --- |
| `energy` (default) | Adaptive-threshold v2 — 300 ms noise calibration, 20th-percentile noise floor +9 dB on / 6 dB hysteresis, 210 ms hangover, 90 ms preroll, speech-only RMS gain | 0 (no extra model) |
| `ai` | Silero VAD v5 tflite (`ASR/silero-vad/silero_vad_16k.tflite`, 1.25 MB, MIT). Needs a 0.2 s head pad before whisper | 1.25 MB model |
| `off` | No preprocessing | — |

- **31/31 detected** on a 98 s / 31-utterance stress clip, with device and
  desktop boundaries matching exactly
- Result JSON: `vadMode / vadModeUsed / speechSegments / trimmedSeconds /
  vadGain / speechRms (+vadError)`
- The Unity live microphone path (`LiteRtLmMicVadCapture`) mirrors the same
  parameters in C# for automatic endpointing → 16 kHz WAV → transcription with
  the selected model
- **The quiet 0.79 s clip cannot be fixed by VAD or gain** (proven across 16
  combinations) — it is a model-capacity limit. The fix is tier escalation
  (turbo-acft)

## Korean ACFT 5 s models — training background and how to read the numbers

Stock whisper is trained for a 30 s window, so feeding it a 5-second context
directly produces runaway repetition (CER 1.1–24.9) — this is an
out-of-distribution effect, not a defect in the upstream models. These models were trained in-house with futo ACFT
self-distillation plus two corrections: an `n_ctx` floor of 250 and a 70:30
Korean:English mix (zeroth + FLEURS). The encoder is **~12× faster** than the
30 s window. Model card:
[leuconoe/whisper-acft-ko](https://huggingface.co/leuconoe/whisper-acft-ko).

Gate results (40-clip TTS-synthesized holdout, Korean short-utterance CER at
5 s ctx): turbo 0.182 · medium 0.208 · base 0.305 · tiny 0.457.

⚠️ **Do not quote these as absolute quality numbers.**

- The holdout is edge-tts synthesized, so it is **validated as a ranking metric
  only** (Spearman ρ = 1.00, Pearson 0.99 against real recordings)
- For a strong base it **overstates real error by ~2.8×**
- References average 3.6 characters, so one period is worth CER +0.25–0.50 —
  while the shipped matcher ignores punctuation
  (`LiteRtLmAsrTestRunner.cs`)
- **tiny is not recommended for Korean voice commands** — real-recording
  command CER 0.896, device 1/4 exact (cycle-4 REJECT). The shipped tiers are
  base (1st pick) and turbo (accuracy fallback)

Full analysis: [`benchmarks/asr-model-matrix.md`](benchmarks/asr-model-matrix.md)
Addendum 3, [`handoffs/asr-training-program-handoff.md`](handoffs/asr-training-program-handoff.md) §3–5.
