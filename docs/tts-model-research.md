# TTS: model research and implementation plan

Written 2026-07-27. The project has ASR (device + desktop) and LLM; TTS is the
missing half of the voice loop. This picks an engine and lays out how to build it.

**Assumptions used** (say so if any is wrong — they change the answer):

- Android on device **46a880a0** is the deployment target; Windows is for editor
  parity, as everywhere else in this repo.
- Fully offline. No cloud TTS (the project's own test audio was made with
  `edge-tts`, which is a cloud endpoint and is a test-data tool only).
- Korean first, English second.
- **No new C++/bazel build** — that constraint is what killed the Whisper Windows
  port ([estimate](handoffs/whisper-windows-port-estimate.md)), so it applies here
  too. Prebuilt native libraries are fine; compiling litert or onnxruntime is not.

## Candidates, verified

Facts below were read from the model repositories and release artifacts on
2026-07-27, not from summaries.

| Engine / model | Korean | Size | License | On-device path | Verdict |
| --- | --- | ---: | --- | --- | --- |
| **Supertonic (v1/v2 int8, via sherpa-onnx)** | yes | **92 MB** total (text_encoder 26.2, vector_estimator 38.8, vocoder 24.8, dp 1.5, voice.bin 0.5, unicode_indexer 0.3) | code MIT, **weights OpenRAIL-M** | sherpa-onnx prebuilt libs, C# API | **Android pick** |
| **Qwen3-TTS-12Hz-0.6B-CustomVoice** | yes | 1.69 GB bf16 / ~2 GB int8 ONNX | **Apache-2.0** | no ready runtime — hand-built ONNX pipeline | **desktop pick**, see below |
| **Chatterbox Multilingual (ONNX q4f16)** | yes (23 languages) | **291 MB** q4f16 / 337 MB q4 | **MIT**, weights included | ONNX Runtime, hand-built driver; transformers.js reference exists | **best licence-clean device option**, see below |
| Supertonic 3 (fp32) | yes (31 languages) | 380 MB | same | same, if converted/quantized | too big as-is; int8 build not published by upstream |
| Piper `ko_KR-kss-medium` | yes (only Korean piper voice of 170) | ~63 MB | **CC BY-NC-SA 4.0** (KSS dataset) | sherpa-onnx / piper runtime | **rejected — non-commercial** |
| MMS-TTS kor (facebook) | yes | ~145 MB | **CC-BY-NC 4.0** | onnx export | **rejected — non-commercial** |
| KRAFTON Raon family (5 checkpoints) | ko/en for Raon-Speech; **English only** for Raon-OpenTTS | 5.2–18.1 GB | **CC-BY-NC 4.0, all of them** | PyTorch / vLLM-Omni; AWQ build is GPU-only | **rejected — see below** |
| Kokoro-82M | **no** — sherpa's kokoro packages are Chinese + English only | 82 M params | Apache-2.0 | sherpa-onnx | rejected — no Korean, despite blog claims |
| MeloTTS-Korean | yes | checkpoint.pth only | **MIT** (weights too) | needs ONNX export + Korean G2P port | fallback if OpenRAIL-M is refused |
| Windows SAPI `Microsoft Heami` | yes | 0 (in Windows) | OS component | `System.Speech` / SAPI | baseline, desktop only |
| Android `android.speech.tts` | device-dependent | 0 | OS component | `AndroidJavaObject`, no AAR change | baseline, Android only |

Verified on this machine: `Microsoft Heami` (ko-KR, female) **and** `Microsoft
Heami Desktop` are installed — Windows TTS in Korean needs nothing downloaded.

## KRAFTON Raon — checked, and out on licence

Worth checking because it is Korean-native and recent. Every checkpoint carries
**CC-BY-NC 4.0**, which rules the whole family out of a delivered product for the
same reason Piper's Korean voice and MMS-TTS are out. Each also fails at least
one more test:

| Checkpoint | Licence | Korean | Size | Also fails on |
| --- | --- | --- | ---: | --- |
| Raon-Speech-9B | CC-BY-NC 4.0 | yes | 16.9 GB | 9 B on a 7.5 GB device |
| Raon-Speech-9B-AWQ-INT4 | CC-BY-NC 4.0 | yes | 7.30 GB | AWQ is a **GPU** kernel (benchmarked on an L40S via vLLM-Omni) — no CPU or mobile path |
| Raon-SpeechChat-9B | CC-BY-NC 4.0 | yes | 18.1 GB | size |
| Raon-OpenTTS-0.3B | CC-BY-NC 4.0 | **no — English only** | 5.17 GB | language; 5 GB for a 0.3 B model (unquantized `.pt`) |
| Raon-OpenTTS-1B | CC-BY-NC 4.0 | **no — English only** | 15.5 GB | language, size |

### Raon-Speech-9B looked at properly

It is the only candidate here that replaces **ASR and TTS at once** — one model
doing STT, TTS, TextQA and SpeechChat in Korean and English (Qwen3 backbone,
Qwen3OmniMoe audio encoder, Mimi codec with 32 quantizers, ECAPA-TDNN speaker
encoder), with optional voice conditioning from a reference clip.

Published streaming numbers, single GPU:

| | RTX 6000 Pro | L40S |
| --- | --- | --- |
| RTF | 0.27 (3.7× real-time) | 0.45 (2.2× real-time) |
| Time to first audio | 617 ms | 887 ms |
| Time between chunks | 135 ms | 233 ms |

**This workstation can run it.** It has an RTX 4090 with 24 GB, so the AWQ-INT4
build (7.30 GB) fits comfortably and BF16 (16.9 GB) fits with room for KV cache.
Expect somewhere around the two columns above.

**The device cannot, and no amount of work changes that.** kona has 7.5 GB of
system RAM, no CUDA, and no NPU in the litert-community lists; AWQ is a GPU kernel
format; and the architecture is `trust_remote_code` PyTorch with no ONNX, GGUF or
tflite path to port. Adopting Raon therefore means moving speech off the device
and onto a companion machine — a GPU-equipped ground station reached over the
network. That is a legitimate architecture for a drone system, but it is a
different product from the on-device one this repo has been built around: it adds
a network dependency, link latency on top of the 0.6–0.9 s first-audio, and a
second machine to power and maintain.

So Raon is a **system-architecture decision, not an engine swap**. If the answer
is "the ground station may host it", Raon-Speech-9B-AWQ-INT4 is a strong choice
and would also let the ASR side collapse into the same model. If speech must keep
working with nothing but the headset, Raon is out and the choice is between
Supertonic and Chatterbox below.

## Qwen3-TTS — the licence-clean option

Released 2026-01, **Apache-2.0 across all four checkpoints**, Korean among its ten
languages, and from the same family as the Qwen3-ASR model this project already
ships. Licence-wise it is strictly better than Supertonic: no RAIL use
restrictions, no archived upstream.

| Checkpoint | Weights | What it gives | Voice source |
| --- | ---: | --- | --- |
| Qwen3-TTS-12Hz-0.6B-Base | 1.70 GB | 3-second voice cloning; fine-tuning base | reference audio + its transcript, per call |
| Qwen3-TTS-12Hz-1.7B-Base | 3.59 GB | same, higher quality | same |
| **Qwen3-TTS-12Hz-0.6B-CustomVoice** | 1.69 GB | **9 built-in timbres + instruction style control**, streaming | none needed |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | 3.57 GB | same, higher quality | none needed |

**CustomVoice, not Base**, for this product: Base requires shipping a reference
clip and its transcript with every request, and voice cloning is a liability we
have no use for. **0.6B, not 1.7B**, for the device.

Upstream claims a dual-track streaming architecture with first-packet latency as
low as 97 ms, and vLLM has day-0 support — both server-class datapoints, not
device ones.

### The catch: there is no runtime for it here

- **sherpa-onnx does not support it.** Its offline TTS config covers vits,
  matcha, kokoro, zipvoice, kitten, pocket and supertonic — that is the whole list.
- **llama.cpp does not support it.** `convert_hf_to_gguf.py` on master has zero
  mentions of `Qwen3TTS`/`qwen3_tts`; the GGUF repos on the Hub come from other
  tooling and cannot be relied on.
- **LiteRT/tflite: no.** Converting an autoregressive talker plus a code predictor
  plus a 12 Hz codec decoder is far beyond the ai-edge-torch path this project has
  used, and the LiteRT-LM converter is still a stub.
- What does exist are **community ONNX exports** — e.g. `sivasub987/Qwen3-TTS-0.6B-ONNX-INT8`
  (~1.98 GB: talker_prefill 427 MB, talker_decode 427 MB, tokenizer12hz_decode
  435 MB, text_project 303 MB, code_predictor 105 MB, …) and
  `elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX` (~4.25 GB, unquantized).
  Unverified third-party artifacts.

Using it in Unity therefore means writing the inference driver ourselves:
prefill/decode session management with a KV cache across ONNX graphs, the code
predictor loop, then codec decoding — in C# against ONNX Runtime, on both
platforms. That is the same class of work as the C# Whisper option that was
priced at 4–6 days, and bigger, because this is an autoregressive LLM pipeline
rather than a single encoder/decoder pair. **Realistically 1.5–3 weeks**, with a
real risk it lands too slow on kona: the talker is 0.6 B, and this project's own
device numbers put a 1 B int4 model at ~16 tok/s decode.

## Chatterbox Multilingual — MIT, and small enough for the device

Resemble AI's Chatterbox is **MIT-licensed weights and code**, 23 languages with
Korean among them, 0.5 B parameters. What makes it the strongest licence-clean
candidate is the export: `onnx-community/chatterbox-multilingual-ONNX` ships
quantized ONNX graphs at **q4f16 291 MB** and **q4 337 MB** (fp16 992 MB, fp32
3.13 GB), plus `default_voice.wav`, so no reference-clip management is needed
even though the model is a voice cloner underneath.

That is 291 MB against Qwen3-TTS's ~2 GB, at a better licence, with Korean in
both. Resemble also publishes its own `chatterbox-turbo-ONNX` (English only) and
a community `chatterbox-nano-ONNX` (English, 83 MB fp16) — neither covers Korean.

The runtime gap is the same in kind as Qwen3-TTS but smaller in degree: graphs
are `embed_tokens` / `language_model` / `speech_encoder` / `conditional_decoder`,
which is the standard transformers.js layout, so a working reference pipeline
exists in JavaScript to port to C# against ONNX Runtime, rather than being
reverse-engineered from a paper. Still an autoregressive loop with KV-cache
management written by us: **1–2 weeks**, not days.

### Verdict — three-way

| | Supertonic int8 | Chatterbox ML q4f16 | Qwen3-TTS 0.6B CustomVoice |
| --- | --- | --- | --- |
| Licence | OpenRAIL-M (use restrictions) | **MIT** | Apache-2.0 |
| Korean | yes | yes | yes |
| Device size | **92 MB** | 291 MB | ~2 GB (int8, third-party) |
| Architecture | non-autoregressive flow matching | autoregressive 0.5 B | autoregressive 0.6 B |
| Runtime in Unity | **sherpa-onnx, prebuilt, C# API** | ONNX Runtime + our driver (JS reference) | ONNX Runtime + our driver (no reference) |
| Effort | 2–3 days | 1–2 weeks | 1.5–3 weeks |
| Upstream | archived 2026-07-23 | active | active |

**Recommendation:**

- **Android** — Supertonic if OpenRAIL-M clears legal; otherwise Chatterbox
  Multilingual q4f16, budgeting the driver. Do not put Qwen3-TTS on kona: 2 GB
  plus an autoregressive 0.6 B talker on a device whose 1 B int4 decode measures
  ~16 tok/s is the worst size/speed trade of the three.
- **Windows** — whichever engine Android lands on, so there is one code path.
  Qwen3-TTS only earns its place here if desktop voice quality becomes a
  requirement of its own; 2 GB is unremarkable on desktop (gemma-4 is 2.41 GB).
- **Either way, Phase 0 first.** System TTS on both platforms costs a day, has no
  licence exposure at all, and becomes the fallback path when a model is missing.

The C# ONNX driver, once written for one of these, is largely reusable for the
other — it is the same prefill/decode/KV-cache shape.

### Why Supertonic on Android

- Korean is one of 31 supported languages; the sherpa-packaged int8 build is
  `opensource-multilingual`.
- ~99 M parameters total, int8 — in the same weight class as the ASR models
  already shipped here, and far below the 0.7–2 B TTS models.
- **sherpa-onnx supports it first-class**: `OfflineTtsSupertonicModelConfig`
  exists in the C API and the C# bindings, with fields matching the published
  files one-to-one (`duration_predictor`, `text_encoder`, `vector_estimator`,
  `vocoder`, `tts.json`, `unicode_indexer.bin`, `voice.bin`).
- Prebuilt native libraries exist for both targets in release **v1.13.4** —
  `sherpa-onnx-v1.13.4-android.tar.bz2` (43 MB, jniLibs) and win-x64 shared
  libraries. Native name is `sherpa-onnx-c-api`, so Unity sees
  `sherpa-onnx-c-api.dll` / `libsherpa-onnx-c-api.so`.
- The C# surface is small: `new OfflineTts(config).Generate(text, speed, speakerId)`
  returns samples + sample rate, and `GenerateWithCallback` streams chunks —
  enough to start playback before synthesis finishes.

### Two risks worth stating plainly

1. **Upstream is archived.** Supertone posted on 2026-07-23 that the Supertonic
   repository will be archived with no further development or official support,
   and Voice Builder closes 2026-08-31. The weights stay downloadable and
   sherpa-onnx keeps maintaining the *runtime*, so this is a "no new versions"
   risk, not a "stops working" risk. Mirror the model files into our own storage
   rather than depending on the upstream repo staying up.
2. **OpenRAIL-M is not a plain open licence.** Its use restrictions (a)–(m) ban
   illegal use, harming minors, disinformation, impersonation/deepfakes,
   undisclosed machine-generated content, automated decisions with legal effect,
   discrimination, medical advice, and law-enforcement/justice profiling. There
   is **no military or defence restriction** in the list, and a drone HUD reading
   status aloud is not any of the above. The clause that actually touches us is
   (e): machine-generated audio must be disclosed as such. **This needs a legal
   sign-off on the customer side, not an engineering decision** — if it comes
   back "permissive licences only", switch to MeloTTS-Korean (MIT weights) and
   budget the ONNX export plus a Korean G2P port.

Not yet verified — flagged rather than assumed:

- Nobody has listened to Supertonic Korean output. Quality is claimed, not heard.
- No RTF measurement on kona (SM8250). The ASR encoders in this project run
  0.03–0.8 s per clip on desktop CPU; a 99 M TTS should be real-time on device,
  but that is an expectation, not a number.
- Whether the Android device has a Korean system TTS voice installed — the device
  was not attached when this was written (`adb devices` empty).
- Some sherpa release artifacts are built **`-no-tts`**. Pick a TTS-enabled build
  when downloading.

## Plan

### Phase 0 — system TTS baseline — **built 2026-07-27**

Shipped in `Packages/com.leuconoe.litert-lm-unity/Runtime/`:

| File | What |
| --- | --- |
| `ILiteRtLmTts.cs` | backend-agnostic interface: `Speak(text, language, onComplete)` coroutine, `Stop()` for barge-in, `IsAvailable`, `LiteRtLmTtsResult` |
| `LiteRtLmAndroidSystemTts.cs` | `android.speech.tts.TextToSpeech` via `AndroidJavaObject`; init listener as `AndroidJavaProxy`, `isLanguageAvailable` checked so a missing Korean voice is reported instead of failing silently |
| `LiteRtLmWindowsSapiTts.cs` | SAPI through a PowerShell sidecar → WAV → `AudioSource`; text staged as UTF-8 so Korean survives the console |
| `LiteRtLmSystemTts.cs` | picks the backend for the current platform; no-op stand-in elsewhere |

Verified:

- `Tools/Windows/Invoke-LiteRtLmRuntimeCompileCheck.ps1` (new) compiles the Runtime
  under both define sets — editor/Windows and Android player — in seconds, without
  opening Unity. It immediately caught a field whose type lived inside a
  `#if UNITY_ANDROID` block; that only breaks the editor, so a device-only check
  would have missed it. Both passes green.
- End-to-end on Windows, by round-trip rather than by ear: `Microsoft Heami
  Desktop` synthesized *"고도 백 미터로 상승합니다. 배터리 잔량 칠십 퍼센트."* in
  **49 ms for 5.82 s of audio (≈119× real-time)**, and feeding that WAV back
  through `Run-WhisperTfliteWindows.ps1` (whisper-base i8) returned
  *"고도 100m로 상승합니다. 배터리 자량 70%"* — numerals normalized as Whisper
  always does, one character misheard. The Korean is intelligible to an ASR model,
  which is the objective form of "it works".

Sample scene, generated (not hand-authored) and run:

- `Samples~/TestScenes/Runtime/Tts/LiteRtLmTtsTestRunner.cs` + generator entry
  `LiteRT-LM/Scenes/Generate/TTS Test Scene`, registered in build settings and
  enabled so the navigator walks into it. Language toolbar, five Korean and five
  English preset lines, editable text, Speak / Stop / speak-every-preset, and a
  results log showing backend, seconds and WAV path.
- Editor play-mode run through MCP: backend `Windows SAPI`, `available=True`,
  spoke *"고도 백 미터로 상승합니다. 배터리 잔량 칠십 퍼센트."*, status
  `Done in 7.01s`, WAV written to `%TEMP%\litertlm-tts\`. Screenshot:
  `Builds/Logs/SceneShots/tts-scene-play.png`.
- 7.01 s against 5.82 s of audio — the extra ~1.2 s is PowerShell process start.
  Acceptable for a baseline; a persistent sidecar would remove it if the system
  voice ever becomes more than a fallback.
- Generator fix found while doing this: newly registered test scenes were added
  to build settings **disabled**, so they existed but the navigator skipped them.

### The device has no system voice at all

**METALENSE2 (kona) is an AOSP build with no TTS engine, and one cannot be
installed** (user-confirmed, 2026-07-27). Phase 0 therefore covers Windows only,
and the Android leg of the fallback plan is gone: `LiteRtLmAndroidSystemTts` will
report unavailable on the target and stays in the tree for other devices.

This makes the engine decision blocking rather than deferrable — **on the device
there is no voice at all until a bundled-model backend ships**. It also removes
"system TTS as the safety net when the model is missing" from the design; the
neural backend has to be the primary path, not the upgrade.

### Phase 0 — original scope (for reference)

Ship a working voice output immediately, with zero model bytes and zero licence
exposure, and use it as the fallback when the neural engine is unavailable.

- `ILiteRtLmTts` in `Packages/…/Runtime/`, mirroring the existing ASR client
  split: `Speak(text, lang)`, `IsAvailable`, cancellation, and a WAV/PCM output
  path so Unity can play it through an `AudioSource`.
- Windows: shell out to SAPI (`Microsoft Heami`) writing a WAV, then play it —
  the same subprocess pattern `LiteRtLmWindowsCliClient` already uses. Unity's
  Mono profile cannot reference `System.Speech` directly.
- Android: `android.speech.tts.TextToSpeech` through `AndroidJavaObject`. No AAR
  change, no native build. Must check `isLanguageAvailable(Locale.KOREAN)` and
  degrade honestly when the voice data is missing.
- Gate: Korean sentence spoken on desktop **and** on 46a880a0.

### Phase 1a — Supertonic proven on the desktop, 2026-07-27

The model runs, speaks Korean, and is fast. Measured, not claimed:

- Package: `Assets/StreamingAssets/TTS/supertonic-int8/` (92 MB, gitignored like
  every other model here), driven by `Tools/Windows/Run-SupertonicTts.ps1` +
  `Tools/Windows/TtsBench/supertonic_tts.py` (sherpa-onnx 1.13.4, CPU, 4 threads).
- **RTF 0.034–0.044 — 23–29× real-time** on the 7950X, 44.1 kHz output, across
  five Korean lines of 2.5–4.7 s. Synthesis of a 4.65 s utterance takes 174 ms.
- Quality judged by round-trip ASR rather than by ear, using the accuracy-best
  tier on this project's device ledger (`acft_turbo_5s_drq`), with the Windows
  system voice as the control:

| Reference line | Supertonic → heard | SAPI (Heami) → heard |
| --- | --- | --- |
| 고도 백 미터로 상승합니다. | 고도 100m로 상승합니다. ✅ | 고도 100m로 상승합니다. ✅ |
| 배터리 잔량 칠십 퍼센트. | 배터리 70% — *잔량* dropped | 배터리 잔량 70% ✅ |
| 임무 지점에 도착했습니다. 촬영을 시작합니다. | ✅ exact | ✅ exact |
| 경고. 강풍이 감지되었습니다. 고도를 낮춥니다. | ✅ exact | 방풍 ✗ (강풍) |
| 귀환을 시작합니다. 예상 소요 시간 삼 분. | ✅ exact | 비환 ✗ (귀환) |

**Supertonic 4/5 clean vs the system voice 3/5.** Numerals normalize to digits in
every case — that is Whisper's behaviour, not a synthesis defect.

Worth recording because it nearly caused a wrong conclusion: judged with
`whisper_base_30s_i8` instead, Supertonic scored 2/5 and looked mediocre. The
extra errors were the ASR's, not the TTS's. Always judge synthesis with the
accuracy-best tier; `Run-SupertonicTts.ps1 -RoundTrip` now defaults to it and
takes `-AsrModel` to override.

### Phase 1c — Supertonic on LiteRT (the actual target)

Running TTS on **LiteRT** rather than adding onnxruntime as a third runtime is the
goal: the LLM and ASR paths already run on LiteRT on both Windows and Android, so
a converted Supertonic reuses the runtime, the packaging and the AAR bridge
instead of doubling them.

Supertonic is convertible in principle because it is **not autoregressive** — four
plain graphs, no KV cache:

```
duration_predictor(text_ids, style_dp, text_mask)                -> duration
text_encoder(text_ids, style_ttl, text_mask)                     -> text_emb
vector_estimator(noisy_latent, text_emb, style_ttl, text_mask,
                 latent_mask, current_step, total_step)          -> latent   ×N steps
vocoder(latent)                                                  -> wav
```

(Pipeline taken from the MIT reference `py/helper.py`, vendored as
`Tools/Windows/TtsBench/supertonic_helper.py`; `vector_estimator` is the one that
runs per flow-matching step and therefore dominates both size and time.)

Tooling, all in the repo:

| File | Role |
| --- | --- |
| `Tools/Windows/Convert-SupertonicToLiteRt.ps1` | one entry point: `-Bootstrap`, `-Describe`, `-Convert`, `-Run -RoundTrip` |
| `Tools/Windows/TtsBench/convert_supertonic_to_tflite.py` | onnx2tf per graph, output-checked against onnxruntime (`-cotof`), writes a conversion report |
| `Tools/Windows/TtsBench/quantize_supertonic_tflite.py` | ai-edge-quantizer with this project's recipes (`dynamic_wi8_afp32`, `dynamic_wi4b64_afp32`; never wi4c), per-graph tiers |
| `Tools/Windows/TtsBench/supertonic_litert.py` | the pipeline on LiteRT, binding inputs by name then by shape, resizing per utterance, reporting per-stage timings |

Conversion venv is separate (`TtsBench/.venv-convert`, Python 3.12) because
TensorFlow and onnx2tf are heavy and must not disturb the ASR bench venv.

**Status: working, 2026-07-27.** Supertonic runs end to end on LiteRT and speaks
intelligible Korean. All four graphs convert, all four match onnxruntime, and the
synthesized audio passes the same round-trip ASR gate as the onnxruntime baseline.

| Graph | fp32 tflite | fp16 tflite | LiteRT vs onnxruntime |
| --- | ---: | ---: | --- |
| duration_predictor | 1.6 MB | 0.9 MB | max abs 2.4e-07 |
| text_encoder | 26.3 MB | 13.4 MB | max abs 3.7e-06, corr 0.9999999999995 |
| vector_estimator | 126.6 MB | 63.6 MB | max abs 3.3e-05, corr 0.9999999999924 |
| vocoder | 96.8 MB | 48.4 MB | max abs 3.7e-05, **corr 0.9999999995** |

Compared stage by stage at a fixed latent seed, because the flow-matching loop
feeds its own output back in and a divergence would compound
(`compare_supertonic_runtimes.py`, report in
`Builds/Logs/TtsSupertonic/litert-vs-onnx-full.json`). Worst relative error over
the whole pipeline is 1.9e-04 — fp32 round-off, not a different computation.

**End-to-end on LiteRT**, *"고도 백 미터로 상승합니다. 배터리 잔량 칠십 퍼센트."*,
8 flow steps, CPU, 4 threads:

| | |
| --- | ---: |
| audio produced | 4.54 s |
| synthesis | 2.28 s → **RTF 0.50** |
| ‑ duration_predictor | 0.010 s |
| ‑ text_encoder | 0.119 s |
| ‑ vector_estimator | 1.777 s (0.222 s × 8 steps) |
| ‑ vocoder | 0.371 s |
| model load | 0.164 s |

Round-trip ASR (`acft_turbo`): **"고도 100m로 상승합니다. 배터리 잔량 70%"** — every
word correct, numerals normalized to digits as Whisper always does. Same result as
the onnxruntime baseline, so the conversion costs nothing audible.

### Performance: what was tried, what worked

RTF 0.50 against onnxruntime's 0.037. Four things were measured rather than
assumed, and the first three did not work:

| Attempt | Result |
| --- | --- |
| More threads | Nothing: 176 ms at 4 threads, 173 ms at 8. |
| fp16 tflite (63.6 MB, half the size) | **Unusable** — kernels reject fp16 inputs (`batch_matmul.cc:342`, `conv.cc:363`); it needs a fp16-capable delegate. |
| int8 dynamic-range quantization (33.3 MB) | **10× slower**, 1,844 ms vs 183 ms. Without a delegate the integer kernels dequantize per op. Good for size, disastrous for speed. |
| Fewer flow-matching steps | Works, linear in the step count. See below. |
| **Bucketed (fixed-shape) conversion** | **Works — 6.6× faster.** See below. |

#### Bucketed conversion: XNNPACK attaches, and the output is identical

On the dynamic graphs XNNPACK never attaches: it fails at `vector_estimator` node
2055 with "failed to reshape runtimeNode". Resizing *before* the first
`allocate_tensors`, so the delegate would prepare at the final shape, does not
help either — the cause is that onnx2tf preserves the dynamic-shape arithmetic,
leaving **145 SHAPE, 147 RESHAPE and 248 SLICE** ops in a 2,053-node graph.

Converting at fixed shapes (`onnxsim --overwrite-input-shape`, via
`--text-len/--latent-len`) folds most of that away — 55 SHAPE ops in 1,665 nodes —
and **the delegate attaches**:

| `vector_estimator`, one step | |
| --- | ---: |
| dynamic graph, reference kernels | 183 ms |
| bucketed graph, reference kernels | 292 ms |
| **bucketed graph, XNNPACK** | **28 ms** |

Correct as well as fast: max |diff| **1.4e-05** against onnxruntime,
correlation 0.9999999999994.

**Two graphs do not survive the rewrite.** `duration_predictor` and
`text_encoder` convert without complaint and then fail to allocate
(`transpose perm 4 != 5`) — onnxsim leaves a rank-5 transpose behind a rank-4
perm, in the embedding lookup both share. They are also the cheap half (10 ms and
119 ms), while the two that *do* convert are the expensive half, so the pipeline
runs mixed: `vector_estimator` and `vocoder` bucketed, the other two from the
dynamic build (`BUCKETED_UNSUPPORTED` in `supertonic_litert.py`, and
`--fallback-tflite-dir`).

End to end, *"고도 백 미터로 상승합니다."*, 4 steps, CPU, 4 threads:

| | dynamic | **bucketed mixed** | speed-up |
| --- | ---: | ---: | ---: |
| RTF | 0.893 | **0.135** | **6.6×** |
| synthesis | 2.12 s | 0.32 s | |
| vector_estimator / step | 0.361 s | 0.032 s | 11× |
| vocoder | 0.556 s | 0.074 s | 7.5× |

*(The dynamic column is padded to the same buckets so the comparison is like for
like; unpadded it is 0.47.)*

**Output is unchanged**: rendered from the same seed with the same padding, the
bucketed pipeline differs from the dynamic one by log-mel L1 **0.0009** at mel
correlation **1.00000**. Round-trip ASR is correct. So the 6.6× costs nothing.

This also matters because it makes the earlier finding reusable: the reason
int8 was 10× slower was the missing delegate, so quantization is worth re-testing
on the bucketed graphs, where the delegate is present.

#### The bucket ladder

A fixed shape means the text must be padded to a size that was converted, and
text longer than the bucket is rejected rather than truncated — *"고도 백 미터로
상승합니다. 배터리 잔량 칠십 퍼센트."* needs 69 ids and does not fit 64. So the
ladder is 64/128/256 and the runner picks the smallest bucket that fits
(`--bucket-root`, `bucket_dirs`/`choose_bucket`).

Selection has to happen *mid-pipeline*, which shapes the call order: the bucket
depends on the latent length, the latent length comes from `duration_predictor`,
and only then can the text be padded for `text_encoder` and the two bucketed
graphs. That falls out naturally because `duration_predictor` is one of the two
dynamic-shape graphs anyway.

Measured across the ladder, 4 steps:

| Text ids | Latent | Bucket | Audio | Synthesis | RTF |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 39 | 35 | 64 | 2.40 s | 0.27 s | **0.111** |
| 69 | 66 | 128 | 4.54 s | 0.44 s | **0.096** |
| 112 | 115 | 128 | 7.95 s | 0.47 s | **0.060** |

Longer utterances are *more* efficient — the per-call overhead amortizes — so RTF
improves with sentence length, from 9× to 17× real-time.

#### Quantization: weight-only, not dynamic

int8 was 10× slower on the dynamic graphs purely because there was no delegate. On
the bucketed graphs the question reopens, and the answer turns on *which kind* of
int8. Measured at bucket 128, all against the fp32 rendering:

| Configuration | Size (ve + voc) | RTF | mel corr | Verdict |
| --- | ---: | ---: | ---: | --- |
| fp32 both | 222.3 MB | 0.090 | — (reference) | |
| dynamic i8 both | 57.7 MB | 0.086 | **0.694** | rejected |
| dynamic i8 ve, fp32 voc | 129.6 MB | 0.107 | 0.990 | good, but heavy |
| dynamic i8 ve, weight-only voc | 57.7 MB | 0.103 | 0.980 | |
| **weight-only int8 both** | **57.7 MB** | 0.107 | **0.983** | **ship this** |

**The vocoder is the quantization-sensitive graph, and it is the *activations*
that matter.** `dynamic_wi8_afp32` quantizes activations per invocation, and on the
graph that emits the waveform that costs more than halving the flow steps does
(0.694 against 0.984). `weight_only_wi8_afp32` leaves activations in fp32 and keeps
**0.983 at the same 3.9× weight saving** — so weight-only wins outright, and using
one recipe for both graphs is simpler than mixing.

fp16 is not an option: onnx2tf's `_float16.tflite` converts the *inputs* to fp16
too, and the Conv/BatchMatMul kernels require fp32/int8 inputs
(`conv.cc:363`). ai-edge-quantizer has no fp16 recipe.

#### Deployed

`Tools/Windows/Deploy-SupertonicLiteRt.ps1` stages the ladder into
`Assets/StreamingAssets/TTS/supertonic-litert/` — **201.7 MB total**:

| Part | Size |
| --- | ---: |
| `dynamic/duration_predictor` + `dynamic/text_encoder` (fp32) | 27.9 MB |
| `st-b64`, `st-b128`, `st-b256` × (`vector_estimator` + `vocoder`), weight-only int8 | 3 × 57.7 MB |
| `assets/` — `tts.json`, `unicode_indexer.json`, voice style | 0.7 MB |

Verified from the staged copy, `-Verify`: RTF **0.056 / 0.097 / 0.132** for
7.95 s / 4.54 s / 2.40 s of audio, round-trip ASR correct on all three.

Per-graph cost at each bucket, for sizing future decisions
(`bench_supertonic_graph.py`):

| Bucket | `vector_estimator` / step | `vocoder` |
| ---: | ---: | ---: |
| 64 | 20 ms | 43 ms |
| 128 | 41 ms | 83 ms |
| 256 | 48 ms | 148 ms |

Both scale with length, so a coarser ladder trades size for latency on short
utterances. The three rungs cost 173 MB of the 202; dropping to two (128/256)
would save 58 MB and slow the shortest utterances by roughly 60 ms.

**Note for Android:** the XNNPACK finding is desktop-specific. The AAR reaches
LiteRT's own GPU/OpenCL accelerator, a different code path, so device numbers must
be measured rather than extrapolated — but the bucketing that unlocked the
delegate here is likely to help there too, for the same reason.

### Flow-matching steps: the setting that actually pays

`vector_estimator` runs once per step and is ~78 % of synthesis, so RTF tracks the
step count almost linearly. Judged two ways — round-trip ASR for intelligibility,
log-mel L1 against the 8-step rendering for the artefacts ASR cannot hear
(`Optimize-SupertonicLiteRt.ps1`, `spectral_distance.py`):

| Steps | RTF | Synth (4.54 s audio) | log-mel L1 vs 8-step | mel corr | Transcript |
| ---: | ---: | ---: | ---: | ---: | --- |
| 8 | 0.519 | 2.36 s | — | — | correct |
| 6 | 0.435 | 1.98 s | 0.133 | 0.9944 | correct |
| **4** | **0.332** | **1.51 s** | 0.245 | 0.9838 | correct |
| 3 | 0.294 | 1.34 s | 0.328 | 0.9701 | correct |
| 2 | 0.256 | 1.16 s | 0.407 | 0.9574 | correct |

Every setting transcribes perfectly, which is exactly why the spectral column
matters — ASR is insensitive to the degradation. **4 steps is the recommended
default**: 1.6× faster than 8, mel correlation still 0.984. Use 6 where quality
matters more than latency; 2 changes the audio audibly (corr 0.957) for only a
further 1.3×.

Four traps, each of which produced a plausible-looking wrong answer first:

1. **onnx2tf rewrites 3-D tensors NCW → NWC, per input, not uniformly.** Feeding
   ONNX-shaped tensors gave `Given shapes, [1,64,39] and [1,39,1], are not
   broadcastable`. Transposing everything then broke `text_encoder` instead,
   because `style_ttl` is *not* rewritten while `text_mask` is.
2. **Forcing the layout back with `-kat` produces an invalid graph** — it
   allocates into `[1,1,1,2] vs [1,4,64,128] are not broadcastable`. Do not use
   it here; let the converter rewrite, and permute at the call site.
3. **Dynamic axes make the layout unrecoverable.** With every variable dimension
   at 1 there is nothing to match against, and a wrong guess is silent: duration
   came out 1.73 instead of 2.52 — a plausible number, not an error. Converting
   at **fixed shapes** makes the dimensions concrete, so the permutation can be
   derived by matching them (`Tools/.../supertonic_litert.py::_permutation`) and
   the conversion report records the ONNX shapes to match against.
4. **XNNPACK cannot re-prepare after `resize_tensor_input`** ("failed to reshape
   runtimeNode number N"). The runner falls back to the reference kernels, but
   that gives up the delegate that makes tflite fast — another reason the
   shipping build wants fixed, bucketed shapes rather than resizing.

Also: `onnx2tf -b` and `-ois` are mutually exclusive, and `-ois` had no effect on
these graphs at all — shapes are pinned in the ONNX itself instead
(`freeze_onnx_shapes`, which rewrites only `dim_param` axes so it cannot mangle
dimensions the model already fixes; an earlier version that took whole shapes got
`style_dp` wrong, [1,16,8] when the graph says [1,8,16]).

The layout is recorded as data, not rediscovered: `TRANSPOSED_INPUTS` in
`supertonic_litert.py` lists which inputs are NWC per graph, derived from a
fixed-shape conversion's input details and then confirmed numerically. Outputs
always come back in the ONNX layout.

Remaining for the LiteRT path:

1. **Fixed-shape buckets** to get XNNPACK back (the 13× gap above). Freezing via
   `onnx.shape_inference` produced a graph that fails to allocate
   (`[1,1,1,2] vs [1,2,65,130] not broadcastable`), so try `onnxsim` with
   `--overwrite-input-shape` instead.
2. **Quantization** — `quantize_supertonic_tflite.py` is written and takes
   per-graph tiers; `vector_estimator` is the one worth i8/i4, being 127 of the
   251 MB and 78 % of the time.
3. **The C# text front end is done and verified**:
   `Runtime/LiteRtLmSupertonicText.cs` reproduces the reference NFKD → indexer
   pipeline **exactly** — 8/8 cases byte-identical against Python
   (`Test-SupertonicTextParity.ps1`), including Korean jamo decomposition, curly
   quotes, bracket-to-space and the auto-period rule.
4. `text_ids` arrives as **int64**; force it to int32 before the Android leg.

#### Where to split the pipeline between C# and native

Supertonic's front end normalizes text with Unicode **NFKD** before mapping code
points through `unicode_indexer.json`. Doing NFKD in C++ means pulling in ICU,
which is exactly the kind of dependency this AAR does not want. C# has it built
in (`string.Normalize(NormalizationForm.FormKD)`).

So the split is:

| Stage | Where | Why |
| --- | --- | --- |
| NFKD, emoji/symbol cleanup, code point → id, masks | **C#** | `Normalize` is in the BCL; keeps ICU out of the AAR; the tables are plain JSON |
| duration_predictor / text_encoder / vector_estimator ×N / vocoder | **native, LiteRT** | one JNI call per utterance instead of one per flow step; the latent never crosses the boundary |
| WAV → `AudioSource` | C# | same as the SAPI backend already does |

That keeps the JNI surface to a single `nativeRunSupertonicTts(textIds, textMask,
stylePath, steps, speed)` returning PCM, mirroring how the whisper ASR entry point
is shaped today. On Windows the same native code would be reached through
`libLiteRt.dll` — which is the piece the earlier Whisper-on-Windows estimate
priced, so it stays a known quantity rather than a new unknown.

### Phase 2 — into Unity and onto the device

Written 2026-07-28. The pieces below exist and compile; the device run is the
gate that has not passed yet.

**C# backend** — `Runtime/LiteRtLmSupertonicTts.cs`, an `ILiteRtLmTts` beside the
system voices, so the sample scene swaps between them with a toolbar and nothing
else changes. It owns the two decisions that belong on the managed side:

- the text front end (`LiteRtLmSupertonicText`, already 8/8 byte-identical to the
  Python reference), because step one is Unicode NFKD and doing it natively would
  pull ICU into the AAR;
- the bucket choice, because a fixed-shape graph needs the text padded to a size
  that was converted. `ChooseBucket` takes the smallest of 64/128/256 that fits,
  and the native side re-checks against the graph it actually loaded rather than
  trusting the caller.

Everything else is one bridge call per utterance: `RunSupertonicTts` on
`LiteRtLmUnityClient`. The flow-matching loop stays native, so neither the latent
nor the PCM crosses JNI — a 5 s utterance is ~440 k floats, and marshalling that
per step would cost more than the synthesis.

**AAR** — `RunSupertonicTtsToJson` in the JNI, reached through
`UnityLiteRtLmBridge.runSupertonicTts`. Two details worth keeping:

- Inputs and outputs are bound through LiteRT's **named-map `Run`**, not
  positionally. The converter reorders signature inputs — `duration_predictor`
  comes out as `(style_dp, text_ids, text_mask)`, not the ONNX order — and a
  positional bind would feed the wrong tensor without erroring. Names survive
  conversion, which is what makes this possible:

  | Graph | Inputs | Output |
  | --- | --- | --- |
  | duration_predictor | style_dp, text_ids, text_mask | `duration` |
  | text_encoder | style_ttl, text_ids, text_mask | `text_emb` |
  | vector_estimator | current_step, latent_mask, noisy_latent, style_ttl, text_emb, text_mask, total_step | `denoised_latent` |
  | vocoder | latent | `wav_tts` |

- The NCW→NWC rewrite is applied natively for `noisy_latent` and `text_emb`, and
  deliberately *not* for the masks — `[1, 1, len]` has the same bytes either way.
  Outputs come back in the ONNX layout.

The latent is drawn from `std::mt19937` with an explicit seed so a device run can
be reproduced against the desktop bench; without that, a conversion fault and a
different noise draw look identical.

The patch was regenerated (`Tools/UnityAar/litert-lm-unity-aar.patch`, 17 files
as before plus the TTS code; the previous revision is kept as `.patch.take8`) and
the AAR is building as take9.

**Device**: 46a880a0 (METALENSE2, kona, Android 12) — arm64-v8a, 98 GB free on
`/data`, 7.9 GB RAM against a 202 MB model set. Backend is CPU for this first
pass: bucketing is what let the *desktop* delegate attach, and whether the Android
GPU/OpenCL accelerator takes these graphs is a measurement to make, not an
assumption to ship.

#### First device run — the harness works, two API mistakes found

`Tools/Windows/Run-LiteRtLmAndroidTtsSmokeTest.ps1` against a 261 MB APK
(`LiteRT-LM/Build/Android/Build TTS Smoke Test APK`, which packages only
`TTS/supertonic-litert`). Everything up to inference is correct: the models stage
out of the APK, and the **bucket chooser picks the right rung on device** — 39 ids
→ 64, 69 → 128, 112 → 128.

All six synthesis attempts then failed with one error:

```
writing 'text_ids': TensorBuffer host memory buffer size is
smaller than the given data size, 8 vs 256
```

Two causes, both in the JNI rather than in the models:

1. **`text_ids` is int64, not int32.** The ONNX dtype survived conversion. Widening
   happens in the JNI now (the Java side keeps `int[]`, which is the natural type
   for a caller).
2. **The dynamic-shape graphs need `ResizeInputTensor`.** `duration_predictor` and
   `text_encoder` are converted with dynamic axes, and the converter collapses
   every variable dimension to 1 — so their `text_ids` buffer is literally 8 bytes
   (one int64) until resized. The Python bench called `resize_tensor_input`; the
   C++ path has `CompiledModel::ResizeInputTensor(signature, input_name, dims)`,
   which was simply not being called. The bucketed graphs are already the right
   size, so the call is made for both and a rejection is ignored.

The 8-vs-256 arithmetic is what identified this: 256 = 64 ids × 4 bytes at the
*int32* width being written, against 8 bytes of *un-resized int64* buffer — two
independent bugs visible in one message.

Worth noting for the next device run: the app must have window focus. With the
notification shade pulled down, Unity never resumes and the runner produces no
status file at all, which looks like a hang rather than a paused app
(`dumpsys window | grep mCurrentFocus` shows it; `cmd statusbar collapse` fixes
it). `Run-LiteRtLmAndroidTtsSmokeTest.ps1` now wakes the screen, dismisses the
keyguard, collapses the shade, and verifies focus before waiting.

#### Second device run — `text_ids` fixed, one more input off by three floats

```
writing 'style_dp': TensorBuffer host memory buffer size is
smaller than the given data size, 512 vs 524
```

512 bytes is correct (128 floats = 8 × 16). 524 is 131 floats — three too many,
and the three are the entries of `dims`. The voice style files are
`{"data": [...], "dims": [1, 8, 16], "type": "..."}`, and the loader flattened the
whole entry rather than `data`, so the shape numbers were appended to the tensor.
`nlohmann::json::flatten()` will happily do that: it walks every leaf, and there
is no error to catch.

Fixed by reading `data` explicitly (falling back to the entry itself if a future
file stores a bare array).

#### Third device run — the resize was being rejected, silently

```
writing 'text_mask': ... buffer size is smaller than the given data size, 4 vs 256
```

4 bytes is one float: `text_mask` was never resized, while `text_ids` clearly had
been. Reading the tflite signatures explains it — **only the batch dimension is
dynamic in these graphs**:

| Input | `shape_signature` |
| --- | --- |
| `text_ids` | `[-1, -1]` |
| `text_mask` | `[-1, 1, 1]` |
| `style_dp` | `[-1, 16, 8]` |

So a *strict* resize of the mask length is refused, and the JNI was discarding the
`Expected<void>` with `(void)` — the refusal never surfaced. The Python bench had
passed `strict=False` all along; the C++ equivalent is
`ResizeInputTensorNonStrict`. Failures are now written into the result JSON
(`resizeTextIdsError` / `resizeTextMaskError`) instead of being dropped.

That signature table also caught a second, quieter bug: `style_dp` is `[-1, 16, 8]`
against the ONNX `[batch, 8, 16]`, i.e. NWC. Its byte count is 128 floats either
way, so feeding the ONNX layout **passes every size check and produces wrong
audio**. The desktop `TRANSPOSED_INPUTS` map had it; the JNI port only carried
`noisy_latent` and `text_emb` across.

#### Fourth device run — the desktop XNNPACK finding, again

Every input write passed. The failure moved to inference, and logcat gave the
reason verbatim:

```
E tflite: XNNPack delegate failed to reshape runtime
E tflite: Node number 843 (TfLiteXNNPackDelegate) failed to prepare.
```

Word for word the desktop failure. **Android LiteRT's CPU path is XNNPACK too**, so
the graphs that need a per-utterance resize break there for the same reason — the
delegate cannot re-prepare. On the desktop this was worked around by dropping to
the reference kernels; the JNI was not doing the equivalent.

LiteRT exposes it: `CpuOptions::SetKernelMode` with `kLiteRtCpuKernelModeBuiltin`
(built-in *optimized* kernels, not the slow reference path). Applied per graph,
which is exactly the desktop conclusion carried into the device code:

| Graph | Kernel mode | Why |
| --- | --- | --- |
| duration_predictor, text_encoder | **builtin** | need a resize, so XNNPACK is impossible; they are the cheap half (~10 ms + ~110 ms on desktop) |
| vector_estimator, vocoder | **XNNPACK** | fixed buckets, no resize — the delegate is what makes them fast (28 ms vs 183 ms per flow step) |

This is the same trade the desktop bench arrived at, now expressed in options
rather than in a fallback path.

#### Fifth device run — the kernel split works; output buffers were the next gap

logcat confirms both halves of the plan, on device:

```
compiling …/duration_predictor… (builtin kernels)
compiling …/vector_estimator… (xnnpack)
tflite: Replacing 1474 out of 1789 node(s) with delegate (TfLiteXNNPackDelegate)
tflite: Replacing  344 out of  356 node(s) with delegate
```

So the bucketing pays off on kona exactly as it did on the desktop — **XNNPACK
takes 1474 of 1789 nodes in `vector_estimator`** — and `duration_predictor` now
invokes successfully on the built-in kernels.

`text_encoder` then failed, and again the device named the cause:

```
E tflite: Custom allocation is too small for tensor idx: 2486
```

`CreateOutputBuffer` allocates from the size the model reports, and a **non-strict
resize does not update that size**, so the output buffer was the pre-resize one.
This also explains why `duration_predictor` was fine: its output is the scalar
`duration`, whose size does not depend on the text length, while `text_emb` is
`[1, embed, text_len]` and does.

Fixed by allocating the output of a resized graph explicitly with
`TensorBuffer::CreateManaged` at the shape we know it must be; `embed` comes from
the voice style (`style_ttl` is `[1, 50, embed]`).

Also tried and rejected: converting `text_encoder` at a fixed bucket to sidestep
the resize altogether. `onnx2tf -nodaftc 8` reproduces the rank-5-transpose
failure and `-nodaftc 2` trades it for `DEPTHWISE_CONV_2D` receiving 3 dimensions —
so the dynamic graph plus an explicitly sized output is the right path, not a
workaround.

### Sixth device run — it runs; the audio is wrong

**The pipeline completes on kona: 6 of 6 runs, real WAVs, no failures.**

| Sentence | ids | Bucket | Audio | RTF (cold → warm) | ve / step | vocoder | compile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 39 | 64 | 1.30 s | 0.484 → 0.433 | 0.060 s | 0.24 s | 0.65 s |
| 2 | 69 | 128 | 2.33 s | 0.554 → 0.513 | 0.124 s | 0.52 s | 0.64 s |
| 3 | 112 | 128 | 3.65 s | 0.317 / 0.325 | 0.112 s | 0.53 s | cached |

**RTF 0.32–0.55 — two to three times faster than real time on the device**, with the
model cache working (second run compiles in 0.000 s). Same shape as the desktop
result: longer utterances are more efficient.

**But the speech is wrong.** Round-tripping the device audio through the desktop
ASR:

| Said | Heard |
| --- | --- |
| 고도 백 미터로 상승합니다. | 저도 배는 잘하셨습니다 |
| 고도 백 미터로 상승합니다. 배터리 잔량 칠십 퍼센트. | 모두의 헤미터로 상자하래, 타다. |

It is fluent Korean-sounding speech carrying the wrong content — the signature of a
layout error, i.e. **the one class of bug that passes every size check**, exactly as
the earlier `style_dp` case warned. A quantitative handle: `duration_predictor`
returns 1.30 s on device where the desktop returns 2.38 s for the same sentence,
54 % of it.

Padding was ruled out rather than assumed: replaying the device's *padded* input on
the desktop gives 2.377 s against 2.398 s unpadded, so padding to a bucket is not
what changes the duration.

Rather than guess at the next candidate, the JNI now reports the **md5 of each
tensor as actually fed** (reusing the Md5 helpers already in that file), and
`Tools/Windows/TtsBench/dump_supertonic_input_md5.py` prints the desktop
equivalents in the same dtype and byte order:

```
idsMd5      6f627893d9ec6d029c4f82e7c3f65da7
styleDpMd5  de4c395ea751a66cd85cfb6ad0ac823c   (after the NWC transpose)
styleTtlMd5 2e3f4ea81f7bb015e447a9b15e78f5a4
textMaskMd5 5e63b4bba0c2f6173f402a208b4e2cbe
durationSeconds 2.3769   textEmbMd5 9d0e8f7d1636b8a23cff9514b84de911
```

One device run against those five values says which input diverges. A checksum is
the right instrument here precisely because the failure is invisible to size
checks.

### Seventh device run — found it, in one run

| Tensor | Desktop | Device | |
| --- | --- | --- | --- |
| `ids` | `6f627893…` | `6f627893…` | match |
| `textMask` | `5e63b4bb…` | `5e63b4bb…` | match |
| `styleDp` | `de4c395e…` | `acdae6cc…` | **differs** |
| `styleTtl` | `2e3f4ea8…` | `c438372a…` | **differs** |

`ids` matching is itself worth having: it re-verifies the C# front end in situ, not
just against the Python reference on a desktop. And because `styleTtl` is fed
without any transpose on both sides, the difference cannot be a layout mistake —
it is the **values**, i.e. the loader.

**Cause: `nlohmann::json::flatten()`.** It returns an *object* keyed by JSON
pointer, and iterating an nlohmann object walks keys in **string** order:

```
array order   : 0, 1, 2, 3, … 9, 10, 11
flatten order : 0, 1, 10, 11, 2, 3, … 9      ("/0/0/10" sorts before "/0/0/2")
```

So all 12,800 style values arrived permuted. Every size check passes — the byte
count is exactly right — and the model simply says something else. This is the
second silent failure of the session and, like the `style_dp` transpose, it could
not have been found by reading an error message.

Fixed by recursing over the arrays instead.

**Rule worth keeping: never use `flatten()` to read tensor data.** It looks like a
convenience for "give me all the leaf numbers" and it is, except that it does not
preserve order and says nothing when it reorders. The same call had already caused
the `512 vs 524` failure by sweeping in `dims`; two distinct bugs from one function.

### Working on the device, 2026-07-28

**Supertonic TTS runs correctly on METALENSE2 (kona, Android 12) through LiteRT.**
Every checksum matches the desktop and the audio transcribes back correctly.

| Tensor | Desktop | Device |
| --- | --- | --- |
| `ids` | `6f627893…` | `6f627893…` ✅ |
| `styleDp` | `de4c395e…` | `de4c395e…` ✅ |
| `styleTtl` | `2e3f4ea8…` | `2e3f4ea8…` ✅ |
| `textMask` | `5e63b4bb…` | `5e63b4bb…` ✅ |
| `duration` | 2.3769 s | **2.3769 s** ✅ (was 1.2972) |

Round-trip ASR on the audio pulled off the device:

| Said | Heard |
| --- | --- |
| 고도 백 미터로 상승합니다. | 고도 100m로 상승합니다. ✅ |
| 고도 백 미터로 상승합니다. 배터리 잔량 칠십 퍼센트. | 고도 100m로 상승합니다. 배터리 자량 70% |
| 경고. 강풍이 감지되었습니다. 고도를 낮춥니다. 귀환을 시작합니다. 예상 소요 시간 삼 분. | 경고, 강풍이 감지되었습니다. 고도를 낮춥니다. 귀환을 시작합니다. 예상소요 시간 3분. |

(`자량` is the whisper-base tier's own single-character error — the accuracy-best
tier reads the same audio correctly, as on the desktop.)

#### Device performance, 4 flow steps, CPU

| ids | Bucket | Audio | RTF cold → warm | ve / step | vocoder | compile |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 39 | 64 | 2.38 s | 0.265 → **0.248** | 0.064 s | 0.25 s | 0.65 s |
| 69 | 128 | 4.52 s | 0.279 → **0.261** | 0.118 s | 0.52 s | 0.64 s |
| 112 | 128 | 7.87 s | **0.150 / 0.150** | 0.113 s | 0.49 s | cached |

**RTF 0.15–0.27 — four to seven times faster than real time on the target
device**, with the model cache eliminating the ~0.65 s compile after the first
utterance. Longer utterances are more efficient, exactly as on the desktop.

Against the desktop (0.060–0.111) the device is roughly 2.4× slower, which for a
Snapdragon 865-class CPU against a 7950X is a reasonable ratio and leaves a wide
margin over real time.

The audio the device produced is kept at
`Builds/Logs/AndroidTtsSmoke/<run>-wav/` alongside the status log, so a later
change can be compared against it rather than re-derived.

#### GPU / OpenCL: measured, and it does not work

`Tools/Windows/Compare-SupertonicDeviceBackends.ps1` runs the device smoke test
once per backend and tabulates the result. `backend` selects the accelerator for
the **bucketed graphs only** — `duration_predictor` and `text_encoder` are resized
per utterance and a GPU delegate can no more re-prepare after a resize than
XNNPACK can, and they are the cheap half anyway.

| Backend | Verdict | Warm RTF (39 / 69 / 112 ids) |
| --- | --- | --- |
| **CPU (XNNPACK)** | SUCCESS 6/6 | **0.242 / 0.257 / 0.153** |
| GPU (fp32) | **FAILURE 0/6** | — |
| GPU_FP16 | **FAILURE 0/6** | — |

Both GPU modes fail at model compilation:

```
FAIL: vector_estimator_w8.tflite: Failed to compile model
```

and logcat gives the reason:

```
I tflite : Replacing 572 out of 1789 node(s) with delegate (LITERT_CL), yielding 2 partitions
I native : Initializing OpenCL-based API from graph.
E native : Failed to create litert::ml_drift::DelegateKernelLiteRt: INVALID_ARGUMENT:
           Shape mismatch: {bhwc, {256, 1, 1, 256}} vs {bhwc, {1, 1, 256, 256}}
E tflite : Restored original execution plan after delegate application failure.
```

The OpenCL delegate does claim 572 of 1,789 nodes, then its `ml_drift` model
builder rejects the graph on a **BHWC shape mismatch** — `{256,1,1,256}` against
`{1,1,256,256}`. That is the NCW→NWC rewrite from the ONNX conversion meeting a
GPU backend that assumes BHWC: the converted graph's 3-D tensors do not map onto
the layout the OpenCL path expects. tflite restores the CPU plan, but LiteRT still
reports the compile as failed, so nothing runs.

**Conclusion: CPU with XNNPACK is the shipping configuration on kona.** This is not
a tuning result to revisit — it is a layout incompatibility between onnx2tf's
output and the OpenCL delegate, so it would need a different conversion route
(one that keeps 4-D BHWC-friendly shapes) rather than different GPU options. Worth
noting the CPU path is already 4–7× real time, so there is no pressing need.

For the record, the CPU numbers reproduced across the two runs — 0.248/0.261/0.150
earlier, 0.242/0.257/0.153 here — so the measurement is stable to about ±2 %.

**What the first three runs have in common:** the size arithmetic in the message names
the cause outright — 8 vs 256 was "int32 into an un-resized int64 buffer", 512 vs
524 was "three extra floats from `dims`", 4 vs 256 was "not resized at all". None
needed a debugger. And the one bug that *cannot* announce itself this way is a
transpose, because the byte count is unchanged — which is the argument for keeping
the layout map as data and checking it against the signatures rather than
remembering it.

### Phase 1b — Supertonic through sherpa-onnx in Unity (2–3 days)

- Vendor the C# bindings (`scripts/dotnet/*.cs`, ~20 files) into the package, or
  reference the NuGet `org.k2fsa.sherpa.onnx` for desktop and vendor for Android.
- Native libs: `sherpa-onnx-c-api.dll` → `Runtime/Plugins/x86_64/`,
  `libsherpa-onnx-c-api.so` (arm64-v8a) → `Runtime/Plugins/Android/`. Set the
  Unity plugin import platforms; do not let the editor load the arm64 binary.
- Model into `Assets/StreamingAssets/TTS/supertonic-int8/`, following the
  existing per-model-subfolder convention, and staged into the APK the same way
  the ASR models are. Mirror the files to our own storage (upstream archived).
- Implement `LiteRtLmSherpaTtsClient : ILiteRtLmTts`. Use `GenerateWithCallback`
  so playback starts on the first chunk instead of after full synthesis.
- Test scene generated through the existing generator, not hand-authored, and
  added to `LiteRT-LM/Verify Sample Scenes` invariants.
- Gate: Korean and English on device, RTF and first-audio latency recorded in
  `docs/benchmarks/`, A/B against the Phase 0 baseline.

### Phase 2 — wire into the voice loop (0.5 day)

ASR → LLM → TTS end to end in one scene, with barge-in (stop speaking when the
microphone opens) and the machine-generated disclosure that OpenRAIL-M (e) wants.

**Total 3.5–4.5 days**, and Phase 0 alone already gives a usable feature on both
platforms in one day — so the risky part can be gated on a real listen to
Supertonic Korean before committing to it.

## Decisions needed before Phase 1

1. **Is OpenRAIL-M acceptable to the customer?** This single answer picks the
   engine: yes → Supertonic, 2–3 days; no → Chatterbox Multilingual q4f16 (MIT),
   1–2 weeks for the ONNX driver. Nothing else about the plan changes.
2. **Streaming or full-utterance?** Supertonic's C# API has a chunk callback;
   an autoregressive engine streams naturally. Command acknowledgements are short
   enough that it may not matter for the first release.
3. **Voice.** Supertonic int8 carries one `voice.bin` (v3 has 10 styles, F1–F5 /
   M1–M5); Chatterbox ships `default_voice.wav` and can clone from a 3-second
   clip. Pick the voice before recording any demo.

## Sources — Qwen3-TTS and Chatterbox

- [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) · [1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) · [0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) · [1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — streaming claims, vLLM support
- [sivasub987/Qwen3-TTS-0.6B-ONNX-INT8](https://huggingface.co/sivasub987/Qwen3-TTS-0.6B-ONNX-INT8), [elbruno/…-CustomVoice-ONNX](https://huggingface.co/elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX) — third-party exports
- [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) — MIT, 23 languages
- [onnx-community/chatterbox-multilingual-ONNX](https://huggingface.co/onnx-community/chatterbox-multilingual-ONNX) — q4/q4f16 builds, `default_voice.wav`

## Sources

- [supertone-inc/supertonic](https://github.com/supertone-inc/supertonic) — README, language list, archive notice
- [Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3) — assets, LICENSE (OpenRAIL-M)
- [csukuangfj2/sherpa-onnx-supertonic-tts-int8-2026-03-06](https://huggingface.co/csukuangfj2/sherpa-onnx-supertonic-tts-int8-2026-03-06) — int8 package
- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — C API/C# bindings, v1.13.4 release artifacts
- [sherpa-onnx TTS pretrained models](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/index.html) — Kokoro/Matcha/Kitten/VITS coverage
- [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) — `voices.json`, `ko_KR-kss-medium` model card
- [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) — Apache-2.0, language coverage
- [myshell-ai/MeloTTS](https://github.com/myshell-ai/MeloTTS) and [MeloTTS-Korean](https://huggingface.co/myshell-ai/MeloTTS-Korean) — MIT
