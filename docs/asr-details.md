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

---

## 전체 배포 라인업 — Android 기준 (2026-07-26, README에서 이동)

디바이스 46a880a0(Snapdragon 865 / 7.5 GB / Android 12) 실측 기준 순위입니다.
각 폴더에 티어 전용 `tokenizer.json`을 함께 배치해야 합니다(medium과
large-v3/turbo는 각자 전용). 전 티어 CER/WER/RTF 매트릭스는
[`benchmarks/asr-model-matrix.md`](benchmarks/asr-model-matrix.md).

| 용도 (device) | Model (`Assets/StreamingAssets/` 하위) | Size | Device 결과 |
| --- | --- | ---: | --- |
| 음성 명령 제1픽 | `ASR/whisper-base-acft-ko/acft_base_5s_drq.tflite` | 101 MB | 한국어 ACFT 5 s 윈도우 학습 — 정상 음량 명령 전부 exact, E2E 0.7–0.8 s (stock base 대비 ~3.5×, turbo-30s 대비 ~30× 빠름). 조용한 녹음은 turbo-acft로 폴백 |
| 음성 명령 정확도 폴백 | `ASR/whisper-turbo-acft-ko/acft_turbo_5s_drq.tflite` | 883 MB | **디바이스 5/5 유일 모델** (조용한 `볼륨 업` 구녹음 + `음량 증가` + 숫자 표기까지 전부 exact). 콜드 ~4 s / 웜 ~1.9 s |
| 장문 (>30 s) 전체 전사 | `ASR/qwen3-asr-0.6b/qwen3_asr_0.6b_5s_i8.tflite` | 794 MB | 5 s 청크 루프로 길이 무제한 — 98 s 오디오 20청크 완전 전사(4.2분, RAM 평탄). **유일한 장문 경로** |
| 문장 전사 최고 정확도 | `ASR/whisper-large-v3-turbo/whisper_large_v3_turbo_30s_i4.tflite` | 755 MB | 디바이스 게이트 3/3 — whisper 중 유일하게 볼륨 명령까지 인식. 매트릭스 종합 1위(8/9, CER 0.000). 단 클립당 ~21–24 s CPU (비실시간 배치용) |
| 균형(크기·속도·정확도) | `ASR/whisper-base/whisper_base_30s_i8.tflite` | 77 MB | 문장 한국어 CER 0.000, 긴 클립 ~2.7 s. 1.2 s 미만 초단클립은 디바이스에서 불안정(mel 수치 특성) |
| 초소형(영어 위주) | `ASR/whisper-tiny/whisper_tiny_30s_i8.tflite` | 41 MB | 영어 CER 0.000. 한국어 연도 오인 + 초단클립 불안정 |
| 정확도 레퍼런스 | `ASR/whisper-large-v3/whisper_large_v3_30s_i4.tflite` | 1148 MB | 문자 단위 완벽하나 turbo보다 3–7배 느림 — 비교 기준용 |
| 중간 티어 | `ASR/whisper-medium/whisper_medium_30s_i8.tflite` (i4: 664 MB) | 832 MB | 7/9 정확 — base와 turbo 사이 절충 |
| medium-acft-ko | `ASR/whisper-medium-acft-ko/` | 826 MB | 배치돼 있으나 **비권장** — turbo-acft보다 느리고 부정확 |
| tiny-acft-ko | `ASR/whisper-tiny-acft-ko/` | 46 MB | **한국어 명령 비권장** — 디바이스 1/4 exact(사이클 4 REJECT) |

⚠️ **whisper 30 s 모델에 30 s 초과 오디오를 직접 넣지 말 것** — 절단 + 토큰 캡 +
조기 종료 3중 실패. 장문은 qwen3 청크 경로 사용.

**대안 경로**: gemma-4 오디오 입력(멀티모달 LLM)으로도 전사 가능 — LLM이 이미
상주할 때 추가 모델 없이 **전사+펑션콜링을 한 턴에** 처리(디바이스 4.1 s).

### 모델 출처

whisper tiny/base는 [litert-community](https://huggingface.co/litert-community/whisper-tiny)
([base](https://huggingface.co/litert-community/whisper-base)); medium /
large-v3 / turbo 및 모든 i8/i4 티어는 **프로젝트 자체 양자화**(int4 최소 티어
정책, `dynamic_wi4b64_afp32` + 민감 스코프 i8 혼합 —
`External/community-release/`에 커뮤니티 공개용 사본과 매니페스트).
토크나이저는 [openai/whisper-*](https://huggingface.co/openai/whisper-tiny)
티어별. Qwen3-ASR은 공식 tflite + 프로젝트 JNI 포팅.

## VAD (음성 구간 검출) — Android 실행 기준

모든 ASR 경로가 `vadMode`를 지원합니다.

| 모드 | 내용 | 비용 |
| --- | --- | --- |
| `energy` (기본) | 적응 임계값 v2 — 300 ms 노이즈 캘리브레이션, 20th-pct 노이즈 플로어 +9 dB 온/6 dB 히스테리시스, 210 ms 행오버, 90 ms 프리롤, speech-only RMS 게인 | 0 (추가 모델 없음) |
| `ai` | Silero VAD v5 tflite (`ASR/silero-vad/silero_vad_16k.tflite`, 1.25 MB, MIT). whisper 입력에는 0.2 s 헤드 패드 필요 | 모델 1.25 MB |
| `off` | 전처리 없음 | — |

- 98 s / 31발화 스트레스 클립에서 **31/31 검출**, 디바이스↔데스크톱 경계 완전 일치
- 결과 JSON: `vadMode / vadModeUsed / speechSegments / trimmedSeconds / vadGain /
  speechRms (+vadError)`
- Unity 라이브 마이크(`LiteRtLmMicVadCapture`)는 동일 파라미터를 C#으로 미러링해
  자동 엔드포인팅 → 16 kHz WAV → 선택 모델 전사
- **저음량 0.79 s 클립은 VAD·게인으로 해결 불가**(16조합 실증) = 모델 용량 문제.
  해법은 티어 에스컬레이션(turbo-acft)

## 한국어 ACFT 5 s 모델 — 학습 배경과 수치 해석

stock whisper를 5초 짧은 컨텍스트에 그대로 넣으면 붕괴합니다(반복 폭주,
CER 1.1–24.9). futo ACFT 자기증류에 두 가지 보정(ctx 하한 250, 한70:영30
zeroth+fleurs 혼합)을 더해 자체 학습했습니다. 인코더는 30초 창 대비 **~12배**
빠릅니다. 모델 카드:
[leuconoe/whisper-acft-ko](https://huggingface.co/leuconoe/whisper-acft-ko).

게이트 결과(TTS 합성 40클립 홀드아웃, 5 s ctx 한국어 단문 CER):
turbo 0.182 · medium 0.208 · base 0.305 · tiny 0.457.

⚠️ **이 수치를 절대 품질로 인용하지 말 것.**

- edge-tts 합성 홀드아웃이라 **순위 지표로만 검증**됨(실녹음 대비
  Spearman ρ = 1.00, Pearson 0.99)
- 강한 베이스에서는 실제 오차를 **~2.8× 과대평가**
- 참조 문자열 평균 3.6자 → 마침표 1개가 CER +0.25~0.50인데 실제 매처는
  구두점을 무시(`LiteRtLmAsrTestRunner.cs`)
- **tiny는 한국어 음성 명령 비권장** — 실녹음 명령 CER 0.896, 디바이스 1/4
  exact(사이클 4 REJECT). 배포 티어는 base(1픽) / turbo(정확도 폴백)

상세 분석: [`benchmarks/asr-model-matrix.md`](benchmarks/asr-model-matrix.md)
Addendum 3, [`handoffs/asr-training-program-handoff.md`](handoffs/asr-training-program-handoff.md) §3–5.
