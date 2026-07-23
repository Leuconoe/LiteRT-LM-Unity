# ASR Model Matrix — Re-validation on 2026-07-23 Re-recorded Audio

Full re-validation ("벤치마크 재검수") of every deployed ASR model tier against the
**re-recorded** test clips in `Assets/StreamingAssets/TestAssets/Audio/`
(all 9 files replaced/added 2026-07-23 13:33–13:37). All prior transcript
validations used the old recordings and are superseded by this document.

**Second pass (same day)**: clip 2 was re-recorded at **13:57** (mislabel fix,
see clip 2 notes) and the clip-7 command was re-recorded as a **new** file
`volume-볼륨, 업.mp3` at **14:00** (louder/longer; the old file remains).
Affected rows and the summary were re-measured on the 8 deployed tiers.
`whisper-large-v3-turbo f32` and `qwen3-asr-0.6b i4` were **removed from
StreamingAssets** between the passes and could not be re-run; their rows are
first-pass values on the old recordings, marked †.

## Environment

- Date: 2026-07-23
- Host: Windows 11 x64, CPU only (`ai-edge-litert` Python interpreter,
  XNNPACK delegate, 8 threads). Desktop reference numbers — Android device
  timings will differ, but relative ordering between tiers holds.
- Whisper decode is greedy, full-sequence re-run per step (SEQ=128), matching
  the existing probe methodology; ms/step is therefore comparable across tiers
  but not equal to a KV-cached runtime.
- Language forcing: Whisper Korean clips `<|ko|>`=50264, English `<|en|>`=50259.
  Qwen3-ASR auto-detects language (5 s chunks).

### Models under test (deployed in `Assets/StreamingAssets/ASR/`)

| Model | Tier | File | Size MB |
| --- | --- | --- | ---: |
| whisper-tiny | f32 / i8 / i4 | `whisper_tiny_30s_{f32,i8,i4}.tflite` | 151.0 / 41.1 / 36.5 |
| whisper-base | f32 / i8 / i4 | `whisper_base_30s_{f32,i8,i4}.tflite` | 290.1 / 77.0 / 45.3 |
| whisper-large-v3-turbo | f32 † / i8 | `whisper_large_v3_turbo_30s_{f32,i8}.tflite` | 3234.3 / 1088.3 |
| qwen3-asr-0.6b | i8 / i4 † | `qwen3_asr_0.6b_5s_{i8,i4}.tflite` | 793.9 / 625.1 |

† removed from `StreamingAssets/ASR/` after the first pass (turbo f32: no
accuracy gain over i8 at 3× the size; qwen3 i4: silent failure modes) — not
re-run on the second-pass re-recordings.

`whisper-medium/` and `whisper-large-v3/` contained only `tokenizer.json` at
benchmark time (no `.tflite` deployed) and are not covered — **see the
Addendum at the end** (desktop, added later the same day): whisper-medium
i8/i4, whisper-large-v3 i8/i4, and whisper-large-v3-turbo i4 were validated
and deployed at 16:00.

### Scoring

- Normalization before scoring: strip punctuation (`.,!?"'()-:;·、。，`),
  collapse whitespace, lowercase.
- **CER**: Levenshtein on the normalized string with all spaces removed.
- **WER**: Levenshtein on whitespace tokens of the normalized string.
- **Match**: normalized exact match (spacing differences count as mismatch;
  see per-clip notes — several "mismatches" are spacing-only, CER = 0).
- **RTF**: real-time factor = total wall s / audio duration s (lower is better;
  < 1.0 = faster than real time).

## Per-clip results

### Clip 1 — `2025년 3월 5일 전술평가 결과 보고.mp3` (ko, 3.98 s)

Expected: `2025년 3월 5일 전술평가 결과 보고`

| Model | Transcript | Match | CER | WER | Enc s | Dec s | ms/step | Total s | RTF |
| --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny f32 | 2015년 3월 후에 전술 평가 결과보고 | ✗ | 0.176 | 0.833 | 0.24 | 0.74 | 53 | 0.99 | 0.25 |
| tiny i8 | 2015년 3월호일 전술 평가 결과 보고 | ✗ | 0.118 | 0.667 | 0.11 | 0.40 | 31 | 0.51 | 0.13 |
| tiny i4 | 2015년 3월의 전술 평가 결과 보고 | ✗ | 0.176 | 0.667 | 0.15 | 0.65 | 54 | 0.79 | 0.20 |
| base f32 | 2025년 3월 5일 전술 평가 결과 보고 | ✗ (spacing) | 0.000 | 0.333 | 0.69 | 1.77 | 136 | 2.46 | 0.62 |
| base i8 | 2025년 3월 5일 전술 평가 결과 보고 | ✗ (spacing) | 0.000 | 0.333 | 0.64 | 1.97 | 152 | 2.60 | 0.65 |
| base i4 | 2025년 3월 5일 전술 평가 결과 보고. | ✗ (spacing) | 0.000 | 0.333 | 0.48 | 1.32 | 94 | 1.80 | 0.45 |
| turbo f32 | 2025년 3월 5일 전술평가 결과 보고 | ✓ | 0.000 | 0.000 | 5.77 | 2.26 | 161 | 8.03 | 2.02 |
| turbo i8 | 2025년 3월 5일 전술평가 결과 보고 | ✓ | 0.000 | 0.000 | 2.53 | 1.07 | 76 | 3.60 | 0.90 |
| qwen3 i8 | 이천이십오년 삼월오일 전술평가 결과보고. | ✗ (number style) | 0.412 | 0.833 | 0.87 | 10.68 | 509 | 11.56 | 2.90 |
| qwen3 i4 | 이천이십오 년 삼 월 오 일 전술평가 결과보고 | ✗ (number style) | 0.412 | 1.333 | 1.57 | 6.04 | 275 | 7.60 | 1.91 |

Notes: whisper-tiny (all tiers) mishears the year as **2015년** on the new
recording — the old recording was transcribed exactly by tiny i8/f32
(prior validation in `docs/asr-details.md` and probe logs). Qwen3's CER is
inflated purely by number style (spells out `이천이십오 년` instead of digits);
the content is phonetically correct.

### Clip 2 — `Tactical Evaluation Results Report - March 5, 2025.mp3` (en, 4.87 s — re-recorded 13:57)

Expected: `Tactical Evaluation Results Report - March 5, 2025`

| Model | Transcript | Match | CER | WER | Enc s | Dec s | ms/step | Total s | RTF |
| --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny f32 | Tactical Evaluation Results Report, March 5, 2025. | ✓ | 0.000 | 0.000 | 0.11 | 0.44 | 32 | 0.55 | 0.11 |
| tiny i8 | Tactical Evaluation Results Report, March 5, 2025. | ✓ | 0.000 | 0.000 | 0.07 | 0.36 | 26 | 0.43 | 0.09 |
| tiny i4 | Tactical Evaluation Results Report, March 5, 2025. | ✓ | 0.000 | 0.000 | 0.08 | 0.36 | 26 | 0.44 | 0.09 |
| base f32 | Tactical Evaluation Results Report March 5, 2025 | ✓ | 0.000 | 0.000 | 0.20 | 0.60 | 50 | 0.80 | 0.17 |
| base i8 | Tactical Evaluation Results Report March 5, 2025 | ✓ | 0.000 | 0.000 | 0.13 | 0.43 | 36 | 0.57 | 0.12 |
| base i4 | Tactical evaluation Results Report March 5, 2025. | ✓ | 0.000 | 0.000 | 0.19 | 0.42 | 35 | 0.61 | 0.13 |
| turbo i8 | Tactical Evaluation Results Report, March 5, 2025. | ✓ | 0.000 | 0.000 | 2.21 | 0.99 | 71 | 3.20 | 0.66 |
| qwen3 i8 | Tactical evaluation results report, March fifth, twenty twenty-five. | ✗ (number style) | 0.512 | 0.571 | 0.40 | 2.94 | 173 | 3.34 | 0.69 |

**✔ Mislabel resolved** — the clip was re-recorded 2026-07-23 **13:57** (the
previous take pronounced "Technical"; every tier heard it that way). On the new
take all 8 deployed tiers hear **"Tactical"**: all 7 Whisper tiers are exact
after normalization (CER/WER 0.000). Qwen3 i8 also hears "Tactical" correctly
but spells the date out ("March fifth, twenty twenty-five") — the same
number-style behavior as clip 1, phonetically correct. Its first-pass trailing
"Five." hallucination is gone: the shorter 4.87 s take fits a single 5 s chunk.
turbo f32 / qwen3 i4 (†) were removed from StreamingAssets before the re-run;
on the old recording both heard "Technical" like every other tier.

### Clip 3 — `현재 서울의 날씨는, 흐림. 입니다.mp3` (ko, 3.24 s)

Expected (filename): `현재 서울의 날씨는, 흐림. 입니다`

| Model | Transcript | Match | CER | WER | Enc s | Dec s | ms/step | Total s | RTF |
| --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny f32 | 현재 서울의 날씨는 흐림입니다. | ✗ (spacing) | 0.000 | 0.400 | 0.10 | 0.66 | 55 | 0.76 | 0.23 |
| tiny i8 | 현재 서울의 날씨는 흐림입니다. | ✗ (spacing) | 0.000 | 0.400 | 0.06 | 0.43 | 36 | 0.50 | 0.15 |
| tiny i4 | 현재 서울의 날씨는 흐림입니다. | ✗ (spacing) | 0.000 | 0.400 | 0.09 | 0.57 | 48 | 0.66 | 0.21 |
| base f32 | 현재 서울의 날씨는 흐림입니다. | ✗ (spacing) | 0.000 | 0.400 | 0.43 | 1.75 | 146 | 2.17 | 0.67 |
| base i8 | 현재 서울의 날씨는 흐림입니다. | ✗ (spacing) | 0.000 | 0.400 | 0.27 | 1.33 | 110 | 1.59 | 0.49 |
| base i4 | 현재 서울의 날씨는 흐림입니다. | ✗ (spacing) | 0.000 | 0.400 | 0.37 | 0.99 | 82 | 1.36 | 0.42 |
| turbo f32 | 현재 서울의 날씨는 흐림입니다. | ✗ (spacing) | 0.000 | 0.400 | 5.16 | 2.82 | 235 | 7.98 | 2.46 |
| turbo i8 | 현재 서울의 날씨는 흐림입니다. | ✗ (spacing) | 0.000 | 0.400 | 2.14 | 0.83 | 70 | 2.98 | 0.92 |
| qwen3 i8 | 현재 서울의 날씨는 흐림입니다. | ✗ (spacing) | 0.000 | 0.400 | 0.23 | 9.74 | 541 | 9.97 | 3.08 |
| qwen3 i4 | 현재 서울의 날씨는 흐림입니다. | ✗ (spacing) | 0.000 | 0.400 | 0.09 | 3.62 | 201 | 3.71 | 1.15 |

**Filename artifact, not a model error**: every model outputs the natural
`현재 서울의 날씨는 흐림입니다.` — CER 0.000 across the board. The filename's
`흐림. 입니다` spacing is non-standard TTS-pause markup; treat all rows as
correct. WER 0.400 here purely measures the filename's odd tokenization.

### Clip 4 — `The current weather in Seoul is cloudy.mp3` (en, 2.90 s)

Expected: `The current weather in Seoul is cloudy`

| Model | Transcript | Match | CER | WER | Enc s | Dec s | ms/step | Total s | RTF |
| --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny f32 | The current weather in Seoul is cloudy. | ✓ | 0.000 | 0.000 | 0.13 | 0.42 | 47 | 0.55 | 0.19 |
| tiny i8 | The current weather in Seoul is cloudy. | ✓ | 0.000 | 0.000 | 0.11 | 0.56 | 62 | 0.67 | 0.23 |
| tiny i4 | The current weather in Seoul is cloudy. | ✓ | 0.000 | 0.000 | 0.11 | 0.44 | 49 | 0.55 | 0.19 |
| base f32 | The current weather and soul is cloudy. | ✗ | 0.094 | 0.286 | 0.33 | 1.21 | 134 | 1.54 | 0.53 |
| base i8 | The current weather and soul is cloudy. | ✗ | 0.094 | 0.286 | 0.26 | 1.05 | 117 | 1.31 | 0.45 |
| base i4 | The current weather in Seoul is cloudy. | ✓ | 0.000 | 0.000 | 0.36 | 0.89 | 99 | 1.25 | 0.43 |
| turbo f32 | The current weather in Seoul is cloudy. | ✓ | 0.000 | 0.000 | 7.96 | 3.98 | 442 | 11.94 | 4.11 |
| turbo i8 | The current weather in Seoul is cloudy. | ✓ | 0.000 | 0.000 | 2.34 | 1.22 | 136 | 3.56 | 1.23 |
| qwen3 i8 | The current weather in Seoul is cloudy. | ✓ | 0.000 | 0.000 | 0.21 | 7.70 | 642 | 7.90 | 2.72 |
| qwen3 i4 | The current weather in Seoul is cloudy. | ✓ | 0.000 | 0.000 | 0.09 | 2.24 | 186 | 2.33 | 0.80 |

Note: base f32/i8 mishear "in Seoul" as "and soul" on the new recording;
interestingly base **i4** gets it right (tie-break noise near the decision
boundary, not an i4 quality win).

### Clip 5 — `변경사항을 검토 중입니다. 앱을 검토하는 과정에서 추가 문제가 발견될 수도 있습니다..mp3` (ko, 7.03 s)

Expected: `변경사항을 검토 중입니다. 앱을 검토하는 과정에서 추가 문제가 발견될 수도 있습니다.`

| Model | Transcript | Match | CER | WER | Enc s | Dec s | ms/step | Total s | RTF |
| --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny f32 | 병경사항을 검토 중입니다. 앱을 검토하는 과정에서 추가 문제가 발견될 수도 있습니다. | ✗ | 0.029 | 0.091 | 0.08 | 1.12 | 37 | 1.20 | 0.17 |
| tiny i8 | 병경사항을 검토 중입니다. 앱을 검토하는 과정에서 추가 문제가 발견될 수도 있습니다. | ✗ | 0.029 | 0.091 | 0.11 | 1.64 | 55 | 1.75 | 0.25 |
| tiny i4 | 평경 사항을 검토 중입니다. 앱을 검토하는 과정에서 추가 문재가 발견될 수도 있습니다. | ✗ | 0.057 | 0.273 | 0.11 | 1.64 | 51 | 1.75 | 0.25 |
| base f32 | 변경사항을 검토 중입니다. 앱을 검토하는 과정에서 추가 문제가 발견될 수도 있습니다. | ✓ | 0.000 | 0.000 | 0.33 | 4.38 | 146 | 4.71 | 0.67 |
| base i8 | 변경사항을 검토 중입니다. 앱을 검토하는 과정에서 추가 문제가 발견될 수도 있습니다. | ✓ | 0.000 | 0.000 | 0.25 | 3.26 | 109 | 3.51 | 0.50 |
| base i4 | 병경 상황을 검토 중입니다. 앱을 검토하는 과정에서 추가 문제가 발견될 수도 있습니다. | ✗ | 0.086 | 0.182 | 0.33 | 3.05 | 105 | 3.39 | 0.48 |
| turbo f32 | 변경 사항을 검토 중입니다. 앱을 검토하는 과정에서 추가 문제가 발견될 수도 있습니다. | ✗ (spacing) | 0.000 | 0.182 | 10.39 | 12.67 | 422 | 23.07 | 3.28 |
| turbo i8 | 변경 사항을 검토 중입니다. 앱을 검토하는 과정에서 추가 문제가 발견될 수도 있습니다. | ✗ (spacing) | 0.000 | 0.182 | 2.91 | 2.83 | 94 | 5.74 | 0.82 |
| qwen3 i8 | 변경 사항을 검토 중입니다. 앱을 검토하는 과정에서 추가. 문제가 발견될 수도 있습니다. | ✗ (spacing) | 0.000 | 0.182 | 0.64 | 21.99 | 579 | 22.63 | 3.22 |
| qwen3 i4 | 변경 사항을 검토 중입니다. 앱을 검토하는 과정에서 추가. 문제가 발견될 수도 있습니다. | ✗ (spacing) | 0.000 | 0.182 | 0.18 | 6.94 | 183 | 7.12 | 1.01 |

Notes: base f32/i8 are exact. Turbo/qwen are character-perfect (CER 0) with
spacing-only differences. Qwen's mid-sentence `추가.` period is a 5 s chunk
boundary artifact (sentence split across chunks). tiny hears 병경/평경 for 변경;
base i4 degrades 사항→상황.

### Clip 6 — `We are currently reviewing the changes Additional issues. may be discovered during the app review process..mp3` (en, 6.55 s)

Expected: `We are currently reviewing the changes Additional issues. may be discovered during the app review process.`

| Model | Transcript | Match | CER | WER | Enc s | Dec s | ms/step | Total s | RTF |
| --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny f32 | We are currently reviewing the changes. Additional issues may be discovered during the app review process. | ✓ | 0.000 | 0.000 | 0.07 | 1.28 | 67 | 1.35 | 0.21 |
| tiny i8 | We are currently reviewing the changes. Additional issues may be discovered during the app review process. | ✓ | 0.000 | 0.000 | 0.11 | 0.72 | 38 | 0.84 | 0.13 |
| tiny i4 | We are currently reviewing the changes. Additional issues may be discovered during the app review process. | ✓ | 0.000 | 0.000 | 0.08 | 0.92 | 48 | 1.00 | 0.15 |
| base f32 | We are currently reviewing the changes. Additional issues may be discovered during the app review process. | ✓ | 0.000 | 0.000 | 0.34 | 2.54 | 134 | 2.88 | 0.44 |
| base i8 | We are currently reviewing the changes. Additional issues may be discovered during the app review process. | ✓ | 0.000 | 0.000 | 0.28 | 2.56 | 135 | 2.84 | 0.43 |
| base i4 | We are currently reviewing the changes. Additional issues may be discovered during the app review process. | ✓ | 0.000 | 0.000 | 0.47 | 2.37 | 125 | 2.84 | 0.43 |
| turbo f32 | We are currently reviewing the changes. Additional issues may be discovered during the app review process. | ✓ | 0.000 | 0.000 | 11.89 | 9.77 | 514 | 21.66 | 3.31 |
| turbo i8 | We are currently reviewing the changes. Additional issues may be discovered during the app review process. | ✓ | 0.000 | 0.000 | 2.64 | 2.06 | 108 | 4.70 | 0.72 |
| qwen3 i8 | We are currently reviewing the changes. Additional issues may be discovered during. The app review process. | ✓ | 0.000 | 0.000 | 0.36 | 13.61 | 504 | 13.97 | 2.13 |
| qwen3 i4 | We are currently reviewing the changes. Additional issues may be discovered. During the app review process. | ✓ | 0.000 | 0.000 | 0.18 | 5.25 | 194 | 5.43 | 0.83 |

All 10 tiers perfect after normalization (the filename's `issues. may` period
placement is a filename artifact). Qwen's mid-sentence periods are chunk
boundary artifacts.

### Clip 7 — `볼륨 업` (ko FC command) — NEW, two takes

Expected: `볼륨 업`

#### Old take — `volume-볼륨 업.mp3` (0.79 s, 13:36, quiet)

| Model | Transcript | Match | CER | WER | Enc s | Dec s | ms/step | Total s | RTF |
| --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny f32 | 보일해봐 | ✗ | 1.333 | 1.000 | 0.10 | 0.25 | 50 | 0.35 | 0.45 |
| tiny i8 | 보일해봐 | ✗ | 1.333 | 1.000 | 0.07 | 0.17 | 35 | 0.25 | 0.31 |
| tiny i4 | 보일이 안되요. | ✗ | 2.000 | 1.000 | 0.16 | 0.62 | 78 | 0.78 | 0.99 |
| base f32 | 볼륨어 | ✗ | 0.333 | 1.000 | 0.29 | 0.59 | 99 | 0.88 | 1.12 |
| base i8 | 볼륨어 | ✗ | 0.333 | 1.000 | 0.29 | 0.79 | 131 | 1.08 | 1.36 |
| base i4 | 볼륙 먹. | ✗ | 0.667 | 1.000 | 0.32 | 0.59 | 84 | 0.91 | 1.15 |
| turbo f32 | 볼륨업 | ✗ (spacing) | 0.000 | 1.000 | 9.29 | 3.47 | 579 | 12.76 | 16.11 |
| turbo i8 | 볼륨업 | ✗ (spacing) | 0.000 | 1.000 | 3.08 | 0.61 | 102 | 3.70 | 4.67 |
| qwen3 i8 | 볼륨업 | ✗ (spacing) | 0.000 | 1.000 | 0.14 | 2.91 | 416 | 3.05 | 3.85 |
| qwen3 i4 | 播放音乐。 | ✗ hallucination | 1.333 | 1.000 | 0.11 | 1.76 | 220 | 1.87 | 2.36 |

**Hardest clip in the set** (0.79 s, RMS 0.071 — quietest and shortest).
Only turbo (both tiers) and qwen3 i8 hear `볼륨업` (character-perfect).
tiny/base fail completely. qwen3 **i4** hallucinates Chinese
(`播放音乐` = "play music") — a language-detection collapse on the quantized
tier. For FC keyword matching, `볼륨업` (no space) should be accepted.

#### Re-recorded take — `volume-볼륨, 업.mp3` (1.10 s, 14:00, louder/longer; new file, old file kept)

Deployed tiers re-run on the re-recorded command:

| Model | Transcript | Match | CER | WER | Enc s | Dec s | ms/step | Total s | RTF |
| --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny f32 | 볼륨업 | ✗ (spacing) | 0.000 | 1.000 | 0.06 | 0.18 | 30 | 0.24 | 0.22 |
| tiny i8 | 볼륨업 | ✗ (spacing) | 0.000 | 1.000 | 0.05 | 0.13 | 22 | 0.19 | 0.17 |
| tiny i4 | 별념 | ✗ | 1.000 | 1.000 | 0.04 | 0.11 | 22 | 0.15 | 0.14 |
| base f32 | 볼륨 업 | ✓ | 0.000 | 0.000 | 0.16 | 0.33 | 56 | 0.49 | 0.45 |
| base i8 | 볼륨 업 | ✓ | 0.000 | 0.000 | 0.13 | 0.25 | 42 | 0.38 | 0.34 |
| base i4 | 볼륨 업 | ✓ | 0.000 | 0.000 | 0.11 | 0.22 | 37 | 0.34 | 0.31 |
| turbo i8 | 볼륨 억 | ✗ | 0.333 | 0.500 | 2.44 | 0.58 | 82 | 3.02 | 2.73 |
| qwen3 i8 | 볼륨 업 | ✓ | 0.000 | 0.000 | 0.10 | 1.85 | 231 | 1.95 | 1.76 |

**Re-recording largely fixes the clip**: tiny f32/i8 go from garbage
(`보일해봐`) to `볼륨업` (CER 0, space-insensitive match); **all three base
tiers** and qwen3 i8 are now exact `볼륨 업`. tiny **i4** still fails (`별념`).
One surprise: **turbo i8 regresses** on this take — `볼륨 억` (업→억,
CER 0.333) where it was character-perfect on the old quiet take. With
space/comma-insensitive FC matching, the re-recorded command is recognized by
7 of 8 deployed tiers (all but tiny i4); turbo i8 needs a fuzzy match
(`볼륨 억`) or should use the old-take behavior as reference. The summary
below uses this re-recorded take for clip 7.

### Clip 8 — `volume-소리 키워줘.mp3` (ko FC command, 1.32 s) — NEW

Expected: `소리 키워줘`

| Model | Transcript | Match | CER | WER | Enc s | Dec s | ms/step | Total s | RTF |
| --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny f32 | 소리 키워줘 | ✓ | 0.000 | 0.000 | 0.11 | 0.23 | 47 | 0.35 | 0.26 |
| tiny i8 | 소리 키워줘 | ✓ | 0.000 | 0.000 | 0.07 | 0.18 | 36 | 0.25 | 0.19 |
| tiny i4 | 소리 키워죠 | ✗ | 0.200 | 0.500 | 0.15 | 0.39 | 78 | 0.54 | 0.41 |
| base f32 | 소리 키워줘 | ✓ | 0.000 | 0.000 | 0.36 | 0.53 | 106 | 0.89 | 0.67 |
| base i8 | 소리 키워줘 | ✓ | 0.000 | 0.000 | 0.24 | 0.55 | 111 | 0.79 | 0.60 |
| base i4 | 소리 키워줘. | ✓ | 0.000 | 0.000 | 0.44 | 0.70 | 116 | 1.14 | 0.86 |
| turbo f32 | 소리 키워줘. | ✓ | 0.000 | 0.000 | 9.50 | 2.27 | 379 | 11.77 | 8.92 |
| turbo i8 | 소리 키워줘. | ✓ | 0.000 | 0.000 | 2.85 | 0.70 | 116 | 3.55 | 2.69 |
| qwen3 i8 | 소리 키워줘. | ✓ | 0.000 | 0.000 | 0.11 | 5.32 | 484 | 5.43 | 4.12 |
| qwen3 i4 | *(empty)* | ✗ empty output | 1.000 | 1.000 | 0.11 | 0.22 | 225 | 0.33 | 0.25 |

qwen3 **i4** emits EOS immediately (empty transcript) — unusable on this clip.
tiny i4 degrades 줘→죠. Everything else exact.

### Clip 9 — `volume-음량 증가.mp3` (ko FC command, 1.15 s) — NEW

Expected: `음량 증가`

| Model | Transcript | Match | CER | WER | Enc s | Dec s | ms/step | Total s | RTF |
| --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny f32 | 능량 증가 | ✗ | 0.250 | 0.500 | 0.10 | 0.35 | 50 | 0.45 | 0.39 |
| tiny i8 | 능량 증가 | ✗ | 0.250 | 0.500 | 0.06 | 0.23 | 33 | 0.29 | 0.26 |
| tiny i4 | 능량 증가 | ✗ | 0.250 | 0.500 | 0.16 | 0.47 | 67 | 0.63 | 0.55 |
| base f32 | 음량 증가 | ✓ | 0.000 | 0.000 | 0.39 | 0.69 | 115 | 1.08 | 0.94 |
| base i8 | 음량 증가 | ✓ | 0.000 | 0.000 | 0.24 | 0.66 | 111 | 0.90 | 0.78 |
| base i4 | 음향 증가. | ✗ | 0.250 | 0.500 | 0.42 | 0.42 | 69 | 0.84 | 0.73 |
| turbo f32 | 음량 증가 | ✓ | 0.000 | 0.000 | 9.01 | 1.69 | 282 | 10.70 | 9.29 |
| turbo i8 | 음량 증가 | ✓ | 0.000 | 0.000 | 2.64 | 0.56 | 93 | 3.19 | 2.77 |
| qwen3 i8 | 음량 증가 | ✓ | 0.000 | 0.000 | 0.12 | 3.84 | 427 | 3.96 | 3.44 |
| qwen3 i4 | 음량 증가 | ✓ | 0.000 | 0.000 | 0.10 | 2.00 | 223 | 2.10 | 1.82 |

tiny hears 능량; base i4 hears 음향. base f32/i8, turbo, qwen3 exact.

## Summary

Averages over all 9 clips; ko = 6 Korean clips, en = 3 English clips.
Exact = normalized exact matches (clip 3's spacing artifact counts against all
models equally). Rows for the 8 deployed tiers use the **re-recorded** clip 2
(13:57) and re-recorded clip 7 (`volume-볼륨, 업.mp3`, 14:00); † rows are
first-pass values on the old recordings (tier removed from StreamingAssets,
not re-run — not directly comparable on clips 2 and 7).

| Model | Size MB | Exact /9 | CER ko | WER ko | CER en | WER en | Avg RTF | Avg ms/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| whisper-tiny f32 | 151.0 | 4 | 0.076 | 0.471 | **0.000** | **0.000** | 0.23 | 46 |
| whisper-tiny i8 | 41.1 | 4 | 0.066 | 0.443 | **0.000** | **0.000** | **0.18** | 38 |
| whisper-tiny i4 | 36.5 | 3 | 0.281 | 0.557 | **0.000** | **0.000** | 0.24 | 49 |
| whisper-base f32 | 290.1 | **6** | **0.000** | **0.122** | 0.031 | 0.095 | 0.57 | 114 |
| whisper-base i8 | 77.0 | **6** | **0.000** | **0.122** | 0.031 | 0.095 | 0.48 | 102 |
| whisper-base i4 | 45.3 | 5 | 0.056 | 0.236 | **0.000** | **0.000** | 0.47 | 85 |
| whisper-large-v3-turbo f32 † | 3234.3 | 5 | 0.000 | 0.264 | 0.024 | 0.048 | 5.66 | 351 |
| whisper-large-v3-turbo i8 | 1088.3 | **6** | 0.056 | 0.180 | **0.000** | **0.000** | 1.49 | 94 |
| qwen3-asr-0.6b i8 | 793.9 | 5 | 0.069* | 0.236 | 0.171* | 0.190 | 2.67 | 454 |
| qwen3-asr-0.6b i4 † | 625.1 | 3 | 0.458 | 0.652 | 0.073 | 0.143 | 1.22 | 209 |

\* qwen3 i8 CER inflation is entirely **number style**: Korean CER is 0.000 on
5 of 6 Korean clips (clip 1: `이천이십오 년` vs `2025년`), and English CER
0.171 comes solely from clip 2's spelled-out date ("March fifth, twenty
twenty-five" vs "March 5, 2025") — both phonetically correct. Content accuracy
is on par with base/turbo.

### Audio-set issues found (all models agree — not model errors)

1. **Clip 2 mispronunciation — RESOLVED**: the first take was heard as
   "Technical" by every tier. Re-recorded 2026-07-23 13:57; all 8 deployed
   tiers now hear "Tactical" (7 Whisper tiers exact, CER/WER 0.000).
2. **Clip 3 filename markup**: `흐림. 입니다` — all models output the natural
   `흐림입니다`; treat as correct (CER 0 for all tiers).
3. **Clip 7 (`볼륨 업`) recording quality — RESOLVED (mostly)**: the old take
   (0.79 s, RMS 0.071) was unrecognizable to tiny/base. Re-recorded as
   `volume-볼륨, 업.mp3` (1.10 s, 14:00): tiny f32/i8, all base tiers, and
   qwen3 i8 now recognize it; tiny i4 still fails; turbo i8 newly hears
   `볼륨 억` on this take. Keep space/comma-insensitive FC matching
   (`볼륨업` = `볼륨 업`).

### Regressions on the new recordings vs old validation

- **whisper-tiny (all tiers), clip 1**: old recording → exact
  `2025년 3월 5일 전술 평가 결과보고`; new recording → `2015년 ...` (year
  wrong on f32, i8, and i4). Recording-driven; tiny remains unreliable for
  Korean numerics.
- **whisper-base f32/i8, clip 4**: new recording heard as "and soul" instead of
  "in Seoul" (old recording was clean on base f32 in the Android smoke run).
- **whisper-large-v3-turbo i8**: no regression — still exact on clip 1 as in
  the pre-replacement validation.
- **qwen3 i4**: newly exposed failure mode on short clips — Chinese
  hallucination (clip 7) and empty output (clip 8). The 5 s-window i4 export
  had previously only been validated on the longer clip 1.

### Second-pass changes (13:57 / 14:00 re-recordings)

- **Clip 2**: all 8 deployed tiers flip from "Technical" to "Tactical";
  Whisper tiers all become exact. base f32/i8 gain +1 exact; qwen3 i8 loses
  the "Five." hallucination but gains a spelled-out date.
- **Clip 7 (re-recorded take)**: tiny f32/i8 and base f32/i8/i4 flip from
  failure to recognition (base tiers exact); qwen3 i8 exact.
- **turbo i8, clip 7 re-recorded take**: only second-pass regression —
  `볼륨 억` (CER 0.333) vs character-perfect `볼륨업` on the old take. Its
  Korean CER average is now 0.056 (was 0.000).

### Recommendation

| Use case | Pick | Why |
| --- | --- | --- |
| Best Korean accuracy (incl. voice commands) | **whisper-base i8** (77 MB) or **whisper-large-v3-turbo i8** (1.09 GB) | On the second-pass clip set base i8 matches turbo i8: 6/9 exact, CER ko 0.000 (turbo: 0.056 after the `볼륨 억` miss on re-recorded clip 7) at 1/14 the size and 3× the speed. turbo i8 remains the safest on hard/quiet audio (only tier that read the old quiet clip 7 perfectly); f32 turbo added nothing and was removed. |
| Balanced size/accuracy for sentences | **whisper-base i8** (77 MB) | Identical accuracy to base f32 at 27% of the size; Korean CER 0.000, 6/9 exact; now also exact on all 3 FC command clips with the re-recorded clip 7. |
| Smallest/fastest, English-lean | **whisper-tiny i8** (41 MB) | RTF 0.18, English CER 0.000 on the re-recorded set, but Korean CER 0.066 with year errors (2015년) — English-only or non-critical Korean. |
| Not recommended | whisper-tiny i4, whisper-base i4 | Both strictly worse than their i8 siblings in Korean accuracy for only 4–32 MB saved (tiny i4 still fails even the re-recorded clip 7). |
| Not recommended | **qwen3-asr-0.6b i4** (removed) | Silent failure modes on short clips (Chinese hallucination, empty output) — removed from StreamingAssets. qwen3 i8 is content-accurate (incl. both clip-7 takes) but decodes at ~450–520 ms/step CPU (RTF ~2.7) and spells out numbers (`이천이십오 년`, "March fifth") — unsuitable when digit-form output or latency matters. |

Function-calling voice pipeline (the 3 `volume-*` commands, using the
re-recorded clip 7): **base f32/i8** and **qwen3 i8** now recognize all three
commands exactly (`볼륨 업`/`소리 키워줘`/`음량 증가`); tiny f32/i8 get clip 7
space-insensitively but still fail clip 9 (`능량 증가`); base i4 fixes clip 7
but keeps its clip-9 miss (`음향 증가`); **turbo i8** recognizes all three on
the old takes but hears `볼륨 억` on the re-recorded clip 7 — keep
space/comma-insensitive matching and consider a `볼륨 억` alias if routing FC
audio to turbo.

## Reproduction

- Batch probe: scratchpad `bench_asr.py` (`<out.json> <w80|w128|qwen> <model.tflite> <tokenizer.json>`),
  greedy decode, 8 CPU threads, `PYTHONIOENCODING=utf-8`.
- Scoring: scratchpad `score_asr.py` (Levenshtein CER/WER, punctuation-stripped,
  whitespace-collapsed, lowercase; CER on space-removed string).
- Raw per-clip JSON incl. qwen chunk-level output: scratchpad `bench_results/*.json`
  (first pass), `bench_results/clip2/*.json` (clip-2 re-run),
  `bench_results/vol/*.json` (both clip-7 takes). `bench_asr.py` takes an
  optional 5th arg: a clip-filename substring filter for single-clip re-runs.

## Addendum — late-added tiers (desktop, added later, 2026-07-23 16:00)

Five new tiers converted/validated on desktop **after** the main matrix above
and deployed to `Assets/StreamingAssets/ASR/`: whisper-medium i8/i4,
whisper-large-v3 i8/i4, and whisper-large-v3-turbo **i4**. All were quantized
from freshly exported f32 tflites (transformers `TFWhisperForConditionalGeneration`
→ 30 s litert-community-style two-signature graph; f32 sources stay in
`External/whisper-large-variants/`, not deployed: medium 3.05 GB,
turbo 3.23 GB, large-v3 6.17 GB buffer-offset flatbuffer).

Recipes (per the int4-minimum-tier policy):

- **i8** = converter-time dynamic-range int8 weights (same scheme as
  `ai_edge_quantizer` `dynamic_wi8_afp32`, validated transcript-identical on
  whisper-base).
- **i4** = **mixed "mixD" recipe**: `dynamic_wi4b64_afp32` default with the
  token-embedding/logits scopes kept at wi8 channelwise (pure `wi4b64`
  corrupts Korean output on the large family; `wi4c` is never used).
  medium i4 = the same emb/logits-at-i8 mix ("L1"); the encoder-also-at-i8
  variant ("L2", 798 MB) produced identical transcripts and was discarded
  for size.

Methodology: scratchpad `bench_asr_v2.py` with the **as-shipped preprocessing**
(`--norm-boost --vad`: boost-only RMS normalization + VAD trim), greedy decode,
8 CPU threads — same host as the main matrix. Note clip 3 has since been
renamed on disk to `현재 서울의 날씨는 흐림 입니다.mp3` (markup punctuation
dropped); its `WER 0.400` remains the filename spacing artifact
(`흐림 입니다` vs the natural `흐림입니다`), CER 0.000 for every tier.
Raw JSON: scratchpad `bench_results/resume/*.json` (`*_p2.json` = the
4 clips missing from the first pass).

### Summary (same 9-clip set and scoring as the main summary; clip 7 = re-recorded take)

| Model | Size MB | Exact /9 | CER ko | WER ko | CER en | WER en | Avg RTF | Avg ms/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| whisper-medium i8 | 831.6 | 7 | 0.042 | 0.150 | **0.000** | **0.000** | 1.67 | 295 |
| whisper-medium i4 | 664.3 | 7 | 0.042 | 0.233 | **0.000** | **0.000** | 1.74 | 311 |
| whisper-large-v3 i8 | 1631.7 | 7 | **0.000** | 0.097 | **0.000** | **0.000** | 4.04 | 732 |
| whisper-large-v3 i4 | 1148.1 | 7 | **0.000** | 0.097 | **0.000** | **0.000** | 3.38 | 582 |
| whisper-large-v3-turbo i4 | 755.3 | **8** | **0.000** | 0.067 | **0.000** | **0.000** | 1.25 | **72** |

Every "miss" against exact match in these five tiers is **spacing-only or a
filename artifact** except one real error: **medium i8/i4 hear clip 9
`음량 증가` as `음향 증가` / `음향증가`** (CER 0.250 on that clip — the same
음량→음향 confusion as whisper-base i4). Neither large-v3 tier nor turbo i4
makes any character error on any clip.

### Deploy-gate clips (per-tier transcripts)

| Model | Clip 1 KO `전술평가` | Clip 2 EN (re-recorded) | Clip 7 re-recorded `볼륨 업` |
| --- | --- | --- | --- |
| medium i8 | `2025년 3월 5일 전술평가 결과 보고` ✓ | `Tactical Evaluation Results Report, March 5, 2025.` ✓ | `볼륨 업` ✓ |
| medium i4 | `2025년 3월 5일 전술평가 결과 보고` ✓ | `Tactical Evaluation Results Report, March 5, 2025.` ✓ | `볼륨 업` ✓ |
| large-v3 i8 | `2025년 3월 5일 전술평가 결과 보고` ✓ | `Tactical Evaluation Results Report, March 5, 2025.` ✓ | `볼륨 업` ✓ |
| large-v3 i4 | `2025년 3월 5일 전술평가 결과 보고` ✓ | `Tactical Evaluation Results Report, March 5, 2025.` ✓ | `볼륨 업` ✓ |
| turbo i4 | `2025년 3월 5일 전술평가 결과 보고` ✓ | `Tactical Evaluation Results Report, March 5, 2025.` ✓ | `볼륨 업` ✓ |

All five tiers pass all three gate clips exactly (after punctuation/space
normalization). On the old quiet clip-7 take (`volume-볼륨 업.mp3`) all five
output the character-perfect `볼륨업`.

### Addendum observations

- **turbo i4 (755 MB) is the best tier in the entire matrix**: 8/9 exact,
  CER 0.000 in both languages, 72 ms/step (fastest large-family tier), and it
  **fixes turbo i8's only regression** — it hears the re-recorded clip 7 as
  exact `볼륨 업` where the deployed turbo i8 heard `볼륨 억`. It also gets
  clip 5 fully exact (turbo i8 was spacing-off). At 69 % of turbo i8's size
  it beats it on every axis measured here.
- **large-v3 i8/i4**: character-perfect on all 9 clips (CER 0.000 ko+en);
  both misses vs exact are spacing (`변경 사항` for `변경사항`, clip 3
  filename artifact). But 3–7× slower than turbo per step on CPU with no
  accuracy gain over turbo i4 on this clip set — large-v3 is the
  accuracy-reference tier, not the practical pick. i4 (1148 MB) is both
  smaller **and** ~20 % faster per step than i8 (1632 MB).
- **medium i8/i4**: exact on all three FC gate clips and all English clips;
  its only content error is clip 9's `음향 증가`. i4 matches i8's transcripts
  everywhere except degrading clip 9's spacing further (`음향증가`) — at
  167 MB less. medium sits between base and turbo in both size and accuracy,
  and 80-mel medium decodes ~4× slower per step than turbo i4 (128-mel) on
  this host.
- The deployed large-v3 **i8 is the converter-DRQ export** (1632 MB) — the
  `ai_edge_quantizer` wi8 control (1831 MB, `whisper_large_v3_30s_i8.tflite`
  pre-rename in `External/whisper-large-variants/`) produced identical
  transcripts at +200 MB and ~10 % slower steps and was not deployed.

### Deployed state after the addendum (`Assets/StreamingAssets/ASR/`)

| Folder | Files (tokenizer.json in every folder) |
| --- | --- |
| whisper-medium | `whisper_medium_30s_i8.tflite` (832 MB), `whisper_medium_30s_i4.tflite` (664 MB) |
| whisper-large-v3 | `whisper_large_v3_30s_i8.tflite` (1632 MB), `whisper_large_v3_30s_i4.tflite` (1148 MB) |
| whisper-large-v3-turbo | `whisper_large_v3_turbo_30s_i8.tflite` (1088 MB), `whisper_large_v3_turbo_30s_i4.tflite` (755 MB) |

Tokenizer per tier verified by md5: medium uses the 80-mel/51865-vocab medium
tokenizer, large-v3 and turbo use their own 128-mel/51866-vocab tokenizers
(all three differ from whisper-base's).
