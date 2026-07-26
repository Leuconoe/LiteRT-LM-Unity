# LiteRT-LM-Unity

**Android 온디바이스 AI**를 위한 Unity 통합 프로젝트. 네트워크 없이 기기 단독으로
LLM 채팅 · 음성 인식(ASR) · 이미지 인식 · 펑션콜링을 실행합니다.

- **LiteRT-LM v0.14.0** (`.litertlm` 1.5.0), 커스텀은
  `Tools/UnityAar/litert-lm-unity-aar.patch`로 관리
- 실기기 검증: **Snapdragon 865 / 7.5 GB RAM / Android 12** — 4대 기능 전부 PASS,
  6회 PDCA 사이클, 80+ 런, 크래시 0
  ([원장](docs/benchmarks/device-cycle1-baseline.md))
- 공개 모델: [whisper-acft](https://huggingface.co/litert-community/whisper-acft) ·
  [whisper-acft-ko](https://huggingface.co/leuconoe/whisper-acft-ko) ·
  [litert-lm-unity-quantized](https://huggingface.co/leuconoe/litert-lm-unity-quantized)

## 기능 (디바이스 실측)

| 기능 | 속도 | 적중률 | 모델 |
| --- | --- | --- | --- |
| LLM 채팅 | 35.5 tok/s | — | Qwen2.5-0.5B i4 |
| 음성 인식 | 0.7–0.8 s | 4/5 | whisper-base-acft-ko 5s |
| 이미지 인식 | 7.6 s (GPU) | 정확 | gemma-4-E2B QAT |
| 펑션콜링 | 15.5 s E2E (음성→도구) | 19/20 | gemma-4-E2B / Qwen3-0.6B |

## Requirements

Unity `6000.4.6f1` + Android Build Support · Android 기기(`adb`, Snapdragon 865급
이상 / RAM 4 GB+) · Windows PowerShell(빌드 스크립트) · Docker(AAR 재빌드 시)

## Quick Start

1. **모델 배치** — 아래 표에서 골라 `Assets/StreamingAssets/` 하위에 배치
   (모델 파일은 저장소에 없음)
2. **APK 빌드** — Unity 메뉴 `LiteRT-LM/Android/...` 또는
   `Tools/Windows/Build-LiteRtLmAndroid*.ps1`
3. **스모크** — `Run-LiteRtLmAndroidAsrSmokeTest.ps1 -DeviceSerial <serial>`,
   결과는 `Builds/Logs/AndroidDeviceRuns/`

## 권장 모델

### LLM — 기기 RAM으로 선택

| 기기 RAM | 모델 | Size | 디바이스 실측 |
| --- | --- | ---: | --- |
| 4~6 GB | `LLM/qwen2.5-0.5b/…_wi4b64_ekv1280.litertlm` | 265 MB | 35.5 tok/s — 채팅 전용(FC 불가) |
| 6~8 GB | `LLM/qwen3-0.6b/qwen3_0_6b_mixed_int4.litertlm` | 475 MB | 20.9 tok/s, FC 18/20 |
| 8 GB+ | `Multimodal/gemma-4-e2b/gemma-4-E2B-it.litertlm` | 2.6 GB | FC 19/20, 이미지 7.6 s, 오디오 4.1 s. 이미지 턴 PSS 3.6 GB |

채팅(디코드)은 **CPU**, 긴 프롬프트·이미지는 **GPU**.
[모델별 상세 →](docs/llm-details.md)

### ASR — 발화 길이로 선택 (보통 1종이면 충분)

| 발화 길이 | 모델 | Size | 디바이스 실측 |
| --- | --- | ---: | --- |
| ≤5 s (명령·짧은 문장) | `ASR/whisper-base-acft-ko/acft_base_5s_drq.tflite` | 101 MB | 0.7–0.8 s, 4/5 exact |
| 5~30 s (받아쓰기) | `ASR/whisper-base/whisper_base_30s_i8.tflite` | 77 MB | 2.7 s, 문장 CER 0.000 |
| >30 s (배치) | `ASR/qwen3-asr-0.6b/qwen3_asr_0.6b_5s_i8.tflite` | 794 MB | 청크 루프, RTF ≈2.6 |

- 티어별 `tokenizer.json`을 같은 폴더에 함께 배치
- **VAD** 기본 활성 — `energy`(비용 0) / `ai`(Silero 1.25 MB) / `off`
- 조용한 녹음까지 필요하면 turbo-acft-ko 5s(883 MB, 5/5)를 온디맨드로 로드

[전체 10티어 비교·선택 근거·ACFT 학습 배경 →](docs/asr-details.md)

## Test Scenes

`Assets/Scenes/Tests/` — 생성은 메뉴 `LiteRT-LM/Test Scenes/Generate All`.

| Scene | 용도 |
| --- | --- |
| `LiteRtLmLlmChatTestScene` | 멀티턴 채팅, think/no_think 토글 |
| `LiteRtLmAsrTestScene` | ASR — 파일/마이크/상시청취(Continuous) |
| `LiteRtLmMultimodalTestScene` | 이미지 + 오디오 입력 |
| `LiteRtLmAsrFunctionCallingTestScene` | 음성 → 도구 호출 (15.5 s) |
| `LiteRtLmMultimodalFunctionCallingTestScene` | 이미지 + 발화 → 도구 호출 (40.7 s) |
| `LiteRtLmTranslateTestScene` | 번역 — Whisper Direct / ASR+LLM |

## AAR 재빌드 (네이티브 변경 시)

`Tools/Windows/Build-LiteRtLmUnityAarFromPatch.ps1 -SourceRoot <pristine v0.14.0>`
— Docker로 패치를 적용해 빌드 후 `Assets/Plugins/Android/`에 배포.
⚠️ `-SkipImageBuild`는 이미지에 구워진 낡은 소스를 빌드하므로 패치 변경 후에는
이미지 재빌드 필수.

## Docs

| 문서 | 내용 |
| --- | --- |
| [`docs/llm-details.md`](docs/llm-details.md) | LLM 티어·백엔드·디바이스 실측 |
| [`docs/asr-details.md`](docs/asr-details.md) | ASR 전 티어·VAD·ACFT-KO 학습 배경 |
| [`docs/README.md`](docs/README.md) | 벤치마크·핸드오프 전체 인덱스 |

Windows 에디터는 배포 전 로직 검증 보조용입니다 — 성능 특성이 Android와
반대(GPU 우세)이니 기기 판단 근거로 쓰지 마세요. 상세는 위 두 문서 참조.
