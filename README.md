# LiteRT-LM-Unity

**Android 온디바이스 AI**를 위한 Unity 통합 프로젝트입니다. 네트워크 없이 기기
단독으로 LLM 채팅 · 음성 인식(ASR) · 이미지 인식 · 펑션콜링(음성/멀티모달 →
도구 호출)을 실행합니다. Windows 에디터 환경은 개발·검증 보조용으로 함께
지원합니다.

- Framework: **LiteRT-LM v0.14.0** (`External/LiteRT-LM`, 커스텀은
  `Tools/UnityAar/litert-lm-unity-aar.patch`로 관리), `.litertlm` 포맷 1.5.0
- 실기기 검증: **Snapdragon 865 (kona) / 7.5 GB RAM / Android 12** 기기에서
  4대 기능 전부 PASS — 3회 PDCA 사이클, 40+ 런, 크래시 0
  ([검증 원장](docs/benchmarks/device-cycle1-baseline.md))

## 온디바이스 핵심 기능 (실기기 검증 완료)

| 기능 | 디바이스 실측 | 사용 모델 |
| --- | --- | --- |
| **LLM 채팅** | 16~35.5 tok/s (CPU) | Qwen2.5/Qwen3/gemma3/LFM2.5 계열 |
| **음성 인식 (ASR)** | 짧은 명령 ~2 s, 문장 전사 CER 0.000 | Qwen3-ASR-0.6B, whisper turbo i4 등 |
| **이미지 인식** | 이미지 묘사 7.6 s (GPU) | gemma-4-E2B (QAT 멀티모달) |
| **펑션콜링** | 음성→도구 호출 15.5 s / 이미지→도구 호출 40.7 s E2E | gemma3-1b, gemma-4-E2B |

## Requirements

- Unity `6000.4.6f1`
- Android 기기 (`adb` 연결) — 권장 사양: Snapdragon 865급 이상, RAM 6 GB+
- Windows + PowerShell (빌드/테스트 스크립트 실행)
- Unity Android Build Support (SDK/NDK)
- Docker Desktop + Git for Windows Bash — 커스텀 AAR 재빌드 시에만

## Quick Start (Android)

1. **모델 배치**: 아래 추천 표의 모델을 받아 `Assets/StreamingAssets/` 하위
   경로에 배치 (모델 파일은 저장소에 포함되지 않음).
2. **APK 빌드**: Unity 메뉴 `LiteRT-LM/Android/...` 또는 커맨드라인
   (`Tools/Windows/Build-LiteRtLmAndroid*.ps1`). 테스트 씬별 APK는 필요한
   모델만 패키징합니다.
3. **실행/스모크**: `Tools/Windows/Run-LiteRtLmAndroidDeviceBenchmarks.ps1`
   (LLM 벤치), `Run-LiteRtLmAndroidAsrSmokeTest.ps1 -DeviceSerial <serial>`
   (ASR 스모크). 결과는 `Builds/Logs/AndroidDeviceRuns/`에 JSON/로그로 저장.

네이티브 브리지(`Assets/Plugins/Android/litertlm-unity-bridge.aar`)가 제공하는
Unity API (`LiteRtLmUnityClient`):

- `Initialize(modelPath, backend, …, visionBackend, audioBackend)` — 멀티모달은
  vision/audio 백엔드 지정 필수, 이미지 턴은 `maxNumTokens 4000` 권장
- `SendMessage(text)` / `SendMessageWithMedia(text, imageBytes|imagePath, audioPath)`
- `RunWhisperAsrSmoke(...)` / `RunQwen3AsrSmoke(...)` — ASR 전용 경로
  (whisper 80/128-mel 자동 감지)

## Recommended LLM Models (Android 기준)

모델은 `Assets/StreamingAssets`의 모델별 하위 폴더(`LLM/<model>/`,
`Multimodal/<model>/`)에 배치합니다. 순위는 **실기기(Snapdragon 865) 측정
기준**이며 Windows 수치는 [`docs/llm-details.md`](docs/llm-details.md) 참조.

| Rank | Model (StreamingAssets path) | Source | Device 성능 / 용도 |
| ---: | --- | --- | --- |
| 1 | `Multimodal/gemma-4-e2b/gemma-4-E2B-it.litertlm` (2.6 GB) | [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) | 메모리가 허용되는 기기에서 제1픽. 공식 QAT wNa8o8. 디바이스에서 이미지 인식(GPU 7.6 s)·오디오 전사(4.1 s)·멀티모달 FC(38 s) 검증. FC 19/20 |
| 2 | `LLM/qwen2.5-0.5b/Qwen2.5-0.5B-Instruct_wi4b64_ekv1280.litertlm` (265 MB) | 자체 int4 (source: [litert-community/Qwen2.5-0.5B-Instruct](https://huggingface.co/litert-community/Qwen2.5-0.5B-Instruct) f32) | **디바이스 최속 채팅: 35.5 tok/s CPU** — 공식 q8(521 MB) 대비 절반 크기에 +38 % 속도. 저사양·저지연 제1픽 (FC는 부적합) |
| 3 | `LLM/qwen3-0.6b/qwen3_0_6b_mixed_int4.litertlm` (475 MB) | [litert-community/Qwen3-0.6B](https://huggingface.co/litert-community/Qwen3-0.6B) | 소형 FC 픽: 디바이스 20.9 tok/s CPU, FC 18/20 (QwenHermes 프롬프트). think/no_think 지원 |
| 4 | `LLM/lfm2.5-1.2b/LFM2.5-1.2B-Instruct_int4.litertlm` (702 MB) | LiquidAI LFM2.5-1.2B-Instruct (LiteRT export) | 중형 FC 픽: 디바이스 16.8 tok/s CPU, FC 17/20. v0.14 런타임 필수 |
| 5 | `LLM/gemma3-1b/gemma3-1b-it-int4.litertlm` (557 MB) | [litert-community/Gemma3-1B-IT](https://huggingface.co/litert-community/Gemma3-1B-IT) | 채팅 폴백: 16.0 tok/s CPU. GPU는 prefill만 유리(184 vs 101 tok/s — 긴 입력용). FC 부적합(3/20) |

**디바이스 백엔드 지침**: 채팅(디코드 위주)은 **CPU**, 긴 프롬프트/이미지
인코딩은 **GPU**. Adreno에서 GPU 디코드가 CPU보다 느린 것은 스텝당 OpenCL
왕복 오버헤드 때문인 구조적 특성입니다 (회귀 아님 — 샘플러는 GPU 정상 동작).

## Recommended ASR Models (Android 기준)

ASR 모델은 `Assets/StreamingAssets/ASR/<model>/`에 해당 `tokenizer.json`과
함께 배치합니다 (티어별 토크나이저가 다름 — medium과 large-v3/turbo는 각자
전용). 순위는 **실기기 검증 기준**. 전체 CER/WER 매트릭스:
[`docs/benchmarks/asr-model-matrix.md`](docs/benchmarks/asr-model-matrix.md).

| Use case (device) | Model (StreamingAssets path) | Size | Device 결과 |
| --- | --- | ---: | --- |
| 음성 명령 (짧은 발화) 제1픽 | `ASR/qwen3-asr-0.6b/qwen3_asr_0.6b_5s_i8.tflite` | 794 MB | 짧은 한국어 명령("볼륨 업" 구/신 녹음 모두) 전부 인식하는 유일 모델군. ~0.5 s/step CPU. 숫자를 한글로 표기(`이천이십오년`) |
| 문장 전사 최고 정확도 | `ASR/whisper-large-v3-turbo/whisper_large_v3_turbo_30s_i4.tflite` | 755 MB | 디바이스 게이트 3/3 통과 — whisper 중 유일하게 볼륨 명령까지 인식. 매트릭스 종합 1위(8/9, CER 0.000). 단 클립당 ~21–24 s CPU (배치/비실시간 용도) |
| 균형(크기·속도·정확도) | `ASR/whisper-base/whisper_base_30s_i8.tflite` | 77 MB | 문장 한국어 CER 0.000, 긴 클립 ~2.7 s. 주의: 1.2 s 미만 초단클립은 디바이스에서 불안정(mel 수치 특성) — 음성 명령은 위 두 모델 사용 |
| 초소형(영어 위주) | `ASR/whisper-tiny/whisper_tiny_30s_i8.tflite` | 41 MB | 영어 CER 0.000. 한국어 연도 오인 + 초단클립 불안정 |
| 정확도 레퍼런스 | `ASR/whisper-large-v3/whisper_large_v3_30s_i4.tflite` | 1148 MB | 문자 단위 완벽하나 turbo보다 3–7배 느림 — 비교 기준용 |
| 중간 티어 | `ASR/whisper-medium/whisper_medium_30s_i8.tflite` (i4: 664 MB) | 832 MB | 7/9 정확 — base와 turbo 사이 절충 |

**대안 경로**: gemma-4 오디오 입력(LLM 1번)으로도 전사 가능 — LLM이 이미
상주할 때 추가 모델 없이 **전사+펑션콜링을 한 턴에** 처리 (디바이스 4.1 s,
내용 정확).

모델 출처: whisper tiny/base는
[litert-community](https://huggingface.co/litert-community/whisper-tiny)
([base](https://huggingface.co/litert-community/whisper-base)); medium /
large-v3 / turbo 및 모든 i8/i4 티어는 **프로젝트 자체 양자화**(int4 최소 티어
정책, `dynamic_wi4b64_afp32` + 민감 스코프 i8 혼합 — `External/community-release/`
에 커뮤니티 공개용 사본과 매니페스트). 토크나이저:
[openai/whisper-*](https://huggingface.co/openai/whisper-tiny) 티어별.
Qwen3-ASR은 공식 tflite + 프로젝트 JNI 포팅.

## Test Scenes

테스트 씬은 `Assets/Scenes/Tests/`에 있으며
`Assets/Scripts/LiteRTLM/Editor/LiteRtLmTestSceneGenerator.cs`로 생성합니다
(메뉴 `LiteRT-LM/Test Scenes/Generate All`, 배치모드
`-executeMethod LiteRTLM.Unity.Editor.LiteRtLmTestSceneGenerator.GenerateAllFromCommandLine`).

| Scene | 용도 (모두 Android 대상, 디바이스 검증 상태) |
| --- | --- |
| `LiteRtLmLlmChatTestScene` | 멀티턴 채팅 — 모델 5종 드롭다운, Qwen3 think/no_think 토글 |
| `LiteRtLmAsrTestScene` | ASR — 모델 드롭다운 × 10클립 오디오 드롭다운 (whisper + qwen3 모드) |
| `LiteRtLmMultimodalTestScene` | 이미지 + 오디오 입력 (`SendMessageWithMedia`, gemma-4) — ✅ 디바이스 PASS |
| `LiteRtLmAsrFunctionCallingTestScene` | 음성 → 전사 → 도구 호출 파이프라인 — ✅ 디바이스 PASS (15.5 s E2E) |
| `LiteRtLmMultimodalFunctionCallingTestScene` | 이미지 + 발화 → 도구 호출 — ✅ 디바이스 PASS (40.7 s E2E) |

레거시 스모크/벤치 씬(`AndroidSmokeTest`, `ConversationTest`,
`FunctionCallingBenchmark`)도 같은 폴더에 있습니다.

## Benchmark Results (2026-07-23, 실기기 46a880a0)

측정 지표 읽는 법:

- **prefill tok/s** — 프롬프트(입력) 처리 속도. 긴 문서/이미지 입력의 대기시간을 좌우.
- **decode tok/s** — 응답 생성 속도. 채팅 체감 속도 (10 tok/s ≈ 한글 5~7자/초).
- **CER / WER** — 문자/단어 오류율. 0.000 = 기대 문장과 완전 일치.
- **RTF** — 처리시간 ÷ 오디오 길이. 1.0 미만 = 실시간보다 빠름.

### LLM (디바이스 CPU decode)

| 모델 | 크기 | decode tok/s | prefill tok/s | 메모 |
| --- | ---: | ---: | ---: | --- |
| Qwen2.5-0.5B wi4b64 (자체 int4) | 265 MB | **35.5** | — | 최속. q8 대비 +38 % — int4는 대역폭이 병목인 모바일에서 크기와 속도를 동시에 얻음 |
| Qwen3-0.6B mixed_int4 | 475 MB | 20.9 | 31.8 | FC 소형 픽 |
| LFM2.5-1.2B int4 | 702 MB | 16.8 | — | FC 중형 픽 |
| gemma3-1b int4 | 557 MB | 16.0 (GPU 13.7) | 101 (GPU **184**) | GPU는 prefill만 유리 |
| gemma-4-E2B QAT | 2.6 GB | (멀티모달 전용) | — | 이미지 7.6 s(GPU) / 오디오 4.1 s / 멀티모달 FC 38 s |

### ASR (디바이스 게이트 + 데스크톱 정밀 매트릭스)

| 모델 | 크기 | 정확 일치 | CER (한/영) | 디바이스 특기 |
| --- | ---: | ---: | --- | --- |
| **whisper-turbo i4** | 755 MB | **8/9** | 0.000 / 0.000 | 게이트 3/3 — 볼륨 명령 포함 |
| whisper-large-v3 i4 | 1.1 GB | 7/9 | 0.000 / 0.000 | 레퍼런스 |
| whisper-base i8 | 77 MB | 6/9 | 0.000 / — | 문장 전사 실용 픽 |
| Qwen3-ASR-0.6B i8 | 794 MB | 4/9* | 0.069 / 0.057 | *숫자 표기 차이만. 음성 명령 최강 |
| whisper-tiny i8 | 41 MB | 3/9 | 0.288 / 0.024 | 영어 특화 |

양자화 요약: **i8 = 속도, i4 = 크기** — 품질은 혼합 레시피(민감 스코프만 i8)로
유지하며 전 티어 한국어 클립 검증 후 배포. int2/Q5/1.58-bit는 LiteRT 커널이
없어 불가.

### Function Calling (20케이스 채점 + 디바이스 E2E)

| 티어 | 모델 | 통과 | 판정 |
| --- | --- | ---: | --- |
| 플래그십 | gemma-4-E2B QAT | **19/20** | 메모리 여유 시 최선. 디바이스 멀티모달 FC 40.7 s E2E PASS |
| 중형 | LFM2.5-1.2B int4 (+Hermes) | 17/20 | 최속 FC. pythonic 형식 → 파서 확장 시 상승 여지 |
| 소형 | Qwen3-0.6B mixed_int4 (+QwenHermes) | 18/20 | 475 MB |
| 부적합 | gemma3-1b / Qwen2.5 계열 | 2–8/20 | FC 라우터 사용 금지 (단 gemma3-1b은 음성 FC 파이프라인의 LLM으로는 디바이스 PASS — 단일 도구 시나리오) |

디바이스 E2E: 음성 → 도구 호출 **15.5 s**, 이미지+발화 → 도구 호출 **40.7 s**.

### 벤치마크 문서 (근거 데이터)

| 문서 | 내용 |
| --- | --- |
| [`asr-model-matrix.md`](docs/benchmarks/asr-model-matrix.md) | ASR 전 티어 × 10클립 CER/WER/RTF 매트릭스, 클립별 전사 원문 |
| [`fc-model-benchmark.md`](docs/benchmarks/fc-model-benchmark.md) | FC 20케이스 모델별 상세 (케이스별 실패 분석 포함) |
| [`device-cycle1-baseline.md`](docs/benchmarks/device-cycle1-baseline.md) | 실기기 PDCA 사이클 1–3 전체 기록 + 최종 PASS/FAIL 원장 |
| [`gemma4-gguf-vs-litertlm.md`](docs/benchmarks/gemma4-gguf-vs-litertlm.md) | GGUF(llama.cpp) 대비 비교 — Windows 실험용 참고 |
| [`short-utterance-asr-research.md`](docs/benchmarks/short-utterance-asr-research.md) | 짧은 음성 인식 개선 연구 (적용: RMS 정규화·VAD·EOS 가드) |
| [`session-final-report-20260723.md`](docs/benchmarks/session-final-report-20260723.md) | v0.14 업그레이드 세션 종합 보고서 |

## AAR 재빌드 (네이티브 커스텀 변경 시)

Docker 기반: `Tools/Windows/Build-LiteRtLmUnityAarFromPatch.ps1 -SourceRoot
<pristine v0.14.0 체크아웃>` — `Tools/UnityAar/litert-lm-unity-aar.patch`를
적용해 빌드 후 `Assets/Plugins/Android/`에 배포합니다. 패치에는 Qwen3-ASR
모드, 멀티모달 브리지, 128-mel whisper 지원, 단발화 오디오 전처리(RMS
정규화·VAD·EOS 가드)가 포함됩니다.

## Windows (개발·검증 보조)

Windows 에디터에서 디바이스 배포 전 로직을 검증하는 용도입니다.

- `Tools/Windows/litert_lm_main.windows_x86_64.exe` — 커스텀 펑션콜링 플래그
  (`--tools_json_file`, `--enable_constrained_decoding`, `--output_message_json`,
  `--system_message_file`, `--messages_json_file`) 포함 v0.14 빌드.
- `litert_lm_advanced_main.windows_x86_64.exe` — `[audio:<path>]`/`[image:<path>]`
  프롬프트 태그 + `--audio_backend` (Windows ASR/멀티모달 경로).
- GPU가 Windows 기본 백엔드 (WebGPU/Dawn/D3D12, RTX 4090에서 CPU 대비 2배) —
  `LiteRtLmWindowsCliClient`가 GPU 실패 시 CPU로 자동 폴백. ※ Android와
  반대 특성이니 혼동 주의.
- Windows ASR 스모크: `Tools/Windows/Run-LiteRtLmWindowsAsrSmokeTest.ps1`.

## Details

- LLM 세부 설명: [`docs/llm-details.md`](docs/llm-details.md)
- ASR 세부 설명: [`docs/asr-details.md`](docs/asr-details.md)
- 커뮤니티 공개용 자체 양자화 모델 + 매니페스트: `External/community-release/`
- 세션/업그레이드 핸드오프: [`docs/handoffs/v0.14-upgrade-handoff.md`](docs/handoffs/v0.14-upgrade-handoff.md)
