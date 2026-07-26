# LiteRT-LM-Unity

**Android 온디바이스 AI**를 위한 Unity 통합 프로젝트입니다. 네트워크 없이 기기
단독으로 LLM 채팅 · 음성 인식(ASR) · 이미지 인식 · 펑션콜링(음성/멀티모달 →
도구 호출)을 실행합니다. Windows 에디터 환경은 배포 전 로직 검증 보조용입니다.

- Framework: **LiteRT-LM v0.14.0** (`External/LiteRT-LM`, 커스텀은
  `Tools/UnityAar/litert-lm-unity-aar.patch`로 관리), `.litertlm` 포맷 1.5.0
- 실기기 검증: **Snapdragon 865 (kona) / 7.5 GB RAM / Android 12**에서 4대 기능
  전부 PASS — 6회 PDCA 사이클, 80+ 런, 크래시 0
  ([검증 원장](docs/benchmarks/device-cycle1-baseline.md))
- 자체 학습·양자화 모델 공개:
  [litert-community/whisper-acft](https://huggingface.co/litert-community/whisper-acft) ·
  [leuconoe/whisper-acft-ko](https://huggingface.co/leuconoe/whisper-acft-ko) ·
  [leuconoe/litert-lm-unity-quantized](https://huggingface.co/leuconoe/litert-lm-unity-quantized)

## 온디바이스 핵심 기능 (실기기 검증 완료)

| 기능 | 디바이스 실측 | 사용 모델 |
| --- | --- | --- |
| **LLM 채팅** | 16~35.5 tok/s (CPU) | Qwen2.5/Qwen3/gemma3/LFM2.5 계열 |
| **음성 인식 (ASR)** | 짧은 명령 0.7~0.8 s, 문장 전사 CER 0.000 | whisper ACFT-KO 5s (base/turbo) |
| **이미지 인식** | 이미지 묘사 7.6 s (GPU) | gemma-4-E2B (QAT 멀티모달) |
| **펑션콜링** | 음성→도구 15.5 s / 이미지→도구 40.7 s E2E | gemma3-1b, gemma-4-E2B |

## Requirements

- Unity `6000.4.6f1` + Android Build Support (SDK/NDK)
- Android 기기 (`adb` 연결) — 권장 Snapdragon 865급 이상, RAM 6 GB+
- Windows + PowerShell (빌드/테스트 스크립트), Docker Desktop (AAR 재빌드 시에만)

## Quick Start (Android)

1. **모델 배치** — 아래 추천 모델을 받아 `Assets/StreamingAssets/` 하위에 배치
   (모델 파일은 저장소에 포함되지 않음).
2. **APK 빌드** — Unity 메뉴 `LiteRT-LM/Android/...` 또는
   `Tools/Windows/Build-LiteRtLmAndroid*.ps1`. 씬별 APK는 필요한 모델만 패키징.
3. **실행/스모크** — `Run-LiteRtLmAndroidDeviceBenchmarks.ps1` (LLM),
   `Run-LiteRtLmAndroidAsrSmokeTest.ps1 -DeviceSerial <serial>` (ASR).
   결과는 `Builds/Logs/AndroidDeviceRuns/`에 JSON/로그 저장.

네이티브 브리지 API(`LiteRtLmUnityClient`): `Initialize(...)` /
`SendMessage(text)` / `SendMessageWithMedia(text, image, audio)` /
`RunWhisperAsrSmoke(...)` / `RunQwen3AsrSmoke(...)`.
인자·주의사항은 [`docs/llm-details.md`](docs/llm-details.md) ·
[`docs/asr-details.md`](docs/asr-details.md).

## Recommended Models — 기기 RAM 예산 기준

**모바일에서 쓸 수 있는지**를 1차 기준으로 골랐습니다. 아래 표의 조합은 전부
Snapdragon 865 / 7.7 GiB 기기에서 크래시·OOM 없이 실측된 것입니다.
데스크톱 정확도 1위 티어라도 디바이스 지연이 실사용 범위를 벗어나면 제외했고,
그 근거는 [`docs/asr-details.md`](docs/asr-details.md) ·
[`docs/llm-details.md`](docs/llm-details.md)에 있습니다.

| 기기 RAM | 권장 조합 | 상주 합계 |
| --- | --- | ---: |
| **4~6 GB** | Qwen2.5-0.5B i4 (채팅) + whisper-base-acft-ko 5s (명령) | ~370 MB |
| **6~8 GB** | Qwen3-0.6B i4 (채팅+FC) + base-acft-ko + whisper-base 30s (받아쓰기) | ~650 MB |
| **8 GB+** | gemma-4-E2B QAT (채팅·이미지·오디오·FC 단일 모델) | ~3.6 GB PSS |

ASR 정확도 폴백(turbo-acft 883 MB)이나 장문 전사(qwen3-asr 794 MB)는 **필요할
때만 로드하고 즉시 해제**하는 전제입니다. LLM과 동시 상주시키지 마세요.

### LLM — `Assets/StreamingAssets/`

| Model | Size | Device 실측 | 적합 |
| --- | ---: | --- | --- |
| `LLM/qwen2.5-0.5b/…_wi4b64_ekv1280.litertlm` | 265 MB | **35.5 tok/s** (최속), init 1.4 s | 저사양 채팅 제1픽. FC는 부적합 |
| `LLM/qwen3-0.6b/qwen3_0_6b_mixed_int4.litertlm` | 475 MB | 20.9 tok/s, FC 18/20 | 소형 FC 제1픽. think/no_think |
| `LLM/gemma3-1b/gemma3-1b-it-int4.litertlm` | 557 MB | 16.0 tok/s, PSS 399 MB | 채팅 폴백. FC 부적합 |
| `LLM/lfm2.5-1.2b/LFM2.5-1.2B-Instruct_int4.litertlm` | 702 MB | 16.8 tok/s, FC 17/20 | 중형 FC |
| `Multimodal/gemma-4-e2b/gemma-4-E2B-it.litertlm` | 2.6 GB | 이미지 7.6 s / 오디오 4.1 s / FC 19-20, **PSS ~3.6 GB** | 8 GB+ 전용. 멀티모달이 필요할 때만 |

**백엔드**: 채팅(디코드)은 **CPU**, 긴 프롬프트·이미지 인코딩은 **GPU**.
Adreno에서 GPU 디코드가 CPU보다 느린 것은 스텝당 OpenCL 왕복 오버헤드로 인한
구조적 특성입니다(회귀 아님).

### ASR — `Assets/StreamingAssets/ASR/<model>/` (티어별 `tokenizer.json` 필수)

| 용도 | Model | Size | Device 실측 |
| --- | --- | ---: | --- |
| **음성 명령 제1픽** | `whisper-base-acft-ko/acft_base_5s_drq.tflite` | 101 MB | E2E **0.7–0.8 s**, 정상 음량 명령 전부 exact |
| 받아쓰기 / 문장 전사 | `whisper-base/whisper_base_30s_i8.tflite` | 77 MB | 한국어 CER 0.000, 긴 클립 ~2.7 s |
| 정확도 폴백 (온디맨드) | `whisper-turbo-acft-ko/acft_turbo_5s_drq.tflite` | 883 MB | 웜 **1.9 s** / 콜드 4.0 s. 조용한 녹음까지 **5/5 유일** |
| 장문 전사 (배치) | `qwen3-asr-0.6b/qwen3_asr_0.6b_5s_i8.tflite` | 794 MB | 5 s 청크 루프로 길이 무제한. 98 s 오디오 → 4.2분(RTF ≈2.6) |

**실시간 대화형은 앞의 두 줄로 충분합니다** (합계 178 MB). 뒤의 두 줄은 무겁고
느려서 상시 상주용이 아니라 *필요 시 로드 → 해제* 패턴 전용입니다.

⚠️ **모바일 부적합 — README에서 제외한 티어**: whisper 30 s의 large-v3-turbo
i4(755 MB, **클립당 21–24 s**) · large-v3 i4(1.1 GB, 그보다 3–7배 느림) ·
medium i8(832 MB) · medium-acft-ko(826 MB, turbo-acft보다 느리고 부정확).
데스크톱 정확도 매트릭스 1위는 turbo-30s지만 **디바이스 지연이 20초를 넘어
대화형으로 못 씁니다** — 비실시간 배치·정확도 레퍼런스 용도로만 쓰세요
([근거·전체 10티어 표](docs/asr-details.md)).
⚠️ **tiny 티어는 한국어 명령 비권장** (디바이스 1/4 exact).
⚠️ **whisper 30 s 모델에 30 s 초과 오디오 직접 입력 금지** (절단+토큰 캡+조기
종료). 장문은 qwen3 청크 경로 사용.

**VAD**: 모든 ASR 경로가 `vadMode` 지원 — `energy`(기본, 비용 0) /
`ai`(Silero 1.25 MB) / `off`. Unity 라이브 마이크도 동일 파라미터로 자동
엔드포인팅.

전체 라인업(10티어) · VAD 파라미터 · ACFT-KO 학습 배경과 **수치 해석 주의**:
[`docs/asr-details.md`](docs/asr-details.md).

## Test Scenes

`Assets/Scenes/Tests/` — 생성은 메뉴 `LiteRT-LM/Test Scenes/Generate All`
(`LiteRtLmTestSceneGenerator.cs`, 배치모드 `-executeMethod` 지원).

| Scene | 용도 (전부 Android 대상) |
| --- | --- |
| `LiteRtLmLlmChatTestScene` | 멀티턴 채팅 — 모델 5종 드롭다운, think/no_think 토글 |
| `LiteRtLmAsrTestScene` | ASR — 모델 × 10클립 드롭다운, **Mic** 라이브 캡처(VAD 자동 엔드포인팅), **Continuous** 상시 청취 루프(유한 큐 + 백그라운드 전사) |
| `LiteRtLmMultimodalTestScene` | 이미지 + 오디오 입력 — ✅ 디바이스 PASS |
| `LiteRtLmAsrFunctionCallingTestScene` | 음성 → 전사 → 도구 호출 — ✅ PASS (15.5 s) |
| `LiteRtLmMultimodalFunctionCallingTestScene` | 이미지 + 발화 → 도구 호출 — ✅ PASS (40.7 s) |
| `LiteRtLmTranslateTestScene` | 번역 2엔진 — Whisper Direct(`<\|translate\|>`) / ASR+LLM(EN·JA·ZH) |

레거시 스모크 씬(`AndroidSmokeTest`, `ConversationTest`,
`FunctionCallingBenchmark`)도 같은 폴더에 있습니다.

## Benchmarks (실기기 46a880a0, 2026-07-23~26)

지표 읽는 법 — **decode tok/s**: 응답 생성 속도(10 tok/s ≈ 한글 5~7자/초) ·
**prefill tok/s**: 입력 처리 속도(긴 문서·이미지 대기시간) · **CER/WER**:
문자/단어 오류율(0.000 = 완전 일치) · **RTF**: 처리시간÷오디오 길이(<1 = 실시간 초과).

핵심 결과:

- **LLM 최속** Qwen2.5-0.5B 자체 int4 265 MB / 35.5 tok/s — 공식 q8 대비 +38 %
- **ASR 실사용 최적** base-acft-ko 5s 101 MB / 0.7–0.8 s — 정확도만 보면
  whisper-turbo 30s i4가 8-9 exact로 1위지만 디바이스에서 클립당 21–24 s라
  대화형 불가. **정확도 1위 ≠ 모바일 1픽**
- **FC 최고** gemma-4-E2B QAT 19/20, 소형은 Qwen3-0.6B 18/20 (475 MB)
- **양자화**: i8 = 속도, i4 = 크기 — 민감 스코프만 i8로 혼합해 품질 유지.
  전 티어 한국어 클립 검증 후 배포. int2/Q5/1.58-bit는 LiteRT 커널 부재로 불가

| 문서 | 내용 |
| --- | --- |
| [`asr-model-matrix.md`](docs/benchmarks/asr-model-matrix.md) | ASR 전 티어 × 10클립 CER/WER/RTF + 클립별 전사 원문 (+게이트 지표 유효성 Addendum 3) |
| [`fc-model-benchmark.md`](docs/benchmarks/fc-model-benchmark.md) | FC 20케이스 모델별 상세 + 케이스별 실패 분석 |
| [`device-cycle1-baseline.md`](docs/benchmarks/device-cycle1-baseline.md) | 실기기 PDCA 사이클 전체 기록 + 최종 PASS/FAIL 원장 |
| [`short-utterance-asr-research.md`](docs/benchmarks/short-utterance-asr-research.md) | 짧은 음성 인식 개선 연구 (적용: RMS 정규화·VAD·EOS 가드) |
| [`gemma4-gguf-vs-litertlm.md`](docs/benchmarks/gemma4-gguf-vs-litertlm.md) | GGUF(llama.cpp) 대비 비교 (참고) |
| [`session-final-report-20260723.md`](docs/benchmarks/session-final-report-20260723.md) | v0.14 업그레이드 세션 종합 보고서 |

## AAR 재빌드 (네이티브 변경 시)

`Tools/Windows/Build-LiteRtLmUnityAarFromPatch.ps1 -SourceRoot <pristine v0.14.0>`
— `Tools/UnityAar/litert-lm-unity-aar.patch`를 적용해 Docker로 빌드 후
`Assets/Plugins/Android/`에 배포. 패치에 Qwen3-ASR 모드, 멀티모달 브리지,
whisper 시그니처 자동 감지(80/128-mel), 단발화 전처리(RMS·VAD·EOS 가드),
translate 태스크 토큰이 포함됩니다.

⚠️ `-SkipImageBuild`는 Docker 이미지에 구워진 **낡은 소스**를 빌드합니다 —
패치 변경 후에는 이미지 재빌드 필수.

## Windows (개발·검증 보조)

디바이스 배포 전 로직 검증 용도이며, 성능 특성은 Android와 다릅니다.

- `Tools/Windows/litert_lm_main.windows_x86_64.exe` — 커스텀 FC 플래그 포함 v0.14 빌드
- `litert_lm_advanced_main.windows_x86_64.exe` — `[audio:]`/`[image:]` 프롬프트 태그
- GPU가 Windows 기본 백엔드(WebGPU/Dawn/D3D12), 실패 시 CPU 자동 폴백.
  ※ Android와 반대 특성이니 혼동 주의
- ASR 스모크: `Tools/Windows/Run-LiteRtLmWindowsAsrSmokeTest.ps1`

## Details

| 문서 | 내용 |
| --- | --- |
| [`docs/llm-details.md`](docs/llm-details.md) | LLM 티어·백엔드·디바이스 벤치 상세 |
| [`docs/asr-details.md`](docs/asr-details.md) | ASR 전체 라인업·VAD·ACFT-KO 학습 배경·스모크 커맨드 |
| [`docs/handoffs/asr-training-program-handoff.md`](docs/handoffs/asr-training-program-handoff.md) | ASR 학습 프로그램 인계 (클린 ACFT 레시피, kspon 폐기 기록, 재학습 규칙) |
| [`docs/handoffs/v0.14-upgrade-handoff.md`](docs/handoffs/v0.14-upgrade-handoff.md) | 프레임워크 v0.14 업그레이드 인계 |
| `External/community-release/` | 커뮤니티 공개용 자체 양자화 모델 + 매니페스트 |
