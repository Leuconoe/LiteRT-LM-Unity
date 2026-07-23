# 최종 보고서 — LiteRT-LM v0.14 업그레이드 및 모델 확장 (2026-07-23)

요청 범위: LiteRT-LM 프레임워크 업데이트, gemma-4 QAT 모델 LiteRT화·테스트, 신규
ASR/LLM 모델 평가, Windows ASR/LLM 환경, GPU 가속+CPU 폴백, 테스트 씬, 양자화
(int4 최소 기준) 확산, 실기기(46a880a0) PDCA 검증.

## 1. 완료 요약 (19개 태스크 중 15개 완료)

### 프레임워크 / 빌드
| 항목 | 결과 |
| --- | --- |
| LiteRT-LM v0.14.0 업그레이드 | ✅ `unity-v0.14.0` 브랜치. 패치 재작성(v0.11→v0.14 API 변경 2건 수정 포함). `.litertlm` 포맷 1.5.0 호환 유지 |
| Android AAR | ✅ take4 배포(16:32). 신규: qwen3 ASR 모드, `sendMessageWithMedia`(이미지/오디오), vision/audio 백엔드 init, RMS 정규화+VAD+EOS 가드, 128-mel whisper 자동 감지. take5(디코드 형상 바인딩+featureMd5) 빌드 진행 중 |
| Windows 바이너리 | ✅ v0.14 exe 2종(13:53) + DLL 7종(신규 libwebgpu_dawn 포함). 커스텀 FC 플래그 5종 검증. 빌드 함정 해결: 한국어 CP949 코드페이지(`/utf-8`), MAX_PATH(`output_base=C:/bzl-lm`), VS2022 고정 |
| Windows GPU + CPU 폴백 | ✅ v0.14로 GPU 부활(RTX 4090, 53 tok/s = CPU 2배). `LiteRtLmWindowsCliClient`에 GPU 실패→CPU 1회 재시도 + 세션 건강 플래그. Windows 기본 백엔드 GPU 전환 |
| Windows ASR 환경 | ✅ gemma-4 오디오 경로 검증(전사 정확) + `Run-LiteRtLmWindowsAsrSmokeTest.ps1` 신규 |

### 모델 판정
| 모델 | 판정 | 근거 |
| --- | --- | --- |
| gemma-4-E2B QAT (mobile-transformers) | ✅ 변환 불필요 | 기존 `gemma-4-E2B-it.litertlm`이 이미 QAT wNa8o8 공식 빌드(SHA 일치). DIY 변환은 도구가 양자화 체크포인트 거부로 불가 |
| Qwen3-ASR-0.6B | ✅ 채택·완료 | 공식 tflite + 자체 JNI 포팅. 한국어 검증, 디바이스 전 클립 통과 |
| Qwen3.5-0.8B-MTP | ❌ 불가 | MTP=llama.cpp 전용, 아키텍처 미지원, 커뮤니티 포트는 v0.14에서 출력 붕괴 |
| VibeVoice-ASR | ❌ 불가 | 8.7B — 온디바이스 부적합 |
| Qwen3-ASR-1.7B | ⏸ 보류 | 포트 없음(1-3주 공수). 0.6B 품질 충분 |
| Bonsai-1.7B | ⏸ 스킵 권장(결정 대기) | 1-bit 이점이 LiteRT 변환 시 소멸 |
| NVFP4 | ❌ 폐기(사용자 지시) | LiteRT 경로 없음 |
| 이미지 생성 | ❌ 패스(사용자 지시) | gemma-4는 입력 전용, SD-on-LiteRT 306s/장 |
| LFM2.5-1.2B int4 (FC) | ✅ 채택 | 파인튜닝 불필요 기준 충족. Windows/디바이스 로드·추론 통과 |
| FunctionGemma-270M | 비교군 유지 | 파인튜닝 전제 설계라 선정 제외 |
| Hammer2.1 | ❌ 제외 | CC-BY-NC 라이선스 |

### 양자화 (int4 최소 기준 — 사용자 정책)
- **레시피 확정**: `dynamic_wi4b64_afp32`(블록64) 기본 + 민감 스코프(임베딩/인코더) i8 혼합.
  wi4c(채널)·wi4b32·int2·Q5·1.58b는 실측/조사로 불가 판정.
- **wi4b64 = 가중치 4bit + 64개 블록당 fp16 스케일 + 활성 fp32** — 크기 절반(i8 대비),
  대역폭 바운드 기기에서 속도 향상(+38% 실측).
- **자체 제작·배치**: whisper-base i8/i4, whisper-tiny i4, whisper-medium i8/i4,
  whisper-large-v3 i8/i4, whisper-turbo i4, Qwen3-ASR i4(→환각으로 제거),
  qwen2.5-0.5b i4(265MB), qwen2.5-1.5b i4(790MB). `.litertlm` 언팩→양자화→리팩
  파이프라인 확립(litert-lm-builder).
- 적용 불가: gemma3-270m/FunctionGemma(F32 소스 없음, i8→i4는 no-op 실증).

### ASR 벤치마크 (재녹음 10클립 × CER/WER/RTF — asr-model-matrix.md)
- **최고**: whisper-turbo i4 (755MB, 8/9 정확, CER 0.000/0.000) — 데스크톱.
- 균형: base i8 (77MB, 재녹음 클립 기준 6/9, CER ko 0.000).
- 디바이스 음성명령 최강: qwen3-asr i8 (볼륨 업 구/신 모두 통과).
- 단발화 개선: RMS 정규화+VAD+EOS 가드 구현(핫워드는 사용자 지시로 제외 — 특정 값
  예측 금지). 라벨 오류 2건은 재녹음으로 해소.
- 잔여: 디바이스 whisper 초단클립 수치 불일치(원인 후보: mel/STFT i8 수치) — take5
  featureMd5 진단으로 A/B 예정.

### FC 벤치마크 (20케이스, Windows CPU — fc-model-benchmark.md)
| 티어 | 권장 | 성적 |
| --- | --- | --- |
| 플래그십 | gemma-4-E2B QAT | 19/20, 15.5 tok/s |
| 중형 | LFM2.5-1.2B int4 (+Hermes 프롬프트) | 17/20, 23.6 tok/s(최속) |
| 소형 | qwen3_0_6b_mixed_int4 (+QwenHermes) | 18/20, 475MB |
| 탈락 | gemma3-1b, qwen2.5 계열 | 2~8/20 |

### GGUF 비교 (gemma4-gguf-vs-litertlm.md)
- llama.cpp CUDA 디코드 241-245 tok/s vs litertlm CPU 13.6 (~16배) — 단 GGUF는
  Android 경로 없음. 역할: litertlm=제품, llama.cpp=Windows 실험.
- v0.14로 litertlm Windows GPU 부활(이전 3/3 실패 해소).

### 디바이스 PDCA (46a880a0, kona — device-cycle1-baseline.md)
- **사이클 1**: LLM 5종 전원 PASS. 자체 i4가 최속(35.5 tok/s, q8 +38%). LFM2.5 디바이스
  지원 확인. GPU 디코드 열세는 Adreno 650 구조적 특성으로 규명(샘플러 정상, 회귀 아님
  — 디코드는 CPU, 프리필·멀티모달은 GPU 권장). qwen3-asr 전 클립 통과.
  과거 크래시 원인 규명(낡은 APK의 SIGSEGV — 메모리 아님).
- **사이클 2**: **이미지 인식 PASS**(apples.jpg 정확 묘사, GPU 7.6s = CPU의 3.1배),
  **오디오 멀티모달 PASS**(전사 일치, 4.1s). tiny/base 회귀 0. 13런 크래시 0.
  멀티모달 필수 조건 확립: visionBackend/audioBackend init + maxNumTokens 4000.
- **사이클 3(진행)**: take5로 turbo i4 디바이스 재시도 + featureMd5 A/B + FC 씬 검증.

### 테스트 씬 / UX (#11 — 코드 완료, 씬 생성 대기)
- 씬 5종 러너 + 생성기(`LiteRtLmTestSceneGenerator`, 메뉴 `LiteRT-LM/Test Scenes/Generate All`)
  구현·컴파일 완료. 기존 씬 `Assets/Scenes/Tests/`로 이동.
- ASR 씬: 모델 드롭다운 + 10클립 오디오 드롭다운. 채팅 씬: think/no_think 토글(Qwen3),
  모델 5종 드롭다운. 멀티모달 씬: SendMessageWithMedia 배선 완료(사이클 2).
- **대기 사유**: 사용자 지정 unity-mcp 경로 — CC 세션에 MCP 서버 미연결.
  대안: 에디터 메뉴 클릭 1회 or MCP 연결 후 요청.

## 2. 산출물 위치
- 벤치마크: `docs/benchmarks/{asr-model-matrix, fc-model-benchmark, gemma4-gguf-vs-litertlm, short-utterance-asr-research, device-cycle1-baseline}.md`
- 핸드오프: `docs/handoffs/v0.14-upgrade-handoff.md`
- 모델: `Assets/StreamingAssets/{LLM,ASR,Multimodal,TestAssets}/…` (카테고리+모델별 폴더)
- 도구: `Tools/Windows/Run-LiteRtLmWindowsAsrSmokeTest.ps1`, i4 리팩 파이프라인(스크래치패드+`External/ModelWork/README-i4-prototypes.md`)

## 3. 잔여 작업
1. **사이클 3**(진행 중): take5 AAR 빌드 → turbo i4 디바이스, featureMd5 A/B, 멀티모달 FC 러너 디바이스 검증
2. **씬 생성**: unity-mcp 연결(에디터 Window > MCP For Unity → CC 재시작) 또는 메뉴 실행
3. **#8 최종 문서화**: README/llm-details/asr-details 갱신 (벤치 문서들은 완료)
4. **#13 Linux/macOS**: 후순위 — 업스트림 macos_arm64 바이너리 존재, 커스텀 플래그는 OS별 빌드 필요
5. **#7 Bonsai**: 스킵 권장 — 사용자 결정 대기
