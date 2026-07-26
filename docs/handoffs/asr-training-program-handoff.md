# ASR 학습·배포 프로그램 핸드오프 (2026-07-23 ~ 07-26)

이 문서 하나로 ASR 관련 작업 전체를 이어받을 수 있게 정리했습니다.
프레임워크(v0.14) 업그레이드 자체는 `v0.14-upgrade-handoff.md` 참조.

## 1. 최종 상태 한 줄 요약

**배포·공개된 것은 "클린 ACFT" 계보뿐**입니다 — stock openai whisper에
zeroth-korean(70%) + FLEURS en(30%)로 ACFT 자기증류한 4모델. KsponSpeech
계보는 **사용자 지시로 폐기**(2026-07-26), 산출물은 기록용으로만 보존.

## 2. 배포 중인 ASR 라인업 (StreamingAssets)

| 용도 | 파일 | 크기 | 근거 |
| --- | --- | ---: | --- |
| 음성 명령 제1픽 | `ASR/whisper-base-acft-ko/acft_base_5s_drq.tflite` | 101 MB | 디바이스 정상음량 명령 전부 exact, E2E 0.7–0.8 s |
| 명령 정확도 폴백 | `ASR/whisper-turbo-acft-ko/acft_turbo_5s_drq.tflite` | 883 MB | 디바이스 5/5 (조용한 구녹음 포함) — 유일 |
| 장문(>30 s) | `ASR/qwen3-asr-0.6b/qwen3_asr_0.6b_5s_i8.tflite` | 794 MB | 5 s 청크 루프, 98 s 디바이스 완주(4.2분, RAM 평탄) |
| 문장 균형 | `ASR/whisper-base/whisper_base_30s_i8.tflite` | 77 MB | 한국어 문장 CER 0.000 |
| 기타 | tiny/medium/large-v3/turbo 30 s 티어(i8/i4) | — | 비교·레퍼런스 |
| (참고) medium-acft | `ASR/whisper-medium-acft-ko/` | 826 MB | 배치돼 있으나 **비권장**(turbo보다 느리고 부정확) |

VAD: 모든 ASR 경로가 `vadMode` off/energy(기본)/ai(Silero 1.25 MB) 지원.

## 3. 클린 ACFT 학습 레시피 (재현용 — 이것이 성공한 방법)

- 시작점: **stock openai/whisper-{tiny,base,medium,large-v3-turbo}** (komixv2
  등 한국어 파인튜닝본에서 시작하지 말 것 — 영어가 파괴되어 있음)
- 방법: futo-org/whisper-acft 자기증류(MSE on decoder hidden states, 동결
  풀윈도우 교사), Adam lr 1e-6, batch 1, 최대 8 epoch + 조기종료
- **핵심 보정 2가지**: ① `n_ctx` 하한 250(=5 s 고정 배포창과 일치 — 이전
  komixv2-acft-ggml 실패의 근본 원인이 ctx 64까지 내려간 분포 밖 학습)
  ② 한70:영30 혼합(zeroth 51.6 h + FLEURS en_us) — 단일 언어 학습은 반대
  언어를 파괴
- 짧은 발화: zeroth <3 s ×3 오버샘플 + 0.5–3 s 크롭 증강(p=0.15)
- 게이트 결과(5 s ctx 한국어 단문 CER): turbo 0.182 · medium 0.208 ·
  base 0.305 · tiny 0.457 (stock은 1.07–24.9로 붕괴)
- 스크립트: `External/acft-training/train_acft.py`, `run_queue.py`
  (**미공개 유지** — futo 노트북 이식본이라 공개 시 MIT 고지 의무 발생)

## 4. KsponSpeech 프로그램 — 폐기 (음성 결과 기록)

- 구성: komixv2(한국어 FT) 체크포인트 + KsponSpeech 100 h(자유대화, 1–3 s
  66.5%) CE 파인튜닝(C) → ACFT(D) → TTS 연속학습(E)
- 실행분: tiny·base 체인 완주, turbo는 C에서 손상 확인 후 D를 원본
  베이스로 재시작하던 중 폐기. small/medium 미시작
- **결과: C 단계가 모든 크기에서 단문 CER 악화** (tiny 0.417→0.609,
  base 0.331→0.473, turbo 0.200→0.575). turbo는 학습 도메인인 kspon조차
  악화(0.134→0.141) = 파국적 망각. 유일한 이득은 komixv2 tiny/base의
  영어 회복(0.99→0.39, 1.00→0.25)
- 미해결 의심(감사 진행 중, `runs/METHODOLOGY-AUDIT.md`): 단문 평가셋이
  **TTS 합성음**이라 자연발화 학습 모델이 불리했을 가능성 / 리플레이 없는
  전면 FT / composite 기준 체크포인트 선택 편향
- 보존물: `External/acft-training/runs/kspon-*`, `export/kspon-*`,
  `runs/C-STAGE-FINDING.md`, `runs/KSPON-PROGRAM-CLOSURE.md`,
  `datasets/kspon-acft-mix`(100 h HF 데이터셋), prep 스크립트 4종.
  **배포·공개된 kspon 계보 모델은 없음**

## 5. 공개된 모델 (HuggingFace)

| 레포 | 내용 |
| --- | --- |
| [litert-community/whisper-acft](https://huggingface.co/litert-community/whisper-acft) | futo 원본 ACFT 6종(tiny/base/small ±.en) × 5s/10s/30s drq, 통합 카드 |
| [leuconoe/whisper-acft-ko](https://huggingface.co/leuconoe/whisper-acft-ko) | 한국어 클린 ACFT 4모델 × 3윈도우(12파일) |
| [leuconoe/litert-lm-unity-quantized](https://huggingface.co/leuconoe/litert-lm-unity-quantized) | 프로젝트 자체 양자화 whisper/Qwen2.5 모음 |
| litert-community/whisper-{tiny,base,medium,large-v3,large-v3-turbo} | 자체 양자화 i8/i4 기여분(PR·직접 커밋) |

## 6. 재개 시 유용한 사실 / 함정

- JNI는 시그니처에서 **mel 빈·vocab·윈도우 프레임을 자동 감지**(take5/6):
  80/128-mel, 51865/51866, 100–3000 프레임 모두 한 코드로 처리. 결과 JSON에
  `melBins/vocabSize/windowFrames/featureMd5/vadMode` 보고
- whisper decode 입력은 **형상 기반 바인딩**(모델별 순서 상이) — 위치 기반으로
  되돌리지 말 것
- AAR 빌드: `-SkipImageBuild`는 Docker 이미지에 구운 **낡은 소스**를 빌드함.
  패치 변경 후에는 이미지 재빌드 필수 (take8에서 발견)
- 양자화: i8=`dynamic_wi8_afp32`, i4=`dynamic_wi4b64_afp32`(+민감 스코프 i8
  혼합). wi4c/wi4b32는 품질 붕괴. int2/Q5/1.58b는 LiteRT 커널 부재
- 저음량 0.79 s 클립은 VAD·게인으로 해결 불가(16조합 실증) = **모델 용량
  문제**. 해법은 티어 에스컬레이션(turbo-acft)
- 디바이스 46a880a0은 **터치스크린 없음 + FLAG_SECURE** — adb 탭/스크린샷
  불가. 검증은 `LiteRtLmAsrTest.autotest.json` 훅으로 수행(스피커 에코
  주입으로 연속 ASR 사이클 유도 가능)
- 학습 재개 시 큐 스크립트는 완료 마커를 보고 자동 스킵/채택(adoption)함 —
  중단 후 재실행이 안전

## 7. 남은 작업

1. **#13 Linux/macOS 바이너리** — 미착수. 업스트림 macos_arm64 릴리스에는
   커스텀 FC 플래그가 없으므로 패치 적용 후 OS별 빌드 필요
2. 감사 문서 `runs/METHODOLOGY-AUDIT.md` 완료 대기(무학습 분석만 진행 중):
   TTS 평가셋 유효성, 체크포인트 선택 기준, 향후 한국어 명령 파인튜닝 레시피
3. 선택 과제(문서에 후보로만 기록): VAD 세그먼트 기반 whisper 30 s 슬라이딩
   윈도우 청킹(장문 8–10배 가속 기대), 저음량 캡처용 AGC
