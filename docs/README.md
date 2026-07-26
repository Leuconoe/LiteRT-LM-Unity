# Documentation

프로젝트 README는 짧게 유지하고, 세부 내용은 여기에 둡니다.
**모든 문서의 기준은 Android 온디바이스 실행**이며, Windows 수치는 참고용입니다.

## Details

- [LLM 세부 설명](llm-details.md) — 티어·백엔드 선택·디바이스 벤치
- [ASR 세부 설명](asr-details.md) — 전체 라인업·VAD·ACFT-KO 학습 배경·스모크 커맨드

## Benchmarks (근거 데이터)

- [ASR 모델 매트릭스](benchmarks/asr-model-matrix.md) — 전 티어 × 10클립 CER/WER/RTF
- [FC 모델 벤치마크](benchmarks/fc-model-benchmark.md) — 20케이스 채점
- [실기기 PDCA 원장](benchmarks/device-cycle1-baseline.md) — 사이클 1–6 전체 기록
- [짧은 음성 인식 연구](benchmarks/short-utterance-asr-research.md)
- [gemma-4 GGUF 대비 비교](benchmarks/gemma4-gguf-vs-litertlm.md)
- [v0.14 세션 종합 보고서](benchmarks/session-final-report-20260723.md)

## Handoffs

작업 인계·추적용 기록입니다.

- [ASR 학습 프로그램](handoffs/asr-training-program-handoff.md) — 클린 ACFT 레시피,
  kspon 폐기 기록(산출물 삭제됨, 이 문서가 유일 기록), 재학습 시 규칙
- [v0.14 업그레이드](handoffs/v0.14-upgrade-handoff.md)
- [Android 디바이스 벤치마크](handoffs/android-device-benchmark-handoff.md)
- [펑션콜링](handoffs/function-calling-handoff.md)
