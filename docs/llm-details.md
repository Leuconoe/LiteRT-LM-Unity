# LLM Details

Unity LiteRT-LM 브리지의 LLM 설정·벤치마크 기록입니다.
**판단 기준은 Android 온디바이스 실행**(Snapdragon 865 / kona / 7.5 GiB RAM,
기기 `46a880a0`)이며, Windows 수치는 개발 보조용 참고입니다.
README에는 요약만 두고 근거는 여기에 둡니다.

## 1. 기기 RAM 예산별 권장 조합

`모바일` 판정: **상시** = 항상 상주 가능 · **온디맨드** = 필요할 때 로드했다
해제 · **부적합** = 디바이스에서 쓸 이유 없음.

| Model | Size | 유휴 PSS | 피크 PSS | 모바일 | 역할 |
| --- | ---: | ---: | ---: | --- | --- |
| `LLM/qwen2.5-0.5b/…_wi4b64_ekv1280.litertlm` (자체 i4) | 265 MB | ~0.38 GB | — | **상시** | 채팅 최속 35.5 tok/s. FC 부적합 |
| `LLM/qwen3-0.6b/qwen3_0_6b_mixed_int4.litertlm` | 475 MB | ~0.36 GB | — | **상시** | 소형 FC 제1픽 18/20 + 채팅 20.9 tok/s |
| `LLM/gemma3-1b/gemma3-1b-it-int4.litertlm` | 557 MB | 0.40 GB | — | **상시** | 채팅 폴백 16.0 tok/s. FC 라우터로는 3/20 — 금지 |
| `LLM/lfm2.5-1.2b/LFM2.5-1.2B-Instruct_int4.litertlm` | 702 MB | — | — | 상시(6 GB+) | 중형 FC 17/20, 16.8 tok/s. v0.14 런타임 필수 |
| `Multimodal/gemma-4-e2b/gemma-4-E2B-it.litertlm` | 2.6 GB | 0.48 GB | **3.6 GB** | **온디맨드(8 GB+)** | 채팅·이미지·오디오·FC를 한 모델로. 아래 주의 참조 |
| `LLM/qwen2.5-1.5b/…_q8_ekv4096.litertlm` | 1.5 GB | 0.35 GB | — | **부적합** | 8.5 tok/s CPU, 턴당 3.6–3.8 s — 크기 대비 이득 없음 |
| `LLM/gemma3-270m/gemma3-270m-it-q8.litertlm` | 290 MB | 0.35 GB | — | **부적합** | 27.7 tok/s로 빠르나 출력 품질이 실사용 미달. i4화 불가(f32 소스 없음) |

권장 조합 (README와 동일). ASR은 **다루는 발화 길이에 맞는 것 하나만** 얹으면
됩니다 — 선택 기준은 [`asr-details.md`](asr-details.md) 참조.

| 기기 RAM | LLM | + ASR 1종 | 상주 합계 |
| --- | --- | --- | ---: |
| 4~6 GB | Qwen2.5-0.5B i4 (265 MB) | base-acft-ko 5s (101 MB) | ~370 MB |
| 6~8 GB | Qwen3-0.6B i4 (475 MB) | 용도에 맞는 1종 (77~101 MB) | ~570 MB |
| 8 GB+ | gemma-4-E2B QAT (2.6 GB) | 불필요 — 오디오 입력 내장(4.1 s, 내용 정확) | 유휴 0.48 GB / 이미지 턴 3.6 GB |

⚠️ **gemma-4-E2B의 실제 제약은 파일 크기(2.6 GB)가 아니라 이미지 턴의 메모리
스파이크**입니다. 텍스트·FC 턴은 PSS 0.48 GB로 가볍지만, 700×467 이미지
1장(2340 vision patch) 처리 중 **3.6 GB까지 상승**합니다(측정 시
MemAvailable 3.4 GB 유지, lowmemorykiller·크래시 0). 6 GB 기기에서 다른 모델과
동시 상주시키면 위험하므로 **멀티모달이 필요한 구간에만 로드**하세요.

⚠️ **FC(펑션콜링)에 채팅 모델을 쓰지 마세요.** gemma3-1b 3/20, qwen2.5 계열
2–8/20으로 라우터 역할을 못 합니다. FC는 Qwen3-0.6B(소형) / LFM2.5-1.2B(중형) /
gemma-4-E2B(플래그십) 중에서 고르세요.

## 2. 디바이스 실측 (v0.14, 2026-07-23)

스모크(채팅 2턴) + 벤치마크(3런 × prefill 64 / decode 32 토큰), 웜 상태
52–57 °C, take3 AAR. 전체 로그:
[`benchmarks/device-cycle1-baseline.md`](benchmarks/device-cycle1-baseline.md).

**적중률**은 20케이스 한국어 FC 라우팅 채점(§4)이며, 채팅 품질은 별도로
한/영 응답 일관성 PASS/FAIL로 표기합니다.

| Model | Backend | Init s | Prefill tok/s | Decode tok/s | FC 적중률 | 채팅 | Note |
| --- | --- | ---: | ---: | ---: | ---: | :-: | --- |
| `qwen2.5-0.5b` 자체 i4 (264 MB) | CPU | **1.37** | **218.3** | **35.5** | 2–8/20 ✗ | PASS | 최속. 공식 q8 대비 절반 크기 +38 % 디코드 |
| `qwen3-0.6b` mixed_int4 (475 MB) | CPU | 1.66 | 31.8 | 20.9 | **18/20** | PASS | `/think` 기본 on → 턴이 길어짐. prefill이 크기 대비 낮음 |
| `lfm2.5-1.2b` int4 (702 MB) | CPU | 7.73 | 57.1 | 16.8 | 17/20 | PASS | v0.14에서만 로드됨 |
| `gemma3-1b` int4 (557 MB) | CPU | 4.14 | 100.9 | 16.0 | 3/20 ✗ | PASS | 웜 상태라 쿨 기록 대비 ~6 % 낮음 |
| `gemma3-1b` int4 | GPU | 9.73 | **184.2** | 13.7 | 3/20 ✗ | PASS | OpenCL 델리게이트 + GPU TopK 샘플러 |
| `gemma-4-E2B` QAT (2.6 GB) | CPU/GPU | 1.2~18.5 | 141 / 431 | 5.2 / 7.0 | **19/20** | PASS | 멀티모달 전용 운용. 디코드는 느리지만 한 모델로 전부 커버 |

전 구성 PASS(한/영 정상 출력, 사이클 전체 크래시·OOM 0).
**속도와 FC 적중률은 상관이 없습니다** — 최속 모델(35.5 tok/s)이 라우터로는
최하위이고, 소형 FC 픽(20.9 tok/s)이 플래그십에 1점 차입니다.

### 멀티모달 턴 (gemma-4-E2B, 사이클 2–3)

| 입력 | 백엔드 (llm/vision/audio) | 지연 | 결과 |
| --- | --- | ---: | --- |
| 이미지 700×467 | CPU/CPU/– | 23.4 s | 정확 (PSS 3.6 GB) |
| 이미지 700×467 | GPU/GPU/– | **7.6 s** | 동일 내용 — GPU 3.1× |
| 오디오 3.8 s 한국어 | CPU/–/CPU | 4.1 s | 내용 정확(연도 포함) |
| 이미지 + 발화 → 도구 호출 | CPU | 40.7 s E2E | 순수 tool JSON, 제약 디코딩 없이 |

`maxNumTokens 4000` 필수(이미지 2340패치 + 도구 프롬프트 ~1.3k 토큰).
오디오 전사는 4.1 s로 전용 ASR보다 빨랐고 띄어쓰기까지 정확 — **LLM이 이미
상주 중이라면 별도 ASR 없이 전사+FC를 한 턴에** 처리할 수 있습니다.

## 3. 백엔드 선택 (CPU vs GPU)

- **Android (Adreno 650 / kona)**: GPU 디코드가 CPU보다 **느립니다**
  (13.7 vs 16.0 tok/s). 샘플러 폴백도 발열도 아니고 구조적 특성 —
  단일 토큰 디코드는 대역폭 병목이라 스텝당 OpenCL 디스패치·동기화
  오버헤드가 연산 이득을 넘습니다. 반대로 GPU는 prefill ~1.8×
  (184 vs 101 tok/s), 멀티모달 이미지 턴 ~3.1× 빠릅니다.
  → **채팅은 CPU, 긴 프롬프트 prefill·멀티모달은 GPU**
- **Windows (RTX 4090, WebGPU/Dawn over D3D12, 참고)**: v0.14에서 GPU 백엔드가
  고쳐져 디코드 49.3 tok/s vs CPU 17.8–26.8(~2×). Windows 기본은 GPU이고
  `LiteRtLmWindowsCliClient`가 실패 시 세션당 1회 CPU로 폴백합니다.
  **Android와 반대 특성이니 데스크톱 결과로 기기 백엔드를 결정하지 마세요.**

## 4. 펑션콜링 티어

20케이스 한국어 라우팅 채점은 Windows에서 측정하고, 상위 티어는 디바이스
E2E로 재확인했습니다. 케이스별 상세:
[`benchmarks/fc-model-benchmark.md`](benchmarks/fc-model-benchmark.md).

| 티어 | 모델 | 점수 | 디바이스 확인 |
| --- | --- | --- | --- |
| 플래그십 (~2.5 GB) | gemma-4-E2B (공식 QAT wNa8o8) | **19/20** | 이미지+발화 → 도구 호출 40.7 s PASS |
| 소형 (475 MB) | qwen3_0_6b_mixed_int4 + Hermes 프롬프트 | 18/20 | 채팅 경로 PASS |
| 중형 (702 MB) | LFM2.5-1.2B int4 + Hermes 프롬프트 | 17/20 | 로드·채팅 PASS (최속 FC 23.6 tok/s) |
| 라우터 불가 | gemma3-1b (3/20), qwen2.5 계열 (2–8/20) | — | 단 gemma3-1b은 단일 도구 음성 FC 파이프라인의 LLM으로는 디바이스 PASS (15.5 s E2E) |

## 5. i4 자체 양자화 파이프라인

int4 최소 티어 정책에 따라 `.litertlm` 번들을 unpack → quantize
(`ai_edge_quantizer`) → repack(`litert-lm-builder`)하는 경로를 구축했습니다.

- 레시피: `dynamic_wi4b64_afp32` (4-bit 가중치, 64값 블록당 fp16 스케일,
  fp32 활성) + 민감 스코프(임베딩/로짓, 인코더)는 i8 유지
- 채널와이즈 `wi4c`는 **사용 금지**(품질 붕괴)
- 산출물: qwen2.5-0.5b i4 (265 MB, 디바이스 검증 +38 % 디코드),
  qwen2.5-1.5b i4 (790 MB)
- gemma3-270m / FunctionGemma는 i4 불가(f32 소스 없음, i8→i4는 무의미)
- 상세: [`External/ModelWork/README-i4-prototypes.md`](../External/ModelWork/README-i4-prototypes.md)

**모바일 관점 요약**: int4는 크기와 속도를 동시에 얻습니다. 디코드가
메모리 대역폭 병목이라 가중치가 작아지면 그대로 속도가 됩니다(q8 → i4에서
크기 절반 + 디코드 +38 %). 반대로 int2/Q5/1.58-bit는 LiteRT 커널이 없어 불가.

## 6. 검토 후 탈락 (2026-07-23)

- **Qwen3.5-0.8B-MTP** — 불가. MTP는 llama.cpp 전용이고 litert-torch가 아키텍처
  미지원, 커뮤니티 litertlm 포팅본은 v0.14에서 출력이 붕괴
- **Bonsai-1.7B** — 권장 안 함. 1-bit 크기 이점이 LiteRT 변환 시 int4/8
  재양자화로 소멸
- **VibeVoice-ASR** — 8.7B, 온디바이스 불가
- **GGUF / llama.cpp** — Windows CUDA 디코드 ~243 tok/s(litertlm CPU의 ~16×)
  이지만 **Android 경로가 없음**. llama.cpp는 데스크톱 실험용, litertlm이 제품
  런타임입니다.
  [`benchmarks/gemma4-gguf-vs-litertlm.md`](benchmarks/gemma4-gguf-vs-litertlm.md)

## 7. 스모크 테스트

벤치마크 러너는 단일 APK를 재사용합니다. 모델과 런타임 JSON 설정을 앱
스토리지에 푸시한 뒤 같은 빌드를 실행하세요.

```powershell
.\Tools\Windows\Run-LiteRtLmAndroidDeviceBenchmarks.ps1 `
  -DeviceSerial <device-serial> `
  -BenchmarkName gemma-4-e2b-it-gpu,gemma-4-e2b-it-cpu `
  -SingleApkPath Builds\Android\LiteRtLmAndroidSmokeTest-gemma3-270m-it-q8-CPU.apk `
  -TimeoutSeconds 900
```

디바이스 파일 크기가 로컬과 일치하면 이후 런에서는 모델 전송을 건너뜁니다.

---

## 부록 — 2026-05-16 쿨 디바이스 기록 (v0.14 이전, 대체됨)

현재 수치는 §2를 보세요. 이 표는 **쿨 상태 레퍼런스**로만 남깁니다
(§2는 웜 52–57 °C 측정이라 ~6 % 낮습니다). 여기 등장하는 q8 모델들은 이후
자체 i4로 대체되었으므로 **모델 선택 근거로 쓰지 마세요.**

| Model | Backend | Status | File MB | PSS MB | Init s | Prefill tok/s | Decode tok/s |
| --- | --- | :-: | ---: | ---: | ---: | ---: | ---: |
| gemma-4-E2B-it | GPU | PASS | 2468.3 | 470.2 | 10.06 | 431.1 | 7.0 |
| gemma-4-E2B-it | CPU | PASS | 2468.3 | 386.4 | 4.78 | 141.0 | 5.2 |
| gemma3-270m-it-q8 | GPU | PASS | 289.9 | 425.2 | 6.49 | 377.6 | 26.0 |
| gemma3-270m-it-q8 | CPU | PASS | 289.9 | 346.3 | 0.87 | 101.3 | 27.7 |
| gemma3-1b-it-int4 | GPU | PASS | 557.3 | 471.4 | 7.91 | 197.0 | 16.4 |
| gemma3-1b-it-int4 | CPU | PASS | 557.3 | 404.1 | 3.44 | 108.1 | 17.5 |
| Qwen2.5-0.5B-Instruct-q8 | GPU | **FAIL** | 520.7 | 458.7 | — | — | — |
| Qwen2.5-0.5B-Instruct-q8 | CPU | PASS | 520.7 | 384.8 | 1.31 | 206.8 | 25.7 |
| Qwen2.5-1.5B q8 ekv4096 | GPU | PASS | 1523.9 | 365.2 | 9.41 | 88.9 | 9.9 |
| Qwen2.5-1.5B q8 ekv4096 | CPU | PASS | 1523.9 | 347.3 | 4.29 | 30.7 | 8.5 |
| Qwen3-0.6B (stock) | GPU | PASS | 585.8 | 395.5 | 5.72 | 96.5 | 9.1 |
| Qwen3-0.6B (stock) | CPU | PASS | 585.8 | 359.4 | 1.74 | 30.1 | 6.1 |

기록해 둘 만한 점:

- **Qwen2.5-0.5B q8은 GPU에서 엔진 생성 실패**
  (`llm_litert_compiled_model_executor.cc:1546`, 1272 op GPU / 54 op CPU 부분
  델리게이션 후 실패). HF 모델 카드도 Android CPU 결과만 공개합니다 —
  이 파일은 CPU 전용으로 취급하세요. 현재는 자체 i4로 대체됨
- 2026-05 AAR은 `GPU sampler unavailable. Falling back to CPU sampling.`을
  출력했습니다. v0.14 AAR은 네이티브 OpenCL TopK 샘플러
  (`LiteRtTopKOpenClSampler`)를 로드하므로 이 폴백은 더 이상 발생하지 않으며,
  그럼에도 Adreno 650에서 GPU 디코드가 느린 것은 §3의 구조적 이유입니다
- stock Qwen3-0.6B는 디코드 6–9 tok/s였으나, mixed_int4 변환 후 20.9 tok/s로
  올라 소형 FC 픽이 되었습니다
