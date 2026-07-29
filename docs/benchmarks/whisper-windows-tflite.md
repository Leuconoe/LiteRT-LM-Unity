# Whisper tflite on Windows

**Question:** can the Whisper tflite exports — the ones the Android AAR runs —
also run on Windows?

**Answer: yes.** 20 of 20 transcriptions succeeded on 2026-07-27 across five
exports and four clips, on CPU through `ai_edge_litert`. This closes the open
question left in the [sample scene session handoff](../handoffs/sample-scene-session-handoff.md).

This document exists so the answer stays proven rather than re-derived: the
driver is now in the repository and the sweep is one command.

## Reproduce

```powershell
.\Tools\Research\Whisper\Run-WhisperTfliteWindows.ps1 -Sweep          # the table below
.\Tools\Research\Whisper\Run-WhisperTfliteWindows.ps1 -Sweep -All     # + stock medium / large-v3
.\Tools\Research\Whisper\Run-WhisperTfliteWindows.ps1 `
    -Model "Assets\StreamingAssets\ASR\whisper-turbo-acft-ko\acft_turbo_5s_drq.tflite" `
    -Audio "Assets\StreamingAssets\TestAssets\Audio\volume-볼륨 업.mp3"
```

Results land in `Builds/Logs/whisper-windows-tflite-sweep.jsonl`, one JSON
object per run.

The wrapper finds a Python with the four required packages; if none exists,
`-Bootstrap` creates `Tools/Research/Whisper/WhisperTflite/.venv` (gitignored) from
`requirements.txt`. The driver itself is
`Tools/Research/Whisper/WhisperTflite/whisper_tflite_runner.py`: 16 kHz mono → slaney
log-mel → `encode` → greedy decode with a forced
`[SOT, lang, transcribe, notimestamps]` prefix. `n_mels`, the mel-frame window
and the vocabulary size are read from the signatures, so 80-mel and 128-mel,
500-frame and 3000-frame exports all work unmodified.

## Results

AMD Ryzen 9 7950X, CPU only (XNNPACK delegate), `ai-edge-litert` 2.1.6,
Python 3.12.13. `enc` is the median of three encoder runs; `dec` is the whole
greedy loop.

| Export | mel | window | `소리 키워줘` (1.32 s) | `볼륨 업` (0.79 s) | `현재 서울의 날씨는 흐림 입니다` (3.24 s) | `2025년 3월 5일 전술평가 결과 보고` (3.98 s) |
| --- | ---: | ---: | --- | --- | --- | --- |
| acft_base_5s_drq | 80 | 500 | 소리 키워줘 | **볼륨어** | 현재 서울의 날씨는 흐림입니다. | 2025년 3월 5일 전술 평가 결과 보고 |
| acft_medium_5s_drq | 80 | 500 | 소리 키워줘 | 볼륨업 | 현재 서울의 날씨는 흐림입니다. | 2025년 3월 5일 전술평가 결과 보고 |
| acft_turbo_5s_drq | 128 | 500 | 소리 키워줘 | 볼륨업 | 현재 서울의 날씨는 흐림입니다. | 2025년 3월 5일 전술평가 결과 보고 |
| whisper_base_30s_i8 | 80 | 3000 | 소리 키워줘 | **볼륨어** | 현재 서울의 날씨는 흐림입니다. | 2025년 3월 5일 전술 평가 결과 보고 |
| whisper_tiny_30s_i8 | 80 | 3000 | 소리 키워줘 | **보일해봐** | 현재 서울의 날씨는 흐림입니다. | **2015년 3월호일** 전술 평가 결과 보고 |

Timings, seconds (encode / decode):

| Export | 1.32 s clip | 0.79 s clip | 3.24 s clip | 3.98 s clip |
| --- | --- | --- | --- | --- |
| acft_base_5s_drq | 0.027 / 0.195 | 0.027 / 0.229 | 0.027 / 0.456 | 0.027 / 0.503 |
| acft_medium_5s_drq | 0.411 / 1.681 | 0.364 / 1.983 | 0.360 / 4.571 | 0.361 / 4.600 |
| acft_turbo_5s_drq | 0.735 / 0.632 | 0.737 / 0.792 | 0.791 / 1.767 | 0.785 / 1.738 |
| whisper_base_30s_i8 | 0.358 / 0.385 | 0.355 / 0.477 | 0.349 / 0.959 | 0.346 / 0.986 |
| whisper_tiny_30s_i8 | 0.166 / 0.206 | 0.162 / 0.206 | 0.167 / 0.496 | 0.165 / 0.510 |

## What this changes

**128-mel / 51866-vocab exports work here too.** `acft_turbo_5s_drq` transcribes
correctly, matching the device. This is agreement, not a new finding: the AAR hit
`TensorBuffer 65536 vs 7680000` on turbo in cycle 1 (take3, 80-mel hardcode and
positional decode binding), take4 made mel/vocab dynamic and take5 bound the
decode inputs by shape — after which turbo passed 3/3 gate clips on device
([device ledger](device-cycle1-baseline.md) cycle 3). The Python driver resolves
the same three things from the signatures, so desktop and device now agree on the
same exports.

**The short-clip tier boundary reproduces on desktop.** The 0.79 s `볼륨 업`
clip comes out as `볼륨어` on both base-tier exports and `보일해봐` on tiny,
while medium and turbo get it right. That matches the on-device finding and the
[short-utterance research](short-utterance-asr-research.md): this is model
capacity, not preprocessing, and no VAD or gain setting fixes it at base tier.

**A 5 s window is 13× cheaper to encode than a 30 s one** at the same tier
(0.027 s vs 0.349 s for base). Decode dominates only above ~10 output tokens.

## Caveats

- **Not wired into Unity.** The desktop ASR path in the samples still routes
  through gemma-4 audio via `litert_lm_main`, because that CLI is an LLM runner
  and cannot drive an encoder/decoder pair. Shipping Whisper tflite on Windows
  from Unity means porting this driver out of the AAR into C#/C++ against
  `libLiteRt.dll` (already in `Tools/Windows/`) — a build task, not a flag.
  Python is a bench harness here, not a runtime.
- Decode timings are pessimistic: the driver re-runs the decoder over the whole
  token buffer each step instead of reusing a KV cache, so `decode_s` grows
  quadratically in output length. Compare exports against each other, not
  against the AAR.
- CPU only. No GPU delegate was attempted.
- Clips are the project's standard Korean test set; see the
  [ASR model matrix](asr-model-matrix.md) for CER/WER on the full 10-clip set,
  and note the standing caveat that the 40-clip gate set is TTS-synthesized and
  valid for ranking only.
