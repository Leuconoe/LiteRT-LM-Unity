# SenseVoice-Small for Korean commands, on LiteRT

Research run 2026-10-16. Question: is FunAudioLLM SenseVoice-Small a better
on-device ASR than the Whisper ACFT tiers for short Korean commands, and can it run
on LiteRT?

**Answer: yes on accuracy, yes on LiteRT (desktop proven, device port not done).**
Published (public):

- [litert-community/SenseVoiceSmall](https://huggingface.co/litert-community/SenseVoiceSmall) — stock weights
- [litert-community/SenseVoiceSmall-ko-zeroth](https://huggingface.co/litert-community/SenseVoiceSmall-ko-zeroth) — Zeroth-Korean fine-tune (a trade, see §3)

Only SenseVoice-**Small** (~234 M params) is released; there is no smaller tier
and SenseVoice-Large weights are not public.

## 1. Accuracy against what we ship

CER with spaces and punctuation removed, `woitn`, language forced to `ko`, dither 0.

| Model | gate40 (TTS, ranking only) | real quiet (13) | real noise (9) | FLEURS-ko (64) |
| --- | ---: | ---: | ---: | ---: |
| **SenseVoice-Small stock, tflite w8** | 0.204 · 20 exact | **0.006 · 12** | **0.269 · 5** | 0.068 |
| SenseVoice-Small stock, PyTorch | 0.169 · 21 | 0.009 · 11 | 0.296 · 4 | 0.071 |
| roboseasylabs SenseVoice-Small-ko (EDIE fine-tune) | 0.243 · 16 | 0.368 · 2 | 0.407 · 2 | 0.536 |

Against the Whisper ACFT tiers on the gate's own protocol (first 24 clips,
punctuation kept): turbo-acft 0.182 · **SenseVoice 0.198 (PyTorch) / 0.247 (tflite)** ·
medium-acft 0.208 · base-acft 0.305 · tiny-acft 0.457 — between turbo and base at a
quarter of turbo's size. It also reads the quiet 0.79 s `볼륨 업` clip that no
base-tier fix could recover. Weak spots: field noise (5/9 exact) and soft language
forcing (a few one-syllable clips come out in kana even with `ko`).

## 2. LiteRT conversion

ONNX (funasr export) → onnx2tf → `.tflite`, one **fixed-shape** graph per window
(3 / 5 / 10 / 30 s → LFR 50 / 84 / 167 / 500), because a dynamic graph keeps
XNNPACK from attaching (the Supertonic lesson; here 10× fp32 / 25× i8).

- Parity: fp32 tflite vs onnxruntime max |Δlogit| ≤ 3.6e-4 on commands, CTC argmax
  100 %, transcripts identical on 583/583 clips.
- `w8` (`weight_only_wi8_afp32`) keeps fp32 accuracy at 231 MB per window;
  `i8` (`dynamic_wi8_afp32`) is ~25 % faster but worse in noise (0.269 → 0.333).
- Desktop CPU p50, model only: 5 s window w8 172 ms, i8 121 ms.

Traps:

- **onnx2tf silently made `speech` feature-major `[1, 560, N]`.** Bind inputs by
  name (signature inputs are alphabetical).
- **Pad features, not audio.** Zero-padded features + the true `speech_lengths` is
  bit-identical to the unpadded graph; zero-padded *audio* without the length takes
  real-command CER from 0.009 to 0.44.
- funasr's default `dither=1.0` makes every evaluation non-deterministic — use 0.
- onnx2tf `-cotof` hangs on Windows (subprocess never exits).

Device port needs, in the AAR: a JNI entry that runs the fixed-shape graph on
CPU/XNNPACK, the Kaldi fbank 80 + LFR (m 7, n 6) + CMVN front end in C++/C#
(dump-compare against desktop features), and greedy CTC through `tokens.txt` (the
AAR has no SentencePiece). Only the 5 s window needs to be resident for commands.

## 3. Zeroth-Korean fine-tune

No public SenseVoice trained on Zeroth exists, so one was trained (FunASR
`train_ds.py`, Zeroth train 95 % + FLEURS-en ~11 % as a forgetting guard).

| Run | real quiet | real noise | real exact (22) | pooled real CER | FLEURS-ko |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock | **0.006 · 12** | 0.269 · 5 | **17** | 0.114 | 0.068 |
| **lr 1e-5, 1 epoch (published)** | 0.011 · 11 | **0.222 · 5** | 16 | **0.097** | **0.066** |
| lr 5e-5, 1–4 epochs | 0.011–0.020 | 0.333–0.444 | 14–15 | 0.158 (ep1) | 0.070–0.077 |

At 5e-5 the model drifts toward read speech: noisy commands and FLEURS-ko get
worse every epoch. At 1e-5 one epoch helps noise and sentences and costs one
quiet-room clip — a trade, stated plainly on the model card. **Zeroth test is not a
held-out measure:** 456 of its 457 sentences also appear in train (speakers differ),
so it was never used for selection.

FunASR notes: FunASR's acc-based "best" checkpoint is always 0 for this CTC model —
select with `sv_eval.py`; half-epoch saves need `validate_interval` equal to the save
interval or the run dies with `KeyError: 'model.pt.ep0.1345'`; torch/torchaudio 2.8
(2.9+ needs torchcodec inside FunASR's loader).

## 4. Whisper hotword A/B (same session)

The WebServer feeds `assets/WhisperHotwords.txt` to faster-whisper as a decoder
prompt. Same prefix (`<|startofprev|>` + hotwords) on our tflite tiers, desktop CPU:

| Set | tiny off | tiny hotwords | base off | base hotwords |
| --- | ---: | ---: | ---: | ---: |
| real recordings (4) | 1/4 | 1/4 | 3/4 | 3/4 |
| BenchmarkAudio (13) | 3 | 3 | 8 | 7 |
| TTS commands with hotword terms (54) | 4 | 10 | 20 | **31** |
| out-of-list real clips (22) | 3 | 3 | 10 | 8 |
| clips where a hotword was invented | — | **4** | — | 0 |

- **tiny-acft-ko stays rejected** — no gain on real audio, and it starts reading the
  callsign list aloud (`볼륨 업` → `퀘벡`).
- base-acft-ko gains on in-list words (`재억원` → `제어권`) at no latency cost (the
  decoder is a fixed 128 tokens) but only 91 of the 151 hotword tokens fit.
- End to end through FunctionGemma the gain shrinks to +10 commands of 504
  ([functiongemma-korean-fc.md](functiongemma-korean-fc.md) §7), while SenseVoice
  beats both. Hotwords stay a research flag (`--hotwords-file`), not shipped.

## 5. Reproduce

```powershell
.\Tools\Research\SenseVoice\Run-SenseVoiceEval.ps1 -Bootstrap
.\Tools\Research\SenseVoice\Run-SenseVoiceEval.ps1 --backend tflite --model <graph-dir> --sets gate40,real_clean,real_noise,fleurs_ko
.\Tools\Research\SenseVoice\Convert-SenseVoiceToLiteRt.ps1 -Bootstrap
.\Tools\Research\SenseVoice\Convert-SenseVoiceToLiteRt.ps1 -Model <funasr-dir> -Name <name>     # -Quant w8,i8 by default
.\Tools\Research\SenseVoice\Train-SenseVoiceZeroth.ps1 -Bootstrap
.\Tools\Research\SenseVoice\Train-SenseVoiceZeroth.ps1 -Prepare
.\Tools\Research\SenseVoice\Train-SenseVoiceZeroth.ps1 -OutDir External\sensevoice-work\zeroth-ft-lr1e5 -MaxEpoch 2 -Lr 1e-5 -WarmupSteps 500 -SaveSteps 1345
.\Tools\Research\Whisper\Run-WhisperTfliteWindows.ps1 -HotwordsFile <WhisperHotwords.txt>   # A/B driver: WhisperTflite/hotword_ab.py
```

## 6. On the device (AAR take20, kona / SM8250, 2026-10-18)

`UnityLiteRtLmBridge.runSenseVoiceAsr` (C# `LiteRtLmSenseVoice`) runs the fbank/LFR/CMVN front end, picks the
smallest bucket present that fits, runs the w8 graph on LiteRT CPU (XNNPACK, 4 threads) and decodes greedy CTC
with a confidence. Parity with the desktop tflite path on the same wavs: **tokens identical 62/62**, feature
max |diff| 9.6e-5, logit max |diff| 1.2e-4 (10 dumped clips), confidence diff ≤ 1.6e-4.

| set | n | CER | exact | infer ms mean / p50 | feat ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| gate40 (TTS, ranking only) | 40 | 0.2042 | 20 | 402 / 385 | 3 |
| real_clean | 13 | 0.0064 | 12 | 465 / 478 | 5–9 |
| real_noise | 9 | 0.2685 | 5 | 499 / 449 | 11 |

Per bucket: 3 s 396 ms p50, 5 s 766 ms p50, 10 s 1.35 s (one clip); first use of a bucket compiles for ~2.9 s.
App PSS 2.46 GB with one bucket resident; hottest thermal zone 39.7 → 49.8 °C over the 62 clips.

- **One bucket resident.** A w8 bucket costs ~2 GB RSS on XNNPACK; caching three got the app killed by the
  low-memory killer at 5.6 GB, so the native side frees the previous graph before compiling the next. Each
  bucket file is a full 242 MB copy of the weights — ship one (the FC demo APK ships only 5 s; longer audio is
  truncated to the largest bucket present).
- The CPU default is one thread (930 ms for the 3 s bucket); the JNI sets 4, like the other entries.
- Test clips are float WAV: the noise recordings decode past ±1 and PCM16 export clipped them (CER 0.27 → 0.32).

### Confidence as a confirmation gate

`meanMaxProb` (mean best softmax probability over speech frames) and `minTokenProb` were joined with the v6e
router's outcomes on the heldout5 voice sets (svw8 arm), on top of the high-impact `ConfirmationPolicy`.
Aggregate counts only.

| set · MinAsrConfidence (meanMaxProb) | wrong actions held | correct commands asked to confirm |
| --- | ---: | ---: |
| TTS 7 voices (2,024) · policy only | 19/216 (8.8 %) | 109/1070 (10.2 %) |
| · 0.90 | 26/216 (12.0 %) | 123/1070 (11.5 %) |
| · 0.93 | 50/216 (23.1 %) | 182/1070 (17.0 %) |
| · 0.95 | 99/216 (45.8 %) | 323/1070 (30.2 %) |
| OSS 4,499 · policy only | 41/607 (6.8 %) | 156/1686 (9.3 %) |
| · 0.90 | 87/607 (14.3 %) | 198/1686 (11.7 %) |
| · 0.93 | 162/607 (26.7 %) | 303/1686 (18.0 %) |
| · 0.95 | 298/607 (49.1 %) | 568/1686 (33.7 %) |

The gate is weak: AUROC wrong-vs-correct 0.57 (`meanMaxProb`), 0.55 (`minTokenProb`, worse at every
threshold — BPE pieces split probability even on correct text, median 0.57). It does flag transcription
errors (AUROC 0.73), but 18–31 % of wrong actions come from exact transcripts. `LiteRtLmSenseVoice.RecommendedMinConfidence`
is 0.90: +3–8 points of wrong actions held for +1–2 points of extra confirmations; 0.93 is the stricter option.
Driver: `Tools/Research/SenseVoice/SenseVoiceBench/sv_device.py` (`clips` / `score` / `conf`) and
`Tools/Windows/Tests/Run-LiteRtLmAndroidSenseVoiceSmokeTest.ps1`.
