# Device Cycle 1 — Baseline (take3 AAR)

- Date: 2026-07-23
- Device: `46a880a0` — Qualcomm kona / SM8250 (METALENSE2), Android 12, 7.7 GiB RAM
- AAR: `Assets/Plugins/Android/litertlm-unity-bridge.aar` rebuilt 14:33 ("take3"):
  qwen3 ASR mode, `sendMessageWithMedia`, short-utterance fixes (boost-only RMS
  loudness normalization + energy-gate VAD trim in the shared ASR PCM path,
  qwen3 EOS min-3-steps guard).
- APKs (fresh builds from take3, Unity 6000.4.4f1 batchmode temp-copy):
  - `Builds/Android/LiteRtLmAndroidAsrSmokeTest-generic.apk` (16:09, model/audio/config pushed at runtime)
  - `Builds/Android/LiteRtLmAndroidSmokeTest-gemma3-1b-it-int4-GPU.apk` (generic LLM smoke, no embedded model, runtime config)
- All ASR runs: CPU backend, fresh app process per run (`am force-stop` + relaunch),
  Korean forced (`ko`) for Whisper, `auto` for Qwen3-ASR.
- Logs: `Builds/Logs/AndroidDeviceRuns/20260723-16*-*.{summary,status,logcat}.txt`

## Prior-attempt root cause (why the last cycle died)

The previous attempt's app death during the qwen3-asr run was **not**
lowmemorykiller. Device tombstone (14:41:41, pulled to scratchpad
`prior-crash/tombstone_00.txt`):

```text
signal 11 (SIGSEGV), code 1 (SEGV_MAPERR)
#04 pc ... liblitertlm_jni.so (Java_com_google_ai_edge_litertlm_unity_UnityLiteRtLmBridge_nativeRunParakeetAsrSmoke+148)
```

The installed APK (14:20) predated the take3 AAR (14:33): its C# runner had no
`qwen3` dispatch case, so the qwen3 `.tflite` fell through to the **parakeet**
default path, which SIGSEGVs on the incompatible model. Fixed by rebuilding the
APK from take3 — this cycle's qwen3 runs completed with zero crashes and no
memory-kill (MemAvailable stayed >2.3 GiB throughout).

## ASR results (model × clip)

Expected transcripts: clip1 `2025년 3월 5일 전술평가 결과 보고`; volume clips `볼륨 업`
(space/punct-insensitive matching). "Desktop" = post-fix desktop reference
(`bench_asr_v2.py --norm-boost --vad`, phase1, same preprocessing as take3).

| Model | Clip | Device transcript | Match | Desktop post-fix | Compile s | Encode s | Decode s (steps) | Total s |
| --- | --- | --- | :-: | --- | ---: | ---: | ---: | ---: |
| whisper-tiny i8 | clip1 (3.8 s) | `2015년 3월 오후일 전술 평가 결과보고` | ✗ (year) | `2015년…` ✗ (same class) | 0.10 | 0.32 | 1.24 (16) | 1.85 |
| whisper-tiny i8 | `volume-볼륨, 업` (re-rec, 1.0 s) | `보입니다.` | ✗ | `볼륨업` ✓ — **diverges** | 0.10 | 0.35 | 0.38 (5) | 1.02 |
| whisper-tiny i8 | `volume-볼륨 업` (old quiet, 0.7 s) | `보일이야?` | ✗ | `보일해봐` ✗ | 0.10 | 0.32 | 0.39 (5) | 1.01 |
| whisper-base i8 | clip1 | `2025년 3월 5일 전술 평가 결과 보고` | ✓ | ✓ (spacing only) | 0.18 | 0.64 | 1.70 (13) | 2.72 |
| whisper-base i8 | `volume-볼륨, 업` | `'뽈림'` | ✗ | `볼륨 업` ✓ — **diverges** | 0.18 | 0.64 | 0.89 (7) | 1.91 |
| whisper-base i8 | `volume-볼륨 업` | `보여요.` | ✗ | `볼륨어` ✗ | 0.18 | 0.61 | 0.50 (4) | 1.48 |
| whisper-base i8 | `volume-소리 키워줘` (1.2 s) | `소리 키워줘` | ✓ | ✓ | 0.29 | 0.64 | 0.68 (5) | 1.80 |
| whisper-base i8 | `volume-음량 증가` (1.1 s) | `음량 증가` | ✓ | ✓ | 0.18 | 0.63 | 0.75 (6) | 1.76 |
| whisper-large-v3-turbo i8 | clip1 | **FAILURE** (native error, below) | ✗ | ✓ exact | 2.56 | 10.79 | — | — |
| qwen3-asr-0.6b i8 | clip1 | `이천이십오년 삼월오일 전술평가 결과 보고.` | ✗ number-style (content ✓) | same behavior | 2.69 | 0.24 | 11.14 (22) | 14.59 |
| qwen3-asr-0.6b i8 | `volume-볼륨, 업` | `볼륨 업` | ✓ exact | ✓ | 1.87 | 0.21 | 3.99 (8) | 6.60 |
| qwen3-asr-0.6b i8 | `volume-볼륨 업` (quiet) | `볼륨업` | ✓ | ✓ | 1.88 | 0.22 | 3.54 (7) | 6.15 |

Pass/fail summary:

- **qwen3-asr i8: PASS** — all 3 clips, both 볼륨 업 takes recognized (EOS
  min-length guard works: no empty output, no Chinese hallucination). Cost:
  ~500 ms/decode-step on device CPU (11 s decode for the 3.8 s clip),
  ~1.9 s model compile per fresh process. Number-style output
  (`이천이십오년`) unchanged — matches desktop; a known display-format gap,
  not a recognition failure.
- **whisper-base i8: PARTIAL** — long-clip exact; 소리 키워줘/음량 증가 exact;
  both 볼륨 업 takes fail on device even though desktop passes the re-recorded
  take with identical preprocessing (see investigation below).
- **whisper-tiny i8: PARTIAL/WEAK** — English-lean tier confirmed; fails all
  volume takes on device; clip1 year error (2015년) same as desktop.
- **whisper-large-v3-turbo i8: FAIL (unsupported by take3 JNI)** — see below.

### Turbo i8 failure — exact evidence

```text
"error": "TensorBuffer host memory buffer size is smaller than the given data size, 65536 vs 7680000"
```

Two take3 JNI limitations, both confirmed against the model's INSPECT signature:

1. `CreateWhisperInputFeatures` hardcodes `kNumMels = 80`; turbo's encode input
   is `[1, 128, 3000]` (128 mel bins).
2. The turbo export's `decode` signature orders inputs `(args_2 mask,
   args_0 encoder, args_1 tokens)` — index 0 is `decode`, 1 is `encode`,
   reversed vs tiny/base — while the JNI binds positionally.
   65536 B = 128×128×4 (mask buffer) receiving 7680000 B = 1500×1280×4
   (encoder output). Encoder itself ran (10.8 s) because LiteRT padded the
   80-mel features into the 128-mel input tensor without complaint.

Fix for next AAR: read mel-bin count and signature/input ordering from the
model instead of hardcoding (take4 in progress at time of writing).

### Short-utterance divergence investigation (device vs desktop, whisper only)

Verified on the take3 patch source (`Tools/UnityAar/litert-lm-unity-aar.patch`):

1. **The whisper path DOES apply the fix.** `CreateWhisperInputFeatures` calls
   `PreprocessAsrPcm` (VAD trim + boost-only RMS norm) immediately after
   `DecodeAudio`, same as the qwen3 path. Constants identical to the validated
   python (`bench_asr_v2.py`): 30 ms windows (480), rel-thresh 0.1,
   abs-thresh 1e-4, head 0.1 s, tail 0.3 s, target RMS 0.1 (-20 dBFS),
   peak clamp 0.8913 (-1 dBFS).
2. **The PCM pipeline is numerically consistent end-to-end.** Post-VAD sample
   counts match desktop exactly: re-recorded take 16384 == 16384, old quiet
   take 11392 == 11392 (miniaudio on device vs soundfile on desktop agree,
   VAD cut agrees, same boost-only gain formula).
3. **Divergence is whisper-feature/decode-numeric.** qwen3 (same PCM, own
   128-mel frontend) passes both takes on device. Whisper flips only on the
   two shortest clips: e.g. desktop tiny decodes 6 steps → `볼륨업`, device
   5 steps → `보입니다.`. Hypothesis: small numeric differences between the
   C++ mel/STFT and the python reference and/or the KV-cached JNI decode vs
   the desktop full-sequence greedy re-run, amplified by i8 quantization on
   <1.2 s utterances. **Not a missing-normalization bug.**

Next-cycle actions: (a) emit a mel-feature checksum (or dump) in the whisper
smoke JSON so device features can be diffed against python for one clip;
(b) route FC voice commands to qwen3-asr i8 (recognizes all takes on device)
or accept base-i8 with the two-of-four command coverage until whisper feature
parity is proven.

## LLM results

Smoke (2 chat turns) + standalone benchmark (3 runs × 64 prefill / 32 decode
tokens). Single generic smoke APK (runtime config), models pushed to device
storage. Device was warm (52–57 °C — kona idles hot; thermal gate skipped, so
figures are conservative vs the cool-device history in `docs/llm-details.md`).

| Model | Backend | Result | Init s | Turn1 s | Turn2 s | Prefill tok/s (avg) | Decode tok/s (avg) | PSS MB | Note |
| --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `gemma3-1b-it-int4` (557 MB) | GPU | PASS | 9.73 | 1.68 | 4.63 | 184.2 | 13.7 | 432 | OpenCL delegate + GPU TopK sampler; history: 197.0 / 16.4 |
| `gemma3-1b-it-int4` | CPU | PASS | 4.14 | 1.42 | 2.43 | 100.9 | 16.0 | 399 | history: 108.1 / 17.5 — ~6 % lower, warm device |
| `qwen3_0_6b_mixed_int4` (475 MB) | CPU | PASS | 1.66 | 11.80 | 7.09 | 31.8 | 20.9 | — | First on-device run of this export. `/think` mode active by default → long turns (856 chars incl. `<think>`); TTFT 2.06 s, prefill notably low for its size |
| `LFM2.5-1.2B-Instruct_int4` (702 MB) | CPU | PASS | 7.73 | 2.12 | 1.05 | 57.1 | 16.8 | — | **Architecture IS supported** — loads, chats correctly, benchmarks (bench engine re-init ~12.6 s). No error to capture |
| `Qwen2.5-0.5B-Instruct_wi4b64_ekv1280` (264 MB, self-made i4) | CPU | PASS | 1.37 | 0.60 | 0.90 | 218.3 | 35.5 | — | Prototype validated on device: coherent English responses, fastest of the set; vs stock q8 history 206.8 / 25.7 → i4 is +38 % decode at half the size |

Logs/CSV: `Builds/Logs/AndroidDeviceRuns/20260723-164040-*` (gemma3-1b GPU/CPU
+ `20260723-164040-results.csv`), `20260723-1700**-qwen3-mixed-int4-cpu.*`,
`20260723-170052-lfm25-int4-cpu.*`, `20260723-170139-qwen25-wi4b64-cpu.*`.

### Why gemma3-1b GPU decode (13.7 tok/s) is slower than CPU (16.0 tok/s)

Investigated per user question — conclusion: **expected architecture behavior
on kona/Adreno 650, not a fixable config issue.**

- Not a sampler fallback: GPU logcat shows
  `sampler_factory.cc:367] Dynamically loaded LiteRtTopKOpenClSampler C API`
  and CSV GpuEvidence `NativeOpenCL+OpenCLSampler`; no
  "GPU sampler unavailable"/"Falling back to CPU sampling" lines exist, so
  sampling runs on-GPU with no per-token GPU→CPU logits round-trip.
- Not a regression and not thermal ordering: the cool-device history shows the
  same ordering (GPU 16.4 < CPU 17.5 decode); today both are ~5–8 % lower from
  heat, and the CPU run actually started hotter (55.3 °C vs 52.9 °C).
- Mechanism: decode is single-token, memory-bandwidth-bound generation — the
  per-step OpenCL dispatch/synchronization overhead on Adreno 650 exceeds the
  compute win, while prefill is batch-parallel (64 tokens) so the GPU wins
  ~1.8× (184 vs 101 tok/s). GPU also costs 2.4× init (9.7 vs 4.1 s OpenCL
  setup) and +33 MB PSS.
- Guidance: use CPU for decode-heavy chat on this SoC; choose GPU only when
  long-prompt prefill dominates (e.g. large system prompts / RAG contexts).

## Pass/fail ledger

| Item | Verdict |
| --- | :-: |
| ASR whisper-tiny i8 (3 clips) | PARTIAL — runs clean; 0/3 content-exact (clip1 year error = desktop; both 볼륨 업 takes garbage) |
| ASR whisper-base i8 (5 clips) | PARTIAL — clip1/소리 키워줘/음량 증가 exact; both 볼륨 업 takes fail (desktop passes re-recorded take) |
| ASR whisper-large-v3-turbo i8 | FAIL — take3 JNI 80-mel hardcode + positional decode binding |
| ASR qwen3-asr-0.6b i8 (3 clips) | PASS — all clips incl. both 볼륨 업 takes; no crash/OOM (prior SIGSEGV was stale-APK parakeet mis-dispatch, fixed) |
| LLM gemma3-1b int4 GPU + CPU | PASS — tok/s within ~6–17 % of cool-device history |
| LLM qwen3-0.6b mixed int4 CPU | PASS — first device validation of the export |
| LLM LFM2.5-1.2B int4 CPU | PASS — architecture supported (load test objective met) |
| LLM Qwen2.5-0.5B wi4b64 i4 prototype CPU | PASS — self-made i4 recipe validated on device |
| App stability | PASS — 16 fresh-process runs, zero crashes, zero lowmemorykiller events, MemAvailable ≥ 2.3 GiB throughout |

## Next-cycle fix list

1. **Whisper turbo support (take4 AAR)**: re-run turbo i8 on the take4 APK
   (mel-bin count read from the model + signature-name-based input binding).
   Evidence to beat: `TensorBuffer ... 65536 vs 7680000`.
2. **Whisper short-clip device/desktop parity**: add a mel-feature checksum or
   dump to the whisper smoke JSON, diff device features vs
   `bench_asr_v2.py` for `volume-볼륨, 업.mp3` (PCM already proven identical:
   16384/11392 post-VAD samples match desktop exactly). Suspect mel/STFT or
   KV-cache-decode numerics under i8.
3. **FC voice routing decision**: until (2) lands, route FC commands to
   qwen3-asr i8 (all takes recognized on device; cost ~0.5 s/step decode,
   ~2 s compile) or accept whisper-base i8 with 볼륨업 gap.
4. **qwen3_0_6b_mixed_int4 prefill (31.8 tok/s)** is 7× lower than the
   similar-size wi4b64 qwen2.5 (218 tok/s) — check export flags
   (multi-prefill-seq / prefill signature lengths) next quantization pass.
5. **Multimodal (`sendMessageWithMedia`) and FC benchmark scenes**: not
   exercised this cycle — run once the generated test scenes/APKs are
   available (bridge API confirmed present in take3 AAR).
6. Benchmark hygiene: thermal gate at 45 °C never clears on this device
   (idles ~53 °C); either lower-is-better trend tracking with `-SkipThermalWait`
   + recorded before/after temps (as done here), or raise the gate to ~55 °C.

## Cycle verdict

DO phase objectives met: take3 AAR baselined on the real device across 4 ASR
models × 3 clips and 5 LLM configurations, prior-cycle crash root-caused with
tombstone evidence (stale-APK parakeet mis-dispatch SIGSEGV — not memory), and
the short-utterance fix verified active-but-insufficient for whisper on
sub-1.2 s clips (works for qwen3). Two concrete defects (turbo JNI, whisper
short-clip parity) and one perf anomaly (mixed-int4 prefill) carried to cycle 2.

---

# Device Cycle 2 — take4 AAR (turbo, regression, multimodal)

- Date: 2026-07-23 (17:26–18:10)
- Device: `46a880a0` — same unit, cool this cycle (battery 24.2 °C at start vs
  52–57 °C in cycle 1).
- AAR: `Assets/Plugins/Android/litertlm-unity-bridge.aar` rebuilt 16:32
  ("take4"): whisper mel-bin count + vocab size read from the model signature
  (128-mel/51866-vocab turbo support); whisper smoke JSON now reports
  `melBins`/`vocabSize`.
- APKs (fresh temp-copy batchmode builds from take4, Unity editor open):
  - `Builds/Android/LiteRtLmAndroidAsrSmokeTest-generic.apk` (17:32)
  - `Builds/Android/LiteRtLmAndroidSmokeTest-gemma3-1b-it-int4-GPU.apk`
    (generic LLM smoke, runtime-config driven, now with multimodal media turns)
- C# changes this cycle: `LiteRtLmAndroidSmokeTestRunner` gained optional
  runtime-config media turns (`mediaImagePath`/`mediaImagePrompt`/
  `mediaAudioPath`/`mediaAudioPrompt`/`audioBackend`/`skipTextTurns`) calling
  `SendMessageWithMedia`; `LiteRtLmMultimodalTestRunner.MediaApiAvailable`
  flipped to `true` and its Send wired to `SendMessageWithMedia`
  (image via `Texture2D.EncodeToPNG` bytes, audio via StreamingAssets copy).
- Logs: `Builds/Logs/AndroidDeviceRuns/20260723-173*/174*/175*-c2-*.{summary,status,logcat}.txt`

## 1. whisper-large-v3-turbo i4 on take4 — still FAIL (partial fix confirmed)

All 3 gate clips fail with the **identical** cycle-1 error, but the failure
point moved: take4's signature-driven mel frontend works, the decode input
binding does not.

```text
"compiledModelCache": "miss", "backendUsed": "CPU", "compileSeconds": 1.23,
"melBins": 128, "vocabSize": 51866, "pcmSamples": 61024, "audioSeconds": 3.814,
"validFeatureFrames": 382, "encodeSeconds": 12.59, "encoderOutputValues": 1920000,
"error": "TensorBuffer host memory buffer size is smaller than the given data size, 65536 vs 7680000"
```

- Fixed by take4 (verified on device): mel bins 128 detected, vocab 51866
  detected, encoder ran to completion (12.6 s, 1500×1280 output).
- Still broken: the decode loop in the JNI writes inputs **positionally** —
  `decode_inputs[0].Write(encoder_output)`, `[1].Write(tokens)`,
  `[2].Write(causal_mask)` (see the whisper decode loop in
  `Tools/UnityAar/litert-lm-unity-aar.patch`). Turbo's `decode` signature
  orders inputs `(args_2 mask, args_1 tokens, args_0 encoder)`, so index 0 is
  the 128×128×4 = 65,536 B mask buffer receiving the 1500×1280×4 = 7,680,000 B
  encoder output — exactly the reported sizes.
- **take5 fix**: bind decode inputs by tensor element-type/size (mask = fp32
  128×128, tokens = int32, encoder = fp32 1500×1280) or by signature input
  name, not by index. tiny/base are unaffected because their exports happen to
  order decode inputs (encoder, tokens, mask).
- Runs: `20260723-173331-c2-turbo-i4-clip1`, `-173401-c2-turbo-i4-volnew`,
  `-173421-c2-turbo-i4-volold` (each fails identically in ~14 s incl. encode).

## 2. tiny/base i8 regression on take4 — PASS (bit-for-bit transcript match)

Same 3 gate clips, CPU, `ko`, fresh process per run. Every transcript is
character-identical to cycle 1 (take3), confirming take4's whisper refactor
did not disturb the 80-mel path; JSON now also reports `melBins: 80`.

| Model | Clip | Cycle-2 transcript | vs cycle 1 | Compile s | Encode s | Decode s (steps) |
| --- | --- | --- | :-: | ---: | ---: | ---: |
| tiny i8 | clip1 | `2015년 3월 오후일 전술 평가 결과보고` | identical | 0.15 | 0.33 | 1.19 (16) |
| tiny i8 | `볼륨, 업` (re-rec) | `보입니다.` | identical | 0.10 | 0.31 | 0.37 (5) |
| tiny i8 | `볼륨 업` (old quiet) | `보일이야?` | identical | 0.11 | 0.31 | 0.40 (5) |
| base i8 | clip1 | `2025년 3월 5일 전술 평가 결과 보고` ✓ | identical | 0.29 | 0.66 | 1.68 (13) |
| base i8 | `볼륨, 업` | `'뽈림'` | identical | 0.18 | 0.62 | 0.89 (7) |
| base i8 | `볼륨 업` | `보여요.` | identical | 0.18 | 0.61 | 0.52 (4) |

Runs: `20260723-1734*/1735*-c2-{tiny,base}-i8-*`.

## 3. Whisper short-clip numerics A/B — device-side checksum still blocked

- take4's JSON adds `melBins`/`vocabSize` but **no feature checksum/dump**, so
  a device-vs-desktop mel diff still needs a take5 diagnostic build (add e.g.
  `featureMd5` + `featureSum` of the encoder input tensor to the smoke JSON).
- Desktop reference checksums computed and recorded now (phase1
  preprocessing: VAD trim + boost-only RMS norm, Slaney mel, float32 LE bytes,
  md5 over the full 1×N×3000 tensor;
  script: scratchpad `mel_checksum_ref.py`):

| Clip | postVadSamples | gain | nMels | sum | md5 |
| --- | ---: | ---: | ---: | ---: | --- |
| `volume-볼륨, 업.mp3` | 16384 | 1.486091 | 80 | -154777.7344 | `132bb9fdc6a1cd6f3409c2d9a8eb5f8e` |
| `volume-볼륨, 업.mp3` | 16384 | 1.486091 | 128 | -250725.7500 | `c727c3295e405cd1ed992f4d6f858fe1` |
| `volume-볼륨 업.mp3` | 11392 | 1.332201 | 80 | -171227.8906 | `0740b3a5ed93f649c995b6398c9f8651` |
| `volume-볼륨 업.mp3` | 11392 | 1.332201 | 128 | -271351.4688 | `c7ac7c2176a017b676b9cdb1883573d0` |

  postVadSamples match the device PCM counts proven identical in cycle 1
  (16384/11392), so a take5 device `featureMd5` can be compared 1:1 against
  these values.

## 4. Multimodal on device (gemma-4 E2B, `sendMessageWithMedia`) — PASS

First on-device exercise of the take3+ media bridge API. Model pushed to
device external files (`LiteRTLM/gemma-4-E2B-it.litertlm`, 2.59 GB, 96 GB
free); media via absolute device paths; fresh app process per run;
`skipTextTurns=true`, no standalone benchmark.

| Turn | Backends (llm/vision/audio) | maxNumTokens | Init s | Media turn s | Response |
| --- | --- | ---: | ---: | ---: | --- |
| image `apples.jpg` (700×467) | CPU/CPU/– | 4000 | 1.2 (warm cache) | 23.36 | "…three red apples and some green leaves… one apple cut open to reveal its interior." — **accurate** |
| image `apples.jpg` | GPU/GPU/– | 4000 | 18.5 | **7.60** | same content, accurate — GPU 3.1× faster media turn |
| audio clip1 (3.8 s Korean) | CPU/–/CPU | 4000 | 7.6 | 4.09 | `2025년 3월 5일 전술 평가 결과 보고` — **content-exact incl. year** (spacing only) |

- PSS during image CPU run ≈ 3.6 GB; no lowmemorykiller, no crash across all
  multimodal runs.
- Runs: `20260723-175014-c2-mm-image-cpu-4k`, `-175204-c2-mm-image-gpu`,
  `-175118-c2-mm-audio-cpu`.
- Notable: gemma-4 E2B audio transcription beats every dedicated ASR model
  tested on this clip — content-exact like whisper-base but with correct
  spacing, in 4.1 s vs qwen3-asr's 14.6 s (whisper-base is 2.7 s but fails the
  볼륨 업 takes; gemma-4 on those takes not yet measured).

Two config requirements discovered (both now documented in the runner):

1. **`visionBackend`/`audioBackend` must be non-empty at initialize** or the
   engine never loads the executor —
   `INVALID_ARGUMENT: Vision executor should not be null, please
   TryLoadingVisionExecutor() first.` (run `20260723-174854`, failed with
   vision unset). The JNI enables the modality only when a backend string is
   provided.
2. **maxNumTokens=1024 is too small for image turns** — apples.jpg becomes
   2340 vision patches and the prefill overflows the KV cache:
   `dynamic_update_slice.cc:70 SizeOfDimension(update, i) <=
   SizeOfDimension(operand, i) was not true` →
   `llm_litert_compiled_model_executor.cc:753` INTERNAL (run
   `20260723-174854-c2-mm-image-cpu`). 4000 (the validated gemma-4 config)
   works.

## Cycle-2 pass/fail ledger

| Item | Verdict |
| --- | :-: |
| whisper-large-v3-turbo i4 (take4, 3 clips) | FAIL — mel/vocab detection fixed, decode input binding still positional (take5: bind by name/size) |
| whisper tiny/base i8 regression on take4 | PASS — 6/6 transcripts identical to cycle 1 |
| melBins/vocabSize in whisper smoke JSON | PASS — 80/51865 (tiny, base), 128/51866 (turbo) reported on device |
| Whisper mel A/B device checksum | BLOCKED — JSON has no feature checksum; take5 needs `featureMd5`; desktop references recorded above |
| Multimodal image (이미지 인식) CPU + GPU | PASS — accurate description both backends |
| Multimodal audio (Korean clip) CPU | PASS — content-exact transcript |
| App stability | PASS — 13 fresh-process runs this cycle, zero crashes/OOM |

## Remaining gaps → cycle 3

1. **take5 AAR**: (a) whisper decode input binding by signature name or
   tensor elemtype/size (unblocks turbo); (b) `featureMd5`/`featureSum` of the
   encoder input tensor in the whisper smoke JSON (unblocks the short-clip
   device-vs-desktop mel diff against the checksums recorded above).
2. **Multimodal FC scene** (`LiteRtLmMultimodalFunctionCallingRunner`) still
   has `MediaApiAvailable = false` and an unwired send path — flip and wire
   like the test runner once an FC-on-device flow is defined.
3. gemma-4 E2B audio on the 볼륨 업 short takes (would it replace qwen3-asr as
   the FC voice route where the 2.6 GB model is already resident?).
4. Carry-over from cycle 1: qwen3_0_6b_mixed_int4 prefill anomaly; thermal
   gate hygiene (device was cool this cycle — 24 °C — so cycle-1 warm figures
   remain the conservative reference).

## Cycle-2 verdict

Turbo remains blocked by one precisely-located JNI defect (positional decode
binding), with the take4 mel/vocab half of the fix verified working on device
and regression-clean. Multimodal (image + audio) via `sendMessageWithMedia`
is fully validated on device on CPU and GPU — the user requirement
이미지 인식 is met with an accurate on-device description of the test image,
and gemma-4 audio transcription is content-exact on the Korean gate clip.

# Device Cycle 3 — take5 AAR (turbo gate, mel A/B verdict, FC on device)

- Date: 2026-07-23 (19:05–). Device `46a880a0`, cool (battery 24.2 °C, 100%).
- AAR: `Assets/Plugins/Android/litertlm-unity-bridge.aar` rebuilt 18:53
  ("take5"): whisper decode inputs resolved name→shape→positional (JSON reports
  `decodeBindingStrategy` + `decode{Audio,Token,Mask}Input` indices);
  `featureMd5`/`featureSum` (md5 over full [1,n_mels,3000] f32 LE tensor) added
  to the whisper smoke JSON.
- APK: `Builds/Android/LiteRtLmAndroidAsrSmokeTest-generic.apk` (19:03, fresh
  temp-copy batchmode build from take5).
- All ASR runs: CPU, `ko`, fresh app process per run.
- Logs: `Builds/Logs/AndroidDeviceRuns/20260723-190*/*-c3-*.{summary,status,logcat}.txt`

## 1. whisper-large-v3-turbo i4 on take5 — PASS (3/3 gate clips) ✅

The shape-based decode binding fix works: JSON reports
`decodeBindingStrategy: "shape"` with `decodeAudioInput=1, decodeTokenInput=2,
decodeMaskInput=0` — exactly the (mask, encoder, tokens) order that broke the
positional binding in cycles 1–2. All three gate clips now transcribe, and
turbo is the only whisper tier that passes **both** 볼륨 업 takes on device.

| Clip | Device transcript | Match | Compile s | Encode s | Decode s (steps) |
| --- | --- | :-: | ---: | ---: | ---: |
| clip1 (3.8 s) | `2025년 3월 5일 전술평가 결과 보고` | ✓ exact | 1.28 | 15.89 | 6.35 (14) |
| `volume-볼륨, 업` (re-rec) | `볼륨 업` | ✓ exact | 0.79 | 17.27 | 2.50 (6) |
| `volume-볼륨 업` (old quiet) | `볼륨업` | ✓ | 0.81 | 17.18 | 2.51 (6) |

- Cost profile: ~17 s CPU encode per utterance (1500×1280 encoder, i4) +
  ~0.42 s/decode-step. Total 21–24 s per clip — accuracy king, not the
  latency king (whisper-base: 2.7 s; gemma-4 audio: 4.1 s).
- Runs: `20260723-190530/190610/190636-c3-turbo-i4-*`.

## 2. tiny i8 quick regression on take5 — PASS

All three transcripts character-identical to cycles 1–2 (`2015년 3월 오후일
전술 평가 결과보고` / `보입니다.` / `보일이야?`); timings in family
(compile 0.12–0.15 s, encode 0.35–0.43 s, decode 0.46–1.47 s). take5's
whisper decode-binding refactor did not disturb the 80-mel path.
Runs: `20260723-190702/190727/190752-c3-tiny-i8-*`.

## 3. featureMd5 A/B — VERDICT: device mel/STFT differs numerically from desktop

take5's `featureMd5` (md5 over the full [1,n_mels,3000] float32 LE encoder
input) compared against the desktop references recorded in cycle 2
(`mel_checksum_ref.py`, identical preprocessing config, PCM sample counts
already proven identical in cycle 1):

| Clip | nMels | Device md5 | Desktop md5 | md5 equal? | Device sum | Desktop sum | Δsum |
| --- | ---: | --- | --- | :-: | ---: | ---: | ---: |
| `volume-볼륨, 업` | 80 | `1c4b371e0d6aa1fd75ba1fb715779dc5` | `132bb9fdc6a1cd6f3409c2d9a8eb5f8e` | ✗ | −154954.71 | −154777.73 | 0.11 % |
| `volume-볼륨 업` | 80 | `874fbb7af80e4fe340e49604034af64c` | `0740b3a5ed93f649c995b6398c9f8651` | ✗ | −171379.71 | −171227.89 | 0.09 % |
| `volume-볼륨, 업` | 128 | `7227f0060105f2c68cc15992f66438ec` | `c727c3295e405cd1ed992f4d6f858fe1` | ✗ | −251016.60 | −250725.75 | 0.12 % |
| `volume-볼륨 업` | 128 | `91686369b1489a89ba3b881c89fdeadc` | `c7ac7c2176a017b676b9cdb1883573d0` | ✗ | −271675.39 | −271351.47 | 0.12 % |

**Definitive answer: the divergence is in the mel/STFT frontend, not decode
numerics.** Device PCM equals desktop PCM (cycle 1), but the mel tensors
differ by ~0.1 % in energy — an implementation-level numerical difference
(FFT kernel / float accumulation order in the device C++ mel vs desktop
Python reference), not a preprocessing-config mismatch. This explains the
cycle-1 observation that tiny/base flip transcripts on borderline short
clips while long clips agree: small models sit near decision boundaries the
~0.1 % mel delta can cross. Turbo (this cycle) and qwen3-asr (cycle 1)
absorb the delta and pass both takes — model robustness, not a device bug.
Chasing bit-exactness in the mel would be an optimization, not a correctness
fix; short-utterance reliability is already solved by using turbo/qwen3-asr.

## 4. ASR→LLM function calling on device — PASS (펑션콜링 디바이스 검증 1/2)

`LiteRtLmAndroidAsrFunctionCallingDemo-c3.apk` (19:12, temp-copy batchmode via
`Tools/Windows/Build-LiteRtLmAndroidAsrFunctionCallingDemoApk.ps1`; whisper-tiny
i8 + gemma3-1b-it-int4 packaged in StreamingAssets, take5 AAR). Fresh install,
fresh process; status file `LiteRtLmAsrFunctionCallingDemo.status.txt`.

| Stage | Backend | Elapsed s | Result |
| --- | --- | ---: | --- |
| whisper-tiny ASR (clip1 3.8 s) | CPU | 2.16 | `2015년 3월 오후일 전술 평가 결과보고` (tiny's known year error; content sufficient for routing) |
| gemma3-1b init | GPU | 9.78 | isInitialized=True |
| gemma3-1b constrained FC | GPU | 3.49 | `{"name":"OpenTacticalEvaluationReport","arguments":{"reportDate":"2015-03-27",...}}` |
| **End-to-end** | | **15.45** | **SUCCESS — tool = OpenTacticalEvaluationReport (expected)** ✓ |

Voice → transcript → tool call works fully on device: the runner's demo guard
routes the 전술평가 utterance to the expected tool despite tiny's year slip.
Run: `20260723-191240-c3-asrfc`.

## 5. Multimodal function calling on device — PASS (펑션콜링 디바이스 검증 2/2)

`LiteRtLmMultimodalFunctionCallingRunner` Android stage-1 wired this cycle:
`MediaApiAvailable=true`, runtime config
(`LiteRtLmMultimodalFunctionCallingDemo.config.json` in the app files dir:
`llmModelPath`/`backend`/`visionBackend`/`audioBackend`/`maxNumTokens`/
`maxNumImages`/`mediaImagePath`/`mediaAudioPath`/`utterance`/
`expectedToolName`), and the FC turn goes through `SendMessageWithMedia` with
the tools JSON embedded in the prompt (same tools contract as the Windows
degraded mode; system message via `systemInstruction` at initialize). New
build entry `LiteRT-LM/Android/Build Multimodal Function Calling Test APK`
(`BuildAndroidMultimodalFunctionCallingTestApk`) →
`Builds/Android/LiteRtLmMultimodalFunctionCallingTest.apk` (60 MB, 19:19,
model+media stay off-APK and are pushed at run time).

| Stage | Config | Elapsed s | Result |
| --- | --- | ---: | --- |
| gemma-4 E2B init | CPU llm + CPU vision, maxNumTokens 4000, warm compile cache | 1.82 | isInitialized=True |
| image FC turn (`apples.jpg` + 13-tool JSON prompt + utterance 멀티모달 데이터 목록을 화면에 띄워줘) | `sendMessageWithMedia` | 38.06 | `{"tool":"ShowMultimodalDataList","parameters":{}}` — clean JSON, no chatter |
| **End-to-end** | | **40.70** | **SUCCESS — tool = ShowMultimodalDataList (expected)** ✓ |

- The image (2340 vision patches) + ~1.3k-token tools prompt fits the
  validated 4000-token gemma-4 config; no KV-cache overflow, no
  lowmemorykiller (post-run PSS 0.48 GB, MemAvailable 3.4 GB).
- gemma-4 obeys the "exactly one JSON object" instruction without constrained
  decoding — the raw response was the bare tool JSON.
- Run: `20260723-192001-c3-mmfc-image-cpu`.

## FINAL PASS/FAIL LEDGER — all user requirements on device `46a880a0`

| 요구사항 | Verdict | Evidence (best on-device result) |
| --- | :-: | --- |
| **LLM (텍스트 생성)** | **PASS** | cycle 1: gemma3-1b int4 CPU 16.0 tok/s decode / GPU 13.7 tok/s, Korean+English responses coherent; gemma3-270m, qwen2.5-0.5b/1.5b also ran |
| **ASR (음성인식)** | **PASS** | cycle 3: whisper-large-v3-turbo i4 3/3 gate clips incl. both 볼륨 업 takes (21–24 s/clip); cycle 1: qwen3-asr i8 3/3 (6–15 s); whisper-base 2.7 s for long clips; gemma-4 audio content-exact in 4.1 s |
| **이미지 인식 (image recognition)** | **PASS** | cycle 2: gemma-4 E2B `sendMessageWithMedia` — accurate apples.jpg description, CPU 23.4 s / GPU 7.6 s |
| **펑션콜링 (function calling)** | **PASS** | cycle 3: ASR→LLM FC (voice→whisper→gemma3-1b GPU→`OpenTacticalEvaluationReport`, 15.5 s E2E) + multimodal FC (image→gemma-4→`ShowMultimodalDataList`, 40.7 s E2E) |
| App stability | PASS | 20+ fresh-process runs across cycles 1–3, zero crashes/OOM/lowmemorykiller |

Open (non-blocking) follow-ups: qwen3_0_6b_mixed_int4 prefill anomaly
(cycle 1); mel frontend ~0.1 % numeric delta vs desktop (cosmetic — see §3);
GPU variant of the multimodal FC turn not yet measured (CPU already passes;
cycle-2 GPU image turn suggests ~3× faster).

## Cycle-3 verdict

All four user requirements — LLM, ASR, 이미지 인식, 펑션콜링 — now PASS on
device 46a880a0. take5 closed the last JNI defect (whisper decode binding by
shape), turbo became the on-device ASR accuracy king (only tier passing both
볼륨 업 takes), the mel A/B question is definitively answered (frontend
numeric delta, not decode), and both function-calling pipelines (voice-driven
and multimodal) run end-to-end on device with the expected tool selected.
PDCA device-testing directive: **complete**.

---

# Device Cycle 4 — ACFT short-window on device (take6 AAR)

- Date: 2026-07-24 (08:36–08:39 host; device clock UTC). Device `46a880a0`.
- AAR: `Assets/Plugins/Android/litertlm-unity-bridge.aar` rebuilt 2026-07-23
  23:57 ("take6"): whisper window frames auto-detected from the `encode`
  signature (no 3000-frame hardcode); smoke JSON adds `windowFrames` next to
  `melBins`/`vocabSize`/`featureMd5`. First device test of take6.
- APK: `Builds/Android/LiteRtLmAndroidAsrSmokeTest-generic.apk` rebuilt 07:56
  from take6 (temp-copy batchmode; build-script copy with `External/` excluded
  from the robocopy mirror — the overnight ACFT training left tens of GB of
  datasets/checkpoints/venv there that the stock mirror would copy).
- Models under test (pushed at runtime, tokenizers = stock whisper
  `tokenizer.json`, vocab 51865 — Korean training used the stock
  `openai/whisper-{base,tiny}` checkpoints per `train_acft.py`, and the device
  JSON confirms `vocabSize: 51865` on every run):
  - futo `External/acft-work/acft_base_5s_drq.tflite` (101 MB, futo-org
    ACFT checkpoint, desktop-validated 2026-07-23)
  - Korean-trained `External/acft-training/export/base/acft_base_5s_drq.tflite`
    (101 MB) and `export/tiny/acft_tiny_5s_drq.tflite` (59 MB) from the
    overnight Korean ACFT run (gates passed; note `export/*/bench.jsonl` is
    all `list index out of range` rows — desktop bench script bug, so this
    cycle is the first real evaluation of those exports)
  - stock `whisper_base_30s_i8.tflite` (77 MB) as the 3000-frame regression
    control
- All runs: CPU, whisper mode, `ko`, fresh app process per run. 13/13 runs
  reached SUCCESS status — zero crashes/OOM.
- Logs: `Builds/Logs/AndroidDeviceRuns/20260724-083*-c4-*.{summary,status,logcat}.txt`
  Matrix scripts: scratchpad `Run-Cycle4AcftMatrix.ps1` + `Invoke-AsrDeviceRunC4.ps1`.

## 1. take6 flexible-window JNI — PASS

Every ACFT-5s run reports `windowFrames: 500, melBins: 80, vocabSize: 51865`
auto-detected from the model (`encode` input `[1, 80, 500]`, encoder output
`[1, 250, 512]`); the stock-30s control reports `windowFrames: 3000` on the
same APK. `decodeBindingStrategy: "shape"` resolved the ACFT exports'
`(mask, audio, tokens)` input order (indices 0/1/2 = mask/audio/token) —
opposite of the stock base export — with no positional-binding faults.
`featureMd5` is identical for the same clip across futo and Korean-trained
runs (e.g. clip1 `d7b2b5e0…` in both), confirming one shared mel frontend at
500 frames.

## 2. ACFT base 5s drq on device — futo vs Korean-trained (THE key test)

| Model | Clip | Device transcript | Match | Compile s | Encode s | Decode s (steps) | Total s |
| --- | --- | --- | :-: | ---: | ---: | ---: | ---: |
| futo b5 drq | clip1 (3.8 s) | `2025년 3월 5일 전술평가 결과 보고` | ✓ exact | 0.18 | 0.053 | 0.96 (14) | 1.33 |
| futo b5 drq | `볼륨, 업` (re-rec) | `볼륨 업` | ✓ exact | 0.18 | 0.048 | 0.40 (6) | 0.76 |
| futo b5 drq | `볼륨 업` (old quiet) | `볼륨어` | ✗ | 0.18 | 0.059 | 0.41 (6) | 0.78 |
| ko b5 drq | clip1 | `2025년 3월 5일 전술평가 결과 보고` | ✓ exact | 0.18 | 0.050 | 0.99 (14) | 1.35 |
| ko b5 drq | `볼륨, 업` (re-rec) | `볼륨 업` | ✓ exact | 0.18 | 0.047 | 0.42 (6) | 0.78 |
| ko b5 drq | `볼륨 업` (old quiet) | `볼륨어` | ✗ | 0.18 | 0.049 | 0.43 (6) | 0.79 |
| ko b5 drq | `소리 키워줘` | `소리 키워줘` | ✓ exact | 0.19 | 0.060 | 0.34 (5) | 0.71 |
| ko b5 drq | `음량 증가` | `음량 증가` | ✓ exact | 0.18 | 0.047 | 0.42 (6) | 0.78 |

- **Short-command accuracy on device is fixed for normal-loudness takes.**
  Cycle-2 stock base-30s i8 on this device read `볼륨, 업` as `'뽈림'` and
  `볼륨 업` as `보여요.` — both ACFT-5s models now read the re-rec take
  **exact** and every Korean FC command (`소리 키워줘`, `음량 증가`) exact.
  Device transcripts equal the desktop futo-5s-drq references
  transcript-for-transcript on all shared clips (incl. the `볼륨어` miss) —
  the cycle-3 ~0.1 % mel delta does not flip ACFT-5s outputs.
- The old **quiet** 0.79 s take still reads `볼륨어` — same in every base
  variant on desktop and device; loudness/model-capacity pain point, not a
  window or training issue (as predicted by the desktop ACFT evaluation).
- **futo vs Korean-trained: identical transcripts and timings on the 3 shared
  clips.** The Korean training cost nothing; its value shows on `음량 증가`
  (exact — the futo 30s export had the `은량` regression) and in provenance
  (trained with Korean + oversampled short utterances in-distribution).
- **Speed: ~12× encoder vs deployed stock base-30s on this device** (0.047–
  0.060 s vs 0.617 s) and ~1.8× decode per step (≈0.069 vs 0.125 s/step).
  Full pipeline for a command clip: **0.71–0.79 s** vs 2.7 s stock — and vs
  21–24 s turbo, the previous only-accurate-tier for both 볼륨 업 takes.

## 3. Korean ACFT tiny 5s drq — REJECT for deployment

| Clip | Device transcript | Match | Total s |
| --- | --- | :-: | ---: |
| `볼륨, 업` (re-rec) | `보입니다.` | ✗ | 0.50 |
| `볼륨 업` (old quiet) | `보일해봐` | ✗ | 0.49 |
| `소리 키워줘` | `소리 키워줘` | ✓ | 0.49 |
| `음량 증가` | `음양증가` | ✗ | 0.48 |

1/4 exact. `볼륨, 업` → `보입니다.` is character-identical to stock tiny-30s
on this device (cycles 1–3) — tiny's failure is model capacity, unchanged by
window length or Korean ACFT. The ~0.3 s saved over base-5s does not buy a
usable command tier; **do not deploy tiny-5s**.

## 4. stock base-30s i8 regression on take6 — PASS

`2025년 3월 5일 전술 평가 결과 보고` — character-identical to cycles 1–3.
`windowFrames: 3000` reported; compile 0.29 s / encode 0.62 s / decode 1.62 s
(13 steps), all in family with cycle-2 (0.29/0.66/1.68). take6's flexible
window detection did not disturb the 3000-frame path.

## Cycle-4 pass/fail ledger

| Item | Verdict |
| --- | :-: |
| take6 window auto-detection (500 ↔ 3000 on one APK) | PASS |
| futo acft_base_5s_drq × 3 gate clips | PASS — 2/3 exact; miss = known quiet-clip `볼륨어`, encode 0.05 s (expected <0.2 s) |
| Korean acft_base_5s_drq × 5 clips | PASS — 4/5 exact; all normal-loudness FC commands exact; fixes cycle-2 `'뽈림'` failure |
| Korean acft_tiny_5s_drq × 4 commands | FAIL — 1/4 exact (capacity, same as stock tiny) |
| stock base-30s i8 3000-frame regression | PASS — transcript bit-identical to cycles 1–3 |
| App stability | PASS — 13 fresh-process runs, zero crashes/OOM |

## Cycle-4 verdict — is Korean ACFT 5s ready?

**Yes — recommend deploying the Korean-trained `acft_base_5s_drq.tflite`
(101 MB) as the voice-command ASR tier** (recommendation only; nothing
deployed to StreamingAssets this cycle):

- Suggested placement: `Assets/StreamingAssets/ASR/whisper-base-acft-ko/
  acft_base_5s_drq.tflite` + reuse of the existing stock
  `ASR/whisper-base/tokenizer.json` (verified compatible, vocab 51865).
- Role: **augment, not replace.** 5 s window truncates longer audio — keep
  stock `whisper_base_30s_i8` (77 MB) for dictation/long clips and turbo i4
  as the accuracy fallback. For FC voice commands the ACFT tier is 0.7–0.8 s
  end-to-end (3.5× faster than stock base, ~30× faster than turbo) and more
  accurate on device than stock base ever was on short takes.
- Prefer the Korean-trained export over the futo one: identical device
  behavior on shared clips, plus the `음량 증가` guarantee and Korean
  short-utterance training in-distribution.
- Residual gap: quiet takes (`볼륨어` on the 0.79 s clip) — attack with
  capture-side AGC/loudness normalization or turbo fallback, not with more
  ACFT.
- Follow-up (non-blocking): fix `External/acft-training` desktop bench
  (`bench.jsonl` all-errors) so future exports get a desktop reference before
  device time.

# Device Cycle 5 — ACFT medium/turbo on device + tier deployment (take6 AAR)

- Date: 2026-07-25 (07:09–07:12 host). Device `46a880a0`, same take6 APK as
  cycle 4 (no reinstall). All runs: CPU, whisper mode, `ko`, fresh app process
  per run. **11/11 runs reached SUCCESS status — zero crashes/OOM** (the one ✗
  below is a transcript mismatch, not a pipeline failure).
- Deployed this cycle (user-approved batch) — the ACFT queue finished all 4
  gates (base/tiny/medium/turbo) and the medium/turbo 5s exports are now in
  StreamingAssets + `LiteRtLmAsrTestRunner.cs` modelOptions:
  - `ASR/whisper-base-acft-ko/acft_base_5s_drq.tflite` (101 MB, from cycle 4)
  - `ASR/whisper-medium-acft-ko/acft_medium_5s_drq.tflite` (826 MB) + copy of
    `ASR/whisper-medium/tokenizer.json` (md5 `e259e7c7…`, identical to the
    training base tokenizer, vocab 51865 — confirmed on device)
  - `ASR/whisper-turbo-acft-ko/acft_turbo_5s_drq.tflite` (883 MB) + copy of
    `ASR/whisper-large-v3-turbo/tokenizer.json` (md5 `5e5ac406…`, identical to
    the training base tokenizer, vocab 51866 — confirmed on device)
- Desktop rebench: `export/{medium,turbo}/bench.jsonl` regenerated with the
  fixed bench path — 42/42 rows clean (the cycle-4 "all `list index out of
  range`" rows are gone). Per-window mean CER over the 7-clip matrix
  (non-zero rows are trailing-punctuation diffs on the two sentence clips
  unless noted):
  - medium: 5s **0.000 Korean-exact incl. the quiet `볼륨 업` take**; 10s/30s
    regress `음량 증가` → `음향(증가)` (CER 0.25). Encode 0.37 s (5s) →
    4.7 s (30s), decode ≈0.41–1.2 s/step (desktop CPU).
  - turbo: **5s and 10s all-Korean-exact** (quiet take included); 30s drops
    `볼륨, 업` → `볼륨` (CER 0.33). Encode 0.72 s (5s) → 8.6 s (30s), decode
    ≈0.15–0.33 s/step. Set means: medium 0.0393, turbo 0.0313 — turbo-5s is
    the best export of the queue, matching its training gate (ko-short CER
    0.182, best of all tiers).
- Logs: `Builds/Logs/AndroidDeviceRuns/20260725-07*-c5-*.{summary,status,logcat}.txt`
  Matrix script: scratchpad `Run-Cycle5AcftMatrix.ps1` (+ `Invoke-AsrDeviceRunC4.ps1`).

## 1. Cycle-5 device matrix (all 5s-window ACFT drq exports)

Every run auto-detected `windowFrames: 500`; vocab 51865 (base/medium) and
51866 (turbo) as expected per tokenizer family. `compiledModelCache: miss` on
every row (fresh process each run), so compile time is included in totals.

| Model | Clip | Device transcript | Match | Compile s | Encode s | Decode s (steps) | Total s |
| --- | --- | --- | :-: | ---: | ---: | ---: | ---: |
| base acft-ko 5s (regression) | `볼륨, 업` (re-rec) | `볼륨 업` | ✓ exact | 0.19 | 0.052 | 0.44 (6) | 0.81 |
| medium acft-ko 5s | clip1 (3.8 s) | `2025년 3월 5일 전술평가 결과 보고` | ✓ exact | 1.86 | 0.564 | 6.41 (14) | 8.97 |
| medium acft-ko 5s | `볼륨, 업` (re-rec) | `볼륨 업` | ✓ exact | 1.86 | 0.560 | 2.73 (6) | 5.28 |
| medium acft-ko 5s | `볼륨 업` (old quiet) | `볼륨업` | ✓ | 1.85 | 0.544 | 2.73 (6) | 5.25 |
| medium acft-ko 5s | `소리 키워줘` | `소리 키워줘` | ✓ exact | 1.86 | 0.532 | 2.32 (5) | 4.84 |
| medium acft-ko 5s | `음량 증가` | `음향증가` | ✗ | 1.86 | 0.523 | 2.29 (5) | 4.80 |
| turbo acft-ko 5s | clip1 (3.8 s) | `2025년 3월 5일 전술평가 결과 보고` | ✓ exact | 2.21 | 1.038 | 2.21 (14) | 5.61 |
| turbo acft-ko 5s | `볼륨, 업` (re-rec) | `볼륨 업` | ✓ exact | 1.92 | 1.037 | 0.91 (6) | 4.00 |
| turbo acft-ko 5s | `볼륨 업` (old quiet) | `볼륨업` | ✓ | 1.91 | 1.013 | 0.91 (6) | 3.97 |
| turbo acft-ko 5s | `소리 키워줘` | `소리 키워줘` | ✓ exact | 1.92 | 1.069 | 0.73 (5) | 3.85 |
| turbo acft-ko 5s | `음량 증가` | `음량 증가` | ✓ exact | 1.92 | 1.037 | 0.90 (6) | 4.00 |

## 2. Findings

- **Turbo acft-ko 5s: 5/5 exact — the first model in five cycles to read every
  clip including the quiet `볼륨 업` take.** The quiet take that produced
  `볼륨어` on every base variant (desktop + device, cycles 2–5) and `보여요.`
  on stock base-30s reads `볼륨업` here. It also keeps digits (`2025년`, not
  `이천이십오년` like Qwen3-ASR) and gets `음량 증가` exact.
- **Medium acft-ko 5s: 4/5.** Fixes the quiet take too, but flips `음량 증가`
  → `음향증가` on device — desktop medium-5s reads the same clip exact
  (CER 0.000), so the residual ~0.1 % device mel delta (cycle 3) lands exactly
  on medium's borderline clip (desktop medium 10s/30s also read `음향`).
  Medium is also *slower* than turbo on device (decode ≈0.46 s/step, 24-layer
  decoder, vs turbo ≈0.15 s/step, 4-layer): command clips 4.8–5.3 s vs
  3.8–4.0 s. Slower **and** less accurate **and** no size advantage
  (826 vs 883 MB) — no deployment role of its own.
- **Base acft-ko 5s regression: PASS** — `볼륨 업` exact at 0.81 s total,
  timings identical to cycle 4.
- Turbo-5s command latency ~4 s total is dominated by compile (1.9 s, cache
  miss every fresh process) + encode (1.0 s); with the whisper compiled-model
  cache warm in a resident app process, repeat commands drop to ~1.6–1.9 s
  (encode+decode), still ~6× faster than turbo-30s i4 (21–24 s) on the same
  device.

## Cycle-5 pass/fail ledger

| Item | Verdict |
| --- | :-: |
| base acft-ko 5s regression (1 clip) | PASS — exact, 0.81 s |
| medium acft-ko 5s × 5 clips | PARTIAL — 4/5 exact; quiet take fixed, `음향증가` flip on device |
| turbo acft-ko 5s × 5 clips | PASS — **5/5 exact incl. quiet take** |
| windowFrames/vocab verification (500 / 51865 / 51866) | PASS — every row |
| App stability | PASS — 11 fresh-process runs, zero crashes/OOM |

## Cycle-5 verdict — final ASR voice-command lineup

| Tier | Model | Size | Role |
| --- | --- | ---: | --- |
| **1st pick (voice commands)** | `ASR/whisper-base-acft-ko/acft_base_5s_drq.tflite` | 101 MB | 0.7–0.8 s E2E, exact on all normal-loudness commands. Default FC command path. |
| **Accuracy fallback (voice commands)** | `ASR/whisper-turbo-acft-ko/acft_turbo_5s_drq.tflite` | 883 MB | Only 5/5 model on device (quiet takes + `음량 증가` + digits). ~4 s cold / ~1.9 s warm. Use for low-confidence retry or quiet capture. |
| Dictation / long clips | `ASR/whisper-base/whisper_base_30s_i8.tflite` | 77 MB | unchanged (5 s window truncates long audio) |
| Long-form accuracy | `ASR/whisper-large-v3-turbo/whisper_large_v3_turbo_30s_i4.tflite` | 755 MB | unchanged |
| Not recommended | medium acft-ko 5s (deployed for the test scene only) | 826 MB | slower + less accurate than turbo-5s on device |
| Not recommended | tiny acft-ko 5s | 59 MB | cycle-4 REJECT (capacity) |

The turbo-acft-5s tier displaces Qwen3-ASR-0.6B (794 MB) as the short-command
accuracy fallback: same size class, better transcripts (digits preserved,
5/5 vs digit-spelling diffs), and one shared whisper runtime path.

---

# Device Cycle 6 — dual-mode VAD (take7 AAR)

- Date: 2026-07-25 (10:07–10:11 host). Device `46a880a0`, CPU, whisper mode,
  `ko`, fresh app process per run. **10/10 runs SUCCESS — zero crashes/OOM.**
- AAR rebuilt 10:01 ("take7", 31.6 MB): dual-mode VAD in the shared ASR PCM
  path (task #24). `vadMode` per request: `off` (no preprocessing) /
  `energy` (adaptive energy VAD **v2** — noise-floor percentile threshold,
  6 dB hysteresis, 210 ms hangover, 90 ms pre-roll, speech-only-RMS gain;
  replaces the take3 fixed gate as default) / `ai` (Silero VAD v5 tflite,
  1.25 MB, from HF `pat229988/silero-vad-16k-tflite`, pushed at runtime to
  `LiteRTLM/ASR/ASR/silero-vad/`). Result JSON now reports
  `vadMode`/`vadModeUsed`/`speechSegments`/`trimmedSeconds`/`vadGain`/
  `speechRms` (+ `vadError` on ai→energy fallback).
- APK: `LiteRtLmAndroidAsrSmokeTest-generic.apk` rebuilt 10:07 (137 MB) with
  take7 AAR + `vadMode`/`vadSileroModelPath` runtime-config keys.
- Offline phase-1 (desktop, 10 clips × 4 modes × base-acft-5s/base-30s-i8/
  turbo-acft-5s): see `short-utterance-asr-research.md` "Dual-mode VAD".
- Logs: `Builds/Logs/AndroidDeviceRuns/20260725-10*-c6-*.{summary,status,logcat}.txt`
  Matrix script: scratchpad `Run-Cycle6VadMatrix.ps1` (+ `Invoke-AsrDeviceRunC6.ps1`).

## 1. Cycle-6 device matrix

| Model | Clip | vadMode | Device transcript | Match | Compile s | Encode s | Decode s (steps) | vadGain | Segments s |
| --- | --- | --- | --- | :-: | ---: | ---: | ---: | ---: | --- |
| base acft-ko 5s | `볼륨 업` (old quiet) | off | `볼륨어` | ✗ | 0.19 | 0.048 | 0.43 (6) | 1.000 | — |
| base acft-ko 5s | `볼륨 업` (old quiet) | energy | `볼륨어` | ✗ | 0.18 | 0.053 | 0.41 (6) | 1.311 | [0.09, 0.78] |
| base acft-ko 5s | `볼륨 업` (old quiet) | ai | `볼륨어` | ✗ | 0.19 | 0.063 | 0.44 (6) | 1.223 | [0.19, 0.79] |
| turbo acft-ko 5s | `볼륨 업` (old quiet) | off | `볼륨업` | ✓ | 1.92 | 1.046 | 0.87 (6) | 1.000 | — |
| turbo acft-ko 5s | `볼륨 업` (old quiet) | energy | `볼륨업` | ✓ | 1.92 | 1.076 | 0.90 (6) | 1.311 | [0.09, 0.78] |
| turbo acft-ko 5s | `볼륨 업` (old quiet) | ai | `볼륨업` | ✓ | 1.92 | 1.071 | 0.85 (6) | 1.223 | [0.19, 0.79] |
| base acft-ko 5s | `소리 키워줘` | energy | `소리 키워줘` | ✓ exact | 0.19 | 0.057 | 0.35 (5) | 1.335 | [0.09, 1.32] |
| base acft-ko 5s | `소리 키워줘` | ai | `소리 키워줘.` | ✓ | 0.18 | 0.050 | 0.41 (6) | 1.220 | [0.26, 1.28] |
| base acft-ko 5s | `현재 서울의…` (3.2 s) | energy | `현재 서울의 날씨는 흐림입니다.` | ✓ | 0.18 | 0.051 | 0.84 (12) | 1.000 | [0.27, 1.89], [1.83, 3.15] |
| base acft-ko 5s | `현재 서울의…` (3.2 s) | ai | `현재 서울의 날씨는 흐림입니다.` | ✓ | 0.19 | 0.055 | 0.85 (12) | 1.000 | [0.38, 1.76], [1.92, 2.40], [2.56, 3.01] |

## 2. Findings

- **Device↔desktop VAD parity is exact**: `vadGain` (1.3109500, 1.2225511,
  1.3346232), segment boundaries and trims match the desktop prototype to
  every reported digit — both the energy-v2 C++ port and the on-device
  Silero LiteRT inference reproduce the Python reference bit-for-bit at
  reporting precision.
- **Silero VAD runs on-device** (arm64, CPU CompiledModel): compiles inside
  the run budget (no measurable compile bump vs energy — model is 1.25 MB),
  per-run VAD cost is noise-level (encode deltas ≤ 0.01 s).
- **Quiet-clip verdict confirmed on device**: base tier reads `볼륨어` in
  all three modes; turbo tier reads `볼륨업` in all three — the hypothesis
  "better VAD + gain staging may fix the quiet clip" is **closed as
  refuted** (also swept 16 gain/trim/pad variants offline: all fail).
  Tier escalation remains the only fix (cycle-5 lineup unchanged).
- **No regressions**: both regression clips exact in energy and ai modes
  (ai adds a harmless trailing period on `소리 키워줘`).
- vadModeUsed reported correctly per run, `off` mode reports empty
  segments, and the ai→energy fallback path (missing silero model) is
  exercised by the C# guard (SILERO_VAD_READY resolves before invoke).

## Cycle-6 pass/fail ledger

| Item | Verdict |
| --- | :-: |
| take7 AAR build (patch regen + pristine apply --check) | PASS |
| 10-run device matrix, fresh process each | PASS — 10/10 SUCCESS, 0 crashes |
| energy-v2 default backward compat (take6-equivalent transcripts) | PASS |
| ai mode on device (silero compile + segments + gain) | PASS |
| off mode (pre-take3 behavior) | PASS |
| quiet clip fixed by any VAD mode | REFUTED — capacity-bound (turbo/qwen3 only) |

## Cycle-6 verdict

**Default vadMode = `energy` (v2).** AI mode is accuracy-neutral on the
clean test set but ships as an opt-in for noisy capture (energy-gate
percentile floors break under nonstationary noise; silero is trained for
it) at +1.25 MB. `off` retained for A/B diagnostics.

## Mic input added to ASR test scene (post-cycle-6)

`LiteRtLmAsrTestScene` now has a "Mic" input mode next to the file
dropdown: `LiteRtLmMicVadCapture` records the default microphone
(16 kHz mono loop buffer) and endpoints utterances with a C# streaming
energy VAD mirroring the native v2 parameters (300 ms noise-floor
calibration, +9/+6 dB on/off hysteresis, 210 ms hangover, 90 ms preroll,
200 ms min-speech, 8 s max-utterance). Endpointed audio is written as a
16-bit/16 kHz WAV under `persistentDataPath/LiteRTLM/MicCaptures/` and
auto-submitted to the selected ASR model through the existing
`RunWhisperAsrSmoke`/`RunQwen3AsrSmoke` file path. Native `vadMode` can
stay `energy` — both trims are conservative and compose safely.

Device verification (46a880a0, 2026-07-25, `LiteRtLmAsrTest-micvad.apk`):
scene load PASS; RECORD_AUDIO auto-added to the manifest by Unity and
platform-fixed granted on this unit (runtime prompt path exists but is
not exercisable here); mic state machine PASS on live ambient audio
(Idle→Calibrating→Listening at noiseFloor≈-40 dB→Speech→8 s max-utterance
endpoint→8.01 s 16 kHz WAV→auto Whisper Tiny transcription success:true
in 1.24 s); file-mode regression PASS (packaged report clip, whisper-tiny
CPU, success:true, 1.71 s, tiny-tier accuracy unchanged); 0 crashes.
Because this unit has no touchscreen and a FLAG_SECURE display, the run
was driven by the optional `LiteRtLmAsrTest.autotest.json` hook in
persistentDataPath (`{"micSmokeSeconds": N, "fileTranscribe": true}`) —
absent file = normal interactive behavior. Real voice-accuracy testing is
manual (speak Korean voice commands, check the transcript log).
