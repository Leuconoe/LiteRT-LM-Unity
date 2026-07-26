# ASR training & deployment program handoff (2026-07-23 – 07-26)

This single document should be enough to pick up all ASR-related work.
**Every judgement here is against Android on-device execution** (Snapdragon 865
class, device `46a880a0`); desktop numbers are reference only. For the
framework (v0.14) upgrade itself see `v0.14-upgrade-handoff.md`.

## 1. Final state in one line

**Only the "clean ACFT" lineage was deployed and published** — four models
self-distilled from stock openai whisper on zeroth-korean (70 %) + FLEURS en
(30 %). The KsponSpeech lineage was **abandoned on user instruction**
(2026-07-26) and its artifacts were deleted; §4 is the only surviving record.

## 2. Deployed ASR lineup (StreamingAssets)

| Use case | File | Size | Rationale |
| --- | --- | ---: | --- |
| Voice commands, 1st pick | `ASR/whisper-base-acft-ko/acft_base_5s_drq.tflite` | 101 MB | Every normal-loudness command exact on device, 0.7–0.8 s E2E |
| Command accuracy fallback | `ASR/whisper-turbo-acft-ko/acft_turbo_5s_drq.tflite` | 883 MB | Device 5/5 including the quiet legacy take — the only model that manages it |
| Long-form (>30 s) | `ASR/qwen3-asr-0.6b/qwen3_asr_0.6b_5s_i8.tflite` | 794 MB | 5 s chunk loop; 98 s clip completed on device (4.2 min, flat RAM) |
| Sentence balance | `ASR/whisper-base/whisper_base_30s_i8.tflite` | 77 MB | Korean sentence CER 0.000 |
| Other | tiny/medium/large-v3/turbo 30 s tiers (i8/i4) | — | Comparison and reference |
| (Note) medium-acft | `ASR/whisper-medium-acft-ko/` | 826 MB | Deployed but **not recommended** — slower and less accurate than turbo |

VAD: every ASR path supports `vadMode` off / energy (default) / ai
(Silero, 1.25 MB).

## 3. Clean ACFT training recipe (this is what worked)

- Starting point: **stock openai/whisper-{tiny,base,medium,large-v3-turbo}**.
  Do not start from a Korean fine-tune such as komixv2 — their English is
  destroyed.
- Method: futo-org/whisper-acft self-distillation (MSE on decoder hidden states
  against a frozen full-window teacher), Adam lr 1e-6, batch 1, up to 8 epochs
  with early stopping
- **Two corrections that mattered**: (1) an `n_ctx` floor of 250, matching the
  5 s fixed deployment window — the root cause of the earlier
  komixv2-acft-ggml failure was training out-of-distribution down to ctx 64;
  (2) a 70:30 Korean:English mix (zeroth 51.6 h + FLEURS en_us) — single-language
  training destroys the other language
- Short utterances: zeroth clips <3 s oversampled ×3, plus 0.5–3 s crop
  augmentation (p = 0.15)
- Gate results (**40-clip TTS-synthesized holdout**, Korean short-utterance CER
  at 5 s ctx): turbo 0.182 · medium 0.208 · base 0.305 · tiny 0.457
  (stock collapses at 1.07–24.9)
  - ⚠️ This gate is **valid for ranking only** (Spearman ρ = 1.00 against real
    recordings); it is not calibrated as an absolute quality number. References
    average 3.6 characters, so one period is worth CER +0.25–0.50 while the
    actual matcher ignores punctuation. **tiny is not recommended for Korean
    voice commands** — real-recording command CER 0.896, device 1/4 exact.
    Evidence: §5 below and `docs/benchmarks/asr-model-matrix.md` Addendum 3
- Scripts: `External/acft-training/train_acft.py`, `run_queue.py`
  (**kept private** — they are a port of the futo notebook, so publishing would
  trigger MIT attribution obligations)

## 4. KsponSpeech program — abandoned (full negative result)

Cancelled by the user on 2026-07-26. **All artifacts (runs, exports, datasets,
bases, scripts, venv — about 87 GB) were deleted on 2026-07-26**, so this
document is the only record. No kspon-lineage model was ever deployed or
published.

- Shape: komixv2 (Korean fine-tune) checkpoints + KsponSpeech 100 h
  (spontaneous conversation, 66.5 % in the 1–3 s range) CE fine-tune (C) →
  ACFT (D) → TTS continuation (E). Cost: **≈31.6 GPU-hours**
- Executed: the tiny and base chains completed; turbo was cancelled while D was
  being restarted from the original base after C damaged it. small never
  started; medium completed C only.

### 4.1 C-stage measurements (ko_short = 40-clip TTS short-utterance holdout, ctx 250)

| Base | ko_short | kspon | fleurs_ko | fleurs_en |
| --- | --- | --- | --- | --- |
| komixv2-tiny → C | 0.417 → **0.609** | 0.177 → 0.124 | 0.100 → 0.137 | 0.992 → 0.386 |
| komixv2-base → C | 0.331 → **0.473** | 0.104 → 0.113 | 0.094 → 0.103 | 1.000 → 0.246 |
| komixv2-medium → C | 0.306 → **0.365** | 0.091 → 0.080 | 0.069 → 0.084 | 0.854 → 0.142 |
| turbo-komix-lora → C | 0.200 → **0.575** | 0.134 → **0.141** | 0.071 → 0.142 | 0.038 → 0.055 |

D and E did their job (short-window repetition collapse removed, base ko_short
6.46 → 0.48) but they were distilling an already-damaged C, so the end product
never beat its own pre-C starting point.

### 4.2 Conclusion — it is **register**, not length

Established by a training-free audit (inference-only re-evaluation plus label
statistics):

- **"kspon does not help short commands" is correct, and the damage is larger
  on real recordings**: the C regression is +0.173/+0.104 (tiny/base) on the
  TTS gate but **+0.292/+0.254** on real recordings. The gate *understated* the
  harm by 1.7–2.4×.
- **But the reason is not "conversation is too long."** On kspon's own eval
  split, **real spontaneous speech under 2 s** actually improves after C
  (tiny 0.162 → 0.125, base flat). C did learn short audio; what it broke was
  **output register**.
- The label statistics warned about this before any GPU time:

  | Corpus | Mean chars | Ends with terminal punctuation | Imperative ending (-줘/-주세요) |
  | --- | ---: | ---: | ---: |
  | KsponSpeech train | 14.1 | **78.3 %** | 0.1 % |
  | zeroth (prior Korean bulk) | 41.6 | 0.0 % | 0.1 % |
  | TTS command set (target) | **3.7** | 0.0 % | **19.8 %** |

  After C the model appends a period to almost everything (on a 2-character
  reference that is CER +0.50), `-줘` collapses into the colloquial `-져`
  (`소리 키워줘` → `술이 키워져`), and domain vocabulary is lost
  (`호버링` → `후바링`).
- **33–79 % of the regression is punctuation alone**, and the shipped matcher
  ignores punctuation (`LiteRtLmAsrTestRunner.cs`) — we optimized against a
  penalty the product does not have.
- **Using duration as a proxy for utterance function was the design error.**
  kspon's short clips are back-channels (`맞아.` `어.` `진짜?`), not commands.

### 4.3 Recipe defects independent of the data choice (five)

1. **No LR decay** — flat 1e-5 for 16.8 k steps after warmup, reinforcing the
   style drift all the way to the end
2. **Zero replay of the prior distribution** — kspon plus 15 % English only.
   The textbook catastrophic-forgetting setup; C-turbo even regressed on its
   own training domain
3. **Zero command data in the stage that had to preserve commands** (no
   `--extra-short-data` path)
4. **Full fine-tune** (no LoRA, no freezing) — nothing constrained the decoder
   LM prior
5. **Selection metric ≠ ship metric** — `composite` weighted ko_short at 1/3,
   so the English recovery C bought cheaply outvoted the command regression.
   Evaluation only at epoch boundaries (2 usable points over 16.8 k steps) meant
   the over-training hypothesis could not even be tested

## 5. If Korean command fine-tuning is ever attempted again

Rules from the kspon audit. Read this section before starting any retraining.

**Data**

1. Match **label register** to the target output, not just audio length.
   Measure terminal-punctuation rate, mean length and sentence-ending
   morphology before adopting a corpus, and gate on it at prep time (kspon at
   78 % punctuated / 0.1 % imperative vs the command set at 0 % / 19.8 % was
   visible before any GPU time)
2. If a mismatched corpus is used anyway, **normalize its labels to the target
   style** (strip terminal punctuation, or train with a style tag). That alone
   removes 33–79 % of the observed regression
3. **Never use duration as a proxy for utterance function.** Bucket by function
   (isolated command / read sentence / conversational turn) first, then by length
4. The real blocker is **real recorded short command audio**. Open data has
   none — AIHub in-vehicle commands (dataSetSn=112, registration required) or
   an in-house recording session on the target mic are the only honest fixes

**Replay and optimization**

5. **Always mix in replay of the prior distribution** — at least 20–30 % of steps
6. To repair one capability, use the **smallest sufficient intervention**
   (100 h of Korean conversation to fix an English collapse was massive
   overkill; a short English mix-in would have done it)
7. **Decay the LR to ~0** (linear or cosine after warmup). Flat LR over a long
   run is the most likely driver of style drift
8. Prefer **LoRA or a frozen decoder** for register-sensitive tasks. The one
   checkpoint whose decoder prior stayed intact (turbo-komix-lora) had the best
   pre-C score in the whole matrix
9. Run a **1–2 hour tiny probe** (eval every 500 steps) before committing to a
   multi-day queue

**Evaluation and selection**

10. **Make the selection metric the ship metric.** Weight ko_short ≥ 0.5, or
    hard-reject any checkpoint worse than step 0. That single rule would have
    killed C at the first eval for all four sizes
11. **Score the way the product matches** (strip terminal punctuation, compare
    space-insensitively). Report both raw and normalized CER so drift stays
    visible without dominating
12. **Evaluate every 500–1000 steps**, not per epoch, and evaluate the
    checkpoints you actually save
13. **Use the whole holdout** (do not shrink `--eval-n`) and grow the command
    holdout well past 40 clips — CER is extremely high-variance on 3.6-character
    references
14. **Keep the TTS-40 bucket as the ranking gate** (validated at ρ = 1.00), but
    **never quote it as an absolute number** (it overstates error ~2.8× for a
    strong base). Add a real-recording bucket as the absolute-level check
15. **Never overwrite an eval split without re-reading the decision note that
    created it**

## 6. Published models (HuggingFace)

| Repo | Contents |
| --- | --- |
| [litert-community/whisper-acft](https://huggingface.co/litert-community/whisper-acft) | The six original futo ACFT models (tiny/base/small ±.en) × 5s/10s/30s drq, one consolidated card |
| [leuconoe/whisper-acft-ko](https://huggingface.co/leuconoe/whisper-acft-ko) | Four Korean clean-ACFT models × 3 windows (12 files) |
| [leuconoe/litert-lm-unity-quantized](https://huggingface.co/leuconoe/litert-lm-unity-quantized) | This project's quantized whisper / Qwen2.5 collection |
| litert-community/whisper-{tiny,base,medium,large-v3,large-v3-turbo} | Project i8/i4 contributions (PRs and direct commits) |

## 7. Useful facts and traps when resuming

- The JNI **auto-detects mel bins, vocab size and window frames** from the
  signature (take5/6): 80/128-mel, 51865/51866, and 100–3000 frames all go
  through one code path. The result JSON reports
  `melBins/vocabSize/windowFrames/featureMd5/vadMode`
- whisper decode inputs are **bound by shape** (the order differs per model) —
  do not revert to positional binding
- AAR builds: `-SkipImageBuild` builds the **stale sources baked into the Docker
  image**. After changing the patch you must rebuild the image (found at take8)
- Quantization: i8 = `dynamic_wi8_afp32`, i4 = `dynamic_wi4b64_afp32` (+ i8 on
  sensitive scopes). `wi4c` and `wi4b32` collapse quality; int2/Q5/1.58b have no
  LiteRT kernels
- The quiet 0.79 s clip **cannot be fixed by VAD or gain** (16 combinations
  tested) — it is a model-capacity limit. The fix is tier escalation (turbo-acft)
- Device `46a880a0` has **no touchscreen and sets FLAG_SECURE** — adb taps and
  screenshots do not work. Validation runs through the
  `LiteRtLmAsrTest.autotest.json` hook (speaker echo injection can drive
  continuous-ASR cycles)
- On resume, the queue scripts read completion markers and skip or adopt
  automatically — stopping and re-running is safe

## 8. Remaining work

1. **#13 Linux/macOS binaries** — not started. No impact on Android deployment;
   it is a desktop verification aid. The upstream macos_arm64 release lacks the
   custom FC flags, so each OS needs a build from the patched tree
2. Optional, recorded as candidates only: VAD-segment-based 30 s sliding-window
   chunking for whisper (expected 8–10× on device long-form), and capture-side
   AGC for quiet input
