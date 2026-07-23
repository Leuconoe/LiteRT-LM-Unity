# Short-Utterance ASR Accuracy — Research & Ranked Recommendations

Research date: 2026-07-23. Companion to `asr-model-matrix.md` (same-day
re-validation), which established the empirical problem:

- Short Korean function-calling commands (0.8–1.5 s: `볼륨 업`, `소리 키워줘`,
  `음량 증가`) are the weakest clips in the set.
- Clip 7 (`볼륨 업`, **0.79 s, RMS 0.071** — shortest *and* quietest) is failed
  outright by whisper-tiny/base (all tiers); only whisper-large-v3-turbo i8/f32
  and qwen3-asr i8 transcribe it (`볼륨업`, spacing aside).
- qwen3 **i4** shows silent failure modes on short clips: Chinese hallucination
  (clip 7) and immediate-EOS empty output (clip 8).

Pipeline under study (custom JNI in the Unity AAR,
`Tools/UnityAar/litert-lm-unity-aar.patch`): mp3/wav → miniaudio decode →
16 kHz mono PCM → log-mel (80 mel / 30 s zero-padded window for Whisper;
128 mel / 5 s chunks for Qwen3) → TFLite encoder → greedy full-sequence
decoder (Whisper SEQ=128; Qwen3 kTextSeq=64). **No VAD, no gain/loudness
preprocessing, no prompt conditioning** (Whisper prefix is exactly
`<|startoftranscript|><|lang|><|transcribe|><|notimestamps|>`; the Qwen3 chat
template ships an **empty system prompt**).

---

## Why the quiet 0.79 s clip fails: two compounding mechanisms

**1. Gain sensitivity of the Whisper mel frontend.** Our Whisper path computes
`log10(mel)`, clamps at `max − 8.0`, then applies the *absolute* affine
`(x + 4)/4` (patch, `CreateWhisper30sInputFeatures`). A uniform input gain `g`
shifts every log-mel bin by `2·log10(g)`; the max-relative clamp is
gain-invariant, but the fixed `(x+4)/4` normalization is **not** — quiet audio
lands the whole feature map in a low-value region the model saw less of during
training. Whisper was trained on loudness-diverse but broadly normalized data;
very low-RMS input measurably degrades small models first (largest models have
enough capacity to stay robust — matching our observation that turbo survives
clip 7 while tiny/base collapse into `보일해봐`-style noise decodes.)

**2. The 30 s zero-pad on sub-second audio.** Whisper is trained on 30 s
windows; 0.8 s of speech followed by 29 s of digital-zero padding is
out-of-distribution. Research on mobile Whisper confirms the trade-off: padding
is what suppresses hallucination (unpadded short inputs hallucinate *more*),
but the model becomes sensitive to *how* the window is filled — the
[NPUsper paper](https://arxiv.org/pdf/2607.01108) measures both effects and
proposes appending a short (~0.5 s) "hush" segment after the speech as the
best-accuracy configuration for short inputs. The
[HF transformers issue #26241](https://github.com/huggingface/transformers/issues/26241)
documents the converse: `padding='longest'` (no 30 s pad) outright breaks
Whisper output. Conclusion for us: **keep the 30 s pad** (we already do), but
control what sits between speech and padding (trim noise, leave a short
silence gap) and fix loudness *before* the mel.

Qwen3-ASR's 5 s window is inherently better matched to command-length audio —
its failures on our set are decode-side (i4 quantization collapse), not
window-mismatch.

---

## Option catalog, by category

Effort codes: **XS** ≤ ~20 LOC · **S** ≤ ~80 LOC · **M** = a few hundred
LOC / new component · **L** = training or new model asset.

### 1. Audio preprocessing (JNI, immediately after `DecodeAudio`)

| # | Technique | Expected impact | Effort (our stack) | Helps |
| - | --- | --- | --- | --- |
| 1a | **Peak/RMS loudness normalization** — compute clip RMS, scale PCM to a target (≈ −20 dBFS RMS, clamp gain so peak ≤ −1 dBFS) | High on quiet clips. Clip 7 RMS 0.071 is the direct failure driver for tiny/base; the mel affine is gain-sensitive (see above). Deterministic, no quality risk on already-loud audio (gain ≈ 1) | **XS–S** (~15–25 LOC C++, one helper shared by the Whisper-80, Whisper-30s and Qwen3-128 paths) | **Both** |
| 1b | **Energy-based VAD trim + controlled re-pad** — trim leading/trailing sub-threshold frames (simple RMS gate over 20–30 ms windows, hysteresis), then re-insert a fixed ~0.2–0.4 s of silence before/after the speech | Medium. Removes low-level noise tails that greedy decode turns into garbage tokens; the retained short silence gap matches Whisper's trained expectation better than speech butted against digital zero. NPUsper's "hush" result (~0.5 s buffer best) supports keeping a small gap rather than trimming to the sample | **S** (~40–70 LOC C++; no new model). Full Silero VAD (1.2 MB, sub-ms/chunk) is better but is ONNX — pulling onnxruntime into the AAR for this is not worth it when an energy gate suffices for push-to-talk clips | **Both** |
| 1c | Noise gate / noise suppression | Low for our data (clips are clean TTS/close-mic). AOSP explicitly recommends **against** enabling NoiseSuppressor for the VOICE_RECOGNITION source — ASR frontends prefer unprocessed spectra ([AOSP pre-processing docs](https://source.android.com/docs/core/audio/implement-pre-processing)) | S, but skip | — |

What production stacks do: whisper.cpp integrated **Silero-VAD-based
segmentation** (reduces hallucination/repetition on silence — see
[whisper.cpp discussion #2286](https://github.com/ggml-org/whisper.cpp/discussions/2286));
WhisperX made VAD-first segmentation its core accuracy/speed win
([WhisperX](https://github.com/m-bain/whisperx)); Android's platform
recognizer records from `VOICE_RECOGNITION` with tuned mic gain and no NS/AEC.
The [OpenAI cookbook](https://developers.openai.com/cookbook/examples/whisper_processing_guide)
lists trimming/segmenting on pauses and avoiding <1 s fragments as standard
pre-processing.

### 2. Whisper-specific decode tricks

| # | Technique | Expected impact | Effort | Helps |
| - | --- | --- | --- | --- |
| 2a | **`initial_prompt` via `<|startofprev|>`** — prepend `<|startofprev|>` + domain-vocab tokens (e.g. `볼륨, 음량, 소리, 키워줘, 증가, 감소`) before `<|startoftranscript|>` | Medium-high for domain words. Zero-training contextual biasing is a studied technique ([arXiv 2410.18363](https://arxiv.org/abs/2410.18363), [rare-word zero-shot study](https://arxiv.org/html/2502.11572v1)); only the last ≤224 prompt tokens count and later tokens weigh more. Known risk: prompts can *induce* hallucination on tiny models (they may copy prompt text on silence) — must be A/B'd with 1a/1b in place | **S** in JNI (extend the prefix loop at the decoder — token IDs precomputed offline/C#, passed as int array; no C++ tokenizer-encode needed). Watch SEQ=128 budget and the +1 special-token shift on large-v3-family vocab (`<|startofprev|>` = 50361 for tiny/base/v2, 50362 for large-v3/turbo) | Whisper only |
| 2b | **Beam search (beam≈5) or best-of-N instead of greedy** | Medium. Beam-5 is OpenAI's reference decode; consistently lower WER than greedy, biggest deltas on ambiguous/short audio ([Min Lookahead beam study](https://arxiv.org/html/2309.10299)). Cost: ~5× decode compute in our no-KV-cache full-sequence decoder — on turbo i8 that turns 0.6 s decode into ~3 s | **M** (decode loop restructure in JNI) — defer until KV-cache decode lands | Whisper (& Qwen) |
| 2c | **Min-length / EOS suppression for first N steps** — forbid EOS before ~2–3 content tokens when audio ≥ 0.5 s | Low for shipped tiers, but it is the exact fix for the qwen3-i4 empty-output mode and a cheap guard everywhere | **XS** (~5 LOC in each decode loop) | Both |
| 2d | **Repeat/tile short audio ×N before the mel** | Unproven. Community guidance is about *concatenating different consecutive segments* for context, not tiling one clip ([whisper discussion #1913](https://github.com/openai/whisper/discussions/1913), cookbook); no credible benchmark shows tiling the same sub-second clip helps, and it doubles/triples decoded text that must then be deduped. Only worth a 30-minute offline A/B via the existing `bench_asr.py` before dismissal | XS to test offline; don't ship untested | Whisper |
| 2e | Encoder-output cropping (feed decoder only the valid frames) | Speed win, not accuracy (NPUsper shows accuracy holds only with careful buffering); touching encoder shapes means re-exporting TFLite graphs | L, skip for accuracy purposes | — |
| 2f | Suppress-tokens tweaks (numerals, punctuation) | Marginal for this failure mode (failures are phonetic, not formatting) | S, skip | — |

Language+task forcing (`<|ko|>` + `<|transcribe|>` + `<|notimestamps|>`) is
already implemented — confirmed in the patch decode prefix.

### 3. Qwen3-ASR-specific

| # | Technique | Expected impact | Effort | Helps |
| - | --- | --- | --- | --- |
| 3a | **Context biasing via the system prompt** — our JNI hardcodes `<|im_start|>system\n<|im_end|>` (empty). Qwen3-ASR was explicitly trained with **context-SFT** to use system-prompt text as biasing background ([Qwen3-ASR tech report](https://arxiv.org/html/2601.21337v1)); third-party runtimes expose it as `--prompt` hotword injection ([antirez/qwen-asr](https://github.com/antirez/qwen-asr)); practitioners report it fixes homophone confusions by supplying a glossary ([field report](https://note.com/veltrea/n/n7dd0b7ffffe9)). Biasing is soft — tilts probabilities, doesn't force output | Medium-high for a fixed command vocabulary; this is the *intended* customization mechanism of the model | **S–M** in JNI: insert precomputed hotword token IDs between the `system\n` and `<|im_end|>` template tokens. **Constraint: kTextSeq=64 total** (15 template tokens + prompt + generated output share the budget, and kJointSeq=134 is baked into the TFLite export) → hotword list must stay ≈ ≤ 25 tokens. Enough for ~6–10 short Korean command words; commands themselves decode in ~5 tokens | Qwen3 only |
| 3b | Ship i8, never i4 (already the matrix recommendation); add 2c EOS-guard as belt-and-braces | Removes the two silent i4 failure modes | XS/policy | Qwen3 |

### 4. Model-level

| # | Technique | Expected impact | Effort | Helps |
| - | --- | --- | --- | --- |
| 4a | **KWS hybrid** — tiny keyword-spotting model for the fixed command set, full ASR as fallback | High robustness for *fixed* commands: speech-commands-style models hit ~96 % on 35 keywords at <1 MB ([BC-ResNet](https://github.com/re9ulus/BC-ResNet), [TC-ResNet](https://github.com/hyperconnect/TC-ResNet), [google-research kws_streaming](https://github.com/google-research/google-research/blob/master/kws_streaming/README.md) — all TFLite-convertible, so **no new runtime**). Korean commands require recording/collecting a few hundred samples per keyword + augmentation and training from scratch or fine-tuning. Alternative without training: [sherpa-onnx open-vocabulary KWS](https://k2-fsa.github.io/sherpa/onnx/kws/index.html) accepts arbitrary keywords at runtime, but is ONNX (new runtime dep) and its zipformer models are zh/en-focused | **L** (data collection + training + new inference path + arbitration logic) | New path |
| 4b | **LoRA fine-tune whisper-tiny/base on short Korean commands** | High for the small tiers: LoRA updates ~1.6 % of weights; a comparable low-resource case (Cantonese, edge deployment) cut CER 49.5 → 11.1 ([LoRA-INT8 Whisper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12431075/)); recipe is mature ([HF fine-tune blog](https://huggingface.co/blog/fine-tune-whisper), [fast-whisper-finetuning](https://github.com/Vaibhavs10/fast-whisper-finetuning)). Needs a command+general-Korean mix to avoid catastrophic forgetting, then merge weights → re-run our existing TFLite export → re-run int8/int4 recipes and the Korean-clip validation gate | **L** (≈ 2–4 days incl. data prep; GPU-hours small for tiny/base) | Whisper tiny/base |
| 4c | Distil-Whisper as a smaller accurate tier | Not available: official distil checkpoints are **English-only** ([distil-large-v3 discussion](https://huggingface.co/distil-whisper/distil-large-v3/discussions/1)); a Korean distillation is a bigger project than 4b | XL, skip | — |
| 4d | Alternative small models (Moonshine etc.) | [Moonshine](https://arxiv.org/pdf/2410.15608) targets exactly this use case (variable-length windows, voice commands, no 30 s pad) but is English-only today | Track, skip | — |

### 5. App-level (C# runners / future mic capture)

No live microphone path exists yet (all runners read StreamingAssets files),
so these apply to the upcoming mic-capture feature:

| # | Technique | Expected impact | Effort | Helps |
| - | --- | --- | --- | --- |
| 5a | **Min-duration enforcement + capture tail** — require ≥ ~1.0 s of captured audio; keep recording ~0.3–0.5 s after release/energy-drop so final consonants aren't clipped | Medium; cookbook explicitly warns against <1 s fragments; clip 7 at 0.79 s sits below that line | **XS–S** C# | Both |
| 5b | **Record from `VOICE_RECOGNITION` source, enable platform AGC, leave NS/AEC off** — Android's `AutomaticGainControl` audiofx normalizes capture level in hardware/driver ([Android AGC docs](https://stuff.mit.edu/afs/sipb/project/android/docs/reference/android/media/audiofx/AutomaticGainControl.html)); AOSP guidance: no NS for VOICE_RECOGNITION | Medium (prevents quiet captures at the source; complements 1a which then becomes a safety net) | **S** (AudioRecord config in AAR or Unity mic settings) | Both |
| 5c | **Confidence thresholding + retry UX** — average token logprob (or EOS-step logprob) from the greedy loop; below threshold → "다시 말씀해 주세요" retry instead of executing a wrong command | Medium-high *system* accuracy: converts silent mis-executions into retries; standard practice (Whisper's own `avg_logprob`/`no_speech_prob` fallback machinery) | **S** (JNI already has per-step logits; export avg logprob in the result JSON ~10 LOC; C# thresholding ~20 LOC) | Both |
| 5d | Space-insensitive + alias keyword matching in the FC layer (`볼륨업`; optionally `볼륨어` etc.) | Immediate: turbo i8 and qwen3 i8 already produce character-perfect `볼륨업` — only the matcher rejects it | **XS** C# | Both |

---

## Ranked recommendations (quick wins first)

| Rank | Item | Effort | Impact | Models |
| - | --- | :-: | :-: | --- |
| 1 | **1a RMS loudness normalization in JNI** (target ≈ −20 dBFS RMS, peak-clamped) | XS | High on quiet clips (root cause of clip-7 tiny/base failure) | both |
| 2 | **5d space-insensitive/alias FC matching** | XS | Converts 2 existing near-misses into passes today | both |
| 3 | **2c min-length EOS guard** (no EOS before 2–3 content tokens) | XS | Kills qwen empty-output mode; harmless elsewhere | both |
| 4 | **1b energy-gate VAD trim + 0.2–0.4 s re-pad** | S | Medium; cleans noise tails, normalizes silence context | both |
| 5 | **3a Qwen3 hotword system prompt** (≤ ~25 tokens, precomputed IDs) | S–M | Medium-high; the model's designed customization channel | qwen3 |
| 6 | **2a Whisper `<|startofprev|>` domain-vocab prompt** | S | Medium; A/B for prompt-copy hallucination on tiny/base | whisper |
| 7 | **5c confidence threshold + retry** (avg logprob export) | S | Medium-high at system level | both |
| 8 | **5a/5b recording UX** (min duration, tail capture, VOICE_RECOGNITION + AGC) | S | Medium; lands with the mic feature | both |
| 9 | **2b beam/best-of decode** | M | Medium; wait for KV-cache decoder | both |
| 10 | **4b LoRA fine-tune tiny/base on Korean commands** | L | High for small tiers, only if tiny/base-class hardware is a hard requirement | whisper |
| 11 | **4a KWS hybrid** (TFLite BC/TC-ResNet, Korean data collection) | L | High for fixed commands; biggest architectural change | new path |
| — | 2d audio tiling | XS test | Unproven — 30-min offline A/B in `bench_asr.py` only | — |
| — | 1c noise suppression, 2e encoder crop, 2f suppress-tokens, 4c/4d | — | Skip (counter-indicated / unavailable / wrong problem) | — |

### Top-3 to implement now

1. **RMS loudness normalization (1a)** — one ~20-line helper in the JNI audio
   path shared by all three feature builders. Directly attacks the measured
   failure (clip 7 RMS 0.071). Re-run the matrix afterwards; expect tiny/base
   to move from garbage to at least near-miss on clip 7, and no regression on
   loud clips.
2. **Energy-gate trim + controlled re-pad (1b), bundled with the min-length
   EOS guard (2c) and space-insensitive FC matching (5d)** — all S/XS, all in
   files already being touched; together they remove the remaining
   mechanical failure modes (noise-tail decodes, qwen empty output, matcher
   rejecting `볼륨업`).
3. **Qwen3 hotword system prompt (3a)** — the shipped FC tier is qwen3 i8 /
   turbo i8; qwen3's context biasing is trained-in and fits our fixed command
   vocabulary inside the 64-token text budget. Prototype offline first in
   `bench_asr.py` (inject token IDs into the prompt prefix) to validate gain
   before touching the AAR; if it proves out, do the Whisper
   `<|startofprev|>` twin (2a) for turbo i8 next.

Defer: fine-tuning (4b) and KWS hybrid (4a) until the preprocessing tier is
measured — if 1a+1b lift tiny/base to acceptable command accuracy, the
training-scale options may be unnecessary; if the product truly must run FC
voice on 40–80 MB models, 4b is the highest-leverage L-size option.

---

## Implemented (2026-07-23)

Offline prototype (`bench_asr_v2.py`, same numpy log-mel + greedy harness as
the model matrix) A/B'd each candidate on the 4 volume clips + 2 long Korean
clips before any native change. Configs: RMS normalization to −20 dBFS
(peak-clamped at −1 dBFS), energy-gate VAD trim (30 ms windows, gate at
−20 dB below the loudest window, keep 0.1 s lead-in / 0.3 s tail), qwen3
EOS min-length guard (suppress EOS for the first 3 generated tokens).

**Directive change during implementation:** the Qwen3 hotword system-prompt
(3a) was prototyped (22 prompt tokens, fit the budget, no regressions) but
**dropped by user directive** — the ASR must not be pre-biased toward
specific expected values. No hotword code ships anywhere (native or C#).

### Phase-1 before/after (baseline → VAD trim + boost-only RMS norm [+ qwen EOS guard])

CER after punctuation/space stripping; `OK` = exact match after stripping.

| Clip | tiny i8 | base i8 | turbo i8 | qwen3 i8 | qwen3 i4* |
| --- | --- | --- | --- | --- | --- |
| 볼륨 업 (0.79 s, quiet) | 보일해봐 1.33 → 1.33 | 볼륨어 0.33 → 0.33 | OK → OK | OK → OK | 宝鱼帽 1.00 → 1.00 |
| 볼륨, 업 (re-recorded) | OK → OK | OK → OK | 볼륨 억 0.33 → **OK** | OK → OK | Volume up 2.67 → 2.67 |
| 소리 키워줘 | OK → OK | OK → OK | OK → OK | OK → OK | 소리 퓭쳐줘 0.40 → **OK** |
| 음량 증가 | 능량 0.25 → 0.25 | OK → OK | OK → OK | OK → OK | 音量增加 1.00 → **OK** |
| long 전술평가 (3.98 s) | 0.12 → 0.18 | OK → OK | OK → OK | 0.41 → 0.41† | 0.06 → 0.47† |
| long 변경사항 (7.03 s) | 0.03 → 0.09 | OK → OK | OK → OK | OK → OK | OK → OK |

\* qwen3 i4 is not a shipped tier (int8-minimum policy); scratchpad copy.
† spoken-form numbers (`이천이십오년` vs `2025년`) — CER artifact, not a
semantic regression; qwen3 i8 baseline has the same reading.

Net: **turbo i8 goes 6/6 exact**, qwen3 i4 gains 2 clips, shipped tiers
(base/turbo/qwen i8) show zero regressions. tiny's long-clip jitter
(0.12→0.18 / 0.03→0.09) is decode noise on already-wrong transcripts; tiny
is not a shipped FC tier.

### Per-fix findings

1. **RMS loudness normalization (1a) — shipped boost-only.** Full
   normalization to −20 dBFS was rejected: on the quiet clip the required
   gain is only 1.4× (−23 dBFS → −20 dBFS) and a sweep of 1.33×/2×/4×/8×
   gains **never** fixes tiny/base (`보일해봐`/`볼륨어` unchanged up to 2×,
   degrading beyond) — the clip-7 failure is phonetic, not loudness; and the
   attenuation branch (gain < 1 on loud clips) regressed tiny on both long
   clips (0.03→0.11). The shipped form never attenuates (gain ≥ 1 only,
   peak-clamped), measured strictly no regressions, and remains a safety net
   for genuinely quiet future mic captures.
2. **Energy-gate VAD trim + 0.3 s tail (1b) — shipped.** The one clear
   accuracy win: fixes turbo i8 `볼륨 억` → `볼륨 업` on the re-recorded
   clip and contributes to both qwen3-i4 short-clip fixes; no shipped-tier
   regressions.
3. **Qwen3 EOS min-length guard (2c) — shipped.** The immediate-EOS empty
   output reproduces offline on two i4 quant variants
   (`i4b32_encembI8`, `wi4b32`: steps=1, empty transcript on 소리 키워줘);
   suppressing EOS for the first 3 generated tokens recovers the **correct**
   transcript (`소리 키워줘.`, 11 steps) on both. On i8 the guard changes
   nothing (validated identical output on all 6 clips).
4. **Qwen3 hotwords (3a) — dropped by directive** (see above).
5. **Space/comma-insensitive expected matching (5d) — display-only** in the
   C# runners; the comparison never touches model input.

### Native changes (AAR patch)

`External/LiteRT-LM/kotlin/java/com/google/ai/edge/litertlm/jni/litertlm.cc`:

- `TrimPcmWithEnergyGate` + `NormalizePcmLoudness` (boost-only) +
  `PreprocessAsrPcm`, applied after `DecodeAudio` in
  `CreateWhisperInputFeatures` and in the Qwen3 smoke path (Parakeet
  untouched — its per-feature mean/std normalization is gain-invariant).
- Qwen3 decode loop: `kMinDecodeStepsBeforeEos = 3` EOS suppression.
- Note: the native Whisper path currently supports the 80-mel family
  (tiny/base); the turbo-i8 VAD win transfers when a 128-mel native path
  lands.

C# (display-only): `LiteRtLmAsrTestRunner.cs` gains per-clip expected
transcripts and a space/punct-insensitive `expectedMatch` log line;
`LiteRtLmAsrSmokeTestRunner.cs` gains an optional `expectedTranscript`
field (+ config override) emitting an `EXPECTED_MATCH` status line.

Patch regenerated as `git diff v0.14.0` (17 files, ~3.9k insertions),
`git apply --check` verified against `External/LiteRT-LM-v0.14-pristine`;
AAR rebuilt via `Tools/Windows/Build-LiteRtLmUnityAarFromPatch.ps1`
(log: `Builds/Logs/aar-build-v0.14-take3.log`).

---

## Sources

- [NPUsper: Eliminating Redundant Computation for Real-Time Whisper on Mobile NPUs](https://arxiv.org/pdf/2607.01108) — 30 s pad vs hallucination trade-off, hush-word buffering
- [HF transformers #26241 — padding='longest' breaks Whisper](https://github.com/huggingface/transformers/issues/26241)
- [OpenAI cookbook — Whisper pre/post-processing](https://developers.openai.com/cookbook/examples/whisper_processing_guide)
- [whisper.cpp discussion #2286 — VAD to curb hallucination/repetition](https://github.com/ggml-org/whisper.cpp/discussions/2286)
- [WhisperX (VAD-first segmentation)](https://github.com/m-bain/whisperx)
- [openai/whisper discussion #1913 — short segments vs long audio](https://github.com/openai/whisper/discussions/1913)
- [Contextual Biasing without Fine-Tuning of Whisper (arXiv 2410.18363)](https://arxiv.org/abs/2410.18363)
- [Improving Rare-Word Recognition of Whisper in Zero-Shot Settings](https://arxiv.org/html/2502.11572v1)
- [HF transformers PR #22496 — `<|startofprev|>` prompt implementation](https://github.com/huggingface/transformers/pull/22496)
- [Min Lookahead beam search for Whisper (arXiv 2309.10299)](https://arxiv.org/html/2309.10299)
- [Qwen3-ASR Technical Report — context-SFT / biasing](https://arxiv.org/html/2601.21337v1)
- [antirez/qwen-asr — `--prompt` hotword biasing in a C runtime](https://github.com/antirez/qwen-asr)
- [Qwen3-ASR context biasing field report (glossary → homophone fix)](https://note.com/veltrea/n/n7dd0b7ffffe9)
- [Silero VAD (1.2 MB, sub-ms on Android)](https://github.com/snakers4/silero-vad/wiki/FAQ)
- [AOSP audio pre-processing — no NS for VOICE_RECOGNITION](https://source.android.com/docs/core/audio/implement-pre-processing)
- [Android AutomaticGainControl audiofx](https://stuff.mit.edu/afs/sipb/project/android/docs/reference/android/media/audiofx/AutomaticGainControl.html)
- [BC-ResNet KWS](https://github.com/re9ulus/BC-ResNet) · [TC-ResNet KWS](https://github.com/hyperconnect/TC-ResNet) · [google-research kws_streaming](https://github.com/google-research/google-research/blob/master/kws_streaming/README.md)
- [sherpa-onnx open-vocabulary keyword spotting](https://k2-fsa.github.io/sherpa/onnx/kws/index.html)
- [LoRA-INT8 Whisper for edge Cantonese (CER 49.5→11.1)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12431075/)
- [HF blog — Fine-Tune Whisper](https://huggingface.co/blog/fine-tune-whisper) · [fast-whisper-finetuning (LoRA)](https://github.com/Vaibhavs10/fast-whisper-finetuning)
- [distil-whisper — English-only checkpoints](https://huggingface.co/distil-whisper/distil-large-v3/discussions/1)
- [Moonshine — variable-length ASR for voice commands (arXiv 2410.15608)](https://arxiv.org/pdf/2410.15608)
