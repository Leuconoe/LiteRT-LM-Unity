# Moonshine Korean ASR — screened and rejected (2026-09-21)

Question: Moonshine now ships Korean models (`moonshine-tiny-ko`, `moonshine-base-ko`,
2025-09) and an official LiteRT package (`litert-community/moonshine-tiny`, English).
Its no-30-s-padding design targets exactly our short-command problem, which is why
[short-utterance-asr-research.md](short-utterance-asr-research.md) §4d kept it on
watch. Is it worth a conversion + JNI spike?

**Answer: no.** Two independent reasons, either one sufficient.

1. **Licence (gate 2).** The Korean models are the "legacy non-streaming, non-English"
   set that Moonshine's own [LICENSE](https://github.com/moonshine-ai/moonshine/blob/main/LICENSE)
   enumerates as staying under the **Moonshine AI Community License** — non-commercial,
   with commercial use only below USD 1 M annual revenue. Everything MIT (all streaming
   models, English tiny/base) is English-only. Same class of verdict as Kanana.
2. **Accuracy on the target regime.** On sentence-length real speech Moonshine-ko beats
   same-size Whisper by 2–4×. On sub-1.5 s Korean commands — the product's regime — it
   loses to whisper-base, and **base-ko is broken**: it emits the literal string
   `audiotext` for 37 of 40 command clips regardless of padding or level.

## Harness

`Tools/Research/Moonshine/Run-MoonshineEval.ps1` (transformers, RTX 4090, fp32, greedy).
Whisper runs through the same script as the same-harness baseline; it always pads to
30 s. Moonshine is fed raw 16 kHz audio; the `-PadTo 5` variant zero-pads to the 5 s
window the litert-community package uses. CER is the training queue's definition
(spaces removed, lowercase, Levenshtein); *np* strips punctuation as the shipped
matcher does. Latency is a desktop GPU number and says nothing about kona.

| Bucket | n | What it is | Use |
| --- | ---: | --- | --- |
| ko_short | 40 | TTS-synthesized Korean commands, 0.46–1.43 s (mean 0.83 s) | **ranking only** (ρ = 1.00 vs real; ~2.8× pessimistic in absolute terms) |
| real_cmd | 7 | Korean clips in `TestAssets/Audio` (4 device-recorded `볼륨` commands + 3 sentences) | real-audio sanity check |
| fleurs_ko | 64 | FLEURS ko validation (first 64 rows of `eval_fleurs_ko`) | absolute level, real read speech |

## Results — CER (np = punctuation stripped), exact-match count

| Model | Params | ko_short | real_cmd | fleurs_ko |
| --- | ---: | --- | --- | --- |
| moonshine-tiny-ko, raw | 27 M | 0.643 / np 0.543, 11/40 | 0.417 / 0.354, 3/7 | **0.113 / 0.082, 23/64** |
| moonshine-tiny-ko, pad 5 s | | 0.466 / 0.327, 13/40 | 0.494 / 0.335, 4/7 | — |
| moonshine-tiny-ko, pad 5 s + 0.5 s head + peak-norm | | 0.462 / 0.315, 15/40 | 0.301 / 0.202, 4/7 | — |
| moonshine-base-ko, raw | 61 M | 2.143 / 2.110, 4/40 | 1.398 / 1.494, 2/7 | **0.117 / 0.078, 29/64** |
| moonshine-base-ko, pad 5 s | | 2.677 / 2.669, 2/40 | 1.530 / 1.637, 1/7 | — |
| moonshine-base-ko, pad 5 s + head + norm | | 1.937 / 1.833, 7/40 | 1.170 / 1.237, 3/7 | — |
| whisper-tiny (stock, 30 s) | 38 M | 0.407 / 0.396, 9/40 | 0.373 / 0.363, 3/7 | 0.213 / 0.199, 1/64 |
| whisper-base (stock, 30 s) | 73 M | **0.300 / 0.283, 15/40** | **0.218 / 0.143, 6/7** | 0.181 / 0.157, 6/64 |
| whisper-large-v3-turbo (stock) | 809 M | 0.179 / 0.132, 25/40 | 0.119 / 0.048, 6/7 | 0.073 / 0.048, 37/64 |

Cross-check: stock whisper-base scores 0.300 on ko_short here; the ACFT gate recorded
0.305 for acft-base at 5 s context and 0.182 for turbo (0.179 here). The two harnesses
agree, so the Moonshine rows are comparable to every number in
[asr-model-matrix.md](asr-model-matrix.md).

### Digits

Moonshine writes numbers out in Hangul (`이천 이십 오 년 삼 월 오 일`); Whisper writes
digits, as the references do. Excluding refs that contain digits (14 of 64 in fleurs_ko,
1 of 7 in real_cmd) does not change the ranking:

| Model | fleurs_ko np CER, no-digit refs (n = 50) |
| --- | ---: |
| moonshine-tiny-ko | 0.050 |
| moonshine-base-ko | 0.040 |
| whisper-tiny | 0.174 |
| whisper-base | 0.100 |
| whisper-turbo | 0.031 |

## What the transcripts show

- **base-ko on commands.** With 5 s padding, 37/40 ko_short hypotheses are exactly
  `audiotext` and one more is `audi audi audiotext`; only `아니요.` and `소리다운`
  survive. `audiotext` is not a special token — it tokenizes as `▁audi ote xt` — so this
  is a training-data label leaking out whenever the encoder sees too little speech.
  It persists with 5 s zero-padding, 0.5 s head silence and peak normalization; the
  3.98 s report clip and the 4–10 s sentences transcribe perfectly. Real device
  `볼륨 업` clips: `audiotext` ×4 raw, still 3/4 after preprocessing.
- **tiny-ko on commands.** No collapse, but consonant/vowel confusions on nearly every
  clip (`음량 켜줘 → 넘냥 켜져`, `이륙 → 일이요`, `호버링 → 호발인`). Padding helps
  (np 0.543 → 0.315) and it then edges whisper-tiny (0.396) but not whisper-base
  (0.283) — at 5–6× the decode latency of whisper-tiny in this harness.
- **On sentences** both Moonshine tiers are clearly better than whisper tiny/base and
  tiny-ko is within 1.7× of turbo at 1/30 the parameters. That is the regime the model
  card measured (Fleurs CER 8.9 published vs 8.2 np here) and it is not ours.

## Decision

Rejected for the product and **not worth the LiteRT conversion spike**: the only tier
that survives the short-command test (tiny-ko, padded) ties or loses to the whisper-base
we already ship, and neither Korean model can be delivered under its licence. Recorded
in [asr-details.md](../asr-details.md) § Screened and rejected and
[litert-lm-model-support.md](../litert-lm-model-support.md) § Verdicts.

What would reopen it: an MIT Korean *streaming* Moonshine release (today the streaming
family is English-only). Re-run with `Run-MoonshineEval.ps1 -Models <id> -PadTo 5`.

Raw outputs: `Builds/Logs/MoonshineEval/moonshine-eval-20260921*.json` (+ `.clips.jsonl`
per-clip transcripts), gitignored.
