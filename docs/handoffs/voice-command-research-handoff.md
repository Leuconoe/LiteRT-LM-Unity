# Voice-command research handoff (2026-10-18)

Everything a new session needs to continue the FunctionGemma / SenseVoice / Jev / RAG
work of 2026-10-16…18 without re-deriving it. Numbers are on the headset (kona /
SM8250, serial `46a880a0`, CPU unless noted). Detail lives in
[`../benchmarks/functiongemma-korean-fc.md`](../benchmarks/functiongemma-korean-fc.md)
and [`../benchmarks/sensevoice-korean-eval.md`](../benchmarks/sensevoice-korean-eval.md).

## 1. Git state

| What | Where |
| --- | --- |
| Branch | `research/functiongemma-sensevoice-jev-rag` |
| PR | [#2](https://github.com/Leuconoe/LiteRT-LM-Unity/pull/2) — the first 10 commits (`c327d91` … `8773ec1`) |
| Committed, **not pushed** | `2ff3fa5` SenseVoice on the headset (AAR take20), `db9db32` confirmation policy |
| Uncommitted, in progress | `fg_train.py` (per-epoch checkpoint + `--resume`), `make_dataset.py` (filter covers the locked sets), `asr_aug/`, `data_v7_empty/`, `locked/heldout6_*`, `locked/overlap_with_data_v6.json`, edits to `locked/make_h5_voice.py` and `locked/gen_heldout5b.py`, and the rebuild script (§4.2) |

`git push` is refused by Claude Code's auto-mode classifier in this setup — ask the
user to run `! git push` themselves; `gh pr create` worked once the branch existed.
Never `git add -A` (CLAUDE.md); stage explicit paths. Left out on purpose: editor
reserialization churn (URP / ProjectSettings / `.slnx`), `External/`, `outputs/`,
`.bkit/`, `Tools/Builds/`, training data v1–v5, `paraphrase/out/`, patch backups.

## 2. What is decided and shipped

| Piece | State |
| --- | --- |
| **FunctionGemma-270M ko, v6e** | Fine-tuned with **no tool declarations** (31-token prompt). eval 147/147 with 0 wrong actions, blind heldout4 130/137, ~0.3–0.4 s per command, int8 277 MB (cache 256 / prefill 64·128). Staged at `Assets/StreamingAssets/LLM/functiongemma-270m-ko/functiongemma_270m_ko_v6e_c256_p64.litertlm` (gitignored). |
| **Router** `Runtime/LiteRtLmFunctionGemmaRouter.cs` | Parses the raw call; schema guard + grounding guard (numbers / team / soldier name must be in the utterance); clause split when the whole utterance is refused; `ConfirmationPolicy` (+ `MinAsrConfidence`) marks high-impact calls `NeedsConfirmation`. Twin in `fg_common.py` — keep them in sync. |
| **SenseVoice-Small** | LiteRT buckets published as `litert-community/SenseVoiceSmall` and `…-ko-zeroth`. On the headset (AAR take20, `runSenseVoiceAsr`): device = desktop token for token; quiet commands 12/13; ~0.4 s (3 s bucket). Keep one bucket resident. |
| **Jev sample** (jevlike offline) | Frozen gemma-3-270m trunk + jevlike head; area accuracy 0.80, 63 ms. Package `.jev`. |
| **RAG sample** | e5-small i8 + gemma-4-E2B GPU on KorQuAD: hit@3 0.92, 2.2 s to answer. Package `.rag`. |
| **AAR take20** | JNI: `nativeRunSenseVoiceAsr`, `nativeRunTextEncoder`, `nativeGetLastBenchmark`, `nativeRunJevTrunk`, `nativeRunSupertonicTts`, `nativeSendMessage`, whisper/qwen3. After any AAR change, rebuild **and reinstall** every APK (FC prompt-set, Jev, RAG, SenseVoice) and check the package id on disk is back to `com.Leuconoe.LiteRTLMUnity`. |

The voicecommand-webserver session applied the same data recipe (no tool
definitions) to EXAONE 3.5 2.4B LoRA: locked 5a/5b/6 correct 80.7 % → 91.5 %, wrong
actions 16 % → ~4 %.

## 3. The finding that drives the open work

On hard synthetic voices every model keeps **10–15 % wrong actions** (a broken or
even a correct transcript turned into the wrong command). Measured countermeasures:

| Countermeasure | Wrong actions stopped | Cost |
| --- | --- | --- |
| Confirmation policy (high-impact commands) | 7–11 % | asks to confirm ~10 % of correct commands |
| + SenseVoice confidence ≥ 0.90 | 12–14 % on heldout5; **0 extra on heldout6** | +1–2 pp confirmations |
| Train on real ASR transcripts (v7) | pending — §4.1 | — |

18–31 % of wrong actions come from a correct transcript, which is why the ASR
confidence gate is weak and v7 is the main bet.

## 4. In progress — how to resume

### 4.1 v7e: train on real recogniser output

- Data ready: `FunctionGemmaBench/data_v7_empty/` = data_v6_empty + 7,067 ASR-transcript
  records (TTS → SenseVoice w8 / whisper base-acft-ko, relabelled by a Qwen2.5-14B
  teacher; unrecoverable transcripts became refusals): 24,147 train, 27.8 % ASR,
  25 % refusals. Driver `asr_aug/v7_asr_data.py`. Transcripts were exact-filtered
  against eval and heldout 1–6.
- Judge on the **locked heldout6** (`locked/heldout6_cases.json`, 196 utterances,
  385 gate-passed clips; voices not used for v7 data). v6e baseline, router on:

  | Set | ASR | all: correct / wrong | clean: correct / wrong |
  | --- | --- | --- | --- |
  | heldout6 (385) | SenseVoice w8 | 307 / 37 | 289 / 36 (n 364) |
  | heldout6 | whisper base | 282 / 34 | 268 / 33 |
  | heldout5 (2,024) | SenseVoice w8 | 1552 / 214 | 1488 / 210 (n 1,935) |
  | heldout5 | whisper base | 1335 / 307 | 1280 / 304 |

- Train (GPU alone, ~1 h; output dir `External/functiongemma-work/ft-functiongemma-none-v7e/`,
  log `Builds/Logs/fg-voice-v7/`):
  `.\Tools\Research\FunctionGemma\Run-FunctionGemmaEval.ps1 -Train -PassThru @('--train','…\data_v7_empty\train.jsonl','--val','…\data_v7_empty\val.jsonl','--format','functiongemma','--decl-mode','none','--epochs','3','--lr','2e-5','--batch-size','8','--label','ft-functiongemma-none-v7e')`
  — add `'--resume'` after a crash. The first run died at 78 % (CUDA illegal memory
  access while sharing the GPU) and lost 4.4 h; resume exists since then.
- Then: `fg_eval.py --batch-size 1 --segment` on eval_cases (must stay 147/147, 0
  wrong); voice E2E for v7e on heldout6 and heldout5 (`fg_voice_e2e.py fc --manifest …
  --cases … --model … --label v7e --cli-check 0`, reuse the existing ASR jsonl);
  report all + clean. **Convert only if** v7e has fewer wrong actions on heldout6
  without losing more than ~1 pp correct and keeps eval 147/147:
  `Convert-FunctionGemmaToLiteRtLm.ps1 -CacheLength 256 -PrefillLengths @(64,128)`
  (call it directly — `pwsh -File` turns `64,128` into 64128). Then device run,
  stage, and update the sample constant.

### 4.2 Retrain-on-contract-change script

`Tools/Research/FunctionGemma/Rebuild-FunctionGemmaRouter.ps1` (being written; check
its state before relying on it). Intended flow: contract diff vs
`FunctionGemmaBench/contract_snapshot.json` → **stop with a to-do list** when new
tools lack templates, a callsign/filter name is missing from `CALLSIGN_KO` /
`FILTER_NAMES` (in `fg_common.py` **and** the C# router), or eval labels went stale →
data (templates + paraphrases that still validate + v6 additions, filtered against
every eval set) → GPU guard → train with `--resume` retries → gate (0 wrong, ≥ 98 %)
→ convert + CLI parity → export `functiongemma_tools.json`, optional `-Stage` and
`-Device`. `-DryRun` = contract check only, `-Smoke` = CPU plumbing test.
Why it exists: with no declarations in the prompt, any change to tools, parameters
or enum values (e.g. a new callsign) needs a retrain; mapping a call to a UI handler
or editing the confirmation list does not.

## 5. Rules that keep the numbers honest

- **Blind sets are spent once they guide a fix**: heldout (→ v2), heldout2 (→ v3),
  heldout3 (→ v4), heldout4 (→ v6). **heldout5 and heldout6 are locked**: score final
  candidates, report aggregates only, never read their failures to design data.
- Locked sets written after a training set can still collide with it: 24 texts of
  5a/5b/6 occur in data_v6 (`locked/overlap_with_data_v6.json`). Report **all and
  clean**; filter space- **and** punctuation-insensitively, in both directions.
- Synthetic speech ranks, it does not measure absolute accuracy. Gate every synthetic
  voice with an STT intelligibility check (a wrong-generation Supertonic style produced
  mumbling that nearly entered the scores; edge-tts < 7.3.1 now returns HTTP 403).
- Score HF models with `--batch-size 1` (batched left-padded generate garbles output).

## 6. Shared machine etiquette

- **One training at a time on the RTX 4090.** The voicecommand-webserver session
  (`voicecommand-webserver-2a`) trains EXAONE LoRA and runs a server container that
  holds ~7 GB; training plus its server inference has crashed both sides. Message that
  session before starting a GPU training and when you finish.
- Unity AI TTS (`_AI/UnityAiTts`, ElevenLabs presets, 5 points per clip, shared org
  credits ~1,715 left) has no cancel API — a started batch runs to the end. Korean-native
  ElevenLabs voices are not reachable through Unity AI.
- Device scripts end with force-stop **and** `input keyevent KEYCODE_SLEEP`: the
  headset's home launcher holds `KEEP_SCREEN_ON`, so it never sleeps on its own.

## 7. Open decisions for the user

1. Ship v7e if it beats v6e on heldout6 (pending §4.1).
2. Which high-impact commands require confirmation in the product (default list in
   the router; the product owns the final list).
3. Whether to wire LiteRT-LM's `Conversation::Clone` prefix reuse into the JNI, which
   would allow tool declarations in the prompt (no retrain on contract changes) at
   acceptable latency.
4. Merge PR #2 once the round-2 commits are pushed.
