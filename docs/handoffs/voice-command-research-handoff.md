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
| Round 2 (after PR #2 opened) | SenseVoice on the headset (AAR take20), confirmation policy, this handoff, `fg_train.py` checkpoint + `--resume`, locked heldout6 + v7 data, the rebuild script (§4.2), v7e shipped |

`git push` was refused twice by Claude Code's auto-mode classifier on 2026-10-17 and
went through on 2026-10-19 — if it is refused, ask the user to run `! git push`.
Never `git add -A` (CLAUDE.md); stage explicit paths. Left out on purpose: editor
reserialization churn (URP / ProjectSettings / `.slnx`), `External/`, `outputs/`,
`.bkit/`, `Tools/Builds/`, training data v1–v5, `paraphrase/out/`, patch backups.

## 2. What is decided and shipped

| Piece | State |
| --- | --- |
| **FunctionGemma-270M ko, v7e** (shipped 2026-10-19; v6e before) | Fine-tuned with **no tool declarations** (31-token prompt) on v6 data + real recogniser transcripts. eval 147/147 with 0 wrong actions, 40–50 % fewer wrong actions than v6e on the locked voice sets (§4.1), ~0.35 s per command, int8 277 MB (cache 256 / prefill 64·128). Staged at `Assets/StreamingAssets/LLM/functiongemma-270m-ko/functiongemma_270m_ko_v7e_c256_p64.litertlm` (gitignored), named by `FunctionGemmaRouterModel`. |
| **Router** `Runtime/LiteRtLmFunctionGemmaRouter.cs` | Parses the raw call; schema guard + grounding guard (numbers / team / soldier name must be in the utterance); clause split when the whole utterance is refused; `ConfirmationPolicy` (+ `MinAsrConfidence`) marks high-impact calls `NeedsConfirmation`. Twin in `fg_common.py` — keep them in sync. |
| **SenseVoice-Small** | LiteRT buckets published as `litert-community/SenseVoiceSmall` and `…-ko-zeroth`. On the headset (AAR take20, `runSenseVoiceAsr`): device = desktop token for token; quiet commands 12/13; ~0.4 s (3 s bucket). Keep one bucket resident. |
| **Jev sample** (jevlike offline) | Frozen gemma-3-270m trunk + jevlike head; area accuracy 0.80, 63 ms. Package `.jev`. |
| **RAG sample** | e5-small i8 + gemma-4-E2B GPU on KorQuAD: hit@3 0.92, 2.2 s to answer. Package `.rag`. |
| **AAR take20** | JNI: `nativeRunSenseVoiceAsr`, `nativeRunTextEncoder`, `nativeGetLastBenchmark`, `nativeRunJevTrunk`, `nativeRunSupertonicTts`, `nativeSendMessage`, whisper/qwen3. After any AAR change, rebuild **and reinstall** every APK (FC prompt-set, Jev, RAG, SenseVoice) and check the package id on disk is back to `com.Leuconoe.LiteRTLMUnity`. |

The voicecommand-webserver session applied the same data recipe (no tool
definitions) to EXAONE 3.5 2.4B LoRA. Final, on locked 5a/5b/6 minus the 24 overlap
cases (469): correct 80.4 % → 92.5 %, wrong actions 15.1 % → 5.8 % (WebServer
`docs/04-report/ai-tactical-staff.handoff.md`).

## 3. The finding that drives the open work

On hard synthetic voices every model keeps **10–15 % wrong actions** (a broken or
even a correct transcript turned into the wrong command). Measured countermeasures:

| Countermeasure | Wrong actions stopped | Cost |
| --- | --- | --- |
| Confirmation policy (high-impact commands) | 7–11 % | asks to confirm ~10 % of correct commands |
| + SenseVoice confidence ≥ 0.90 | 12–14 % on heldout5; **0 extra on heldout6** | +1–2 pp confirmations |
| Train on real ASR transcripts (v7e) | **40–50 %** (heldout6 SenseVoice 37 → 22, heldout5 214 → 127) | none — correct commands go up too (§4.1) |

18–31 % of wrong actions come from a correct transcript, which is why the ASR
confidence gate is weak and v7 was the main bet — it paid off.

## 4. In progress — how to resume

### 4.1 v7e: train on real recogniser output — done and shipped 2026-10-19

Result (router on; detail in functiongemma-korean-fc.md §7c):

| Set · ASR | v6e correct / wrong | v7e correct / wrong | clean v7e |
| --- | ---: | ---: | ---: |
| heldout6 (385) · SenseVoice w8 | 307 / 37 | **320 / 22** | 301 / 21 |
| heldout6 · whisper base | 282 / 34 | **296 / 22** | 282 / 21 |
| heldout5 (2,024) · SenseVoice w8 | 1552 / 214 | **1620 / 127** | 1542 / 127 |
| heldout5 · whisper base | 1335 / 307 | **1455 / 149** | 1388 / 148 |

Device: eval 147/147, 0 wrong, 346 ms mean; heldout4 (spent) 129/137, 6 wrong (v6e
130 / 4). Trained in 19 min alone on the GPU, first attempt. Converted bundle staged
beside v6e (`functiongemma_270m_ko_v7e_c256_p64.litertlm`, gitignored);
`FunctionGemmaRouterModel` points at it and the rebuild script trains the v7
composition. Score locked sets with `locked/score_locked.py` (counts only, never rows).
heldout5 and heldout6 are now spent for v7: a v8 needs a new locked set.

What was done, for reference:

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

`Tools/Research/FunctionGemma/Rebuild-FunctionGemmaRouter.ps1` — `-DryRun`, a fake
contract (new tool + callsign → exit 2 with 4 to-dos) and `-Smoke` (CPU, train →
convert → CLI parity 8/8) pass, and the first real run (2026-10-19, `-Label v7u
-WaitForGpu -Device`, ~1 h) went through every stage on GPU and the headset —
results in the benchmark doc §7d. Still unexercised: `-Stage` and
`-RefreshParaphrases`. Stage 7 needs the headset attached when it starts; if it is
not, everything else is already done — rerun just the device part by hand:
`Run-LiteRtLmAndroidFcPromptSet.ps1 -DeviceSerial 46a880a0 -Bundle <bundle> -SystemInstructionFile <log>\device\system.txt -PromptsFile <log>\device\prompts.txt -MaxNumTokens 256 -OutDir <log>\device`,
then `rebuild_check.py device-grade --cases <log>\cases\eval_cases.json --results <log>\device\results.jsonl --log <log>`.
Locked-set rows land only in the quiet `04-eval-heldout5*/6.log` files — do not open
them. Usage in
[functiongemma-korean-fc.md](../benchmarks/functiongemma-korean-fc.md) §9. Flow: contract diff vs
`FunctionGemmaBench/contract_snapshot.json` → **stop with a to-do list** when new
tools lack templates, a callsign/filter name is missing from `CALLSIGN_KO` /
`FILTER_NAMES` (in `fg_common.py` **and** the C# router), or eval labels went stale →
data (templates + paraphrases that still validate + v6 additions + the v7 ASR records
that still validate, filtered against every eval set) → GPU guard → train with `--resume` retries → gate (0 wrong, ≥ 98 %)
→ convert + CLI parity → export `functiongemma_tools.json`, optional `-Stage` and
`-Device`. `-DryRun` = contract check only, `-Smoke` = CPU plumbing test.
Why it exists: with no declarations in the prompt, any change to tools, parameters
or enum values (e.g. a new callsign) needs a retrain; mapping a call to a UI handler
or editing the confirmation list does not.

'유지비' (resolved 2026-10-19): TEAM_TYPES has the alias and the app's
`functiongemma_tools.json` matches the current ToolsPrompt again. The v7u rebuild
that trained it was not shipped — v7e already maps '유지비' to UGV, and v7u fell
outside the ship rule on two voice arms (§7d). Data it rebuilds is filtered against every eval set (0 hits); the old
data_v6 holds 35 records of the 24 locked overlap cases, already covered by
all/clean reporting.

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

1. Which high-impact commands require confirmation in the product (default list in
   the router; the product owns the final list).
2. Whether to wire LiteRT-LM's `Conversation::Clone` prefix reuse into the JNI, which
   would allow tool declarations in the prompt (no retrain on contract changes) at
   acceptable latency.
3. Merge PR #2 once the round-2 commits are pushed.
