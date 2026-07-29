# Function-Calling Model Benchmark — Windows v0.14 CLI (2026-07-23)

> **Scope.** Scores come from this project's own 20-case Korean tool-routing set
> with our prompt templates and parser, and describe fit for that specific task.
> A low score here frequently means the output format did not match our parser
> rather than that the model misunderstood the request; it is not a general
> capability ranking of the models involved.

20-case Korean Unity command-routing benchmark (+ 7-case English mobile-actions set)
run against the **v0.14 fork Windows binaries** deployed 2026-07-23 13:53
(`Tools/Windows/Bin/litert_lm_main.windows_x86_64.exe`). Cases, prompt profiles, tools
JSON, output parsing, deterministic guards, and validation are an exact CLI-level
port of `Assets/Scripts/LiteRTLM/LiteRtLmFunctionCallingBenchmarkRunner.cs`
(driver: session scratchpad `fc_bench.py`, raw records `results.jsonl`).

This doc also records the **Windows ASR smoke (#4)** and the **v0.14 GPU backend
re-test (#10)** performed the same session.

## Environment

- Date: 2026-07-23, Windows 11 x64
- CPU: AMD Ryzen 9 7950X (16C/32T), 128 GB RAM; runtime uses 4 XNNPACK threads
- GPU: NVIDIA RTX 4090 (WebGPU accelerator → Direct3D 12, `libwebgpu_dawn.dll`)
- CLI flags per case: `--backend=cpu --system_message_file --tools_json_file
  --enable_constrained_decoding=true --output_message_json=true --input_prompt_file`
  (QwenHermes profile: raw prompt only, unconstrained, no tools JSON — as in the runner)
- Fresh process per case → `init_s` is the real cold/warm model init cost.
  Reference time fixed at 2026-04-24 10:30:00 (runner constant).

## LFM2.5 load verdict (gate check)

**PASS.** `LFM2.5-1.2B-Instruct_int4.litertlm` (702 MB) loads on the v0.14 CLI and
answers Korean correctly (`대한민국의 수도는 **서울**입니다.`).
Init 2.98 s, prefill 55.4 tok/s, decode 17.9 tok/s (CPU). The old exe could not
load LFM2.5 at all — v0.14 resolves this.

## Results (pass rate × decode tok/s × init s, CPU)

| Model | File (size) | Profile | Pass/20 | Decode tok/s | Prefill tok/s | Init s | Avg case s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| **gemma-4-E2B-it** | 2469 MB | CurrentTuned | **19/20** | 15.5 | 399 | 0.25† | 5.5 |
| **LFM2.5-1.2B int4** | 702 MB | QwenHermes | **17/20** | 23.6 | 344 | 3.0 | 7.7 |
| LFM2.5-1.2B int4 | 702 MB | CurrentTuned | 2/20 (14/20‡) | 16.9 | 286 | 3.5 | 9.4 |
| **Qwen3-0.6B** | 586 MB | QwenHermes | **20/20** | 4.4 | 141 | 0.6 | 22.5 |
| Qwen3-0.6B | 586 MB | QwenNoThink | 2/20 | 10.2 | 162 | 0.2 | 3.1 |
| **qwen3_0_6b_mixed_int4** | 475 MB | QwenHermes | **18/20** | 7.1 | 269 | 0.6 | 10.7 |
| qwen3_0_6b_mixed_int4 | 475 MB | QwenNoThink | 9/20 | 9.4 | 329 | 0.4 | 9.4 |
| gemma3-1b-it-int4 | 558 MB | CurrentTuned | 3/20 (3/20 unconstrained) | 17.8–26.8 | 387–523 | 0.4 | 1.6–2.4 |
| Qwen2.5-1.5B wi4b64 (proto) | 790 MB | CurrentTuned | 8/20 | 18.2 | 231 | 1.0 | 8.5 |
| Qwen2.5-0.5B wi4b64 (proto) | 265 MB | CurrentTuned | 2/20 | 56.7 | 1031 | 0.4 | 2.2 |
| functiongemma-270m (mobile_actions) | 276 MB | MobileActions | **6/7** | 38.2 | 1295 | 0.3 | 3.5 |
| kanana-2-1.3b-instruct §| 2.58 GB bf16 | CurrentTuned | 17/20 | 22.0 | — | 52.5 | 1.5 |

§ **Not measured on this runtime.** Kanana has no LiteRT conversion path, so it
was scored on an RTX 4090 through `transformers`
(`Tools/Research/Kanana/Run-KananaEval.ps1`) with the same cases, tools and
grading. The pass rate is comparable; the timings are not — they are unquantized
bf16 on a desktop GPU, and `init s` is a Hugging Face load, not a LiteRT engine
init. Detail and the two blockers: [`../llm-details.md` §6](../llm-details.md#6-evaluated-and-rejected).

† gemma-4-E2B init is 0.25 s because its XNNPACK weight caches already exist next
to the bundle; first-ever load is much slower.
‡ 14/20 with a format-aware parser (see LFM2.5 notes) — the runner's JSON parser
scores the native LFM tool-call format as 0.

## Per-model notes

### gemma-4-E2B (19/20) — best accuracy
Emits proper `tool_calls` message JSON under constrained decoding; correct
full-day/full-month date ranges. Only failure: B19 (지난달/last month →
`2026-03-24` start instead of `2026-03-01`, a month-boundary slip). CPU decode
15.5 tok/s is usable but the 2.5 GB bundle is the cost.

### LFM2.5-1.2B int4 — QwenHermes 17/20, native format caveat
- **CurrentTuned (tools JSON path): the model always answers in its native
  pythonic format** `<|tool_call_start|>[FuncName(startTime="...")]<|tool_call_end|>`
  — the runner's JSON regex parser extracts nothing → 2/20 (guard-assisted only).
  `--enable_constrained_decoding` does not redirect it to JSON.
  With a format-aware parser the same transcripts score **14/20**; residual
  failures are semantic: B02/B03 direction confusion (밝기 낮춰→IncreaseBrightness,
  볼륨 줄여→IncreaseVolume), B10/B14 date args copy the current time instead of
  full-day ranges, B19 wrong month, B17 no call.
- **QwenHermes profile scores 17/20** (fails B12 wrong tool, B13 wrong date,
  B14 Visualize/View confusion) at 23.6 tok/s — best LFM configuration today.
- **Action item**: to use LFM2.5 in the Unity runner, either keep the Hermes-style
  rule prompt or extend `ParseToolCall` with the pythonic `[Func(...)]` format.

### Qwen3-0.6B (20/20 with QwenHermes) — accuracy champion, slow with that profile
The QwenHermes profile (explicit routing rules + date table in the prompt) yields
**20/20**, but decode drops to 4.4 tok/s / 22.5 s per command on CPU — the long
rule prompt plus unconstrained decode is expensive. QwenNoThink+tools JSON is fast
(3.1 s/case) but only 2/20: the model mostly answers plain-text `DefaultResponse`
instead of calling tools.

### qwen3_0_6b_mixed_int4 (18/20 with QwenHermes) — best small-tier balance
18/20 at 10.7 s/case (both failures are 지난달/2025년 3월 month-range slips,
B12/B19). Notably better than the FP base under QwenNoThink too (9/20 vs 2/20).
475 MB, int4-mixed — consistent with the int4-minimum-tier policy.

### gemma3-1b-it-int4 (3/20) — below our FC routing bar
Constrained or unconstrained, it answers the literal text `DefaultResponse` for
nearly every request (3/20 both ways; unconstrained retry confirmed no change).
Fine as a chat model (fluent Korean), wrong tool for routing.

### Qwen2.5 wi4b64 prototypes — runtime validated, low FC score
Both `External/ModelWork` wi4b64 bundles **load and infer correctly** on the v0.14
Windows CLI (inference validation goal met; recipe confirmed at runtime).
FC accuracy is low with CurrentTuned: 0.5B = 2/20 (fast: 56.7 tok/s), 1.5B = 8/20
(emits proper `tool_calls` JSON but date args copy the 10:30 current time instead
of 00:00:00 full-day starts). A Hermes-style rule prompt would likely lift the
1.5B substantially; not measured this session.

### functiongemma-270m mobile_actions (6/7)
Only miss: M03 createContact (answers prose "I need..." instead of a call).
38 tok/s, 0.3 s init — good for its English mobile-action niche; its tool set does
not cover the Korean Unity commands.

## Korean output quality

- LFM2.5 int4: fluent, correct Korean prose (load-check answer natural).
- gemma3-1b int4 (GPU and CPU): fluent Korean lists, no broken jamo observed.
- gemma-4-E2B: Korean audio transcription exact (below); tool-call args clean.
- Qwen3/Qwen2.5: tool-call JSON output only in this benchmark — no Korean prose
  regression check performed this session.

## Per-tier recommendation (Windows CPU, FC routing)

| Tier | Pick | Why |
| --- | --- | --- |
| Flagship (~2.5 GB) | **gemma-4-E2B** | 19/20 with plain CurrentTuned profile + constrained decoding; also the multimodal/ASR carrier |
| Mid (~0.7 GB) | **LFM2.5-1.2B int4 + QwenHermes prompt** (17/20, 23.6 tok/s) | best accuracy/speed balance; add pythonic-format parsing to unlock the tools-JSON path |
| Small (~0.5 GB) | **qwen3_0_6b_mixed_int4 + QwenHermes** (18/20) | near-mid-tier accuracy at 475 MB; FP Qwen3-0.6B hits 20/20 but at 22.5 s/command |
| Not recommended | gemma3-1b (3/20), Qwen2.5-0.5B wi4b64 (2/20) | refuse/route-to-DefaultResponse behavior; keep them for chat/prototype roles |

## Windows ASR smoke — gemma-4 audio path (#4)

**PASS.** `litert_lm_advanced_main.windows_x86_64.exe --backend=cpu
--audio_backend=cpu` on `gemma-4-E2B-it.litertlm` with
`Transcribe the audio: [audio:<path>.mp3]`:

- The E2B bundle **contains audio sections** (`tf_lite_audio_encoder_hw`,
  `tf_lite_audio_adapter`, `tf_lite_end_of_audio`); audio engine creation OK,
  XNNPACK audio caches created next to the bundle.
- **mp3 is supported** (miniaudio) — no wav conversion needed.
- Transcript of `2025년 3월 5일 전술평가 결과 보고.mp3`:
  `2025년 3월 5일 전술 평가 결과 보고` — exact (spacing variant only).
- End-to-end 3.9–5.3 s per run (warm caches).
- **Pitfall**: the CLI media-tag regex is `\[(image|audio):([^\s\]]+)\]` —
  **paths with whitespace are silently not parsed** (the model then sees the
  literal path text and refuses). Stage audio under a space-free path.
- Scripted entry point: `Tools/Windows/Tests/Run-LiteRtLmWindowsAsrSmokeTest.ps1`
  (params `-ModelPath -AudioPath -Prompt -Backend -AudioBackend -TimeoutSeconds
  -Benchmark`; auto-stages whitespace paths, UTF-8 prompt file, writes
  raw + summary logs under `Builds/Logs/WindowsAsrSmoke/`). Verified PASS twice.

## Windows GPU backend re-test (#10 fallback design input)

**v0.14 fixes the GPU backend.** The May exe failed 3/3 on `--backend=gpu`; the
new exe + `libwebgpu_dawn.dll` works:

- Adapter: `NVIDIA GeForce RTX 4090, backend=Direct3D 12` via WebGPU/Dawn.
- gemma3-1b-it-int4: init 5.7 s, decode **49.3 tok/s** (vs 17.8–26.8 CPU),
  fluent Korean output, rc=0 on both runs.
- Implication for the fallback design: GPU-first with CPU fallback is now viable
  on Windows; keep CPU fallback for the ~5.7 s GPU init and for machines without
  D3D12-capable adapters.
