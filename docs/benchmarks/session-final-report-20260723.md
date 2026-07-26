# Session report — LiteRT-LM v0.14 upgrade and model expansion (2026-07-23)

Historical snapshot of the 2026-07-23 session. Statuses marked "in progress"
were completed in later cycles — see
[`device-cycle1-baseline.md`](device-cycle1-baseline.md) and
[`../handoffs/asr-training-program-handoff.md`](../handoffs/asr-training-program-handoff.md)
for the current state.

Requested scope: update the LiteRT-LM framework, convert and test the gemma-4
QAT model for LiteRT, evaluate new ASR/LLM models, build a Windows ASR/LLM
environment, add GPU acceleration with CPU fallback, create test scenes, roll
out quantization (int4 as the minimum tier), and run PDCA verification on the
physical device (46a880a0).

## 1. Summary (15 of 19 tasks complete)

### Framework / build

| Item | Result |
| --- | --- |
| LiteRT-LM v0.14.0 upgrade | ✅ `unity-v0.14.0` branch. Patch rewritten (including two v0.11→v0.14 API fixes). `.litertlm` format 1.5.0 compatibility retained |
| Android AAR | ✅ take4 deployed (16:32). New: qwen3 ASR mode, `sendMessageWithMedia` (image/audio), vision/audio backend init, RMS normalization + VAD + EOS guard, 128-mel whisper auto-detection. take5 (shape-based decode binding + featureMd5) building |
| Windows binaries | ✅ Two v0.14 executables (13:53) plus seven DLLs (including the new libwebgpu_dawn). Five custom FC flags verified. Build traps solved: Korean CP949 codepage (`/utf-8`), MAX_PATH (`output_base=C:/bzl-lm`), pinning VS2022 |
| Windows GPU + CPU fallback | ✅ GPU works again on v0.14 (RTX 4090, 53 tok/s = 2× CPU). `LiteRtLmWindowsCliClient` retries once on CPU after a GPU failure and keeps a session health flag. Windows default backend switched to GPU |
| Windows ASR environment | ✅ gemma-4 audio path verified (transcription accurate) plus the new `Run-LiteRtLmWindowsAsrSmokeTest.ps1` |

### Model verdicts

| Model | Verdict | Rationale |
| --- | --- | --- |
| gemma-4-E2B QAT (mobile-transformers) | ✅ No conversion needed | The existing `gemma-4-E2B-it.litertlm` is already the official QAT wNa8o8 build (SHA matches). DIY conversion is impossible — the tooling rejects quantized checkpoints |
| Qwen3-ASR-0.6B | ✅ Adopted | Official tflite plus an in-house JNI port. Korean verified, all device clips pass |
| Qwen3.5-0.8B-MTP | ❌ Not feasible | MTP is llama.cpp-only, the architecture is unsupported, and the community port collapses on v0.14 |
| VibeVoice-ASR | ❌ Not feasible | 8.7B — unsuitable on-device |
| Qwen3-ASR-1.7B | ⏸ Deferred | No port exists (1–3 weeks of work). 0.6B quality is sufficient |
| Bonsai-1.7B | ⏸ Skip recommended | The 1-bit advantage disappears during LiteRT conversion |
| NVFP4 | ❌ Dropped (user decision) | No LiteRT path |
| Image generation | ❌ Passed over (user decision) | gemma-4 is input-only; SD-on-LiteRT is 306 s per image |
| LFM2.5-1.2B int4 (FC) | ✅ Adopted | Meets the "usable without fine-tuning" bar. Loads and infers on Windows and device |
| FunctionGemma-270M | Kept as a comparison point | Designed around fine-tuning, so excluded from selection |
| Hammer2.1 | ❌ Excluded | CC-BY-NC license |

### Quantization (int4 minimum tier — user policy)

- **Recipe settled**: `dynamic_wi4b64_afp32` (block-64) as the base, mixed with
  i8 on sensitive scopes (embeddings/encoder). wi4c (channelwise), wi4b32,
  int2, Q5 and 1.58b were all ruled out by measurement or research.
- **wi4b64 = 4-bit weights + one fp16 scale per 64-value block + fp32
  activations** — half the size of i8, and faster on bandwidth-bound devices
  (+38 % measured).
- **Produced and deployed**: whisper-base i8/i4, whisper-tiny i4,
  whisper-medium i8/i4, whisper-large-v3 i8/i4, whisper-turbo i4, Qwen3-ASR i4
  (later removed for hallucination), qwen2.5-0.5b i4 (265 MB), qwen2.5-1.5b i4
  (790 MB). The `.litertlm` unpack → quantize → repack pipeline is established
  (litert-lm-builder).
- Not applicable: gemma3-270m / FunctionGemma (no f32 source; i8→i4 demonstrated
  to be a no-op).

### ASR benchmark (10 re-recorded clips × CER/WER/RTF — asr-model-matrix.md)

- **Best**: whisper-turbo i4 (755 MB, 8/9 exact, CER 0.000/0.000) — desktop.
- Balanced: base i8 (77 MB, 6/9 on the re-recorded clips, Korean CER 0.000).
- Strongest device voice-command tier: qwen3-asr i8 (passes both `볼륨 업` takes).
- Short-utterance work: RMS normalization + VAD + EOS guard shipped. Hotwords
  were excluded on user instruction — no pre-biasing toward expected values.
  Two label errors were resolved by re-recording.
- Open: the device disagrees with desktop on very short whisper clips
  (suspect: mel/STFT i8 numerics) — take5 featureMd5 diagnostics will A/B it.

### FC benchmark (20 cases, Windows CPU — fc-model-benchmark.md)

| Tier | Pick | Score |
| --- | --- | --- |
| Flagship | gemma-4-E2B QAT | 19/20, 15.5 tok/s |
| Mid | LFM2.5-1.2B int4 (+Hermes prompt) | 17/20, 23.6 tok/s (fastest) |
| Small | qwen3_0_6b_mixed_int4 (+QwenHermes) | 18/20, 475 MB |
| Rejected | gemma3-1b, qwen2.5 family | 2–8/20 |

### GGUF comparison (gemma4-gguf-vs-litertlm.md)

- llama.cpp CUDA decode 241–245 tok/s vs litertlm CPU 13.6 (~16×) — but GGUF
  has no Android path. Roles: litertlm is the product, llama.cpp is the Windows
  experiment stack.
- v0.14 restored litertlm Windows GPU (previously 3/3 failures).

### Device PDCA (46a880a0, kona — device-cycle1-baseline.md)

- **Cycle 1**: all five LLMs PASS. The project i4 is fastest (35.5 tok/s, +38 %
  over q8). LFM2.5 confirmed supported on device. GPU decode losing to CPU was
  root-caused as an Adreno 650 structural property (sampler is fine, not a
  regression — use CPU for decode, GPU for prefill and multimodal). qwen3-asr
  passes every clip. The earlier crash was traced to a stale APK's SIGSEGV, not
  memory pressure.
- **Cycle 2**: **image understanding PASS** (accurate description of
  apples.jpg, GPU 7.6 s = 3.1× CPU), **audio multimodal PASS** (transcription
  matches, 4.1 s). Zero regressions on tiny/base. 13 runs, zero crashes.
  Multimodal prerequisites established: visionBackend/audioBackend init plus
  maxNumTokens 4000.
- **Cycle 3 (in progress)**: retry turbo i4 on device with take5, featureMd5
  A/B, and FC scene verification.

### Test scenes / UX (#11 — code complete, scene generation pending)

- Five scene runners plus the generator (`LiteRtLmTestSceneGenerator`, menu
  `LiteRT-LM/Test Scenes/Generate All`) implemented and compiling. Existing
  scenes moved to `Assets/Scenes/Tests/`.
- ASR scene: model dropdown + 10-clip audio dropdown. Chat scene: think/no_think
  toggle (Qwen3) and a five-model dropdown. Multimodal scene:
  SendMessageWithMedia wired up (cycle 2).
- **Blocked on**: the user-specified unity-mcp route — the MCP server was not
  connected to the CC session. Alternatives: one editor menu click, or connect
  MCP and re-request.

## 2. Where the outputs live

- Benchmarks: `docs/benchmarks/{asr-model-matrix, fc-model-benchmark,
  gemma4-gguf-vs-litertlm, short-utterance-asr-research,
  device-cycle1-baseline}.md`
- Handoff: `docs/handoffs/v0.14-upgrade-handoff.md`
- Models: `Assets/StreamingAssets/{LLM,ASR,Multimodal,TestAssets}/…`
  (by category, then by model)
- Tools: `Tools/Windows/Run-LiteRtLmWindowsAsrSmokeTest.ps1`, the i4 repack
  pipeline (scratchpad + `External/ModelWork/README-i4-prototypes.md`)

## 3. Remaining work (as of 2026-07-23)

1. **Cycle 3** (in progress): build the take5 AAR → turbo i4 on device,
   featureMd5 A/B, multimodal FC runner device verification
2. **Scene generation**: connect unity-mcp (editor Window > MCP For Unity, then
   restart CC) or run the menu item
3. **#8 final documentation**: update README / llm-details / asr-details (the
   benchmark documents are done)
4. **#13 Linux/macOS**: low priority — upstream macos_arm64 binaries exist, but
   the custom flags require a per-OS build
5. **#7 Bonsai**: skip recommended — awaiting the user's decision
