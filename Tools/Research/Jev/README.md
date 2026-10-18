# Jev sample — "jevlike offline" (stage 1)

A Jev-style **decision model** that runs on the device: frozen `google/gemma-3-270m-it`, its layer-12
token states, and a small jevlike pointer head that scores **declared options** and returns
probabilities — no generation. This is a standalone sample, **not** a function-calling replacement
(it picks among closed options; it cannot fill numbers or names).

Why not Jev itself: TypeSafe Jev is a hosted API with no weights, so it cannot run offline on kona.
The head is [jevlike](https://github.com/vinnylarouge/jevlike)'s `AttentionHead` (MIT) with the recipe
skykim/voice-agent-unity found best (layer 12, 48 tokens, drop `<bos>`, rank 128, AdamW 2e-3, dropout 0.1,
label smoothing 0.1, ≤ 20 epochs, run ≥ 0.5 / ask ≥ 0.35 with a 0.2 lead).

## Files

| Path | What |
| --- | --- |
| `Run-JevHead.ps1` | `-Bootstrap` builds `JevBench/.venv` (py 3.13, torch 2.11 cu128, transformers 5.20, ai-edge-litert 2.3); then `train` / `parity --tflite <path>` / `selfcheck` |
| `JevBench/jev_head.py` | state extraction, the head, training on two questions, scoring + routing + calibration, head export for C#, tflite parity |
| `JevBench/export_trunk.py` | runs in the `litert-torch-converter:0.9.4` image: layers 0..11 → `.tflite` fp32 / w8 / i8 |
| `External/jev-work/head/` | `jev_head.pt`, `jev_head.bin` + `jev_head.json` (C# contract below), `report.md` |
| `External/jev-work/trunk/` | `gemma3_270m_l12_t48_{fp32,w8,i8}.tflite` |

```powershell
.\Tools\Research\Jev\Run-JevHead.ps1 -Bootstrap
.\Tools\Research\Jev\Run-JevHead.ps1 train                      # ~20 s states on the 4090 + ~1 min head
docker run --rm -e HF_TOKEN -v fg-hf-cache:/root/.cache/huggingface -v <out>:/out -v <JevBench>:/src:ro `
  --entrypoint python litert-torch-converter:0.9.4 /src/export_trunk.py --out /out
.\Tools\Research\Jev\Run-JevHead.ps1 parity --tflite External\jev-work\trunk\gemma3_270m_l12_t48_w8.tflite --n 123
```

## The two Jev questions

- **`area`** (choice, 15 options): team · drone · control · marker_preset · tactics · window · camera ·
  camera_window · filter · confirm · video_through · view · panel · recording · **none**. Each option is
  declared by a Korean criterion sentence (`AREAS` in `jev_head.py`); its vector is the mean layer-12 state
  of that sentence, so options can be re-declared at run time without retraining the trunk.
- **`noul`** (asked as a two-option choice through the same head): "바로 실행할 수 있는 UI 명령" vs "실행하지 않을 말".

Labels come from `FunctionGemma/FunctionGemmaBench/data_v2_empty` (function → area, DefaultResponse → none).

## Results (2026-10-16)

Head: val area acc **0.997** (649). Same training data as FunctionGemma **v2e**, so heldout2/3 compare
like for like (FG's implied area = area of the function it called; invented names → none).

| set | n | Jev area acc | Jev noul acc | ECE | run ✓ | **run ✗** | refuse (want none) | refuse (want cmd) | FG area acc | **FG wrong runs** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eval | 147 | 0.993 | 0.986 | 0.095 | 125 | 0 | 22 | 0 | 1.000 (v3e) | 0 |
| heldout | 118 | 0.983 | 0.975 | 0.104 | 103 | 1 | 12 | 1 | 0.983 (v3e) | 0 |
| heldout2 | 108 | 0.833 | 0.824 | 0.037 | 77 | 9 | 13 | 9 | 0.861 (v2e) | 6 |
| heldout3 | 123 | 0.813 | 0.813 | 0.075 | 84 | 10 | 16 | 13 | 0.829 (v2e) | 10 |

- The decision head is **close to, not better than**, the fine-tuned router on area, and it makes as many
  or more wrong runs. What it buys is the *Jev shape*: per-option probabilities in one pass, options
  re-declarable at run time, and a ~250 k-parameter head that trains in a minute on top of a frozen trunk.
- Calibration is usable but not a safety net: on heldout3 the 0.9–1.0 bucket is 91 % right, 0.6–0.7 only
  43 %. The `ask` band (0.35–0.5 with a 0.2 lead) never fired — the head is decisive, so the thresholds
  mostly decide run vs refuse.

## Trunk export (works, after one fix)

Converting HF's Gemma3 layers as-is (litert-torch 0.9.4 generic `convert`) gives a graph that is exact at
position 0 and off by thousands at every other position. Bisected stage by stage: embeddings, norms,
q/k projections, q/k norms and RoPE all match to ~1e-5, but
`matmul(q, repeat_kv(k).transpose(2, 3))` does not (Δ 288). gemma-3-270m has a single kv head, so
`export_trunk.py` skips the kv expand: q is flattened to `[1, H·T, D]` and multiplied with the one k / v
(constant RoPE tables and causal mask; exact while tokens ≤ the 512 sliding window).

| graph | size | vs HF states | head argmax agree (123 heldout3) | desktop CPU invoke p50 (4 thr, XNNPACK) |
|---|---:|---:|---:|---:|
| fp32 | 939 MB | max Δ 0.020 (values reach ~7 500) | **123/123** | 40 ms |
| w8 (`weight_only_wi8_afp32`) | 239 MB | max Δ 488 | 120/123 | 42 ms |
| i8 (`dynamic_wi8_afp32`) | 239 MB | max Δ 1176 | 119/123 | 25 ms |

The 168 M-parameter embedding table is most of the size. int8 drift comes from Gemma's massive
activations; the cheap fix is to train the head on states taken *from the w8 graph* (stage 2).

## Head tensor contract (`jev_head.bin` + `jev_head.json`)

Little-endian float32, concatenated in the manifest's order with byte offsets:
`context_norm.{weight,bias}` [640], `option_norm.{weight,bias}` [640], `query.weight` / `key.weight` /
`value.weight` [128, 640] (nn.Linear layout: out × in), `area_options` [15, 640], `noul_options` [2, 640].

```
C  = LayerNorm(states[1..n-1])            # eps 1e-5; <bos> (position 0) and pads excluded
Q  = LayerNorm_opt(options) · Wqᵀ         # [N, 128]
K  = C · Wkᵀ ; V = C · Wvᵀ                # [L, 128]
S  = softmax_L(Q · Kᵀ / √128)             # per option, over real tokens only
logit = Σ_r Q ⊙ (S · V) / √128            # [N]  → softmax over options
```

## Stage 2 — on the device (2026-10-17, AAR take18)

| Piece | Where |
|---|---|
| JNI `nativeRunJevTrunk` + static Java `UnityLiteRtLmBridge.runJevTrunk(model, tokenizer.model, text) → float[]` | `Tools/UnityAar/litert-lm-unity-aar.patch` (take18; previous kept as `.patch.take17`). SentencePiece via `litert::support::SentencePieceTokenizer`, `<bos>`=2 prepended, right-padded with 0, one cached `CompiledModel` on CPU/XNNPACK. Returns `[n, tokenizeMs, invokeMs, compileMs]` + `(n-1)·640` states |
| C# head + routing | `Runtime/LiteRtLmJevDecider.cs` (`Decide(text)` on device, `Decide(float[] raw)` anywhere) — matches the Python head to 1e-6 on 5 sentences |
| Sample scene (on-screen field + top-5 bars + noul + verdict; headless prompt-set mode) | `Samples~/AutomatedTests/Runtime/Smoke/LiteRtLmJevSmokeTestRunner.cs`, scene `Samples~/AutomatedTests/Scenes/LiteRtLmJevSmokeTestScene.unity` |
| APK (own id `com.Leuconoe.LiteRTLMUnity.jev`, no StreamingAssets) | `Samples~/AutomatedTests/Editor/LiteRtLmJevSmokeApkBuild.cs` → menu `LiteRT-LM/Build/Android/Build Jev Smoke Test APK` |
| Device run | `Tools/Windows/Tests/Run-LiteRtLmAndroidJevSmokeTest.ps1 [-Trunk …] [-PromptsFile cases.json]`; grade/compare with `Run-JevHead.ps1 device --results …` |

The head shipped to the device is trained on **w8 trunk states** (`train --trunk … --out External/jev-work/head-w8`).
The gemma-3 SentencePiece `tokenizer.model` gives the same ids as the HF tokenizer on all 1 282 eval texts.

Device 46a880a0 (kona, CPU), heldout3 (123), device already hot (74–78 °C):

| trunk | trunk ms p50 | head ms p50 | wall ms p50 | first compile | area acc (device = desktop) | top-1 agreement device vs desktop |
|---|---:|---:|---:|---:|---:|---:|
| w8 (239 MB) | 201 | 1.8 | 204 | 0.78 s | 0.805 | 123/123 (max Δp 0.001) |
| **i8 (239 MB)** | **61** | 1.8 | **63** | 0.34 s | 0.797 | 123/123 (max Δp 0.085) |

i8 is 3.3× faster on the device (weight-only int8 dequantizes per op, the Supertonic lesson again).
Demo (w8): `알파팀 제어권 회수해` → control 0.91 · run; `3번 드론 카메라 띄워` → camera 0.89 · run;
`위협 기동로 해제` → filter 0.95 · run; `녹화 시작` → recording 0.90 · run; `오늘 날씨 어때` → none 0.87, noul(yes) 0.06 · refuse.
Next if this sample is kept: ship the i8 trunk with a head retrained on i8 states.

## Stage 2 plan (as written before stage 2)

1. **AAR (rebuild required).** The bridge only has purpose-built entry points (`nativeRunWhisperAsrSmoke`,
   `…Qwen3Asr…`, `…Parakeet…`, `nativeRunSupertonicTts`, `nativeInspectLiteRtModel`); there is no generic
   "run this .tflite on int32 ids" call and no tokenizer exposed. Add `nativeRunJevTrunk(modelPath,
   tokenizerModelPath, text) → float[]`: `litert_lm/support/tokenizer/sentencepiece_tokenizer.h`
   (`CreateFromFile` + `TextToTokenIds`, present in the builder image) for tokens, prepend `<bos>`=2, right-pad
   with 0 to 48, one cached `CompiledModel` on CPU/XNNPACK, return states [n-1, 640] for the real tokens.
   Verify the new symbol inside `jni/arm64-v8a/liblitertlm_jni.so` after the build (CLAUDE.md).
2. **C# head** (`Runtime/`): read `jev_head.json/.bin`, implement the five lines above (~9 M MACs, single-digit
   ms), re-declarable options = run the trunk on the option sentence once and cache its mean state.
3. **Retrain on w8 states** so the device head matches the device trunk.
4. **Sample scene** via unity-mcp: text field (+ the existing mic/ASR path) → bar chart of the 15 `area`
   probabilities, the `noul` probability and the routing verdict. Generate through the scene generator
   (`LiteRT-LM/Test Scenes/Generate All`), then `Sync-LiteRtLmGeneratedScenes.ps1`. The Windows editor has no
   in-process LiteRT path for plain .tflite (the CLI only runs .litertlm), so the editor shows the device
   build's results or a note.
5. **Device measurement on 46a880a0**: a prompt-set runner like `Run-LiteRtLmAndroidFcPromptSet.ps1`
   (push trunk w8 + `tokenizer.model` + head, run the heldout sets, pull per-prompt probabilities and trunk /
   head ms), `-SkipThermalWait` semantics and recorded temperatures.
