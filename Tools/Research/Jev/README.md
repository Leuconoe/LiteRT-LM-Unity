# Jev sample — "jevlike offline"

A Jev-style **decision model** that runs on the device: frozen `google/gemma-3-270m-it`, its layer-12
token states, and a small jevlike pointer head that scores **declared options** and returns
probabilities — no generation. It picks among closed options; it cannot fill numbers or names, so it is
a companion to a function-calling router, not a replacement
([FunctionGemma router](../../../docs/functiongemma-function-calling.md)).

TypeSafe Jev itself is a hosted API with no weights, so it cannot run offline. The head is
[jevlike](https://github.com/vinnylarouge/jevlike)'s `AttentionHead` (MIT) with the recipe
skykim/voice-agent-unity found best (layer 12, 48 tokens, drop `<bos>`, rank 128, AdamW 2e-3, dropout 0.1,
label smoothing 0.1, ≤ 20 epochs, run ≥ 0.5 / ask ≥ 0.35 with a 0.2 lead).

## Files

| Path | What |
| --- | --- |
| `Run-JevHead.ps1` | `-Bootstrap` builds `JevBench/.venv` (py 3.12/3.13, torch cu128); then `train` / `parity --tflite <path>` / `device --results <path>` / `selfcheck` |
| `JevBench/sample_areas.json` | the two questions for the sample smart-home contract: `area` options (criterion sentence + the contract functions each covers, `none` for refusals) and `noul` |
| `JevBench/jev_head.py` | state extraction, the head, training on both questions, scoring + routing + calibration, head export for C#, tflite parity, device grading |
| `JevBench/export_trunk.py` | runs in the `litert-torch-converter:0.9.4` image: layers 0..11 → `.tflite` fp32 / w8 / i8 |

Training records are the FunctionGemma generator's output (`FunctionGemmaBench/make_dataset.py`, text +
function → area), so one contract and one data set serve both samples. Outputs go to `External/jev-work/`
(untracked); no model or head file is committed.

```powershell
.\Tools\Research\Jev\Run-JevHead.ps1 -Bootstrap
python Tools\Research\FunctionGemma\FunctionGemmaBench\make_dataset.py          # -> FunctionGemmaBench\data
.\Tools\Research\Jev\Run-JevHead.ps1 train                                       # HF states; minutes on a CPU
docker run --rm -e HF_TOKEN -v fg-hf-cache:/root/.cache/huggingface -v <out>:/out -v <JevBench>:/src:ro `
  --entrypoint python litert-torch-converter:0.9.4 /src/export_trunk.py --out /out
.\Tools\Research\Jev\Run-JevHead.ps1 train --trunk External\jev-work\trunk\gemma3_270m_l12_t48_w8.tflite --out External\jev-work\head-w8
.\Tools\Research\Jev\Run-JevHead.ps1 parity --tflite External\jev-work\trunk\gemma3_270m_l12_t48_w8.tflite
```

Options are re-declarable: each option vector is the mean layer-12 state of its criterion sentence, so a
new criterion needs no trunk retraining (retrain the head when the option set changes much).

## Trunk export (works, after one fix)

Converting HF's Gemma3 layers as-is (litert-torch 0.9.4 generic `convert`) gives a graph that is exact at
position 0 and off by thousands at every other position. Bisected stage by stage: embeddings, norms,
q/k projections, q/k norms and RoPE all match to ~1e-5, but `matmul(q, repeat_kv(k).transpose(2, 3))`
does not. gemma-3-270m has a single kv head, so `export_trunk.py` skips the kv expand: q is flattened to
`[1, H·T, D]` and multiplied with the one k / v (constant RoPE tables and causal mask; exact while tokens
≤ the 512 sliding window).

| graph | size | vs HF states | desktop CPU invoke p50 (4 thr, XNNPACK) |
|---|---:|---|---:|
| fp32 | 939 MB | max Δ 0.020 (values reach ~7 500) | 40 ms |
| w8 (`weight_only_wi8_afp32`) | 239 MB | drifts on Gemma's massive activations | 42 ms |
| i8 (`dynamic_wi8_afp32`) | 239 MB | drifts more | 25 ms |

The 168 M-parameter embedding table is most of the size. Train the head on states taken *from the graph
you ship* (`train --trunk …`) so the device head matches the device trunk.

## Head tensor contract (`jev_head.bin` + `jev_head.json`)

Little-endian float32, concatenated in the manifest's order with byte offsets:
`context_norm.{weight,bias}` [640], `option_norm.{weight,bias}` [640], `query.weight` / `key.weight` /
`value.weight` [128, 640] (nn.Linear layout: out × in), `area_options` [N, 640], `noul_options` [2, 640].
The manifest also carries the option keys, criteria and routing thresholds.

```
C  = LayerNorm(states[1..n-1])            # eps 1e-5; <bos> (position 0) and pads excluded
Q  = LayerNorm_opt(options) · Wqᵀ         # [N, 128]
K  = C · Wkᵀ ; V = C · Wvᵀ                # [L, 128]
S  = softmax_L(Q · Kᵀ / √128)             # per option, over real tokens only
logit = Σ_r Q ⊙ (S · V) / √128            # [N]  → softmax over options
```

## On the device (AAR JNI `runJevTrunk`)

| Piece | Where |
|---|---|
| JNI `nativeRunJevTrunk` + static Java `UnityLiteRtLmBridge.runJevTrunk(model, tokenizer.model, text) → float[]` | `Tools/UnityAar/litert-lm-unity-aar.patch`. SentencePiece tokenizer, `<bos>`=2 prepended, right-padded with 0, one cached `CompiledModel` on CPU/XNNPACK. Returns `[n, tokenizeMs, invokeMs, compileMs]` + `(n-1)·640` states |
| C# head + routing | `Runtime/LiteRtLmJevDecider.cs` (`Decide(text)` on device, `Decide(float[] raw)` anywhere) |
| Sample scene (on-screen field + top-5 bars + noul + verdict; headless prompt-set mode) | `Samples~/AutomatedTests/Runtime/Smoke/LiteRtLmJevSmokeTestRunner.cs`, scene `Samples~/AutomatedTests/Scenes/LiteRtLmJevSmokeTestScene.unity` |
| APK (own id `com.Leuconoe.LiteRTLMUnity.jev`, no StreamingAssets) | menu `LiteRT-LM/Build/Android/Build Jev Smoke Test APK` |
| Device run | `Tools/Windows/Tests/Run-LiteRtLmAndroidJevSmokeTest.ps1 [-Trunk …] [-PromptsFile cases.json]`; grade with `Run-JevHead.ps1 device --results …` |

On kona / SM8250 (CPU) the trunk takes ~61 ms per utterance with the i8 graph and ~200 ms with w8; the
head adds ~2 ms; the first call compiles the graph (0.3–0.8 s). Accuracy depends on your options and data
— measure it on your own eval set.
