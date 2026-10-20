# RAG sample — embedder, Korean data, LiteRT encoder, on-device results

The RAG sample is a standalone capability, **not** a function-calling alternative
(user decision 2026-10-16). Stage 1 answers: which embedder, which Korean data, and
can the embedder run on LiteRT on the headset.

**Answer:** `intfloat/multilingual-e5-small`, int8 (`i8`), 122 MB per graph — same
retrieval quality as the HF model (KorQuAD-dev R@10 0.97), ~10 ms per query on desktop
CPU, 39–41 ms on kona (stage 2). The device needs only the query graph; passage
embeddings are precomputed and shipped (≈0.7 MB for the sample corpus).

## Data

| Role | Set | Size | Why |
| --- | --- | --- | --- |
| **Sample corpus** | KorQuAD 1.0 dev (`KorQuAD/squad_kor_v1`, validation) | 960 unique paragraphs, 5,774 questions with answer spans | fits an on-device index; every question points at its paragraph and carries the answer text, so stage 2 can check generated answers |
| Benchmark | Ko-StrategyQA (`mteb/Ko-StrategyQA`, dev) | 9,251 docs, 592 queries | MTEB-ko, multi-hop, hardest of the three |
| Benchmark | AutoRAGRetrieval (`mteb/AutoRAGRetrieval`) | 720 docs, 114 queries | the allganize RAG-Evaluation-Dataset-KO domains (finance / public / medical / law / commerce) in MTEB form |

`allganize/RAG-Evaluation-Dataset-KO` itself ships PDFs to download and parse; its MTEB
version above is the usable form. MIRACL-ko's corpus is far beyond an on-device index.

## Embedder comparison (desktop, cosine top-10, max 512 tokens)

nDCG@10 / R@1 / R@10. CPU = batch-1 query encode, 4 threads, p50.

| Model | Params | KorQuAD-dev | Ko-StrategyQA | AutoRAG | CPU ms/query |
| --- | ---: | --- | --- | --- | ---: |
| **e5-small** | 118 M | 0.881 / 0.778 / 0.972 | 0.752 / 0.466 / 0.822 | 0.801 / 0.658 / 0.947 | **19.7** |
| e5-base | 278 M | 0.878 / 0.774 / 0.971 | 0.763 / 0.480 / 0.821 | 0.797 / 0.640 / 0.947 | 56.3 |
| e5-large | 560 M | **0.897 / 0.802 / 0.979** | 0.803 / **0.507** / 0.854 | **0.813 / 0.675** / 0.939 | 211.2 |
| e5-large-instruct | 560 M | 0.887 / 0.786 / 0.973 | **0.805** / 0.496 / 0.864 | 0.777 / 0.632 / 0.930 | 869.9 |
| embeddinggemma-300m | 308 M | 0.895 / **0.805** / 0.974 | 0.801 / 0.476 / **0.872** | 0.782 / 0.640 / 0.930 | 98.5 |
| bge-m3 (dense) | 568 M | 0.892 / 0.799 / 0.973 | 0.794 / 0.493 / 0.852 | 0.805 / 0.658 / 0.947 | 167.4 |

- e5-base buys nothing over e5-small here; the large models gain 1–5 nDCG points at
  5–40× the latency.
- embeddinggemma-300m is the strongest mid-size option and litert-community already
  ships LiteRT builds (`litert-community/embeddinggemma-300m`, seq 256–2048, ~183 MB
  mixed precision) — the natural upgrade path if e5-small's quality is not enough.

## LiteRT conversion (`RagBench/export_e5.py`, litert-torch 0.9.4 in Docker)

One fixed-shape graph per length (`t64` for queries, `t512` for passages):
`(input_ids int32[1,L], attention_mask int32[1,L]) → float32[1,384]`, mean pooling and
L2 norm inside the graph. Inputs are named `args_0` (ids) and `args_1` (mask) — bind by
name. The generic `litert_torch.convert` handles XLM-R attention correctly (no
`repeat_kv`: e5 has full multi-head kv), unlike the Gemma3 case in CLAUDE.md.

| e5-small graph | MB | KorQuAD R@1 / R@10 | AutoRAG R@1 / R@10 | min cos vs HF | query t64 ms | passage t512 ms |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| fp32 | 470 | 0.779 / 0.972 | 0.658 / 0.947 | 1.00000 | 15.1 | 91.3 |
| w8 (`weight_only_wi8_afp32`) | 122 | 0.780 / 0.972 | 0.658 / 0.947 | 0.99973 | 18.8 | 95.2 |
| **i8 (`dynamic_wi8_afp32`)** | 122 | 0.778 / 0.970 | 0.658 / 0.947 | 0.99866 | **9.9** | **51.0** |

Desktop CPU, XNNPACK, 4 threads, p50 over all KorQuAD/AutoRAG queries and passages.
e5-base converts too (w8 282 MB, query 36 ms) but i8 drops to cos 0.98 and it is no more
accurate, so it is not pursued. 96 M of e5-small's 118 M parameters are the 250k-token
embedding table — pruning the vocabulary to Korean/English/digits would shrink the graph
to roughly 30 MB (not done; only worth it if 122 MB matters).

## Device tokenisation (checked: `rag_eval.py tokcheck`)

The AAR's `litert_lm/support/tokenizer/sentencepiece_tokenizer.h` returns raw
SentencePiece ids. XLM-R's HF ids are those **+1** (fairseq offset), with spm `<unk>` 0
mapped to 3, wrapped as `<s>`=0 … `</s>`=2, padded with `<pad>`=1 and attention_mask 0 on
pads. Prefixes `query: ` / `passage: ` are part of the text.
`tokcheck --n 3000` (texts from all three sets plus five edge strings): 3,000 / 3,005
identical. On the sample corpus alone (KorQuAD + AutoRAG, 3,000 texts) 2,999 match,
every query included; the one mismatch is a long digit string inside a passage
(`1TB = 1000000000000바이트`), embedding cosine 0.99998 — negligible.

## Stage 2 — on-device RAG (kona, 2026-10-17)

User decisions: answer LLM gemma-4-E2B (official QAT `.litertlm`), KorQuAD wiki demo
corpus, compare e5-small i8 against litert-community embeddinggemma-300m on device.

**Pipeline.** Query text → `runTextEncoder` (AAR take19) → cosine top-3 over an fp16
index → prompt "문서 1..3 + 질문 + 답:" (each passage ≤ 700 chars) with a system
instruction to answer with one short span from the documents or '모름' → gemma-4-E2B.

- **JNI (AAR take19).** `UnityLiteRtLmBridge.runTextEncoder(model, spm, text, idOffset,
  unkId, bosId, eosId, padId, useMask)` → `[tokens, tokenizeMs, invokeMs, compileMs] +
  embedding`. In-JNI SentencePiece, fixed-shape CompiledModel on CPU/XNNPACK, cached per
  model+tokenizer. `getLastBenchmark()`
  returns LiteRT-LM's per-turn `BenchmarkInfo` (TTFT, prefill/decode tok/s); the engine
  enables timing-only benchmark params (no synthetic turns).
- **Index** (`rag_eval.py index` → `External/rag-work/index/`): 960 passages, the same 100
  KorQuAD-dev questions (`random.Random(0)`) for every run, per embedder
  `<name>.f16` (int32 N, int32 D, N·D halves), `<name>.query.tflite`, the tokenizer and a
  JSON spec. e5-small: t64 i8 query graph, 0.74 MB index. embeddinggemma: litert-community
  seq256 mixed-precision query graph (179 MB; smallest length it ships), seq512 for
  passages, 1.47 MB index; desktop KorQuAD R@1 0.801 / R@10 0.972, 82 ms per query.
- **C#:** `Runtime/LiteRtLmRag.cs` (`LiteRtLmTextEncoder`, `LiteRtLmRagIndex`: load,
  top-k, prompt), `Samples~/AutomatedTests/.../LiteRtLmRagSmokeTestRunner.cs` (prompt-set
  mode + an ask box), scene `LiteRtLmRagSmokeTestScene`, APK `com.Leuconoe.LiteRTLMUnity.rag`
  (`LiteRT-LM/Build/Android/Build RAG Smoke Test APK`).

Device results: 100 questions per run; p50 latencies; EM / char-F1 / contains after
NFKC, lower-case, punctuation/space strip.

| Embedder | LLM | hit@1 | hit@3 | embed ms | retrieve ms | TTFT s | answer s | EM | F1 | contains | °C before → after |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **e5-small i8** | **GPU** | 0.82 | 0.92 | **41** | 0.5 | **1.9** | **2.2** | **0.55** | 0.731 | 0.81 | 80.2 → 70.1 |
| embeddinggemma | GPU | 0.81 | 0.94 | 742 | 1.1 | 1.9 | 2.3 | 0.52 | **0.744** | **0.88** | 74.0 → 73.6 |
| e5-small i8 | CPU | 0.82 | 0.92 | 39 | 0.5 | 8.9 | 9.5 | 0.54 | 0.739 | 0.84 | 77.1 → 78.3 |
| embeddinggemma | CPU | 0.81 | 0.94 | 741 | 1.2 | 8.8 | 9.5 | 0.51 | 0.726 | 0.84 | 78.3 → 84.5 |

- Device retrieval equals the desktop TFLite numbers (e5 0.82 / 0.92, embeddinggemma
  0.82 / 0.94 desktop, 0.81 on device), so the JNI tokenisation and id rules are right.
- Prompts are ~1,000 tokens, answers ~6 tokens. GPU prefill ~550 tok/s and decode
  ~18 tok/s; CPU prefill ~112 tok/s, so CPU answers take 4× as long. Answer time is
  prefill-dominated: fewer or shorter passages are the latency lever, not output length.
- With the gold passage in the top 3, the answer contains the gold span 86–91 % of the
  time; without it, 25–33 % (gemma's own knowledge). '모름' 4–5 / 100.
- embeddinggemma buys +2 hit@3 points and a few contains points at 18× the query cost
  (≈ 0.74 s vs 0.04 s) and +57 MB. **e5-small i8 + gemma-4-E2B GPU is the default**;
  embeddinggemma only pays off on a larger or harder corpus.
- EM is strict against KorQuAD's spans (particles, units, parenthesised aliases, e.g.
  "유니시" vs "유니시(Unish)"); contains/F1 are the fairer quality read.

Open issues: no max-output-token control on the bridge (answers stay short only through
the instruction); embeddinggemma has no graph shorter than 256 tokens (a seq64 export
would cut its query cost ~4×); a scene left modified blocks chained in-editor APK builds
with a modal dialog (open an empty scene before each build).

## Reproduce

```powershell
.\Tools\Research\Rag\Run-RagBench.ps1 -Bootstrap
.\Tools\Research\Rag\Run-RagBench.ps1 bench                         # all six embedders × three sets
.\Tools\Research\Rag\Run-RagBench.ps1 -Export -Model intfloat/multilingual-e5-small -Lengths @(64,512)
.\Tools\Research\Rag\Run-RagBench.ps1 tflite --dir External\rag-work\multilingual-e5-small --sets korquad,autorag
.\Tools\Research\Rag\Run-RagBench.ps1 tokcheck --n 3000
.\Tools\Research\Rag\Run-RagBench.ps1 index                         # External\rag-work\index (both embedders)
.\Tools\Windows\Tests\Run-LiteRtLmAndroidRagSmokeTest.ps1 -Embedder e5-small -LlmBackend GPU   # pwsh 7
.\Tools\Research\Rag\Run-RagBench.ps1 grade-device @files   # $files = (dir Builds\Logs\AndroidRagSmoke\*\results.jsonl).FullName
```

`-Export` runs in the `litert-torch-converter:0.9.4` Docker image (built on first
use). Graphs land in
`External/rag-work/<model>/` (untracked); results in `Builds/Logs/rag-*.jsonl`.
