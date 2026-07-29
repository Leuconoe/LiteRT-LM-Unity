# Gemma 4 E2B on Windows: GGUF (llama.cpp) vs .litertlm (LiteRT-LM)

Date: 2026-07-23. Host: Windows 11, RTX 4090 (24 GB), AMD Ryzen (16 threads used for CPU runs).

- llama.cpp build: `b4d6c7d8f (10091)`, CUDA 12 prebuilt release (`External/gguf-bench/llamacpp/`).
- LiteRT-LM runner: `Tools/Windows/Bin/litert_lm_main.windows_x86_64.exe` (**May 8 build** — pre-v0.14; the numbers below are from this build).
- Korean smoke prompt: `대한민국의 수도는 어디인가요? 한 문장으로 답하세요.` (English smoke: drone-gimbal one-liner.)
- Raw logs: `External/gguf-bench/results/`.

## Models

| Variant | File | Size |
|---|---|---|
| QAT Q4_K_XL (unsloth) | `qat/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf` | 2.44 GB |
| QAT Q2_K_XL (unsloth, "mobile") | `qat-mobile/gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf` | 2.04 GB |
| Non-QAT Q4_0 (bartowski) | `nonqat/gemma-4-E2B-it-Q4_0.gguf` | 2.83 GB |
| Google official QAT Q4_0 | `google-qat/gemma-4-E2B_q4_0-it.gguf` | 3.12 GB |
| LiteRT-LM | `Assets/StreamingAssets/Multimodal/gemma-4-e2b/gemma-4-E2B-it.litertlm` | 2.59 GB |

Note: `llama-bench` prints all variants as "gemma4 E2B Q4_0"; rows were mapped to files by the reported size (2.43 / 2.02 / 2.82 / 3.10 GiB).

## Comparison table

Prefill = `llama-bench` pp512; decode = tg128 (mean ± σ). LiteRT-LM numbers are the runner's printed BenchmarkInfo for the same Korean prompt (short, 25 tok) and a 1077-token long Korean prompt.

| Model × backend | File size | Prefill tok/s | Decode tok/s | Korean quality |
|---|---:|---:|---:|---|
| GGUF Q4_K_XL — CPU (16t) | 2.44 GB | 415.4 ± 9.6 | 15.6 ± 3.9 | Correct ("대한민국의 수도는 서울입니다"), coherent thinking trace |
| GGUF Q4_K_XL — CUDA | 2.44 GB | 16 622.6 ± 1 855.2 | 243.1 ± 7.9 | Correct, coherent |
| GGUF Q2_K_XL — CPU (16t) | 2.04 GB | 291.8 ± 80.3 | 28.2 ± 1.1 | Correct answer; coherent (fastest CPU decode — smallest weights) |
| GGUF Q2_K_XL — CUDA | 2.04 GB | 652.3 ± 47.7 (!) | 34.3 ± 8.4 (!) | Correct; **GPU anomaly** — see findings |
| GGUF Q4_0 non-QAT — CPU (16t) | 2.83 GB | 375.4 ± 15.5 | 17.3 ± 1.4 | Correct, coherent |
| GGUF Q4_0 non-QAT — CUDA | 2.83 GB | 17 830.7 ± 3 978.4 | 244.8 ± 7.4 | Correct, coherent |
| GGUF Q4_0 Google QAT — CPU (16t) | 3.12 GB | 343.5 ± 63.7 | 16.2 ± 2.3 | Correct, coherent |
| GGUF Q4_0 Google QAT — CUDA | 3.12 GB | 16 065.4 ± 2 141.5 | 241.3 ± 15.5 | Correct, coherent |
| litertlm — CPU (short prompt) | 2.59 GB | 62.4 (25 tok) | 15.1 | Correct: "대한민국의 수도는 서울입니다." TTFT 0.47 s |
| litertlm — CPU (1077-tok prompt) | 2.59 GB | 342.6 | 13.6 | Coherent long-context Korean; TTFT 3.22 s |
| litertlm — GPU | 2.59 GB | — | — | **Fails** (3/3 attempts, no output; see findings) |

llama-cli quality-smoke throughput (short prompts, single run — noisy, listed for reference): CPU generation 16.1–26.6 t/s across variants; CUDA generation 201–222 t/s except Q2_K_XL at 44.6–49.4 t/s.

## Findings

1. **Q2_K_XL GPU anomaly.** With `-ngl 99` the Q2_K_XL file benches at only 652 pp / 34 tg (vs ~16–18k pp / ~243 tg for every other variant) and llama-cli generation tops out at ~45–49 t/s. The behavior is consistent with the UD-Q2_K_XL tensor mix not being fully offloadable/accelerated by this CUDA build, so much of the graph effectively runs on CPU. Q2_K_XL is only interesting as the smallest-footprint CPU option (best CPU decode at 28.2 t/s).
2. **litertlm GPU does not run on this Windows build.** All three `--backend=gpu` attempts produced empty stdout. The log shows the WebGPU delegate initializing successfully on the RTX 4090 via Direct3D 12 ("Created a WebGPU environment", "Failed to create OpenCL context" is informational), then the process dies silently during weight upload (`# of threads to upload weights = 2` is the last line). Reproduced 3/3 (`results/litertlm-gpu*.err`). Numbers are from the May 8 exe; retest when the v0.14 Windows exe lands.
3. **CPU-to-CPU the two stacks are comparable.** litertlm CPU: 342.6 tok/s prefill / 13.6 tok/s decode on a 1077-token prompt vs llama.cpp Q4 CPU: 343–415 pp512 / 15.6–17.3 tg128. Same ballpark; llama.cpp's Q2_K_XL pulls ahead on decode (28.2) at a quality/size trade.
4. **Korean quality is fine everywhere.** Every variant (including Q2_K_XL) answered the Korean prompt correctly and idiomatically; Gemma 4's thinking trace is emitted in English, final answer in Korean. English smoke (gimbal) also coherent on all variants.
5. **VRAM footprint is small.** llama.cpp memory breakdown for Q4_K_XL @ full offload: ~2.2 GiB on CUDA0 (1223 model + 780 context + 242 compute MiB) — trivial for a 24 GB card.

## Verdict

On Windows desktop, **GGUF + llama.cpp is decisively ahead**: CUDA offload delivers ~243 tok/s decode and 16–18k tok/s prefill — roughly **16× the decode speed** of LiteRT-LM, whose GPU path crashes outright on the current Windows runner and whose CPU path (13–15 tok/s decode) merely matches llama.cpp's CPU mode. llama.cpp also offers a real quantization menu (Q4_K_XL is the sweet spot: same speed as Q4_0 at 0.4–0.7 GB smaller; Q2_K_XL only for CPU-bound small-footprint cases) and mature tooling (`llama-bench`, `llama-cli`, server). LiteRT-LM's value remains its Android/mobile deployment story and the single `.litertlm` artifact shared with the Unity Android runtime — not Windows inference performance. For any Windows-hosted evaluation, demo, or data-generation work with Gemma 4 E2B, use llama.cpp (Q4_K_XL on GPU); keep `.litertlm` for device targets, and re-test its GPU backend when the new v0.14 Windows exe is deployed.
