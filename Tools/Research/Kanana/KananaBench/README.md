# KananaBench — scoring candidate LLMs that LiteRT cannot run yet

The FC benchmark in `docs/benchmarks/fc-model-benchmark.md` is driven by
`litert_lm_main`, which only loads `.litertlm` bundles. A model with no
conversion path therefore cannot be scored at all — which is exactly when you
most want a number, because the conversion is the expensive part.

This driver closes that gap. It asks the **same 20 Korean routing questions**
with the **same tools JSON** and the **same grading rules**, on the desktop GPU
through Hugging Face `transformers`, so a candidate can be ranked against the
existing table before anyone commits to a port.

```powershell
.\Tools\Windows\Run-KananaEval.ps1 -Bootstrap        # first run: build the venv
.\Tools\Windows\Run-KananaEval.ps1                   # kanana-2-1.3b-instruct
.\Tools\Windows\Run-KananaEval.ps1 -Model Qwen/Qwen3-0.6B -Label qwen3-0.6b-bf16
```

Results land in `Builds/Logs/kanana-fc-bench.jsonl`, one JSON object per case
plus a summary record.

## Read the numbers correctly

- **The pass rate transfers. The latency does not.** Timings here are an
  RTX 4090 in bf16 with a stock `generate()` loop, and say nothing about kona.
  For device throughput a model has to be converted and measured on device.
- Each model is allowed **its own tool-call format**. The driver reads the
  OpenAI-style JSON block, the Hermes `<tool_call>` envelope and the
  qwen3-coder `<function=Name>` form, because a model should not be marked
  wrong for answering in its native syntax. The standing caveat in the
  benchmark doc — that a low score often means a format mismatch rather than a
  misunderstanding — applies here too, so read the `raw` field before
  concluding anything.
- Greedy decoding, matching the deterministic-router requirement.

## Keeping it honest

`fc_cases.json` and `fc_tools.json` are ported verbatim from
`Samples~/AutomatedTests/Runtime/Benchmark/LiteRtLmFunctionCallingBenchmarkRunner.cs`.
They are duplicated rather than generated, so **if you change the cases in the
C# runner, change them here too** — otherwise the two scores stop being
comparable and nothing will tell you.

## Licence note for Kanana specifically

Kanana-2 weights are under the **Kanana Open License Agreement**, not a
permissive licence. §4.1(ii) requires a separate commercial licence from Kakao,
at Kakao's sole discretion, to offer the model to third parties "as part of a
system integration (SI) or on-premise deployment solution". Evaluating it is
fine; shipping it in a delivered product is a negotiation. See
`docs/llm-details.md`.
