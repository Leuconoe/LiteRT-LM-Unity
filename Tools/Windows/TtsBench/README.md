# Supertonic TTS bench and conversion tools

Desktop-side scripts for the TTS work: converting the Supertonic ONNX graphs to
tflite, quantizing them, running them on LiteRT, and judging the result. Driven
from the PowerShell wrappers in `..` rather than invoked directly:

| Wrapper | What it does |
| --- | --- |
| `Convert-SupertonicToLiteRt.ps1` | `-Bootstrap` (venv), `-Describe`, `-Convert`, `-Run -RoundTrip` |
| `Run-SupertonicTts.ps1` | synthesize through onnxruntime (the reference path) |
| `Optimize-SupertonicLiteRt.ps1` | flow-step sweep with quality measurement |
| `Deploy-SupertonicLiteRt.ps1` | stage the bucket ladder into StreamingAssets |
| `Test-SupertonicTextParity.ps1` | C# front end vs the Python reference |
| `Compare-SupertonicDeviceBackends.ps1` | device CPU vs GPU |

## Scripts

- `convert_supertonic_to_tflite.py` — onnx2tf per graph, output-checked against
  onnxruntime; `--text-len/--latent-len` freeze shapes for the fast bucketed build.
- `quantize_supertonic_tflite.py` — ai-edge-quantizer; `w8` (weight-only int8) is
  the shipping recipe, `i8` (dynamic) wrecks the vocoder.
- `supertonic_litert.py` — the pipeline on LiteRT, with the NCW/NWC layout map and
  automatic bucket selection.
- `compare_supertonic_runtimes.py` — LiteRT vs onnxruntime, stage by stage.
- `bench_supertonic_graph.py` — one graph across delegate/thread/precision options.
- `spectral_distance.py` — log-mel L1 between renderings. **Round-trip ASR alone
  cannot judge TTS quality**; it passes on audio that is audibly degraded.
- `dump_supertonic_input_md5.py` — desktop checksums of every fed tensor, to
  compare against the device. A wrong tensor *layout* passes every size check, so
  this is the only cheap way to find one.
- `dump_reference_text_ids.py` — ground truth for the C# front-end parity test.
- `check_tflite_runs.py` — does a converted graph allocate and invoke, with and
  without XNNPACK.

Findings and measurements live in [docs/tts-model-research.md](../../../docs/tts-model-research.md).

## Third-party code

`supertonic_helper.py` is vendored verbatim from
[supertone-inc/supertonic](https://github.com/supertone-inc/supertonic)
(`py/helper.py`), **MIT licensed**, © 2025 Supertone Inc. It is kept here because
it defines the reference pipeline — text normalization, latent sampling, mask
construction — that every other script and the C# port are checked against. The
upstream repository was announced for archival on 2026-07-23, which is the other
reason not to depend on fetching it.

The Supertonic *model weights* are separate and carry **OpenRAIL-M**, not MIT.
Commercial use is permitted and accepted for this project (2026-07-29); shipping
the weights still requires including the licence with them, passing the use
restrictions downstream, and disclosing machine-generated audio. See the licence
discussion in the doc above.
