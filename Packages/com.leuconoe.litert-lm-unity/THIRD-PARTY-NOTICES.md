# Third-party notices

**This package is licensed under Apache-2.0** ([`LICENSE.md`](LICENSE.md)). The
components below are embedded or built upon; the licenses here apply to them,
not to the integration code.

| Component | Where | License |
| --- | --- | --- |
| [google-ai-edge/LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) v0.14.0 | Compiled into `Runtime/Plugins/Android/litertlm-unity-bridge.aar` | Apache-2.0 |
| LiteRT / TensorFlow Lite runtime | Linked inside the same AAR | Apache-2.0 |
| [supertone-inc/supertonic](https://github.com/supertone-inc/supertonic) `py/helper.py` | Vendored at `Tools/Research/Supertonic/TtsBench/supertonic_helper.py` (repository only, not in the package) | MIT, © 2025 Supertone Inc. |

The AAR and the Windows executables are **derivative works of LiteRT-LM**, built
from its sources with `Tools/UnityAar/litert-lm-unity-aar.patch` applied. Apache-2.0
was chosen for this project so that the integration code and those binaries carry
one consistent license rather than two.

Models are **not** shipped with this package. When you download them, the model
licenses apply separately:

| Model family | License |
| --- | --- |
| OpenAI Whisper (`openai/whisper-*`, and every quantized or ACFT derivative) | MIT |
| Qwen2.5 / Qwen3 / Qwen3-ASR | Apache-2.0 |
| Gemma / gemma-4 | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |
| LiquidAI LFM2.5 | See the model card |
| Silero VAD (`pat229988/silero-vad-16k-tflite`) | MIT |

The audio-context fine-tuning method used for the ACFT models comes from
[futo-org/whisper-acft](https://github.com/futo-org/whisper-acft) (MIT). Only
the method is used here; the training scripts are not distributed.

**Model weights are never redistributed by this project.** StreamingAssets is
gitignored and the package ships no `.litertlm`, `.tflite` or voice file. Two
model licenses need a decision rather than a glance before you ship the download
you chose:

- **Supertonic TTS weights are OpenRAIL-M.** Commercial use is permitted — the
  restrictions are on use cases, not commerce — but it is a carry-forward
  license: ship the license text with the weights, pass the use restrictions to
  whoever receives them, and disclose that the audio is machine-generated
  (restriction (e); `LiteRtLmTtsDisclosure` exists for this). See
  [`docs/tts-details.md`](https://github.com/Leuconoe/LiteRT-LM-Unity/blob/main/docs/tts-details.md).
- **Gemma models** are under the Gemma Terms of Use, not a standard open license.
