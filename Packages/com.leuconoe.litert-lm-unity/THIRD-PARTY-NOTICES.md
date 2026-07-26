# Third-party notices

This package embeds or builds on the following components.

| Component | Where | License |
| --- | --- | --- |
| [google-ai-edge/LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) v0.14.0 | Compiled into `Runtime/Plugins/Android/litertlm-unity-bridge.aar` | Apache-2.0 |
| LiteRT / TensorFlow Lite runtime | Linked inside the same AAR | Apache-2.0 |

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

> The Unity integration code in this package does not yet carry a top-level
> license file. Add one at the repository root before redistributing.
