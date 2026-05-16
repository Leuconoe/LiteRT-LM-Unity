# LiteRT-LM-Unity

Unity integration sample for LiteRT-LM on Windows Editor and Android.

## Requirements

- Unity `6000.4.6f1`
- Windows and PowerShell for Editor and Android build scripts
- Unity Android Build Support with SDK/NDK
- Android device with `adb` access for hardware tests
- Docker Desktop and Git for Windows Bash only when rebuilding the custom Android AAR

## Recommended LLM Models

| Rank | Model | Source | Note |
| ---: | --- | --- | --- |
| 1 | `gemma-4-E2B-it.litertlm` | [litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) | Primary quality model when memory allows. |
| 2 | `Qwen2.5-0.5B-Instruct-q8.litertlm` | [litert-community/Qwen2.5-0.5B-Instruct](https://huggingface.co/litert-community/Qwen2.5-0.5B-Instruct) | Fast CPU fallback for smaller memory and low latency. |
| 3 | `gemma3-1b-it-int4.litertlm` | [litert-community/Gemma3-1B-IT](https://huggingface.co/litert-community/Gemma3-1B-IT) | Compact Android native-GPU-capable fallback. |

## Recommended ASR Models

ASR models need a matching tokenizer directory under `Assets/StreamingAssets`.
For example, Whisper Tiny uses `whisper-tiny/tokenizer.json`, and Whisper Base
uses `whisper-base/tokenizer.json`. Use the tokenizer source links below when
restoring those tokenizer folders.

| Rank | Model | Model source | Tokenizer source | Note |
| ---: | --- | --- | --- | --- |
| 1 | `whisper_tiny_30s_i8.tflite` | [litert-community/whisper-tiny](https://huggingface.co/litert-community/whisper-tiny) | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) | Recommended Korean/English ASR model for Android CPU use. |
| 2 | `whisper_base_30s_f32.tflite` | [litert-community/whisper-base](https://huggingface.co/litert-community/whisper-base) | [openai/whisper-base](https://huggingface.co/openai/whisper-base) | Quality comparison only; it is large and not the default recommendation. |

## Details

- LLM 세부 설명: [`docs/llm-details.md`](docs/llm-details.md)
- ASR 세부 설명: [`docs/asr-details.md`](docs/asr-details.md)
