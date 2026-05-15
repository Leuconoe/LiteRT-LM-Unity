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
| 2 | `Qwen2.5-0.5B-Instruct-q8.litertlm` | [litert-community/Qwen2.5-0.5B-Instruct](https://huggingface.co/litert-community/Qwen2.5-0.5B-Instruct) | Fast CPU fallback for smaller memory and lower latency. |
| 3 | `gemma3-1b-it-int4.litertlm` | [litert-community/Gemma3-1B-IT](https://huggingface.co/litert-community/Gemma3-1B-IT) | Compact Android GPU-capable fallback. |

## Recommended ASR Models

| Rank | Model | Source | Note |
| ---: | --- | --- | --- |
| 1 | `whisper_tiny_30s_i8.tflite` | [litert-community/whisper-tiny](https://huggingface.co/litert-community/whisper-tiny) | Recommended Korean/English ASR model for Android CPU use. |
| 2 | `parakeet_tdt_0.6b_v3_5s_i8.tflite` | [litert-community/parakeet-tdt-0.6b-v3](https://huggingface.co/litert-community/parakeet-tdt-0.6b-v3) | English-only ASR smoke model. |
| 3 | `whisper_base_30s_f32.tflite` | [litert-community/whisper-base](https://huggingface.co/litert-community/whisper-base) | Quality comparison only; it is large and not the default recommendation. |

## Details

- ASR benchmarks: [`docs/android-asr-smoke-benchmarks.md`](docs/android-asr-smoke-benchmarks.md)
- Android LLM benchmarks: [`docs/benchmarks/android-device-llm-benchmarks.md`](docs/benchmarks/android-device-llm-benchmarks.md)
- AAR patch/build notes are in the ASR benchmark document.
