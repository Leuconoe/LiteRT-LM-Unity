# ASR Details

This document records Android ASR setup and benchmark results for the Unity
LiteRT-LM bridge. The README keeps only requirements and recommended models.

## Setup

- Report date: 2026-05-17
- Unity package: `com.Leuconoe.LiteRTLMUnity`
- Device class: Qualcomm Android 12 physical device, about 7.52 GiB RAM
- Test audio: Korean `2025년 3월 5일 전술평가 결과 보고.mp3`
- Runtime inputs: one ASR smoke-test APK plus model/audio/tokenizer/config files
  pushed to app device storage.

ASR requires a tokenizer that matches the model family. Whisper Tiny uses
`whisper-tiny/tokenizer.json`; Whisper Base uses
`whisper-base/tokenizer.json`.

## Required Assets

| Model | Required files | Model source | Tokenizer source |
| --- | --- | --- | --- |
| Whisper Tiny i8 CPU | `whisper_tiny_30s_i8.tflite`, `whisper-tiny/tokenizer.json` | [litert-community/whisper-tiny](https://huggingface.co/litert-community/whisper-tiny) | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) |
| Whisper Tiny i8 GPU attempt | `whisper_tiny_30s_i8.tflite`, `whisper_tiny_30s_i8_encoder.tflite`, `whisper-tiny/tokenizer.json` | [litert-community/whisper-tiny](https://huggingface.co/litert-community/whisper-tiny) | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) |
| Whisper Tiny f32 GPU split | `whisper_tiny_30s_f32.tflite`, `whisper_tiny_30s_f32_encoder.tflite`, `whisper-tiny/tokenizer.json` | [litert-community/whisper-tiny](https://huggingface.co/litert-community/whisper-tiny) | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) |
| Whisper Base f32 CPU | `whisper_base_30s_f32.tflite`, `whisper-base/tokenizer.json` | [litert-community/whisper-base](https://huggingface.co/litert-community/whisper-base) | [openai/whisper-base](https://huggingface.co/openai/whisper-base) |

## Benchmarks

### Latest Results

The current run tested the `.tflite` ASR files that were present in
`Assets/StreamingAssets`. CPU and GPU were both attempted for each full ASR
model.

| Model | Backend requested | Backend used | Status | File size | Compile s | Encode s | Decode s | Elapsed s | Transcript / failure |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `whisper_tiny_30s_i8.tflite` | CPU | CPU | PASS | 41.12 MB | 0.079 | 0.333 | 1.028 | 1.629 | `2025년 3월 5일 전술 평가 결과보고` |
| `whisper_tiny_30s_i8.tflite` | GPU | GPU encoder + CPU decoder | FAIL | 41.12 MB + 33.15 MB encoder | N/A | N/A | N/A | N/A | GPU encoder companion was found, but encoder compilation failed: `Failed to compile model`. |
| `whisper_tiny_30s_f32.tflite` | CPU | CPU | PASS | 150.98 MB | 0.092 | 0.462 | 2.147 | 2.899 | `2025년 3월 5일 전술 평가 결과보고` |
| `whisper_tiny_30s_f32.tflite` | GPU_FP16 | GPU encoder + CPU decoder | PASS | 150.98 MB + 32.94 MB encoder | 1.535 first run, 0.000 cached | 0.097 | 1.959 | 2.372 10-run avg | `2025년 3월 5일 전술 평가 결과 보고` |
| `whisper_base_30s_f32.tflite` | CPU | CPU | PASS | 290.08 MB | 0.242 | 1.082 | 4.205 | 5.739 | `2025년 3월 5일 전술평가 결과 보고` |
| `whisper_base_30s_f32.tflite` | GPU | N/A | FAIL | 290.08 MB | N/A | N/A | N/A | N/A | Current GPU split path requires `whisper_base_30s_encoder_f32.tflite`, but that companion is not present in `StreamingAssets`. |

### Repeat Stability

The same Korean audio was repeated 10 times on the physical test device with
`whisper_tiny_30s_f32.tflite`.

| Backend requested | Backend used | Runs | Avg compile s | Avg encode s | Avg decode s | Avg elapsed s | Transcript |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| CPU | CPU | 10 | 0.095 | 0.491 | 2.145 | 2.919 | `2025년 3월 5일 전술 평가 결과보고` |
| GPU_FP16 | GPU encoder + CPU decoder | 10 | 0.153 | 0.097 | 1.959 | 2.372 | `2025년 3월 5일 전술 평가 결과 보고` |

The GPU encoder is about 5x faster than CPU encoding for this clip, but the
native smoke path now keeps compiled encoder and decoder models alive while the
process is running. In the 10-run GPU_FP16 test, run 1 compiled the models
(`compiledModelCache=miss`, 1.535 s compile) and runs 2-10 reused them
(`compiledModelCache=hit`, 0.000 s compile). This makes the cached GPU split
path faster than the CPU path for repeated utterances on the same process.

### GPU Notes

The Unity ASR GPU path is split execution: the encoder is compiled for GPU and
the decoder runs on CPU. The full Whisper model alone is sufficient for CPU, but
GPU currently needs a matching encoder companion next to the full model.

The `whisper_tiny_30s_f32_encoder.tflite` companion used for the successful
Whisper Tiny f32 GPU test is a project-generated split encoder artifact. It is
not part of the upstream `litert-community/whisper-tiny` model files. Keep it
with the full f32 model when testing Unity GPU ASR. The Unity runner creates a
legacy data-file alias named `whisper_tiny_30s_encoder_f32.tflite` because the
current native AAR still derives that internal name.

`whisper_tiny_30s_i8.tflite` has an i8 encoder companion in the project, but the
Qualcomm device rejected it during GPU compilation. The `litert-community`
Whisper Tiny model card exposes multiple hardware-specific f32 variants and an
i8 full model, but does not document this i8 encoder companion as a guaranteed
GPU path.

`whisper_base_30s_f32.tflite` passed on CPU. The `litert-community/whisper-base`
model card currently provides the full f32 model; no matching encoder companion
was present in the project, so the GPU split path could not start.

### Recommendations

1. Use `whisper_tiny_30s_i8.tflite` on CPU as the smallest default
   Korean/English ASR model.
2. Use `whisper_tiny_30s_f32.tflite` plus
   `whisper_tiny_30s_f32_encoder.tflite` when validating the GPU split path.
   The encoder step is much faster on GPU. The first utterance still pays the
   compile cost, but repeated utterances reuse the compiled encoder and decoder.
3. Use `whisper_base_30s_f32.tflite` only for quality comparison; it is much
   larger and slower than Tiny.

## Smoke Tests

The current ASR runner can reuse a single APK. Push the selected model, audio,
tokenizer, and runtime JSON config into app storage, then launch the same build:

```powershell
.\Tools\Windows\Run-LiteRtLmAndroidAsrSmokeTest.ps1 `
  -DeviceSerial <device-serial> `
  -ApkPath Builds\Android\LiteRtLmAndroidAsrSmokeTest-generic.apk `
  -ModelFileName whisper_tiny_30s_i8.tflite `
  -AudioFileName "2025년 3월 5일 전술평가 결과 보고.mp3" `
  -TokenizerJsonPath "whisper-tiny/tokenizer.json" `
  -AsrMode whisper `
  -AsrLanguage ko `
  -Backend CPU `
  -TimeoutSeconds 300
```

Use `-Backend GPU` only when the matching encoder companion model is present and
known to compile on the target device.
