# ASR Details

This document records ASR setup, benchmark results, and smoke-test commands for
the Unity LiteRT-LM bridge. The README keeps only requirements and recommended
models.

## Setup

### Current Recommendation

- Default ASR model: `whisper_tiny_30s_i8.tflite` on CPU.
- GPU ASR status: `whisper_tiny_30s_f32.tflite` can use the split path
  (`GPU_ENCODER_CPU_DECODER`) with `whisper_tiny_30s_encoder_f32.tflite`.
- Quantized GPU status: q8/int8 encoder companions currently fail during
  Adreno GPU delegate compilation with `Failed to compile model`.
- Base status: `whisper_base_30s_f32.tflite` runs and transcribes Korean
  correctly on CPU, but it is large and not a default recommendation.
- Parakeet status: `parakeet_tdt_0.6b_v3_5s_i8.tflite` is English-only.

### Required Assets

| Model | Required files | Tokenizer source |
| --- | --- | --- |
| Whisper Tiny CPU | `whisper_tiny_30s_i8.tflite`, `whisper-tiny/tokenizer.json` | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) |
| Whisper Tiny GPU split | `whisper_tiny_30s_f32.tflite`, `whisper_tiny_30s_encoder_f32.tflite`, `whisper-tiny/tokenizer.json` | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) |
| Whisper Base CPU | `whisper_base_30s_f32.tflite`, `whisper-base/tokenizer.json` | [openai/whisper-base](https://huggingface.co/openai/whisper-base) |
| Parakeet | `parakeet_tdt_0.6b_v3_5s_i8.tflite`, `parakeet-tdt-0.6b-v3/tokenizer.json` | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |

The ASR tokenizer path must match the model family. Pass the tokenizer with
`-TokenizerJsonPath` when building a smoke APK.

### Custom AAR

The custom AAR is built from `External/LiteRT-LM` plus
`Tools/UnityAar/litert-lm-unity-aar.patch`. Do not commit direct changes inside
the submodule. Regenerate the AAR after patch changes, then replace:

```powershell
.\temp\Run-ReusableUnityAarBuild.ps1 -ForceSync
Copy-Item -Force temp\docker-bazel-out\aar\litertlm-unity-bridge.aar `
  Assets\Plugins\Android\litertlm-unity-bridge.aar
```

## Benchmarks

### Device

Public device details are limited to chipset and memory.

| Field | Value |
| --- | --- |
| Chipset/platform | Qualcomm `kona` / `qcom` |
| RAM | about 7.52 GiB |
| Package | `com.Leuconoe.LiteRTLMUnity` |

### Latest Results

Korean test audio: `2025년 3월 5일 전술평가 결과 보고.mp3`.

| Case | Language | Backend used | File size | Compile s | Encoder compile s | Encode s | Decode s | Elapsed s | Result |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Whisper Tiny i8 | `ko` | `CPU` | 41.1 MB | 0.079 | N/A | 0.311 | 1.020 | 1.598 | Pass: `2025년 3월 5일 전술 평가 결과보고` |
| Whisper Tiny f32 | `ko` | `CPU` | 151.0 MB | 0.094 | N/A | 0.460 | 2.130 | 2.887 | Pass: `2025년 3월 5일 전술 평가 결과보고` |
| Whisper Tiny q8 experimental | `ko` | `CPU` | 151.9 MB | 0.008 | N/A | 0.621 | 7.564 | 8.396 | Pass, but not recommended |
| Whisper Tiny f32 split | `ko` | `GPU_ENCODER_CPU_DECODER` | 151.0 MB + 32.9 MB | 1.547 | 1.453 | 0.094 | 1.939 | 3.790 | Pass: `2025년 3월 5일 전술 평가 결과 보고` |
| Whisper Tiny q8 split | `ko` | `GPU_ENCODER_CPU_DECODER` | 151.9 MB + 33.1 MB | N/A | N/A | N/A | N/A | N/A | Fail: encoder GPU compile |
| Whisper Tiny int8 split | `ko` | `GPU_ENCODER_CPU_DECODER` | 41.1 MB + 33.1 MB | N/A | N/A | N/A | N/A | N/A | Fail: encoder GPU compile |
| Whisper Base f32 | `ko` | `CPU` | 290.1 MB | 0.171 | N/A | 0.992 | 4.301 | 5.680 | Pass: `2025년 3월 5일 전술평가 결과 보고` |

Earlier English/Parakeet runs are still useful for regression checks:

| Case | Language | Backend used | Audio s | Compile s | Encoder compile s | Encode s | Decode s | Result quality | Status file |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Parakeet English | English only | `GPU_FP16` | 5.904 | 4.094 | N/A | 0.347 | 6.522 | Partial but recognizable: `Evaluation Res Report, March 5 2025` | `20260515-174137-asr-smoke.status.txt` |
| Whisper Tiny English | `auto` | `GPU_ENCODER_CPU_DECODER` | 5.904 | 1.534 | 1.440 | 0.094 | 2.019 | Pass: `Tactical Evaluation results report March 5, 2025.` | `20260515-171922-asr-smoke.status.txt` |

### Notes

- The Whisper GPU path in the current AAR is split execution: encoder on GPU,
  decoder/full model on CPU.
- q8/int8 full models generated through the generic PT2E path are not useful
  for Base. They do not shrink the model because large f32 buffers remain.
- `SM8250` is not listed in the current LiteRT Qualcomm AOT target enum. The
  available Qualcomm targets are `SA8255`, `SA8295`, `SM8350`, `SM8450`,
  `SM8550`, `SM8650`, `SM8750`, and `SM8850`.

## Smoke Tests

Build and run Whisper Tiny i8 CPU:

```powershell
.\Tools\Windows\Build-LiteRtLmAndroidAsrSmokeApk.ps1 `
  -AsrMode whisper `
  -AsrLanguage ko `
  -ModelFileName whisper_tiny_30s_i8.tflite `
  -AudioFileName "2025년 3월 5일 전술평가 결과 보고.mp3" `
  -TokenizerJsonPath "whisper-tiny/tokenizer.json" `
  -Backend CPU `
  -OutputApk LiteRtLmAndroidAsrSmokeTest-whisper-tiny-ko-i8-CPU.apk

.\Tools\Windows\Run-LiteRtLmAndroidAsrSmokeTest.ps1 `
  -DeviceSerial <device-serial> `
  -ApkPath Builds\Android\LiteRtLmAndroidAsrSmokeTest-whisper-tiny-ko-i8-CPU.apk `
  -TimeoutSeconds 600 `
  -ClearAppData
```

Build and run Whisper Tiny f32 GPU split:

```powershell
.\Tools\Windows\Build-LiteRtLmAndroidAsrSmokeApk.ps1 `
  -AsrMode whisper `
  -AsrLanguage ko `
  -ModelFileName whisper_tiny_30s_f32.tflite `
  -AudioFileName "2025년 3월 5일 전술평가 결과 보고.mp3" `
  -TokenizerJsonPath "whisper-tiny/tokenizer.json" `
  -Backend GPU_FP16 `
  -OutputApk LiteRtLmAndroidAsrSmokeTest-whisper-tiny-ko-gpu-split.apk

.\Tools\Windows\Run-LiteRtLmAndroidAsrSmokeTest.ps1 `
  -DeviceSerial <device-serial> `
  -ApkPath Builds\Android\LiteRtLmAndroidAsrSmokeTest-whisper-tiny-ko-gpu-split.apk `
  -TimeoutSeconds 600 `
  -ClearAppData
```
