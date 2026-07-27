# LiteRT-LM for Unity

On-device AI for Android in Unity — LLM chat, speech recognition, image
understanding and function calling, all running on the device with no network.
Wraps [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) v0.14.0 through a
native Android bridge.

Device-verified on Snapdragon 865 / 7.5 GB RAM / Android 12: all four
capabilities PASS across 6 PDCA cycles, 80+ runs, zero crashes.

- Minimum Unity **2022.3**; developed and device-verified on 6000.4.6f1
- Android is the target platform. A Windows editor path is included for
  pre-device verification only
- Models are **not** bundled — see the
  [repository README](https://github.com/Leuconoe/LiteRT-LM-Unity#readme) for
  the download tables

## Install

Package Manager → **Add package from git URL…**

```
https://github.com/Leuconoe/LiteRT-LM-Unity.git?path=/Packages/com.leuconoe.litert-lm-unity
```

Or add to `Packages/manifest.json`:

```json
"com.leuconoe.litert-lm-unity": "https://github.com/Leuconoe/LiteRT-LM-Unity.git?path=/Packages/com.leuconoe.litert-lm-unity"
```

Pin a release by appending `#v0.14.0a-unity`.

## Sample

Package Manager → LiteRT-LM for Unity → **Samples** → *Test Scenes* → Import.
Six scenes plus the Android build menu (`LiteRT-LM/Android/...`) and the scene
generator (`LiteRT-LM/Test Scenes/Generate All`).

Scene paths are resolved from wherever the sample was imported, so the version
folder in `Assets/Samples/...` does not need patching.

Each sample scene shows a **◀ Prev / Next ▶** bar in the top-right, so the whole
set can be walked through on a device. Switching scenes calls
`LiteRtLmModelMemory.ReleaseAll()` first: a LiteRT-LM engine holds native memory
the garbage collector does not track, so the outgoing model is disposed before
the next one loads instead of both being resident. Your own components can join
that by implementing `ILiteRtLmModelHost`.

## API

```csharp
using LiteRTLM.Unity;

var client = new LiteRtLmUnityClient();
client.Initialize(modelPath, backend: "CPU");          // vision/audio backends for multimodal
string reply = client.SendMessage("Hello");
string caption = client.SendMessageWithMedia("Describe this", imagePath: path);

// ASR — mel bins, vocab and window frames are detected from the model signature
string json = client.RunWhisperAsrSmoke(
    modelPath, audioPath, tokenizerJsonPath,
    language: "ko", task: "transcribe", vadMode: "energy");
```

`LiteRtLmMicVadCapture` adds live microphone capture with VAD endpointing and a
continuous always-listening mode; `LiteRtLmStatusHudOverlay` draws runtime
status. On Windows the editor uses `LiteRtLmWindowsCliClient`, which falls back
from GPU to CPU automatically.

Backends: use **CPU** for chat decode and **GPU** for long-prompt prefill and
images. On Adreno, GPU decode is slower than CPU — this is a per-step dispatch
cost, not a misconfiguration.

## Third-party components and model licenses

See [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
