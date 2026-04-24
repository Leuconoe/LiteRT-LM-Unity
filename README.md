# LiteRT-LM-Unity

Unity integration sample for running LiteRT-LM from a Unity project.

This project includes a Unity Editor sample flow, a Windows CLI fallback path,
and batchmode self-tests that verify the Editor integration without relying on
manual UI steps.

## Requirements

- Unity `6000.4.4f1`
- Windows for the included Editor CLI fallback scripts
- PowerShell

## Included

- `Assets/Scenes/LiteRtLmSampleScene.unity`
  - Manual Editor sample scene.
- `Assets/Scenes/LiteRtLmConversationTestScene.unity`
  - Automated 10-turn conversation test scene.
- `Assets/Scripts/LiteRTLM/LiteRtLmSampleController.cs`
  - IMGUI sample UI with IME-aware prompt input.
- `Assets/Scripts/LiteRTLM/LiteRtLmWindowsCliClient.cs`
  - Windows Editor CLI fallback client.
- `Tools/Windows/Run-LiteRtLmSample.ps1`
  - Stable wrapper around `litert_lm_main.windows_x86_64.exe`.
- `Tools/Windows/Run-LiteRtLmEditorSelfTest.ps1`
  - Unity batchmode self-test runner.

## Model Files

`Assets/StreamingAssets/model.litertlm` is committed as the small test model so
the Editor self-test can run after checkout.

Other `Assets/StreamingAssets` files are local artifacts and are ignored by
default. This includes downloaded models, generated `.xnnpack_cache` files, and
runtime cache metadata.

For manual testing with a larger model, place it under `Assets/StreamingAssets`
locally and select it in the sample scene. Large model files are intentionally
not committed.

## Run The Editor Self-Test

From the Unity project root:

```powershell
.\Tools\Windows\Run-LiteRtLmEditorSelfTest.ps1 `
  -MaxAttempts 1 `
  -ExecuteMethod 'LiteRTLM.Unity.Editor.LiteRtLmBuild.RunWindowsConversationSceneTestBatchmode' `
  -StatusRelativePath 'Builds\Logs\LiteRtLmConversationTest.status.txt' `
  -TestName 'Unity conversation scene test'
```

The test performs a Unity domain reload, opens the conversation test scene, and
runs 10 prompts covering short prompts, Korean input, mixed-language prompts,
longer diagnostic prompts, and context recall.

Expected result:

- Unity process exits with code `0`.
- `Builds/Logs/LiteRtLmConversationTest.status.txt` ends with `SUCCESS`.
- The final context recall response includes `LRT-CTX-042`.

## Notes

- The Windows Editor path starts the CLI process through PowerShell for process
  and encoding stability.
- Korean prompt input in the sample UI uses IME-aware text fields.
- UTF-8 stdout and stderr handling is enabled for Korean text and emoji output.
