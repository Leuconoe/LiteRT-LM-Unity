# Release Notes

## v0.1.0 - Initial Unity Editor test release

Date: 2026-04-24

This is the clean first release of the LiteRT-LM Unity integration. The release
keeps the repository history as a single `first commit` and includes the Unity
Editor validation path used to verify Windows Editor behavior.

### Included

- Unity sample scene and controller for running LiteRT-LM from the Editor.
- Windows CLI fallback client for Unity Editor execution.
- PowerShell sample runner for `litert_lm_main.windows_x86_64.exe`.
- Batchmode Unity Editor self-test launcher.
- Conversation test scene with 10 prompts covering short prompts, Korean input,
  mixed-language prompts, longer diagnostic prompts, and context recall.
- Status-file and Unity Console debug logging for self-test progress.
- UTF-8 stdout/stderr handling for Korean text and emoji output.
- IME-aware IMGUI text fields for Korean prompt input.

### Repository Contents

- `Assets/StreamingAssets/model.litertlm` is committed as the small test model.
- `Assets/StreamingAssets/model.litertlm.meta` is committed for Unity asset
  stability.
- Other `StreamingAssets` files are ignored by default, including downloaded
  models, generated cache files, and runtime cache metadata.
- Generated `.xnnpack_cache` files are not part of this release.

### Validation

Validated in Unity `6000.4.4f1` using the Windows Editor batchmode test path.

```powershell
.\Tools\Windows\Run-LiteRtLmEditorSelfTest.ps1 `
  -MaxAttempts 1 `
  -ExecuteMethod 'LiteRTLM.Unity.Editor.LiteRtLmBuild.RunWindowsConversationSceneTestBatchmode' `
  -StatusRelativePath 'Builds\Logs\LiteRtLmConversationTest.status.txt' `
  -TestName 'Unity conversation scene test'
```

Result:

- Unity domain reload completed before the scene test.
- 10/10 conversation turns completed.
- Final context recall returned `LRT-CTX-042`.
- Unity process exit code was `0`.

### Notes

- The Windows Editor path starts the CLI process through the PowerShell wrapper
  for stability.
- Conversation continuity in the Editor fallback path is maintained by injecting
  summarized prior turns into each subsequent prompt.
- Large local models such as `gemma-4-E2B-it.litertlm` can be used for manual
  testing, but they are intentionally not committed.
