# Working notes for this repository

Traps that have already cost time here, and the conventions that avoid them.
Read before editing samples, building the AAR, or trusting a benchmark number.

## Unity samples

- **`Packages/com.leuconoe.litert-lm-unity/Samples~/` is not compiled by Unity.**
  Editing a file there changes nothing in the editor until you run
  `.\Tools\Windows\Restore-LiteRtLmSamples.ps1 -Force` and let Unity reimport.
  Several "the fix didn't work" investigations have traced back to this — if a
  change appears to have no effect, check this first.
- **Never call `AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport)`.**
  StreamingAssets holds multi-GB models; it reimports them and hangs the editor
  and the MCP bridge. A plain `Refresh()` is enough.
- Scenes are generated, not hand-authored: menu `LiteRT-LM/Test Scenes/Generate All`.
  Static invariants: menu `LiteRT-LM/Verify Sample Scenes` → `Builds/Logs/SampleSceneVerification.txt`.
- **A generated scene is not saved until you sync it back.** Generators write into
  the imported copy under `Assets/Samples/…`, which the next
  `Restore-LiteRtLmSamples.ps1 -Force` overwrites from `Samples~` — deleting any
  scene that only ever existed in Assets. Run
  `.\Tools\Windows\Sync-LiteRtLmGeneratedScenes.ps1` after generating, then commit.
- If a generator fails with *"Overwriting the same path as another open scene is
  not allowed"*, the editor is holding that scene (often one whose asset a restore
  just deleted). Open an empty scene first, then regenerate.
- Asset paths in sample code are string literals against StreamingAssets. Renaming
  a model or clip silently breaks the entry that names it — re-run the scene, not
  just the compiler.

## Android

- The physical test device is serial **46a880a0** (kona / SM8250, METALENSE2,
  Android 12). An emulator is usually attached too — always pass
  `-DeviceSerial 46a880a0`.
- kona is not in the litert-community per-SoC NPU lists (those start at SM8450):
  CPU and GPU (OpenCL) paths only.
- The device idles around 53 °C while the benchmark script waits for ≤45 °C, so
  the thermal gate never clears. Use `-SkipThermalWait` and record temperatures.
- **`Build-LiteRtLmUnityAarFromPatch.ps1 -SkipImageBuild` builds from source baked
  into the Docker image.** After native edits, rebuild the image (omit the flag)
  or use a warm container with `-SyncWorktree`; otherwise you silently ship the
  previous AAR.
- **`-SyncWorktree` alone does nothing.** The sync only runs inside the
  reuse-container branch, so it needs `-ReuseContainer <name>` as well. Without
  one, `-SkipImageBuild -SyncWorktree` builds happily from the stale baked source
  and reports success — verify instead of trusting it:
  `python -c "import zipfile;print(b'<newSymbol>' in zipfile.ZipFile('Packages/com.leuconoe.litert-lm-unity/Runtime/Plugins/Android/litertlm-unity-bridge.aar').read('jni/arm64-v8a/liblitertlm_jni.so'))"`.
- The AAR patch is regenerated from `External/LiteRT-LM` with
  `git diff -- . ':(exclude)cxxbridge_cmd/Cargo.lock'` (the Cargo.lock churn is
  noise). Compare the file list against the previous revision before replacing
  `Tools/UnityAar/litert-lm-unity-aar.patch` — a dropped file fails much later.
- Always rebuild *and reinstall* the APK after an AAR change. A stale APK with a
  new model routes to the wrong native path and dies with SIGSEGV; look for
  `files/tombstone_*` in the app directory.

## Running the tooling

- The `Tools/Windows/*.ps1` scripts are unsigned, so `powershell.exe` launched from
  bash refuses them (`PSSecurityException`). Run them through the PowerShell tool
  directly, or add `-ExecutionPolicy Bypass` when shelling out.
- Batchmode Unity (`Unity.exe -batchmode -projectPath …`) is refused while the
  editor has the project open. Either close the editor or schedule the work inside
  it; `EditorApplication.delayCall` is dropped by a domain reload, so hook
  `EditorApplication.update` and unsubscribe on the first tick.
- Android device runs need the app to hold window focus. Unity pauses otherwise
  and a headless runner writes *no* status file at all, which reads as a hang —
  check `adb shell dumpsys window | grep mCurrentFocus`.

## Results and artifacts

- **`External/` is untracked scratch.** Nothing under it is in git and it has been
  deleted wholesale before (87 GB of training artifacts, 2026-07-26). If a result
  matters, the driver belongs in `Tools/` and the numbers in `docs/`. Do not leave
  the only copy of a working script in `External/`.
- Do not run `git add -A` at the repository root — it stages the untracked
  `External/` working directories. Stage explicit paths.
- `Builds/` is gitignored; logs there are evidence for the current session only.
- Docs convention: the README stays short and Android-first, all detail goes in
  `docs/`. Desktop numbers are reference only.
- Standing caveat for ASR claims: the 40-clip ACFT gate set is TTS-synthesized —
  valid for ranking (ρ=1.00 vs real recordings), ~2.8× optimistic in absolute CER.

## Python tooling

- Use `.\Tools\Windows\Run-WhisperTfliteWindows.ps1` to run Whisper tflite on the
  desktop; `-Bootstrap` builds its own venv. `litert_lm_main` cannot do this — it
  is an LLM runner and does not drive an encoder/decoder pair.
- The `py` launcher defaults to Python 3.14 on this machine, which has no numpy
  wheel; building it from source fails. Bootstrap targets 3.12/3.13 deliberately.

## Windows LLM CLI

- `Tools/Windows/litert_lm_main.windows_x86_64.exe` is a prebuilt binary with
  custom flags from `Tools/UnityAar/litert-lm-unity-aar.patch`; its source is not
  committed. Rebuilds must pin VS2022 via `BAZEL_VC` — VS2026 / MSVC 14.51
  miscompiles litert.
- The default Windows backend is CPU. OpenCL context creation fails on this build,
  so GPU runs through WebGPU (Dawn → D3D12): prefill is ~6× slower and the CLI is
  stateless, so each request repays a ~5 s executor init. See
  [docs/llm-details.md](docs/llm-details.md) §3.
