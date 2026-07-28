# Sample scene session handoff

State at the end of the session. Nothing is committed — `HEAD` is still
`5bd24560`, and all work below sits in the working tree (43 tracked files
changed, +2,161 / −4,329; 20 new untracked source files under
`Packages/com.leuconoe.litert-lm-unity/Samples~/`). Committing was deliberately
left to the repository owner.

## What was done

The eleven reported sample-scene defects (ten numbered plus layout unification)
are implemented and were verified by running each scene, not by reading code.
Per-item cause, change and evidence:
[sample scene rework](sample-scene-rework-verification.md).

Two verification artifacts exist and are re-runnable:

| What | Where |
| --- | --- |
| Static invariants, 65 checks | Unity menu `LiteRT-LM/Verify Sample Scenes` → `Builds/Logs/SampleSceneVerification.txt` |
| Runtime sweep, 10 scenes played | `Builds/Logs/SampleSceneRuntimeVerification.txt` |
| Screenshots | `Builds/Logs/SceneShots/` |

Roslyn compiles both sample assemblies clean (exit 0, warnings only); the Unity
console was clear at the end of the last run.

### Late changes not covered by the rework document

- **ASR continuous transcription now works on Windows.** `ContinuousWorkerRoutine`
  gated on `_client.IsAvailable` (the Android bridge), so a desktop session
  captured WAVs and transcribed none of them. It now falls back to the CLI, and
  `WarmUpBridge()` is skipped when there is no bridge. Verified by feeding a real
  utterance into the capture handler: `#1 (1.08s) transcript=화면 밝게,
  latency=1.37s, desktop CLI`.
- **ASR model dropdown says what it actually does.** Whisper tflite tiers run on
  the Android bridge only; the desktop CLI transcribes with gemma-4 audio. The
  scene silently ignored the selection, which read as a bug ("picked ACFT-ko, got
  gemma-4"). A line under the dropdown now states it.
- **Windows default backend is CPU** (`windowsBackend`), Android stays GPU.
  Measured with `litert_lm_main`, gemma-4-E2B, one process per request:

  | | CPU | GPU |
  | --- | ---: | ---: |
  | Init executor | 305 ms | 5,097 ms |
  | Time to first token | 0.49 s | 2.38 s |
  | Prefill | 43.9 tok/s | 7.6 tok/s |
  | Decode | 13.2 tok/s | 53.3 tok/s |

  The CLI is stateless, so each request repays executor init — roughly 4.8 s that
  only ~84 output tokens of faster decode would recover. Chat turns and FC
  routing are far shorter, and prefill is 6× slower on GPU here. OpenCL context
  creation fails on this build, so GPU runs through WebGPU (Dawn → D3D12) rather
  than the native path Android uses. Recorded in
  [LLM details](../llm-details.md) §3.
- **Voice FC prompt gained worked examples.** With prose rules alone the 0.6B
  router answered `{"tool":"None"}` to `소리 키워 줘`; with examples it returns
  `SetVolume(direction=up, amount=70)`. Non-commands still correctly return
  `None`.
- **One stale asset reference fixed.** The ASR audio catalogue named
  `현재 서울의 날씨는, 흐림. 입니다.mp3`, but the file on disk had been renamed to
  `현재 서울의 날씨는 흐림 입니다.mp3`, so selecting that entry failed. All 114
  StreamingAssets references across the samples were then audited — none missing.

### Model release on scene switch

Confirmed present and running. `LiteRtLmSceneNavigator` calls
`LiteRtLmModelMemory.ReleaseAll()` before `SceneManager.LoadScene`, which invokes
`ReleaseModels()` on every `ILiteRtLmModelHost` in the scene. Observed:

```
[LiteRT-LM] Loading 'LiteRtLmMultimodalTestScene'; released 1 model host(s) first.
```

The conversation and benchmark scenes have no host to release — they shell out to
the CLI rather than holding a native engine.

## Open question — ANSWERED 2026-07-27

**Can Whisper tflite run on Windows? Yes.** 20 of 20 transcriptions across five
exports and four clips, on CPU through `ai_edge_litert`. Full results, timings
and caveats: [Whisper tflite on Windows](../benchmarks/whisper-windows-tflite.md).

The driver no longer lives in untracked scratch. It is
`Tools/Windows/WhisperTflite/whisper_tflite_runner.py`, driven by
`Tools/Windows/Run-WhisperTfliteWindows.ps1` (`-Sweep` reproduces the table,
`-Bootstrap` builds its own venv). The old `External/acft-work/bench_acft.py`
was the only copy and is not in git.

Two findings came out of it:

- The 128-mel turbo export transcribes correctly on desktop, agreeing with the
  device: the old `TensorBuffer 65536 vs 7680000` failure was take3-era and was
  fixed by take4/take5 (dynamic mel+vocab, shape-based decode binding).
- The 0.79 s `볼륨 업` clip fails on base and tiny (`볼륨어`, `보일해봐`) and
  passes on medium and turbo, reproducing the device-side tier boundary on
  desktop.

Unchanged: `litert_lm_main` still cannot do this — it is an LLM runner, while
whisper needs a mel frontend, an encoder/decoder pair and a KV loop driven from
outside. Shipping Whisper tflite from Unity on Windows still means porting that
driver out of the AAR into C#/C++ against `libLiteRt.dll`. Python is a bench
harness, not a runtime. The ASR scene's desktop path therefore still uses
gemma-4 audio.

## Also outstanding

Linux/macOS binary support (task #13). Needs a Mac; untouched.

## Notes for whoever picks this up

These, and the AAR/device/artifact traps from earlier sessions, are now collected
in `CLAUDE.md` at the repository root so they are read before the next mistake
rather than after.

- Sample sources live in `Samples~/`, which Unity does not compile. After editing
  them run `.\Tools\Windows\Restore-LiteRtLmSamples.ps1 -Force`, then let Unity
  reimport, or the editor keeps running the previous assembly. Several
  false "the fix didn't work" moments in this session traced back to that.
- Do not run `AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport)`
  here: StreamingAssets holds multi-GB models and it reimports them, which hangs
  the editor and the MCP bridge for a long while. A plain `Refresh()` is enough.
- `git add -A` at the repository root would stage the untracked `External/`
  working directories. Stage explicit paths instead.
