# Sample scene rework — verification record

Covers the eleven items raised against the sample scenes (ten numbered defects
plus item 0, layout unification). Each row names the change and the runtime
evidence produced by running the scene, not by reading the code.

Environment: Unity 2022.3.62f3, Windows editor, `litert_lm_main` /
`litert_lm_advanced_main` CLI backends, `gemma-4-E2B-it.litertlm`.
Screenshots: `Builds/Logs/SceneShots/`.

## Item 0 — layout unification

One grammar for every scene: a fixed-width control column on the left, output
on the right, and the same widget for the same kind of thing.

| Rule | Widget |
| --- | --- |
| Chosen once per session (model, backend, language, VAD mode) | `LiteRtLmUi.Dropdown` |
| Toggled constantly during a test (input source, engine) | `LiteRtLmUi.OptionRow` |
| File paths | `LiteRtLmUi.PathRow` |
| Free text (prompts, tool JSON) | `LiteRtLmUi.TextArea` |
| Current state | `LiteRtLmUi.Status`, last in the control column |

The rules are written into the `LiteRtLmUi` class documentation so a scene added
later still matches.

Gaps found by auditing the scenes rather than assuming:

- `LiteRtLmLlmChatTestScene` did not use the shared two-column screen at all; it
  drew its own 780 px box. Converted — this is also the root cause of item 6.
- `LiteRtLmSampleScene` used free text fields for the backend. Now a dropdown,
  with the executable path on a `PathRow`.
- `LiteRtLmSampleScene`'s camera was the only one still on Skybox
  (`m_ClearFlags: 1`), so its background did not match. Now solid dark.
- ASR-FC, Multimodal and MM-FC had backends as button rows rather than
  dropdowns. Unified.

Evidence: `chat_twocolumn.png`, `chat_final.png`, `quickstart_dropdown.png`,
`asrfc_dropdowns.png`, `multimodal_fixed.png`, `mmfc_interactive.png`,
`asr_final.png`, `translate_verify.png`.

## Item 1 — URP downgrade, smoke and benchmark scenes split out

URP `14.0.12` (the 2022.3 line), `"unity": "2022.3"` in the package manifest,
editor 2022.3.62f3.

Two samples are declared:

| Sample | Scenes |
| --- | --- |
| Test Scenes | Quick start, LLM chat, ASR, Multimodal, Voice FC, Multimodal FC, Translate |
| Automated Tests | Android smoke, Conversation, Function-calling benchmark |

Runners, editor menus and asmdefs moved with the scenes, so the automated
assembly (`LiteRTLM.Unity.Samples.Automated`) no longer ships with the
hand-driven sample.

## Item 2 — Voice FC scene: editable prompt and input UI

Added a system-prompt text area, a tools-JSON text area, a Reset prompt button,
and three utterance sources (audio clip / microphone / typed text).

Evidence — the edited prompt and tool list actually drive the routing:

```
clip  → transcript: 2025년 3월 5일 전술 평가 결과 보고
      → tool call:  {"tool":"OpenTacticalEvaluationReport","parameters":{"date":"2025-03-05"}}
typed → "화면이 너무 어두워"
      → tool call:  {"tool":"SetBrightness","parameters":{"direction":"up","amount":"50"}}
```

Screenshots: `asrfc_prompt_ui.png`, `asrfc_dropdowns.png`, `asrfc_final.png`.

## Item 3 — ASR scene: transcript field instead of a wav-path log

The wav path moved to a separate "Metrics / raw log" area; a persistent
Transcript field accumulates recognised text (200 lines) with Copy and Clear.

A real defect surfaced while testing this: audio dropdown entry [2] pointed at a
file name that no longer existed, so selecting it failed with
`Error: Audio file not found:`.

```
catalogue: TestAssets/Audio/현재 서울의 날씨는, 흐림. 입니다.mp3   (comma, period)
on disk:   TestAssets/Audio/현재 서울의 날씨는 흐림 입니다.mp3
```

Only the ASR scene carried the stale name; Translate, `LiteRtLmSampleAssets` and
the build script already used the correct one. After fixing it, all 114
StreamingAssets references across the samples were audited for existence — none
missing.

Evidence (`asr_final.png`), two entries accumulated:

```
[13:30:03] 현재 서울의 날씨는 흐림입니다.
[13:30:29] 소리 키워 줘.
```

## Item 4 — Conversation scene: camera, UI, model path

`LiteRtLmRunLogOverlay.Attach` adds a camera and an on-screen log when the scene
has neither. The `FileNotFoundException` came from a model path that predated
the StreamingAssets reorganisation; `LiteRtLmStreamingAssets.Resolve` now falls
back to a search by file name and reports what it found.

Evidence (`conversation_final.png`): ten turns, no exception, and the model
recalls the code word planted in turn 4.

```
[RESPONSE] 9/10: 기억시키라고 한 코드워드는 LRT-CTX-042 입니다.
[SUCCESS] Completed 10 turns.
```

## Item 5 — Benchmark scene: no visible progress

Three separate causes, all fixed:

1. `runOnStart` was `0` in the scene asset (the Conversation scene had `1`), so
   pressing Play left the scene at `Idle` and nothing ran.
2. Per-case results went only to the status file, and `LiteRtLmLog` was not
   thread-safe — the benchmark logs from a worker thread. The log now hands out
   an immutable snapshot under a lock.
3. The model path had no self-healing fallback, so a relocated model threw.

Two further defects found while verifying:

- `LogStatus` wrote to the overlay and then called `WriteStatusLine`, which
  wrote to the overlay again — every lifecycle line appeared twice.
- The main thread (`STARTED_ASYNC`) and the worker thread (`RUN_START`) appended
  to the status file concurrently and interleaved mid-line, leaving a truncated
  record. Writes are now serialised.

Evidence (`benchmark_dedup.png`) — Play only, no manual invocation:

```
[RUN_START] model=gemma-4-E2B-it, cases=20, constrained=True
[INFO] Model '…\gemma-4-E2B-it.litertlm' relocated to Multimodal/gemma-4-e2b/…
[TURN] B01: expected=IncreaseBrightness, prompt=화면을 조금 더 밝게 해줘.
[PASS] B01: elapsedSeconds=5.7, tool=IncreaseBrightness, reason=ok
```

## Item 6 — Chat scene: buttons pushed off screen

Root cause was the scene's own fixed-size area, not the transcript height.
Rebuilt on the shared two-column screen: the transcript owns the right panel,
and the composer plus Send/Reset sit below a height-bounded settings scroll view
on the left, so they cannot be displaced.

Evidence (`chat_final.png`): after a real exchange the composer and both buttons
remain in place, status reads `Response received (last 6.70s)`.

```
user      드론 배터리가 20% 남았을 때 취해야 할 조치를 한 문장으로 알려줘.
assistant 드론 배터리가 20% 남았을 때는 즉시 안전한 장소로 이동하여 충전하거나
          비상 전원을 확보해야 합니다.
```

## Item 7 — Multimodal FC scene behaved like a smoke test

Rewritten as a user-driven scene: image source (none / bundled / file with a
picker), optional audio, an editable utterance, an editable system prompt and
tool list, Run and Reset.

The original scene classified from the file name because the desktop path was
text-only. It now sends a real `[image:]` tag to `litert_lm_advanced_main`, so
the tool chosen depends on the picture:

```
apples.jpg    → {"tool":"HandleFruit","parameters":{"fruit":"apple","count":3}}
puppy-run.jpg → {"tool":"HandleAnimal","parameters":{"species":"dog","count":1}}
```

Tools cover six categories: fruit, person, cartoon, appliance, animal, other.

Evidence: `mmfc_interactive.png`, `mmfc_verify.png`.

## Item 8 — Multimodal scene: image picker, mic audio, silent audio selection

Added a Browse file dialog for images, microphone capture for audio, an image
preview, and a Windows path through the advanced CLI. The silent audio selection
was the `[audio:]` tag decoding **wav only** — mp3 and ogg were passed through as
plain text, so the model answered as if there were no audio. Non-wav input is
now converted to 16 kHz mono wav and cached.

Evidence (`multimodal_verify.png`), file-picked image plus audio:

```
prompt   What fruit is in the image and how many?
response The image contains apples. There are three apples in the image.  (5.8s)
```

Audio alone transcribes correctly (`화면 밝게`, 1.3 s). With image and audio in
the same turn gemma-4-E2B tends to describe only the image — model behaviour,
not wiring: each modality works on its own through the same code path.

## Item 9 — Quick start scene: FileNotFoundException

Same self-healing resolver as the other scenes. The scene now resolves a
relocated model and completes a request.

Evidence (`quickstart_verify.png`): `Windows CLI response received`, TTFT 0.34 s,
prefill 70.4 tok/s, no exception.

## Item 10 — Translate scene: disabled button, no mic transcript

The Translate button's enable condition evaluated at the default selection:

```
androidBridge=False  desktopFallback=True  busy=False  →  enabled = True
```

The mic path previously only wrote a wav because there was no desktop ASR stage.
Both paths now report transcript and translation.

Evidence (`translate_verify.png`):

```
file  audio=2025년 3월 5일 전술평가 결과 보고.mp3
      transcript=2025년 3월 5일 전술 평가 결과 보고
      translation=Tactical Evaluation Results Report for March 5, 2025   asr=3.7s
mic   audio=fc_20260727_121235_115.wav
      transcript=화면 밝게
      translation=Screen bright                                          asr=1.4s
```

## Cross-cutting fixes

Found while testing, each one a real failure rather than a cleanup:

| Symptom | Cause |
| --- | --- |
| `I cannot access local files` | `[audio:]` decodes wav only |
| Transcript polluted with the instruction | the media tag parser stops at the first space, so `volume-소리 키워줘.wav` was read as text; media is now copied to an ASCII, space-free cache name |
| `INVALID_ARGUMENT: Model path is empty` | `advanced_main` was called with the legacy `run <model>` argument form |
| `Unknown command line flag 'system_message_file'` | custom FC flags exist only in the patched `litert_lm_main`; the system message is folded into the prompt for the stock binary |
| Progress stalls when the editor loses focus | `Application.runInBackground` was unset — the CLI child process was never the problem |
| Router replied `{'tool': 'None'}` | single-quoted, Python-style output; quotes are normalised before parsing |
| Horizontal scrollbar hid Browse, Send and the mic row | an unconstrained `TextField` takes its minimum width from its text; long absolute paths widened the control column |

## Verification status

- Roslyn compile of both sample assemblies: exit 0 (warnings only).
- Unity console: 0 errors.
- Not committed — awaiting review.

Outstanding, unrelated to these items: Linux/macOS binary support, which needs a
Mac.
