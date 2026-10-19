# FunctionGemma function calling on the device

How to turn `google/functiongemma-270m-it` into an on-device voice-command router for **your own tool
contract**, and run it from Unity on LiteRT-LM. Everything here ships as scripts and a sample contract
(a Korean smart-home assistant, `Tools/Research/FunctionGemma/FunctionGemmaBench/sample_contract.json`);
no model file is committed — you train one for your contract.

Device figures are from kona / SM8250 (CPU). Accuracy is not quoted: it depends entirely on your contract
and data — measure it on your own eval set.

## 1. Why fine-tune, and why with no declarations in the prompt

Stock FunctionGemma reads the tool declarations from its prompt. For a few dozen tools that prompt runs to
hundreds or thousands of tokens, and on a phone-class CPU prefill dominates every command. Stock accuracy
on Korean voice commands is also poor without fine-tuning.

The recipe here fine-tunes with **decl-mode none**: the prompt is FunctionGemma's trigger sentence as the
developer turn plus the bare utterance (~30 tokens), and the contract lives in the weights.

- **Fast:** with a cache-256 / prefill-64·128 bundle, a command takes **~0.3–0.4 s** on kona CPU, with
  **0.9 s** init. The int8 bundle is **277 MB**.
- **The cost:** any change to a tool, a parameter, an enum value or a range is a **retrain**.
  `Rebuild-FunctionGemmaRouter.ps1` makes that one command (§5). Mapping a call to a UI handler, or editing
  which calls need confirmation, does not need a retrain.

`fg_eval.py --decl-mode full|compact|none` scores the stock model with declarations, for comparison.

## 2. The contract

An OpenAI-style tools JSON. Router settings ride along as `x-` keys that are stripped before anything
reaches the model:

| Key | On | Meaning |
| --- | --- | --- |
| `"x-grounded": true` | parameter | the value must be traceable to the utterance (digits, Sino-Korean, native counter forms, the enum value or one of its aliases) |
| `"x-aliases": {value: [spoken forms]}` | enum parameter | how a value is said (`"Kitchen": ["주방", "부엌"]`) |
| `"x-confirm": true` | function | the app asks before running it (`UnlockDoor` in the sample) |

Also used: `enum`, `minimum` / `maximum` (both validated by the guard), `required`. A `DefaultResponse`
tool is the refusal: the model emits it for out-of-scope, ambiguous, incomplete or multi-action utterances.

## 3. Data and training

`make_dataset.py` fills **templates** (`sample_templates.json`) from the contract:

- **Positives:** every template of every tool, with enum values said through their aliases and numbers as
  digits, Sino-Korean or native counter forms inside `[minimum, maximum]`. Particles agree with the value
  (`{room|을}`).
- **Dressing:** fillers, a lead-in clause, a dropped or added space, a trailing period — what a recogniser
  produces.
- **Refusals (~20 %):** out-of-scope and negated requests, commands missing a required value, and two
  commands joined in one utterance (the router never executes two actions from one utterance).
- **Filtering:** every text is exact-filtered (spaces and punctuation ignored) against the eval cases, so
  the eval stays unseen.

`fg_train.py` does a full fine-tune: 3 epochs, lr 2e-5, batch 8, loss on the completion only. After every
epoch it greedy-generates on the validation records and keeps the epoch with the best exact match.
Training writes a checkpoint per epoch, and `--resume` continues a crashed run. The sample trains in
minutes on one desktop GPU.

Ideas that helped on a real contract and fit this pipeline when you need them:
- LLM-written paraphrases of the templates, with each label checked by a teacher LLM.
- Records built from real recogniser output (TTS → ASR → teacher relabel), so broken transcripts are
  refused instead of "repaired" into the wrong command. This was the largest single gain in wrong actions.

## 4. The router (app side)

`Runtime/LiteRtLmFunctionGemmaRouter.cs`, with a Python twin in `FunctionGemmaBench/fg_common.py`:

1. **Parse:** `<start_function_call>call:Name{k:<escape>v<escape>,n:3}<end_function_call>` → call.
2. **Guard:** an unknown function, a missing required or unknown parameter, the wrong type, an enum miss,
   an out-of-range number, or an `x-grounded` value not traceable to the utterance → `DefaultResponse`.
   The model may refuse; it may not invent a number or a room.
3. **Clause split:** when the whole utterance is refused and it has punctuation, each clause runs on its
   own. It executes only if exactly one distinct executable call comes out (remarks around a command).
   Two different actions stay refused.
4. **Confirmation:** `ConfirmationPolicy` marks `x-confirm` calls `NeedsConfirmation`. With
   `MinAsrConfidence` set (e.g. SenseVoice's CTC confidence), it also marks any executable call heard with
   low confidence. `ParseConfirmation` reads a 예/아니오 answer.

A fine-tuned 270M router still makes some wrong calls on hard audio, so keep irreversible commands behind
confirmation. The C# router and `fg_common.py` run the same golden table
(`Samples~/AutomatedTests/Editor/functiongemma_router_golden.json`): menu
`LiteRT-LM/Tests/FunctionGemma Router Self-Test`, and `rebuild_check.py selftest`.

The sample scene **Voice → tool call** (`AsrFunctionCalling`) runs ASR (Whisper tiers or SenseVoice), then
the router, and shows the confirmation buttons. `LiteRtLmSampleAssets.FunctionGemmaRouterModel` names the
bundle.

## 5. One command: `Rebuild-FunctionGemmaRouter.ps1`

```powershell
.\Tools\Research\FunctionGemma\Run-FunctionGemmaEval.ps1 -Bootstrap                        # once
.\Tools\Research\FunctionGemma\Rebuild-FunctionGemmaRouter.ps1 -DryRun                     # stage 1 only
.\Tools\Research\FunctionGemma\Rebuild-FunctionGemmaRouter.ps1 -Smoke                      # CPU plumbing test
.\Tools\Research\FunctionGemma\Rebuild-FunctionGemmaRouter.ps1 -Label sample -WaitForGpu -Stage -Device
.\Tools\Research\FunctionGemma\Rebuild-FunctionGemmaRouter.ps1 -Contract my.json -Templates my_t.json -Cases my_cases.json -Label myapp
```

| Stage | What it does |
| --- | --- |
| 1 contract | Diffs the contract against `contract_snapshot.json`. **Stops with a to-do list** when a tool has no template, a template misses a required parameter, or a parameter cannot be generated or grounded. Eval cases whose label no longer validates are listed and left out. |
| 2 data | `make_dataset.py` |
| 3 train | Behind a GPU guard; retries with `--resume` |
| 4 gate | Batch 1, guard + clause split: 0 wrong actions and ≥ 95 % pass |
| 5 convert | Converts the bundle and checks Windows-CLI parity against the HF calls |
| 6 export | Writes `Resources/functiongemma_tools.json`; `-Stage` copies the bundle and repoints `FunctionGemmaRouterModel` |
| 7 device | `-Device`: the prompt set on the phone, graded with the router |

Every stage logs to `Builds/Logs/fg-rebuild-<label>/report.md`.

## 6. Conversion and the traps

`Convert-FunctionGemmaToLiteRtLm.ps1` runs litert-torch 0.9.4 in Docker (no Windows wheels):

- `--litert_lm_model_type_override=function_gemma`
- `--tokenizer_path_override=<tokenizer.model>`: the runtime refuses constrained decoding without a
  SentencePiece tokenizer.
- `--use_jinja_template=False`: the HF template does not render in the v0.14 runtime.
- No `--externalize_embedder`: Gemma ties `lm_head` to the embedding, so that only duplicates the table.
- int8 `dynamic_wi8_afp32`, `-CacheLength 256 -PrefillLengths @(64,128)`.

Traps that cost time:

- **Score HF models at batch 1.** Batched, left-padded `generate` garbles some Gemma3 outputs and
  under-reports the model (`fg_eval.py` defaults to batch 1).
- **`pwsh -File` turns `-PrefillLengths 64,128` into the integer 64128.** Call the script directly with
  `@(64,128)`.
- **Send the pre-rendered prompt, not tools JSON.** With `--tools_json_file` the runtime renders the
  declarations differently from HF. `fg_eval_litertlm.py` sends the HF render through
  `--system_message_file` and checks that the CLI's prefill token count equals the HF count on every case.
- **litert-torch's generic `convert` mis-lowers HF Gemma3 attention** (`repeat_kv`): the graph matches at
  position 0 only. This hits the Jev trunk export, not `export_hf`; see `Tools/Research/Jev/README.md`.

## 7. Related: the Jev decision sample

`Tools/Research/Jev` scores **closed options** with probabilities, from a frozen gemma-3-270m trunk (an
exported LiteRT graph, ~61 ms per utterance with i8 on kona) plus a small head. It reuses this pipeline's
data and contract. It cannot fill arguments, so it complements the router rather than replacing it: for
example, picking the UI area, or gating "is this a command at all".
