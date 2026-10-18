"""End-to-end voice test: TTS clip -> ASR -> fine-tuned FunctionGemma -> graded call.

Audio is the DroneVRV2 VoiceTtsCheck cache (file stem = utterance); only stems that
are an eval_cases/heldout_cases text carry a label. ALL of it is synthesized speech —
ranking only; real recordings are ~2.8x harder.

Two stages, two venvs (the ASR and FC stacks do not share one):
  asr  SenseVoiceBench/.venv (litert + funasr)     -> Builds/Logs/fg-voice-e2e/asr-<arm>.jsonl
  fc   FunctionGemmaBench/.venv (transformers)     -> fc.jsonl, summary.md

  <sv-venv>\\python fg_voice_e2e.py asr --arm wb5 | wb5hw | svpt | svw8
  <fg-venv>\\python fg_voice_e2e.py fc [--cli-check 20]

Failure split: safe = DefaultResponse/unparsed instead of the command (nothing runs);
wrong = a different function or argument executed (incl. a command where DefaultResponse
was expected) — the dangerous one for a voice UI.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TTS = ROOT.parent.parent / "_contents" / "DroneVRV2" / "VoiceTtsCheck" / "tts"
OUT = ROOT / "Builds" / "Logs" / "fg-voice-e2e"
HOTWORDS = ROOT.parent / "VoiceCommand-WebServer" / "assets" / "WhisperHotwords.txt"
WHISPER = ROOT / "Assets" / "StreamingAssets" / "ASR" / "whisper-base-acft-ko"
SV_W8 = ROOT / "External" / "sensevoice-work" / "tflite" / "stock" / "w8"
FG_MODEL = ROOT / "External" / "functiongemma-work" / "ft-functiongemma-none" / "merged"
FG_BUNDLE = ROOT / "External" / "functiongemma-work" / "ft-functiongemma-none" / "litertlm-c256-p64" / "model.litertlm"
VOICES = {"brian": "unity", "george": "unity", "sarah": "unity", "heami": "sapi",
          "sunhi": "edge", "injoon": "edge", "hyunsu": "edge"}
ARMS = {"wb5": "whisper base-acft-ko 5s", "wb5hw": "whisper base-acft-ko 5s + hotwords",
        "svpt": "SenseVoice-Small (funasr)", "svw8": "SenseVoice-Small LiteRT w8"}


MANIFEST: Path | None = None  # --manifest: clip list {id, voice, kind, text, spoken_text, path16k} instead of the TTS cache
CASE_FILES = ("eval_cases.json", "heldout_cases.json")  # --cases: label files (keyed by text and by id)


def labels() -> dict[str, dict]:
    lab = {}
    for f in CASE_FILES:
        for c in json.loads((HERE / f).read_text(encoding="utf-8"))["cases"]:
            lab[c["text"]] = c
            lab[c["id"]] = c
    lab.setdefault("장애물 설정", {"id": "x001", "suite": "extra", "text": "장애물 설정",
                                "function": "ApplyDisplayFilters", "arguments": {"filterNames": "장애물"}})
    return lab


def clips() -> list[dict]:
    lab, out = labels(), []
    if MANIFEST:
        for m in (json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()):
            if Path(m["path16k"]).exists():
                out.append({**lab[m["id"]], **m, "path": m["path16k"]})
        return out
    for voice, kind in VOICES.items():
        for p in sorted((TTS / voice).glob("*.wav")):
            stem = unicodedata.normalize("NFC", p.stem)
            if stem in lab:
                out.append({**lab[stem], "voice": voice, "kind": kind, "path": str(p)})
    return out


# ---------------------------------------------------------------- asr stage (SenseVoiceBench venv)

def make_asr(arm: str, threads: int):
    sys.path.insert(0, str(ROOT / "Tools" / "Research" / "SenseVoice" / "SenseVoiceBench"))
    import sv_eval as sv

    if arm.startswith("wb5"):
        sys.path.insert(0, str(ROOT / "Tools" / "Research" / "Whisper" / "WhisperTflite"))
        import whisper_tflite_runner as w
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(WHISPER / "tokenizer.json"))
        m = w.load_model(str(WHISPER / "acft_base_5s_drq.tflite"))
        prompt, eot = w.forced_prefix(m["vocab"], "ko")
        if arm == "wb5hw":  # same budget as hotword_ab: keep 32 output slots in the 128-token decoder
            prompt = w.hotword_prefix(tok, w.load_hotwords(HOTWORDS), w.SEQ - len(prompt) - 1 - 32) + prompt
        return lambda path: tok.decode(w.transcribe(m, w.load_audio(path), prompt, eot)[0]).strip(), sv.cer

    if arm == "svpt":
        be = sv.FunasrBackend("FunAudioLLM/SenseVoiceSmall", "cuda:0")
    else:  # private copy: S1 may rewrite its graphs while this runs (tflite mmaps the file)
        local = OUT / "cache" / "sv-w8"
        if not local.exists():
            shutil.copytree(SV_W8, local)
        be = sv.TfliteBackend(str(local), threads)
    return lambda path: sv.TAG.sub("", be(sv.decode_audio(path), "ko", False)[0]).strip(), sv.cer


def run_asr(arm: str, threads: int) -> None:
    asr, cer = make_asr(arm, threads)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = clips()
    with (OUT / f"asr-{arm}.jsonl").open("w", encoding="utf-8") as f:
        for c in rows:
            t = time.perf_counter()
            hyp = asr(c["path"])
            spoken = c.get("spoken_text", c["text"])  # CER against what the TTS actually said
            f.write(json.dumps({"id": c["id"], "voice": c["voice"], "kind": c["kind"], "text": c["text"], "spoken_text": spoken,
                                "hyp": hyp, "cer": round(cer(spoken, hyp, True), 4), "asr_s": round(time.perf_counter() - t, 3)},
                               ensure_ascii=False) + "\n")
    print(f"asr-{arm}: {len(rows)} clips")


# ---------------------------------------------------------------- fc stage (FunctionGemmaBench venv)

def outcome(case: dict, got: dict | None, passed: bool) -> str:
    if passed:
        return "ok"
    if got is None or got["function"] == "DefaultResponse":
        return "safe"
    return "wrong"


def run_fc(cli_check: int, model_dir: Path = FG_MODEL, label: str = "") -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import fg_common as fg
    from fg_eval import generate

    render, parse, tools = fg.make_format("functiongemma", "none")
    idx = {t["function"]["name"]: t for t in tools}
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to("cuda").eval()
    suffix = f"-{label}" if label else ""
    cache: dict[str, str] = {}

    def call(text: str) -> str:
        if text not in cache:  # batch 1: batched left-padded generate garbles output (known trap)
            cache[text] = generate(model, tok, [render(tok, text)], batch_size=1)[0]
        return cache[text]

    def route(text: str) -> dict:
        """What the app does (LiteRtLmFunctionGemmaRouter): guard + grounding, clause split on refusal."""
        full = fg.validate(parse(call(text)), idx, text)
        parts = fg.segments(text) if full["function"] == "DefaultResponse" else []
        return fg.combine_segments(full, [fg.validate(parse(call(p)), idx, p) for p in parts]) if parts else full

    lab = labels()
    rows = []
    for arm in ARMS:
        path = OUT / f"asr-{arm}.jsonl"
        if not path.exists():
            print(f"skip {arm}: no {path.name}")
            continue
        for a in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()):
            case = lab.get(a.get("id")) or lab[a["text"]]
            got_ref, got = parse(call(a["text"])), parse(call(a["hyp"]))
            ok_ref, ok = fg.grade(case, got_ref, idx)[0], fg.grade(case, got, idx)[0]
            routed = route(a["hyp"])
            ok_r = fg.grade(case, routed, idx)[0]
            rows.append({**a, "arm": arm, "suite": case["suite"], "function": case["function"],
                         "arguments": case["arguments"], "got": got, "oracle_ok": ok_ref,
                         "e2e_ok": ok, "outcome": outcome(case, got, ok),
                         "routed": routed, "routed_ok": ok_r, "routed_outcome": outcome(case, routed, ok_r)})
    with (OUT / f"fc{suffix}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def table(key) -> list[str]:
        g = defaultdict(list)
        for r in rows:
            g[(r["arm"], key(r))].append(r)
        lines = ["| ASR | voice | n | CER | ASR exact | FC oracle | E2E | safe | wrong | routed E2E | routed wrong |",
                 "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for (arm, v), rs in g.items():
            n = len(rs)
            lines.append(f"| {arm} | {v} | {n} | {sum(r['cer'] for r in rs) / n:.3f} | {sum(r['cer'] == 0 for r in rs)} "
                         f"| {sum(r['oracle_ok'] for r in rs)} | {sum(r['e2e_ok'] for r in rs)} "
                         f"| {sum(r['outcome'] == 'safe' for r in rs)} | {sum(r['outcome'] == 'wrong' for r in rs)} "
                         f"| {sum(r['routed_ok'] for r in rs)} | {sum(r['routed_outcome'] == 'wrong' for r in rs)} |")
        return lines

    md = [f"# Voice E2E (TTS -> ASR -> FunctionGemma {model_dir.parent.name}) — synthesized speech, ranking only", "",
          "## By ASR", *table(lambda r: "all"), "",
          "## By voice group", *table(lambda r: r["kind"]), "", "## By voice", *table(lambda r: r["voice"]), "",
          "## Wrong actions (executed something not asked)", "", "| ASR | voice | reference | transcript | expected | called |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        if r["outcome"] == "wrong":
            md.append(f"| {r['arm']} | {r['voice']} | {r['text']} | {r['hyp']} | {r['function']} {json.dumps(r['arguments'], ensure_ascii=False)} "
                      f"| {r['got']['function']} {json.dumps(r['got']['arguments'], ensure_ascii=False)} |")

    if cli_check:  # HF stands in for the device; confirm on the shipped int8 bundle
        from fg_eval_litertlm import DEFAULT_CLI, run_case, split_render

        picked = [r for r in rows if r["text"] != r["hyp"]][::7][:cli_check]
        same = 0
        md += ["", f"## CLI int8 spot-check ({FG_BUNDLE.parent.name}) on {len(picked)} E2E transcripts", ""]
        for r in picked:
            system, prompt = split_render("functiongemma", render(tok, r["hyp"]))
            raw = run_case(DEFAULT_CLI, FG_BUNDLE, "cpu", system, prompt)[0]
            same += parse(raw) == r["got"]
        md.append(f"same parsed call as HF: {same}/{len(picked)}")
    (OUT / f"summary{suffix}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def _selfcheck() -> None:
    c = {"function": "ReturnTeam", "arguments": {"team": "Alpha"}}
    assert outcome(c, None, False) == "safe"
    assert outcome(c, {"function": "DefaultResponse", "arguments": {}}, False) == "safe"
    assert outcome(c, {"function": "ReturnTeam", "arguments": {"team": "Bravo"}}, False) == "wrong"
    assert outcome({"function": "DefaultResponse", "arguments": {}}, {"function": "SelectTeam", "arguments": {}}, False) == "wrong"
    assert labels()["장애물 설정"]["function"] == "ApplyDisplayFilters"


def main() -> int:
    _selfcheck()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    a = sub.add_parser("asr")
    a.add_argument("--arm", choices=list(ARMS), required=True)
    a.add_argument("--threads", type=int, default=4)
    f = sub.add_parser("fc")
    f.add_argument("--cli-check", type=int, default=20)
    f.add_argument("--model", type=Path, default=FG_MODEL, help="merged HF checkpoint dir")
    f.add_argument("--label", default="", help="suffix for fc-<label>.jsonl / summary-<label>.md")
    for p in (a, f):
        p.add_argument("--manifest", type=Path, help="clip manifest jsonl (e.g. External/fg-voice-h5/manifest.jsonl)")
        p.add_argument("--cases", default="", help="comma-separated label files, relative to FunctionGemmaBench")
        p.add_argument("--out", type=Path, help="output dir (default Builds/Logs/fg-voice-e2e)")
    args = ap.parse_args()
    global OUT, MANIFEST, CASE_FILES
    OUT, MANIFEST = args.out or OUT, args.manifest
    CASE_FILES = tuple(args.cases.split(",")) if args.cases else CASE_FILES
    if args.stage == "asr":
        run_asr(args.arm, args.threads)
    else:
        run_fc(args.cli_check, args.model, args.label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
