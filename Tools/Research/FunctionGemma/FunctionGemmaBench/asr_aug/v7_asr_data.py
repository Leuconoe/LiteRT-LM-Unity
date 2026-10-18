"""v7: teach FunctionGemma what real recogniser output looks like, so a broken transcript is refused
instead of being "repaired" into a wrong command.

  python asr_aug/v7_asr_data.py subset            # stratified data_v6 texts -> asr_aug/v7_subset_cases.json
  py -3.13 locked/make_h5_voice.py build --cases asr_aug/v7_subset_cases.json --out External/fg-voice-v7 \
      --voices sunhi,hyunsu,st-m1,sunhi-fast,hyunsu-slow --per-case 1      (+ run --voice ... per voice)
  python asr_aug/v7_asr_data.py noise             # mild additive noise (SNR 12-18 dB) on every 5th clip
  <sv venv> fg_voice_e2e.py asr --arm svw8|wb5 --manifest External/fg-voice-v7/manifest.jsonl ...
  python asr_aug/v7_asr_data.py teach-input       # transcripts that differ from the text -> asr_aug/v7_teach_cases.json
  python paraphrase/paraphrase.py teach --label V7 --cases asr_aug/v7_teach_cases.json   (Qwen2.5-14B, full ToolsPrompt)
  python asr_aug/v7_asr_data.py build             # label rule below -> data_v7_empty/

Label rule for a transcript of a record labelled L (teacher = Qwen's call on the TRANSCRIPT, ungrounded optional
keys stripped):
  L executable: teacher == L and L grounded in the transcript -> keep L
                L no longer grounded (a slot was lost) and teacher != L -> DefaultResponse
                teacher == DefaultResponse -> DefaultResponse (intent lost)
                teacher == another executable call -> drop (conflict)
  L refusal:    teacher == DefaultResponse -> keep; otherwise drop (e.g. ASR ate the negation)
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = BENCH.parents[3]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / "paraphrase"))
import fg_common as fg  # noqa: E402
import make_dataset as md  # noqa: E402

SUBSET = HERE / "v7_subset_cases.json"
TEACH_IN = HERE / "v7_teach_cases.json"
TEACH_OUT = BENCH / "paraphrase" / "out" / "teacher_V7_v7_teach_cases.jsonl"
VOICE_DIR = ROOT / "External" / "fg-voice-v7"
ASR_DIR = ROOT / "Builds" / "Logs" / "fg-voice-v7"
BLOCK = ["eval_cases.json", "heldout_cases.json", "heldout2_cases.json", "heldout3_cases.json", "heldout4_cases.json",
         "locked/heldout5a_cases.json", "locked/heldout5b_cases.json", "locked/heldout6_cases.json"]
CAP = 0.30          # ASR-transcript records as a share of the final train set
SEED = 20261018


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8-sig").splitlines() if l.strip()]


def banned() -> set[str]:
    keys = set()
    for f in BLOCK:
        if (BENCH / f).exists():
            for c in json.loads((BENCH / f).read_text(encoding="utf-8"))["cases"]:
                k = md.norm(c["text"])
                keys |= {k, k.replace(" ", "")}
    assert any("heldout6" in f for f in BLOCK) and (BENCH / "locked" / "heldout6_cases.json").exists(), "write heldout6 first"
    return keys


def cmd_subset(n: int = 6000) -> None:
    rows = read_jsonl(BENCH / "data_v6_empty" / "train.jsonl")
    by_fn = defaultdict(list)
    for r in rows:
        by_fn[r["function"]].append(r)
    rnd, picked = random.Random(SEED), []
    for fn, recs in sorted(by_fn.items()):
        k = min(len(recs), max(40, round(n * len(recs) / len(rows))))
        picked += rnd.sample(recs, k)
    cases = [{"id": f"t{i + 1:05d}", "suite": "v7", **r} for i, r in enumerate(picked)]
    SUBSET.write_text(json.dumps({"cases": cases}, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"subset {len(cases)} of {len(rows)}; DefaultResponse {sum(c['function'] == 'DefaultResponse' for c in cases)}")


def colored_noise(n: int, alpha: float, g: np.random.Generator) -> np.ndarray:
    """1/f^alpha noise (alpha 1 = pink, 2 = brown) — non-speech, no codec artifacts."""
    spec = np.fft.rfft(g.standard_normal(n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    x = np.fft.irfft(spec / f ** (alpha / 2), n)
    return x / (np.abs(x).max() + 1e-9)


def cmd_noise(frac_mod: int = 5) -> None:
    done_path = VOICE_DIR / "noise.json"
    done = set(json.loads(done_path.read_text(encoding="utf-8"))) if done_path.exists() else set()
    n = 0
    for r in read_jsonl(VOICE_DIR / "manifest.jsonl"):
        key = f"{r['voice']}/{r['id']}"
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        if h % frac_mod or key in done or not Path(r["path16k"]).exists():
            continue
        with wave.open(r["path16k"]) as w:
            params, y = w.getparams(), np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64)
        g = np.random.default_rng(h % 2**32)
        snr = 12 + (h >> 8) % 7                                   # 12-18 dB
        nz = colored_noise(len(y), 1.0 + (h >> 16) % 2, g)
        nz *= np.sqrt(np.mean(y ** 2) / (np.mean(nz ** 2) + 1e-9) / 10 ** (snr / 10))
        with wave.open(r["path16k"], "wb") as w:
            w.setparams(params)
            w.writeframes(np.clip(y + nz, -32768, 32767).astype(np.int16).tobytes())
        done.add(key)
        n += 1
    done_path.write_text(json.dumps(sorted(done)), encoding="utf-8")
    print(f"noise added to {n} clips ({len(done)} total)")


def cmd_teach_input() -> None:
    cases = {c["id"]: c for c in json.loads(SUBSET.read_text(encoding="utf-8"))["cases"]}
    block = banned()
    seen, out, drops = set(), [], Counter()
    for arm in ("svw8", "wb5"):
        for a in read_jsonl(ASR_DIR / f"asr-{arm}.jsonl"):
            c = cases[a["id"]]
            hyp = (a.get("hyp") or "").strip()
            k = md.norm(hyp)
            reason = ("empty" if not k else "same" if k.replace(" ", "") == md.norm(c["text"]).replace(" ", "")
                      else "banned" if k in block or k.replace(" ", "") in block else "dup" if (k, c["id"]) in seen else "")
            if reason:
                drops[reason] += 1
                continue
            seen.add((k, c["id"]))
            out.append({"id": f"{c['id']}-{arm}-{a['voice']}", "text": hyp, "function": c["function"], "arguments": c["arguments"],
                        "src_text": c["text"], "arm": arm, "voice": a["voice"]})
    TEACH_IN.write_text(json.dumps({"cases": out}, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"teacher input {len(out)}; skipped {dict(drops)}")


def decide(r: dict, idx: dict) -> str:
    """keep | refuse | drop — see module docstring."""
    from paraphrase import strip_ungrounded_optional
    t = r["teacher"]
    t = strip_ungrounded_optional(t, idx, r["text"]) if t["function"] != "DefaultResponse" else t
    if r["function"] == "DefaultResponse":
        return "keep" if t["function"] == "DefaultResponse" else "drop"
    label = {"function": r["function"], "arguments": r["arguments"]}
    grounded = fg.validate(label, idx, r["text"])["function"] == r["function"]
    if t["function"] == "DefaultResponse":
        return "refuse"
    if fg.grade(r, t, idx)[0] and grounded:
        return "keep"
    if not grounded and not fg.grade(r, t, idx)[0]:
        return "refuse"
    return "drop"


def cmd_build() -> None:
    idx = {t["function"]["name"]: t for t in fg.load_tools()}
    rows = read_jsonl(TEACH_OUT)
    verdict = Counter()
    new = []
    for r in rows:
        v = decide(r, idx)
        verdict[(r["function"] == "DefaultResponse", v)] += 1
        if v == "drop":
            continue
        fn, args = (r["function"], r["arguments"]) if v == "keep" else ("DefaultResponse", {})
        new.append({"text": r["text"], "function": fn, "arguments": args if fn != "DefaultResponse" else {}, "asr": r["arm"]})
    # one record per transcript text (two arms/voices can produce the same string); a string that two
    # source commands collapsed into with different targets is unrecoverable either way, so it goes
    groups = defaultdict(list)
    for r in new:
        groups[md.norm(r["text"])].append(r)
    same = lambda g: len({json.dumps([r["function"], r["arguments"]], sort_keys=True) for r in g}) == 1
    verdict["conflict_texts"] = sum(not same(g) for g in groups.values())
    new = [g[0] for g in groups.values() if same(g)]
    base_train = read_jsonl(BENCH / "data_v6_empty" / "train.jsonl")
    base_val = read_jsonl(BENCH / "data_v6_empty" / "val.jsonl")
    have = {md.norm(r["text"]) for r in base_train + base_val}
    new = [r for r in new if md.norm(r["text"]) not in have]
    rnd = random.Random(SEED)
    rnd.shuffle(new)
    cap = int(CAP / (1 - CAP) * len(base_train))
    new = new[:cap]
    n_val = max(1, len(new) // 20)
    out = BENCH / "data_v7_empty"
    out.mkdir(exist_ok=True)
    strip = lambda r: {k: r[k] for k in ("text", "function", "arguments")}
    train = base_train + [strip(r) for r in new[n_val:]]
    val = base_val + [strip(r) for r in new[:n_val]]
    rnd.shuffle(train)
    for name, recs in (("train", train), ("val", val)):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    dr = sum(r["function"] == "DefaultResponse" for r in train) / len(train)
    print("teacher verdicts (label is refusal?, verdict):", dict(verdict))
    print(f"ASR records added: {len(new)} (train {len(new) - n_val}, val {n_val}); "
          f"refusals among them {sum(r['function'] == 'DefaultResponse' for r in new)}; by arm {dict(Counter(r['asr'] for r in new))}")
    print(f"data_v7_empty: train {len(train)} (ASR share {(len(new) - n_val) / len(train):.1%}, DefaultResponse {dr:.1%}), val {len(val)}")


def _selfcheck() -> None:
    idx = {t["function"]["name"]: t for t in fg.load_tools()}
    lab = {"function": "MoveToMarker", "arguments": {"markerNumber": 3}}
    ok = {**lab, "text": "마커 삼 이동", "teacher": dict(lab)}
    assert decide(ok, idx) == "keep"
    lost = {**lab, "text": "마커 이동", "teacher": {"function": "EnterMarkerCreation", "arguments": {}}}
    assert decide(lost, idx) == "refuse"
    ref = {**lab, "text": "마커 삼 이동", "teacher": {"function": "DefaultResponse", "arguments": {}}}
    assert decide(ref, idx) == "refuse"
    other = {**lab, "text": "마커 삼 삭제", "teacher": {"function": "DeleteMarker", "arguments": {"markerNumber": 3}}}
    assert decide(other, idx) == "drop"
    neg = {"function": "DefaultResponse", "arguments": {}, "text": "녹화 시작", "teacher": {"function": "StartRecording", "arguments": {}}}
    assert decide(neg, idx) == "drop"
    assert colored_noise(1600, 1.0, np.random.default_rng(0)).shape == (1600,)


if __name__ == "__main__":
    _selfcheck()
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    {"subset": cmd_subset, "noise": cmd_noise, "teach-input": cmd_teach_input, "build": cmd_build}[stage]()
