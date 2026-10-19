"""SenseVoice on the device: clip export and device-vs-desktop parity.

  clips  write the sv_eval sets as 16 kHz float32 wavs + clips.tsv for Run-LiteRtLmAndroidSenseVoiceSmokeTest.ps1
         python sv_device.py clips --sets real_clean,real_noise,gate40
  score  device results (+ feats/logits dumps) against the desktop tflite path on the very same wavs:
         token identity, CER, feature / logit max |diff|, confidence diff, feat / infer latency
         python sv_device.py score Builds/Logs/AndroidSenseVoiceSmoke/<run>

Confidence (mirrors sensevoice::Run in litertlm.cc): per row the max softmax prob; meanMaxProb = mean over the
speech rows (the 4 query rows excluded); per emitted non-tag token the max over its CTC run; minTokenProb = the
smallest of those (0 when nothing was emitted).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import soundfile

from sv_eval import ROOT, VOCAB, TfliteBackend, cer, decode_audio, load_set

MODEL_DIR = ROOT / "External" / "sensevoice-work" / "hf-release" / "sensevoice-small-litert"
CLIPS_DIR = ROOT / "External" / "sensevoice-work" / "device-clips"


def load_pieces(model_dir: Path) -> list[str]:
    pieces = [""] * VOCAB
    for line in (model_dir / "tokens.txt").read_text(encoding="utf-8").splitlines():
        i, _, piece = line.partition("\t")
        pieces[int(i)] = piece
    return pieces


def decode(lg: np.ndarray, pieces: list[str]) -> dict:
    """Greedy CTC + confidence, row for row the JNI's decode loop."""
    best = lg.argmax(-1)
    prob = 1.0 / np.exp(lg - lg.max(-1, keepdims=True)).sum(-1)
    ids, probs, text, prev = [], [], "", -1
    for b, p in zip(best.tolist(), prob.tolist()):
        if b != 0 and b == prev and probs:
            if probs[-1] >= 0:
                probs[-1] = max(probs[-1], p)
        elif b != 0 and b != prev:
            ids.append(b)
            if pieces[b].startswith("<|"):
                probs.append(-1.0)
            else:
                probs.append(p)
                text += pieces[b]
        prev = b
    per = [p for p in probs if p >= 0]
    return {"tokens": ids, "text": text.replace("▁", " ").strip(),
            "meanMaxProb": float(prob[4:].mean()) if len(prob) > 4 else 0.0,
            "minTokenProb": min(per) if per else 0.0}


def backend(model_dir: Path, variant: str, threads: int) -> TfliteBackend:
    return TfliteBackend(str(model_dir), threads, pattern=f"sensevoice_small_*_{variant}.tflite")


def stats(v: list[float]) -> str:
    if not v:
        return "n/a"
    s = sorted(v)
    return f"{statistics.mean(s):.0f} / {s[len(s) // 2]:.0f}"


# ---------------------------------------------------------------- clips

def cmd_clips(a) -> int:
    out = Path(a.out)
    lines = []
    for set_name in re.split(r"[,\s]+", a.sets.strip()):
        d = out / set_name
        d.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(load_set(set_name)):
            p = d / f"{i:03d}.wav"
            # float WAV: the noise mp3s decode past ±1 (peaks 1.4-1.8), PCM16 would clip them (real_noise CER 0.27 -> 0.32).
            soundfile.write(p, c["wav"], 16000, subtype="FLOAT")
            lines.append((len(c["wav"]), f"{p}\t{c['ref']}"))
    # Shortest first: the device keeps one bucket graph resident, so sorted clips compile each bucket once.
    lines = [l for _, l in sorted(lines, key=lambda x: x[0])]
    (out / "clips.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} clips -> {out / 'clips.tsv'}")
    return 0


# ---------------------------------------------------------------- score

def cmd_score(a) -> int:
    run = Path(a.run)
    model_dir = Path(a.model_dir)
    variant = json.loads((run / "config.json").read_text(encoding="utf-8-sig"))["variant"]
    local = [l.split("\t", 1) for l in (run / "clips-local.tsv").read_text(encoding="utf-8").splitlines() if l.strip()]
    device = {r["index"]: r for r in map(json.loads, (run / "results.jsonl").read_text(encoding="utf-8-sig").splitlines())}  # C# Encoding.UTF8 writes a BOM
    be, pieces = backend(model_dir, variant, a.threads), load_pieces(model_dir)
    rows = []
    for i, (path, ref) in enumerate(local):
        dv = device.get(i)
        if dv is None or dv.get("error"):
            rows.append({"set": Path(path).parent.name, "error": dv.get("error") if dv else "missing"})
            continue
        f = be.side.feats(decode_audio(path))
        lg = be.logits(f, "ko", False)
        ds = decode(lg, pieces)
        row = {"set": Path(path).parent.name, "bucket": dv["bucket"],
               "cer": cer(ref, dv["text"], True), "cer_desktop": cer(ref, ds["text"], True),
               "tokens_equal": list(dv["tokens"] or []) == ds["tokens"], "text_equal": dv["text"] == ds["text"],
               "d_mean": abs(dv["meanMaxProb"] - ds["meanMaxProb"]), "d_min": abs(dv["minTokenProb"] - ds["minTokenProb"]),
               "featMs": dv["featMs"], "inferMs": dv["inferMs"], "compileMs": dv["compileMs"], "wallMs": dv["wallMs"]}
        dump = run / "dump" / str(i)
        if (dump / "feats.f32").exists():
            df = np.fromfile(dump / "feats.f32", np.float32).reshape(-1, 560)
            dl = np.fromfile(dump / "logits.f32", np.float32).reshape(-1, VOCAB)
            row["feat_dmax"] = float(np.abs(df - f[: len(df)]).max())
            row["logit_dmax"] = float(np.abs(dl - lg[: len(dl)]).max())
            row["argmax_agree"] = float((dl.argmax(-1) == lg[: len(dl)].argmax(-1)).mean())
        rows.append(row)
    (run / "score.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    print(f"## {run.name} ({variant})")
    print("| set | n | errors | CER device | CER desktop | exact | tokens = desktop | max conf diff | feat max diff | logit max diff | feat ms mean/p50 | infer ms mean/p50 (warm) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in dict.fromkeys(r["set"] for r in rows):
        rs = [r for r in rows if r["set"] == s]
        ok = [r for r in rs if "error" not in r]
        dumped = [r for r in ok if "feat_dmax" in r]
        warm = [r for r in ok if r["compileMs"] == 0]
        print(f"| {s} | {len(rs)} | {len(rs) - len(ok)} | {statistics.mean(r['cer'] for r in ok):.4f} "
              f"| {statistics.mean(r['cer_desktop'] for r in ok):.4f} | {sum(r['cer'] == 0 for r in ok)} "
              f"| {sum(r['tokens_equal'] for r in ok)}/{len(ok)} | {max(max(r['d_mean'], r['d_min']) for r in ok):.2e} "
              f"| {max((r['feat_dmax'] for r in dumped), default=float('nan')):.2e} "
              f"| {max((r['logit_dmax'] for r in dumped), default=float('nan')):.2e} "
              f"| {stats([r['featMs'] for r in ok])} | {stats([r['inferMs'] for r in warm])} |")
    print("\n| bucket | clips | infer ms mean/p50 (warm) | compile ms (first) |\n|---|---:|---:|---:|")
    ok = [r for r in rows if "error" not in r]
    for b in ("3s", "5s", "10s", "30s"):
        rb = [r for r in ok if r["bucket"] == b]
        if rb:
            print(f"| {b} | {len(rb)} | {stats([r['inferMs'] for r in rb if r['compileMs'] == 0])} "
                  f"| {max(r['compileMs'] for r in rb):.0f} |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clips")
    c.add_argument("--sets", default="real_clean,real_noise,gate40")
    c.add_argument("--out", default=str(CLIPS_DIR))
    s = sub.add_parser("score")
    s.add_argument("run")
    s.add_argument("--model-dir", default=str(MODEL_DIR))
    s.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    return {"clips": cmd_clips, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    # decode(): repeats inside a CTC run collapse to the run's max prob, tags count for nothing.
    _p = ["<blank>", "<|ko|>", "▁가", "나"]
    _lg = np.log(np.array([[.01, .97, .01, .01], [.01, .01, .9, .08], [.01, .01, .95, .03], [.9, .05, .03, .02],
                           [.01, .01, .02, .96], [.01, .01, .02, .96]], np.float32))
    _r = decode(_lg, _p)
    assert _r["tokens"] == [1, 2, 3] and _r["text"] == "가나", _r
    assert abs(_r["minTokenProb"] - 0.95) < 1e-5 and abs(_r["meanMaxProb"] - 0.96) < 1e-5, _r
    sys.exit(main())
