"""Hotword-prefix A/B for the Whisper tflite exports (desktop CPU).

One process per arm (model x hotwords on/off) writes <out>/<label>.jsonl;
`--summarize` turns every jsonl in <out> into summary.md. Scoring follows
docs/benchmarks/asr-model-matrix.md: strip punctuation, collapse spaces,
lowercase; CER on the space-free string. A "hotword hallucination" is a primed
hotword that appears in the output but not in the reference.

  python hotword_ab.py --model M.tflite --tokenizer T.json --label tiny-off --out DIR
  python hotword_ab.py ... --hotwords-file WhisperHotwords.txt --label tiny-hw
  python hotword_ab.py --summarize --out DIR
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import whisper_tflite_runner as w  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DRONE = os.path.abspath(os.path.join(ROOT, "..", "..", "_contents", "DroneVRV2"))
WEB = os.path.abspath(os.path.join(ROOT, "..", "VoiceCommand-WebServer", "assets"))
PUNCT = re.compile(r"[.,!?\"'()\-:;·、。，]")


def norm(s):
    return " ".join(PUNCT.sub("", s).lower().split())


def cer(ref, hyp):
    r, h = ref.replace(" ", ""), hyp.replace(" ", "")
    d = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hc in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rc != hc))
    return d[len(h)] / max(1, len(r))


def ffmpeg_wav(src, dst):
    if not os.path.isfile(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        exe = os.environ.get("FFMPEG", "ffmpeg")
        subprocess.run([exe, "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "16000", dst], check=True)
    return dst


def build_sets(cache):
    """[(set, path, reference)]"""
    items = []
    from datasets import load_from_disk  # the 40-clip TTS gate set lives as an HF dataset
    gate = load_from_disk(os.path.join(ROOT, "External", "acft-training", "data", "acft_mix", "eval_short"))
    for i, row in enumerate(gate):
        p = os.path.join(cache, "tts40", f"{i:02d}.wav")
        if not os.path.isfile(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            sf.write(p, np.asarray(row["audio"]["array"], dtype=np.float32), row["audio"]["sampling_rate"])
        items.append(("tts40", p, row["text"]))
    test_audio = os.path.join(ROOT, "Assets", "StreamingAssets", "TestAssets", "Audio")
    for p in sorted(glob.glob(os.path.join(test_audio, "volume-*.mp3"))):
        items.append(("real4", p, os.path.basename(p)[len("volume-"):-4]))
    for p in sorted(glob.glob(os.path.join(WEB, "BenchmarkAudio", "*.mp3"))):
        items.append(("bench13", p, os.path.basename(p)[:-4].removeprefix("volume-")))
    noise = json.load(open(os.path.join(WEB, "NoiseUnitTest", "expected.json"), encoding="utf-8"))["cases"]
    for c in noise:
        src = os.path.join(WEB, "NoiseUnitTest", c["audio"])
        items.append(("noise9", ffmpeg_wav(src, os.path.join(cache, "noise9", c["audio"][:-4] + ".wav")), c["text"]))
    # In-vocabulary set: DroneVRV2 voice_tts_check edge-tts cache (wargame commands).
    # ponytail: every 3rd utterance with the voice rotating -> 156 clips; use all 468x3 if variance matters.
    voices = ["sunhi", "injoon", "hyunsu"]
    names = sorted(os.listdir(os.path.join(DRONE, "VoiceTtsCheck", "tts", "sunhi")))
    for k, n in enumerate(names[::3]):
        items.append(("inlist156", os.path.join(DRONE, "VoiceTtsCheck", "tts", voices[k % 3], n), n[:-4]))
    return items


def run_arm(args):
    from tokenizers import Tokenizer
    os.makedirs(args.out, exist_ok=True)
    tok = Tokenizer.from_file(args.tokenizer)
    m = w.load_model(args.model)
    prompt, eot = w.forced_prefix(m["vocab"], "ko")
    primed = []
    if args.hotwords_file:
        budget = w.SEQ - len(prompt) - 1 - args.max_new_tokens
        prefix = w.hotword_prefix(tok, w.load_hotwords(args.hotwords_file), budget)
        primed = [x for x in tok.decode(prefix[1:]).split() if "�" not in x]
        prompt = prefix + prompt
    out_path = os.path.join(args.out, f"{args.label}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for set_name, path, ref in build_sets(os.path.join(args.out, "cache")):
            data = w.load_audio(path)
            ids, enc_s, dec_s = w.transcribe(m, data, prompt, eot)
            hyp = tok.decode(ids)
            f.write(json.dumps({
                "label": args.label, "set": set_name, "audio": os.path.basename(path), "ref": ref, "hyp": hyp,
                "audio_s": round(len(data) / w.SR, 2), "steps": len(ids), "prompt_tokens": len(prompt),
                "hit_cap": len(prompt) + len(ids) >= w.SEQ, "encode_s": round(enc_s, 3), "decode_s": round(dec_s, 3),
                "primed": primed,
            }, ensure_ascii=False) + "\n")
            f.flush()
    print(out_path)


def summarize(out):
    rows = [json.loads(l) for p in sorted(glob.glob(os.path.join(out, "*.jsonl")))
            for l in open(p, encoding="utf-8") if l.strip()]
    primed = sorted({x for r in rows for x in r["primed"]}, key=len, reverse=True)

    def hw_in(s):
        s = s.replace(" ", "")
        return {h for h in primed if h in s}

    for r in rows:
        nr, nh = norm(r["ref"]), norm(r["hyp"])
        r["cer"], r["cer_raw"] = cer(nr, nh), cer(r["ref"].lower(), r["hyp"].strip().lower())
        r["exact"], r["exact_nospace"] = nr == nh, nr.replace(" ", "") == nh.replace(" ", "")
        r["halluc"] = sorted(hw_in(nh) - hw_in(nr))
        r["in_vocab"] = bool(hw_in(nr))
    labels = sorted({r["label"] for r in rows})
    groups = [("tts40", lambda r: r["set"] == "tts40"), ("real4", lambda r: r["set"] == "real4"),
              ("bench13", lambda r: r["set"] == "bench13"), ("noise9", lambda r: r["set"] == "noise9"),
              ("inlist156: primed word in ref", lambda r: r["set"] == "inlist156" and r["in_vocab"]),
              ("inlist156: no primed word", lambda r: r["set"] == "inlist156" and not r["in_vocab"]),
              ("out-of-list probe (bench13+noise9, no primed word)",
               lambda r: r["set"] in ("bench13", "noise9") and not r["in_vocab"])]
    lines = ["# Whisper hotword-prefix A/B", "",
             f"Primed hotwords ({len(primed)}): {' '.join(sorted(primed))}", "",
             "| set | arm | n | CER | CER raw | exact | exact (no space) | halluc clips | cap hits | mean steps | mean dec s |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for gname, pred in groups:
        for lab in labels:
            g = [r for r in rows if r["label"] == lab and pred(r)]
            if not g:
                continue
            lines.append(f"| {gname} | {lab} | {len(g)} | {np.mean([r['cer'] for r in g]):.3f} | "
                         f"{np.mean([r['cer_raw'] for r in g]):.3f} | {sum(r['exact'] for r in g)} | "
                         f"{sum(r['exact_nospace'] for r in g)} | {sum(bool(r['halluc']) for r in g)} | "
                         f"{sum(r['hit_cap'] for r in g)} | {np.mean([r['steps'] for r in g]):.1f} | "
                         f"{np.mean([r['decode_s'] for r in g]):.2f} |")
    lines += ["", "## Hallucinated hotwords (hyp has a primed word the ref lacks)", ""]
    for r in rows:
        if r["halluc"]:
            lines.append(f"- {r['label']} [{r['set']}] ref `{r['ref']}` → hyp `{r['hyp'].strip()}` ({', '.join(r['halluc'])})")
    lines += ["", "## Real-recording clips (real4 / bench13 / noise9), per arm", "",
              "| set | ref | " + " | ".join(labels) + " |", "|---|---|" + "---|" * len(labels)]
    key = lambda r: (r["set"], r["audio"])  # noqa: E731
    by = {(key(r), r["label"]): r for r in rows}
    for k in sorted({key(r) for r in rows if r["set"] in ("real4", "bench13", "noise9")}):
        ref = next(r["ref"] for r in rows if key(r) == k)
        cells = [by[(k, lab)]["hyp"].strip() if (k, lab) in by else "" for lab in labels]
        lines.append(f"| {k[0]} | {ref} | " + " | ".join(cells) + " |")
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines[:4 + 2 + len(groups) * len(labels) + 2]))


def _selftest():
    assert norm("볼륨, 업.") == "볼륨 업"
    assert cer("볼륨 업", "볼륨업") == 0.0 and abs(cer("축소", "축소해") - 0.5) < 1e-9


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    _selftest()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model")
    ap.add_argument("--tokenizer")
    ap.add_argument("--hotwords-file", default="")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--label")
    ap.add_argument("--out", default=os.path.join(ROOT, "Builds", "Logs", "whisper-hotword-ab"))
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    summarize(a.out) if a.summarize else run_arm(a)
    print(f"done in {time.time() - t0:.0f}s", file=sys.stderr)
