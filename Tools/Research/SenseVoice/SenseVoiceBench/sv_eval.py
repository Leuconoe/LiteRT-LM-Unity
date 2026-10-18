"""Score SenseVoice on the Korean sets the whisper tiers were scored on.

Sets (pick with --sets):
  gate40     40-clip TTS short-command holdout = the ACFT ko_short gate
             (External/acft-training/data/acft_mix/eval_short). RANKING ONLY —
             TTS, ~2.8x harsher than real audio in absolute CER.
  real_clean recorded Korean clips: TestAssets/Audio + WebServer BenchmarkAudio
  real_noise WebServer NoiseUnitTest (field noise)      real = both (deduplicated by hash)
  zeroth     zeroth-korean test (457).        fleurs_ko  FLEURS ko dev (64).

Backends (--model is generic):
  funasr  HF id or local funasr dir (config.yaml + model.pt). PyTorch reference.
  onnx    model.onnx path or its dir (needs am.mvn + *.bpe.model + config.yaml beside it).
  tflite  a .tflite or a dir of bucket graphs sv_t{N}.tflite (+ the same side files).

Scores, both per clip and as set means:
  gate_cer  train_acft.cer: spaces removed, lowercase, punctuation KEPT — the exact
            function behind the published gate numbers (turbo 0.182, base 0.305 ...).
  cer       punctuation stripped as well — what the shipped matcher sees.
  exact     cer == 0.
Front end runs with dither 0 everywhere (funasr's default is 1.0 = random per call).

  python sv_eval.py --backend funasr --model FunAudioLLM/SenseVoiceSmall --label stock
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ACFT_DATA = ROOT / "External" / "acft-training" / "data" / "acft_mix"
WEBSERVER_ASSETS = ROOT.parent / "VoiceCommand-WebServer" / "assets"
HF_SPLITS = {"gate40": "eval_short", "zeroth": "eval_full", "fleurs_ko": "eval_fleurs_ko",
             "fleurs_en": "eval_fleurs_en"}  # fleurs_en: English-preservation guard, run with --language en
# SenseVoiceSmall query ids (funasr models/sense_voice/model.py lid_dict / textnorm_dict).
LID = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
TEXTNORM = {True: 14, False: 15}
VOCAB = 25055
PUNCT = re.compile(r"[.,!?\"'()\-:;·、。，？！]")
TAG = re.compile(r"<\|[^|]*\|>")


# ---------------------------------------------------------------- scoring

def _lev(r: str, h: str) -> int:
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str, strip_punct: bool) -> float:
    if strip_punct:
        ref, hyp = PUNCT.sub("", ref), PUNCT.sub("", hyp)
    r, h = ref.replace(" ", "").lower(), hyp.replace(" ", "").lower()
    if not r:
        return 0.0 if not h else 1.0
    return _lev(r, h) / len(r)


# ---------------------------------------------------------------- audio

def decode_audio(src) -> np.ndarray:
    """Any container (path or bytes) -> float32 mono 16 kHz in [-1, 1]."""
    import av

    with av.open(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else str(src)) as c:
        rs = av.AudioResampler(format="flt", layout="mono", rate=16000)
        out = [f.to_ndarray().reshape(-1) for fr in c.decode(audio=0) for f in rs.resample(fr)]
        out += [f.to_ndarray().reshape(-1) for f in rs.resample(None)]
    return np.concatenate(out).astype(np.float32)


def load_set(name: str) -> list[dict]:
    if name in HF_SPLITS:
        from datasets import Audio, load_from_disk

        ds = load_from_disk(str(ACFT_DATA / HF_SPLITS[name])).cast_column("audio", Audio(decode=False))
        return [{"id": f"{name}/{i}", "ref": ex["text"], "wav": decode_audio(ex["audio"]["bytes"])}
                for i, ex in enumerate(ds)]
    if name not in ("real", "real_clean", "real_noise"):
        raise SystemExit(f"unknown set {name}")
    items: list[tuple[Path, str]] = []
    if name != "real_noise":  # quiet-room recordings
        for d in (ROOT / "Assets" / "StreamingAssets" / "TestAssets" / "Audio", WEBSERVER_ASSETS / "BenchmarkAudio"):
            for p in sorted(d.glob("*.mp3")):
                if re.search(r"[가-힣]", p.stem):  # Korean clips only — language is forced to ko
                    items.append((p, re.sub(r"^volume-", "", p.stem)))
    if name != "real_clean":  # field-noise recordings (fan, wind, vehicle, chatter)
        noise = WEBSERVER_ASSETS / "NoiseUnitTest"
        for case in json.loads((noise / "expected.json").read_text(encoding="utf-8"))["cases"]:
            items.append((noise / case["audio"], case["text"]))
    seen, out = set(), []
    for p, ref in items:
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append({"id": f"real/{p.parent.name}/{p.name}", "ref": ref, "wav": decode_audio(p)})
    return out


# ---------------------------------------------------------------- shared front end / CTC

class SideFiles:
    """am.mvn + bpe model + frontend_conf beside an exported graph -> funasr frontend + tokenizer."""

    def __init__(self, d: Path):
        import sentencepiece as spm
        import yaml
        from funasr.frontends.wav_frontend import WavFrontend

        conf = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
        # frontend_conf may carry a stale absolute cmvn_file from training — the side file wins.
        # dither 0: WavFrontend defaults to 1.0 (random noise per call), which made the funasr
        # reference non-deterministic and showed up as ~1.0 logit "parity" gaps. Device fbank has none.
        self.frontend = WavFrontend(**{**conf["frontend_conf"], "cmvn_file": str(d / "am.mvn"), "dither": 0.0})
        self.sp = spm.SentencePieceProcessor(model_file=str(next(d.glob("*.bpe.model"))))

    def feats(self, wav: np.ndarray) -> np.ndarray:
        import torch
        from funasr.utils.load_utils import extract_fbank

        speech, _ = extract_fbank([torch.from_numpy(wav)], data_type="sound", frontend=self.frontend)
        return speech[0].numpy().astype(np.float32)  # [T, 560]

    def ctc_text(self, logits: np.ndarray) -> str:
        ids = logits.argmax(-1)
        keep = [int(t) for i, t in enumerate(ids) if t != 0 and (i == 0 or t != ids[i - 1])]
        return self.sp.decode(keep)


def side_dir(model: Path) -> Path:
    return model if model.is_dir() else model.parent


class FunasrBackend:
    def __init__(self, model: str, device: str):
        from funasr import AutoModel

        kw = {"hub": "hf", "frontend_conf": {"dither": 0.0}}
        d = Path(model)
        if d.is_dir():  # local checkpoint: its config.yaml may point at the trainer's absolute paths
            kw = {"tokenizer_conf": {"bpemodel": str(next(d.glob("*.bpe.model")))},
                  "frontend_conf": {"cmvn_file": str(d / "am.mvn"), "dither": 0.0}}
        self.m = AutoModel(model=model, device=device, disable_update=True, **kw)

    def __call__(self, wav, lang, itn):
        r = self.m.generate(input=[wav], cache={}, language=lang, use_itn=itn, batch_size=1)
        return r[0]["text"], None


class OnnxBackend:
    def __init__(self, model: str):
        import onnxruntime as ort

        p = Path(model)
        p = p / "model.onnx" if p.is_dir() else p
        self.side = SideFiles(side_dir(p))
        self.s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        self.names = [i.name for i in self.s.get_inputs()]  # feats, lengths, language, textnorm

    def logits(self, f: np.ndarray, lang, itn) -> np.ndarray:
        feed = dict(zip(self.names, [f[None], np.array([len(f)], np.int32),
                                     np.array([LID[lang]], np.int32), np.array([TEXTNORM[itn]], np.int32)]))
        return self.s.run(None, feed)[0][0][: len(f) + 4]

    def __call__(self, wav, lang, itn):
        f = self.side.feats(wav)
        lg = self.logits(f, lang, itn)
        return self.side.ctc_text(lg), lg


class TfliteBackend:
    """Fixed-shape bucket graphs sv_t{N}.tflite; features zero-padded to N with the true length fed."""

    def __init__(self, model: str, threads: int, pattern: str = "sv_t*.tflite"):
        from ai_edge_litert.interpreter import Interpreter

        p = Path(model)
        files = sorted(p.glob(pattern)) if p.is_dir() else [p]  # release dir: sensevoice_small_*_w8.tflite
        self.side = SideFiles(side_dir(p))
        self.graphs = []
        for f in files:
            it = Interpreter(model_path=str(f), num_threads=threads)
            it.allocate_tensors()
            ins = it.get_input_details()
            feat = next(d for d in ins if len(d["shape"]) == 3)
            t = int(feat["shape"][1]) if feat["shape"][2] == 560 else int(feat["shape"][2])
            self.graphs.append((t, it, ins, feat))
        self.graphs.sort(key=lambda g: g[0])

    def logits(self, f: np.ndarray, lang, itn) -> np.ndarray:
        t, it, ins, feat = next((g for g in self.graphs if g[0] >= len(f)), self.graphs[-1])
        n = min(len(f), t)
        x = np.zeros((t, 560), np.float32)
        x[:n] = f[:n]
        x = x[None] if feat["shape"][2] == 560 else x.T[None]  # onnx2tf may transpose NCW->NWC
        it.set_tensor(feat["index"], x.astype(feat["dtype"]))
        # Bind scalars by NAME — converters reorder signature inputs (Supertonic lesson).
        for d in ins:
            if d is feat:
                continue
            nm = d["name"].lower()
            v = n if "length" in nm else LID[lang] if "lang" in nm else TEXTNORM[itn] if "norm" in nm else None
            assert v is not None, f"unexpected tflite input {d['name']}"
            it.set_tensor(d["index"], np.array([v], d["dtype"]).reshape(d["shape"]))
        it.invoke()
        out = it.get_tensor(it.get_output_details()[0]["index"])[0]
        out = out if out.shape[-1] == VOCAB else out.T
        return out[: n + 4]

    def __call__(self, wav, lang, itn):
        f = self.side.feats(wav)
        lg = self.logits(f, lang, itn)
        return self.side.ctc_text(lg), lg


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["funasr", "onnx", "tflite"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--sets", default="gate40,real_clean,real_noise,zeroth,fleurs_ko")
    ap.add_argument("--language", default="ko", choices=list(LID))
    ap.add_argument("--itn", default="off", choices=["off", "on", "both"])
    ap.add_argument("--pad-to", type=float, default=0, help="zero-pad audio to N seconds (padding sensitivity)")
    ap.add_argument("--ref-onnx", default="", help="tflite only: also run this ONNX and report logit parity")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import logging
    for noisy in ("httpx", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if a.backend == "funasr":
        import torch
        be = FunasrBackend(a.model, a.device if torch.cuda.is_available() else "cpu")
    elif a.backend == "onnx":
        be = OnnxBackend(a.model)
    else:
        be = TfliteBackend(a.model, a.threads)
    ref_be = OnnxBackend(a.ref_onnx) if a.ref_onnx else None

    label = a.label or Path(a.model).name
    out = Path(a.out or ROOT / "Builds" / "Logs" / f"sensevoice-{label}-{a.backend}{f'-pad{a.pad_to:g}' if a.pad_to else ''}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    itns = {"off": [False], "on": [True], "both": [False, True]}[a.itn]
    summary = []
    with out.open("w", encoding="utf-8") as fo:
        # PowerShell turns `a,b` into an array and joins it with a space — accept both.
        for set_name in re.split(r"[,\s]+", a.sets.strip()):
            clips = load_set(set_name)[: a.limit or None]
            for itn in itns:
                rows = []
                for c in clips:
                    wav = c["wav"]
                    if a.pad_to and len(wav) < a.pad_to * 16000:
                        wav = np.pad(wav, (0, int(a.pad_to * 16000) - len(wav)))
                    t0 = time.perf_counter()
                    raw, lg = be(wav, a.language, itn)
                    ms = (time.perf_counter() - t0) * 1000
                    hyp = TAG.sub("", raw).strip()
                    row = {"set": set_name, "id": c["id"], "itn": itn, "ref": c["ref"], "hyp": hyp,
                           "dur_s": round(len(c["wav"]) / 16000, 2), "ms": round(ms, 1),
                           "gate_cer": round(cer(c["ref"], hyp, False), 4), "cer": round(cer(c["ref"], hyp, True), 4)}
                    if ref_be is not None and lg is not None:
                        rl = ref_be.logits(ref_be.side.feats(wav), a.language, itn)
                        n = min(len(rl), len(lg))
                        row["max_abs_diff"] = float(np.abs(rl[:n] - lg[:n]).max())
                        row["argmax_agree"] = float((rl[:n].argmax(-1) == lg[:n].argmax(-1)).mean())
                        row["ref_hyp"] = TAG.sub("", ref_be.side.ctc_text(rl)).strip()
                    rows.append(row)
                    fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                ms_sorted = sorted(r["ms"] for r in rows[1:] or rows)  # first call = warm-up
                s = {"set": set_name, "itn": "on" if itn else "off", "n": len(rows),
                     "gate_cer": round(statistics.mean(r["gate_cer"] for r in rows), 4),
                     "cer": round(statistics.mean(r["cer"] for r in rows), 4),
                     "exact": sum(r["cer"] == 0 for r in rows),
                     "ms_p50": ms_sorted[len(ms_sorted) // 2], "ms_mean": round(statistics.mean(ms_sorted), 1)}
                if ref_be is not None:
                    s["max_abs_diff"] = round(max(r.get("max_abs_diff", 0) for r in rows), 5)
                    s["argmax_agree_min"] = round(min(r.get("argmax_agree", 1) for r in rows), 4)
                    s["hyp_equal_ref"] = sum(r["hyp"] == r.get("ref_hyp") for r in rows)
                summary.append(s)
                print(json.dumps(s, ensure_ascii=False), flush=True)
    print(f"\n## {label} / {a.backend}{f' / pad {a.pad_to:g}s' if a.pad_to else ''}")
    keys = [k for k in summary[0] if k not in ("set", "itn")]
    print("| set | itn | " + " | ".join(keys) + " |")
    print("|---|---|" + "---:|" * len(keys))
    for s in summary:
        print(f"| {s['set']} | {s['itn']} | " + " | ".join(str(s[k]) for k in keys) + " |")
    out.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"rows: {out}")
    return 0


if __name__ == "__main__":
    assert cer("볼륨 업.", "볼륨업", True) == 0 and cer("볼륨 업.", "볼륨업", False) == 0.25
    sys.exit(main())
