"""Acoustic augmentation, manifest and QC for the open-source TTS voices of the locked heldout5.

Run with the SenseVoiceBench venv (numpy/scipy/librosa/soundfile):
  python build_oss.py augment            # speed 0.9/1.15, pink|brown noise at SNR 10/5 dB, RT60 0.4 s reverb
  python build_oss.py manifest           # External/fg-voice-h5/manifest_oss.jsonl (T's schema + engine, transform)
  python build_oss.py qc --asr <asr-svpt.jsonl> [--delete | --final]
      # STT intelligibility gate (CER > 0.6) + duration/length checks on TTS clips; --delete to regenerate
      # once, --final to drop what still fails and exclude voices under 80 % pass (qc_report.json)
Noise is synthetic only: the NoiseUnitTest recordings have no long speech-free stretches to cut.
Every transform is seeded from crc32(id + transform), so reruns reproduce the same audio.
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

import oss_common as oc

SR = 16000
AUG_BASES = ("melo-s1", "cosy-m1")  # one clean MeloTTS and one CosyVoice native-reference voice
TRANSFORMS = ("speed0.9", "speed1.15", "noise10", "noise5", "reverb")
GROUPS = (("+speed", "aug-speed"), ("+noise10", "aug-noise10"), ("+noise5", "aug-noise5"), ("+reverb", "aug-reverb"),
          ("cosy-", "cosyvoice"), ("melo-", "melo"), ("qwen3-", "qwen3"))


def rng(cid: str, transform: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(f"{cid}|{transform}".encode()))


def colored_noise(n: int, alpha: float, g: np.random.Generator) -> np.ndarray:
    """1/f^alpha noise (alpha 1 = pink, 2 = brown) by spectral shaping of white noise."""
    spec = np.fft.rfft(g.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    x = np.fft.irfft(spec / f ** (alpha / 2), n)
    return x / (np.sqrt(np.mean(x ** 2)) + 1e-12)


def transform(y: np.ndarray, t: str, cid: str, idx: int) -> np.ndarray:
    g = rng(cid, t)
    if t.startswith("speed"):
        import librosa
        rate = float(t[5:])  # > 1 = faster, pitch kept
        # ponytail: librosa 1.0 phase_vocoder's float frame grid overruns the last frame for some lengths
        # at rate > 1; retry with a few more trailing zeros (trimmed off) instead of patching librosa
        for extra in (0, 512, 1024, 1536):
            try:
                out = librosa.effects.time_stretch(np.pad(y, (0, (-len(y)) % 512 + 2048 + extra)), rate=rate)
                return out[: int(round(len(y) / rate))]
            except librosa.util.exceptions.ParameterError:
                continue
        raise RuntimeError(f"time_stretch failed for {cid}")
    if t.startswith("noise"):
        snr = float(t[5:])
        n = colored_noise(len(y), 1.0 if idx % 2 == 0 else 2.0, g)  # alternate pink / brown per clip
        ps = np.mean(y ** 2)
        return y + n * np.sqrt(ps / 10 ** (snr / 10))
    if t == "reverb":
        from scipy.signal import fftconvolve
        rt60, length = 0.4, int(0.5 * SR)
        tt = np.arange(length) / SR
        h = g.standard_normal(length) * np.exp(-6.908 * tt / rt60)  # 60 dB decay over RT60
        h[0] = 1.0  # direct path
        h /= np.sqrt(np.sum(h ** 2))
        wet = fftconvolve(y, h)[: len(y) + int(0.3 * SR)]
        return wet * (np.max(np.abs(y)) / (np.max(np.abs(wet)) + 1e-12))
    raise ValueError(t)


def write(path: Path, y: np.ndarray) -> None:
    peak = np.max(np.abs(y))
    if peak > 0.99:
        y = y * (0.99 / peak)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y.astype(np.float32), SR, subtype="PCM_16")


def augment(force: bool) -> None:
    order = {j["id"]: i for i, j in enumerate(oc.jobs())}
    for base in AUG_BASES:
        for src in sorted((oc.OUT / base).glob("*.wav")):
            y, sr = sf.read(str(src), dtype="float32")
            assert sr == SR, src
            for t in TRANSFORMS:
                dst = oc.wav_path(f"{base}+{t}", src.stem)
                if force or not dst.exists():
                    write(dst, transform(y, t, src.stem, order[src.stem]))
        print(base, "augmented")


def group(variant: str) -> str:
    return next(g for key, g in GROUPS if key in variant)


def manifest() -> None:
    by_id = {j["id"]: j for j in oc.jobs()}
    drop = set(json.loads((oc.OUT / "qc_drop.json").read_text(encoding="utf-8"))) if (oc.OUT / "qc_drop.json").exists() else set()
    gone = excluded_voices()
    rows = []
    for d in sorted(p for p in oc.OUT.iterdir() if p.is_dir()):
        engine, _, tr = d.name.partition("+")
        if engine in gone:  # a voice that failed the intelligibility gate takes its augmentations with it
            continue
        for w in sorted(d.glob("*.wav")):
            if f"{d.name}/{w.stem}" in drop:
                continue
            j = by_id[w.stem]
            rows.append({"id": w.stem, "voice": d.name, "kind": group(d.name), "text": j["text"], "spoken_text": j["spoken_text"],
                         "path16k": str(w), "engine": engine.split("-")[0], "transform": tr or "none"})
    (oc.H5 / "manifest_oss.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    counts = defaultdict(int)
    for r in rows:
        counts[r["voice"]] += 1
    print(len(rows), "clips;", dict(counts))


def chars(s: str) -> int:
    return len("".join(ch for ch in s if ch.isalnum()))


MAX_CER, MIN_VOICE_PASS = 0.6, 0.8  # intelligibility gate (coordinator, after a garbled Supertonic voice in T's run)


def qc(asr_path: Path, delete: bool, final: bool) -> None:
    """Gate TTS clips only (augmented clips are the test condition, they follow their base clip):
    flag when SenseVoice CER vs spoken_text > MAX_CER, when seconds-per-character is far off the
    voice's median, or when SenseVoice hears far more/less text than was spoken (spurious lead/tail
    speech, truncation). `--delete` removes flagged clips so the synth scripts regenerate them once;
    `--final` writes qc_drop.json (still-flagged clips + their augmentations) and excludes every voice
    whose pass rate is below MIN_VOICE_PASS (qc_report.json)."""
    rows = [json.loads(l) for l in asr_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if "+" not in r["voice"]]
    spc = defaultdict(list)
    for r in rows:
        r["spc"] = sf.info(str(oc.wav_path(r["voice"], r["id"]))).duration / max(1, chars(r["spoken_text"]))
        spc[r["voice"]].append(r["spc"])
    med = {v: float(np.median(x)) for v, x in spc.items()}
    for r in rows:
        r["flag"] = (r["cer"] > MAX_CER or not (0.4 <= r["spc"] / med[r["voice"]] <= 2.5)
                     or not (0.35 <= chars(r["hyp"]) / max(1, chars(r["spoken_text"])) <= 2.0))
    report = {}
    for v in sorted(med):
        vr = [r for r in rows if r["voice"] == v]
        ok = sum(not r["flag"] for r in vr)
        report[v] = {"clips": len(vr), "pass": ok, "rate": round(ok / len(vr), 3), "excluded": ok / len(vr) < MIN_VOICE_PASS}
    flags = sorted(f"{r['voice']}/{r['id']}" for r in rows if r["flag"])
    (oc.OUT / "qc_flags.json").write_text(json.dumps(flags, ensure_ascii=False, indent=1), encoding="utf-8")
    (oc.OUT / "qc_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for v, x in report.items():
        print(f"{v}: {x['pass']}/{x['clips']} pass ({x['rate']:.1%}){'  EXCLUDED' if x['excluded'] else ''}")
    if delete:
        for f in flags:
            v, cid = f.split("/")
            oc.wav_path(v, cid).unlink(missing_ok=True)
            for t in TRANSFORMS:
                oc.wav_path(f"{v}+{t}", cid).unlink(missing_ok=True)
    if final:
        drop = flags + [f"{f.split('/')[0]}+{t}/{f.split('/')[1]}" for f in flags for t in TRANSFORMS]
        (oc.OUT / "qc_drop.json").write_text(json.dumps(sorted(drop), ensure_ascii=False, indent=1), encoding="utf-8")


def excluded_voices() -> set[str]:
    p = oc.OUT / "qc_report.json"
    return {v for v, x in json.loads(p.read_text(encoding="utf-8")).items() if x["excluded"]} if p.exists() else set()


def _selfcheck() -> None:
    g = np.random.default_rng(0)
    y = np.sin(2 * np.pi * 220 * np.arange(SR) / SR).astype(np.float32) * 0.5
    noisy = transform(y, "noise10", "x", 0)
    snr = 10 * np.log10(np.mean(y ** 2) / np.mean((noisy - y) ** 2))
    assert abs(snr - 10) < 0.5, snr
    assert np.allclose(transform(y, "noise5", "x", 1), transform(y, "noise5", "x", 1))  # deterministic
    assert abs(len(transform(y, "speed1.15", "x", 0)) - SR / 1.15) < 0.02 * SR
    assert len(transform(y, "reverb", "x", 0)) == SR + int(0.3 * SR)
    assert group("melo-s1+noise5") == "aug-noise5" and group("cosy-f2") == "cosyvoice" and group("qwen3-sohee") == "qwen3"
    del g


def main() -> int:
    _selfcheck()
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["augment", "manifest", "qc"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--asr", type=Path)
    ap.add_argument("--delete", action="store_true")
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    if args.stage == "augment":
        augment(args.force)
    elif args.stage == "manifest":
        manifest()
    else:
        qc(args.asr, args.delete, args.final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
