#!/usr/bin/env python
"""Score Moonshine (and stock Whisper, same harness) on this project's Korean ASR sets.

Buckets
  ko_short   40-clip TTS-synthesized short-command holdout
             (External/acft-training/data/acft_mix/eval_short). RANKING gate only —
             ~2.8x pessimistic in absolute CER for a strong base; see
             docs/handoffs/asr-training-program-handoff.md §3.
  real_cmd   the Korean clips under Assets/StreamingAssets/TestAssets/Audio
             (device-recorded / product clips; reference = file name).
  fleurs_ko  FLEURS ko validation (External/acft-training/data/acft_mix/eval_fleurs_ko),
             real read speech, for the absolute-level check against the model card.

CER = the training queue's definition (run_queue.py:126): spaces removed, lowercased,
Levenshtein / len(ref). Punctuation is NOT removed there, so `cer_nopunct` is also
reported — the shipped matcher (LiteRtLmAsrTestRunner.cs) discards punctuation.

Latency is an RTX 4090 number and says nothing about kona; it is recorded only to
show relative model cost.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ACFT_DATA = PROJECT_ROOT / "External" / "acft-training" / "data" / "acft_mix"
TEST_AUDIO = PROJECT_ROOT / "Assets" / "StreamingAssets" / "TestAssets" / "Audio"

MODEL_ALIASES = {
    "moonshine-tiny-ko": "UsefulSensors/moonshine-tiny-ko",
    "moonshine-base-ko": "UsefulSensors/moonshine-base-ko",
    "whisper-tiny": "openai/whisper-tiny",
    "whisper-base": "openai/whisper-base",
    "whisper-small": "openai/whisper-small",
    "whisper-turbo": "openai/whisper-large-v3-turbo",
}

_PUNCT = re.compile(r"[\s\.,!\?;:~…\-–—'\"“”‘’()\[\]{}<>/\\·]+")


def cer(ref: str, hyp: str) -> float:
    """run_queue.py definition, verbatim: spaces out, lowercase, Levenshtein."""
    r = ref.replace(" ", "").lower()
    h = hyp.replace(" ", "").lower()
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(r)


def strip_punct(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    return _PUNCT.sub("", s)


def cer_nopunct(ref: str, hyp: str) -> float:
    return cer(strip_punct(ref), strip_punct(hyp))


# ---------------------------------------------------------------- audio loading

def load_audio(path: Path, sr: int = 16000) -> np.ndarray:
    import soundfile as sf

    try:
        pcm, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:  # libsndfile without mp3 → librosa/audioread fallback
        import librosa

        pcm, file_sr = librosa.load(str(path), sr=None, mono=False)
        pcm = np.asarray(pcm, dtype=np.float32)
        pcm = pcm.T if pcm.ndim == 2 else pcm[:, None]
    mono = pcm.mean(axis=1)
    if file_sr != sr:
        import librosa

        mono = librosa.resample(mono, orig_sr=file_sr, target_sr=sr)
    return mono.astype(np.float32)


# ---------------------------------------------------------------- buckets

def _decode_hf_audio(audio: dict, sr: int = 16000) -> np.ndarray:
    """Decode a datasets Audio cell without torchcodec (bytes or path → mono float32 @ sr)."""
    import io

    import soundfile as sf

    if audio.get("bytes"):
        pcm, file_sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
    else:
        return load_audio(Path(audio["path"]), sr)
    mono = pcm.mean(axis=1)
    if file_sr != sr:
        import librosa

        mono = librosa.resample(mono, orig_sr=file_sr, target_sr=sr)
    return mono.astype(np.float32)


def _iter_hf_bucket(name: str, subdir: str, limit: int | None, width: int):
    from datasets import Audio, load_from_disk

    ds = load_from_disk(str(ACFT_DATA / subdir))
    ds = ds.cast_column("audio", Audio(decode=False))
    n = len(ds) if limit is None else min(limit, len(ds))
    for i in range(n):
        row = ds[i]
        yield f"{name}/{i:0{width}d}", _decode_hf_audio(row["audio"]), row["text"]


def bucket_ko_short(limit: int | None):
    yield from _iter_hf_bucket("ko_short", "eval_short", limit, 2)


def bucket_fleurs_ko(limit: int | None):
    yield from _iter_hf_bucket("fleurs_ko", "eval_fleurs_ko", limit, 3)


def bucket_real_cmd(limit: int | None):
    files = sorted(p for p in TEST_AUDIO.glob("*.mp3") if re.search(r"[가-힣]", p.stem))
    if limit is not None:
        files = files[:limit]
    for p in files:
        ref = p.stem
        ref = re.sub(r"^[A-Za-z]+-", "", ref)  # "volume-볼륨 업" → "볼륨 업"
        yield f"real_cmd/{p.stem}", load_audio(p), ref


BUCKETS = {
    "ko_short": bucket_ko_short,
    "real_cmd": bucket_real_cmd,
    "fleurs_ko": bucket_fleurs_ko,
}


# ---------------------------------------------------------------- models

class Transcriber:
    def __init__(self, model_id: str, device: str, dtype: torch.dtype, pad_to_s: float = 0.0,
                 head_pad_s: float = 0.0, peak_norm: bool = False):
        from transformers import AutoConfig

        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        # Moonshine-only preprocessing knobs (Whisper's processor pads/normalizes itself)
        self.pad_to_s = pad_to_s
        self.head_pad_s = head_pad_s
        self.peak_norm = peak_norm
        cfg = AutoConfig.from_pretrained(model_id)
        self.kind = cfg.model_type  # "moonshine" | "whisper"
        if self.kind == "moonshine":
            from transformers import AutoProcessor, MoonshineForConditionalGeneration

            self.model = MoonshineForConditionalGeneration.from_pretrained(model_id)
            self.processor = AutoProcessor.from_pretrained(model_id)
            # model card: 13 tokens / s hard cap against hallucination loops
            self.token_limit_factor = 13 / self.processor.feature_extractor.sampling_rate
        elif self.kind == "whisper":
            from transformers import AutoProcessor, WhisperForConditionalGeneration

            self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
            self.processor = AutoProcessor.from_pretrained(model_id)
        else:
            raise SystemExit(f"unsupported model_type {self.kind} for {model_id}")
        self.model = self.model.to(device).to(dtype).eval()
        self.params = sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def transcribe(self, pcm: np.ndarray) -> tuple[str, float]:
        sr = 16000
        if self.kind == "moonshine":
            if self.peak_norm:
                peak = float(np.abs(pcm).max()) or 1.0
                pcm = pcm * (0.9 / peak)
            if self.head_pad_s:
                pcm = np.concatenate([np.zeros(int(self.head_pad_s * sr), dtype=np.float32), pcm])
            want = int(self.pad_to_s * sr)
            if want and len(pcm) < want:
                pcm = np.concatenate([pcm, np.zeros(want - len(pcm), dtype=np.float32)])
            inputs = self.processor(pcm, return_tensors="pt", sampling_rate=sr)
            inputs = inputs.to(self.device, self.dtype)
            seq_len = int(inputs.attention_mask.sum(dim=-1).max().item())
            max_length = max(8, int(seq_len * self.token_limit_factor))
            max_length = min(max_length, int(self.model.config.max_position_embeddings))
            t0 = time.perf_counter()
            ids = self.model.generate(**inputs, max_length=max_length)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            text = self.processor.decode(ids[0], skip_special_tokens=True)
        else:
            feats = self.processor(pcm, sampling_rate=sr, return_tensors="pt").input_features
            feats = feats.to(self.device, self.dtype)
            t0 = time.perf_counter()
            ids = self.model.generate(
                feats, language="ko", task="transcribe", max_new_tokens=128
            )
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        return text.strip(), dt


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["moonshine-tiny-ko", "moonshine-base-ko", "whisper-tiny", "whisper-base"],
                    help="aliases (%s) or HF ids" % ", ".join(MODEL_ALIASES))
    ap.add_argument("--buckets", nargs="+", default=["ko_short", "real_cmd", "fleurs_ko"], choices=sorted(BUCKETS))
    ap.add_argument("--limit", type=int, default=None, help="max clips per bucket (fleurs_ko is large)")
    ap.add_argument("--fleurs-limit", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--pad-to", type=float, default=0.0,
                    help="Moonshine only: zero-pad clips shorter than N seconds to N (Whisper always pads to 30 s; "
                         "the litert-community Moonshine package uses a fixed 5 s window)")
    ap.add_argument("--head-pad", type=float, default=0.0, help="Moonshine only: seconds of silence prepended")
    ap.add_argument("--peak-norm", action="store_true", help="Moonshine only: peak-normalize to 0.9 before inference")
    ap.add_argument("--out", required=True, help="results JSON")
    ap.add_argument("--transcripts", default=None, help="per-clip JSONL (default: <out>.clips.jsonl)")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clips_path = Path(args.transcripts) if args.transcripts else out_path.with_suffix(".clips.jsonl")

    # load buckets once (audio in memory; all three are small)
    data: dict[str, list[tuple[str, np.ndarray, str]]] = {}
    for b in args.buckets:
        lim = args.fleurs_limit if b == "fleurs_ko" else args.limit
        data[b] = list(BUCKETS[b](lim))
        print(f"[bucket] {b}: {len(data[b])} clips", flush=True)

    summary: dict = {"device": args.device, "dtype": args.dtype, "pad_to_s": args.pad_to,
                     "head_pad_s": args.head_pad, "peak_norm": args.peak_norm,
                     "cer_def": "run_queue.py:126 (spaces removed, lowercase)",
                     "buckets": {b: len(v) for b, v in data.items()}, "models": {}}
    clips_f = clips_path.open("w", encoding="utf-8")
    try:
        for alias in args.models:
            model_id = MODEL_ALIASES.get(alias, alias)
            print(f"\n[model] {alias} = {model_id}", flush=True)
            tr = Transcriber(model_id, args.device, dtype, pad_to_s=args.pad_to,
                             head_pad_s=args.head_pad, peak_norm=args.peak_norm)
            # warm-up
            tr.transcribe(np.zeros(16000, dtype=np.float32))
            mrow: dict = {"model_id": model_id, "params_M": round(tr.params / 1e6, 1), "buckets": {}}
            for b, rows in data.items():
                cers, cers_np, exact, lat, audio_s = [], [], 0, [], 0.0
                for clip_id, pcm, ref in rows:
                    hyp, dt = tr.transcribe(pcm)
                    c, cn = cer(ref, hyp), cer_nopunct(ref, hyp)
                    cers.append(c)
                    cers_np.append(cn)
                    exact += int(cn == 0.0)
                    lat.append(dt)
                    audio_s += len(pcm) / 16000
                    clips_f.write(json.dumps({"model": alias, "clip": clip_id, "ref": ref, "hyp": hyp,
                                              "cer": round(c, 4), "cer_nopunct": round(cn, 4),
                                              "sec": round(len(pcm) / 16000, 2), "latency_s": round(dt, 4)},
                                             ensure_ascii=False) + "\n")
                brow = {
                    "n": len(rows),
                    "cer": round(float(np.mean(cers)), 4),
                    "cer_nopunct": round(float(np.mean(cers_np)), 4),
                    "exact_nopunct": f"{exact}/{len(rows)}",
                    "latency_mean_s": round(float(np.mean(lat)), 4),
                    "rtf": round(float(np.sum(lat)) / audio_s, 4) if audio_s else None,
                }
                mrow["buckets"][b] = brow
                print(f"  {b:10s} n={brow['n']:3d}  CER={brow['cer']:.3f}  CER(no punct)={brow['cer_nopunct']:.3f}"
                      f"  exact={brow['exact_nopunct']}  lat={brow['latency_mean_s']*1000:.0f} ms  RTF={brow['rtf']}", flush=True)
            summary["models"][alias] = mrow
            del tr
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    finally:
        clips_f.close()

    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out_path} and {clips_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
