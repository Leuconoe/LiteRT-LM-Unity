"""Export the ACFT Zeroth-Korean + FLEURS-en data to wav + FunASR SenseVoice jsonl.

Source: External/acft-training/data/acft_mix (datasets.save_to_disk; built by the ACFT
program's prep_dataset.py from Bingsu/zeroth-korean and google/fleurs en_us).
Writes <work>/data/{train,dev}.jsonl and <work>/data/wav/<split>/<i>.wav (16 kHz PCM16).

  train = 95 % of zeroth train (<|ko|>, <|woitn|>) + FLEURS-en train capped at
          --en-share of the total (<|en|>, <|withitn|> — raw_transcription is cased and
          punctuated). The English share is the forgetting guard (komixv2 lesson).
  dev   = the other 5 % of zeroth train. The zeroth TEST split is never touched here.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000


def decode(audio: dict) -> np.ndarray:
    wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        import librosa  # funasr dependency
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav


def record(key: str, path: Path, n_samples: int, text: str, lang: str, itn: str) -> dict:
    return {
        "key": key, "source": str(path.resolve()), "source_len": int(n_samples / SR * 100),
        "target": text, "target_len": len(text), "text_language": lang,
        "emo_target": "<|NEUTRAL|>", "event_target": "<|Speech|>", "with_or_wo_itn": itn,
    }


def export(ds, split: str, wav_dir: Path, lang: str, itn: str, keep=None) -> list[dict]:
    out_dir = wav_dir / split
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in (keep if keep is not None else range(len(ds))):
        ex = ds[i]
        text = " ".join(ex["text"].split())
        if not text:
            continue
        path = out_dir / f"{i:06d}.wav"
        if path.exists():
            n = sf.info(str(path)).frames
        else:
            wav = decode(ex["audio"])
            sf.write(str(path), wav, SR, subtype="PCM_16")
            n = len(wav)
        rows.append(record(f"{split}_{i:06d}", path, n, text, lang, itn))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="External/acft-training/data/acft_mix")
    ap.add_argument("--work", default="External/sensevoice-work")
    ap.add_argument("--en-share", type=float, default=0.15)
    ap.add_argument("--dev-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import Audio, load_from_disk

    dd = load_from_disk(args.src)
    data_dir = Path(args.work) / "data"
    wav_dir = data_dir / "wav"
    rnd = random.Random(args.seed)

    z = dd["train"].cast_column("audio", Audio(decode=False))
    idx = list(range(len(z)))
    rnd.shuffle(idx)
    n_dev = int(len(idx) * args.dev_frac)
    dev_idx, train_idx = sorted(idx[:n_dev]), sorted(idx[n_dev:])
    assert not set(dev_idx) & set(train_idx)

    ko_train = export(z, "zeroth_train", wav_dir, "<|ko|>", "<|woitn|>", train_idx)
    ko_dev = export(z, "zeroth_train", wav_dir, "<|ko|>", "<|woitn|>", dev_idx)

    en = dd["train_en"].cast_column("audio", Audio(decode=False))
    n_en = min(len(en), int(len(ko_train) * args.en_share / (1 - args.en_share)))
    en_idx = sorted(rnd.sample(range(len(en)), n_en))
    en_train = export(en, "fleurs_en_train", wav_dir, "<|en|>", "<|withitn|>", en_idx)

    train = ko_train + en_train
    rnd.shuffle(train)
    write_jsonl(data_dir / "train.jsonl", train)
    write_jsonl(data_dir / "dev.jsonl", ko_dev)

    hours = lambda rows: sum(r["source_len"] for r in rows) / 100 / 3600
    print(f"train {len(train)} ({len(ko_train)} ko {hours(ko_train):.1f} h + {len(en_train)} en {hours(en_train):.1f} h)"
          f" | dev {len(ko_dev)} ({hours(ko_dev):.1f} h) | >20.48 s dropped by FunASR: "
          f"{sum(r['source_len'] > 2048 for r in train)}")


if __name__ == "__main__":
    main()
