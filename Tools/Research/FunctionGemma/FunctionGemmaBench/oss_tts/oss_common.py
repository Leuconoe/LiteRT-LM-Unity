"""Shared bits for the open-source TTS voices of the locked heldout5 voice eval.

Stdlib + soundfile/torch only, so every engine venv can import it.
Jobs = the unique (id, spoken_text) pairs of agent T's manifest (TTS-normalised text,
Hangul numbers) — every engine reads the same sentences.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
H5 = ROOT / "External" / "fg-voice-h5"
OUT = H5 / "oss"
SRC = ROOT / "External" / "oss-tts"


def jobs() -> list[dict]:
    seen, out = set(), []
    for line in (H5 / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        m = json.loads(line)
        if m["id"] not in seen:
            seen.add(m["id"])
            out.append({"id": m["id"], "text": m["text"], "spoken_text": m["spoken_text"]})
    return out


def wav_path(variant: str, cid: str) -> Path:
    return OUT / variant / f"{cid}.wav"


def save16(path: Path, wav, sr: int) -> None:
    """Mono 16 kHz s16 like every other clip in the eval. `wav` = 1-D or [1, N] float tensor/array."""
    import soundfile as sf
    import torch
    import torchaudio.functional as F

    t = torch.as_tensor(wav, dtype=torch.float32).reshape(-1)
    if sr != 16000:
        t = F.resample(t, sr, 16000)
    peak = float(t.abs().max()) if t.numel() else 0.0
    if peak > 0.99:  # some engines overshoot; avoid wrap-around in s16
        t = t * (0.99 / peak)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), t.numpy(), 16000, subtype="PCM_16")


if __name__ == "__main__":
    j = jobs()
    assert len(j) == len({x["id"] for x in j}) == 297, len(j)
    assert all(x["spoken_text"] for x in j)
    print("ok —", len(j), "jobs")
