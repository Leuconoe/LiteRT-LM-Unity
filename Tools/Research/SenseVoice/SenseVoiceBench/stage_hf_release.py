"""Stage a Hugging Face release folder from converted variants (does NOT upload).

Copies <work>/<kind>/sv_t{N}.tflite as sensevoice_small_{secs}s_{kind}.tflite, the side files,
writes tokens.txt (id<TAB>piece, for runtimes without SentencePiece) and SHA256SUMS.

  python stage_hf_release.py --work External/sensevoice-work/tflite/stock --kinds w8,i8 \
      --out External/sensevoice-work/hf-release/sensevoice-small-litert
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

SECONDS = {50: 3, 84: 5, 167: 10, 500: 30}  # LFR frames are 60 ms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", required=True)
    ap.add_argument("--kinds", default="w8,i8")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import sentencepiece as spm

    work, out = Path(a.work), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for kind in a.kinds.split(","):
        for f in sorted((work / kind).glob("sv_t*.tflite")):
            n = int(f.stem[4:])
            shutil.copy2(f, out / f"sensevoice_small_{SECONDS[n]}s_{kind}.tflite")
    side = work / a.kinds.split(",")[0]
    for f in ["am.mvn", "config.yaml", *[p.name for p in side.glob("*.bpe.model")]]:
        shutil.copy2(side / f, out / f)
    sp = spm.SentencePieceProcessor(model_file=str(next(side.glob("*.bpe.model"))))
    (out / "tokens.txt").write_text("".join(f"{i}\t{sp.id_to_piece(i)}\n" for i in range(sp.get_piece_size())),
                                    encoding="utf-8")
    sums = []
    for f in sorted(out.iterdir()):
        if f.is_file() and f.name not in ("SHA256SUMS", "README.md"):
            sums.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n")
    (out / "SHA256SUMS").write_text("".join(sums), encoding="utf-8")
    for f in sorted(out.iterdir()):
        print(f"{f.stat().st_size / 2**20:9.1f} MB  {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
