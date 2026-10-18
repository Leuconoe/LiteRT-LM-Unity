"""MeloTTS-Korean (myshell-ai/MeloTTS-Korean), its single KR speaker, one variant per speed.

Run with oss_tts/.venv-melo:  python synth_melo.py [--speeds 1.0,0.9,1.15] [--limit N]
Existing outputs are skipped (delete a file to regenerate it).
"""

from __future__ import annotations

import argparse
import sys
import time

import oss_common as oc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speeds", default="1.0,0.9,1.15")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from melo.api import TTS

    model = TTS(language="KR", device="cuda:0")
    spk = model.hps.data.spk2id["KR"]
    todo = oc.jobs()[: args.limit or None]
    for speed in map(float, args.speeds.split(",")):
        variant = f"melo-s{speed:g}"
        t0, n = time.perf_counter(), 0
        for j in todo:
            out = oc.wav_path(variant, j["id"])
            if out.exists():
                continue
            audio = model.tts_to_file(j["spoken_text"], spk, None, speed=speed, quiet=True)
            oc.save16(out, audio, model.hps.data.sampling_rate)
            n += 1
        print(f"{variant}: {n} new clips, {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
