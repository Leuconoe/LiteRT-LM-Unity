"""Qwen3-TTS CustomVoice, Korean preset speaker(s), one variant per speaker.

Run with oss_tts/.venv-qwen3:
  python synth_qwen3.py [--model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice] [--speakers Sohee] [--limit N]
Existing outputs are skipped (delete a file to regenerate it).
"""

from __future__ import annotations

import argparse
import sys
import time

import oss_common as oc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    ap.add_argument("--speakers", default="Sohee")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--instruct", default="", help="style instruction (CustomVoice), e.g. an urgent radio delivery")
    ap.add_argument("--tag", default="", help="variant suffix for an --instruct run")
    args = ap.parse_args()

    import torch
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(args.model, device_map="cuda:0", dtype=torch.bfloat16)
    print("speakers:", model.get_supported_speakers(), "| languages:", model.get_supported_languages(), flush=True)
    todo = oc.jobs()[: args.limit or None]
    for spk in args.speakers.split(","):
        variant = f"qwen3-{spk.lower()}" + (f"-{args.tag}" if args.tag else "")
        pending = [j for j in todo if not oc.wav_path(variant, j["id"]).exists()]
        t0 = time.perf_counter()
        for i in range(0, len(pending), args.batch):
            chunk = pending[i:i + args.batch]
            wavs, sr = model.generate_custom_voice(text=[j["spoken_text"] for j in chunk], speaker=[spk] * len(chunk),
                                                   language=["Korean"] * len(chunk), instruct=[args.instruct] * len(chunk))
            for j, w in zip(chunk, wavs):
                oc.save16(oc.wav_path(variant, j["id"]), w, sr)
        print(f"{variant}: {len(pending)} new clips, {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
