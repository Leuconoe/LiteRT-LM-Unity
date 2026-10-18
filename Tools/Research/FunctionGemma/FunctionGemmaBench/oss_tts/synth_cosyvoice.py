"""Fun-CosyVoice3-0.5B zero-shot Korean voices, one variant per real Korean reference speaker.

References are Zeroth-Korean train clips (CC BY 4.0) chosen by median F0 for a male/female mix
(see refs.json). Run with oss_tts/.venv-cosyvoice:
  python synth_cosyvoice.py [--refs refs.json] [--only cosy-m1,cosy-f1] [--limit N]
Existing outputs are skipped, so a rerun only fills gaps (delete a file to regenerate it).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import oss_common as oc

COSY = oc.SRC / "CosyVoice"
sys.path[:0] = [str(COSY), str(COSY / "third_party" / "Matcha-TTS")]
MODEL_DIR = oc.SRC / "models" / "Fun-CosyVoice3-0.5B"
PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"  # CosyVoice3 prompt format (example.py)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default=str(Path(__file__).with_name("refs.json")))
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--half", action="store_true", help="every 2nd case (149) — RTF ~1.5 makes all 297 x 6 voices ~2 h")
    ap.add_argument("--rl", action="store_true", help="use llm.rl.pt (Fun-CosyVoice3-0.5B-2512_RL, lower CER upstream)")
    args = ap.parse_args()

    if not (MODEL_DIR / "cosyvoice3.yaml").exists():
        from huggingface_hub import snapshot_download
        snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", local_dir=str(MODEL_DIR))
    model_dir = MODEL_DIR
    if args.rl:  # the loader hard-codes llm.pt; a hard-linked copy of the dir swaps in llm.rl.pt (the _RL model)
        model_dir = MODEL_DIR.with_name(MODEL_DIR.name + "-RL")
        if not (model_dir / "llm.pt").exists():
            import os
            for p in MODEL_DIR.rglob("*"):
                if p.is_file() and ".cache" not in p.parts:
                    dst = model_dir / p.relative_to(MODEL_DIR)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    os.link(MODEL_DIR / "llm.rl.pt" if p.name == "llm.pt" else p, dst)
    from cosyvoice.cli.cosyvoice import AutoModel

    model = AutoModel(model_dir=str(model_dir), fp16=args.fp16)
    refs = json.loads(Path(args.refs).read_text(encoding="utf-8"))
    if args.only:
        refs = [r for r in refs if r["variant"] in args.only.split(",")]
    todo = oc.jobs()[:: 2 if args.half else 1][: args.limit or None]  # cases are grouped by suite, so [::2] keeps the mix
    for ref in refs:
        assert model.add_zero_shot_spk(PROMPT_PREFIX + ref["text"], str(oc.ROOT / ref["wav"]), ref["variant"])
        t0, n = time.perf_counter(), 0
        for j in todo:
            out = oc.wav_path(ref["variant"], j["id"])
            if out.exists():
                continue
            chunks = [c["tts_speech"] for c in model.inference_zero_shot(j["spoken_text"], "", "", zero_shot_spk_id=ref["variant"], stream=False)]
            import torch
            oc.save16(out, torch.cat(chunks, dim=1), model.sample_rate)
            n += 1
        print(f"{ref['variant']}: {n} new clips, {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
