"""Checksum the Supertonic input tensors on the desktop, for device comparison.

The device produces speech but the wrong speech, and a wrong *layout* passes every
size check the runtime performs — so it cannot be found by reading error messages.
This prints the md5 of each tensor exactly as the desktop pipeline feeds it, in the
same byte order and dtype the JNI uses, so one device run identifies which input
diverges instead of a sequence of guesses.

Matches the JNI's `idsMd5` / `styleDpMd5` / `styleTtlMd5` / `textMaskMd5` /
`textEmbMd5`, which are computed over the *fed* buffers (i.e. after any NCW→NWC
transpose).

Usage:
  python dump_supertonic_input_md5.py --assets-dir <dir> --dynamic-dir <dir> \
      --voice F1.json --text "..." [--bucket 64]
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import supertonic_helper as helper
from supertonic_litert import TfliteGraph, pick, TRANSPOSED_INPUTS


def md5(array, dtype):
    return hashlib.md5(np.ascontiguousarray(array, dtype=dtype).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--dynamic-dir", required=True,
                        help="directory holding the dynamic duration_predictor/text_encoder")
    parser.add_argument("--voice", default="F1.json")
    parser.add_argument("--text", default="고도 백 미터로 상승합니다.")
    parser.add_argument("--lang", default="ko")
    parser.add_argument("--bucket", type=int, default=64)
    parser.add_argument("--speed", type=float, default=1.05)
    args = parser.parse_args()

    voice_path = args.voice if os.path.isabs(args.voice) else os.path.join(args.assets_dir, args.voice)
    processor = helper.load_text_processor(args.assets_dir)
    style = helper.load_voice_style([voice_path])

    ids, mask = processor([args.text], [args.lang])
    real = ids.shape[1]
    bucket = max(args.bucket, real)

    padded_ids = np.zeros((1, bucket), dtype=np.int64)
    padded_ids[0, :real] = ids[0]
    padded_mask = np.zeros((1, 1, bucket), dtype=np.float32)
    padded_mask[0, 0, :real] = 1.0

    # The JNI feeds style_dp transposed to NWC; everything else as-is.
    style_dp_nwc = np.ascontiguousarray(np.transpose(style.dp, (0, 2, 1)), dtype=np.float32)

    report = {
        "text": args.text,
        "realTextLen": int(real),
        "bucket": int(bucket),
        "idsMd5": md5(padded_ids, np.int64),
        "idsFirst8": [int(v) for v in padded_ids[0, :8]],
        "styleDpMd5": md5(style_dp_nwc, np.float32),
        "styleTtlMd5": md5(style.ttl, np.float32),
        "textMaskMd5": md5(padded_mask, np.float32),
    }

    dp = TfliteGraph(pick(args.dynamic_dir, "duration_predictor"), 4,
                     transposed=TRANSPOSED_INPUTS["duration_predictor"])
    duration = np.asarray(dp.run({"text_ids": padded_ids, "style_dp": style.dp,
                                  "text_mask": padded_mask})[0], np.float32)
    report["durationSeconds"] = round(float(duration.ravel()[0]) / args.speed, 4)

    encoder = TfliteGraph(pick(args.dynamic_dir, "text_encoder"), 4,
                          transposed=TRANSPOSED_INPUTS["text_encoder"])
    embed_dim = style.ttl.shape[-1]
    text_emb = np.asarray(
        encoder.run({"text_ids": padded_ids, "style_ttl": style.ttl,
                     "text_mask": padded_mask},
                    output_shapes=[(1, embed_dim, bucket)])[0], np.float32)
    report["embedDim"] = int(embed_dim)
    report["textEmbMd5"] = md5(text_emb, np.float32)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
