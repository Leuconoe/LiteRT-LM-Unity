"""Compare Supertonic on LiteRT against onnxruntime, stage by stage.

The end goal is serving TTS from LiteRT, so the question that matters is not
"does the converted model run" but "does it compute the same thing". This drives
both runtimes through the identical pipeline with the identical latent noise and
reports the divergence after every stage, plus the final waveform correlation.

Stage-by-stage matters because the flow-matching loop feeds its own output back
in: a small error in `vector_estimator` compounds over the steps, and only a
per-stage view shows where a discrepancy starts.

Usage:
  python compare_supertonic_runtimes.py --assets-dir <fp32 onnx dir> \
      --tflite-dir <converted dir> --voice F1.json --text "..." \
      [--text-len 64 --latent-len 128] [--steps 8]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import supertonic_helper as helper
from supertonic_litert import OnnxGraph, TfliteGraph, pick, TRANSPOSED_INPUTS

STEMS = ("duration_predictor", "text_encoder", "vector_estimator", "vocoder")


def stats(reference, candidate):
    reference = np.asarray(reference, np.float32).ravel()
    candidate = np.asarray(candidate, np.float32).ravel()
    if reference.shape != candidate.shape:
        return {"shape_mismatch": [list(reference.shape), list(candidate.shape)]}
    diff = np.abs(reference - candidate)
    scale = float(np.abs(reference).max()) or 1.0
    correlation = None
    if reference.size > 1 and reference.std() > 0 and candidate.std() > 0:
        correlation = float(np.corrcoef(reference, candidate)[0, 1])
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel": float(diff.max() / scale),
        "corr": correlation,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--tflite-dir", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--text", default="고도 백 미터로 상승합니다.")
    parser.add_argument("--lang", default="ko")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--speed", type=float, default=1.05)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--text-len", type=int)
    parser.add_argument("--latent-len", type=int)
    parser.add_argument("--only", action="append", choices=STEMS,
                        help="compare a subset; useful while some graphs are still converting")
    parser.add_argument("--report")
    args = parser.parse_args()

    stems = args.only or list(STEMS)
    cfgs = helper.load_cfgs(args.assets_dir)
    processor = helper.load_text_processor(args.assets_dir)
    style = helper.load_voice_style([args.voice])

    sample_rate = cfgs["ae"]["sample_rate"]
    base_chunk = cfgs["ae"]["base_chunk_size"]
    compress = cfgs["ttl"]["chunk_compress_factor"]
    latent_dim = cfgs["ttl"]["latent_dim"] * compress

    text_ids, text_mask = processor([args.text], [args.lang])
    real_len = text_ids.shape[1]
    if args.text_len:
        padded = np.zeros((1, args.text_len), dtype=text_ids.dtype)
        padded[0, :real_len] = text_ids[0]
        mask = np.zeros((1, 1, args.text_len), dtype=np.float32)
        mask[0, 0, :real_len] = 1.0
        text_ids, text_mask = padded, mask
    text_mask = text_mask.astype(np.float32)
    text_len = text_ids.shape[1]

    def load(stem):
        return (OnnxGraph(os.path.join(args.assets_dir, f"{stem}.onnx"), args.threads),
                TfliteGraph(pick(args.tflite_dir, stem), args.threads,
                            transposed=TRANSPOSED_INPUTS.get(stem)))

    report = {"text": args.text, "steps": args.steps, "seed": args.seed,
              "text_len": text_len, "stages": {}}

    # duration_predictor
    dp_feed = {"text_ids": text_ids, "style_dp": style.dp, "text_mask": text_mask}
    onnx_dp, tfl_dp = load("duration_predictor")
    duration_onnx = np.asarray(onnx_dp.run(dp_feed)[0], np.float32)
    if "duration_predictor" in stems:
        duration_tfl = np.asarray(tfl_dp.run(dp_feed)[0], np.float32)
        report["stages"]["duration_predictor"] = stats(duration_onnx, duration_tfl)
        report["stages"]["duration_predictor"]["onnx_value"] = duration_onnx.ravel().tolist()[:4]
        report["stages"]["duration_predictor"]["litert_value"] = duration_tfl.ravel().tolist()[:4]

    # Both runtimes continue from the ONNX duration so later stages are compared
    # on equal input rather than on drift inherited from this one.
    duration = duration_onnx / args.speed

    # text_encoder
    te_feed = {"text_ids": text_ids, "style_ttl": style.ttl, "text_mask": text_mask}
    onnx_te, tfl_te = load("text_encoder")
    emb_onnx = np.asarray(onnx_te.run(te_feed)[0], np.float32)
    if "text_encoder" in stems:
        emb_tfl = np.asarray(
            tfl_te.run(te_feed, output_shapes=[emb_onnx.shape])[0], np.float32)
        report["stages"]["text_encoder"] = stats(emb_onnx, emb_tfl)

    # Latent noise, identical for both runtimes.
    np.random.seed(args.seed)
    wav_lengths = (duration * sample_rate).astype(np.int64)
    chunk = base_chunk * compress
    latent_len = int((duration.max() * sample_rate + chunk - 1) // chunk)
    latent_mask = helper.get_latent_mask(wav_lengths, base_chunk, compress).astype(np.float32)
    if args.latent_len:
        padded = np.zeros((1, 1, args.latent_len), dtype=np.float32)
        padded[0, 0, :latent_mask.shape[-1]] = latent_mask[0, 0, :]
        latent_mask = padded
        latent_len = args.latent_len
    report["latent_len"] = latent_len
    x0 = (np.random.randn(1, latent_dim, latent_len).astype(np.float32) * latent_mask).astype(np.float32)

    latent_onnx = latent_tfl = None
    if "vector_estimator" in stems:
        onnx_ve, tfl_ve = load("vector_estimator")
        total = np.array([args.steps], dtype=np.float32)
        per_step = []
        xo = xt = x0
        for step in range(args.steps):
            common = {"text_emb": emb_onnx, "style_ttl": style.ttl,
                      "text_mask": text_mask, "latent_mask": latent_mask,
                      "current_step": np.array([step], dtype=np.float32),
                      "total_step": total}
            xo = np.asarray(onnx_ve.run({"noisy_latent": xo, **common})[0], np.float32)
            xt = np.asarray(tfl_ve.run({"noisy_latent": xt, **common},
                                       output_shapes=[xo.shape])[0], np.float32)
            per_step.append(stats(xo, xt))
        report["stages"]["vector_estimator"] = {
            "final": per_step[-1],
            "per_step_max_abs": [round(s.get("max_abs", float("nan")), 6) for s in per_step],
        }
        latent_onnx, latent_tfl = xo, xt

    if "vocoder" in stems:
        onnx_vo, tfl_vo = load("vocoder")
        source_onnx = latent_onnx if latent_onnx is not None else x0
        source_tfl = latent_tfl if latent_tfl is not None else x0
        wav_onnx = np.asarray(onnx_vo.run({"latent": source_onnx})[0], np.float32).ravel()
        wav_tfl = np.asarray(tfl_vo.run({"latent": source_tfl})[0], np.float32).ravel()
        report["stages"]["vocoder"] = stats(wav_onnx, wav_tfl)
        report["stages"]["vocoder"]["samples"] = int(wav_onnx.size)
        report["sample_rate"] = sample_rate

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(text)

    for stem in STEMS:
        entry = report["stages"].get(stem)
        if not entry:
            continue
        summary = entry.get("final", entry)
        if "shape_mismatch" in summary:
            print(f"  {stem:<20} SHAPE MISMATCH {summary['shape_mismatch']}")
        else:
            print(f"  {stem:<20} max|d| {summary['max_abs']:.3e}  "
                  f"rel {summary['max_rel']:.3e}  corr {summary['corr']}")
    if args.report:
        print(f"\nreport: {args.report}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
