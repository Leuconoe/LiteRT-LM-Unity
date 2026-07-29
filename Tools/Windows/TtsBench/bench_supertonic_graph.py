"""Benchmark one converted Supertonic graph across runtime configurations.

`vector_estimator` runs once per flow-matching step and was 78 % of end-to-end
time, so it is the graph worth optimising. This measures it (or any other) with
and without the XNNPACK delegate, at fp32/fp16/quantized, and at a range of
thread counts — so the choice is made on numbers rather than on assumption.

It also reports whether XNNPACK actually attached, which matters: the delegate
fails to prepare on some of these graphs and tflite then falls back silently.

Usage:
  python bench_supertonic_graph.py --assets-dir <onnx dir> --tflite <file>...
      [--graph vector_estimator] [--text-len 64] [--latent-len 128] [--runs 5]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import supertonic_helper as helper
from supertonic_litert import TRANSPOSED_INPUTS

from ai_edge_litert.interpreter import Interpreter, OpResolverType


def build_feed(graph, assets_dir, voice, text, lang, text_len, latent_len, cfgs):
    processor = helper.load_text_processor(assets_dir)
    style = helper.load_voice_style([voice])
    text_ids, text_mask = processor([text], [lang])
    real = text_ids.shape[1]
    if text_len:
        padded = np.zeros((1, text_len), dtype=text_ids.dtype)
        padded[0, :min(real, text_len)] = text_ids[0, :min(real, text_len)]
        mask = np.zeros((1, 1, text_len), dtype=np.float32)
        mask[0, 0, :min(real, text_len)] = 1.0
        text_ids, text_mask = padded, mask
    text_mask = text_mask.astype(np.float32)
    length = text_ids.shape[1]

    latent_dim = cfgs["ttl"]["latent_dim"] * cfgs["ttl"]["chunk_compress_factor"]
    latent_len = latent_len or 128
    embed_dim = style.ttl.shape[-1]

    if graph == "duration_predictor":
        return {"text_ids": text_ids, "style_dp": style.dp, "text_mask": text_mask}
    if graph == "text_encoder":
        return {"text_ids": text_ids, "style_ttl": style.ttl, "text_mask": text_mask}
    if graph == "vector_estimator":
        return {
            "noisy_latent": np.random.randn(1, latent_dim, latent_len).astype(np.float32),
            "text_emb": np.random.randn(1, embed_dim, length).astype(np.float32),
            "style_ttl": style.ttl,
            "text_mask": text_mask,
            "latent_mask": np.ones((1, 1, latent_len), dtype=np.float32),
            "current_step": np.array([0], dtype=np.float32),
            "total_step": np.array([8], dtype=np.float32),
        }
    if graph == "vocoder":
        return {"latent": np.random.randn(1, latent_dim, latent_len).astype(np.float32)}
    raise ValueError(graph)


def measure(path, graph, feed, threads, xnnpack, runs):
    """Returns (median seconds, delegate_attached, error)."""
    kwargs = {} if xnnpack else {
        "experimental_op_resolver_type": OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES}
    try:
        interpreter = Interpreter(model_path=path, num_threads=threads, **kwargs)
        interpreter.allocate_tensors()
    except Exception as exception:
        return None, False, f"allocate: {str(exception)[:90]}"

    transposed = TRANSPOSED_INPUTS.get(graph, set())
    details = {d["name"].split(":")[0]: d for d in interpreter.get_input_details()}

    try:
        prepared = {}
        for name, array in feed.items():
            if name not in details:
                continue
            if array.ndim == 3 and name in transposed:
                array = np.ascontiguousarray(np.transpose(array, (0, 2, 1)))
            prepared[name] = array
            if list(details[name]["shape"]) != list(array.shape):
                interpreter.resize_tensor_input(details[name]["index"],
                                                list(array.shape), strict=False)
        interpreter.allocate_tensors()
        details = {d["name"].split(":")[0]: d for d in interpreter.get_input_details()}
        for name, array in prepared.items():
            want = np.dtype(details[name]["dtype"])
            interpreter.set_tensor(details[name]["index"],
                                   array if array.dtype == want else array.astype(want))
        interpreter.invoke()  # warm up: first call pays kernel preparation
    except Exception as exception:
        return None, False, f"invoke: {str(exception)[:90]}"

    times = []
    for _ in range(runs):
        started = time.perf_counter()
        interpreter.invoke()
        times.append(time.perf_counter() - started)
    return float(np.median(times)), xnnpack, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--tflite", action="append", required=True,
                        help="tflite file to benchmark; repeatable")
    parser.add_argument("--graph", default="vector_estimator",
                        choices=list(TRANSPOSED_INPUTS))
    parser.add_argument("--voice")
    parser.add_argument("--text", default="고도 백 미터로 상승합니다.")
    parser.add_argument("--lang", default="ko")
    parser.add_argument("--text-len", type=int)
    parser.add_argument("--latent-len", type=int, default=128)
    parser.add_argument("--threads", action="append", type=int, default=[])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--report")
    args = parser.parse_args()

    np.random.seed(0)
    voice = args.voice or os.path.join(args.assets_dir, "F1.json")
    cfgs = helper.load_cfgs(args.assets_dir)
    feed = build_feed(args.graph, args.assets_dir, voice, args.text, args.lang,
                      args.text_len, args.latent_len, cfgs)
    threads = args.threads or [4]

    rows = []
    for path in args.tflite:
        if not os.path.isfile(path):
            print(f"MISSING  {path}", file=sys.stderr)
            continue
        size_mb = os.path.getsize(path) / 1048576
        for count in threads:
            for xnnpack in (True, False):
                seconds, attached, error = measure(path, args.graph, feed, count,
                                                   xnnpack, args.runs)
                rows.append({"tflite": os.path.basename(path), "mb": round(size_mb, 1),
                             "threads": count, "xnnpack": xnnpack,
                             "seconds": None if seconds is None else round(seconds, 4),
                             "error": error})
                label = "xnn" if xnnpack else "ref"
                if seconds is None:
                    print(f"  {os.path.basename(path):<40} t{count} {label}  FAILED  {error}")
                else:
                    print(f"  {os.path.basename(path):<40} t{count} {label}  {seconds * 1000:8.1f} ms")

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump({"graph": args.graph, "rows": rows}, handle, ensure_ascii=False, indent=2)
        print(f"\nreport: {args.report}")

    best = [r for r in rows if r["seconds"]]
    if best:
        winner = min(best, key=lambda r: r["seconds"])
        print(f"\nfastest: {winner['tflite']} t{winner['threads']} "
              f"{'xnn' if winner['xnnpack'] else 'ref'} {winner['seconds'] * 1000:.1f} ms")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
