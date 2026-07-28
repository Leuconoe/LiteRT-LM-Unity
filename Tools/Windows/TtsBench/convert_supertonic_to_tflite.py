"""Convert the four Supertonic ONNX graphs to tflite for the LiteRT runtime.

Goal of the exercise: run Supertonic TTS through LiteRT (the runtime this project
already ships on Android and Windows) instead of onnxruntime, so TTS lands on the
same stack as the LLM and ASR paths.

Route: fp32 ONNX -> onnx2tf -> tflite (fp32), then quantize with
ai-edge-quantizer, which is this project's established recipe
(`dynamic_wi8_afp32`, `dynamic_wi4b64_afp32` — never channelwise wi4c).
Converting the already-int8 QDQ ONNX directly is the other option and usually
converts worse, so it is not the default here.

The graphs and their signatures (from the MIT reference implementation,
supertone-inc/supertonic py/helper.py):

  duration_predictor(text_ids, style_dp, text_mask)                 -> duration
  text_encoder(text_ids, style_ttl, text_mask)                      -> text_emb
  vector_estimator(noisy_latent, text_emb, style_ttl, text_mask,
                   latent_mask, current_step, total_step)           -> latent
  vocoder(latent)                                                   -> wav

vector_estimator is called `total_step` times per utterance, so it dominates
runtime and is the graph whose conversion matters most.

Usage:
  python convert_supertonic_to_tflite.py --onnx-dir <fp32 dir> --out-dir <dir>
                                         [--only vector_estimator] [--no-quantize]
"""
import argparse
import json
import os
import subprocess
import sys
import time

GRAPHS = ["duration_predictor", "text_encoder", "vector_estimator", "vocoder"]

# Why fixed shapes rather than dynamic axes
# ----------------------------------------
# Converting with dynamic axes left every variable dimension at 1 and required
# resize_tensor_input per utterance. Two things broke:
#   * XNNPACK cannot re-prepare after a resize ("failed to reshape runtimeNode"),
#     so the delegate has to be dropped — and it is the reason tflite is fast.
#   * onnx2tf reorders 3-D tensors NCW -> NWC, and with all dims equal to 1 there
#     is nothing to disambiguate. text_mask came out transposed, and feeding the
#     transposed shape still produced wrong numbers (duration 1.73 vs 2.52).
# Fixed shapes remove both problems, at the cost of padding to a bucket. Text is
# padded with the mask already in the pipeline, so bucketing is nearly free.
TEXT_BUCKETS = [64, 128, 256]
LATENT_BUCKETS = [64, 128, 256]


def freeze_onnx_shapes(onnx_path, out_path, symbolic_sizes):
    """Rewrite the ONNX with its symbolic input dims pinned to concrete sizes.

    Uses onnxsim, which folds the shape arithmetic these graphs perform on their
    own inputs. Editing `dim_param` by hand and re-running `shape_inference`
    produced a graph that converted but could not allocate
    ("[1,1,1,2] and [1,2,65,130] are not broadcastable") — the Reshape/Concat
    chains kept operating on the old symbolic sizes.

    Pinning in the ONNX rather than through onnx2tf's `-ois`, which had no effect
    on these graphs at all. Concrete dimensions are also what make the converted
    layout recoverable, and what lets XNNPACK stay attached: without a resize at
    runtime, the delegate never has to re-prepare.

    Returns the sizes actually applied, for the report.
    """
    import onnx
    from onnxsim import simplify

    model = onnx.load(onnx_path)
    initializers = {i.name for i in model.graph.initializer}
    overwrite = {}
    unresolved = []

    for value in model.graph.input:
        if value.name in initializers:
            continue
        dims = []
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_param:
                if dim.dim_param not in symbolic_sizes:
                    unresolved.append(f"{value.name}.{dim.dim_param}")
                    dims.append(1)
                else:
                    dims.append(symbolic_sizes[dim.dim_param])
            else:
                dims.append(dim.dim_value)
        overwrite[value.name] = dims

    if unresolved:
        raise ValueError(
            f"{os.path.basename(onnx_path)}: no size given for {sorted(set(unresolved))}")

    simplified, ok = simplify(model, overwrite_input_shapes=overwrite)
    if not ok:
        raise RuntimeError(f"onnxsim could not validate {os.path.basename(onnx_path)}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    onnx.save(simplified, out_path, save_as_external_data=False)
    return overwrite


def onnx_input_shapes(onnx_path):
    """{input name: [dims]} for the pinned graph.

    Written into the conversion report so the runtime can work out how onnx2tf
    permuted each input, instead of anyone hard-coding a transpose.
    """
    import onnx

    model = onnx.load(onnx_path, load_external_data=False)
    initializers = {i.name for i in model.graph.initializer}
    shapes = {}
    for value in model.graph.input:
        if value.name in initializers:
            continue
        shapes[value.name] = [
            d.dim_value if d.dim_value else d.dim_param
            for d in value.type.tensor_type.shape.dim
        ]
    return shapes


def describe_onnx(path):
    """Input/output names, shapes and dtypes — the contract the tflite must keep."""
    import onnx
    from onnx import shape_inference

    model = onnx.load(path, load_external_data=False)
    del shape_inference

    def spec(values):
        out = []
        for v in values:
            dims = []
            for d in v.type.tensor_type.shape.dim:
                dims.append(d.dim_param if d.dim_param else d.dim_value)
            out.append({"name": v.name, "shape": dims,
                        "dtype": int(v.type.tensor_type.elem_type)})
        return out

    initializers = {i.name for i in model.graph.initializer}
    inputs = [v for v in model.graph.input if v.name not in initializers]
    ops = {}
    for node in model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    return {
        "inputs": spec(inputs),
        "outputs": spec(model.graph.output),
        "op_types": dict(sorted(ops.items(), key=lambda kv: -kv[1])),
        "node_count": len(model.graph.node),
    }


def convert_one(onnx_path, out_dir, extra_args):
    """Run onnx2tf for a single graph. Returns (ok, seconds, message)."""
    os.makedirs(out_dir, exist_ok=True)
    command = [
        sys.executable, "-m", "onnx2tf",
        "-i", onnx_path,
        "-o", out_dir,
        "-osd",              # keep signature defs, so LiteRT sees named inputs
        "-cotof",            # check outputs against onnxruntime, all ops
        "-n",                # non-verbose
    ] + list(extra_args)

    started = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    seconds = time.perf_counter() - started
    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout or "").splitlines()[-25:])
        return False, seconds, tail
    return True, seconds, ""


def find_tflite(out_dir, stem):
    """onnx2tf writes <stem>_float32.tflite and friends into out_dir."""
    found = []
    for name in sorted(os.listdir(out_dir)):
        if name.endswith(".tflite") and name.startswith(stem):
            found.append(os.path.join(out_dir, name))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-dir", required=True, help="directory holding the four fp32 .onnx files")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--only", action="append", choices=GRAPHS,
                        help="convert a subset; repeatable")
    parser.add_argument("--describe-only", action="store_true",
                        help="print the ONNX signatures and exit, converting nothing")
    parser.add_argument("--keep-going", action="store_true",
                        help="continue after a graph fails instead of stopping")
    parser.add_argument("--text-len", type=int,
                        help="convert with a fixed text length instead of dynamic axes "
                             "(recommended; see the note at the top of this file)")
    parser.add_argument("--latent-len", type=int,
                        help="fixed latent length for vector_estimator/vocoder")
    args = parser.parse_args()

    graphs = args.only or GRAPHS
    report = {"onnx_dir": os.path.abspath(args.onnx_dir), "graphs": {}}

    for stem in graphs:
        onnx_path = os.path.join(args.onnx_dir, f"{stem}.onnx")
        if not os.path.isfile(onnx_path):
            print(f"MISSING  {onnx_path}", file=sys.stderr)
            report["graphs"][stem] = {"status": "missing"}
            continue

        entry = {"onnx_mb": round(os.path.getsize(onnx_path) / 1048576, 1)}
        try:
            entry["signature"] = describe_onnx(onnx_path)
        except Exception as exception:  # onnx not installed, or unreadable graph
            entry["signature_error"] = str(exception)

        if args.describe_only:
            report["graphs"][stem] = entry
            continue

        extra = []
        source_path = onnx_path
        if args.text_len:
            latent_len = args.latent_len or args.text_len
            symbolic = {"batch_size": 1, "text_length": args.text_len,
                        "latent_length": latent_len}
            frozen = os.path.join(args.out_dir, "_frozen", f"{stem}.onnx")
            try:
                pinned = freeze_onnx_shapes(onnx_path, frozen, symbolic)
            except Exception as exception:
                print(f"FAILED   {stem}: shape freeze — {exception}", file=sys.stderr)
                report["graphs"][stem] = {**entry, "status": "failed",
                                          "error_tail": f"shape freeze: {exception}"}
                if not args.keep_going:
                    break
                continue
            source_path = frozen
            entry["fixed_shape"] = pinned
            entry["onnx_input_shapes"] = onnx_input_shapes(frozen)
            # No -kat here. onnx2tf rewrites 3-D tensors NCW -> NWC, and forcing
            # the layout back produced a graph that failed to allocate
            # ("[1,1,1,2] and [1,4,64,128] are not broadcastable"). Letting the
            # converter do its own thing keeps the graph valid; because the
            # shapes are now concrete, the caller can recover the permutation by
            # matching tflite dims against the ONNX dims recorded above.

        out_dir = os.path.join(args.out_dir, stem)
        ok, seconds, message = convert_one(source_path, out_dir, extra)
        entry["convert_seconds"] = round(seconds, 1)
        entry["status"] = "ok" if ok else "failed"
        if ok:
            files = find_tflite(out_dir, stem) or find_tflite(out_dir, "")
            entry["tflite"] = [
                {"path": f, "mb": round(os.path.getsize(f) / 1048576, 1)} for f in files
            ]
            print(f"OK       {stem}  {seconds:6.1f}s  " +
                  ", ".join(f"{os.path.basename(f['path'])} {f['mb']}MB" for f in entry["tflite"]))
        else:
            entry["error_tail"] = message
            print(f"FAILED   {stem}  {seconds:6.1f}s")
            print(message, file=sys.stderr)
            if not args.keep_going:
                report["graphs"][stem] = entry
                break

        report["graphs"][stem] = entry

    out_report = os.path.join(args.out_dir, "conversion-report.json") if not args.describe_only else None
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if out_report:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(out_report, "w", encoding="utf-8") as handle:
            handle.write(text)
        print(f"\nreport: {out_report}")
    else:
        print(text)

    failed = [k for k, v in report["graphs"].items() if v.get("status") == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
