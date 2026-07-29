"""Quantize the converted Supertonic tflite graphs with ai-edge-quantizer.

Applies this project's established recipes rather than inventing new ones
(see docs and the ASR work):

  i8  = recipe.dynamic_wi8_afp32()    — the speed pick, byte-identical output on
                                        the ASR models
  i4  = recipe.dynamic_wi4b64_afp32() — blockwise-64, the size pick.
                                        NEVER channelwise wi4c: validated quality
                                        collapse on this project's models.

Per-graph control matters here: `vector_estimator` runs once per flow-matching
step and dominates both size and time, while `duration_predictor` is 1.5 MB and
not worth degrading. Pass --recipe per graph to mix tiers.

Usage:
  python quantize_supertonic_tflite.py --tflite-dir <dir> --out-dir <dir> --recipe i8
  python quantize_supertonic_tflite.py --tflite-dir <dir> --out-dir <dir> \
      --graph-recipe vector_estimator=i4 --graph-recipe vocoder=i8 --recipe i8
"""
import argparse
import json
import os
import sys

GRAPHS = ["duration_predictor", "text_encoder", "vector_estimator", "vocoder"]


RECIPES = ("i8", "i4", "w8", "w4")


def build_recipe(kind):
    from ai_edge_quantizer import recipe

    if kind == "i8":
        return recipe.dynamic_wi8_afp32()
    if kind == "i4":
        return recipe.dynamic_wi4b64_afp32()
    # Weight-only: activations stay fp32 end to end. Dynamic quantization also
    # quantizes activations per invocation, which is what wrecked the vocoder
    # (mel corr 0.694) — it emits the waveform, so activation error is audible.
    # Weight-only gives the same 4x weight saving without touching activations.
    if kind == "w8":
        return recipe.weight_only_wi8_afp32()
    if kind == "w4":
        return recipe.weight_only_wi4_afp32()
    raise ValueError(f"unknown recipe '{kind}' (use one of {', '.join(RECIPES)})")


def find_source(tflite_dir, stem):
    directory = os.path.join(tflite_dir, stem)
    if not os.path.isdir(directory):
        directory = tflite_dir
    names = [f for f in sorted(os.listdir(directory)) if f.endswith(".tflite")]
    preferred = [f for f in names if f.startswith(stem) and "float32" in f] \
        or [f for f in names if "float32" in f] \
        or [f for f in names if f.startswith(stem)] \
        or names
    if not preferred:
        raise FileNotFoundError(f"no .tflite for {stem} under {directory}")
    return os.path.join(directory, preferred[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tflite-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--recipe", default="i8", choices=list(RECIPES),
                        help="default recipe for every graph")
    parser.add_argument("--graph-recipe", action="append", default=[],
                        metavar="GRAPH=RECIPE",
                        help="override one graph, e.g. vector_estimator=i4")
    parser.add_argument("--only", action="append", choices=GRAPHS)
    args = parser.parse_args()

    from ai_edge_quantizer import quantizer

    overrides = {}
    for item in args.graph_recipe:
        graph, _, kind = item.partition("=")
        if graph not in GRAPHS or kind not in RECIPES:
            print(f"bad --graph-recipe '{item}'", file=sys.stderr)
            return 2
        overrides[graph] = kind

    os.makedirs(args.out_dir, exist_ok=True)
    report = {}

    for stem in (args.only or GRAPHS):
        try:
            source = find_source(args.tflite_dir, stem)
        except FileNotFoundError as exception:
            print(f"SKIP    {stem}: {exception}")
            report[stem] = {"status": "missing"}
            continue

        kind = overrides.get(stem, args.recipe)
        destination = os.path.join(args.out_dir, f"{stem}_{kind}.tflite")

        qt = quantizer.Quantizer(source)
        qt.load_quantization_recipe(build_recipe(kind))
        result = qt.quantize()
        result.export_model(destination)

        before = os.path.getsize(source) / 1048576
        after = os.path.getsize(destination) / 1048576
        report[stem] = {
            "recipe": kind,
            "source": source,
            "output": destination,
            "mb_before": round(before, 1),
            "mb_after": round(after, 1),
            "ratio": round(before / after, 2) if after else None,
        }
        print(f"OK      {stem:<20} {kind}  {before:6.1f} MB -> {after:6.1f} MB "
              f"({report[stem]['ratio']}x)")

    path = os.path.join(args.out_dir, "quantization-report.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"\nreport: {path}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
