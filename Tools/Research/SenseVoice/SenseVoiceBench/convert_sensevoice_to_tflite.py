"""Static per-bucket SenseVoice ONNX (sv_t{N}.onnx) -> tflite fp32 + quantized variants.

Runs in .venv-convert (TensorFlow + onnx2tf). Input = export_sensevoice_onnx.py output.
Writes <out>/{fp32,w8,i8}/sv_t{N}.tflite + side files + conversion-report.json.

  python convert_sensevoice_to_tflite.py --onnx-dir <work>/onnx --out-dir <work> --buckets 50,84,167
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

SIDE = ("am.mvn", "config.yaml")
RECIPES = {
    # Weight-only keeps activations fp32 (dynamic activation quant wrecked Supertonic's vocoder).
    "w8": "weight_only_wi8_afp32",
    "i8": "dynamic_wi8_afp32",
}


def simplify(src: Path, dst: Path) -> None:
    import onnx
    from onnxsim import simplify as sim

    model, ok = sim(onnx.load(str(src)))
    if not ok:
        raise RuntimeError(f"onnxsim could not validate {src.name}")
    onnx.save(model, str(dst))


def onnx2tf(src: Path, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    # No -cotof: on Windows its op-error-report child process wrote the report and then never
    # exited, stalling onnx2tf for 35+ min (t84). Its one result — ctc_logits max_abs 2.3e-5,
    # cosine 1.0 — is re-checked end to end on real audio by sv_eval --ref-onnx instead.
    cmd = [sys.executable, "-m", "onnx2tf", "-i", str(src), "-o", str(out),
           "-osd",    # signature defs -> named inputs (bound by name in sv_eval)
           "-n"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (out / "onnx2tf.log").write_text((r.stdout or "") + "\n" + (r.stderr or ""), encoding="utf-8")
    if r.returncode != 0:
        tail = "\n".join((r.stderr or r.stdout or "").splitlines()[-30:])
        raise RuntimeError(f"onnx2tf failed for {src.name}:\n{tail}")
    return next(out.glob("*_float32.tflite"))


def quantize(src: Path, dst: Path, kind: str) -> None:
    from ai_edge_quantizer import quantizer, recipe

    qt = quantizer.Quantizer(str(src))
    qt.load_quantization_recipe(getattr(recipe, RECIPES[kind])())
    qt.quantize().export_model(str(dst), overwrite=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--buckets", default="50,84,167,500")
    ap.add_argument("--quant", default="w8,i8")
    a = ap.parse_args()

    onnx_dir, out = Path(a.onnx_dir), Path(a.out_dir)
    kinds = [k for k in a.quant.split(",") if k]
    report = {}
    for n in [int(b) for b in a.buckets.split(",")]:
        t0 = time.perf_counter()
        src = onnx_dir / f"sv_t{n}.onnx"
        sim = out / "tmp" / f"sv_t{n}.sim.onnx"
        sim.parent.mkdir(parents=True, exist_ok=True)
        simplify(src, sim)
        f32 = onnx2tf(sim, out / "tmp" / f"t{n}")
        (out / "fp32").mkdir(parents=True, exist_ok=True)
        shutil.copy2(f32, out / "fp32" / f"sv_t{n}.tflite")
        entry = {"fp32_mb": round(f32.stat().st_size / 2**20, 1)}
        for k in kinds:
            (out / k).mkdir(parents=True, exist_ok=True)
            dst = out / k / f"sv_t{n}.tflite"
            quantize(f32, dst, k)
            entry[f"{k}_mb"] = round(dst.stat().st_size / 2**20, 1)
        entry["seconds"] = round(time.perf_counter() - t0, 1)
        report[f"t{n}"] = entry
        print(f"OK t{n}: {entry}", flush=True)

    for v in ["fp32"] + kinds:
        for f in SIDE:
            shutil.copy2(onnx_dir / f, out / v / f)
        for f in onnx_dir.glob("*.bpe.model"):
            shutil.copy2(f, out / v / f.name)
    (out / "conversion-report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
