"""Per-bucket invoke latency of SenseVoice tflite graphs + proof that XNNPACK delegated.

Each graph runs in a child process so the runtime's stderr ("Replacing N out of M node(s)
with delegate (TfLiteXNNPackDelegate)") can be captured — it is printed by C++, not Python.

  python bench_tflite.py <variant dir with sv_t*.tflite> [--threads 4] [--runs 10]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

CHILD = r"""
import sys, time, numpy as np
from ai_edge_litert.interpreter import Interpreter
it = Interpreter(model_path=sys.argv[1], num_threads=int(sys.argv[2]))
it.allocate_tensors()
ins = it.get_input_details()
feat = next(d for d in ins if len(d["shape"]) == 3)
t = int(feat["shape"][1] if feat["shape"][2] == 560 else feat["shape"][2])  # full bucket
for d in ins:
    nm = d["name"].lower()
    v = (np.random.default_rng(0).standard_normal(d["shape"]) if d is feat
         else np.full(d["shape"], t if "length" in nm else 12 if "lang" in nm else 15))
    it.set_tensor(d["index"], v.astype(d["dtype"]))
it.invoke()  # warm-up (XNNPACK packs weights here)
ts = []
for _ in range(int(sys.argv[3])):
    t = time.perf_counter(); it.invoke(); ts.append((time.perf_counter() - t) * 1000)
ts.sort()
print("RESULT", ts[len(ts) // 2], sum(ts) / len(ts))
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("variant_dir")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=10)
    a = ap.parse_args()
    rows = []
    for f in sorted(Path(a.variant_dir).glob("sv_t*.tflite"), key=lambda p: int(p.stem[4:])):
        r = subprocess.run([sys.executable, "-c", CHILD, str(f), str(a.threads), str(a.runs)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        m = re.search(r"RESULT ([\d.]+) ([\d.]+)", r.stdout)
        rep = re.findall(r"Replacing (\d+) out of (\d+) node\(s\) with delegate \((\w+)\)", r.stderr)
        rows.append({"graph": f.name, "mb": round(f.stat().st_size / 2**20, 1),
                     "p50_ms": round(float(m.group(1)), 1) if m else None,
                     "mean_ms": round(float(m.group(2)), 1) if m else None,
                     "delegated": [f"{a_}/{b}" for a_, b, _ in rep] or r.stderr.strip().splitlines()[-1:]})
        print(json.dumps(rows[-1]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
