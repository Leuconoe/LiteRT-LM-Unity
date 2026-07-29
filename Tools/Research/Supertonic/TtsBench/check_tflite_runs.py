"""Check that a converted tflite allocates and invokes, with and without XNNPACK.

Used while hunting conversion settings: a graph can convert cleanly and then fail
at allocate (onnx2tf's FlexTranspose compression once emitted a rank-4 perm for a
rank-5 transpose), and the delegate can fail where the built-in kernels succeed.
Both matter, because the device picks the kernel mode per graph.

Feeds random data of the right shape, so it proves the graph runs — not that it is
numerically correct. Use compare_supertonic_runtimes.py for that.

Usage:
  python check_tflite_runs.py model.tflite [model2.tflite ...]
"""
import os
import sys

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType


def check(path, label, resolver):
    kwargs = {} if resolver is None else {"experimental_op_resolver_type": resolver}
    try:
        interpreter = Interpreter(model_path=path, num_threads=4, **kwargs)
        interpreter.allocate_tensors()
        for detail in interpreter.get_input_details():
            dtype = np.dtype(detail["dtype"])
            shape = [int(v) for v in detail["shape"]]
            value = (np.random.randn(*shape).astype(dtype)
                     if dtype == np.float32 else np.zeros(shape, dtype))
            interpreter.set_tensor(detail["index"], value)
        interpreter.invoke()
        outputs = [(d["name"].split(":")[0], [int(v) for v in d["shape"]])
                   for d in interpreter.get_output_details()]
        return f"[{label}] OK  outputs={outputs}"
    except Exception as exception:
        return f"[{label}] FAIL {str(exception)[:110]}"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    np.random.seed(0)
    failures = 0
    for path in sys.argv[1:]:
        name = os.path.basename(path)
        if not os.path.isfile(path):
            print(f"{name:<42} MISSING")
            failures += 1
            continue
        for label, resolver in (("xnn", None),
                                ("builtin", OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)):
            line = check(path, label, resolver)
            print(f"{name:<42} {line}")
            if "FAIL" in line and label == "builtin":
                failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
