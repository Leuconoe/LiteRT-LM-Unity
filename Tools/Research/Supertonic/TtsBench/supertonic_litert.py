"""Run Supertonic TTS through LiteRT (tflite) instead of onnxruntime.

This is the target runtime for the project: LiteRT already carries the LLM and
ASR paths on Windows and Android, so putting TTS on it means one runtime instead
of three.

The pipeline is the MIT reference one from supertone-inc/supertonic
(`py/helper.py`, vendored next to this file as `supertonic_helper.py`); only the
four inference calls are swapped from ORT sessions to tflite interpreters:

  duration_predictor(text_ids, style_dp, text_mask)                 -> duration
  text_encoder(text_ids, style_ttl, text_mask)                      -> text_emb
  vector_estimator(noisy_latent, text_emb, style_ttl, text_mask,
                   latent_mask, current_step, total_step)           -> latent
  vocoder(latent)                                                   -> wav

Inputs are bound by name when the converted model kept its signature names, and
by shape/dtype otherwise — conversion tools rename freely, and the whisper driver
in this repo learned the same lesson.

Every graph takes dynamic-length inputs, so tensors are resized per utterance.

Emits one JSON line with per-stage timings so it can be compared directly with
the onnxruntime baseline from Run-SupertonicTts.ps1.

Usage:
  python supertonic_litert.py --tflite-dir <dir> --assets-dir <fp32 onnx dir>
                              --voice <voice_styles/F1.json> --text "..." --out out.wav
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import supertonic_helper as helper  # vendored MIT reference

from ai_edge_litert.interpreter import Interpreter


# Input layout of the converted graphs
# -----------------------------------
# onnx2tf rewrites some 3-D inputs from NCW to NWC and leaves others alone, and
# which is which cannot be inferred from the shape: `style_ttl` [1,50,256] is
# kept, while `style_dp` [1,8,16], `text_mask` [1,1,L] and the latents are
# transposed. Outputs come back in the ONNX layout.
#
# The map was derived by converting each graph at fixed shapes and reading the
# resulting input details — concrete dimensions make the permutation unambiguous
# — and then confirmed numerically against onnxruntime (duration_predictor
# max|diff| 2.4e-07, text_encoder 3.7e-06). It is data, not a guess, but it is
# specific to onnx2tf's behaviour on these graphs: re-derive it with
# `--probe-layout` if the converter or the model version changes.
TRANSPOSED_INPUTS = {
    "duration_predictor": {"style_dp", "text_mask"},
    "text_encoder": {"text_mask"},
    "vector_estimator": {"noisy_latent", "text_emb", "latent_mask", "text_mask"},
    "vocoder": {"latent"},
}


class TfliteGraph:
    """One converted graph, callable with the ONNX input names.

    XNNPACK note: these graphs are converted with dynamic axes collapsed to 1 and
    resized per utterance, and XNNPACK cannot re-prepare some nodes after a
    resize — it fails with "failed to reshape runtimeNode number N". The
    interpreter is therefore rebuilt without the default delegates as soon as that
    happens. The same limit applies on Android, so the shipping build should
    convert with fixed (bucketed) shapes rather than rely on resizing.
    """

    def __init__(self, path, num_threads=4, allow_xnnpack=True, transposed=None):
        self.path = path
        self.num_threads = num_threads
        self._xnnpack = allow_xnnpack
        # Names whose 3-D input must be fed as NWC. Empty set = feed as-is.
        self.transposed = set(transposed or ())
        self._build()

    def _build(self):
        if self._xnnpack:
            self.interpreter = Interpreter(model_path=self.path, num_threads=self.num_threads)
        else:
            from ai_edge_litert.interpreter import OpResolverType

            self.interpreter = Interpreter(
                model_path=self.path, num_threads=self.num_threads,
                experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        self.interpreter.allocate_tensors()
        self._inputs = self.interpreter.get_input_details()
        self._outputs = self.interpreter.get_output_details()

    def _fallback_without_xnnpack(self):
        """Rebuild without delegates; returns False when already there."""
        if not self._xnnpack:
            return False
        print(f"[litert] {os.path.basename(self.path)}: XNNPACK cannot resize this "
              f"graph; falling back to the reference kernels", file=sys.stderr)
        self._xnnpack = False
        self._build()
        return True

    @staticmethod
    def _permutation(source_shape, target_shape):
        """Axis order that turns source_shape into target_shape, or None.

        onnx2tf rewrites 3-D tensors from NCW to NWC, and it does so for some
        inputs and not others in the same graph. Rather than hard-code which,
        the permutation is recovered by matching the concrete dimensions — which
        works because the graphs are converted at fixed shapes, so nothing is 1
        except real singleton axes.
        """
        if len(source_shape) != len(target_shape):
            return None
        if list(source_shape) == list(target_shape):
            return tuple(range(len(source_shape)))

        used = [False] * len(source_shape)
        order = []
        for want in target_shape:
            for i, have in enumerate(source_shape):
                if not used[i] and have == want:
                    used[i] = True
                    order.append(i)
                    break
            else:
                return None
        return tuple(order)

    def _match(self, feed):
        """Map ONNX input names onto tflite tensor indices.

        First by name (onnx2tf -osd keeps them), then by rank+dtype, which is
        unambiguous for these graphs because no two inputs share both.
        """
        remaining = list(self._inputs)
        bound = {}

        for name, array in feed.items():
            hit = None
            for detail in remaining:
                tf_name = detail["name"].split(":")[0].split("/")[-1]
                if tf_name == name:
                    hit = detail
                    break
            if hit is not None:
                bound[name] = hit
                remaining.remove(hit)

        for name, array in feed.items():
            if name in bound:
                continue
            candidates = [
                d for d in remaining
                if len(d["shape"]) == array.ndim and np.dtype(d["dtype"]) == array.dtype
            ]
            if not candidates:
                raise ValueError(
                    f"{os.path.basename(self.path)}: cannot bind '{name}' "
                    f"(ndim={array.ndim}, dtype={array.dtype}); "
                    f"free inputs={[(d['name'], list(d['shape']), str(np.dtype(d['dtype']))) for d in remaining]}")
            bound[name] = candidates[0]
            remaining.remove(candidates[0])

        return bound

    def run(self, feed, output_shapes=None):
        """Run the graph with ONNX-shaped tensors.

        `feed` uses ONNX names and ONNX layouts; inputs are permuted into the
        converted graph's layout here, and outputs are permuted back, so callers
        never see the NCW/NWC rewrite. `output_shapes` gives the expected ONNX
        output shapes when the permutation cannot be inferred otherwise.
        """
        feed = {k: np.ascontiguousarray(v) for k, v in feed.items()}

        for attempt in range(2):
            bound = self._match(feed)

            prepared = {}
            resized = False
            for name, detail in bound.items():
                array = feed[name]
                # Dynamic-axis graphs report all-ones shapes, which carry no
                # layout information, so the permutation comes from the map
                # above; where the shapes are concrete they are used to confirm.
                if array.ndim == 3 and name in self.transposed:
                    array = np.ascontiguousarray(np.transpose(array, (0, 2, 1)))
                else:
                    order = self._permutation(array.shape, tuple(detail["shape"]))
                    if order and order != tuple(range(array.ndim)):
                        array = np.ascontiguousarray(np.transpose(array, order))
                if list(detail["shape"]) != list(array.shape):
                    self.interpreter.resize_tensor_input(
                        detail["index"], list(array.shape), strict=False)
                    resized = True
                prepared[name] = array

            if resized:
                self.interpreter.allocate_tensors()
                self._inputs = self.interpreter.get_input_details()
                self._outputs = self.interpreter.get_output_details()
                bound = self._match(prepared)

            for name, detail in bound.items():
                # tflite is strict about dtype; the converter keeps int64 ids.
                want = np.dtype(detail["dtype"])
                array = prepared[name]
                self.interpreter.set_tensor(
                    detail["index"], array if array.dtype == want else array.astype(want))

            try:
                self.interpreter.invoke()
            except RuntimeError as error:
                if attempt == 0 and "XNNPack" in str(error) and self._fallback_without_xnnpack():
                    continue
                raise

            outputs = [self.interpreter.get_tensor(d["index"]) for d in self._outputs]
            if output_shapes:
                fixed = []
                for i, array in enumerate(outputs):
                    want = output_shapes[i] if i < len(output_shapes) else None
                    order = self._permutation(array.shape, tuple(want)) if want else None
                    fixed.append(np.transpose(array, order) if order and
                                 order != tuple(range(array.ndim)) else array)
                outputs = fixed
            return outputs


class OnnxGraph:
    """Same interface as TfliteGraph, backed by onnxruntime.

    Present so the two runtimes can be driven by *identical* pipeline code with
    the same latent seed: any waveform difference is then conversion error, not a
    different sampler or a different mask.
    """

    def __init__(self, path, num_threads=4):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads
        self.path = path
        self.session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
        self._input_types = {i.name: i.type for i in self.session.get_inputs()}

    def run(self, feed):
        # ORT is strict about int64 vs int32 where tflite is not.
        typed = {}
        for name, array in feed.items():
            want = self._input_types.get(name, "")
            if "int64" in want:
                array = array.astype(np.int64)
            elif "int32" in want:
                array = array.astype(np.int32)
            elif "float" in want:
                array = array.astype(np.float32)
            typed[name] = array
        return self.session.run(None, typed)


# Graphs the bucketed (fixed-shape) conversion emits but that do not work:
# onnxsim's rewrite leaves a rank-5 transpose behind a rank-4 perm, so they fail
# to allocate ("transpose perm 4 != 5" on the reference kernels, an XNNPACK
# reshape failure with the delegate). Both take text_ids through an embedding
# lookup, which is the part onnxsim reshapes.
#
# They are also the cheap half — duration_predictor 10 ms and text_encoder 119 ms
# of a synthesis whose expensive graphs are vector_estimator and vocoder, and
# those two *do* convert and do keep XNNPACK. So they are served from the dynamic
# build. They must be listed explicitly: the broken files exist and would
# otherwise be preferred.
BUCKETED_UNSUPPORTED = {"duration_predictor", "text_encoder"}


def bucket_dirs(root):
    """{bucket size: directory} for a ladder laid out as <root>/st-b<N>.

    Buckets exist because the fixed-shape graphs are the fast ones, and a fixed
    shape means the text has to be padded to a size that was converted. A ladder
    keeps the padding waste bounded: the cost of a bucket is proportional to its
    size, so a 40-id utterance in a 256 bucket would throw away most of the win.
    """
    found = {}
    if not root or not os.path.isdir(root):
        return found
    for name in sorted(os.listdir(root)):
        if not name.startswith("st-b"):
            continue
        try:
            size = int(name[4:])
        except ValueError:
            continue
        if os.path.isdir(os.path.join(root, name)):
            found[size] = os.path.join(root, name)
    return found


def choose_bucket(buckets, text_len, latent_len=0):
    """Smallest bucket that fits both lengths, or None when none does."""
    for size in sorted(buckets):
        if text_len <= size and latent_len <= size:
            return size, buckets[size]
    return None, None


def pick(tflite_dir, stem, fallback_dir=None):
    """Prefer float32 exports; fall back to whatever single file is present.

    `fallback_dir` covers the mixed setup the fixed-shape work produced: the
    bucketed conversion is much faster where it is valid, and the graphs listed in
    BUCKETED_UNSUPPORTED come from the dynamic build instead.
    """
    roots = (tflite_dir, fallback_dir)
    if fallback_dir and stem in BUCKETED_UNSUPPORTED:
        roots = (fallback_dir,)

    for root in roots:
        if not root:
            continue
        directory = os.path.join(root, stem)
        if not os.path.isdir(directory):
            directory = root
        if not os.path.isdir(directory):
            continue
        names = [f for f in sorted(os.listdir(directory)) if f.endswith(".tflite")]
        candidates = [f for f in names if f.startswith(stem)] or names
        if not candidates:
            continue
        preferred = [f for f in candidates if "float32" in f] or candidates
        return os.path.join(directory, preferred[0])
    raise FileNotFoundError(f"no .tflite for {stem} under {tflite_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tflite-dir", required=True)
    parser.add_argument("--assets-dir", required=True,
                        help="directory with tts.json, unicode_indexer.json")
    parser.add_argument("--voice", required=True, help="voice style json (e.g. F1.json)")
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lang", default="ko")
    parser.add_argument("--steps", type=int, default=8, help="flow-matching steps")
    parser.add_argument("--speed", type=float, default=1.05)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234,
                        help="fixes the latent noise so runs are comparable")
    parser.add_argument("--runtime", default="litert", choices=["litert", "onnx"],
                        help="onnx runs the same pipeline on the source graphs, "
                             "which is how conversion fidelity is measured")
    parser.add_argument("--text-len", type=int,
                        help="bucket the graphs were converted at; pads the text to it")
    parser.add_argument("--latent-len", type=int,
                        help="latent bucket the graphs were converted at")
    parser.add_argument("--fallback-tflite-dir",
                        help="second directory to source graphs the primary one lacks "
                             "or cannot convert (duration_predictor, in practice)")
    parser.add_argument("--bucket-root",
                        help="directory holding a bucket ladder (st-b64, st-b128, …); "
                             "the smallest bucket that fits the utterance is chosen "
                             "automatically, which is what makes the fast fixed-shape "
                             "graphs usable for arbitrary text")
    args = parser.parse_args()

    np.random.seed(args.seed)

    cfgs = helper.load_cfgs(args.assets_dir)
    text_processor = helper.load_text_processor(args.assets_dir)
    style = helper.load_voice_style([args.voice])

    sample_rate = cfgs["ae"]["sample_rate"]
    base_chunk_size = cfgs["ae"]["base_chunk_size"]
    chunk_compress_factor = cfgs["ttl"]["chunk_compress_factor"]
    latent_dim_base = cfgs["ttl"]["latent_dim"]

    timings = {}
    latent_dim = latent_dim_base * chunk_compress_factor
    chunk = base_chunk_size * chunk_compress_factor
    on_litert = args.runtime != "onnx"
    buckets = bucket_dirs(args.bucket_root) if on_litert else {}

    def load(stem, primary=None):
        if not on_litert:
            return OnnxGraph(os.path.join(args.assets_dir, f"{stem}.onnx"), args.threads)
        path = pick(primary or args.tflite_dir, stem,
                    args.fallback_tflite_dir or (args.tflite_dir if primary else None))
        timings.setdefault("graph_files", {})[stem] = \
            os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)
        return TfliteGraph(path, args.threads, transposed=TRANSPOSED_INPUTS.get(stem))

    started = time.perf_counter()
    text_ids, text_mask = text_processor([args.text], [args.lang])
    real_text_len = text_ids.shape[1]

    # duration_predictor and text_encoder are the dynamic-shape graphs, so they
    # can run at the real length. Their duration output is what decides the latent
    # length, which is needed before a bucket can be chosen for the fixed-shape
    # graphs — hence this order rather than padding everything up front.
    dp = load("duration_predictor")
    duration = dp.run({"text_ids": text_ids, "style_dp": style.dp,
                       "text_mask": text_mask.astype(np.float32)})[0]
    duration = duration / args.speed
    timings["duration_s"] = round(time.perf_counter() - started, 3)

    wav_lengths = (duration * sample_rate).astype(np.int64)
    latent_len = int((duration.max() * sample_rate + chunk - 1) // chunk)
    latent_mask = helper.get_latent_mask(
        wav_lengths, base_chunk_size, chunk_compress_factor).astype(np.float32)
    timings["real_text_len"] = int(real_text_len)
    timings["real_latent_len"] = latent_len

    bucket_dir = None
    pad_to = args.text_len
    latent_pad_to = args.latent_len
    if buckets:
        size, bucket_dir = choose_bucket(buckets, real_text_len, latent_len)
        if size is None:
            raise ValueError(
                f"utterance needs text {real_text_len} / latent {latent_len} ids but the "
                f"largest converted bucket is {max(buckets)}; convert a bigger one")
        pad_to = latent_pad_to = size
        timings["bucket"] = size

    if pad_to:
        if real_text_len > pad_to:
            raise ValueError(
                f"text needs {real_text_len} ids but the graphs were converted for {pad_to}")
        padded_ids = np.zeros((1, pad_to), dtype=text_ids.dtype)
        padded_ids[0, :real_text_len] = text_ids[0]
        padded_mask = np.zeros((1, 1, pad_to), dtype=np.float32)
        padded_mask[0, 0, :real_text_len] = 1.0
        text_ids, text_mask = padded_ids, padded_mask
    text_mask = text_mask.astype(np.float32)

    t0 = time.perf_counter()
    text_len = text_ids.shape[1]
    text_emb = load("text_encoder").run(
        {"text_ids": text_ids, "style_ttl": style.ttl, "text_mask": text_mask},
        output_shapes=[(1, style.ttl.shape[-1], text_len)])[0]
    timings["text_encoder_s"] = round(time.perf_counter() - t0, 3)

    if latent_pad_to:
        if latent_len > latent_pad_to:
            raise ValueError(
                f"utterance needs {latent_len} latent frames but the graphs were "
                f"converted for {latent_pad_to}")
        padded_mask = np.zeros((1, 1, latent_pad_to), dtype=np.float32)
        padded_mask[0, 0, :latent_mask.shape[-1]] = latent_mask[0, 0, :]
        latent_mask = padded_mask
        latent_len = latent_pad_to
    timings["latent_len"] = latent_len

    graphs = {"vector_estimator": load("vector_estimator", bucket_dir),
              "vocoder": load("vocoder", bucket_dir)}
    timings["load_s"] = round(time.perf_counter() - started, 3)

    np.random.seed(args.seed)
    xt = (np.random.randn(1, latent_dim, latent_len).astype(np.float32) * latent_mask).astype(np.float32)

    total_step = np.array([args.steps], dtype=np.float32)
    t0 = time.perf_counter()
    for step in range(args.steps):
        xt = graphs["vector_estimator"].run({
            "noisy_latent": xt,
            "text_emb": text_emb,
            "style_ttl": style.ttl,
            "text_mask": text_mask,
            "latent_mask": latent_mask,
            "current_step": np.array([step], dtype=np.float32),
            "total_step": total_step,
        }, output_shapes=[(1, latent_dim, latent_len)])[0].astype(np.float32)
    timings["vector_estimator_s"] = round(time.perf_counter() - t0, 3)
    timings["vector_estimator_per_step_s"] = round(timings["vector_estimator_s"] / max(1, args.steps), 4)

    t0 = time.perf_counter()
    wav = graphs["vocoder"].run({"latent": xt})[0]
    timings["vocoder_s"] = round(time.perf_counter() - t0, 3)

    samples = np.asarray(wav, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        print("vocoder produced no samples", file=sys.stderr)
        return 3

    # A padded latent bucket synthesizes silence past the utterance; trim it so
    # the reported RTF and the WAV reflect the real speech, not the padding.
    real_samples = int(wav_lengths.max())
    if 0 < real_samples < samples.size:
        timings["trimmed_samples"] = int(samples.size - real_samples)
        samples = samples[:real_samples]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    sf.write(args.out, samples, sample_rate)

    audio_s = samples.size / float(sample_rate)
    synth_s = sum(v for k, v in timings.items()
                  if k.endswith("_s") and k not in ("load_s", "vector_estimator_per_step_s"))
    print(json.dumps({
        "runtime": args.runtime,
        "text": args.text,
        "lang": args.lang,
        "wav": os.path.abspath(args.out),
        "audio_s": round(audio_s, 2),
        "synth_s": round(synth_s, 3),
        "rtf": round(synth_s / audio_s, 4) if audio_s else None,
        "steps": args.steps,
        "sample_rate": sample_rate,
        "threads": args.threads,
        **timings,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
