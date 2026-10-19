"""funasr SenseVoice -> ONNX: one STATIC graph per bucket (sv_t{N}.onnx) + a dynamic model.onnx.

Runs in .venv. Uses funasr's own export wrapper (export_meta.export_rebuild_model), so the
graph is exactly funasr's export_forward: 4 query embeddings (language, event, emotion,
textnorm) prepended, SANM encoder masked from speech_lengths, ctc_lo logits out.
Inputs: speech [1,N,560] f32, speech_lengths [1] i32, language [1] i32, textnorm [1] i32.
Outputs: ctc_logits [1,N+4,25055], encoder_out_lens [1].

  python export_sensevoice_onnx.py --model FunAudioLLM/SenseVoiceSmall --out-dir <work>/onnx
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

NAMES_IN = ["speech", "speech_lengths", "language", "textnorm"]
NAMES_OUT = ["ctc_logits", "encoder_out_lens"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF id or local funasr dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--buckets", default="50,84,167,500")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()

    from funasr import AutoModel
    from funasr.models.sense_voice.export_meta import export_rebuild_model

    kw = {} if Path(a.model).exists() else {"hub": "hf"}
    am = AutoModel(model=a.model, device="cpu", disable_update=True, **kw)
    src = Path(am.kwargs["model_path"])
    m = export_rebuild_model(model=am.model.eval(), max_seq_len=512, device="cpu")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def export(n: int, path: Path, dynamic: bool) -> None:
        dummy = (torch.zeros(1, n, 560), torch.tensor([n], dtype=torch.int32),
                 torch.tensor([12], dtype=torch.int32), torch.tensor([15], dtype=torch.int32))
        axes = {"speech": {1: "t"}, "ctc_logits": {1: "t4"}} if dynamic else None
        with torch.no_grad():
            torch.onnx.export(m, dummy, str(path), input_names=NAMES_IN, output_names=NAMES_OUT,
                              opset_version=a.opset, dynamic_axes=axes, do_constant_folding=True,
                              dynamo=False)
        print(f"OK {path.name} ({path.stat().st_size / 2**20:.0f} MB)", flush=True)

    for n in [int(b) for b in a.buckets.split(",")]:
        export(n, out / f"sv_t{n}.onnx", dynamic=False)
    export(84, out / "model.onnx", dynamic=True)  # parity reference for any length

    for f in ["am.mvn", "config.yaml", *[p.name for p in src.glob("*.bpe.model")]]:
        shutil.copy2(src / f, out / f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
