"""Export multilingual-e5 as fixed-shape .tflite encoders: (input_ids, attention_mask) -> L2-normalised mean-pooled embedding.

Runs inside the litert-torch converter image (litert-torch 0.9.4, ai-edge-quantizer, ai-edge-litert). One graph per
sequence length (fixed shapes so XNNPACK attaches); pooling and normalisation are in the graph, so the device only
tokenises, pads with <pad>=1 and reads a [1, D] vector. fp32 + w8 (weight-only) + i8 (dynamic) per length.

  python export_e5.py --model intfloat/multilingual-e5-small --lengths 64,512 --out /out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

SAMPLES = ["query: 독도는 어느 나라 영토인가?", "passage: 세종대왕은 조선의 제4대 왕으로 훈민정음을 창제하였다.",
           "query: 서울역 3번 출구는 어디에 있나요", "passage: The quick brown fox jumps over the lazy dog. 2026년 10월 17일"]


class E5(torch.nn.Module):
    def __init__(self, encoder) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        e = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return torch.nn.functional.normalize(e, dim=-1)


def main() -> int:
    import litert_torch
    from ai_edge_litert.interpreter import Interpreter
    from ai_edge_quantizer import quantizer, recipe
    from transformers import AutoModel, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--lengths", default="64,512")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = E5(AutoModel.from_pretrained(args.model, attn_implementation="eager").eval()).eval()
    stem = args.model.split("/")[-1].replace("multilingual-", "").replace("-", "_")
    meta = {"model": args.model, "pad_id": tok.pad_token_id, "graphs": {}}
    for length in (int(x) for x in args.lengths.split(",")):
        enc = tok(SAMPLES, padding="max_length", truncation=True, max_length=length, return_tensors="pt")
        ids, mask = enc["input_ids"].to(torch.int32), enc["attention_mask"].to(torch.int32)
        with torch.no_grad():
            ref = model(ids.long(), mask.long()).numpy()
        fp32 = out / f"{stem}_t{length}_fp32.tflite"
        litert_torch.convert(model, (ids[:1], mask[:1])).export(str(fp32))
        paths = [fp32]
        for name, rec in (("w8", recipe.weight_only_wi8_afp32()), ("i8", recipe.dynamic_wi8_afp32())):
            qt = quantizer.Quantizer(str(fp32))
            qt.load_quantization_recipe(rec)
            path = out / f"{stem}_t{length}_{name}.tflite"
            qt.quantize().export_model(str(path), overwrite=True)
            paths.append(path)
        for path in paths:
            runner = Interpreter(model_path=str(path)).get_signature_runner()
            names = list(runner.get_input_details())  # converter names them args_0 (ids), args_1 (mask)
            cos = []
            for i in range(len(SAMPLES)):
                got = list(runner(**{names[0]: ids[i:i + 1].numpy(), names[1]: mask[i:i + 1].numpy()}).values())[0][0]
                cos.append(float(got @ ref[i]))
            meta["graphs"][path.name] = {"length": length, "inputs": names, "min_cos_vs_hf": min(cos),
                                         "mb": round(path.stat().st_size / 1e6, 1)}
            print(f"[export] {path.name} {path.stat().st_size / 1e6:.1f} MB inputs={names} min cos vs HF {min(cos):.6f}")
    (out / "e5_tflite.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
