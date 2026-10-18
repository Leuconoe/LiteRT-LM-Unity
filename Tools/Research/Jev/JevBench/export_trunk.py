"""Export gemma-3-270m-it layers 0..11 as a fixed 48-token .tflite that returns layer-12 hidden states.

Runs inside the litert-torch converter image (litert-torch 0.9.4, ai-edge-quantizer). LiteRT-LM never
exposes hidden states, so the Jev head needs its own trunk graph.

Why the attention is written out by hand: converting the HF layer as-is gives a graph that is exact at
position 0 and off by thousands everywhere else. Bisected to `matmul(q, repeat_kv(k).transpose(2, 3))` —
q and k after RoPE match to 1e-5, the scores do not (Δ 288). gemma-3-270m has ONE kv head, so the kv
expand is skipped entirely: q is flattened to [1, H·T, D] and multiplied with the single k/v.
No attention-mask input: tokens are right-padded and the constant causal mask hides the pads.

  python export_trunk.py --out /out [--layers 12] [--tokens 48]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers.models.gemma3 import modeling_gemma3 as g3


class Trunk(torch.nn.Module):
    """Embedding + the first `layers` decoder layers, constant RoPE and causal mask, no final norm."""

    def __init__(self, lm, layers: int, tokens: int) -> None:
        super().__init__()
        text, cfg = lm.model, lm.model.config
        assert cfg.num_key_value_heads == 1, "the flattened attention assumes a single kv head"
        self.embed, self.layers, self.types = text.embed_tokens, text.layers[:layers], cfg.layer_types[:layers]
        self.heads, self.hd, self.tokens = cfg.num_attention_heads, cfg.head_dim, tokens
        self.scale = cfg.query_pre_attn_scalar ** -0.5
        pos, dummy = torch.arange(tokens)[None], torch.zeros(1, tokens, cfg.hidden_size)
        with torch.no_grad():
            for kind in set(self.types):
                cos, sin = text.rotary_emb(dummy, pos, kind)
                self.register_buffer(f"cos_{kind}", cos.clone())
                self.register_buffer(f"sin_{kind}", sin.clone())
        # ponytail: sliding layers use the plain causal mask — exact while tokens <= sliding_window (512).
        assert tokens <= cfg.sliding_window
        self.register_buffer("mask", torch.full((tokens, tokens), torch.finfo(torch.float32).min).triu(1)[None])

    def attn(self, a, h, kind):
        t, hd, heads = self.tokens, self.hd, self.heads
        q = a.q_norm(a.q_proj(h).view(1, t, heads, hd).transpose(1, 2))
        k = a.k_norm(a.k_proj(h).view(1, t, 1, hd).transpose(1, 2))
        v = a.v_proj(h).view(1, t, hd)
        q, k = g3.apply_rotary_pos_emb(q, k, getattr(self, f"cos_{kind}"), getattr(self, f"sin_{kind}"))
        scores = torch.matmul(q.reshape(1, heads * t, hd), k.reshape(1, t, hd).transpose(1, 2)) * self.scale
        weights = torch.softmax(scores.view(1, heads, t, t) + self.mask, -1)
        out = torch.matmul(weights.reshape(1, heads * t, t), v).view(1, heads, t, hd)
        return a.o_proj(out.transpose(1, 2).reshape(1, t, heads * hd))

    def forward(self, input_ids):
        h = self.embed(input_ids)
        for layer, kind in zip(self.layers, self.types):
            h = h + layer.post_attention_layernorm(self.attn(layer.self_attn, layer.input_layernorm(h), kind))
            h = h + layer.post_feedforward_layernorm(layer.mlp(layer.pre_feedforward_layernorm(h)))
        return h


def main() -> int:
    import litert_torch
    from ai_edge_litert.interpreter import Interpreter
    from ai_edge_quantizer import quantizer, recipe
    from transformers import AutoModelForCausalLM

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-270m-it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--tokens", type=int, default=48)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="eager").eval()
    trunk = Trunk(lm, args.layers, args.tokens).eval()
    ids = torch.randint(3, lm.config.vocab_size, (1, args.tokens), dtype=torch.int32)
    ids[0, 0] = 2  # <bos>
    with torch.no_grad():
        ref = lm(input_ids=ids.long(), output_hidden_states=True).hidden_states[args.layers][0].numpy()
        print(f"[trunk] hand-written vs HF hidden_states[{args.layers}]: max |Δ| "
              f"{abs(trunk(ids)[0].numpy() - ref).max():.2e}")

    stem = f"gemma3_270m_l{args.layers}_t{args.tokens}"
    fp32 = out / f"{stem}_fp32.tflite"
    litert_torch.convert(trunk, (torch.zeros_like(ids),)).export(str(fp32))
    paths = [fp32]
    for name, rec in (("w8", recipe.weight_only_wi8_afp32()), ("i8", recipe.dynamic_wi8_afp32())):
        qt = quantizer.Quantizer(str(fp32))
        qt.load_quantization_recipe(rec)
        path = out / f"{stem}_{name}.tflite"
        qt.quantize().export_model(str(path), overwrite=True)
        paths.append(path)
    for path in paths:  # random ids, not the trace input, so a baked-in constant would show
        it = Interpreter(model_path=str(path))
        it.allocate_tensors()
        it.set_tensor(it.get_input_details()[0]["index"], ids.numpy())
        it.invoke()
        got = it.get_tensor(it.get_output_details()[0]["index"])[0]
        print(f"[export] {path.name} {path.stat().st_size / 1e6:.1f} MB, tflite vs HF max |Δ| {abs(got - ref).max():.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
