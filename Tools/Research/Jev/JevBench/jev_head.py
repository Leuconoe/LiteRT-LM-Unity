"""Offline Jev sample: a jevlike decision head on frozen gemma-3-270m-it layer-12 states.

  train     extract layer-12 states, train ONE head for two Jev questions — `area` (which option of the
            spec, or none) and `noul` (is this an executable command?) — score the eval cases and save the
            head for the C# port (jev_head.json + jev_head.bin).
  parity    check an exported trunk .tflite (embed + layers 0..11 -> layer-12 states, 48 tokens) against HF.
  device    grade a device run (Run-LiteRtLmAndroidJevSmokeTest.ps1 results) against the same head on the desktop.
  selfcheck tiny shape / mapping asserts.

The options come from a spec (sample_areas.json: criterion sentence + the contract functions each covers);
training records are FunctionGemma make_dataset.py output (text + function), so a contract and its data
serve both samples. The head is jevlike's AttentionHead (github.com/vinnylarouge/jevlike, MIT) plus
dropout, with the recipe skykim/voice-agent-unity found best: layer 12, 48 tokens, drop <bos> (its
activations dwarf the rest), rank 128, AdamW 2e-3, dropout 0.1, label smoothing 0.1, <=20 epochs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FG = ROOT / "Tools" / "Research" / "FunctionGemma" / "FunctionGemmaBench"
WORK = ROOT / "External" / "jev-work"
MODEL = "google/gemma-3-270m-it"
LAYER, MAX_TOKENS = 12, 48
THRESHOLDS = {"run": 0.5, "ask": 0.35, "margin": 0.2}  # voice-agent-unity TurnRouting


class Spec:
    """area options (key -> criterion, functions) and the noul question, from a spec JSON."""

    def __init__(self, path: Path):
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        self.areas = {k: (v["criterion"], v["functions"]) for k, v in doc["areas"].items()}
        self.keys = list(self.areas)
        self.fn_to_area = {fn: k for k, (_, fns) in self.areas.items() for fn in fns}
        self.noul = doc["noul"]
        self.noul_keys = list(self.noul)
        assert "none" in self.areas, "the spec needs a `none` area (refusals)"

    def area_of(self, function: str | None) -> str:
        return self.fn_to_area.get(function or "", "none")  # unknown / invented names fall to none, as the app guard does


class AttentionHead(nn.Module):
    """Vendored from jevlike/model.py (MIT) — one score per option from context tokens and option vectors."""

    def __init__(self, width: int, rank: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.context_norm = nn.LayerNorm(width)
        self.option_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, rank, bias=False)
        self.key = nn.Linear(width, rank, bias=False)
        self.value = nn.Linear(width, rank, bias=False)
        self.drop = nn.Dropout(dropout)
        self.rank = rank

    def forward(self, context, context_mask, options):  # [B,L,W], [B,L] bool, [N,W] -> [B,N]
        context = self.drop(self.context_norm(context.float()))
        query = self.query(self.option_norm(options.float()))
        scores = torch.einsum("nr,blr->bnl", query, self.key(context)) / math.sqrt(self.rank)
        scores = scores.masked_fill(~context_mask[:, None, :], torch.finfo(scores.dtype).min)
        attended = torch.einsum("bnl,blr->bnr", scores.softmax(-1), self.value(context))
        return (query[None] * attended).sum(-1) / math.sqrt(self.rank)


def load_encoder(device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device).eval()
    return tok, model


@torch.no_grad()
def encode(tok, model, texts: list[str], batch: int = 128):
    """Layer-12 states with <bos> dropped: ([N, 47, W] fp16, [N, 47] bool). Right padding is exact for a causal LM."""
    width = model.config.hidden_size
    states = torch.zeros(len(texts), MAX_TOKENS - 1, width, dtype=torch.float16)
    mask = torch.zeros(len(texts), MAX_TOKENS - 1, dtype=torch.bool)
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], max_length=MAX_TOKENS, truncation=True, padding="max_length", return_tensors="pt")
        hidden = model(input_ids=enc.input_ids.to(model.device), attention_mask=enc.attention_mask.to(model.device),
                       output_hidden_states=True).hidden_states[LAYER]
        states[i:i + batch] = hidden[:, 1:].half().cpu()
        mask[i:i + batch] = enc.attention_mask[:, 1:].bool()
    return states, mask


def tflite_encoder(path: str):
    """Same contract as `encode`, but the states come from an exported trunk graph — train the head on
    exactly what the device trunk produces (w8 drifts from HF on Gemma's massive activations)."""
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path, num_threads=8)
    it.allocate_tensors()
    inp, outp = it.get_input_details()[0], it.get_output_details()[0]

    def run(tok, _model, texts: list[str]):
        states = torch.zeros(len(texts), MAX_TOKENS - 1, int(outp["shape"][2]), dtype=torch.float16)
        mask = torch.zeros(len(texts), MAX_TOKENS - 1, dtype=torch.bool)
        for i, text in enumerate(texts):
            enc = tok(text, max_length=MAX_TOKENS, truncation=True, padding="max_length", return_tensors="np")
            it.set_tensor(inp["index"], enc["input_ids"].astype(np.int32))
            it.invoke()
            states[i] = torch.from_numpy(it.get_tensor(outp["index"])[0, 1:]).half()
            mask[i] = torch.from_numpy(enc["attention_mask"][0, 1:].astype(bool))
        return states, mask

    return run


def pooled(states, mask):
    w = mask.unsqueeze(-1).float()
    return (states.float() * w).sum(1) / w.sum(1).clamp_min(1)


def read_jsonl(path: Path) -> list[dict]:
    # utf-8-sig: the device runner's first File.AppendAllText(…, Encoding.UTF8) writes a BOM
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def train_head(train, val, area_opts, noul_opts, device, epochs=20, seed=7):
    (xs, ms, ya, yn), (vxs, vms, vya, _) = train, val
    torch.manual_seed(seed)
    random.seed(seed)
    head = AttentionHead(xs.shape[-1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=0.01)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    best, best_state, area_opts, noul_opts = -1.0, None, area_opts.to(device), noul_opts.to(device)
    order = list(range(len(xs)))
    for epoch in range(epochs):
        head.train()
        random.shuffle(order)
        for i in range(0, len(order), 64):
            idx = order[i:i + 64]
            x, m = xs[idx].to(device), ms[idx].to(device)
            loss = ce(head(x, m, area_opts), ya[idx].to(device)) + ce(head(x, m, noul_opts), yn[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = (score(head, vxs, vms, area_opts, device).argmax(1) == vya).float().mean().item()
        print(f"[head] epoch {epoch + 1}: val area acc {acc:.4f}")
        if acc > best:
            best, best_state = acc, {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head.eval(), best


@torch.no_grad()
def score(head, xs, ms, opts, device) -> torch.Tensor:
    head.eval()
    return torch.cat([head(xs[i:i + 256].to(device), ms[i:i + 256].to(device), opts.to(device)).softmax(-1).cpu()
                      for i in range(0, len(xs), 256)])


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10):
    edges = np.linspace(0, 1, bins + 1)
    total, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            total += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
            rows.append((f"{lo:.1f}-{hi:.1f}", int(sel.sum()), float(correct[sel].mean())))
    return total, rows


def route(p: np.ndarray, keys: list[str]) -> tuple[str, int]:
    """voice-agent-unity TurnRouting: run >= 0.5; ask >= 0.35 with a 0.2 lead; otherwise refuse. `none` never runs."""
    order = np.argsort(-p)
    top, p1, p2 = int(order[0]), float(p[order[0]]), float(p[order[1]])
    if keys[top] == "none":
        return "refuse", top
    if p1 >= THRESHOLDS["run"]:
        return "run", top
    if p1 >= THRESHOLDS["ask"] and p1 - p2 >= THRESHOLDS["margin"]:
        return "ask", top
    return "refuse", top


def cmd_train(args) -> int:
    spec = Spec(args.spec)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load_encoder(device)
    encode_fn = tflite_encoder(args.trunk) if args.trunk else encode
    data_dir = Path(args.data)
    train_rows, val_rows = read_jsonl(data_dir / "train.jsonl"), read_jsonl(data_dir / "val.jsonl")
    t0 = time.perf_counter()
    xs, ms = encode_fn(tok, model, [r["text"] for r in train_rows])
    vxs, vms = encode_fn(tok, model, [r["text"] for r in val_rows])
    print(f"[states] {len(train_rows)}+{len(val_rows)} sentences in {time.perf_counter() - t0:.1f}s")

    def labels(rows):
        a = torch.tensor([spec.keys.index(spec.area_of(r["function"])) for r in rows])
        return a, (a == spec.keys.index("none")).long()  # noul: 0 = yes, 1 = no

    ya, yn = labels(train_rows)
    vya, vyn = labels(val_rows)
    ao_s, ao_m = encode(tok, model, [spec.areas[k][0] for k in spec.keys])
    no_s, no_m = encode(tok, model, [spec.noul[k] for k in spec.noul_keys])
    area_opts, noul_opts = pooled(ao_s, ao_m), pooled(no_s, no_m)
    head, best = train_head((xs, ms, ya, yn), (vxs, vms, vya, vyn), area_opts, noul_opts, device, args.epochs)
    print(f"[head] best val area acc {best:.4f}")

    report = ["| set | n | area acc | noul acc | ECE | run ✓ | run ✗ | ask | refuse (want none) | refuse (want cmd) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    cx, cm = encode(tok, model, [c["text"] for c in cases])
    pa, pn = score(head, cx, cm, area_opts, device).numpy(), score(head, cx, cm, noul_opts, device).numpy()
    want = np.array([spec.keys.index(spec.area_of(c["function"])) for c in cases])
    acc = (pa.argmax(1) == want).mean()
    noul_acc = (pn.argmax(1) == (want == spec.keys.index("none"))).mean()
    e, rows = ece(pa.max(1), (pa.argmax(1) == want).astype(float))
    counts = defaultdict(int)
    for p, w in zip(pa, want):
        decision, top = route(p, spec.keys)
        if decision == "refuse":
            counts["refuse_none" if spec.keys[w] == "none" else "refuse_cmd"] += 1
        elif decision == "run":
            counts["run_ok" if top == w else "run_bad"] += 1
        else:
            counts["ask"] += 1
    report.append(f"| {Path(args.cases).stem} | {len(cases)} | {acc:.3f} | {noul_acc:.3f} | {e:.3f} | {counts['run_ok']} | "
                  f"{counts['run_bad']} | {counts['ask']} | {counts['refuse_none']} | {counts['refuse_cmd']} |")
    print("\n".join(report))
    print("reliability: " + ", ".join(f"{b} n={n} acc={a:.2f}" for b, n, a in rows))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": head.state_dict(), "area_opts": area_opts, "noul_opts": noul_opts, "spec": str(args.spec)}, out / "jev_head.pt")
    export_head(head, area_opts, noul_opts, spec, out)
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[saved] {out}")
    return 0


def export_head(head, area_opts, noul_opts, spec: Spec, out: Path) -> None:
    """Little-endian float32 blob + manifest — what a C# head reads (LayerNorm eps 1e-5, nn.Linear weight = [rank, width])."""
    sd = head.state_dict()
    tensors = [("context_norm.weight", sd["context_norm.weight"]), ("context_norm.bias", sd["context_norm.bias"]),
               ("option_norm.weight", sd["option_norm.weight"]), ("option_norm.bias", sd["option_norm.bias"]),
               ("query.weight", sd["query.weight"]), ("key.weight", sd["key.weight"]), ("value.weight", sd["value.weight"]),
               ("area_options", area_opts), ("noul_options", noul_opts)]
    blob, manifest, offset = bytearray(), [], 0
    for name, t in tensors:
        arr = t.detach().cpu().float().contiguous().numpy().astype("<f4")
        blob += arr.tobytes()
        manifest.append({"name": name, "shape": list(arr.shape), "offset": offset})
        offset += arr.nbytes
    (out / "jev_head.bin").write_bytes(bytes(blob))
    (out / "jev_head.json").write_text(json.dumps({
        "model": MODEL, "layer": LAYER, "max_tokens": MAX_TOKENS, "drop_bos": True, "layernorm_eps": 1e-5,
        "rank": head.rank, "width": int(sd["query.weight"].shape[1]), "tensors": manifest,
        "questions": {"area": spec.keys, "noul": spec.noul_keys}, "thresholds": THRESHOLDS,
        "area_criteria": {k: spec.areas[k][0] for k in spec.keys}, "noul_criteria": spec.noul,
    }, ensure_ascii=False, indent=1), encoding="utf-8")


def load_head(path: Path, width: int):
    payload = torch.load(path, weights_only=False)
    head = AttentionHead(width)
    head.load_state_dict(payload["state"])
    return head.eval(), payload  # eval: dropout off — otherwise agreement counts are noise


def cmd_parity(args) -> int:
    from ai_edge_litert.interpreter import Interpreter

    tok, model = load_encoder("cpu")
    texts = [c["text"] for c in json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"][: args.n]]
    it = Interpreter(model_path=args.tflite, num_threads=4)
    it.allocate_tensors()
    inp, outp = it.get_input_details()[0], it.get_output_details()[0]
    print(f"[tflite] input {inp['name']} {inp['shape']} {inp['dtype'].__name__}, output {outp['name']} {outp['shape']}")
    head, payload = load_head(Path(args.head) / "jev_head.pt", model.config.hidden_size)
    max_diff, pooled_diff, agree, times = 0.0, 0.0, 0, []
    for text in texts:
        enc = tok(text, max_length=MAX_TOKENS, truncation=True, padding="max_length", return_tensors="pt")
        ids, real = enc.input_ids, enc.attention_mask[0, 1:].bool()
        with torch.no_grad():  # no attention_mask: the device graph sees pads as tokens, causal masking hides them
            ref = model(input_ids=ids, output_hidden_states=True).hidden_states[LAYER][0, 1:]
        it.set_tensor(inp["index"], ids.numpy().astype(inp["dtype"]))
        t0 = time.perf_counter()
        it.invoke()
        times.append((time.perf_counter() - t0) * 1000)
        got = torch.from_numpy(it.get_tensor(outp["index"]))[0, 1:]
        max_diff = max(max_diff, (got[real] - ref[real]).abs().max().item())
        pooled_diff = max(pooled_diff, (got[real].mean(0) - ref[real].mean(0)).abs().max().item())
        agree += head(ref[None], real[None], payload["area_opts"]).argmax().item() == head(got[None], real[None], payload["area_opts"]).argmax().item()
    for _ in range(30):
        t0 = time.perf_counter()
        it.invoke()
        times.append((time.perf_counter() - t0) * 1000)
    print(f"[parity] {len(texts)} texts: max |Δstate| {max_diff:.4f}, max |Δpooled| {pooled_diff:.4f}, "
          f"head argmax agree {agree}/{len(texts)}; invoke p50 {statistics.median(times):.1f} ms (4 threads)")
    return 0


def cmd_device(args) -> int:
    """Grade a device run (Run-LiteRtLmAndroidJevSmokeTest.ps1 results.jsonl) and compare it, prompt by prompt,
    with the same head on the same trunk graph on the desktop."""
    spec = Spec(args.spec)
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    rows = read_jsonl(Path(args.results))
    assert [r["prompt"] for r in rows] == [c["text"] for c in cases[: len(rows)]], "prompt order mismatch"
    tok, model = load_encoder("cpu")
    head, payload = load_head(Path(args.head) / "jev_head.pt", model.config.hidden_size)
    xs, ms = tflite_encoder(args.trunk)(tok, model, [r["prompt"] for r in rows])
    desk = score(head, xs, ms, payload["area_opts"], "cpu").numpy()
    dev = np.array([r["probabilities"] for r in rows])
    want = np.array([spec.keys.index(spec.area_of(c["function"])) for c in cases[: len(rows)]])
    print(f"[device] {len(rows)} prompts: area acc device {(dev.argmax(1) == want).mean():.3f} / desktop "
          f"{(desk.argmax(1) == want).mean():.3f}; top-1 agree {(dev.argmax(1) == desk.argmax(1)).mean():.3f}; "
          f"max |Δp| {np.abs(dev - desk).max():.4f}; trunk ms p50 {statistics.median(r['trunkMs'] for r in rows):.1f}, "
          f"head ms p50 {statistics.median(r['headMs'] for r in rows):.1f}")
    return 0


def selfcheck(spec_path: Path) -> int:
    spec = Spec(spec_path)
    contract = json.loads((FG / "sample_contract.json").read_text(encoding="utf-8"))
    names = {t["function"]["name"] for t in contract}
    assert names == set(spec.fn_to_area), names ^ set(spec.fn_to_area)  # every contract function has exactly one area
    assert spec.area_of("Nope") == "none" and spec.area_of("DefaultResponse") == "none"
    head = AttentionHead(16, rank=4)
    out = head(torch.randn(2, 5, 16), torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool), torch.randn(3, 16))
    assert out.shape == (2, 3)
    n = len(spec.keys)
    first_real = next(i for i, k in enumerate(spec.keys) if k != "none")
    p = np.full(n, 0.4 / (n - 1)); p[first_real] = 0.6
    assert route(p, spec.keys)[0] == "run"
    p = np.full(n, 0.45 / (n - 2)); p[first_real] = 0.4; p[spec.keys.index("none")] = 0.15
    assert route(p, spec.keys)[0] == "ask"
    p = np.full(n, 0.1 / (n - 1)); p[spec.keys.index("none")] = 0.9
    assert route(p, spec.keys)[0] == "refuse"
    print(f"ok — {len(names)} functions mapped to {n} areas")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--spec", default=str(HERE / "sample_areas.json"))
    common.add_argument("--cases", default=str(FG / "sample_eval_cases.json"))
    t = sub.add_parser("train", parents=[common])
    t.add_argument("--data", default=str(FG / "data"), help="FunctionGemma make_dataset.py output dir")
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--trunk", default="", help="take states from this exported trunk .tflite instead of HF")
    t.add_argument("--out", default=str(WORK / "head"))
    p = sub.add_parser("parity", parents=[common])
    p.add_argument("--tflite", required=True)
    p.add_argument("--head", default=str(WORK / "head"))
    p.add_argument("--n", type=int, default=20)
    d = sub.add_parser("device", parents=[common])
    d.add_argument("--results", required=True)
    d.add_argument("--trunk", default=str(WORK / "trunk" / "gemma3_270m_l12_t48_w8.tflite"))
    d.add_argument("--head", default=str(WORK / "head-w8"))
    sub.add_parser("selfcheck", parents=[common])
    args = ap.parse_args()
    return {"train": cmd_train, "parity": cmd_parity, "device": cmd_device, "selfcheck": lambda a: selfcheck(Path(a.spec))}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
