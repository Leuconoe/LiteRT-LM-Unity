"""Offline Jev sample, stage 1: a jevlike decision head on frozen gemma-3-270m-it layer-12 states.

  train     extract layer-12 states, train ONE head for two Jev questions — `area` (which UI area, or none) and
            `noul` (is this an executable UI command?) — score the FunctionGemma eval sets, compare with the
            fine-tuned router's implied area, and save the head for a C# port.
  parity    check an exported trunk .tflite (embed + layers 0..11 -> layer-12 states, 48 tokens) against HF.
  selfcheck tiny shape / mapping asserts.

The head is jevlike's AttentionHead (github.com/vinnylarouge/jevlike, MIT) plus dropout, with the recipe
skykim/voice-agent-unity found best: layer 12, 48 tokens, drop <bos> (its activations dwarf the rest),
rank 128, AdamW 2e-3, dropout 0.1, label smoothing 0.1, <=20 epochs.
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

ROOT = Path(__file__).resolve().parents[4]
FG = ROOT / "Tools" / "Research" / "FunctionGemma" / "FunctionGemmaBench"
WORK = ROOT / "External" / "jev-work"
MODEL = "google/gemma-3-270m-it"
LAYER, MAX_TOKENS = 12, 48
CASE_SETS = ["eval_cases.json", "heldout_cases.json", "heldout2_cases.json", "heldout3_cases.json", "heldout4_cases.json"]
THRESHOLDS = {"run": 0.5, "ask": 0.35, "margin": 0.2}  # voice-agent-unity TurnRouting

# Jev "choice" question: option key -> (declared criterion text, functions it covers).
AREAS = {
    "team": ("팀을 선택하거나 만들고 복귀·해체하거나 팀 전술을 시작·편집하는 명령",
             ["SelectTeam", "EnterTeamCreation", "CreateTeam", "ReturnTeam", "DisbandTeam", "StartTeamTactics", "EditTeamTactics"]),
    "drone": ("번호로 지정한 드론을 선택·복귀·팀 제외·개별제어하거나 드론 전술을 편집하는 명령",
              ["SelectDrone", "EditDroneTactics", "ReturnDrone", "DetachDrone", "StartManualFlight"]),
    "control": ("팀 제어권을 지정·이양·회수·요청·반납하는 명령",
                ["AssignTeamControl", "RevokeTeamControl", "RequestTeamControl", "ReturnTeamControl"]),
    "marker_preset": ("마커나 프리셋으로 이동하거나 만들거나 지우는 명령",
                      ["MoveToMarker", "MoveToPreset", "EnterMarkerCreation", "CreatePreset", "DeleteMarker", "DeletePreset"]),
    "tactics": ("AI 전술 추천을 요청하거나 추천을 취소하는 명령", ["RequestTacticsRecommendation", "CancelTacticsRecommendation"]),
    "window": ("설정·필터·알림 상세·전술 지도 창을 열거나 닫는 명령", ["OpenWindow", "CloseWindow"]),
    "camera": ("팀이나 드론의 카메라 영상을 띄우거나 짐벌을 초기화하는 명령", ["ShowTeamCamera", "ShowDroneCamera", "ResetCameraGimbal"]),
    "camera_window": ("카메라 창을 확대하거나 축소하는 명령", ["EnlargeCameraWindow", "ShrinkCameraWindow"]),
    "filter": ("지도 표시 필터를 켜거나 끄는 명령", ["ApplyDisplayFilters", "ClearDisplayFilters"]),
    "confirm": ("컨펌 창을 승인하거나 거절하는 명령", ["AcceptConfirmWindow", "RejectConfirmWindow"]),
    "video_through": ("비디오 쓰루(패스스루)를 켜거나 끄는 명령", ["EnableVideoThrough", "DisableVideoThrough"]),
    "view": ("팀 뷰·팀 전체 뷰·드론 뷰로 시점을 옮기는 명령", ["MoveToTeamView", "MoveToTeamOverview", "MoveToDroneView"]),
    "panel": ("패널 위치를 원위치로 초기화하는 명령", ["ResetPanelPositions"]),
    "recording": ("전술 녹화를 시작하거나 중지하는 명령", ["StartRecording", "StopRecording"]),
    "none": ("지원하지 않거나 모호하거나 두 가지 동작이 섞인, 실행하지 않을 말", ["DefaultResponse"]),
}
AREA_KEYS = list(AREAS)
FN_TO_AREA = {fn: key for key, (_, fns) in AREAS.items() for fn in fns}
# Jev "noul" question, asked as a two-option choice through the same head.
NOUL = {"yes": "바로 실행할 수 있는 UI 명령", "no": "실행하지 않을 말"}
NOUL_KEYS = list(NOUL)


def area_of(function: str | None) -> str:
    return FN_TO_AREA.get(function or "", "none")  # unknown / invented names fall to none, as the app guard does


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


def route(p: np.ndarray) -> tuple[str, int]:
    """voice-agent-unity TurnRouting: run >= 0.5; ask >= 0.35 with a 0.2 lead; otherwise refuse. `none` never runs."""
    order = np.argsort(-p)
    top, p1, p2 = int(order[0]), float(p[order[0]]), float(p[order[1]])
    if AREA_KEYS[top] == "none":
        return "refuse", top
    if p1 >= THRESHOLDS["run"]:
        return "run", top
    if p1 >= THRESHOLDS["ask"] and p1 - p2 >= THRESHOLDS["margin"]:
        return "ask", top
    return "refuse", top


def fg_predictions(case_file: str) -> tuple[str, dict[str, str]] | None:
    stem = case_file.replace("_cases.json", "")
    for label in ("v3e", "v2e"):
        path = ROOT / "Builds" / "Logs" / f"fg-ft-functiongemma-none-{label}-{stem}.jsonl"
        if path.exists():
            rows = [r for r in read_jsonl(path) if r.get("id")]
            return label, {r["id"]: area_of((r.get("got") or {}).get("function")) for r in rows}
    return None


def cmd_train(args) -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load_encoder(device)
    encode = tflite_encoder(args.trunk) if args.trunk else globals()["encode"]
    data_dir = FG / args.data
    train_rows, val_rows = read_jsonl(data_dir / "train.jsonl"), read_jsonl(data_dir / "val.jsonl")
    t0 = time.perf_counter()
    xs, ms = encode(tok, model, [r["text"] for r in train_rows])
    vxs, vms = encode(tok, model, [r["text"] for r in val_rows])
    print(f"[states] {len(train_rows)}+{len(val_rows)} sentences in {time.perf_counter() - t0:.1f}s")

    def labels(rows):
        a = torch.tensor([AREA_KEYS.index(area_of(r["function"])) for r in rows])
        return a, (a == AREA_KEYS.index("none")).long()  # noul: 0 = yes, 1 = no

    ya, yn = labels(train_rows)
    vya, vyn = labels(val_rows)
    ao_s, ao_m = encode(tok, model, [AREAS[k][0] for k in AREA_KEYS])
    no_s, no_m = encode(tok, model, [NOUL[k] for k in NOUL_KEYS])
    area_opts, noul_opts = pooled(ao_s, ao_m), pooled(no_s, no_m)
    head, best = train_head((xs, ms, ya, yn), (vxs, vms, vya, vyn), area_opts, noul_opts, device, args.epochs)
    print(f"[head] best val area acc {best:.4f}")

    report = ["| set | n | area acc | noul acc | ECE | run ✓ | run ✗ | ask | refuse (want none) | refuse (want cmd) | FG area acc | FG wrong runs |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    reliability = {}
    for name in CASE_SETS:
        path = FG / name
        if not path.exists():
            continue
        cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
        cx, cm = encode(tok, model, [c["text"] for c in cases])
        pa, pn = score(head, cx, cm, area_opts, device).numpy(), score(head, cx, cm, noul_opts, device).numpy()
        want = np.array([AREA_KEYS.index(area_of(c["function"])) for c in cases])
        acc = (pa.argmax(1) == want).mean()
        noul_acc = (pn.argmax(1) == (want == AREA_KEYS.index("none"))).mean()
        e, rows = ece(pa.max(1), (pa.argmax(1) == want).astype(float))
        reliability[name] = rows
        counts = defaultdict(int)
        for p, w in zip(pa, want):
            decision, top = route(p)
            wants_none = AREA_KEYS[w] == "none"
            if decision == "refuse":
                counts["refuse_none" if wants_none else "refuse_cmd"] += 1
            elif decision == "run":
                counts["run_ok" if top == w else "run_bad"] += 1
            else:
                counts["ask"] += 1
        fg = fg_predictions(name)
        fg_acc = fg_bad = "—"
        if fg:
            label, pred = fg
            hits = [(pred.get(c["id"]), area_of(c["function"])) for c in cases if c["id"] in pred]
            fg_acc = f"{sum(p == w for p, w in hits) / len(hits):.3f} ({label})"
            fg_bad = str(sum(p != w and p != "none" for p, w in hits))
        report.append(f"| {name.replace('_cases.json', '')} | {len(cases)} | {acc:.3f} | {noul_acc:.3f} | {e:.3f} | "
                      f"{counts['run_ok']} | {counts['run_bad']} | {counts['ask']} | {counts['refuse_none']} | "
                      f"{counts['refuse_cmd']} | {fg_acc} | {fg_bad} |")
    print("\n".join(report))
    for name, rows in reliability.items():
        print(f"reliability {name}: " + ", ".join(f"{b} n={n} acc={a:.2f}" for b, n, a in rows))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": head.state_dict(), "area_opts": area_opts, "noul_opts": noul_opts}, out / "jev_head.pt")
    export_head(head, area_opts, noul_opts, out)
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[saved] {out}")
    return 0


def export_head(head, area_opts, noul_opts, out: Path) -> None:
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
        "questions": {"area": AREA_KEYS, "noul": NOUL_KEYS}, "thresholds": THRESHOLDS,
        "area_criteria": {k: AREAS[k][0] for k in AREA_KEYS}, "noul_criteria": NOUL,
    }, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_parity(args) -> int:
    from ai_edge_litert.interpreter import Interpreter

    tok, model = load_encoder("cpu")
    texts = [c["text"] for c in json.loads((FG / "heldout3_cases.json").read_text(encoding="utf-8"))["cases"][: args.n]]
    it = Interpreter(model_path=args.tflite, num_threads=4)
    it.allocate_tensors()
    inp, outp = it.get_input_details()[0], it.get_output_details()[0]
    print(f"[tflite] input {inp['name']} {inp['shape']} {inp['dtype'].__name__}, output {outp['name']} {outp['shape']}")
    payload = torch.load(WORK / "head" / "jev_head.pt", weights_only=False)
    head = AttentionHead(model.config.hidden_size)
    head.load_state_dict(payload["state"])
    head.eval()  # dropout off — otherwise the agreement count is noise
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
        p_ref, p_got = ref[real].mean(0), got[real].mean(0)
        pooled_diff = max(pooled_diff, (p_got - p_ref).abs().max().item())
        a_ref = head(ref[None], real[None], payload["area_opts"]).argmax().item()
        a_got = head(got[None], real[None], payload["area_opts"]).argmax().item()
        agree += a_ref == a_got
    for _ in range(30):
        t0 = time.perf_counter()
        it.invoke()
        times.append((time.perf_counter() - t0) * 1000)
    print(f"[parity] {len(texts)} commands: max |Δstate| {max_diff:.4f}, max |Δpooled| {pooled_diff:.4f}, "
          f"head argmax agree {agree}/{len(texts)}; invoke p50 {statistics.median(times):.1f} ms (4 threads)")
    return 0


def cmd_device(args) -> int:
    """Grade a device run (Run-LiteRtLmAndroidJevSmokeTest.ps1 results.jsonl) and compare it, prompt by prompt,
    with the same head on the same trunk graph on the desktop."""
    cases = json.loads((FG / args.cases).read_text(encoding="utf-8"))["cases"]
    rows = read_jsonl(Path(args.results))
    assert [r["prompt"] for r in rows] == [c["text"] for c in cases[: len(rows)]], "prompt order mismatch"
    tok, model = load_encoder("cpu")
    payload = torch.load(Path(args.head) / "jev_head.pt", weights_only=False)
    head = AttentionHead(model.config.hidden_size)
    head.load_state_dict(payload["state"])
    xs, ms = tflite_encoder(args.trunk)(tok, model, [r["prompt"] for r in rows])
    desk = score(head, xs, ms, payload["area_opts"], "cpu").numpy()
    dev = np.array([r["probabilities"] for r in rows])
    want = np.array([AREA_KEYS.index(area_of(c["function"])) for c in cases[: len(rows)]])
    agree = (dev.argmax(1) == desk.argmax(1)).mean()
    print(f"[device] {len(rows)} prompts: area acc device {(dev.argmax(1) == want).mean():.3f} / desktop "
          f"{(desk.argmax(1) == want).mean():.3f}; top-1 agree {agree:.3f}; max |Δp| {np.abs(dev - desk).max():.4f}; "
          f"trunk ms p50 {statistics.median(r['trunkMs'] for r in rows):.1f}, head ms p50 {statistics.median(r['headMs'] for r in rows):.1f}")
    return 0


def selfcheck() -> int:
    fns = {r["function"] for r in read_jsonl(FG / "data_v2_empty" / "train.jsonl")}
    assert fns <= set(FN_TO_AREA), fns - set(FN_TO_AREA)
    assert len(FN_TO_AREA) == 44 and area_of("MoveToDrone") == "none"
    head = AttentionHead(16, rank=4)
    out = head(torch.randn(2, 5, 16), torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool), torch.randn(3, 16))
    assert out.shape == (2, 3)
    assert route(np.array([0.6] + [0.4 / 14] * 14))[0] == "run"
    assert route(np.array([0.4, 0.15] + [0.45 / 13] * 13))[0] == "ask"
    assert route(np.array([0.05] * 14 + [0.3]))[0] == "refuse"
    print("ok — 44 functions mapped to", len(AREA_KEYS), "areas")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="data_v2_empty")
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--trunk", default="", help="take states from this exported trunk .tflite instead of HF")
    t.add_argument("--out", default=str(WORK / "head"))
    p = sub.add_parser("parity")
    p.add_argument("--tflite", required=True)
    p.add_argument("--n", type=int, default=20)
    d = sub.add_parser("device")
    d.add_argument("--results", required=True)
    d.add_argument("--cases", default="heldout3_cases.json")
    d.add_argument("--trunk", default=str(WORK / "trunk" / "gemma3_270m_l12_t48_w8.tflite"))
    d.add_argument("--head", default=str(WORK / "head-w8"))
    sub.add_parser("selfcheck")
    args = ap.parse_args()
    return {"train": cmd_train, "parity": cmd_parity, "device": cmd_device, "selfcheck": lambda _: selfcheck()}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
