"""Score a FunctionGemma checkpoint (stock or fine-tuned) on voice-command eval cases.

Greedy decode through transformers. Score at --batch-size 1 (the default): batched,
left-padded generation garbles some Gemma3 outputs and under-reports the model. The pass
rate carries over to a converted .litertlm of the same weights; the latency does not.
`generate`/`score` are reused by fg_train.py for picking the best epoch.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import fg_common as fg

HERE = Path(__file__).resolve().parent


def stop_ids(tok) -> list[int]:
    """<eos>, <start_function_response> (FunctionGemma: the call is complete),
    <end_of_turn>. Looked up, not hard-coded — gemma-3 has no function tokens
    and maps the name to <unk>, which is dropped."""
    ids = [tok.eos_token_id, tok.convert_tokens_to_ids("<start_function_response>"), tok.convert_tokens_to_ids("<end_of_turn>")]
    return [i for i in dict.fromkeys(ids) if i is not None and i != tok.unk_token_id]


@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new_tokens: int = 96, batch_size: int = 16) -> list[str]:
    tok.padding_side = "left"
    stops, outs = stop_ids(tok), []
    for i in range(0, len(prompts), batch_size):
        # The rendered prompt already starts with <bos>.
        enc = tok(prompts[i:i + batch_size], add_special_tokens=False, padding=True, return_tensors="pt").to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             eos_token_id=stops, pad_token_id=tok.pad_token_id)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            outs.append(tok.decode(row, skip_special_tokens=False).replace(tok.pad_token, ""))
    return outs


def score(records: list[dict], raws: list[str], parse, tool_index) -> list[dict]:
    rows = []
    for rec, raw in zip(records, raws):
        got = parse(raw)
        passed, reason = fg.grade(rec, got, tool_index)
        guarded = fg.validate(got, tool_index, rec["text"])
        rows.append({**rec, "passed": passed, "reason": reason, "got": got, "raw": raw,
                     "guarded_passed": fg.grade(rec, guarded, tool_index)[0],
                     "wrong_action": not fg.grade(rec, guarded, tool_index)[0] and guarded["function"] != "DefaultResponse"})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=fg.DEFAULT_MODEL, help="HF id or a merged fine-tune dir")
    ap.add_argument("--cases", default=str(HERE / "sample_eval_cases.json"))
    ap.add_argument("--decl-mode", choices=fg.DECL_MODES, default="none")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--contract", default=str(fg.DEFAULT_CONTRACT))
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=1, help=">1 is faster but garbles some outputs (left padding)")
    ap.add_argument("--limit", type=int, default=0, help="first N cases only (smoke runs)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--segment", action="store_true", help="also score the clause-split fallback for refused utterances")
    args = ap.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    if args.limit:
        cases = cases[: args.limit]
    model_id, contract = args.model, Path(args.contract)
    render, parse, tools = fg.make_format(args.decl_mode, contract)
    tool_index = {t["function"]["name"]: t for t in tools}

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(args.device).eval()

    sample = cases[0]["text"]
    token_counts = {m: len(tok(fg.make_format(m, contract)[0](tok, sample), add_special_tokens=False)["input_ids"])
                    for m in fg.DECL_MODES}
    print(f"prompt tokens for {sample!r}: " + ", ".join(f"{m}={n}" for m, n in token_counts.items()))

    t0 = time.perf_counter()
    raws = generate(model, tok, [render(tok, c["text"]) for c in cases], args.max_new_tokens, args.batch_size)
    per_case_ms = (time.perf_counter() - t0) * 1000 / len(cases)
    rows = score(cases, raws, parse, tool_index)
    if args.segment:
        for r in rows:
            full = fg.validate(r["got"], tool_index, r["text"])
            parts = fg.segments(r["text"]) if full["function"] == "DefaultResponse" else []
            outs = generate(model, tok, [render(tok, p) for p in parts], args.max_new_tokens, 1) if parts else []
            seg_calls = [fg.validate(parse(o), tool_index, p) for o, p in zip(outs, parts)]
            final = fg.combine_segments(full, seg_calls) if parts else full
            r["segmented"] = final
            r["segmented_passed"] = fg.grade(r, final, tool_index)[0]
            r["segmented_wrong"] = not r["segmented_passed"] and final["function"] != "DefaultResponse"

    by_suite = defaultdict(lambda: [0, 0])
    for r in rows:
        by_suite[r.get("suite", "-")][0] += r["passed"]
        by_suite[r.get("suite", "-")][1] += 1
    total = sum(r["passed"] for r in rows)
    fn_ok = sum(1 for r in rows if r["got"] and r["got"]["function"] == r["function"])
    reasons = Counter(r["reason"] for r in rows if not r["passed"])

    label = args.label or f"{Path(model_id).name}-{args.decl_mode}"
    print(f"\n{label}  decl={args.decl_mode}  cases={Path(args.cases).name}")
    print("| suite | pass |\n| --- | --- |")
    for suite, (p, n) in by_suite.items():
        print(f"| {suite} | {p}/{n} |")
    print(f"| **total** | **{total}/{len(rows)}** |")
    print(f"function-name correct {fn_ok}/{len(rows)}; failures by reason: {dict(reasons)}")
    print(f"with app schema guard: {sum(r['guarded_passed'] for r in rows)}/{len(rows)} pass, "
          f"{sum(r['wrong_action'] for r in rows)} wrong actions (the rest of the misses are refusals)")
    if args.segment:
        seg_suite = defaultdict(lambda: [0, 0])
        for r in rows:
            seg_suite[r.get("suite", "-")][0] += r["segmented_passed"]
            seg_suite[r.get("suite", "-")][1] += 1
        print(f"guard + clause split: {sum(r['segmented_passed'] for r in rows)}/{len(rows)} pass, "
              f"{sum(r['segmented_wrong'] for r in rows)} wrong actions | " + ", ".join(f"{s} {p}/{n}" for s, (p, n) in seg_suite.items()))
    print(f"{per_case_ms:.0f} ms/case at batch {args.batch_size} (desktop, not the device)")
    print("\nfailures:")
    for r in rows:
        if not r["passed"]:
            exp = f"{r['function']}{json.dumps(r['arguments'], ensure_ascii=False)}"
            got = f"{r['got']['function']}{json.dumps(r['got']['arguments'], ensure_ascii=False)}" if r["got"] else repr(r["raw"][:80])
            print(f"  [{r.get('id', '-')}|{r['reason']}] {r['text']}  →  {got}   (want {exp})")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=list) + "\n")
            f.write(json.dumps({"summary": True, "label": label, "model": model_id,
                                "decl_mode": args.decl_mode, "cases": Path(args.cases).name, "passed": total,
                                "total": len(rows), "function_correct": fn_ok, "by_suite": dict(by_suite),
                                "reasons": dict(reasons), "prompt_tokens": token_counts,
                                "ms_per_case": round(per_case_ms, 1), "batch_size": args.batch_size},
                               ensure_ascii=False) + "\n")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
