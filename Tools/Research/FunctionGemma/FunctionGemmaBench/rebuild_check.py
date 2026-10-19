#!/usr/bin/env python3
"""The checks and glue of Rebuild-FunctionGemmaRouter.ps1 that PowerShell would do clumsily.

Each subcommand prints a readable summary; with --log it also writes the markdown fragment
the script joins into report.md.

  contract      contract vs contract_snapshot.json (written on first run), template coverage,
                stale eval cases (non-stale copy -> <log>/cases/). Exit 2 = hand edits needed.
  snapshot      rewrite contract_snapshot.json (only after a passed gate)
  subset        the first N records of a data dir (CPU smoke)
  score         aggregate fg_eval outputs; --gate enforces 0 wrong actions and a pass rate (exit 3)
  parity        HF vs Windows-CLI calls, case by case (exit 4 on any difference)
  export        the contract as the app's functiongemma_tools.json
  stage         point FunctionGemmaRouterModel at a new bundle (LiteRtLmSampleAssets.cs copies)
  device-prep   prompts (cases + their clauses) and system instruction for the device runner
  device-grade  the router (guard + clause split) over the device's raw outputs
  selftest      snapshot diff asserts + the shared router golden table (same table as the C# self-test)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import fg_common as fg
import make_dataset as md

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SNAPSHOT = HERE / "contract_snapshot.json"
GOLDEN = ROOT / "Packages" / "com.leuconoe.litert-lm-unity" / "Samples~" / "AutomatedTests" / "Editor" / "functiongemma_router_golden.json"
SPLITS = ("train", "val")


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def emit(log, name: str, lines: list[str]) -> None:
    print("\n".join(lines))
    if log:
        Path(log, name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def snapshot_of(contract: Path) -> dict:
    """The structure the weights depend on: tools, required, types, enums, bounds, aliases."""
    snap = {"sha256": hashlib.sha256(Path(contract).read_bytes()).hexdigest(), "tools": {}}
    for t in fg.load_tools(contract):
        p = t["function"].get("parameters", {})
        snap["tools"][t["function"]["name"]] = {
            "required": sorted(p.get("required", [])),
            "params": {k: {kk: (sorted(vv) if kk == "enum" else vv) for kk, vv in v.items() if kk != "description"}
                       for k, v in p.get("properties", {}).items()}}
    return snap


def diff(old: dict, new: dict) -> list[str]:
    o, n = old["tools"], new["tools"]
    added, removed, out = sorted(set(n) - set(o)), sorted(set(o) - set(n)), []
    for r in list(removed):  # same signature under a new name reads as a rename
        twin = next((a for a in added if n[a] == o[r]), None)
        if twin:
            out.append(f"renamed tool {r} → {twin}")
            added.remove(twin)
            removed.remove(r)
    out += [f"added tool {a} (params {sorted(n[a]['params'])})" for a in added] + [f"removed tool {r}" for r in removed]
    for t in sorted(set(o) & set(n)):
        op, np_ = o[t]["params"], n[t]["params"]
        out += [f"{t}: added param {k}" for k in sorted(set(np_) - set(op))]
        out += [f"{t}: removed param {k}" for k in sorted(set(op) - set(np_))]
        for k in sorted(set(op) & set(np_)):
            for field in sorted(set(op[k]) | set(np_[k])):
                if op[k].get(field) != np_[k].get(field):
                    out.append(f"{t}.{k}: {field} {op[k].get(field)} → {np_[k].get(field)}")
        if o[t]["required"] != n[t]["required"]:
            out.append(f"{t}: required {o[t]['required']} → {n[t]['required']}")
    return out


def cmd_contract(a) -> int:
    contract, snapshot = Path(a.contract), Path(a.snapshot)
    snap, tools = snapshot_of(contract), fg.load_tools(contract)
    index = fg.index_of(tools)
    lines = ["## 1. Contract", "", f"`{contract.name}` (sha256 {snap['sha256'][:12]}), {len(snap['tools'])} tools; "
             f"templates `{Path(a.templates).name}`.", ""]
    old = json.loads(snapshot.read_text(encoding="utf-8")) if snapshot.exists() else None
    todo = md.coverage(tools, json.loads(Path(a.templates).read_text(encoding="utf-8")))
    if old is None:
        lines.append(f"no `{snapshot.name}` yet — it is written after the first passed gate.")
    else:
        changes = diff(old, snap)
        if changes:
            lines += [f"**{len(changes)} contract change(s)** vs `{snapshot.name}` — the decl-none model must be retrained:"]
            lines += [f"- {c}" for c in changes]
        elif old["sha256"] != snap["sha256"]:
            lines.append("**no structural change** — descriptions only; the decl-none model never sees them.")
        else:
            lines.append(f"**no contract change** (identical to `{snapshot.name}`).")

    stale, out_dir = [], Path(a.log or ".") / "cases"
    doc = json.loads(Path(a.cases).read_text(encoding="utf-8"))
    keep = []
    for c in doc["cases"]:
        call = {"function": c["function"], "arguments": c["arguments"]}
        if c["function"] != fg.REFUSAL_NAME and fg.validate(call, index, c["text"]) is fg.REFUSAL:
            stale.append({"id": c.get("id"), "text": c["text"], **call})
        else:
            keep.append(c)
    if a.log:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / Path(a.cases).name).write_text(json.dumps({**doc, "cases": keep}, ensure_ascii=False, indent=1), encoding="utf-8")

    if todo:
        lines += ["", f"**STOP — {len(todo)} hand edit(s) before data can be generated:**"] + [f"- [ ] {t}" for t in todo]
    if stale:
        lines += ["", f"**{len(stale)} stale eval case(s)** — the expected call no longer validates against the contract "
                  "(or is not grounded in its text); left out of every score until relabelled:"]
        lines += [f"- [ ] [{s['id']}] {s['text']}  →  {s['function']}{json.dumps(s['arguments'], ensure_ascii=False)}" for s in stale]
    else:
        lines += ["", "stale eval cases: none."]
    emit(a.log, "01-contract.md", lines)
    return 2 if todo else 0


def cmd_snapshot(a) -> int:
    Path(a.snapshot).write_text(json.dumps(snapshot_of(Path(a.contract)), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {a.snapshot}")
    return 0


def cmd_subset(a) -> int:
    n_val = max(10, a.n // 20)
    for s, k in (("train", a.n - n_val), ("val", n_val)):
        write_jsonl(Path(a.out) / f"{s}.jsonl", read_jsonl(Path(a.src) / f"{s}.jsonl")[:k])
    print(f"subset: train {a.n - n_val}, val {n_val} -> {a.out}")
    return 0


def eval_rows(path) -> list[dict]:
    return [r for r in read_jsonl(path) if not r.get("summary")]


def outcome(r) -> tuple[bool, bool]:
    """(passed, wrong action) after the guard, and after the clause split when fg_eval ran with --segment."""
    return (r["segmented_passed"], r["segmented_wrong"]) if "segmented_passed" in r else (r["guarded_passed"], r["wrong_action"])


def cmd_score(a) -> int:
    lines = [f"## {a.title}", "", "| set | pass | wrong actions |", "| --- | ---: | ---: |"]
    verdict, gate_line, misses = (3, f"gate ({a.gate}): **not run**", []) if a.gate else (0, "", [])
    for spec in a.files:
        name, path = spec.split("=", 1)
        if not Path(path).exists():
            lines.append(f"| {name} | not run | |")
            continue
        rows = eval_rows(path)
        p, w = (sum(outcome(r)[i] for r in rows) for i in (0, 1))
        lines.append(f"| {name}{' (gate)' if name == a.gate else ''} | {p}/{len(rows)} ({p / max(1, len(rows)):.1%}) | {w} |")
        if name == a.gate:
            ok = w == 0 and p >= a.pass_rate * len(rows)
            verdict = 0 if ok else 3
            gate_line = f"gate ({name}: 0 wrong actions, ≥ {a.pass_rate:.0%} pass): **{'PASS' if ok else 'FAIL'}**"
            misses = [r for r in rows if not outcome(r)[0]]
    if a.gate:
        lines += ["", gate_line] + [f"- [{r.get('id')}] {r['text']}  →  {json.dumps(r.get('segmented') or r.get('got'), ensure_ascii=False)}"
                                    f"  (want {r['function']}{json.dumps(r['arguments'], ensure_ascii=False)})" for r in misses[:40]]
    emit(a.log, a.fragment, lines)
    return verdict


def cmd_parity(a) -> int:
    hf = {(r.get("id"), r["text"]): r for r in eval_rows(a.hf)}
    cli = eval_rows(a.cli)
    diffs = [(r, hf[(r.get("id"), r["text"])]) for r in cli if (r.get("id"), r["text"]) in hf and r["got"] != hf[(r.get("id"), r["text"])]["got"]]
    tokens = sum(r.get("prefill_tokens") != r["hf_tokens"] for r in cli)
    errors = sum("error" in r for r in cli)
    bundle = Path(a.bundle)
    lines = ["## 5. Convert + parity", "", f"bundle `{bundle}` ({bundle.stat().st_size / 2**20:.1f} MB), cache 256 / prefill 64,128, int8.",
             f"Windows CLI vs HF: **{len(cli) - len(diffs)}/{len(cli)} identical calls**, CLI raw pass "
             f"{sum(r['passed'] for r in cli)}/{len(cli)}, prompt-token mismatches {tokens}, CLI errors {errors}."]
    lines += [f"- [{r.get('id')}] {r['text']}: CLI {json.dumps(r['got'], ensure_ascii=False)} vs HF {json.dumps(h['got'], ensure_ascii=False)}"
              for r, h in diffs]
    emit(a.log, "05-parity.md", lines)
    return 4 if diffs or tokens or errors else 0


def cmd_export(a) -> int:
    body = json.dumps(fg.load_tools(Path(a.contract)), ensure_ascii=False, separators=(",", ":")) + "\n"
    lines = ["## 6. App export", ""]
    for d in (Path(p).resolve() for p in a.dest):
        if not d.parent.exists():
            lines.append(f"- `{d}`: skipped (no such directory)")
            continue
        old = d.read_text(encoding="utf-8") if d.exists() else None
        state = "unchanged" if old == body else "would change (dry)" if a.dry else "written"
        if old != body and not a.dry:
            d.write_text(body, encoding="utf-8", newline="\n")
        lines.append(f"- `{d.relative_to(ROOT).as_posix()}`: {state}")
    emit(a.log, "06-export.md", lines)
    return 0


def cmd_stage(a) -> int:
    pat = rb'(FunctionGemmaRouterModel = ")[^"]*(")'
    lines = []
    for f in (Path(p).resolve() for p in a.file):
        if not f.exists():
            continue  # the Assets/Samples copy exists only once the sample is imported
        src = f.read_bytes()
        new = re.sub(pat, rb"\g<1>" + a.path.encode() + rb"\g<2>", src)
        if new == src and a.path.encode() not in src:
            raise SystemExit(f"{f}: FunctionGemmaRouterModel not found")
        f.write_bytes(new)
        lines.append(f"- `{f.relative_to(ROOT).as_posix()}`: FunctionGemmaRouterModel = \"{a.path}\"")
    emit(a.log, "06-stage.md", lines)
    return 0


def gate_cases(path, limit) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["cases"][: limit or None]


def cmd_device_prep(a) -> int:
    cases, out = gate_cases(a.cases, a.limit), Path(a.out)
    prompts = list(dict.fromkeys(t for c in cases for t in [c["text"], *fg.segments(c["text"])]))
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompts.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8", newline="\n")
    (out / "system.txt").write_text(fg.TRIGGER, encoding="utf-8", newline="\n")  # decl none: the trigger alone
    print(f"device prompts: {len(prompts)} ({len(cases)} cases + their clauses) -> {out}")
    return 0


def cmd_device_grade(a) -> int:
    index = fg.index_of(fg.load_tools(Path(a.contract)))
    res = {r["prompt"]: r for r in read_jsonl(a.results)}
    passed = wrong = missing = 0
    ms = []
    for c in gate_cases(a.cases, a.limit):
        r = res.get(c["text"])
        if r is None or r.get("error"):
            missing += 1
            continue
        full = fg.validate(fg.parse_call(r["response"]), index, c["text"])
        parts = [p for p in fg.segments(c["text"]) if p in res] if full["function"] == fg.REFUSAL_NAME else []
        final = fg.combine_segments(full, [fg.validate(fg.parse_call(res[p]["response"]), index, p) for p in parts]) if parts else full
        ok = fg.grade(c, final, index)[0]
        passed += ok
        wrong += not ok and final["function"] != fg.REFUSAL_NAME
        ms.append(r["wallMs"] + sum(res[p]["wallMs"] for p in parts))
    ms.sort()
    n = len(ms)
    lat = f"mean {sum(ms) / n:.0f}, p50 {ms[n // 2]:.0f}, p95 {ms[int(0.95 * (n - 1))]:.0f}" if n else "-"
    emit(a.log, "07-device.md", ["## 7. Device (CPU, router applied host-side)", "",
                                 f"pass **{passed}/{n}**, wrong actions **{wrong}**, missing/errored {missing}; "
                                 f"ms per command incl. clause retries: {lat}."])
    return 0


def run_golden(path: Path, contract: Path) -> list[str]:
    """The shared golden table through fg_common.route + needs_confirmation; returns the failures."""
    index, failures = fg.index_of(fg.load_tools(contract)), []
    for g in json.loads(path.read_text(encoding="utf-8"))["routes"]:
        got = fg.route(g["utterance"], lambda t: g["raw"][t], index)
        confirm = fg.needs_confirmation(got, index, g.get("asr_confidence"), g.get("min_asr_confidence", 0.0))
        if got != g["expected"] or confirm != g.get("confirm", False):
            failures.append(f"{g['utterance']!r}: want {g['expected']} confirm={g.get('confirm', False)}, got {got} confirm={confirm}")
    return failures


def cmd_selftest(a) -> int:
    snap = {"tools": {"A": {"required": ["n"], "params": {"n": {"type": "number"}, "room": {"type": "string", "enum": ["Kitchen"]}}}}}
    moved = {"tools": {"B": snap["tools"]["A"]}}
    grown = json.loads(json.dumps(snap))
    grown["tools"]["A"]["params"]["room"]["enum"].append("Garage")
    assert diff(snap, snap) == [] and diff(snap, moved) == ["renamed tool A → B"], diff(snap, moved)
    assert diff(snap, grown) == ["A.room: enum ['Kitchen'] → ['Kitchen', 'Garage']"], diff(snap, grown)
    failures = run_golden(GOLDEN, Path(a.contract))
    assert not failures, "\n".join(failures)
    print(f"rebuild_check selftest ok — {len(json.loads(GOLDEN.read_text(encoding='utf-8'))['routes'])} golden routes")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--contract", default=str(fg.DEFAULT_CONTRACT))
    common.add_argument("--log", default="")
    sub.add_parser("selftest", parents=[common])
    s = sub.add_parser("contract", parents=[common])
    s.add_argument("--templates", default=str(HERE / "sample_templates.json"))
    s.add_argument("--cases", default=str(HERE / "sample_eval_cases.json"))
    s.add_argument("--snapshot", default=str(SNAPSHOT))
    s = sub.add_parser("snapshot", parents=[common])
    s.add_argument("--snapshot", default=str(SNAPSHOT))
    s = sub.add_parser("subset")
    s.add_argument("--src", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=200)
    s = sub.add_parser("score", parents=[common])
    s.add_argument("files", nargs="+", help="name=fg_eval.jsonl")
    s.add_argument("--gate", default="")
    s.add_argument("--pass-rate", type=float, default=0.95)
    s.add_argument("--title", default="4. Gate (HF bf16, batch 1, guard + clause split)")
    s.add_argument("--fragment", default="04-gate.md")
    s = sub.add_parser("parity", parents=[common])
    s.add_argument("--hf", required=True)
    s.add_argument("--cli", required=True)
    s.add_argument("--bundle", required=True)
    s = sub.add_parser("export", parents=[common])
    s.add_argument("--dest", action="append", required=True)
    s.add_argument("--dry", action="store_true")
    s = sub.add_parser("stage", parents=[common])
    s.add_argument("--path", required=True, help="StreamingAssets-relative bundle path")
    s.add_argument("--file", action="append", required=True)
    for name in ("device-prep", "device-grade"):
        s = sub.add_parser(name, parents=[common])
        s.add_argument("--cases", required=True)
        s.add_argument("--limit", type=int, default=0)
        s.add_argument("--out" if name == "device-prep" else "--results", required=True)
    a = ap.parse_args()
    return globals()["cmd_" + a.cmd.replace("-", "_")](a)


if __name__ == "__main__":
    sys.exit(main())
