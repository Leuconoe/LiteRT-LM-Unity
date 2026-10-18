#!/usr/bin/env python3
"""The checks and glue of Rebuild-FunctionGemmaRouter.ps1 that PowerShell would do clumsily.

Each subcommand prints a readable summary; with --log it also writes the markdown fragment
the script joins into report.md.

  contract      ToolsPrompt vs contract_snapshot.json, make_dataset / router mirror coverage,
                stale eval cases (non-stale copies -> <log>/cases/). Exit 2 = hand edits needed.
  snapshot      rewrite contract_snapshot.json (only after a passed gate)
  carry         new v4 templates + the old paraphrases that still validate -> v5 composition
  subset        the first N records of a data dir (CPU smoke)
  score         aggregate fg_eval outputs; --gate enforces 0 wrong actions and a pass rate (exit 3)
  parity        HF vs Windows-CLI calls, case by case
  export        the app's functiongemma_tools.json copies
  stage         point FunctionGemmaRouterModel at a new bundle (both LiteRtLmSampleAssets.cs copies)
  device-prep   prompts (cases + their clauses) and system instruction for the device runner
  device-grade  the router (guard + clause split) over the device's raw outputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import fg_common as fg

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SNAPSHOT = HERE / "contract_snapshot.json"
ROUTER_CS = ROOT / "Packages" / "com.leuconoe.litert-lm-unity" / "Runtime" / "LiteRtLmFunctionGemmaRouter.cs"
SPLITS = ("train", "val")
# Where each enum param's values live in make_dataset.py.
MD_VOCAB = {"team": "TEAMS (+ Korean reading)", "teamType": "TEAM_TYPES (+ spoken aliases)",
            "windowName": "WINDOWS (+ spoken forms)", "confirmName": "CONFIRMS"}


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


def case_files() -> list[Path]:
    return sorted([*HERE.glob("*_cases.json"), *HERE.glob("locked/*_cases.json")])


def load_contract(tools_prompt, ws_root) -> tuple[dict, dict]:
    """(structural snapshot, tool index). The filter names exist only in the prose, so the
    router check validates filterNames against them while this process runs."""
    tools = fg.load_tools(Path(tools_prompt), Path(ws_root))
    text = Path(tools_prompt).read_text(encoding="utf-8")
    m = re.search(r"^\s*//\s*필터 이름:\s*(.+?)\.?\s*$", text, re.M)
    snap = {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "filter_names": [f.strip() for f in m.group(1).split(",")] if m else [], "tools": {}}
    for t in tools:
        p = t["function"].get("parameters", {})
        snap["tools"][t["function"]["name"]] = {
            "required": sorted(p.get("required", [])),
            "params": {k: {"type": v.get("type"), **({"enum": sorted(v["enum"])} if "enum" in v else {})}
                       for k, v in p.get("properties", {}).items()}}
    return snap, {t["function"]["name"]: t for t in tools}


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
    enums = {}  # one line per enum change, however many tools share the param
    for t in sorted(set(o) & set(n)):
        op, np_ = o[t]["params"], n[t]["params"]
        out += [f"{t}: added param {k}" for k in sorted(set(np_) - set(op))]
        out += [f"{t}: removed param {k}" for k in sorted(set(op) - set(np_))]
        for k in sorted(set(op) & set(np_)):
            if op[k]["type"] != np_[k]["type"]:
                out.append(f"{t}.{k}: type {op[k]['type']} → {np_[k]['type']}")
            oe, ne = set(op[k].get("enum", [])), set(np_[k].get("enum", []))
            for sign, vals in (("+", ne - oe), ("−", oe - ne)):
                if vals:
                    enums.setdefault((k, sign, tuple(sorted(vals))), []).append(t)
        if o[t]["required"] != n[t]["required"]:
            out.append(f"{t}: required {o[t]['required']} → {n[t]['required']}")
    out += [f"{k}: enum {sign} {list(vals)} in {', '.join(ts) if len(ts) <= 3 else f'{len(ts)} tools'}"
            for (k, sign, vals), ts in enums.items()]
    of, nf = set(old["filter_names"]), set(new["filter_names"])
    out += [f"filter names + {sorted(nf - of)}"] if nf - of else []
    out += [f"filter names − {sorted(of - nf)}"] if of - nf else []
    return out


def router_mirrors() -> tuple[set, set]:
    src = ROUTER_CS.read_text(encoding="utf-8")
    block = lambda name: re.search(name + r"\s*=\s*new[^{]*\{(.*?)\};", src, re.S).group(1)  # noqa: E731
    return set(re.findall(r'\["([^"]+)"\]', block("CallsignKo"))), set(re.findall(r'"([^"]+)"', block("FilterNames")))


def coverage(snap: dict, renames: dict) -> list[str]:
    """(a) make_dataset.py can generate the contract, (b) the router mirrors know its callsigns and filters."""
    import make_dataset as md
    tools, todo = snap["tools"], []
    new_only, md_only = set(tools) - set(md.SCHEMA), set(md.SCHEMA) - set(tools)
    for old, new in renames.items():
        if old in md_only and new in new_only:
            todo.append(f"(a) make_dataset.py: rename {old} → {new} in TOOLS / SCHEMA (and INDIRECT / V3_EXTRA / V4_EXTRA where it appears)")
            md_only.discard(old)
            new_only.discard(new)
    for t in sorted(new_only):
        todo.append(f"(a) make_dataset.py: new tool {t} has no generator — add a TOOLS entry (phrasings, target count) "
                    f"and a SCHEMA line: required {tools[t]['required']}, params {tools[t]['params']}")
    for t in sorted(md_only):
        todo.append(f"(a) make_dataset.py: {t} is not in the contract any more — rename or remove it in TOOLS / SCHEMA "
                    "(and INDIRECT / V3_EXTRA / V4_EXTRA where it appears); selfcheck() asserts on it")
    missing, extra = {}, {}
    for t in sorted(set(tools) & set(md.SCHEMA)):
        req, opt = md.SCHEMA[t]
        if set(req) != set(tools[t]["required"]) or set(req) | set(opt) != set(tools[t]["params"]):
            todo.append(f"(a) make_dataset.py SCHEMA[{t!r}]: required {sorted(req)} + optional {sorted(opt)}, "
                        f"contract: required {tools[t]['required']}, params {sorted(tools[t]['params'])}")
        for k, spec in tools[t]["params"].items():
            mine = req.get(k) or opt.get(k)
            if "enum" in spec and isinstance(mine, (set, frozenset)):
                missing.setdefault(k, set()).update(set(spec["enum"]) - mine)
                extra.setdefault(k, set()).update(mine - set(spec["enum"]))
    for k in sorted(missing):
        where = MD_VOCAB.get(k, f"the SCHEMA set of {k}")
        todo += [f"(a) make_dataset.py {where}: add {k} values {sorted(missing[k])}"] if missing[k] else []
        todo += [f"(a) make_dataset.py {where}: remove {k} values {sorted(extra[k])}"] if extra[k] else []
    prose = set(snap["filter_names"])
    if not prose:
        todo.append("(b) no '// 필터 이름: …' line in the ToolsPrompt prose — filter names cannot be checked; restore the line")
    elif set(md.FILTERS) != prose:
        todo.append(f"(a) make_dataset.py FILTERS: add {sorted(prose - set(md.FILTERS))}, remove {sorted(set(md.FILTERS) - prose)}")

    teams = set().union(*(set(p.get("enum", [])) for t in tools.values() for k, p in t["params"].items() if k == "team"))
    cs_callsigns, cs_filters = router_mirrors()
    rel_cs = ROUTER_CS.relative_to(ROOT).as_posix()
    for where, have in (("FunctionGemmaBench/fg_common.py CALLSIGN_KO", set(fg.CALLSIGN_KO)), (f"{rel_cs} CallsignKo", cs_callsigns)):
        if teams - have:
            todo.append(f"(b) {where}: add {sorted(teams - have)} with the Korean reading (grounding refuses these teams until then)")
    if prose:
        for where, have in (("FunctionGemmaBench/fg_common.py FILTER_NAMES", set(fg.FILTER_NAMES)), (f"{rel_cs} FilterNames", cs_filters)):
            if have != prose:
                todo.append(f"(b) {where}: add {sorted(prose - have)}, remove {sorted(have - prose)}")
    return todo


def cmd_contract(a) -> int:
    snap, index = load_contract(a.tools_prompt, a.webserver_root)
    snapshot = Path(a.snapshot)
    lines = ["## 1. Contract", "", f"ToolsPrompt `{a.tools_prompt}` (sha256 {snap['sha256'][:12]}), "
             f"{len(snap['tools'])} tools, {len(snap['filter_names'])} filter names.", ""]
    old = json.loads(snapshot.read_text(encoding="utf-8")) if snapshot.exists() else None
    changes = diff(old, snap) if old else []
    todo = coverage(snap, dict(re.findall(r"^renamed tool (\w+) → (\w+)$", "\n".join(changes), re.M)))
    if old is None and todo:
        lines.append(f"no `{snapshot.name}` yet, and this contract is not a valid baseline (see the to-do list).")
    elif old is not None:
        if changes:
            lines += [f"**{len(changes)} contract change(s)** vs `{snapshot.name}`:"] + [f"- {c}" for c in changes]
        elif old["sha256"] != snap["sha256"]:
            lines.append("**no contract change** — the prose differs only (descriptions, examples); the decl-none model "
                         "never sees prose, retrain only if a phrasing rule changed.")
        else:
            lines.append(f"**no contract change** (identical to `{snapshot.name}`).")
    else:
        snapshot.write_text(json.dumps(snap, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        lines.append(f"first run: wrote `{snapshot.name}` as the baseline.")

    if snap["filter_names"]:  # the stale check reads filterNames against the new prose, not the old mirror
        fg.FILTER_NAMES = frozenset(snap["filter_names"])
    stale, out_dir = [], Path(a.log or ".") / "cases"
    for f in case_files():
        doc = json.loads(f.read_text(encoding="utf-8"))
        keep = []
        for c in doc["cases"]:
            call = {"function": c["function"], "arguments": c["arguments"]}
            if c["function"] != "DefaultResponse" and fg.validate(call, index) is fg.REFUSAL:
                stale.append({"file": f.relative_to(HERE).as_posix(), "id": c.get("id"), "text": c["text"], **call})
            else:
                keep.append(c)
        if a.log:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f.name).write_text(json.dumps({**doc, "cases": keep}, ensure_ascii=False, indent=1), encoding="utf-8")

    if todo:
        lines += ["", f"**STOP — {len(todo)} hand edit(s) before the data can be rebuilt:**"] + [f"- [ ] {t}" for t in todo]
    if stale:
        lines += ["", f"**{len(stale)} stale eval case(s)** — the expected call no longer validates; left out of every score "
                  "until a human relabels them:"]
        lines += [f"- [ ] {s['file']} [{s['id']}] {s['text']}  →  {s['function']}{json.dumps(s['arguments'], ensure_ascii=False)}"
                  for s in stale]
    elif a.log:
        lines += ["", "stale eval cases: none."]
    if a.log:
        Path(a.log, "contract.json").write_text(json.dumps({"changes": changes, "todo": todo, "stale": stale, "snapshot": snap},
                                                           ensure_ascii=False, indent=1), encoding="utf-8")
    emit(a.log, "01-contract.md", lines)
    return 2 if todo else 0


def cmd_snapshot(a) -> int:
    snap, _ = load_contract(a.tools_prompt, a.webserver_root)
    Path(a.snapshot).write_text(json.dumps(snap, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {a.snapshot}")
    return 0


def cmd_carry(a) -> int:
    snap, index = load_contract(a.tools_prompt, a.webserver_root)
    if snap["filter_names"]:
        fg.FILTER_NAMES = frozenset(snap["filter_names"])
    template = {r["text"] for s in SPLITS for r in read_jsonl(Path(a.old_v4) / f"{s}.jsonl")}
    kept, dropped, n = {s: [] for s in SPLITS}, Counter(), 0
    for s in SPLITS:
        for r in read_jsonl(Path(a.old_v5) / f"{s}.jsonl"):
            if r["text"] in template:
                continue
            n += 1
            if (r["function"] == "DefaultResponse" and "DefaultResponse" in index) or fg.validate(r, index, r["text"]) is r:
                kept[s].append(r)
            else:
                dropped[r["function"]] += 1
    for s in SPLITS:
        write_jsonl(Path(a.out) / f"{s}.jsonl", read_jsonl(Path(a.new_v4) / f"{s}.jsonl") + kept[s])
    k = sum(len(v) for v in kept.values())
    print(f"carry: paraphrases {n}, still valid {k}, dropped {sum(dropped.values())}"
          + (f" ({dict(dropped.most_common())})" if dropped else "") + f" -> {a.out}")
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
            gate_line = (f"gate ({name}: 0 wrong actions, ≥ {a.pass_rate:.0%} pass): **{'PASS' if ok else 'FAIL'}**")
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
             f"Windows CLI vs HF on eval_cases: **{len(cli) - len(diffs)}/{len(cli)} identical calls**, CLI raw pass "
             f"{sum(r['passed'] for r in cli)}/{len(cli)}, prompt-token mismatches {tokens}, CLI errors {errors}."]
    lines += [f"- [{r.get('id')}] {r['text']}: CLI {json.dumps(r['got'], ensure_ascii=False)} vs HF {json.dumps(h['got'], ensure_ascii=False)}"
              for r, h in diffs]
    emit(a.log, "05-parity.md", lines)
    return 4 if diffs or tokens or errors else 0


def cmd_export(a) -> int:
    body = json.dumps(fg.load_tools(Path(a.tools_prompt), Path(a.webserver_root)), ensure_ascii=False, separators=(",", ":")) + "\n"
    lines = ["## 6. App export", ""]
    for d in (Path(p).resolve() for p in a.dest):
        old = d.read_text(encoding="utf-8") if d.exists() else None
        state = "unchanged" if old == body else "would change (dry)" if a.dry else "written"
        if old != body and not a.dry:
            d.write_text(body, encoding="utf-8", newline="\n")
        lines.append(f"- `{d.relative_to(ROOT).as_posix()}`: {state}")
    emit(a.log, "06-export.md", lines)
    return 0


def cmd_stage(a) -> int:
    pat = rb'(FunctionGemmaRouterModel = ")[^"]*(")'
    files = [Path(f).resolve() for f in a.file]
    before = [f.read_bytes() for f in files]
    lines = []
    if len(set(before)) > 1:  # diff first: the copies differ outside our line, so edit each in place
        lines.append(f"note: the {len(files)} copies of {files[0].name} differ; only the FunctionGemmaRouterModel line is replaced in each")
    for f, src in zip(files, before):
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
    _, index = load_contract(a.tools_prompt, a.webserver_root)
    res = {r["prompt"]: r for r in read_jsonl(a.results)}
    passed = wrong = missing = 0
    ms = []
    for c in gate_cases(a.cases, a.limit):
        r = res.get(c["text"])
        if r is None or r.get("error"):
            missing += 1
            continue
        full = fg.validate(fg.parse_call(r["response"]), index, c["text"])
        parts = [p for p in fg.segments(c["text"]) if p in res] if full["function"] == "DefaultResponse" else []
        final = fg.combine_segments(full, [fg.validate(fg.parse_call(res[p]["response"]), index, p) for p in parts]) if parts else full
        ok = fg.grade(c, final, index)[0]
        passed += ok
        wrong += not ok and final["function"] != "DefaultResponse"
        ms.append(r["wallMs"] + sum(res[p]["wallMs"] for p in parts))
    ms.sort()
    n = len(ms)
    lat = f"mean {sum(ms) / n:.0f}, p50 {ms[n // 2]:.0f}, p95 {ms[int(0.95 * (n - 1))]:.0f}" if n else "-"
    emit(a.log, "07-device.md", ["## 7. Device (46a880a0, CPU, router applied host-side)", "",
                                 f"pass **{passed}/{n}**, wrong actions **{wrong}**, missing/errored {missing}; "
                                 f"ms per command incl. clause retries: {lat}."])
    return 0


def cmd_selftest(a) -> int:
    snap = {"tools": {"A": {"required": ["n"], "params": {"n": {"type": "number"}, "team": {"type": "string", "enum": ["Alpha"]}}}},
            "filter_names": ["위협"]}
    moved = {"tools": {"B": snap["tools"]["A"]}, "filter_names": ["위협"]}
    grown = json.loads(json.dumps(snap))
    grown["tools"]["A"]["params"]["team"]["enum"].append("Kimchi")
    assert diff(snap, snap) == [] and diff(snap, moved) == ["renamed tool A → B"], diff(snap, moved)
    assert diff(snap, grown) == ["team: enum + ['Kimchi'] in A"], diff(snap, grown)
    callsigns, filters = router_mirrors()  # the C# mirror must parse, and agree with fg_common
    assert callsigns == set(fg.CALLSIGN_KO) and filters == set(fg.FILTER_NAMES), (callsigns ^ set(fg.CALLSIGN_KO), filters ^ set(fg.FILTER_NAMES))
    print("rebuild_check selftest ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tools-prompt", default=str(fg.DEFAULT_TOOLS_PROMPT))
    common.add_argument("--webserver-root", default=str(fg.DEFAULT_WEBSERVER_ROOT))
    common.add_argument("--log", default="")
    s = sub.add_parser("contract", parents=[common])
    s.add_argument("--snapshot", default=str(SNAPSHOT))
    s = sub.add_parser("snapshot", parents=[common])
    s.add_argument("--snapshot", default=str(SNAPSHOT))
    s = sub.add_parser("carry", parents=[common])
    for k in ("--old-v4", "--old-v5", "--new-v4", "--out"):
        s.add_argument(k, required=True)
    s = sub.add_parser("subset")
    s.add_argument("--src", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=200)
    s = sub.add_parser("score", parents=[common])
    s.add_argument("files", nargs="+", help="name=fg_eval.jsonl")
    s.add_argument("--gate", default="")
    s.add_argument("--pass-rate", type=float, default=0.98)
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
