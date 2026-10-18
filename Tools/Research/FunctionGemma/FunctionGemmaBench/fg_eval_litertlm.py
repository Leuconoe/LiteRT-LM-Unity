"""Score a converted .litertlm bundle on the Korean voice-command cases through
the Windows LiteRT-LM CLI — the parity check between the HF checkpoint
(fg_eval.py) and the file that actually ships.

The prompt is the HF render, split into the runtime's two inputs: the first
turn's content goes to --system_message_file, the user content to
--input_prompt_file, and no --tools_json_file is passed, so the runtime's own
tool formatter (which renders differently from HF) never runs. The CLI's
prefill token count is compared with the HF token count of the same render on
every case: equal counts are the prompt-identity proof.

  functiongemma: developer turn = TRIGGER + declarations → system message.
  json (gemma-3): HF folds `system` into the first user turn and Gemma3 has no
                  system role, so the folded user content is sent as the prompt
                  and no system message is passed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transformers import AutoTokenizer

import fg_common as fg

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DEFAULT_CLI = ROOT / "Tools" / "Windows" / "Bin" / "litert_lm_main.windows_x86_64.exe"

_FG = re.compile(r"^<bos><start_of_turn>developer\n(.*)<end_of_turn>\n<start_of_turn>user\n(.*)<end_of_turn>\n<start_of_turn>model\n$", re.S)
_JSON = re.compile(r"^<bos><start_of_turn>user\n(.*)<end_of_turn>\n<start_of_turn>model\n$", re.S)
_BENCH = {
    "prefill_tokens": r"Prefill Turn 1: Processed (\d+) tokens",
    "prefill_tps": r"Prefill Speed: ([\d.]+)",
    "decode_tokens": r"Decode Turn 1: Processed (\d+) tokens",
    "decode_tps": r"Decode Speed: ([\d.]+)",
    "ttft_s": r"Time to first token: ([\d.]+) s",
    "init_ms": r"Init Total: ([\d.]+) ms",
}


def split_render(fmt: str, rendered: str) -> tuple[str, str]:
    """HF render → (system message, user prompt) for the CLI."""
    m = (_FG if fmt == "functiongemma" else _JSON).match(rendered)
    if not m:
        raise ValueError(f"render does not match the {fmt} single-turn shape: {rendered[:120]!r}")
    return (m.group(1), m.group(2)) if fmt == "functiongemma" else ("", m.group(1))


def parse_stdout(stdout: str, prompt: str) -> tuple[str, dict]:
    """CLI stdout = `input_prompt: <prompt>\\n<response>\\n\\nBenchmarkInfo: …`."""
    body, _, bench_text = stdout.partition("\nBenchmarkInfo:")
    echo = f"input_prompt: {prompt}\n"
    raw = body[len(echo):] if body.startswith(echo) else body
    bench = {}
    for key, pattern in _BENCH.items():
        m = re.search(pattern, bench_text)
        if m:
            bench[key] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
    return raw.strip("\n"), bench


def run_case(cli: Path, bundle: Path, backend: str, system: str, prompt: str) -> tuple[str, dict, float, str]:
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [str(cli), f"--backend={backend}", f"--model_path={bundle}"]
        prompt_file = Path(tmp) / "prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        cmd.append(f"--input_prompt_file={prompt_file}")
        if system:
            system_file = Path(tmp) / "system.txt"
            system_file.write_text(system, encoding="utf-8")
            cmd.append(f"--system_message_file={system_file}")
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, cwd=cli.parent)
        wall = time.perf_counter() - t0
    stdout = proc.stdout.decode("utf-8", "replace").replace("\r\n", "\n")
    err = "" if proc.returncode == 0 else proc.stderr.decode("utf-8", "replace")[-400:]
    raw, bench = parse_stdout(stdout, prompt)
    return raw, bench, wall, err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--format", choices=fg.FORMATS, default="functiongemma")
    ap.add_argument("--decl-mode", choices=fg.DECL_MODES, default="none")
    ap.add_argument("--cases", default=str(HERE / "eval_cases.json"))
    ap.add_argument("--tokenizer", default="", help="HF id or merged dir; default: the format's base model")
    ap.add_argument("--label", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4, help="parallel CLI processes; 1 for clean timings")
    ap.add_argument("--cli", default=str(DEFAULT_CLI))
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cases_path = Path(args.cases if Path(args.cases).exists() else HERE / args.cases)
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    if args.limit:
        cases = cases[: args.limit]
    tok = AutoTokenizer.from_pretrained(args.tokenizer or fg.DEFAULT_MODELS[args.format])
    render, parse, tools = fg.make_format(args.format, args.decl_mode)
    tool_index = {t["function"]["name"]: t for t in tools}
    cli, bundle = Path(args.cli), Path(args.bundle).resolve()

    jobs = []
    for c in cases:
        rendered = render(tok, c["text"])
        system, prompt = split_render(args.format, rendered)
        jobs.append((c, system, prompt, len(tok(rendered, add_special_tokens=False)["input_ids"])))

    def work(job):
        c, system, prompt, hf_tokens = job
        raw, bench, wall, err = run_case(cli, bundle, args.backend, system, prompt)
        got = parse(raw)
        passed, reason = fg.grade(c, got, tool_index)
        return {**c, "passed": passed, "reason": reason, "got": got, "raw": raw, "wall_s": round(wall, 3),
                "hf_tokens": hf_tokens, **bench, **({"error": err} if err else {})}

    # The first run writes the bundle's XNNPACK cache next to it; let it finish alone.
    rows = [work(jobs[0])]
    with ThreadPoolExecutor(max(1, args.jobs)) as pool:
        rows += list(pool.map(work, jobs[1:]))

    label = args.label or f"{bundle.parent.name}-{args.format}-{args.decl_mode}"
    by_suite = defaultdict(lambda: [0, 0])
    for r in rows:
        by_suite[r.get("suite", "-")][0] += r["passed"]
        by_suite[r.get("suite", "-")][1] += 1
    total = sum(r["passed"] for r in rows)
    fn_ok = sum(1 for r in rows if r["got"] and r["got"]["function"] == r["function"])
    reasons = Counter(r["reason"] for r in rows if not r["passed"])
    mismatch = [r for r in rows if r.get("prefill_tokens") != r["hf_tokens"]]
    errors = [r for r in rows if "error" in r]

    def mean(key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 2) if vals else None

    timing = {k: mean(k) for k in ("wall_s", "ttft_s", "prefill_tps", "decode_tps", "init_ms", "prefill_tokens")}
    print(f"\n{label}  bundle={bundle}  format={args.format}  decl={args.decl_mode}  cases={cases_path.name}  jobs={args.jobs}")
    print("| suite | pass |\n| --- | --- |")
    for suite, (p, n) in by_suite.items():
        print(f"| {suite} | {p}/{n} |")
    print(f"| **total** | **{total}/{len(rows)}** |")
    print(f"function-name correct {fn_ok}/{len(rows)}; failures by reason: {dict(reasons)}")
    print(f"prompt identity: {len(rows) - len(mismatch)}/{len(rows)} cases CLI prefill tokens == HF tokens")
    for r in mismatch[:10]:
        print(f"  mismatch [{r.get('id')}] cli={r.get('prefill_tokens')} hf={r['hf_tokens']} {r['text']!r}")
    if errors:
        print(f"CLI errors: {len(errors)} — first: {errors[0]['error']!r}")
    print(f"mean: {timing}  (jobs={args.jobs}; use --jobs 1 for uncontended timings)")
    print("\nfailures:")
    for r in rows:
        if not r["passed"]:
            exp = f"{r['function']}{json.dumps(r['arguments'], ensure_ascii=False)}"
            got = f"{r['got']['function']}{json.dumps(r['got']['arguments'], ensure_ascii=False)}" if r["got"] else repr(r["raw"][:80])
            print(f"  [{r.get('id', '-')}|{r['reason']}] {r['text']}  →  {got}   (want {exp})")

    out = Path(args.out) if args.out else ROOT / "Builds" / "Logs" / f"fg-cli-{label}-{cases_path.stem}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=list) + "\n")
        f.write(json.dumps({"summary": True, "label": label, "bundle": str(bundle), "format": args.format,
                            "decl_mode": args.decl_mode, "cases": cases_path.name, "passed": total, "total": len(rows),
                            "function_correct": fn_ok, "by_suite": dict(by_suite), "reasons": dict(reasons),
                            "token_mismatches": len(mismatch), "cli_errors": len(errors), "jobs": args.jobs,
                            "timing": timing}, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    assert parse_stdout("input_prompt: 알파팀\n선택\n<start_function_call>call:X{}<end_function_call>\n\nBenchmarkInfo:\n"
                        "  Prefill Turn 1: Processed 31 tokens in 9ms\n      Prefill Speed: 1892.65 tokens/sec.\n",
                        "알파팀\n선택") == ("<start_function_call>call:X{}<end_function_call>",
                                            {"prefill_tokens": 31, "prefill_tps": 1892.65})
    assert split_render("functiongemma", "<bos><start_of_turn>developer\nS<end_of_turn>\n<start_of_turn>user\nU<end_of_turn>\n<start_of_turn>model\n") == ("S", "U")
    assert split_render("json", "<bos><start_of_turn>user\nS\n\nU<end_of_turn>\n<start_of_turn>model\n") == ("", "S\n\nU")
    raise SystemExit(main())
