"""Run this project's 20-case Korean tool-routing benchmark against a Hugging
Face causal LM, on the desktop GPU.

Why this exists: the LiteRT CLI can only run `.litertlm` bundles, so a model
that has no conversion path cannot be scored with the existing harness. This
driver asks the same 20 questions with the same tools and the same grading, so
a candidate can be ranked against `docs/benchmarks/fc-model-benchmark.md`
before anyone pays for a conversion port.

What it is NOT: an on-device measurement. Latency here is an RTX 4090 in
bf16/fp16 and says nothing about kona. Only the pass rate transfers.

Cases and tools are ported verbatim from
Samples~/AutomatedTests/Runtime/Benchmark/LiteRtLmFunctionCallingBenchmarkRunner.cs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The CurrentTuned profile from the C# runner. Kept as one string so the two
# stay comparable; edit both or neither.
REFERENCE_NOW = "2026-04-24 10:30:00"
SYSTEM_CURRENT_TUNED = (
    "You are a deterministic function-calling router for a Unity command UI.\n"
    "Select exactly one tool from the provided tools.\n"
    f"Current time is {REFERENCE_NOW}.\n"
    "For date ranges, output full-day or full-month ranges in YYYY-MM-DD HH:MM:SS.\n"
    "For 어제/yesterday, use the previous calendar day from 00:00:00 to 23:59:59.\n"
    "For 지난달/last month, use the previous calendar month, not the current month.\n"
    "Use View* tools for requests that say 조회, 열람, 결과 보여줘 with a date range.\n"
    "Use Visualize* tools only when the user asks to visualize or display a prepared result.\n"
    "Use DefaultResponse for unrelated requests.\n"
    "Do not explain your choice."
)

# Korean chat sanity prompts — coherence only, judged by eye, not scored.
CHAT_PROMPTS = [
    "대한민국의 수도는 어디인가요? 한 문장으로 답하세요.",
    "드론이 강풍을 만났을 때 조종사가 먼저 확인할 것 세 가지를 짧게 알려줘.",
    "다음 문장을 영어로 번역해줘: 고도 백 미터로 상승합니다.",
]


def parse_tool_call(raw: str) -> dict:
    """Extract a tool name and the two date arguments from model output.

    Accepts every shape we have actually seen: an OpenAI-style JSON block, a
    Hermes `<tool_call>` envelope, and the qwen3-coder `<function=Name>` form
    that Kanana's own template emits. The C# runner's two regexes are kept as
    the last resort so a model that answers in our old format still scores.
    """
    text = raw or ""

    # qwen3-coder / Kanana native: <function=IncreaseVolume>
    match = re.search(r"<function\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", text)
    tool = match.group(1) if match else ""

    if not tool:
        match = re.search(r'"(?:name|tool)"\s*:\s*"([^"]+)"', text)
        tool = match.group(1) if match else ""

    if not tool:
        stripped = re.sub(r"<[^>]+>", "", text).strip()
        if stripped.startswith("DefaultResponse"):
            tool = "DefaultResponse"

    def argument(name: str, end_of_day: bool) -> str:
        # JSON form: "startTime": "..."
        found = re.search(rf'"{name}"\s*:\s*"([^"]+)"', text)
        if not found:
            # qwen3-coder form: <parameter=startTime>\n2026-04-24 00:00:00
            found = re.search(
                rf"<parameter\s*=\s*{name}\s*>\s*([^<\n]+)", text, re.IGNORECASE
            )
        value = found.group(1).strip() if found else ""
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            value += " 23:59:59" if end_of_day else " 00:00:00"
        return value

    return {
        "tool": tool,
        "startTime": argument("startTime", False),
        "endTime": argument("endTime", True),
    }


def validate(case: dict, call: dict) -> tuple[bool, str]:
    if not call["tool"]:
        return False, "no tool call parsed"
    if call["tool"] != case["expectedTool"]:
        return False, f"tool mismatch expected={case['expectedTool']}, actual={call['tool']}"
    for key, field in (("expectedStartTime", "startTime"), ("expectedEndTime", "endTime")):
        expected = case.get(key)
        if expected and call[field] != expected:
            return False, f"{field} mismatch expected={expected}, actual={call[field] or '(none)'}"
    return True, "ok"


def build_inputs(tokenizer, tools, user_text, use_tools_api):
    """Return the prompt string, preferring the model's own tool template."""
    messages = [
        {"role": "system", "content": SYSTEM_CURRENT_TUNED},
        {"role": "user", "content": user_text},
    ]
    if use_tools_api:
        try:
            return tokenizer.apply_chat_template(
                messages, tools=tools, add_generation_prompt=True, tokenize=False
            )
        except Exception as exc:  # template without tool support
            print(f"  (tools= unsupported by template: {exc}; falling back)", file=sys.stderr)

    # Fallback: hand the tools to the model as text, the way the CLI does.
    messages[0]["content"] += "\n\nAvailable tools:\n" + json.dumps(
        tools, ensure_ascii=False, indent=2
    )
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="kakaocorp/kanana-2-1.3b-instruct")
    parser.add_argument("--label", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--out", default="Builds/Logs/kanana-fc-bench.jsonl")
    parser.add_argument("--no-tools-api", action="store_true",
                        help="Do not use the chat template's tools= path; inline the tools as text.")
    parser.add_argument("--chat-only", action="store_true")
    parser.add_argument("--skip-chat", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    label = args.label or args.model.split("/")[-1]
    dtype = getattr(torch, args.dtype)

    print(f"Loading {args.model} ({args.dtype}) …", flush=True)
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map=args.device, trust_remote_code=True
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started
    parameters = sum(p.numel() for p in model.parameters())
    print(f"Loaded in {load_seconds:.1f}s · {parameters/1e9:.3f}B parameters", flush=True)

    tools = json.loads((HERE / "fc_tools.json").read_text(encoding="utf-8"))
    payload = json.loads((HERE / "fc_cases.json").read_text(encoding="utf-8"))
    cases = payload["cases"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []

    def generate(prompt: str) -> tuple[str, float, int]:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        started = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started
        generated = output[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True), elapsed, len(generated)

    passed = 0
    if not args.chat_only:
        print(f"\n=== 20-case Korean tool routing ({label}) ===", flush=True)
        for case in cases:
            prompt = build_inputs(tokenizer, tools, case["user"], not args.no_tools_api)
            raw, elapsed, tokens = generate(prompt)
            call = parse_tool_call(raw)
            ok, reason = validate(case, call)
            passed += ok
            print(
                f"{case['id']} {'PASS' if ok else 'FAIL'} {elapsed:5.2f}s "
                f"{tokens/elapsed:6.1f} tok/s  {reason}",
                flush=True,
            )
            if not ok:
                print(f"     raw: {raw[:220].replace(chr(10), ' ')}", flush=True)
            records.append({
                "kind": "fc", "model": args.model, "label": label, "id": case["id"],
                "user": case["user"], "expectedTool": case["expectedTool"],
                "expectedStartTime": case.get("expectedStartTime", ""),
                "expectedEndTime": case.get("expectedEndTime", ""),
                "parsed": call, "pass": ok, "reason": reason,
                "seconds": round(elapsed, 3), "newTokens": tokens,
                "decodeTokensPerSecond": round(tokens / elapsed, 2), "raw": raw,
            })
        print(f"\n{label}: {passed}/{len(cases)} = {passed/len(cases):.3f}", flush=True)

    if not args.skip_chat:
        print(f"\n=== Korean chat sanity ({label}) ===", flush=True)
        for index, prompt_text in enumerate(CHAT_PROMPTS, start=1):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                add_generation_prompt=True, tokenize=False,
            )
            raw, elapsed, tokens = generate(prompt)
            print(f"\nC{index} ({elapsed:.2f}s, {tokens/elapsed:.1f} tok/s) {prompt_text}")
            print(f"  → {raw.strip()[:600]}", flush=True)
            records.append({
                "kind": "chat", "model": args.model, "label": label, "id": f"C{index}",
                "user": prompt_text, "raw": raw, "seconds": round(elapsed, 3),
                "newTokens": tokens, "decodeTokensPerSecond": round(tokens / elapsed, 2),
            })

    summary = {
        "kind": "summary", "model": args.model, "label": label,
        "parameters": parameters, "dtype": args.dtype, "device": args.device,
        "loadSeconds": round(load_seconds, 2),
        "passed": passed, "total": 0 if args.chat_only else len(cases),
        "accuracy": None if args.chat_only else round(passed / len(cases), 3),
        "toolsApi": not args.no_tools_api,
    }
    records.append(summary)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
