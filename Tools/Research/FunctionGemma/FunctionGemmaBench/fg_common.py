"""Shared pieces for the FunctionGemma fine-tune and eval: the tool contract, prompt rendering per
declaration mode, parsing, the app-side guard, the clause split and grading.

The contract is an OpenAI-style tools JSON (sample_contract.json). Router settings ride along as
`x-` keys the model never sees:
  parameter  "x-grounded": true          the value must be traceable to the utterance
             "x-aliases": {value: [...]} spoken forms of an enum value (the value itself also counts)
  function   "x-confirm": true            the app asks before running it
LiteRtLmFunctionGemmaRouter.cs is the C# twin of validate / grounded / segments / combine_segments /
needs_confirmation — change both together (rebuild_check.py selftest runs the shared golden table).
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CONTRACT = HERE / "sample_contract.json"
DEFAULT_MODEL = "google/functiongemma-270m-it"
# The model card: this sentence is the switch into function-calling mode.
TRIGGER = "You are a model that can do function calling with the following functions"
DECL_MODES = ("full", "compact", "none")
REFUSAL_NAME = "DefaultResponse"
REFUSAL = {"function": REFUSAL_NAME, "arguments": {}}


def load_tools(contract: Path = DEFAULT_CONTRACT) -> list[dict]:
    return json.loads(Path(contract).read_text(encoding="utf-8"))


def strip_x(obj):
    """The declarations as the model sees them: every `x-` key removed."""
    if isinstance(obj, dict):
        return {k: strip_x(v) for k, v in obj.items() if not k.startswith("x-")}
    return [strip_x(v) for v in obj] if isinstance(obj, list) else obj


def tools_for_mode(tools: list[dict], mode: str) -> list[dict] | None:
    """full: as written. compact: names, types, enums, required — no prose. none: nothing (the fine-tune
    carries the contract, the prompt is the trigger sentence plus the utterance)."""
    if mode == "none":
        return None
    clean = strip_x(tools)
    if mode == "full":
        return clean
    if mode != "compact":
        raise ValueError(f"unknown decl mode {mode!r}")
    compact = copy.deepcopy(clean)
    for tool in compact:
        # The template always prints description:<escape>..<escape>; an empty string costs two tokens.
        tool["function"]["description"] = ""
        for prop in tool["function"]["parameters"]["properties"].values():
            prop["description"] = ""
    return compact


def make_format(mode: str, contract: Path = DEFAULT_CONTRACT):
    """→ (render, parse, tools). render(tok, text, call=None) gives the prompt (ends at the model turn)
    or, with a call, prompt + training target, through FunctionGemma's own chat template."""
    tools = load_tools(contract)
    mode_tools = tools_for_mode(tools, mode)

    def render(tok, text, call=None):
        messages = [{"role": "developer", "content": TRIGGER}, {"role": "user", "content": text}]
        if call is not None:
            messages.append({"role": "assistant", "tool_calls": [
                {"type": "function", "function": {"name": call["function"], "arguments": call["arguments"]}}]})
        return tok.apply_chat_template(messages, tools=mode_tools, tokenize=False, add_generation_prompt=call is None)

    return render, parse_call, tools


def index_of(tools: list[dict]) -> dict[str, dict]:
    return {t["function"]["name"]: t for t in tools}


_CALL = re.compile(r"<start_function_call>call:([A-Za-z_]\w*)\{(.*?)\}<end_function_call>", re.S)


def _scalar(raw: str):
    raw = raw.strip()
    if raw in ("true", "false"):
        return raw == "true"
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    if re.fullmatch(r"-?\d+\.\d*", raw):
        return float(raw)
    return raw  # bare, unescaped string — kept so the guard can call it a type error


def parse_args(body: str) -> dict:
    """`k:<escape>v<escape>,n:3,b:true` → dict. Escaped values may contain ',' and ':'."""
    args, i = {}, 0
    while i < len(body):
        colon = body.find(":", i)
        if colon < 0:
            break
        key = body[i:colon].strip().strip(",").strip()
        j = colon + 1
        if body.startswith("<escape>", j):
            end = body.find("<escape>", j + 8)
            end = len(body) if end < 0 else end
            args[key] = body[j + 8:end]
            i = end + 8
        else:
            end = body.find(",", j)
            end = len(body) if end < 0 else end
            args[key] = _scalar(body[j:end])
            i = end
        if i < len(body) and body[i] == ",":
            i += 1
    return args


def parse_call(raw: str) -> dict | None:
    match = _CALL.search(raw or "")
    return {"function": match.group(1), "arguments": parse_args(match.group(2))} if match else None


# ------------------------------------------------------------------ spoken Korean numbers
_SINO = "영일이삼사오육칠팔구"
_NATIVE_ONES = ["", "하나", "둘", "셋", "넷", "다섯", "여섯", "일곱", "여덟", "아홉"]
_NATIVE_TENS = ["", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"]
_COUNTER = {"하나": "한", "둘": "두", "셋": "세", "넷": "네", "스물": "스무"}


def sino(n: int) -> str:
    """24 → 이십사, 10 → 십, 115 → 백십오 (0-9999)."""
    if n == 0:
        return "영"
    out = ""
    for unit, value in (("천", 1000), ("백", 100), ("십", 10)):
        d, n = divmod(n, value)
        out += ("" if d == 1 else _SINO[d]) + unit if d else ""
    return out + (_SINO[n] if n else "")


def native(n: int, counter: bool = False) -> str:
    """7 → 일곱, 12 → 열둘; counter=True: 1 → 한, 12 → 열두, 20 → 스무 (1-99, else '')."""
    if not 0 < n < 100:
        return ""
    tens, ones = _NATIVE_TENS[n // 10], _NATIVE_ONES[n % 10]
    if counter:
        ones = _COUNTER.get(ones, ones)
        tens = _COUNTER.get(tens, tens) if not ones else tens
    return tens + ones


def number_forms(n: int) -> set[str]:
    """Every way the utterance may carry n: digits, Sino-Korean, native, native counter form."""
    return {f for f in (str(n), sino(n) if 0 <= n < 10000 else "", native(n), native(n, True)) if f}


# ------------------------------------------------------------------ the guard
def grounded(args: dict, text: str, props: dict) -> bool:
    """Every `x-grounded` slot value must be traceable to the utterance — the model may not invent one.
    ponytail: lenient substring match (일 matches 1 inside 일곱) — it catches invented slots, not misreads."""
    flat = text.replace(" ", "").lower()
    for key, value in args.items():
        spec = props.get(key, {})
        if not spec.get("x-grounded"):
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            forms = number_forms(value)
        elif isinstance(value, str):
            forms = {value, *spec.get("x-aliases", {}).get(value, [])}
        else:
            return False
        if not any(f.replace(" ", "").lower() in flat for f in forms):
            return False
    return True


def validate(got: dict | None, tool_index: dict[str, dict], text: str | None = None) -> dict:
    """The app-side guard: an unparsable, off-schema or (given the utterance) ungrounded call becomes a refusal."""
    if got is None or got["function"] not in tool_index:
        return REFUSAL
    params = tool_index[got["function"]]["function"].get("parameters", {})
    props, args = params.get("properties", {}), got["arguments"]
    if not set(params.get("required", [])) <= set(args) or not set(args) <= set(props):
        return REFUSAL
    for key, value in args.items():
        spec = props[key]
        if spec.get("type") == "number" and (isinstance(value, bool) or not isinstance(value, int)):
            return REFUSAL
        if spec.get("type") == "string" and not isinstance(value, str):
            return REFUSAL
        if "enum" in spec and value not in spec["enum"]:
            return REFUSAL
        if ("minimum" in spec and value < spec["minimum"]) or ("maximum" in spec and value > spec["maximum"]):
            return REFUSAL
    if text is not None and not grounded(args, text, props):
        return REFUSAL
    return got


# ponytail: punctuation-only split — a transcript without punctuation is never segmented.
SEGMENT_SPLIT = re.compile(r"[,.!?;…]+")


def segments(text: str) -> list[str]:
    """Clauses of an utterance, or [] when there is only one."""
    parts = [p.strip() for p in SEGMENT_SPLIT.split(text) if p.strip()]
    return parts if len(parts) > 1 else []


def combine_segments(full: dict, segment_calls: list[dict]) -> dict:
    """Fallback for a refused utterance: run each clause, execute only if exactly one distinct executable
    call comes out (remarks around a command); two or more stay a refusal — one utterance, one action."""
    calls = [c for c in segment_calls if c["function"] != REFUSAL_NAME]
    distinct = {json.dumps(c, sort_keys=True, ensure_ascii=False) for c in calls}
    return calls[0] if len(distinct) == 1 else full


def route(text: str, infer, tool_index: dict[str, dict]) -> dict:
    """The whole route: infer(text) → raw model output; refusal → clause split. Mirrors Route() in C#."""
    full = validate(parse_call(infer(text)), tool_index, text)
    parts = segments(text) if full["function"] == REFUSAL_NAME else []
    return combine_segments(full, [validate(parse_call(infer(p)), tool_index, p) for p in parts]) if parts else full


def needs_confirmation(call: dict | None, tool_index: dict[str, dict], asr_confidence: float | None = None,
                       min_asr_confidence: float = 0.0) -> bool:
    """True when the routed call must be confirmed before it runs: an `x-confirm` tool, or any executable call
    below the ASR confidence floor. A refusal never needs confirmation."""
    if not call or call["function"] == REFUSAL_NAME:
        return False
    if asr_confidence is not None and asr_confidence < min_asr_confidence:
        return True
    return bool(tool_index.get(call["function"], {}).get("function", {}).get("x-confirm"))


def grade(case: dict, got: dict | None, tool_index: dict[str, dict]) -> tuple[bool, str]:
    """Returns (passed, reason). reason ∈ ok | parse | function | type | arguments."""
    if got is None:
        return False, "parse"
    if got["function"] != case["function"]:
        return False, "function"
    if case["function"] == REFUSAL_NAME or case["arguments"] == got["arguments"]:
        return True, "ok"
    props = tool_index.get(got["function"], {}).get("function", {}).get("parameters", {}).get("properties", {})
    for key, value in got["arguments"].items():
        expected_type = props.get(key, {}).get("type")
        if (expected_type == "number" and (isinstance(value, bool) or not isinstance(value, (int, float)))) or (
            expected_type == "string" and not isinstance(value, str)
        ):
            return False, "type"
    return False, "arguments"


if __name__ == "__main__":
    assert parse_call("<start_function_call>call:SetAlarm{hour:7,minute:30}<end_function_call>") == {
        "function": "SetAlarm", "arguments": {"hour": 7, "minute": 30}}
    assert parse_call("<start_function_call>call:PauseMusic{}<end_function_call>") == {"function": "PauseMusic", "arguments": {}}
    assert parse_args("room:<escape>a, b:c<escape>,n:3") == {"room": "a, b:c", "n": 3}
    assert parse_call("no call here") is None
    assert sino(24) == "이십사" and sino(10) == "십" and sino(115) == "백십오" and sino(0) == "영"
    assert native(7) == "일곱" and native(12, True) == "열두" and native(20, True) == "스무" and native(1, True) == "한"
    assert number_forms(3) == {"3", "삼", "셋", "세"}
    index = index_of(load_tools())
    light = {"function": "TurnOnLight", "arguments": {"room": "Kitchen"}}
    assert validate(light, index, "부엌 불 켜 줘") == light and validate(light, index, "kitchen 불") == light
    assert validate(light, index, "거실 불 켜 줘") == REFUSAL  # grounded on the wrong room
    assert validate({"function": "TurnOnLight", "arguments": {"room": "Garage"}}, index) == REFUSAL  # enum miss
    assert validate({"function": "TurnOnLight", "arguments": {}}, index) == REFUSAL  # required missing
    assert validate({"function": "SetVolume", "arguments": {"level": "7"}}, index) == REFUSAL  # type
    assert validate({"function": "SetVolume", "arguments": {"level": 70}}, index) == REFUSAL  # maximum
    temp = {"function": "SetTemperature", "arguments": {"degrees": 24}}
    assert validate(temp, index, "온도 이십사 도") == temp and validate(temp, index, "온도 24도") == temp
    assert validate(temp, index, "온도 좀 올려") == REFUSAL  # invented number
    alarm = {"function": "SetAlarm", "arguments": {"hour": 7}}
    assert validate(alarm, index, "일곱 시에 깨워 줘") == alarm
    assert validate(None, index) == REFUSAL and validate({"function": "Nope", "arguments": {}}, index) == REFUSAL
    assert segments("좀 춥네, 온도 이십오 도로 맞춰 줘") == ["좀 춥네", "온도 이십오 도로 맞춰 줘"] and segments("불 꺼") == []
    pause = {"function": "PauseMusic", "arguments": {}}
    assert combine_segments(REFUSAL, [REFUSAL, light]) == light
    assert combine_segments(REFUSAL, [light, pause]) == REFUSAL and combine_segments(REFUSAL, [light, dict(light)]) == light
    unlock = {"function": "UnlockDoor", "arguments": {}}
    assert needs_confirmation(unlock, index) and not needs_confirmation(light, index) and not needs_confirmation(REFUSAL, index)
    assert needs_confirmation(light, index, 0.4, 0.6) and not needs_confirmation(light, index, 0.9, 0.6)
    assert not needs_confirmation(REFUSAL, index, 0.1, 0.6)
    assert grade({"function": "SetVolume", "arguments": {"level": 3}}, {"function": "SetVolume", "arguments": {"level": "3"}}, index) == (False, "type")
    assert grade({"function": REFUSAL_NAME, "arguments": {}}, REFUSAL, index)[0]
    assert "x-aliases" not in json.dumps(tools_for_mode(load_tools(), "full")) and tools_for_mode([], "none") is None
    print(f"ok — {len(index)} tools")
