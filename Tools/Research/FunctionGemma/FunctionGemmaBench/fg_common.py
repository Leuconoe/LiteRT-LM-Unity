"""Shared pieces for the FunctionGemma eval and fine-tune: tool loading,
prompt rendering per output format and declaration mode, parsing and grading.

Tools come from VoiceCommand-WebServer's ToolsPrompt.txt through that repo's
own TypeScript parser, so the declarations the model sees are exactly the
contract the GPU server enforces.

Two output formats over the same neutral records:
  functiongemma — FunctionGemma's chat template and function tokens
                  (`<start_function_call>call:Name{k:<escape>v<escape>}…`).
  json          — a plain Gemma3 chat template; WebServer SystemPrompt.txt +
                  tool text in the first user turn, answer = one JSON object.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
from pathlib import Path

DEFAULT_WEBSERVER_ROOT = Path(r"D:\workspace\___2025___LIGNex1-Drone\_AI\VoiceCommand-WebServer")
DEFAULT_TOOLS_PROMPT = DEFAULT_WEBSERVER_ROOT / "assets" / "ToolsPrompt.txt"

# The model card: this sentence is the switch into function-calling mode.
TRIGGER = "You are a model that can do function calling with the following functions"
DECL_MODES = ("full", "compact", "none")
FORMATS = ("functiongemma", "json")
DEFAULT_MODELS = {"functiongemma": "google/functiongemma-270m-it", "json": "google/gemma-3-270m-it"}


def load_tools(tools_prompt: Path = DEFAULT_TOOLS_PROMPT, webserver_root: Path = DEFAULT_WEBSERVER_ROOT) -> list[dict]:
    """OpenAI-format tools parsed by the WebServer's own parser.

    Loaded by file path, not `import server.tools_parser`, so server/__init__
    and its runtime dependencies never come along. Note the parser keeps only
    the `/** */` docs — the `//` header rules of ToolsPrompt.txt (call-sign
    normalization, number words, STT fixes) are not in the declarations.
    """
    spec = importlib.util.spec_from_file_location("ws_tools_parser", Path(webserver_root) / "server" / "tools_parser.py")
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    return parser.tools_to_openai_format(parser.parse_typescript_tools(Path(tools_prompt).read_text(encoding="utf-8")))


def tools_for_mode(tools: list[dict], mode: str) -> list[dict] | None:
    """functiongemma declarations. full: as written. compact: names, types,
    enums, required — no prose. none: nothing (the fine-tune carries the contract)."""
    if mode == "full":
        return tools
    if mode == "none":
        return None
    if mode != "compact":
        raise ValueError(f"unknown decl mode {mode!r}")
    compact = copy.deepcopy(tools)
    for tool in compact:
        fn = tool["function"]
        # The template always prints description:<escape>..<escape>; an empty
        # string costs two tokens, a missing key would depend on jinja's Undefined.
        fn["description"] = ""
        for prop in fn["parameters"]["properties"].values():
            prop["description"] = ""
    return compact


def compact_tool_lines(tools: list[dict]) -> str:
    """json-format compact: `SelectDrone(droneNumber: number, team?: "Alpha"|"Bravo"|…)`, one per line."""
    lines = []
    for tool in tools:
        fn = tool["function"]
        required = set(fn["parameters"]["required"])
        params = []
        for name, prop in fn["parameters"]["properties"].items():
            kind = "|".join(json.dumps(v, ensure_ascii=False) for v in prop["enum"]) if prop.get("enum") else prop["type"]
            params.append(f"{name}{'' if name in required else '?'}: {kind}")
        lines.append(f"{fn['name']}({', '.join(params)})")
    return "\n".join(lines)


def json_instructions(mode: str, tools: list[dict], tools_prompt: Path) -> str:
    """WebServer SystemPrompt.txt (sits next to ToolsPrompt.txt) minus Qwen's
    /no_think switch, followed by the tool text for the declaration mode."""
    system = "\n".join(
        line for line in (Path(tools_prompt).parent / "SystemPrompt.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() != "/no_think"
    ).strip()
    if mode == "full":
        return f"{system}\n\n{Path(tools_prompt).read_text(encoding='utf-8').strip()}"
    if mode == "compact":
        return f"{system}\n\nTools:\n{compact_tool_lines(tools)}"
    if mode == "none":
        return system
    raise ValueError(f"unknown decl mode {mode!r}")


def json_target(call: dict) -> str:
    return json.dumps({"function": call["function"], "arguments": call["arguments"]}, ensure_ascii=False, separators=(",", ":"))


def make_format(fmt: str, mode: str, tools_prompt: Path = DEFAULT_TOOLS_PROMPT,
                webserver_root: Path = DEFAULT_WEBSERVER_ROOT):
    """→ (render, parse, tools). render(tok, text, call=None) gives the prompt
    (ends at the model turn) or, with a call, prompt + training target."""
    tools = load_tools(tools_prompt, webserver_root)
    if fmt == "functiongemma":
        mode_tools = tools_for_mode(tools, mode)

        def render(tok, text, call=None):
            messages = [{"role": "developer", "content": TRIGGER}, {"role": "user", "content": text}]
            if call is not None:
                messages.append({"role": "assistant", "tool_calls": [
                    {"type": "function", "function": {"name": call["function"], "arguments": call["arguments"]}}]})
            return tok.apply_chat_template(messages, tools=mode_tools, tokenize=False, add_generation_prompt=call is None)

        return render, parse_call, tools
    if fmt == "json":
        # Gemma3 has no system turn; its template folds `system` into the first user turn.
        system = json_instructions(mode, tools, tools_prompt)

        def render(tok, text, call=None):
            messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
            if call is not None:
                messages.append({"role": "assistant", "content": json_target(call)})
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=call is None)

        return render, parse_json_call, tools
    raise ValueError(f"unknown format {fmt!r}")


_CALL = re.compile(r"<start_function_call>call:([A-Za-z_]\w*)\{(.*?)\}<end_function_call>", re.S)


def _scalar(raw: str):
    raw = raw.strip()
    if raw in ("true", "false"):
        return raw == "true"
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    if re.fullmatch(r"-?\d+\.\d*", raw):
        return float(raw)
    return raw  # bare, unescaped string — kept so the grader can call it a type/format error


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
    if not match:
        return None
    return {"function": match.group(1), "arguments": parse_args(match.group(2))}


def parse_json_call(raw: str) -> dict | None:
    """First JSON object in the output with the contract's keys, else None."""
    start = (raw or "").find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("function"), str):
        return None
    args = obj.get("arguments", {})
    return {"function": obj["function"], "arguments": args if isinstance(args, dict) else {}}


def _norm_filters(value) -> object:
    return frozenset(p.strip() for p in value.split(",") if p.strip()) if isinstance(value, str) else value


# ponytail: the filter names exist only in the ToolsPrompt prose (filterNames is a free string); update with the contract.
FILTER_NAMES = frozenset(["드론 시야", "배터리", "팀 연결선", "장애물", "마커", "전장의 안개", "위협", "적 은엄폐",
                          "아군 은엄폐", "주요 지형지물", "기동성", "기동로"])
REFUSAL = {"function": "DefaultResponse", "arguments": {}}


# Spoken forms per value: Sino, native, counter forms, the contract's STT codes (E/이=2, 이오기=2호기 …) and
# English digits the recogniser emits. ponytail: lenient substring match — catches invented slots, not misreads.
NUMBER_WORDS = {1: "일 하나 한 첫 원", 2: "이 둘 두 E 투", 3: "삼 셋 세 쓰리 사모기", 4: "사 넷 네 포", 5: "오 다섯 파이브",
                6: "육 여섯", 7: "칠 일곱", 8: "팔 여덟", 9: "구 아홉", 10: "십 열"}
CALLSIGN_KO = {"Alpha": "알파", "Bravo": "브라보", "Charlie": "찰리", "Delta": "델타", "Echo": "에코", "Foxtrot": "폭스트롯",
               "Golf": "골프", "Hotel": "호텔", "India": "인디아", "Juliet": "줄리", "Kilo": "킬로", "Lima": "리마", "Mike": "마이크",
               "November": "노벰버", "Oscar": "오스카", "Papa": "파파", "Quebec": "퀘벡", "Romeo": "로미오", "Sierra": "시에라",
               "Tango": "탱고", "Uniform": "유니폼", "Victor": "빅터", "Whiskey": "위스키", "X-ray": "엑스레이", "Yankee": "양키", "Zulu": "줄루"}


def grounded(args: dict, text: str) -> bool:
    """Every slot value must be traceable to the utterance — the model may not invent a number, team or name."""
    flat = text.replace(" ", "").lower()
    for key, value in args.items():
        if isinstance(value, int) and not isinstance(value, bool):
            if str(value) not in flat and not any(w.lower() in flat for w in NUMBER_WORDS.get(value, "").split()):
                return False
        elif key == "team" and CALLSIGN_KO.get(value, "\0") not in flat and value.lower().replace("-", "") not in flat.replace("-", ""):
            return False
        elif key == "soldierName" and value.replace(" ", "") not in flat:
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
        if key == "filterNames" and not (_norm_filters(value) and _norm_filters(value) <= FILTER_NAMES):
            return REFUSAL
    if text is not None and not grounded(args, text):
        return REFUSAL
    return got


# ponytail: punctuation-only split — a transcript without punctuation (SenseVoice woitn) is never segmented.
SEGMENT_SPLIT = re.compile(r"[,.!?;…]+")


def segments(text: str) -> list[str]:
    """Clauses of an utterance, or [] when there is only one."""
    parts = [p.strip() for p in SEGMENT_SPLIT.split(text) if p.strip()]
    return parts if len(parts) > 1 else []


def combine_segments(full: dict, segment_calls: list[dict]) -> dict:
    """Runtime fallback for a refused utterance: run each clause, execute only if exactly
    one distinct executable call comes out (remarks around a command); two or more stay
    a refusal, as the contract requires for multiple actions."""
    calls = [c for c in segment_calls if c["function"] != "DefaultResponse"]
    distinct = {json.dumps(c, sort_keys=True, ensure_ascii=False) for c in calls}
    return calls[0] if len(distinct) == 1 else full


# Irreversible / authority-moving commands never run from one utterance: on hard voices every fine-tune still
# executes 12-15 % wrong calls. Mirrors LiteRtLmFunctionGemmaRouter.ConfirmationPolicy.
CONFIRM_FUNCTIONS = frozenset(["DisbandTeam", "RevokeTeamControl", "AssignTeamControl", "DetachDrone"])
CONFIRM_WINDOWS = frozenset(["자폭 공격", "팀 해체", "게임 종료", "시뮬레이션 종료", "제어권 회수", "드론봇 팀 제외"])


def needs_confirmation(call: dict | None, asr_confidence: float | None = None, min_asr_confidence: float = 0.0,
                       functions=CONFIRM_FUNCTIONS, windows=CONFIRM_WINDOWS) -> bool:
    """True when the routed call must be confirmed by the user before it runs (a refusal never does)."""
    if not call or call["function"] == "DefaultResponse":
        return False
    if asr_confidence is not None and asr_confidence < min_asr_confidence:
        return True
    if call["function"] in functions:
        return True
    return call["function"] == "AcceptConfirmWindow" and call["arguments"].get("confirmName") in windows


def grade(case: dict, got: dict | None, tool_index: dict[str, dict]) -> tuple[bool, str]:
    """Returns (passed, reason). reason ∈ ok | parse | function | type | arguments."""
    if got is None:
        return False, "parse"
    if got["function"] != case["function"]:
        return False, "function"
    if case["function"] == "DefaultResponse":
        return True, "ok"
    want, have = dict(case["arguments"]), dict(got["arguments"])
    if "filterNames" in want:
        want["filterNames"] = _norm_filters(want["filterNames"])
        have["filterNames"] = _norm_filters(have.get("filterNames"))
    if want == have:
        return True, "ok"
    props = tool_index.get(got["function"], {}).get("function", {}).get("parameters", {}).get("properties", {})
    for key, value in have.items():
        expected_type = props.get(key, {}).get("type")
        if (expected_type == "number" and (isinstance(value, bool) or not isinstance(value, (int, float)))) or (
            expected_type == "string" and not isinstance(value, (str, frozenset))
        ):
            return False, "type"
    return False, "arguments"


if __name__ == "__main__":
    assert parse_call("<start_function_call>call:SelectDrone{droneNumber:3,team:<escape>Alpha<escape>}<end_function_call>") == {
        "function": "SelectDrone", "arguments": {"droneNumber": 3, "team": "Alpha"}}
    assert parse_call("<start_function_call>call:StartRecording{}<end_function_call>") == {"function": "StartRecording", "arguments": {}}
    assert parse_args("filterNames:<escape>위협, 기동성<escape>") == {"filterNames": "위협, 기동성"}
    assert parse_call("no call here") is None
    assert parse_json_call('```json\n{"function":"SelectDrone","arguments":{"droneNumber":3}} trailing') == {
        "function": "SelectDrone", "arguments": {"droneNumber": 3}}
    assert parse_json_call('{"name":"X"}') is None and parse_json_call("prose") is None
    case = {"function": "ClearDisplayFilters", "arguments": {"filterNames": "위협, 기동성"}}
    assert grade(case, {"function": "ClearDisplayFilters", "arguments": {"filterNames": "기동성,위협"}}, {}) == (True, "ok")
    tools = load_tools()
    index = {t["function"]["name"]: t for t in tools}
    assert grade({"function": "SelectDrone", "arguments": {"droneNumber": 3}},
                 {"function": "SelectDrone", "arguments": {"droneNumber": "3"}}, index) == (False, "type")
    assert grade({"function": "DefaultResponse", "arguments": {"text": "x"}}, {"function": "DefaultResponse", "arguments": {}}, index)[0]
    assert validate({"function": "MoveToDrone", "arguments": {"droneNumber": 1}}, index) == REFUSAL
    assert validate({"function": "ClearDisplayFilters", "arguments": {"filterNames": "팀 연결선, 지저분"}}, index) == REFUSAL
    assert validate({"function": "SelectTeam", "arguments": {"team": "알파"}}, index) == REFUSAL
    assert validate({"function": "SelectDrone", "arguments": {"droneNumber": 3, "team": "Alpha"}}, index)["function"] == "SelectDrone"
    assert validate(None, index) == REFUSAL
    cam = {"function": "ShowDroneCamera", "arguments": {"droneNumber": 3}}
    assert validate(cam, index, "드론 카메라 띄워") == REFUSAL and validate(cam, index, "삼번 드론 카메라") == cam
    assert validate(cam, index, "3번 드론 카메라") == cam and validate(cam, index, "드론 셋 카메라") == cam
    ctl = {"function": "AssignTeamControl", "arguments": {"team": "Charlie", "soldierName": "서방선박"}}
    assert validate(ctl, index, "찰리팀 제어권 서하사한테 넘겨") == REFUSAL
    jul = {"function": "SelectTeam", "arguments": {"team": "Juliet"}}
    assert validate(jul, index, "줄리에팀 선택") == jul and validate(jul, index, "킬로팀 선택") == REFUSAL
    assert validate({"function": "SelectTeam", "arguments": {"team": "X-ray"}}, index, "Xray팀")["function"] == "SelectTeam"
    assert segments("좌측에 움직임 있다, 오스카팀 선택, 빨리") == ["좌측에 움직임 있다", "오스카팀 선택", "빨리"]
    assert segments("오스카팀 선택") == []
    sel = {"function": "SelectTeam", "arguments": {"team": "Oscar"}}
    assert combine_segments(REFUSAL, [REFUSAL, sel, REFUSAL]) == sel
    assert combine_segments(REFUSAL, [sel, {"function": "StartRecording", "arguments": {}}]) == REFUSAL
    assert combine_segments(REFUSAL, [sel, dict(sel)]) == sel
    assert needs_confirmation({"function": "DisbandTeam", "arguments": {"team": "Alpha"}})
    assert needs_confirmation({"function": "AcceptConfirmWindow", "arguments": {"confirmName": "자폭 공격"}})
    assert not needs_confirmation({"function": "AcceptConfirmWindow", "arguments": {"confirmName": "전술 확정"}})
    assert not needs_confirmation({"function": "RejectConfirmWindow", "arguments": {"confirmName": "자폭 공격"}})
    assert not needs_confirmation(sel) and not needs_confirmation(REFUSAL) and not needs_confirmation(None)
    assert not needs_confirmation(sel, asr_confidence=0.1)  # gate off by default
    assert needs_confirmation(sel, asr_confidence=0.4, min_asr_confidence=0.6)
    assert not needs_confirmation(sel, asr_confidence=0.9, min_asr_confidence=0.6)
    assert not needs_confirmation(REFUSAL, asr_confidence=0.1, min_asr_confidence=0.6)
    sel = index["SelectDrone"]["function"]["parameters"]
    assert sel["required"] == ["droneNumber"] and len(sel["properties"]["team"]["enum"]) == 26, sel
    assert compact_tool_lines(tools).splitlines()[0].startswith('SelectTeam(team: "Alpha"|"Bravo"')
    assert "SelectDrone(droneNumber: number, team?: " in compact_tool_lines(tools)
    assert "/no_think" not in json_instructions("none", tools, DEFAULT_TOOLS_PROMPT)
    print(f"ok — {len(tools)} tools")
