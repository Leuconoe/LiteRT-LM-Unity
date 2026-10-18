"""LLM paraphrases of the template data, kept only where two teacher LLMs agree with the label.

Template data (make_dataset.py) cannot reach real mid-mission speech — remark + command +
urgency, radio endings, honorific mixing. A local LLM rewrites template seeds in that style;
each rewrite inherits its seed's label and survives only if two teachers of different model
families, given the WebServer's own system prompt + ToolsPrompt.txt and constrained to the
WebServer's tool-call JSON schema, call exactly that function with exactly those arguments
(DefaultResponse: function name only, as everywhere in this bench).

Stages (run with the FunctionGemmaBench venv; servers are vLLM in Docker, one at a time):
  serve  --model HF_ID          start vllm/vllm-openai on :8000 (served name "m")
  stop                          remove the container
  seeds                         stratified seeds from data_v4_empty/train.jsonl
  gen                           paraphrase seeds            -> out/raw.jsonl
  teach  --label A              teacher calls on raw        -> out/teacher_A.jsonl
  teach  --label A --cases eval_cases.json                  (teacher accuracy check)
  build                         data_v4_empty + kept        -> ../data_v5_empty/
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))
import fg_common as fg  # noqa: E402
import make_dataset as md  # noqa: E402  (norm + slot vocab; module has a __main__ guard)

OUT = HERE / "out"
URL = "http://127.0.0.1:8000/v1/chat/completions"
IMAGE = "vllm/vllm-openai:v0.29.0"
CONTAINER = "fg-vllm"
HF_CACHE = Path.home() / ".cache" / "huggingface"
BLOCK_SETS = ["eval_cases.json", "heldout_cases.json", "heldout2_cases.json", "heldout3_cases.json", "heldout4_cases.json"]
SEEDS_PER_TOOL, DEFAULT_SEEDS, REWRITES = 45, 550, 4

PARA_SYSTEM = """너는 한국어 워게임 시뮬레이터용 AR 음성 명령 데이터를 만드는 도우미다.
주어진 음성 명령 한 문장을, 작전 중인 운용자가 무전하듯 실제로 말할 법한 서로 다른 문장으로 바꿔 쓴다.

반드시 지킬 것:
- 의미는 그대로: 실행할 동작 하나와 그 대상. 동작을 더하거나 빼지 않는다. 두 번째 명령을 붙이지 않는다.
- 팀 콜사인(브라보, Bravo 등), 숫자 값, 창 이름, 필터 이름, 컨펌 창 이름, 팀 타입, 전투원 이름은 그대로 둔다.
  숫자는 같은 값이면 3 / 삼 / 셋 / 세 처럼 표기를 바꿔도 된다.
- 문장마다 말투를 다르게: 앞에 상황 설명(관측, 피해, 배터리, 지형, 시간 압박), 뒤에 재촉이나 무전 종결어
  (빨리, 서둘러, 당장, 지금, 이상, 오버, 바람), 존댓말·반말·군대식 어미(~할 것, ~바람, ~합니다, ~해라),
  필러(어, 음, 자), 쉼표 유무를 섞는다. 상황 설명만 있고 명령이 없는 문장은 만들지 않는다.
- 한 문장은 한 번의 발화, 60자 이내.

예) 명령: 패널 원위치 / 의미: ResetPanelPositions()
{"rewrites": ["화면이 정신없다, 패널 원위치 바람", "어 패널들 원래 자리로 돌려 놔, 빨리", "패널 위치 초기화합니다, 이상"]}

JSON {"rewrites": [...]} 만 출력한다."""

DEFAULT_MEANING = ("실행할 수 없는 요청(DefaultResponse). 원래 문장의 성격 — 부정, 상태 질문, 지원하지 않는 기능, "
                   "대상이나 번호 누락, 두 가지 동작, 없는 콜사인 — 을 그대로 유지하고 실행 가능한 명령으로 바꾸지 않는다.")


def post(body: dict, timeout: float = 300) -> dict:
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(system: str, user: str, schema: dict, temperature: float, max_tokens: int) -> str:
    body = {"model": "m", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature, "max_tokens": max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}}
    for attempt in range(3):
        try:
            return post(body)["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001 — server hiccup: retry, then give up on this item
            last = e
            time.sleep(2 * (attempt + 1))
    return f"__error__ {last}"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def banned() -> tuple[set, set]:
    keys = {md.norm(c["text"]) for f in BLOCK_SETS for c in json.loads((BENCH / f).read_text(encoding="utf-8"))["cases"]}
    return keys, {k.replace(" ", "") for k in keys}


def ws_schema() -> dict:
    """The WebServer's closed tool-call schema (one oneOf branch per tool), loaded by path like fg_common."""
    mods = {}
    for name in ("tools_parser", "tool_call_validation"):
        spec = importlib.util.spec_from_file_location(f"ws_{name}", fg.DEFAULT_WEBSERVER_ROOT / "server" / f"{name}.py")
        mods[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mods[name])
    parsed = mods["tools_parser"].parse_typescript_tools(fg.DEFAULT_TOOLS_PROMPT.read_text(encoding="utf-8"))
    return mods["tool_call_validation"].build_tool_call_json_schema(parsed)


def meaning(rec: dict) -> str:
    if rec["function"] == "DefaultResponse":
        return DEFAULT_MEANING
    return f'{rec["function"]}({", ".join(f"{k}={v}" for k, v in rec["arguments"].items())})'


LATIN_OK = {en.lower() for en, _ in md.TEAMS} | {"x", "ray", "ugv", "ai", "e"}


def script_ok(text: str) -> bool:
    """The 7.8B paraphraser sometimes drifts into Chinese/kana or English filler ('Alright', 'Confirm')."""
    if re.search(r"[぀-ヿ㐀-鿿]", text):
        return False
    return all(w.lower() in LATIN_OK for w in re.findall(r"[A-Za-z]+", text))


def slots_kept(rec: dict, text: str) -> bool:
    """Cheap pre-filter: every named slot value must still be spoken (space-insensitive)."""
    t, a = text.replace(" ", ""), rec["arguments"]
    if "team" in a and not any(alias.replace(" ", "") in t for alias in md.TEAM_ALIASES[a["team"]]):
        return False
    for key in ("windowName", "confirmName", "soldierName"):
        if key in a and a[key].replace(" ", "") not in t:
            return False
    if "filterNames" in a and not all(f.strip().replace(" ", "") in t for f in a["filterNames"].split(",")):
        return False
    if "teamType" in a and not any(alias in t for name, aliases in md.TEAM_TYPES if name == a["teamType"] for alias in aliases):
        return False
    return True


# ------------------------------------------------------------------ stages

def cmd_serve(args) -> None:
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
    repo = f"models--{args.model.replace('/', '--')}"
    snap = next((HF_CACHE / "hub" / repo / "snapshots").iterdir()).name
    # Mount the whole hub: snapshot files link to repo blobs, which link on into hub/blobs/xx/.
    # Docker Desktop's WSL2 GPU has no UVA, which vLLM's v2 model runner requires ("UVA is not available").
    subprocess.run(["docker", "run", "-d", "--name", CONTAINER, "--gpus", "all", "-p", "8000:8000", "--ipc=host",
                    "-e", "VLLM_USE_V2_MODEL_RUNNER=0",
                    "-v", f"{HF_CACHE / 'hub'}:/hub:ro", IMAGE, "--model", f"/hub/{repo}/snapshots/{snap}", "--served-model-name", "m",
                    "--max-model-len", str(args.max_len), "--gpu-memory-utilization", str(args.gpu_util),
                    "--enable-prefix-caching", "--trust-remote-code", "--max-num-seqs", "64"], check=True)
    for _ in range(120):   # first start compiles kernels; give it up to 20 min
        time.sleep(10)
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5)
            print("server ready:", args.model)
            return
        except Exception:  # noqa: BLE001
            state = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER], capture_output=True, text=True).stdout.strip()
            if state != "true":
                print(subprocess.run(["docker", "logs", "--tail", "40", CONTAINER], capture_output=True, text=True).stdout)
                raise SystemExit("vLLM container exited")
    raise SystemExit("server not ready after 20 min")


def cmd_stop(args) -> None:
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)


def cmd_seeds(args) -> None:
    rows = read_jsonl(BENCH / "data_v4_empty" / "train.jsonl")
    by_fn = defaultdict(list)
    for r in rows:
        by_fn[r["function"]].append(r)
    rnd = random.Random(md.SEED)
    seeds = []
    for fn, recs in sorted(by_fn.items()):
        seeds += rnd.sample(recs, min(len(recs), DEFAULT_SEEDS if fn == "DefaultResponse" else SEEDS_PER_TOOL))
    for i, s in enumerate(seeds):
        s["seed_id"] = i
    write_jsonl(OUT / "seeds.jsonl", seeds)
    print(len(seeds), "seeds;", Counter(s["function"] == "DefaultResponse" for s in seeds))


def cmd_gen(args) -> None:
    seeds = read_jsonl(OUT / "seeds.jsonl")
    schema = {"type": "object", "properties": {"rewrites": {"type": "array", "items": {"type": "string", "maxLength": 80},
              "minItems": REWRITES, "maxItems": REWRITES}}, "required": ["rewrites"], "additionalProperties": False}

    def one(s):
        raw = chat(PARA_SYSTEM, f"명령: {s['text']} / 의미: {meaning(s)}", schema, 0.9, 400)
        try:
            return s, json.loads(raw)["rewrites"]
        except Exception:  # noqa: BLE001
            return s, []

    keys, keys_ns = banned()
    have = {md.norm(r["text"]) for r in read_jsonl(BENCH / "data_v4_empty" / "train.jsonl")}
    out, drops = [], Counter()
    with ThreadPoolExecutor(48) as ex:
        for s, rewrites in ex.map(one, seeds):
            seen = set()
            for t in rewrites:
                t = t.strip().strip('"').strip()
                k = md.norm(t)
                reason = ("empty" if not k else "dup" if k in seen or k in have else "banned" if k in keys or k.replace(" ", "") in keys_ns
                          else "script" if not script_ok(t) else "slot" if not slots_kept(s, t) else "")
                if reason:
                    drops[reason] += 1
                    continue
                seen.add(k)
                out.append({"text": t, "function": s["function"], "arguments": s["arguments"], "seed_id": s["seed_id"], "seed": s["text"]})
    write_jsonl(OUT / "raw.jsonl", out)
    print(f"{len(out)} paraphrases from {len(seeds)} seeds; dropped {dict(drops)}")


def cmd_teach(args) -> None:
    tools = fg.load_tools()
    index = {t["function"]["name"]: t for t in tools}
    system = fg.json_instructions("full", tools, fg.DEFAULT_TOOLS_PROMPT)
    schema = ws_schema()
    items = json.loads((BENCH / args.cases).read_text(encoding="utf-8"))["cases"] if args.cases else read_jsonl(OUT / "raw.jsonl")

    def one(item):
        raw = chat(system, item["text"], schema, 0.0, 160)
        got = fg.validate(fg.parse_json_call(raw), index)
        return {**item, "teacher": got, "agree": fg.grade(item, got, index)[0]}

    with ThreadPoolExecutor(48) as ex:
        rows = list(ex.map(one, items))
    name = f"teacher_{args.label}" + (f"_{Path(args.cases).stem}" if args.cases else "")
    write_jsonl(OUT / f"{name}.jsonl", rows)
    suites = defaultdict(lambda: [0, 0])
    for r in rows:
        key = r.get("suite") or ("default" if r["function"] == "DefaultResponse" else "exec")
        suites[key][0] += r["agree"]
        suites[key][1] += 1
    print(name, f"{sum(r['agree'] for r in rows)}/{len(rows)}", {k: f"{a}/{n}" for k, (a, n) in suites.items()})


def strip_ungrounded_optional(call: dict, index: dict, text: str) -> dict:
    """Teachers (EXAONE especially) add an optional team the speaker never said; drop such keys before
    grading. Required keys and grounded optional keys still have to match the label exactly."""
    if call["function"] not in index:
        return call
    required = set(index[call["function"]]["function"].get("parameters", {}).get("required", []))
    return {**call, "arguments": {k: v for k, v in call["arguments"].items() if k in required or fg.grounded({k: v}, text)}}


def cmd_build(args) -> None:
    a, b = (read_jsonl(OUT / f"teacher_{x}.jsonl") for x in ("A", "B"))
    assert [r["text"] for r in a] == [r["text"] for r in b], "teacher files are not aligned"
    index = {t["function"]["name"]: t for t in fg.load_tools()}
    for rows in (a, b):
        for r in rows:
            r["agree_strict"] = r["agree"]
            r["agree"] = fg.grade(r, strip_ungrounded_optional(r["teacher"], index, r["text"]), index)[0]
    print(f"both teachers agree — strict {sum(r['agree_strict'] and s['agree_strict'] for r, s in zip(a, b))}, "
          f"ungrounded-optional stripped {sum(r['agree'] and s['agree'] for r, s in zip(a, b))} of {len(a)}")
    kept = [{k: r[k] for k in ("text", "function", "arguments", "seed_id")} for r, s in zip(a, b) if r["agree"] and s["agree"]]
    per_fn = Counter(r["function"] for r in a)
    kept_fn = Counter(r["function"] for r in kept)
    print("kept per tool (kept/raw):", {fn: f"{kept_fn[fn]}/{n}" for fn, n in sorted(per_fn.items())})
    print(f"teacher agreement: A {sum(r['agree'] for r in a)}/{len(a)}, B {sum(r['agree'] for r in b)}/{len(b)}, both {len(kept)}")

    keys, keys_ns = banned()
    hit = lambda r: md.norm(r["text"]) in keys or md.norm(r["text"]).replace(" ", "") in keys_ns  # noqa: E731
    base_train, base_val = (read_jsonl(BENCH / "data_v4_empty" / f) for f in ("train.jsonl", "val.jsonl"))
    # v4's filter stopped at heldout3; heldout4 postdates it.
    print("base v4 rows colliding with eval/heldout1-4, dropped:", [r["text"] for r in base_train + base_val if hit(r)])
    base_train, base_val = [r for r in base_train if not hit(r)], [r for r in base_val if not hit(r)]
    rnd = random.Random(md.SEED + 5)
    val_seeds = {sid for sid in {r["seed_id"] for r in kept} if rnd.random() < md.VAL_FRACTION}
    # Keep DefaultResponse near 20 %: downsample the over-represented side of the kept paraphrases.
    dr = [r for r in kept if r["function"] == "DefaultResponse"]
    ex = [r for r in kept if r["function"] != "DefaultResponse"]
    target = 0.20
    if len(dr) > target / (1 - target) * len(ex):
        dr = rnd.sample(dr, round(target / (1 - target) * len(ex)))
    elif len(ex) > (1 - target) / target * len(dr) * 1.6:   # ponytail: allow exec-heavy mix up to 1.6x; base data keeps the floor
        ex = rnd.sample(ex, round((1 - target) / target * len(dr) * 1.6))
    para = dr + ex
    strip = lambda r: {k: r[k] for k in ("text", "function", "arguments")}  # noqa: E731
    train = base_train + [strip(r) for r in para if r["seed_id"] not in val_seeds]
    val = base_val + [strip(r) for r in para if r["seed_id"] in val_seeds]
    assert not [r for r in train + val if hit(r)]
    rnd.shuffle(train)
    out = BENCH / "data_v5_empty"
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    share = sum(r["function"] == "DefaultResponse" for r in train) / len(train)
    print(f"data_v5_empty: train {len(train)} (paraphrase {len(train) - len(base_train)}), val {len(val)}, DefaultResponse {share:.1%}")


def cmd_check(args) -> None:
    team = {"function": "SelectTeam", "arguments": {"team": "Oscar"}}
    assert slots_kept(team, "적 접근, 오스카팀 잡아 빨리") and not slots_kept(team, "브라보팀 잡아")
    filt = {"function": "ApplyDisplayFilters", "arguments": {"filterNames": "위협, 기동로"}}
    assert slots_kept(filt, "위협 기동로 켜") and not slots_kept(filt, "위협 켜")
    assert slots_kept({"function": "OpenWindow", "arguments": {"windowName": "전술 지도"}}, "전술지도 꺼내")
    assert slots_kept({"function": "CreateTeam", "arguments": {"teamType": "공격", "memberCount": 4}}, "폭격 드론 넷으로 팀 짜")
    assert script_ok("X-ray팀 선택, 빨리") and script_ok("UGV 세 대로 팀") and not script_ok("패널原位置") and not script_ok("승인 Alright")
    index = {t["function"]["name"]: t for t in fg.load_tools()}
    call = {"function": "SelectDrone", "arguments": {"droneNumber": 3, "team": "Alpha"}}
    assert strip_ungrounded_optional(call, index, "3번 드론 선택")["arguments"] == {"droneNumber": 3}
    assert strip_ungrounded_optional(call, index, "알파 3호기 선택") == call
    assert len(ws_schema()["oneOf"]) == 44 and banned()[0]
    print("ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    s = sub.add_parser("serve")
    s.add_argument("--model", required=True)
    s.add_argument("--max-len", type=int, default=12288)
    s.add_argument("--gpu-util", type=float, default=0.55)
    sub.add_parser("stop")
    sub.add_parser("seeds")
    sub.add_parser("gen")
    t = sub.add_parser("teach")
    t.add_argument("--label", required=True)
    t.add_argument("--cases", default="")
    sub.add_parser("build")
    args = ap.parse_args()
    {"check": cmd_check, "serve": cmd_serve, "stop": cmd_stop, "seeds": cmd_seeds, "gen": cmd_gen, "teach": cmd_teach, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
