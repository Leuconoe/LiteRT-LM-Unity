"""Template-driven training data for a FunctionGemma router: contract + templates → train/val JSONL.

  python make_dataset.py [--contract sample_contract.json] [--templates sample_templates.json]
                         [--cases sample_eval_cases.json] [--per-tool 400] [--out data]

One record per line, the shape fg_train.py reads:
  {"text": "거실 불 켜 줘", "function": "TurnOnLight", "arguments": {"room": "LivingRoom"}}

Positives: every template of every tool, slots filled from the contract (enum values and their
`x-aliases`, numbers in [minimum, maximum] as digits / Sino-Korean / native counter form), dressed
like real speech (fillers, a lead-in clause, a dropped or added space, a trailing period). Refusals
(DefaultResponse, empty arguments) are ~20 % of the set: out-of-scope or negated requests, a command
missing a required value, and two commands in one utterance. Every text is exact-filtered —
spaces and punctuation ignored — against the eval cases, so the eval stays unseen.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import fg_common as fg

HERE = Path(__file__).resolve().parent
SLOT = re.compile(r"\{(\w+)(?::(\w+))?(?:\|(\w+))?\}")
_JOSA = {"을": ("를", "을"), "은": ("는", "은"), "이": ("가", "이"), "과": ("와", "과"), "으로": ("로", "으로")}
_DIGIT_FINAL = {"0": True, "1": True, "2": False, "3": True, "4": False, "5": False, "6": True, "7": True, "8": True, "9": False}


def josa(word: str, particle: str) -> str:
    """거실+을 → 거실을, 주방+으로 → 주방으로, 7+으로 → 7로 (ㄹ-final takes 로)."""
    last = word[-1]
    if "가" <= last <= "힣":
        final = (ord(last) - 0xAC00) % 28
        has, rieul = final != 0, final == 8
    else:  # digits read in Korean; Latin letters read vowel-final
        has, rieul = _DIGIT_FINAL.get(last, False), last in "178"
    vowel_form, consonant_form = _JOSA[particle]
    return word + (vowel_form if not has or (particle == "으로" and rieul) else consonant_form)


def key(text: str) -> str:
    """Exact-text filter key: spaces and punctuation ignored."""
    return re.sub(r"[\s.,!?~]+", "", text)


class Generator:
    def __init__(self, tools: list[dict], spec: dict, rng: random.Random):
        self.index, self.spec, self.rng = fg.index_of(tools), spec, rng

    def value(self, props: dict, name: str, fmt: str | None) -> tuple[object, str]:
        p = props[name]
        if "enum" in p:
            v = self.rng.choice(p["enum"])
            return v, self.rng.choice(p.get("x-aliases", {}).get(v) or [v])  # aliases are how it is said
        if p.get("type") == "number":
            n = self.rng.randint(int(p.get("minimum", 0)), int(p.get("maximum", 10)))
            spoken = fg.native(n, True) if fmt == "native" else fg.sino(n)
            return n, self.rng.choice([str(n), spoken or str(n)])
        raise ValueError(f"{name}: only enum strings and bounded numbers can be generated (add a template value list)")

    def render(self, tool: str, template: str) -> dict:
        props = self.index[tool]["function"]["parameters"]["properties"]
        args = {}

        def fill(m):
            v, spoken = self.value(props, m.group(1), m.group(2))
            args[m.group(1)] = v
            return josa(spoken, m.group(3)) if m.group(3) else spoken

        return {"text": SLOT.sub(fill, template), "function": tool, "arguments": args}

    def dress(self, text: str) -> str:
        r, s = self.rng, self.spec
        if r.random() < 0.1:
            text = r.choice(s["clauses"]) + text
        if r.random() < 0.12:
            text = r.choice(s["fillers"]) + text
        if r.random() < 0.15:  # recogniser spacing: a dropped space, or one inside a word
            spaces = [i for i, c in enumerate(text) if c == " "]
            if spaces and r.random() < 0.7:
                i = r.choice(spaces)
                text = text[:i] + text[i + 1:]
            elif len(text) > 3:
                i = r.randrange(1, len(text) - 1)
                text = text[:i] + " " + text[i:] if text[i] != " " and text[i - 1] != " " else text
        if r.random() < 0.2:
            text += r.choice([".", "?", "!"]) if r.random() < 0.3 else "."
        return text

    def positives(self, per_tool: int) -> list[dict]:
        out = []
        for tool, templates in self.spec["tools"].items():
            for i in range(per_tool):
                rec = self.render(tool, templates[i % len(templates)])
                rec["text"] = self.dress(rec["text"])
                out.append(rec)
        return out

    def refusals(self, n: int, taken: set[str]) -> list[dict]:
        """n distinct refusals whose texts are not in `taken` (keys of the positives and the eval cases)."""
        r, s, out = self.rng, self.spec, []
        missing = [t for ts in s["missing"].values() for t in ts]
        for _ in range(50 * n):  # the fixed phrase lists run dry long before the two-action combinations
            if len(out) == n:
                break
            kind = r.random()
            if kind < 0.4:
                text = r.choice(s["refusals"])
            elif kind < 0.65:
                text = r.choice(missing)
            else:  # two actions in one utterance: never executed, the user says them one at a time
                a, b = r.sample(sorted(s["tools"]), 2)
                text = self.render(a, r.choice(s["tools"][a]))["text"] + r.choice(s["joiners"]) + self.render(b, r.choice(s["tools"][b]))["text"]
            text = self.dress(text)
            if key(text) not in taken:
                taken.add(key(text))
                out.append({"text": text, "function": fg.REFUSAL_NAME, "arguments": {}})
        return out


def coverage(tools: list[dict], spec: dict) -> list[str]:
    """What must be fixed by hand before data can be generated (rebuild_check.py contract prints it)."""
    todo, index = [], fg.index_of(tools)
    for name, tool in index.items():
        if name == fg.REFUSAL_NAME:
            continue
        params = tool["function"].get("parameters", {})
        props, required = params.get("properties", {}), set(params.get("required", []))
        templates = spec["tools"].get(name, [])
        if not templates:
            todo.append(f"templates: tool {name} has no templates")
        for t in templates:
            named = {m.group(1) for m in SLOT.finditer(t)}
            if named - set(props):
                todo.append(f"templates: {name} template {t!r} names unknown parameters {sorted(named - set(props))}")
            if required - named:
                todo.append(f"templates: {name} template {t!r} misses required {sorted(required - named)}")
        for p, s in props.items():
            if "enum" not in s and not (s.get("type") == "number" and "minimum" in s and "maximum" in s):
                todo.append(f"contract: {name}.{p} needs an enum or a number minimum/maximum to be generated")
            unsaid = [v for v in s.get("enum", []) if re.search(r"[A-Za-z]", str(v)) and not s.get("x-aliases", {}).get(v)]
            if s.get("x-grounded") and unsaid:
                todo.append(f"contract: {name}.{p} values {unsaid} have no x-aliases — a Korean utterance can never ground them")
    todo += [f"templates: {t} is not in the contract" for t in sorted(set(spec["tools"]) - set(index))]
    return todo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--contract", default=str(fg.DEFAULT_CONTRACT))
    ap.add_argument("--templates", default=str(HERE / "sample_templates.json"))
    ap.add_argument("--cases", default=str(HERE / "sample_eval_cases.json"), help="exact-filtered out of the data")
    ap.add_argument("--per-tool", type=int, default=400)
    ap.add_argument("--refusal-share", type=float, default=0.2)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--out", default=str(HERE / "data"))
    a = ap.parse_args()

    tools, spec = fg.load_tools(Path(a.contract)), json.loads(Path(a.templates).read_text(encoding="utf-8"))
    todo = coverage(tools, spec)
    if todo:
        raise SystemExit("fix first:\n  " + "\n  ".join(todo))
    rng = random.Random(a.seed)
    gen = Generator(tools, spec, rng)
    banned = {key(c["text"]) for c in json.loads(Path(a.cases).read_text(encoding="utf-8"))["cases"]} if Path(a.cases).exists() else set()
    pos = list({key(r["text"]): r for r in gen.positives(a.per_tool) if key(r["text"]) not in banned}.values())
    records = pos + gen.refusals(round(len(pos) * a.refusal_share / (1 - a.refusal_share)), banned | {key(r["text"]) for r in pos})

    index, seen, kept, dropped = fg.index_of(tools), {}, [], {"eval": 0, "dup": 0, "conflict": 0}
    for rec in records:
        k = key(rec["text"])
        if k in banned:
            dropped["eval"] += 1
            continue
        call = json.dumps([rec["function"], rec["arguments"]], sort_keys=True, ensure_ascii=False)
        if k in seen:
            dropped["dup" if seen[k] == call else "conflict"] += 1
            continue
        assert rec["function"] == fg.REFUSAL_NAME or fg.validate(rec, index, rec["text"]) is rec, rec
        seen[k] = call
        kept.append(rec)
    rng.shuffle(kept)
    n_val = max(1, round(len(kept) * a.val_fraction))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("val", kept[:n_val]), ("train", kept[n_val:])):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    share = sum(r["function"] == fg.REFUSAL_NAME for r in kept) / len(kept)
    print(f"exact-text filter: {len(banned)} eval texts; dropped {dropped}")
    print(f"train {len(kept) - n_val}  val {n_val}  DefaultResponse share {share:.1%}  -> {out}")
    return 0


if __name__ == "__main__":
    assert josa("거실", "을") == "거실을" and josa("주방", "으로") == "주방으로" and josa("부엌", "으로") == "부엌으로"
    assert josa("안방", "은") == "안방은" and josa("7", "으로") == "7로" and josa("2", "으로") == "2로" and josa("3", "으로") == "3으로"
    assert josa("침실", "으로") == "침실로" and josa("화장실", "이") == "화장실이" and josa("소리", "을") == "소리를"
    assert key("거실 불, 켜 줘.") == "거실불켜줘"
    raise SystemExit(main())
