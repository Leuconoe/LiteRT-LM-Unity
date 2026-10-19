"""Aggregate-only scorer for the locked voice sets: prints counts per ASR arm, never rows.

  python locked/score_locked.py <fc-*.jsonl from fg_voice_e2e.py fc> <External/fg-voice-* dir> <overlap keys, e.g. heldout6>

"clean" drops the cases in overlap_with_data_v6.json; "policy" is the router's ConfirmationPolicy
(fg_common.needs_confirmation) applied to the routed call.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fg_common as fg  # noqa: E402

fc, voice_dir, keys = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(",")
flags = json.loads((voice_dir / "qc_flags.json").read_text(encoding="utf-8"))
bad = {(v, i) for v, ids in flags.items() for i in ids}
ov = json.loads((HERE / "overlap_with_data_v6.json").read_text(encoding="utf-8"))
overlap = {i for k in keys for i in ov["ids"][k]}
rows = [json.loads(l) for l in fc.read_text(encoding="utf-8").splitlines() if l.strip()]
rows = [r for r in rows if (r["voice"], r["id"]) not in bad]
for arm in sorted({r["arm"] for r in rows}):
    for tag, sub in (("all", [r for r in rows if r["arm"] == arm]), ("clean", [r for r in rows if r["arm"] == arm and r["id"] not in overlap])):
        ok = sum(r["routed_outcome"] == "ok" for r in sub)
        wrong = [r for r in sub if r["routed_outcome"] == "wrong"]
        good_exec = [r for r in sub if r["routed_outcome"] == "ok" and r["routed"] and r["routed"]["function"] != "DefaultResponse"]
        held = sum(fg.needs_confirmation(r["routed"]) for r in wrong)
        asked = sum(fg.needs_confirmation(r["routed"]) for r in good_exec)
        print(f"{arm:5} {tag:5} n {len(sub):5}  correct {ok:5} ({ok / len(sub):.1%})  wrong {len(wrong):4} ({len(wrong) / len(sub):.1%})"
              f"  policy: wrong held {held}/{len(wrong)}, wrong left {len(wrong) - held}, correct asked {asked}/{len(good_exec)}")
