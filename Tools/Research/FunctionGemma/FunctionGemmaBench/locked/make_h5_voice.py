"""Voice version of the LOCKED heldout5 set: Unity AI (ElevenLabs Multilingual V2) TTS.

Score-only set — this script makes audio and a manifest; it never looks at model output.

  py -3.13 make_h5_voice.py build                 # spoken text + manifest + per-voice jobs
  py -3.13 make_h5_voice.py run --voice alice     # synthesize missing clips (Unity editor on port 7800)
  py -3.13 make_h5_voice.py qc                    # duration outliers -> qc_flags.json
  py -3.13 make_h5_voice.py run --voice alice --redo qc_flags.json   # regenerate flagged once

Generation goes through the UnityAiTts editor tool (project _AI/UnityAiTts, `unity command eval`),
5 points per clip, 44.1 kHz WAV converted to 16 kHz mono for the ASR arms.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
OUT = ROOT / "External" / "fg-voice-h5"
PROJECT = ROOT.parent / "UnityAiTts"
CASES = [HERE / "heldout5a_cases.json", HERE / "heldout5b_cases.json"]
VOICES = {  # name -> (engine, engine voice id); Unity AI costs points, the rest are free Korean-native baselines
    "alice": ("unity", "Alice"), "george": ("unity", "George"),
    "sunhi": ("edge", "ko-KR-SunHiNeural"), "injoon": ("edge", "ko-KR-InJoonNeural"),
    "hyunsu": ("edge", "ko-KR-HyunsuMultilingualNeural"),
    "st-f1": ("supertonic", "F1.json"), "st-m1": ("supertonic", "M1.json"),
}
RAW_EXT = {"unity": ".wav", "edge": ".mp3", "supertonic": ".wav"}
ST = ROOT / "Assets" / "StreamingAssets" / "TTS" / "supertonic-litert"
ST_STYLES = OUT / "supertonic_styles"  # F1 from the deployed assets, M1 from HF Supertone/supertonic voice_styles
ST_PY = ROOT / "Tools" / "Research" / "SenseVoice" / "SenseVoiceBench" / ".venv" / "Scripts" / "python.exe"  # has ai_edge_litert
# edge-tts < 7.3 gets HTTP 403 from the service (stale client token); point this at a newer copy
# (`py -3.13 -m pip install --target <dir> edge-tts`) instead of upgrading the shared Python.
EDGE_LIB = OUT / "_edgetts-lib"
GATE_CER = 0.6  # SenseVoice transcript vs spoken text: above this the clip is unintelligible, not hard
POINTS_PER_CLIP = 5
FFMPEG = shutil.which("ffmpeg") or str(Path.home() / "AppData/Local/Microsoft/WinGet/Packages/"
                                      "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1.2-full_build/bin/ffmpeg.exe")

# ---------------------------------------------------------------- spoken text (TTS input only; labels untouched)

SINO = "영일이삼사오육칠팔구"
NATIVE = {1: "한", 2: "두", 3: "세", 4: "네", 5: "다섯", 6: "여섯", 7: "일곱", 8: "여덟", 9: "아홉", 10: "열",
          20: "스무"}
NATIVE_COUNTERS = ("대", "기", "개", "배", "명")  # 번/호기/미터/퍼센트 and bare numbers read Sino
WORDS = {"x-ray": "엑스레이", "alpha": "알파", "bravo": "브라보", "delta": "델타", "tango": "탱고", "victor": "빅터",
         "ugv": "유지브이", "ai": "에이아이", "on": "온", "off": "오프", "stop": "스톱", "close": "클로즈",
         "please": "플리즈", "zoom": "줌", "in": "인", "speed": "스피드", "up": "업", "pan": "팬", "left": "레프트",
         "brightness": "브라이트니스", "replay": "리플레이", "preset": "프리셋"}
LETTERS = {"a": "에이", "b": "비", "c": "씨", "d": "디", "e": "이", "f": "에프", "g": "지", "h": "에이치", "m": "엠",
           "p": "피", "q": "큐", "x": "엑스"}


def sino(n: int) -> str:
    if n == 0:
        return "영"
    out = ""
    for unit, name in ((1000, "천"), (100, "백"), (10, "십")):
        d, n = divmod(n, unit)
        if d:
            out += ("" if d == 1 else SINO[d]) + name
    return out + (SINO[n] if n else "")


def native(n: int) -> str:
    if n in NATIVE:
        return NATIVE[n]
    if 10 < n < 20:
        return "열" + NATIVE[n - 10]
    return sino(n)  # ponytail: native only to 20; larger counts are read Sino, fine for this set


def spoken(text: str) -> str:
    t = re.sub(r"(\d+)x(\d+)", lambda m: f"{sino(int(m[1]))} 바이 {sino(int(m[2]))}", text)
    t = re.sub(r"(\d+)(\s*)([가-힣]?)",
               lambda m: (native(int(m[1])) if m[3] in NATIVE_COUNTERS else sino(int(m[1]))) + (" " if m[3] else m[2]) + m[3], t)
    t = re.sub(r"[A-Za-z][A-Za-z\-]*",
               lambda m: WORDS.get(m[0].lower()) or (LETTERS.get(m[0].lower()) if len(m[0]) == 1 else None) or m[0], t)
    t = t.replace(".", " ")
    return re.sub(r"\s+", " ", t).strip()


def _selfcheck() -> None:
    assert spoken("X-ray 2호기 카메라 보여줘") == "엑스레이 이 호기 카메라 보여줘"
    assert spoken("UGV 열한 대로 팀 만들어") == "유지브이 열한 대로 팀 만들어"
    assert spoken("4대로") == "네 대로" and spoken("4기로") == "네 기로" and spoken("2배로") == "두 배로"
    assert spoken("마커 11 이동") == "마커 십일 이동" and spoken("16번") == "십육 번" and spoken("마커 2로 이동하자") == "마커 이 로 이동하자"
    assert spoken("1920x1080 크기로") == "천구백이십 바이 천팔십 크기로" and spoken("3000으로") == "삼천 으로"
    assert spoken("팀 E 복귀") == "팀 이 복귀" and spoken("Alpha팀 복귀 하지마.") == "알파팀 복귀 하지마"
    assert spoken("2번드론") == "이 번드론" and spoken("패스스루 on") == "패스스루 온"


# ---------------------------------------------------------------- build / run / qc

def cases() -> list[dict]:
    return [c for f in CASES for c in json.loads(f.read_text(encoding="utf-8"))["cases"]]


def raw_path(voice: str, cid: str) -> Path:
    return OUT / "tts" / voice / f"{cid}{RAW_EXT[VOICES[voice][0]]}"


def have(voice: str, cid: str) -> bool:
    p = raw_path(voice, cid)
    return p.exists() and p.stat().st_size > 0  # a failed edge-tts save leaves a 0-byte file


def wav16_path(voice: str, cid: str) -> Path:
    return OUT / "tts16k" / voice / f"{cid}.wav"


def build() -> None:
    rows = []
    for v in VOICES:
        for c in cases():
            rows.append({"id": c["id"], "voice": v, "kind": VOICES[v][0], "text": c["text"], "spoken_text": spoken(c["text"]),
                         "path16k": str(wav16_path(v, c["id"]))})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    changed = sum(r["text"] != r["spoken_text"] for r in rows) // len(VOICES)
    print(f"manifest: {len(rows)} rows ({len(rows) // len(VOICES)} cases x {len(VOICES)} voices), {changed} spoken texts normalised")


def unity_eval(code: str) -> str:
    r = subprocess.run(["unity", "command", "eval", code, "--project-path", str(PROJECT), "--result-only"],
                       capture_output=True, text=True, encoding="utf-8")
    body = json.loads(r.stdout) if r.stdout.strip().startswith("{") else {"errors": [{"message": (r.stdout + r.stderr)[-300:]}]}
    if not body.get("success"):
        raise RuntimeError("; ".join(e.get("message", "") for e in body.get("errors") or []))
    return str(body["result"])


def points() -> int:
    for _ in range(20):  # first call after a domain reload only starts the refresh
        p = unity_eval("return UnityAiTts.Points();")
        if p != "unknown":
            return int(p.split("/")[0])
        time.sleep(1)
    raise RuntimeError("points balance unavailable — check the editor's Unity login")


def run(voice: str, redo: Path | None, floor: int) -> None:
    rows = [r for r in map(json.loads, (OUT / "manifest.jsonl").read_text(encoding="utf-8").splitlines()) if r["voice"] == voice]
    flagged = set(json.loads(redo.read_text(encoding="utf-8")).get(voice, [])) if redo else set()
    todo = [r for r in rows if r["id"] in flagged or not have(voice, r["id"])]
    if not todo:
        convert(rows, voice)  # zero-cost: 16 kHz copies for clips that arrived but were not converted
        print(f"{voice}: nothing to generate")
        return
    raw_path(voice, "x").parent.mkdir(parents=True, exist_ok=True)
    engine, vid = VOICES[voice]
    if engine == "edge":
        synth_edge(vid, todo, voice)
    elif engine == "supertonic":
        synth_supertonic(vid, todo, voice)
    else:
        synth_unity(vid, todo, voice, floor)
    convert(todo, voice, force=True)


def synth_edge(vid: str, todo: list[dict], voice: str) -> None:
    import asyncio
    if EDGE_LIB.exists():
        sys.path.insert(0, str(EDGE_LIB))
    import edge_tts

    async def go():
        gate = asyncio.Semaphore(4)

        async def one(r):
            async with gate:
                for attempt in range(3):
                    try:
                        await edge_tts.Communicate(r["spoken_text"], vid).save(str(raw_path(voice, r["id"])))
                        if have(voice, r["id"]):
                            return
                        raise RuntimeError("empty audio")
                    except Exception as e:  # transient network errors
                        raw_path(voice, r["id"]).unlink(missing_ok=True)
                        if attempt == 2:
                            print(f"  {voice} {r['id']}: {e}")
                        await asyncio.sleep(2)
        await asyncio.gather(*(one(r) for r in todo))
    asyncio.run(go())
    print(f"{voice}: edge-tts {edge_tts.__version__} {sum(have(voice, r['id']) for r in todo)}/{len(todo)}")


def synth_supertonic(style: str, todo: list[dict], voice: str) -> None:
    from concurrent.futures import ThreadPoolExecutor

    def one(r):
        p = subprocess.run([str(ST_PY), str(ROOT / "Tools/Research/Supertonic/TtsBench/supertonic_litert.py"),
                            "--tflite-dir", str(ST / "dynamic"), "--fallback-tflite-dir", str(ST / "dynamic"), "--bucket-root", str(ST),
                            "--assets-dir", str(ST / "assets"), "--voice", str(ST_STYLES / style), "--steps", "4", "--threads", "2",
                            "--text", r["spoken_text"], "--out", str(raw_path(voice, r["id"]))],
                           capture_output=True, text=True, encoding="utf-8")
        if p.returncode:
            print(f"  {voice} {r['id']}: {(p.stderr or p.stdout)[-200:]}")
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, todo))
    print(f"{voice}: supertonic {sum(have(voice, r['id']) for r in todo)}/{len(todo)}")


def synth_unity(vid: str, todo: list[dict], voice: str, floor: int) -> None:
    avail, need = points(), len(todo) * POINTS_PER_CLIP
    if avail - need < floor:
        raise RuntimeError(f"{voice}: need {need}, balance {avail}, floor {floor} — refusing")
    work = OUT / "_unity" / voice
    work.mkdir(parents=True, exist_ok=True)
    jobs, status = work / "jobs.json", work / "status.json"
    status.unlink(missing_ok=True)
    jobs.write_text(json.dumps({"jobs": [{"text": r["spoken_text"], "voice": vid, "path": str(raw_path(voice, r["id"]))}
                                         for r in todo]}, ensure_ascii=False), encoding="utf-8")
    print(f"{voice}: {len(todo)} clips, ~{need} points (balance {avail}) — {unity_eval(f'return UnityAiTts.Run(@\"{jobs}\", @\"{status}\");')}")
    st, last, t_last = {}, -1, time.monotonic()
    while not st.get("finished"):
        time.sleep(5)
        try:
            st = json.loads(status.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if st.get("done", 0) != last:
            last, t_last = st.get("done", 0), time.monotonic()
            print(f"  {last}/{len(todo)}", flush=True)
        elif time.monotonic() - t_last > 600:
            print("  no progress for 10 min — keeping what arrived")
            break
    print(f"{voice}: spent {st.get('spent')} · balance {st.get('available')} · errors {len(st.get('errors', []))}")
    for e in st.get("errors", []):
        print("  error:", e)
    (work / f"log-{int(time.time())}.json").write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")


def convert(rows: list[dict], voice: str, force: bool = False) -> None:
    n = 0
    for r in rows:
        src, dst = raw_path(voice, r["id"]), Path(r["path16k"])
        if have(voice, r["id"]) and (force or not dst.exists()):
            dst.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                            str(dst)], check=True)
            n += 1
    print(f"{voice}: converted {n} clips to 16 kHz")


def duration(path: str) -> float:
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def qc(asr: Path | None) -> None:
    """Duration outliers, plus (given a SenseVoice ASR jsonl from fg_voice_e2e) the intelligibility gate:
    a clip whose transcript CER vs the spoken text exceeds GATE_CER is broken audio, not a hard case."""
    rows = [r for r in map(json.loads, (OUT / "manifest.jsonl").read_text(encoding="utf-8").splitlines()) if Path(r["path16k"]).exists()]
    cer = {(a["voice"], a["id"]): a["cer"] for a in map(json.loads, asr.read_text(encoding="utf-8").splitlines())} if asr else {}
    flags: dict[str, list[str]] = {}
    for v in VOICES:
        vr = [r for r in rows if r["voice"] == v]
        if not vr:
            continue
        spc = {r["id"]: duration(r["path16k"]) / max(1, len(r["spoken_text"].replace(" ", ""))) for r in vr}
        med = statistics.median(spc.values())
        dur = {i for i, s in spc.items() if s > 2 * med or s < 0.4 * med}
        gate = {r["id"] for r in vr if cer.get((v, r["id"]), 0) > GATE_CER}
        flags[v] = sorted(dur | gate)
        scored = sum((v, r["id"]) in cer for r in vr)
        print(f"{v}: {len(vr)} clips, median {med:.3f} s/char, {len(dur)} duration outliers, "
              f"STT gate {scored - len(gate)}/{scored} pass (CER <= {GATE_CER})")
    missing = {v: sorted(r["id"] for r in map(json.loads, (OUT / "manifest.jsonl").read_text(encoding="utf-8").splitlines())
                         if r["voice"] == v and not Path(r["path16k"]).exists()) for v in VOICES}
    print("missing:", {v: len(m) for v, m in missing.items()})
    (OUT / "qc_flags.json").write_text(json.dumps(flags, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> int:
    _selfcheck()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["build", "run", "qc"])
    ap.add_argument("--voice", choices=list(VOICES))
    ap.add_argument("--redo", type=Path, help="qc flags json: regenerate those ids once")
    ap.add_argument("--floor", type=int, default=1500, help="never let the org balance drop below this")
    ap.add_argument("--asr", type=Path, help="qc: SenseVoice ASR jsonl (fg_voice_e2e asr --arm svw8) for the STT gate")
    a = ap.parse_args()
    if a.stage == "build":
        build()
    elif a.stage == "qc":
        qc(a.asr)
    else:
        run(a.voice, a.redo, a.floor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
