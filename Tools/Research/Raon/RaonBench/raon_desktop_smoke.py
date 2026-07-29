"""Prove Raon-Speech runs on this workstation, and measure what it costs.

Raon is the only candidate this project has evaluated that does STT *and* TTS in
one model. It cannot run on the device — 9 B, CUDA-only AWQ kernels, no LiteRT
path — so the question it has to answer is the PC one: does it work here, how
fast, and is the Korean good enough to be worth a companion machine?

Two passes, both on the project's standard Korean assets so the numbers sit
beside the existing ASR and TTS tables:

  STT — transcribe the clips the whisper/ACFT matrix uses.
  TTS — speak the sentences the Supertonic device test speaks, and keep the WAVs.

Everything is written to a JSONL record and the audio to a run directory, so a
later listen does not require re-running a 9 B model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

MODEL_ID = "KRAFTON/Raon-Speech-9B-AWQ-INT4"

# Clips from Assets/StreamingAssets/TestAssets/Audio, with the reference text
# used elsewhere in docs/benchmarks/asr-model-matrix.md.
STT_CLIPS = [
    ("volume-소리 키워줘.mp3", "소리 키워줘"),
    ("volume-볼륨 업.mp3", "볼륨 업"),
    ("현재 서울의 날씨는 흐림 입니다.mp3", "현재 서울의 날씨는 흐림 입니다"),
    ("2025년 3월 5일 전술평가 결과 보고.mp3", "2025년 3월 5일 전술평가 결과 보고"),
]

# The same sentences the Supertonic smoke test speaks on device.
TTS_SENTENCES = [
    "고도 백 미터로 상승합니다.",
    "경고. 강풍이 감지되었습니다. 고도를 낮춥니다. 귀환을 시작합니다. 예상 소요 시간 삼 분.",
    "Altitude one hundred meters. Returning to base.",
]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein over characters, ignoring spaces and trailing punctuation.

    Rough on purpose — it is here to flag a broken transcription, not to rank
    models. The ASR matrix owns the real scoring.
    """
    def normalize(text: str) -> str:
        return "".join(c for c in text if not c.isspace() and c not in ".,!?。、")

    reference, hypothesis = normalize(reference), normalize(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0

    previous = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, start=1):
        current = [i]
        for j, h in enumerate(hypothesis, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (r != h)))
        previous = current
    return previous[-1] / len(reference)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--audio-root", default="Assets/StreamingAssets/TestAssets/Audio")
    parser.add_argument("--out-dir", default="Builds/Logs/RaonDesktop")
    parser.add_argument("--speaker-audio", default="",
                        help="Reference clip for voice-conditioned TTS (needs speechbrain).")
    parser.add_argument("--skip-stt", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available — this model has no CPU path worth measuring.")
    if args.device == "cuda":
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {name}, {total:.1f} GB", flush=True)

    print(f"Loading {args.model} …", flush=True)
    started = time.perf_counter()
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    RaonPipeline = get_class_from_dynamic_module(
        "modeling_raon.RaonPipeline", args.model,
        revision=getattr(config, "_commit_hash", None),
    )
    pipe = RaonPipeline(args.model, device=args.device, dtype=args.dtype)
    load_seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 1e9 if args.device == "cuda" else 0.0
    print(f"Loaded in {load_seconds:.1f}s · {peak:.2f} GB allocated", flush=True)

    if not args.skip_stt:
        print("\n=== STT ===", flush=True)
        for filename, reference in STT_CLIPS:
            path = Path(args.audio_root) / filename
            if not path.exists():
                print(f"  missing: {path}", flush=True)
                continue
            started = time.perf_counter()
            hypothesis = pipe.stt(str(path))
            elapsed = time.perf_counter() - started
            if not isinstance(hypothesis, str):
                hypothesis = str(hypothesis)
            cer = character_error_rate(reference, hypothesis)
            print(f"  {elapsed:5.2f}s CER {cer:.3f}  {hypothesis.strip()}", flush=True)
            records.append({
                "kind": "stt", "model": args.model, "clip": filename,
                "reference": reference, "hypothesis": hypothesis.strip(),
                "cer": round(cer, 4), "seconds": round(elapsed, 3),
            })

    if not args.skip_tts:
        print("\n=== TTS ===", flush=True)
        for index, sentence in enumerate(TTS_SENTENCES, start=1):
            started = time.perf_counter()
            kwargs = {"speaker_audio": args.speaker_audio} if args.speaker_audio else {}
            audio, sample_rate = pipe.tts(sentence, **kwargs)
            elapsed = time.perf_counter() - started
            wav_path = out_dir / f"tts-{index:02d}.wav"
            pipe.save_audio((audio, sample_rate), str(wav_path))

            # RTF needs the produced duration; derive it from the array length
            # rather than trusting a reported value.
            samples = getattr(audio, "shape", [len(audio)])[-1]
            duration = float(samples) / float(sample_rate)
            rtf = elapsed / duration if duration else float("nan")
            print(f"  {elapsed:5.2f}s → {duration:5.2f}s audio  RTF {rtf:.3f}  {wav_path.name}",
                  flush=True)
            print(f"     {sentence}", flush=True)
            records.append({
                "kind": "tts", "model": args.model, "index": index, "text": sentence,
                "wav": str(wav_path), "sampleRate": int(sample_rate),
                "audioSeconds": round(duration, 3), "seconds": round(elapsed, 3),
                "rtf": round(rtf, 4), "speakerAudio": args.speaker_audio,
            })

    records.append({
        "kind": "summary", "model": args.model, "device": args.device,
        "dtype": args.dtype, "loadSeconds": round(load_seconds, 2),
        "peakAllocatedGb": round(peak, 3),
        "peakReservedGb": round(torch.cuda.max_memory_reserved() / 1e9, 3)
        if args.device == "cuda" else 0.0,
    })

    jsonl = out_dir / "raon-desktop-smoke.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nWrote {jsonl}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
