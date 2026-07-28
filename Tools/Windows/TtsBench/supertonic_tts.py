"""Synthesize speech with the Supertonic ONNX model through sherpa-onnx.

Desktop driver for the TTS engine evaluation: it proves the model, the language
coverage and the speed before any Unity or Android work, and it is the reference
the LiteRT conversion has to match.

The model directory is the sherpa-onnx Supertonic package layout:
  duration_predictor.int8.onnx  text_encoder.int8.onnx
  vector_estimator.int8.onnx    vocoder.int8.onnx
  tts.json                      unicode_indexer.bin       voice.bin

Emits one JSON line: {text, lang, wav, seconds, audio_s, rtf, sample_rate,
                      threads, speed}

Usage:
  python supertonic_tts.py --model-dir <dir> --text "..." --out out.wav
Prefer the wrapper Tools/Windows/Run-SupertonicTts.ps1, which resolves the
interpreter and can round-trip the result through Whisper.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import sherpa_onnx


def build_tts(model_dir, threads, provider, debug=False):
    def need(name):
        path = os.path.join(model_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return path

    supertonic = sherpa_onnx.OfflineTtsSupertonicModelConfig(
        duration_predictor=need("duration_predictor.int8.onnx"),
        text_encoder=need("text_encoder.int8.onnx"),
        vector_estimator=need("vector_estimator.int8.onnx"),
        vocoder=need("vocoder.int8.onnx"),
        tts_json=need("tts.json"),
        unicode_indexer=need("unicode_indexer.bin"),
        voice_style=need("voice.bin"),
    )
    model = sherpa_onnx.OfflineTtsModelConfig(
        supertonic=supertonic,
        num_threads=threads,
        provider=provider,
        debug=debug,
    )
    config = sherpa_onnx.OfflineTtsConfig(model=model, max_num_sentences=1)
    if not config.validate():
        raise ValueError("sherpa-onnx rejected the Supertonic configuration")
    return sherpa_onnx.OfflineTts(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lang", default="ko", help="reported only; the model is language-agnostic at the API level")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--sid", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--provider", default="cpu")
    parser.add_argument("--warmup", action="store_true",
                        help="synthesize once before timing, so the reported RTF excludes first-run cost")
    args = parser.parse_args()

    tts = build_tts(args.model_dir, args.threads, args.provider)

    if args.warmup:
        tts.generate("워밍업", sid=args.sid, speed=args.speed)

    started = time.perf_counter()
    audio = tts.generate(args.text, sid=args.sid, speed=args.speed)
    seconds = time.perf_counter() - started

    samples = np.asarray(audio.samples, dtype=np.float32)
    if samples.size == 0:
        print("synthesis produced no samples", file=sys.stderr)
        return 3

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    sf.write(args.out, samples, audio.sample_rate)

    audio_s = samples.size / float(audio.sample_rate)
    print(json.dumps({
        "text": args.text,
        "lang": args.lang,
        "wav": os.path.abspath(args.out),
        "seconds": round(seconds, 3),
        "audio_s": round(audio_s, 2),
        "rtf": round(seconds / audio_s, 4) if audio_s else None,
        "sample_rate": audio.sample_rate,
        "threads": args.threads,
        "speed": args.speed,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
