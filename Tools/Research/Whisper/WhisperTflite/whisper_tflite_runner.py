"""Run a Whisper tflite export on Windows CPU through ai_edge_litert.

This is the desktop counterpart of the Whisper driver that otherwise lives only
inside the Android AAR: mel frontend -> encoder -> greedy KV decode. It exists
so the "can Whisper tflite run on Windows?" question stays answered and
re-runnable instead of being re-derived every session.

Window-size and mel-count aware: `n_mels` and the encoder mel-frame window are
read from the `encode` signature, so the same script drives 30 s (3000-frame)
and short-window (500/1000-frame) exports, and 80-mel as well as 128-mel
(large-v3 / turbo) checkpoints.

Pipeline mirrors the Unity JNI one: audio -> 16 kHz mono -> slaney log-mel
(zero-padded/truncated to the window) -> encode -> greedy decode with a forced
[SOT, lang, transcribe, notimestamps] prefix.

Emits one JSON line: {model, audio, lang, frames, n_mels, audio_s, text,
                      encode_s (median of --runs), decode_s, steps}

Usage:
  python whisper_tflite_runner.py --model M.tflite --tokenizer T.json \
      --audio A.wav [--lang ko] [--runs 3]

Prefer the PowerShell wrapper `Tools/Windows/Run-WhisperTfliteWindows.ps1`,
which resolves the interpreter and checks dependencies first.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
from ai_edge_litert.interpreter import Interpreter
from tokenizers import Tokenizer

SR = 16000
N_FFT = 400
HOP = 160
SEQ = 128


def hz_to_mel_slaney(f):
    f = np.asarray(f, dtype=np.float64)
    m = 3.0 * f / 200.0
    return np.where(f >= 1000.0,
                    15.0 + np.log(np.maximum(f, 1e-10) / 1000.0)
                    / (np.log(6.4) / 27.0), m)


def mel_to_hz_slaney(m):
    m = np.asarray(m, dtype=np.float64)
    f = 200.0 * m / 3.0
    return np.where(m >= 15.0,
                    1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)), f)


def mel_filterbank(n_mels):
    n_freqs = N_FFT // 2 + 1
    fft_freqs = np.linspace(0, SR / 2, n_freqs)
    mel_pts = np.linspace(hz_to_mel_slaney(0.0), hz_to_mel_slaney(SR / 2),
                          n_mels + 2)
    hz_pts = mel_to_hz_slaney(mel_pts)
    fb = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        lower, center, upper = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        left = (fft_freqs - lower) / max(center - lower, 1e-10)
        right = (upper - fft_freqs) / max(upper - center, 1e-10)
        fb[i] = np.maximum(0.0, np.minimum(left, right)) * (2.0 / (upper - lower))
    return fb.astype(np.float32)


def log_mel(pcm, n_mels, n_frames):
    n_samples = n_frames * HOP
    if len(pcm) > n_samples:
        pcm = pcm[:n_samples]
    pcm = np.pad(pcm, (0, n_samples - len(pcm)))
    padded = np.pad(pcm, (N_FFT // 2, N_FFT // 2), mode="reflect")
    window = np.hanning(N_FFT + 1)[:-1].astype(np.float64)
    n_frames_total = 1 + (len(padded) - N_FFT) // HOP
    frames = np.lib.stride_tricks.sliding_window_view(
        padded, N_FFT)[::HOP][:n_frames_total]
    stft = np.fft.rfft(frames * window, axis=1)
    power = (np.abs(stft) ** 2).T
    power = power[:, :n_frames]
    mel = mel_filterbank(n_mels) @ power.astype(np.float32)
    logspec = np.log10(np.maximum(mel, 1e-10))
    logspec = np.maximum(logspec, logspec.max() - 8.0)
    logspec = (logspec + 4.0) / 4.0
    return logspec[np.newaxis, :, :].astype(np.float32)


def load_audio(path):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        x_old = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, int(len(data) * SR / sr))
        data = np.interp(x_new, x_old, data).astype(np.float32)
    return data


def forced_prefix(vocab, lang):
    """Whisper prompt ids differ per checkpoint family."""
    if vocab == 51864:  # english-only (.en) checkpoints
        return [50257, 50362], 50256  # [SOT, <|notimestamps|>], EOT
    sot, eot = 50258, 50257
    # large-v3 / turbo (51866) shifted every special token by one.
    transcribe, nots = (50360, 50364) if vocab >= 51866 else (50359, 50363)
    lang_id = {"ko": 50264, "en": 50259}[lang]
    return [sot, lang_id, transcribe, nots], eot


def load_hotwords(path):
    """WebServer rule (server/transcription.py): non-'#' lines, stripped, space-joined."""
    with open(path, encoding="utf-8") as f:
        return " ".join(l.strip() for l in f.read().splitlines()
                        if l.strip() and not l.lstrip().startswith("#"))


def hotword_prefix(tok, hotwords, budget):
    """faster-whisper get_prompt(): <|startofprev|> + encode(" " + hotwords),
    placed before the SOT sequence; over-long lists keep their first tokens."""
    ids = tok.encode(" " + hotwords.strip(), add_special_tokens=False).ids
    return [tok.token_to_id("<|startofprev|>")] + ids[:max(0, budget)]


def load_model(model_path):
    interp = Interpreter(model_path=model_path)
    encode = interp.get_signature_runner("encode")
    decode = interp.get_signature_runner("decode")
    m = {"encode": encode, "decode": decode}
    for name, det in encode.get_input_details().items():
        shape = det["shape"]
        if len(shape) == 3:
            m.update(feat=name, n_mels=int(shape[1]), n_frames=int(shape[2]))
    assert "feat" in m, encode.get_input_details()
    m["vocab"] = int(list(decode.get_output_details().values())[0]["shape"][-1])
    # Decode inputs are bound by shape/dtype, not by name: exports disagree on
    # naming and on argument order (turbo swaps two of them).
    for name, det in decode.get_input_details().items():
        shape, dtype = det["shape"], det["dtype"]
        if len(shape) == 3:
            m["audio"] = name
        elif len(shape) == 4:
            m["mask"] = name
        elif dtype == np.int32:
            m["tokens"] = name
    return m


def transcribe(m, data, prompt, eot, runs=1):
    """Returns (output ids, median encode s, decode s)."""
    feats = log_mel(data, m["n_mels"], m["n_frames"])
    enc_times = []
    for _ in range(max(1, runs)):
        t = time.perf_counter()
        enc_out = m["encode"](**{m["feat"]: feats})
        enc_times.append(time.perf_counter() - t)
    audio_embed = list(enc_out.values())[0]

    mask = np.triu(np.full((SEQ, SEQ), -1e9, dtype=np.float32), k=1)
    mask = mask[np.newaxis, np.newaxis]
    tokens = np.full((1, SEQ), eot, dtype=np.int32)
    tokens[0, :len(prompt)] = prompt

    out_ids = []
    t_dec = time.perf_counter()
    pos = len(prompt)
    while pos < SEQ:
        logits = m["decode"](**{m["audio"]: audio_embed,
                                m["tokens"]: tokens,
                                m["mask"]: mask})
        logits = list(logits.values())[0]
        nxt = int(np.argmax(logits[0, pos - 1]))
        if nxt == eot:
            break
        tokens[0, pos] = nxt
        out_ids.append(nxt)
        pos += 1
    return out_ids, float(np.median(enc_times)), time.perf_counter() - t_dec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--lang", default="ko", choices=["ko", "en"])
    parser.add_argument("--runs", type=int, default=3,
                        help="encoder timing repeats; the median is reported")
    parser.add_argument("--hotwords-file", default="",
                        help="faster-whisper style hotwords (WebServer WhisperHotwords.txt); off by default")
    parser.add_argument("--max-new-tokens", type=int, default=32,
                        help="decoder positions kept free for output when a hotword prefix is used")
    args = parser.parse_args()

    for label, path in (("model", args.model), ("tokenizer", args.tokenizer),
                        ("audio", args.audio)):
        if not os.path.isfile(path):
            print(f"{label} not found: {path}", file=sys.stderr)
            return 2

    data = load_audio(args.audio)
    try:
        m = load_model(args.model)
    except Exception as exc:  # encoder-only or non-whisper export
        print(f"not a whisper encode/decode export: {exc}", file=sys.stderr)
        return 3
    tok = Tokenizer.from_file(args.tokenizer)
    prompt, eot = forced_prefix(m["vocab"], args.lang)
    if args.hotwords_file:
        # ponytail: this export's decoder is a fixed 128-token window (faster-whisper
        # has 448), so a long list is cut to what fits; re-export with a longer
        # decoder window if the whole list must be used.
        budget = SEQ - len(prompt) - 1 - args.max_new_tokens
        prompt = hotword_prefix(tok, load_hotwords(args.hotwords_file), budget) + prompt

    out_ids, encode_s, decode_s = transcribe(m, data, prompt, eot, args.runs)
    print(json.dumps({
        "model": os.path.basename(args.model),
        "audio": os.path.basename(args.audio),
        "lang": args.lang, "frames": m["n_frames"], "n_mels": m["n_mels"],
        "vocab": m["vocab"], "audio_s": round(len(data) / SR, 2),
        "text": tok.decode(out_ids),
        "encode_s": round(encode_s, 3), "decode_s": round(decode_s, 3),
        "steps": len(out_ids), "prompt_tokens": len(prompt),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
