"""Log-mel L1 distance between WAVs — a quality check ASR cannot give.

Round-trip transcription proves the words are intelligible, but it is insensitive
to buzz, roughness and over-smoothing: a degraded voice still transcribes. This
measures how far the audio itself moved from a reference rendering, which is what
catches quality loss from dropping flow-matching steps or from quantization.

Reports L1 over log-mel frames, plus the same figure per second of audio, and the
correlation of the mel envelopes. Files of different lengths are compared over
their common prefix.

Usage:
  python spectral_distance.py --reference ref.wav candidate1.wav candidate2.wav ...
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf

SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 80


def hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64)
    return np.where(f >= 1000.0,
                    15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0),
                    3.0 * f / 200.0)


def mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    return np.where(m >= 15.0,
                    1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)),
                    200.0 * m / 3.0)


def filterbank():
    n_freqs = N_FFT // 2 + 1
    freqs = np.linspace(0, SR / 2, n_freqs)
    points = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(SR / 2), N_MELS + 2))
    bank = np.zeros((N_MELS, n_freqs))
    for i in range(N_MELS):
        lower, center, upper = points[i], points[i + 1], points[i + 2]
        left = (freqs - lower) / max(center - lower, 1e-10)
        right = (upper - freqs) / max(upper - center, 1e-10)
        bank[i] = np.maximum(0.0, np.minimum(left, right)) * (2.0 / (upper - lower))
    return bank.astype(np.float32)


def log_mel(path):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        old = np.linspace(0, 1, len(data))
        new = np.linspace(0, 1, int(len(data) * SR / sr))
        data = np.interp(new, old, data).astype(np.float32)
    if data.size < N_FFT:
        return None, 0.0
    window = np.hanning(N_FFT + 1)[:-1]
    frames = np.lib.stride_tricks.sliding_window_view(data, N_FFT)[::HOP]
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    mel = filterbank() @ spectrum.T.astype(np.float32)
    return np.log10(np.maximum(mel, 1e-10)), data.size / SR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("candidates", nargs="+")
    parser.add_argument("--report")
    args = parser.parse_args()

    reference, ref_seconds = log_mel(args.reference)
    if reference is None:
        print(f"reference too short: {args.reference}", file=sys.stderr)
        return 2

    rows = []
    for path in args.candidates:
        if not os.path.isfile(path):
            print(f"MISSING  {path}", file=sys.stderr)
            continue
        candidate, seconds = log_mel(path)
        if candidate is None:
            print(f"SKIP     {os.path.basename(path)} (too short)")
            continue
        frames = min(reference.shape[1], candidate.shape[1])
        a, b = reference[:, :frames], candidate[:, :frames]
        l1 = float(np.abs(a - b).mean())
        correlation = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        rows.append({"file": os.path.basename(path), "seconds": round(seconds, 2),
                     "frames_compared": int(frames), "logmel_l1": round(l1, 4),
                     "mel_corr": round(correlation, 6),
                     "length_delta_s": round(seconds - ref_seconds, 2)})
        print(f"  {os.path.basename(path):<28} logmel L1 {l1:7.4f}  "
              f"mel corr {correlation:8.5f}  {seconds:5.2f}s "
              f"({seconds - ref_seconds:+.2f}s)")

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump({"reference": os.path.basename(args.reference),
                       "reference_seconds": round(ref_seconds, 2),
                       "candidates": rows}, handle, ensure_ascii=False, indent=2)
        print(f"\nreport: {args.report}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
