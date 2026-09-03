"""Speech-presence pseudo-labels for the auxiliary head.

Vaani is conversational speech with noise events on top. Every DCASE-derived
system treats this audio as a generic soundscape; telling the model which energy
is speech lets it factor the mixture rather than infer the decomposition
implicitly, and boundaries are much easier to place once speech is accounted for.

The labels are free: any VAD produces them, and they are only ever an auxiliary
target - a wrong frame costs the model a little gradient, never a wrong event.

    python scripts/make_vad.py --data data/vaani --out data/vaani/vad

Uses Silero VAD when it can be fetched, and falls back to a spectral-flatness +
energy heuristic otherwise, so this never blocks a training run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_silero():
    try:
        import torch
        model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad",
                                      trust_repo=True, onnx=False)
        return model, utils
    except Exception as e:                                            # noqa: BLE001
        print("[vad] silero unavailable (%s) - using the heuristic fallback" % e)
        return None, None


def heuristic_speech(y: np.ndarray, sr: int, fps: float) -> np.ndarray:
    """Energy x (1 - spectral flatness), smoothed. Crude but monotone with speech.

    Speech is high-energy *and* tonal; a fan or engine is high-energy and flat.
    The product separates them well enough for an auxiliary target.
    """
    hop = max(1, int(sr / fps))
    win = hop * 2
    n = max(1, (len(y) - win) // hop + 1)
    out = np.zeros((n,), "float32")
    eps = 1e-10
    for i in range(n):
        seg = y[i * hop:i * hop + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) + eps
        gm = np.exp(np.mean(np.log(spec)))
        am = np.mean(spec)
        flat = gm / (am + eps)
        energy = np.log(np.mean(seg ** 2) + eps)
        out[i] = (1.0 - flat) * energy
    out = (out - out.min()) / (out.ptp() + eps)
    k = max(1, int(0.12 * fps))
    return np.convolve(out, np.ones(k) / k, mode="same").astype("float32")


def silero_speech(model, utils, y: np.ndarray, sr: int, fps: float) -> np.ndarray:
    import torch
    get_ts = utils[0]
    ts = get_ts(torch.from_numpy(y), model, sampling_rate=sr)
    n = int(np.ceil(len(y) / sr * fps))
    out = np.zeros((n,), "float32")
    for seg in ts:
        a = int(seg["start"] / sr * fps)
        b = min(n, int(seg["end"] / sr * fps) + 1)
        out[max(0, a):b] = 1.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir holding manifest.jsonl + audio/")
    ap.add_argument("--out", default="", help="default <data>/vad")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import soundfile as sf
    data = Path(args.data)
    out = Path(args.out or data / "vad")
    out.mkdir(parents=True, exist_ok=True)

    recs = [json.loads(l) for l in (data / "manifest.jsonl").open(encoding="utf-8") if l.strip()]
    if args.limit:
        recs = recs[:args.limit]
    model, utils = load_silero()

    done = 0
    for r in recs:
        dst = out / (r["uid"] + ".npy")
        if dst.exists():
            continue
        try:
            y, sr = sf.read(str(data / r["path"]), dtype="float32", always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != args.sr:
                import librosa
                y = librosa.resample(y, orig_sr=sr, target_sr=args.sr)
            v = (silero_speech(model, utils, y, args.sr, args.fps) if model is not None
                 else heuristic_speech(y, args.sr, args.fps))
            np.save(dst, v.astype("float32"))
            done += 1
            if done % 2000 == 0:
                print("[vad] %d clips" % done, flush=True)
        except Exception as e:                                        # noqa: BLE001
            print("[vad] %s failed: %s" % (r["uid"], e))
    print("[vad] wrote %d files to %s" % (done, out))
    print("[vad] set `data.vad_dir: %s` in your config to enable the speech head" % out)


if __name__ == "__main__":
    main()
