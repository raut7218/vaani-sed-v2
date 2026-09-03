"""Turn the bronze tier's tags into strong labels by construction.

The unused 32 hours
-------------------
Bronze clips carry a class tag and no timestamps, so v1 could only feed them
through attention pooling - they never supervised a boundary. v1's README listed
self-training on that tier as "step 4" and it was never done.

There is a better move than self-training, and it needs no model at all. If you
*cut* a segment out of a bronze clip tagged `animal_sound` and paste it into a
host clip at a known position, you have manufactured a strong label: you know the
class (the donor's tag) and you know the boundaries exactly (you chose them).
The label is correct by construction rather than by confidence threshold, so
there is no error to accumulate the way self-training does.

Two details make the labels honest rather than convenient:

* The pasted segment is energy-gated, so a donor's silence does not become an
  "event" with crisp edges the model then learns to hallucinate.
* Hosts are drawn from clips with no annotated events in the pasted region, so a
  synthetic span never overlaps a real one it does not know about.

Output is a normal manifest that `train.py` reads alongside the real one.

    python scripts/make_synthetic.py --data data/vaani --out data/vaani_synth -n 20000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.labels import LabelEncoder                              # noqa: E402


def load_manifest(p: Path):
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def read_audio(p: Path, sr: int):
    import soundfile as sf
    y, s = sf.read(str(p), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if s != sr:
        import librosa
        y = librosa.resample(y, orig_sr=s, target_sr=sr)
    return y.astype("float32")


def energetic_window(y: np.ndarray, sr: int, lo: float, hi: float, rng) -> tuple:
    """Pick a sub-window whose RMS is above the clip's median. Returns (a, b) in samples.

    Without this gate, roughly a third of donated segments are near-silence and
    the model gets taught that an event boundary can sit in silence - which is
    the fastest way to reproduce v1's over-dilated spans.
    """
    n = len(y)
    dur = rng.uniform(lo, hi)
    w = int(min(n, dur * sr))
    if w < int(0.05 * sr) or n <= w:
        return 0, n
    hop = max(1, w // 4)
    starts = list(range(0, n - w + 1, hop))
    rms = np.array([float(np.sqrt(np.mean(y[s:s + w] ** 2) + 1e-12)) for s in starts])
    med = float(np.median(rms))
    good = [s for s, r in zip(starts, rms) if r >= med]
    a = rng.choice(good) if good else rng.choice(starts)
    return a, a + w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", "--num", type=int, default=20000)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--clip-len", type=float, default=8.0)
    ap.add_argument("--events-per-clip", type=int, default=2,
                    help="max synthetic events pasted into one host")
    ap.add_argument("--snr-db", type=float, nargs=2, default=(0.0, 15.0))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import soundfile as sf
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    data, out = Path(args.data), Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)

    le = LabelEncoder(expand_vehicle=True)
    recs = load_manifest(data / "manifest.jsonl")

    # Donors: bronze clips whose tag set resolves to exactly one class, so the
    # pasted segment's label is unambiguous.
    donors = []
    for r in recs:
        if r.get("events"):
            continue
        ids = le.encode_clip_categories(r.get("clip_labels"))
        if len(ids) == 1:
            donors.append((r, le.classes[ids[0]]))
    # Hosts: clips with no events at all, so nothing unlabelled is underneath.
    hosts = [r for r in recs if not r.get("events") and not r.get("clip_labels")]
    if not hosts:
        hosts = [r for r in recs if not r.get("events")]

    print("[synth] %d donors, %d hosts" % (len(donors), len(hosts)))
    if not donors or not hosts:
        raise SystemExit("need both bronze donors and hosts; is the manifest complete?")

    n_samp = int(args.clip_len * args.sr)
    written, man = 0, (out / "manifest.jsonl").open("w", encoding="utf-8")

    for i in range(args.num):
        host = rng.choice(hosts)
        try:
            hy = read_audio(data / host["path"], args.sr)
        except Exception:                                             # noqa: BLE001
            continue
        buf = np.zeros((n_samp,), "float32")
        m = min(len(hy), n_samp)
        buf[:m] = hy[:m]
        host_rms = float(np.sqrt(np.mean(buf[:m] ** 2) + 1e-12))

        events, occupied = [], []
        for _ in range(rng.randint(1, args.events_per_clip)):
            drec, dcls = rng.choice(donors)
            try:
                dy = read_audio(data / drec["path"], args.sr)
            except Exception:                                         # noqa: BLE001
                continue
            a, b = energetic_window(dy, args.sr, 0.15, 3.0, rng)
            seg = dy[a:b]
            if len(seg) < int(0.05 * args.sr):
                continue
            # Place it somewhere that is still free.
            for _try in range(8):
                t0 = rng.uniform(0.0, max(0.0, args.clip_len - len(seg) / args.sr))
                t1 = t0 + len(seg) / args.sr
                if all(t1 <= o0 or t0 >= o1 for o0, o1 in occupied):
                    break
            else:
                continue
            s0 = int(t0 * args.sr)
            seg_rms = float(np.sqrt(np.mean(seg ** 2) + 1e-12))
            snr = float(np_rng.uniform(*args.snr_db))
            gain = host_rms / max(seg_rms, 1e-9) * (10.0 ** (snr / 20.0))
            # Short raised-cosine ramps: a hard splice leaves a click, and a
            # click is a boundary cue that will not exist at test time.
            ramp = max(2, int(0.008 * args.sr))
            w = np.ones(len(seg), "float32")
            w[:ramp] = np.hanning(2 * ramp)[:ramp]
            w[-ramp:] = np.hanning(2 * ramp)[ramp:]
            buf[s0:s0 + len(seg)] += (seg * w * gain).astype("float32")
            occupied.append((t0, t1))
            events.append({"cls": dcls, "start": round(t0, 4), "end": round(t1, 4)})

        if not events:
            continue
        peak = float(np.abs(buf).max())
        if peak > 1.0:
            buf = buf / peak * 0.98

        uid = "synth_%06d" % i
        sf.write(str(out / "audio" / (uid + ".wav")), buf, args.sr)
        man.write(json.dumps({
            "uid": uid, "path": "audio/%s.wav" % uid,
            "duration": round(args.clip_len, 4),
            # Boundaries are exact by construction, so this is genuinely gold.
            "tier": "gold", "state": "SYNTH", "district": "SYNTH",
            "events": sorted(events, key=lambda e: e["start"]),
            "clip_labels": sorted({e["cls"] for e in events}),
        }) + "\n")
        written += 1
        if written % 1000 == 0:
            print("[synth] %d clips" % written, flush=True)

    man.close()
    print("[synth] wrote %d clips to %s" % (written, out))
    print("[synth] train on both:  --data %s --extra-data %s" % (args.data, args.out))


if __name__ == "__main__":
    main()
