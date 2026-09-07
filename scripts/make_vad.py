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

Speed
-----
90k clips is enough that the obvious one-clip-at-a-time loop costs hours of a
12 h Kaggle session before a single training step runs. Two things fix that, and
both matter more than any per-clip micro-optimisation:

*   **Silero runs batched, on the GPU.** Each window's forward pass is tiny, so
    the cost is almost all per-call overhead - and one clip at a time means 17 M
    of those calls. Clips are independent, so a batch of 256 pays each call once
    for 256 clips. Decoding runs on a thread pool ahead of the GPU (`soundfile`
    releases the GIL), so neither side waits on the other.
*   **The heuristic fallback is forked across cores and vectorised.** It was
    never the expensive path (~8 min for the corpus on one core), so the fork is
    most of its win; batching the per-frame `rfft` into one call over a strided
    view is worth a further 2x, and produces bit-identical numbers.

Clips are batched in duration order, so a batch pads to roughly its own clips'
length rather than to the longest clip in the corpus.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Silero's own get_speech_timestamps defaults, applied here at frame resolution.
THRESHOLD = 0.5
NEG_THRESHOLD = 0.35
MIN_SPEECH_S = 0.25
MIN_SILENCE_S = 0.10
SPEECH_PAD_S = 0.03


def load_silero():
    try:
        import torch
        model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad",
                                      trust_repo=True, onnx=False)
        return model, utils
    except Exception as e:                                            # noqa: BLE001
        print("[vad] silero unavailable (%s) - using the heuristic fallback" % e)
        return None, None


def read_audio(path: Path, sr: int) -> np.ndarray:
    import soundfile as sf
    y, in_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if in_sr != sr:
        import librosa
        y = librosa.resample(y, orig_sr=in_sr, target_sr=sr)
    return np.ascontiguousarray(y, dtype="float32")


# --------------------------------------------------------------------------
# heuristic fallback
# --------------------------------------------------------------------------
def heuristic_speech(y: np.ndarray, sr: int, fps: float) -> np.ndarray:
    """Energy x (1 - spectral flatness), smoothed. Crude but monotone with speech.

    Speech is high-energy *and* tonal; a fan or engine is high-energy and flat.
    The product separates them well enough for an auxiliary target.

    One strided view and one batched `rfft` over every frame at once. The frame
    grid, the window and the arithmetic are the same as in the per-frame loop
    this replaces; only the number of NumPy calls changes.
    """
    hop = max(1, int(sr / fps))
    win = hop * 2
    eps = 1e-10
    n = max(1, (len(y) - win) // hop + 1)
    need = win + (n - 1) * hop
    if len(y) < need:                       # short clip: pad the single frame out
        y = np.pad(y, (0, need - len(y)))
    frames = np.lib.stride_tricks.sliding_window_view(y, win)[::hop][:n]

    spec = np.abs(np.fft.rfft(frames * np.hanning(win), axis=-1)) + eps
    gm = np.exp(np.mean(np.log(spec), axis=-1))
    am = np.mean(spec, axis=-1)
    flat = gm / (am + eps)
    energy = np.log(np.mean(frames ** 2, axis=-1) + eps)
    out = ((1.0 - flat) * energy).astype("float32")

    out = (out - out.min()) / (np.ptp(out) + eps)
    k = max(1, int(0.12 * fps))
    return np.convolve(out, np.ones(k) / k, mode="same").astype("float32")


_HEUR = {}


def _heur_init(data: str, out: str, sr: int, fps: float) -> None:
    # Each worker owns a core; NumPy's own threads would then oversubscribe it.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    _HEUR.update(data=Path(data), out=Path(out), sr=sr, fps=fps)


def _heur_one(job):
    uid, rel = job
    a = _HEUR
    try:
        y = read_audio(a["data"] / rel, a["sr"])
        v = heuristic_speech(y, a["sr"], a["fps"])
        np.save(a["out"] / (uid + ".npy"), v.astype("float32"))
        return uid, ""
    except Exception as e:                                            # noqa: BLE001
        return uid, "%s: %s" % (type(e).__name__, e)


def run_heuristic(jobs, data: Path, out: Path, sr: int, fps: float,
                  workers: int) -> int:
    """Fork over clips - the work is pure NumPy per clip and perfectly parallel."""
    done = failed = 0
    t0 = time.time()
    pool = None
    if workers <= 1:
        _heur_init(str(data), str(out), sr, fps)
        results = (_heur_one(j) for j in jobs)
    else:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=workers, initializer=_heur_init,
                                   initargs=(str(data), str(out), sr, fps))
        results = pool.map(_heur_one, jobs, chunksize=32)
    for uid, err in results:
        if err:
            failed += 1
            if failed <= 20:
                print("[vad] %s failed: %s" % (uid, err))
        else:
            done += 1
            if done % 5000 == 0:
                _progress(done, len(jobs), t0)
    if pool is not None:
        pool.shutdown()
    if failed:
        print("[vad] %d clips failed" % failed)
    return done


# --------------------------------------------------------------------------
# silero, batched
# --------------------------------------------------------------------------
class SileroBatcher:
    """Batched speech probabilities from the Silero JIT model.

    The model is a per-window recurrent net: it consumes fixed 512-sample windows
    (256 at 8 kHz) and carries state between them. That state has a batch
    dimension, so a whole batch of clips streams through together and each window
    costs one call for 256 clips instead of one call each.

    `audio_forward` runs that window loop inside TorchScript where the build has
    it; the explicit loop below is the same computation with the loop in Python,
    and is used when it does not.
    """

    def __init__(self, model, device, sr: int):
        import torch
        self.torch = torch
        self.model = model.to(device).eval()
        self.device = device
        self.sr = int(sr)
        self.win = 512 if self.sr >= 16000 else 256
        self.use_audio_forward = hasattr(self.model, "audio_forward")

    def _reset(self) -> None:
        # State is sized by the previous batch, so it must be dropped before a
        # batch of a different size - the last batch of a run is almost always
        # short.
        try:
            self.model.reset_states()
        except Exception:                                             # noqa: BLE001
            pass

    def probs(self, x):
        """(B, T) waveform tensor -> (B, n_windows) speech probability."""
        torch = self.torch
        b, t = x.shape
        pad = (-t) % self.win
        if pad:
            x = torch.nn.functional.pad(x, (0, pad))
        with torch.no_grad():
            if self.use_audio_forward:
                self._reset()
                try:
                    p = self.model.audio_forward(x, self.sr)
                    return p.reshape(b, -1).float().cpu().numpy()
                except Exception as e:                                # noqa: BLE001
                    print("[vad] audio_forward unusable (%s: %s) - windowing in "
                          "Python instead" % (type(e).__name__, str(e)[:120]),
                          flush=True)
                    self.use_audio_forward = False
            self._reset()
            out = []
            for i in range(0, x.shape[1], self.win):
                p = self.model(x[:, i:i + self.win], self.sr)
                out.append(p.reshape(b))
            return torch.stack(out, dim=1).float().cpu().numpy()


def probs_to_mask(p: np.ndarray, n: int, win_s: float, fps: float) -> np.ndarray:
    """Window probabilities -> a 0/1 speech mask on the `fps` frame grid.

    This is `get_speech_timestamps`' hysteresis, minimum-duration and padding
    rules applied at frame resolution, so the labels match what the
    one-clip-at-a-time path used to write rather than being raw probabilities
    with quite different statistics.
    """
    if n <= 0:
        return np.zeros((0,), "float32")
    if len(p) == 0:
        return np.zeros((n,), "float32")
    # Windows are 32 ms and frames are 40 ms, so the window grid is the finer of
    # the two: sampling it at frame centres would drop every fifth window, and a
    # short burst with it. `reduceat` partitions the windows across frames
    # instead, so each frame gets the max over the windows that start inside it
    # and no window is ignored.
    a = np.minimum((np.arange(n) / fps / win_s).astype(np.int64), len(p) - 1)
    q = np.maximum.reduceat(p, a)

    min_speech = max(1, int(round(MIN_SPEECH_S * fps)))
    min_sil = max(1, int(round(MIN_SILENCE_S * fps)))
    pad = int(round(SPEECH_PAD_S * fps))

    mask = np.zeros((n,), "float32")
    triggered = False
    start = 0
    temp_end = 0
    for i in range(n):
        v = q[i]
        if v >= THRESHOLD and temp_end:
            temp_end = 0
        if v >= THRESHOLD and not triggered:
            triggered, start = True, i
            continue
        if v < NEG_THRESHOLD and triggered:
            if not temp_end:
                temp_end = i
            if i - temp_end < min_sil:
                continue
            if temp_end - start >= min_speech:
                mask[max(0, start - pad):min(n, temp_end + pad)] = 1.0
            triggered, temp_end = False, 0
    if triggered and n - start >= min_speech:
        mask[max(0, start - pad):n] = 1.0
    return mask


def batching_works(bat) -> bool:
    """Prove the batched path on two dummy clips before betting the corpus on it.

    Batched input is not something Silero's README promises, and the hub pulls
    whatever the repo's master currently is - so a build that only accepts one
    clip at a time is a thing that can happen on some future Tuesday. One probe
    costs milliseconds and turns that into the slow-but-correct path below
    instead of a crash 40,000 clips in.
    """
    import torch
    try:
        p = bat.probs(torch.zeros((2, bat.win * 4), dtype=torch.float32,
                                  device=bat.device))
        ok = p.ndim == 2 and p.shape[0] == 2 and p.shape[1] == 4
        if not ok:
            print("[vad] batched probe returned %s, expected (2, 4)" % (p.shape,))
        return ok
    except Exception as e:                                            # noqa: BLE001
        print("[vad] batched silero unavailable (%s: %s)"
              % (type(e).__name__, str(e)[:160]))
        return False


def silero_speech(model, utils, y: np.ndarray, sr: int, fps: float) -> np.ndarray:
    """One clip at a time, through Silero's own helper. The slow path, kept as
    the fallback for when the batched one does not load."""
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


def split_budget(items, max_samples: int):
    """Group length-sorted (uid, waveform) pairs into padded batches of bounded size.

    A batch is padded to its longest clip, so a fixed clip count would allocate
    `count x longest` - fine for the 6 s median and several GB for a batch that
    happens to contain a long recording. Capping the padded sample count instead
    keeps every batch the same size in memory whatever the durations are.
    """
    cur, cur_max = [], 0
    for uid, y in items:
        m = max(cur_max, len(y))
        if cur and (len(cur) + 1) * m > max_samples:
            yield cur
            cur, cur_max = [], 0
            m = len(y)
        cur.append((uid, y))
        cur_max = m
    if cur:
        yield cur


class Decoder:
    """Decodes chunks of clips one chunk ahead of whoever consumes them.

    `soundfile` releases the GIL, so threads are enough and nothing has to be
    pickled back the way it would with processes. The lookahead submit is what
    makes decode and inference overlap - waiting for a whole chunk before
    starting the next one's reads leaves the GPU idle for exactly as long as
    decoding takes.
    """

    def __init__(self, jobs, data: Path, sr: int, chunk: int, workers: int):
        self.chunks = [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
        self.data, self.sr = data, sr
        self.workers = workers
        self.failed = 0

    def _one(self, job):
        uid, rel = job
        try:
            return uid, read_audio(self.data / rel, self.sr), ""
        except Exception as e:                                        # noqa: BLE001
            return uid, None, "%s: %s" % (type(e).__name__, e)

    def __iter__(self):
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            ahead = ([pool.submit(self._one, j) for j in self.chunks[0]]
                     if self.chunks else [])
            for ci in range(len(self.chunks)):
                cur = ahead
                ahead = ([pool.submit(self._one, j) for j in self.chunks[ci + 1]]
                         if ci + 1 < len(self.chunks) else [])
                good = []
                for fut in cur:
                    uid, y, err = fut.result()
                    if err or y is None or len(y) == 0:
                        self.failed += 1
                        if self.failed <= 20:
                            print("[vad] %s failed: %s" % (uid, err or "empty audio"))
                        continue
                    good.append((uid, y))
                yield ci, good


def run_silero(bat, jobs, data: Path, out: Path, sr: int, fps: float,
               batch: int, workers: int, max_samples: int) -> int:
    import torch

    win_s = bat.win / float(sr)
    done = 0
    t0 = time.time()
    dec = Decoder(jobs, data, sr, batch, workers)

    for ci, good in dec:
        good.sort(key=lambda kv: len(kv[1]))
        for sub in split_budget(good, max_samples):
            t_max = max(len(y) for _, y in sub)
            buf = np.zeros((len(sub), t_max), "float32")
            for j, (_, y) in enumerate(sub):
                buf[j, :len(y)] = y
            probs = bat.probs(torch.from_numpy(buf).to(bat.device))

            for j, (uid, y) in enumerate(sub):
                n = int(np.ceil(len(y) / sr * fps))
                nw = max(1, int(np.ceil(len(y) / bat.win)))
                v = probs_to_mask(probs[j, :nw], n, win_s, fps)
                np.save(out / (uid + ".npy"), v.astype("float32"))
                done += 1
        if ci % 20 == 0:
            _progress(done, len(jobs), t0)
    if dec.failed:
        print("[vad] %d clips failed" % dec.failed)
    return done


def run_silero_serial(model, utils, jobs, data: Path, out: Path, sr: int,
                      fps: float, workers: int) -> int:
    """One clip at a time - only reached when the batched probe fails."""
    done = 0
    t0 = time.time()
    dec = Decoder(jobs, data, sr, 64, workers)
    for ci, good in dec:
        for uid, y in good:
            try:
                v = silero_speech(model, utils, y, sr, fps)
                np.save(out / (uid + ".npy"), v.astype("float32"))
                done += 1
            except Exception as e:                                    # noqa: BLE001
                dec.failed += 1
                if dec.failed <= 20:
                    print("[vad] %s failed: %s" % (uid, e))
        if ci % 20 == 0:
            _progress(done, len(jobs), t0)
    if dec.failed:
        print("[vad] %d clips failed" % dec.failed)
    return done


def _progress(done: int, total: int, t0: float) -> None:
    dt = max(1e-6, time.time() - t0)
    rate = done / dt
    eta = (total - done) / rate if rate else 0.0
    print("[vad] %d/%d clips | %.0f clips/s | eta %.1f min"
          % (done, total, rate, eta / 60.0), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir holding manifest.jsonl + audio/")
    ap.add_argument("--out", default="", help="default <data>/vad")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256,
                    help="clips per Silero forward; the whole speed-up lives here")
    ap.add_argument("--workers", type=int, default=0,
                    help="decode threads (silero) / processes (heuristic); 0 = auto")
    ap.add_argument("--device", default="",
                    help="cuda / cpu; defaults to cuda when one is visible")
    ap.add_argument("--max-samples", type=int, default=64_000_000,
                    help="cap on a padded batch's samples (~256 MB at float32)")
    ap.add_argument("--no-silero", action="store_true",
                    help="skip the download and go straight to the heuristic")
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out or data / "vad")
    out.mkdir(parents=True, exist_ok=True)

    recs = [json.loads(l) for l in (data / "manifest.jsonl").open(encoding="utf-8") if l.strip()]
    if args.limit:
        recs = recs[:args.limit]

    # Resume is one directory listing and a set membership test, not 90k
    # separate exists() calls.
    have = {p.stem for p in out.glob("*.npy")}
    todo = [r for r in recs if r["uid"] not in have]
    print("[vad] %d clips | %d already done | %d to do"
          % (len(recs), len(recs) - len(todo), len(todo)), flush=True)
    if not todo:
        print("[vad] nothing to do")
        return

    # Duration order keeps a batch's padding close to its own clips' length.
    todo.sort(key=lambda r: float(r.get("duration") or 0.0))
    jobs = [(r["uid"], r["path"]) for r in todo]

    cpus = os.cpu_count() or 2
    model, utils = (None, None) if args.no_silero else load_silero()

    t0 = time.time()
    if model is not None:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        workers = args.workers or min(8, max(2, cpus))
        bat = SileroBatcher(model, torch.device(device), args.sr)
        if batching_works(bat):
            print("[vad] silero on %s | batch %d | %d decode threads"
                  % (device, args.batch, workers), flush=True)
            done = run_silero(bat, jobs, data, out, args.sr, args.fps,
                              args.batch, workers, args.max_samples)
        else:
            print("[vad] falling back to one clip at a time - this is the slow "
                  "path (hours for the full corpus)", flush=True)
            # The batcher moved the model to the GPU; the serial helper feeds it
            # CPU tensors, so put it back rather than trip over the mismatch.
            model.to("cpu")
            done = run_silero_serial(model, utils, jobs, data, out, args.sr,
                                     args.fps, workers)
    else:
        workers = args.workers or max(1, cpus)
        print("[vad] heuristic on %d processes" % workers, flush=True)
        done = run_heuristic(jobs, data, out, args.sr, args.fps, workers)

    print("[vad] wrote %d files to %s in %.1f min" % (done, out, (time.time() - t0) / 60.0))
    print("[vad] set `data.vad_dir: %s` in your config to enable the speech head" % out)


if __name__ == "__main__":
    main()
