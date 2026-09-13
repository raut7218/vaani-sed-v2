"""Training stream: corpus gold/silver + validation-fold natural + synthetic re-mix.

The synthetic re-mixer
----------------------
Every synthetic validation clip ships with its clean reference, and
noisy - clean recovers the inserted noise exactly (100% of its energy lies
inside the labelled spans, labels accurate to ~1 ms). So the validation
folds used for training yield a bank of real noise snippets, each with its
class, and a bank of clean speech hosts. Pasting snippets onto hosts at the
measured statistics produces unlimited, exactly-labelled clips from the same
generator as the 4 h synthetic third of the test set. The held-out fold's
snippets and hosts never enter the bank.
"""
from __future__ import annotations

import io
import json
import math
import os
import random
import zlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

SR = 16000
FPS = 50


def val_fold(uid: str, k: int = 5) -> int:
    return zlib.crc32(uid.encode()) % k


def _read(path_or_blob) -> np.ndarray:
    import soundfile as sf
    y, sr = sf.read(path_or_blob, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(1)
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def _i16(y):
    return np.clip(y * 32767.0, -32768, 32767).astype(np.int16)


class ValBank:
    """Validation audio split by fold: natural clips, noise snippets, clean hosts."""

    def __init__(self, meta_path: str, le, hold_fold: int = 0, load_heldout: bool = True):
        from src.data.labels import resolve_event_class
        self.natural, self.snippets, self.hosts, self.synth, self.heldout = [], [], [], [], []
        self.syn_rate = 0.0
        if not meta_path:              # no validation set: the corpus trains alone
            return
        vdir = Path(meta_path).parent
        meta = json.load(open(meta_path))
        for r in meta:
            u = r["segmentFileName"][:-4]
            syn = bool(r["syntheticData"])
            evs = []
            for e in r["NoiseSubCategoryTimeStamp"] or []:
                c = resolve_event_class(e.get("category", ""), e.get("tag", ""), le.expand_vehicle)
                s, t = float(e["start"]), float(e["end"])
                if t > s:
                    evs.append((s, t, le.idx.get(c, -1) if c else -1))
            evs.sort()
            sub = "syntheticNoiseAudio" if syn else "naturalNoisyAudio"
            held = val_fold(u) == hold_fold
            if held:
                if load_heldout:
                    self.heldout.append(dict(uid=u, syn=syn, path=str(vdir / sub / r["segmentFileName"]), events=evs))
                continue
            y = _read(str(vdir / sub / r["segmentFileName"]))
            if not syn:
                self.natural.append((_i16(y), evs))
                continue
            clean = _read(str(vdir / "syntheticCleanRefAudio" / r["segmentFileName"]))
            n = min(len(y), len(clean))
            noise = y[:n] - clean[:n]
            self.hosts.append(_i16(clean[:n]))
            self.synth.append((_i16(y), evs))
            for s, t, c in evs:
                a, b = int(s * SR), int(t * SR)
                if b - a > int(0.03 * SR):
                    self.snippets.append((_i16(noise[a:b]), c))
        self.syn_rate = sum(len(e) for _, e in self.synth) / max(1e-6, sum(len(y) / SR for y, _ in self.synth))


class TraceStream(IterableDataset):
    """Infinite stream of 8 s windows drawn from sources by quota."""

    def __init__(self, bank: ValBank, corpus: list, corpus_root: str, n_class: int, quotas: dict,
                 win: float = 8.0, seed: int = 0, silver_pres_w: float = 0.3):
        self.bank, self.corpus, self.root = bank, corpus, corpus_root
        self.gold = [r for r in corpus if r.get("tier") == "gold" and r.get("events")]
        self.silver = [r for r in corpus if r.get("tier") == "silver" and r.get("events")]
        self.C = n_class
        self.W = int(win * SR); self.T = int(win * FPS)
        self.seed = seed
        self.silver_pres_w = silver_pres_w
        src = {"gold": len(self.gold), "silver": len(self.silver), "valnat": len(bank.natural),
               "remix": len(bank.snippets), "valsyn": len(bank.synth)}
        self.quotas = {k: v for k, v in quotas.items() if v > 0 and src.get(k, 0) > 0}

    # ---- sources -----------------------------------------------------------
    def _corpus_wav(self, rec):
        if "pack" not in rec:          # scripts/download_data.py layout: one FLAC per clip
            return _read(str(Path(rec.get("_root", self.root)) / rec["path"]))
        fds = self.__dict__.setdefault("_fds", {})
        p = str(Path(rec.get("_root", self.root)) / rec["pack"])
        if p not in fds:
            fds[p] = os.open(p, os.O_RDONLY)
        blob = os.pread(fds[p], int(rec["nbytes"]), int(rec["off"]))
        return _read(io.BytesIO(blob))

    def _remix(self, rng):
        b = self.bank
        host = b.hosts[rng.randrange(len(b.hosts))].astype(np.float32) / 32767.0
        dur = len(host) / SR
        lam = b.syn_rate * dur
        n = min(8, int(np.random.default_rng(rng.getrandbits(32)).poisson(lam)))
        x = host.copy(); evs = []
        t = rng.uniform(0.0, 0.6)
        order = [b.snippets[rng.randrange(len(b.snippets))] for _ in range(n)]
        for snip, c in order:
            s = snip.astype(np.float32) / 32767.0
            L = len(s)
            if t * SR + L > len(x):
                break
            a = int(t * SR)
            g = 10 ** (rng.uniform(-4.0, 4.0) / 20.0)
            fade = min(int(0.01 * SR), L // 4)
            env = np.ones(L, np.float32)
            if fade > 0:
                env[:fade] = np.linspace(0, 1, fade); env[-fade:] = np.linspace(1, 0, fade)
            x[a:a + L] += g * s * env
            evs.append((a / SR, (a + L) / SR, c))
            t = (a + L) / SR + rng.uniform(0.08, 1.2)
        return x, evs

    # ---- window + targets ---------------------------------------------------
    def _window(self, y, evs, rng, pres_w, bnd_w, cls_known):
        W, T = self.W, self.T
        off = rng.randint(0, len(y) - W) if len(y) > W else 0
        y = y[off:off + W]
        nv = len(y)
        wav = np.zeros(W, np.float32); wav[:nv] = y
        t0 = off / SR
        pres = np.zeros((T, self.C + 1), np.float32)
        bnd = np.zeros((T, 2), np.float32)
        ext = np.zeros((T, 2), np.float32); ext_w = np.zeros(T, np.float32)
        valid = np.zeros(T, np.float32); valid[:max(1, math.ceil(nv / SR * FPS))] = 1.0
        idx = np.arange(T, dtype=np.float32)
        count = 0
        for s, e, c in evs:
            s, e = (s - t0) * FPS, (e - t0) * FPS
            if e <= 0 or s >= T:
                continue
            if 0 <= (s + e) / 2 < T:
                count += 1
            a, bb = max(0.0, s), min(float(T), e)
            ia, ib = int(math.floor(a)), int(math.ceil(bb))
            pres[ia:ib, -1] = 1.0
            if c >= 0:
                pres[ia:ib, c] = 1.0
            for j, v in ((0, s), (1, e)):
                if 0 <= v < T:          # a boundary cut by the window edge is not a boundary
                    g = np.exp(-0.5 * ((idx - v) / 1.0) ** 2)
                    k = int(round(v)) if int(round(v)) < T else T - 1
                    g[k] = 1.0
                    bnd[:, j] = np.maximum(bnd[:, j], g)
            if s >= 0 and e <= T and ib > ia:
                fr = np.arange(ia, ib, dtype=np.float32) + 0.5
                ext[ia:ib, 0] = fr - s; ext[ia:ib, 1] = e - fr
                ext_w[ia:ib] = 1.0 / (ib - ia)
        return dict(wav=wav, valid=valid, pres=pres, bnd=bnd, ext=ext, ext_w=ext_w,
                    count=np.int64(count), pres_w=np.float32(pres_w), bnd_w=np.float32(bnd_w),
                    cls_known=np.float32(cls_known))

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rank = int(os.environ.get("RANK", "0"))
        rng = random.Random(self.seed * 1000 + rank * 97 + (wi.id if wi else 0))
        keys = list(self.quotas); wts = [self.quotas[k] for k in keys]
        while True:
            k = rng.choices(keys, wts)[0]
            try:
                if k == "gold" or k == "silver":
                    pool = self.gold if k == "gold" else self.silver
                    rec = pool[rng.randrange(len(pool))]
                    y = self._corpus_wav(rec)
                    le_idx = rec["_le_idx"]
                    evs = [(float(e["start"]), float(e["end"]), le_idx.get(e["cls"], -1)) for e in rec["events"]]
                    if k == "gold":
                        yield self._window(y, evs, rng, 1.0, 1.0, 1.0)
                    else:   # silver: agnostic presence only, no boundary/extent/count
                        yield self._window(y, evs, rng, self.silver_pres_w, 0.0, 0.0)
                elif k == "valnat":
                    y, evs = self.bank.natural[rng.randrange(len(self.bank.natural))]
                    yield self._window(y.astype(np.float32) / 32767.0, evs, rng, 1.0, 1.0, 1.0)
                elif k == "valsyn":
                    y, evs = self.bank.synth[rng.randrange(len(self.bank.synth))]
                    yield self._window(y.astype(np.float32) / 32767.0, evs, rng, 1.0, 1.0, 1.0)
                else:
                    x, evs = self._remix(rng)
                    yield self._window(x, evs, rng, 1.0, 1.0, 1.0)
            except Exception as ex:                                 # noqa: BLE001
                print("[data] skip", k, type(ex).__name__, ex, flush=True)


def collate(items):
    return {k: torch.from_numpy(np.stack([it[k] for it in items])) for k in items[0]}
