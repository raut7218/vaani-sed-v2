"""Dataset, tier-balanced sampler, and the state-grouped k-fold split.

Two changes from v1 that matter.

**Spans, not just frame masks.** The model regresses boundaries, so the dataset
emits the events themselves - (onset, offset) in base frames - alongside the
rasterised frame target the auxiliary heads still need.

**k-fold by state, not one holdout.** v1 held out five states
(Himachal / MP / Tripura / Rajasthan / Nagaland) and tuned eight classes' worth
of post-processing against that single slice. The test set spans ~150 districts
across 25+ states, and the val-to-leaderboard drop was 0.16. Selecting a
checkpoint on one narrow slice is how that happens; `--fold` makes the holdout
rotate so model selection averages over folds instead.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from src.data.labels import LabelEncoder

TIER_IDS = {"gold": 0, "silver": 1, "bronze": 2}
MAX_EVENTS = 12          # 99.99th percentile of the corpus is 7


def load_manifest(path: str | Path) -> List[dict]:
    recs = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def state_folds(recs: Sequence[dict], n_folds: int = 5, seed: int = 1234,
                group_by: str = "state") -> List[List[str]]:
    """Partition the grouping keys into `n_folds` roughly equal-sized folds.

    Balanced by clip count rather than by number of states, because Vaani's
    states differ in size by more than an order of magnitude and a naive
    round-robin would give one fold most of the data.
    """
    ts = [r for r in recs if r.get("events")]
    counts: Dict[str, int] = {}
    for r in ts:
        counts[str(r.get(group_by, "")) or "_"] = counts.get(
            str(r.get(group_by, "")) or "_", 0) + 1
    keys = sorted(counts, key=lambda k: -counts[k])
    rng = random.Random(seed)
    rng.shuffle(keys)
    keys.sort(key=lambda k: -counts[k])            # stable: largest first
    folds: List[List[str]] = [[] for _ in range(n_folds)]
    sizes = [0] * n_folds
    for k in keys:                                  # greedy longest-processing-time
        i = min(range(n_folds), key=lambda j: sizes[j])
        folds[i].append(k)
        sizes[i] += counts[k]
    return folds


def split_manifest(recs: List[dict], fold: int = 0, n_folds: int = 5,
                   seed: int = 1234, group_by: str = "state"):
    """Hold out one state-fold. Bronze always trains (it has nothing to score)."""
    ts = [r for r in recs if r.get("events")]
    bronze = [r for r in recs if not r.get("events")]
    if not ts:
        return recs, []
    folds = state_folds(recs, n_folds, seed, group_by)
    held = set(folds[fold % n_folds])
    val = [r for r in ts if (str(r.get(group_by, "")) or "_") in held]
    if not val or len(val) > 0.5 * len(ts):
        # Single-state corpora (or a tiny download) cannot be split by state;
        # fall back to a random clip split rather than returning an empty val.
        rng = random.Random(seed)
        shuffled = list(ts)
        rng.shuffle(shuffled)
        cut = max(1, int(len(ts) / n_folds))
        val = shuffled[fold * cut:(fold + 1) * cut] or shuffled[:cut]
    val_uids = {r["uid"] for r in val}
    train = [r for r in ts if r["uid"] not in val_uids] + bronze
    return train, val


class VaaniSpanDataset(Dataset):
    def __init__(self, records: Sequence[dict], root: str | Path, le: LabelEncoder,
                 clip_len: float = 8.0, sr: int = 16000, fps: float = 25.0,
                 train: bool = True, augment: bool = True,
                 vad_dir: str | Path | None = None):
        self.recs = list(records)
        self.root = Path(root)
        self.le = le
        self.clip_len = float(clip_len)
        self.sr = int(sr)
        self.fps = float(fps)
        self.train = train
        self.augment = augment and train
        self.n_samples = int(round(self.clip_len * self.sr))
        self.n_frames = int(round(self.clip_len * self.fps))
        self.vad_dir = Path(vad_dir) if vad_dir else None

    def __len__(self) -> int:
        return len(self.recs)

    def _load_wav(self, rec: dict) -> np.ndarray:
        import soundfile as sf
        # `_root` lets a synthetic manifest live in its own directory and still
        # be mixed into one training set.
        root = Path(rec.get("_root", self.root))
        y, sr = sf.read(str(root / rec["path"]), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != self.sr:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
        return y.astype("float32")

    def _load_vad(self, uid: str, t_off: float) -> tuple:
        if self.vad_dir is None:
            return np.zeros((self.n_frames,), "float32"), 0.0
        p = self.vad_dir / (uid + ".npy")
        if not p.exists():
            return np.zeros((self.n_frames,), "float32"), 0.0
        v = np.load(p).astype("float32")            # speech prob at self.fps
        a = int(round(t_off * self.fps))
        v = v[a:a + self.n_frames]
        out = np.zeros((self.n_frames,), "float32")
        out[:len(v)] = v
        return out, 1.0

    def __getitem__(self, i: int) -> dict:
        rec = self.recs[i]
        y = self._load_wav(rec)
        events = rec.get("events") or []

        # --- crop / pad to the window, shifting event times with the crop ---
        offset = 0
        if len(y) > self.n_samples:
            if self.train:
                offset = random.randint(0, len(y) - self.n_samples)
            else:
                offset = (len(y) - self.n_samples) // 2
            y = y[offset:offset + self.n_samples]
        valid_samples = len(y)
        if len(y) < self.n_samples:
            y = np.pad(y, (0, self.n_samples - len(y)))
        t_off = offset / self.sr

        if self.augment:
            y = y * float(np.random.uniform(0.85, 1.15))
            if random.random() < 0.5:
                y = y + np.random.randn(len(y)).astype("float32") * 1e-3

        C, F = len(self.le), self.n_frames
        frame_t = np.zeros((F, C), dtype="float32")
        spans = np.full((MAX_EVENTS, 2), -1.0, dtype="float32")
        span_cls = np.full((MAX_EVENTS,), -1, dtype="int64")
        n_ev = 0

        for ev in events:
            ci = self.le.idx.get(ev["cls"])
            if ci is None:
                continue
            s = (float(ev["start"]) - t_off) * self.fps
            e = (float(ev["end"]) - t_off) * self.fps
            # Keep only events that survive the crop with real support. A sliver
            # clipped to 2 frames teaches a boundary that is not in the audio.
            s_c, e_c = max(0.0, s), min(float(F), e)
            if e_c - s_c < 0.25 or e_c <= s_c:
                continue
            a, b = int(math.floor(s_c)), int(math.ceil(e_c))
            if b > a:
                frame_t[a:b, ci] = 1.0
            else:
                frame_t[min(a, F - 1), ci] = 1.0
            if n_ev < MAX_EVENTS:
                spans[n_ev] = (s_c, e_c)
                span_cls[n_ev] = ci
                n_ev += 1

        clip_t = np.zeros((C,), dtype="float32")
        if events:
            clip_t = frame_t.max(axis=0)
        else:
            for ci in self.le.encode_clip_categories(rec.get("clip_labels")):
                clip_t[ci] = 1.0

        tier = rec.get("tier", "bronze")
        n_valid = int(min(F, math.ceil(valid_samples / self.sr * self.fps)))
        frame_valid = np.zeros((F,), dtype="float32")
        frame_valid[:max(1, n_valid)] = 1.0
        speech, has_vad = self._load_vad(rec["uid"], t_off)

        return {
            "wav": torch.from_numpy(y),
            "frame_target": torch.from_numpy(frame_t),
            "clip_target": torch.from_numpy(clip_t),
            "frame_valid": torch.from_numpy(frame_valid),
            "spans": torch.from_numpy(spans),
            "span_cls": torch.from_numpy(span_cls),
            "n_events": torch.tensor(n_ev, dtype=torch.long),
            "speech_target": torch.from_numpy(speech),
            "has_vad": torch.tensor(has_vad, dtype=torch.float32),
            "tier": torch.tensor(TIER_IDS.get(tier, 2), dtype=torch.long),
            "uid": rec["uid"],
        }


def collate(batch: List[dict]) -> dict:
    out = {}
    for k in batch[0]:
        if k == "uid":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


class TierBatchSampler(Sampler):
    """Compose every batch from fixed per-tier quotas.

    Kept from v1: with tiers this imbalanced a plain random sampler produces
    batches with no gold at all, and the boundary terms then have nothing
    trustworthy to learn from for whole stretches of training.
    """

    def __init__(self, records: Sequence[dict], batch_size: int,
                 quotas: Dict[str, float] | None = None, seed: int = 0):
        self.records = list(records)
        self.batch_size = int(batch_size)
        self.seed = seed
        self.by_tier: Dict[str, List[int]] = {}
        for i, r in enumerate(self.records):
            self.by_tier.setdefault(r.get("tier", "bronze"), []).append(i)
        self.by_tier = {k: v for k, v in self.by_tier.items() if v}

        quotas = quotas or {"gold": 0.5, "silver": 0.35, "bronze": 0.15}
        quotas = {k: v for k, v in quotas.items() if k in self.by_tier and v > 0}
        tot = sum(quotas.values()) or 1.0
        raw = {k: self.batch_size * v / tot for k, v in quotas.items()}
        self.counts = {k: int(math.floor(v)) for k, v in raw.items()}
        rem = self.batch_size - sum(self.counts.values())
        for k in sorted(raw, key=lambda k: raw[k] - math.floor(raw[k]), reverse=True):
            if rem <= 0:
                break
            self.counts[k] += 1
            rem -= 1
        self.counts = {k: v for k, v in self.counts.items() if v > 0}
        self._nb = max(1, min(len(self.by_tier[k]) // c for k, c in self.counts.items()))

    def __len__(self) -> int:
        return self._nb

    def __iter__(self):
        rng = random.Random(self.seed)
        self.seed += 1
        pools = {k: list(v) for k, v in self.by_tier.items()}
        for v in pools.values():
            rng.shuffle(v)
        ptr = {k: 0 for k in pools}
        for _ in range(self._nb):
            batch: List[int] = []
            for k, c in self.counts.items():
                pool = pools[k]
                if c > len(pool):
                    batch.extend(rng.choice(pool) for _ in range(c))
                    continue
                if ptr[k] + c > len(pool):
                    rng.shuffle(pool)
                    ptr[k] = 0
                batch.extend(pool[ptr[k]:ptr[k] + c])
                ptr[k] += c
            rng.shuffle(batch)
            yield batch
