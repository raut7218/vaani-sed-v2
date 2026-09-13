"""Count-exact semi-Markov event decoding.

Why a segment DP instead of thresholds
--------------------------------------
Thresholding a frame posterior couples an event's *extent* to the model's
*confidence*: a hesitant model emits short fragments, a confident one emits
blobs, and the event count falls out of wherever the threshold happens to cut.
On validation that is exactly the failure we measured - the count is wrong
(synthetic recall .54 at precision .96) while boundaries, where an event is
found, are accurate.

Here the decoder chooses a *set* of non-overlapping segments directly. Each
segment [a, b) earns

    sum_{a<=t<b} (logit p_t - bias)          presence evidence
  + w_on  * log o_a  +  w_off * log f_b      boundary evidence
  + w_dur * log P(b - a)                     duration prior (log-normal)
  + w_pos * log N(centre; mu_k, sigma)       optional per-event position prior

and background frames earn 0, so a segment is kept only if its evidence
outweighs staying in background. One O(K * T * D) pass yields the best
decoding with exactly k events *for every k <= K*, so choosing K is a separate,
explicit step:

  * K given (transcript): take row K.
  * K unknown: argmax_k best[k] + w_cnt * log P(k)   (count head / subset prior).

Per-event class evidence: when the transcript names each event's type, event
k scores presence on its own class channel, so "horn then barking" cannot be
decoded as "barking then horn".
"""
from __future__ import annotations

import math

import numba
import numpy as np

NEG = -1e18


@numba.njit(cache=True)
def _dp(pres_cum, on_lp, off_lp, dur_lp, pos_lp, ev_ch, K, T, D, gap):
    """best[k, t]: best score using frames [0, t) with k events all ended by t.

    pres_cum: (C, T+1) prefix sums of per-channel presence scores
    on_lp / off_lp: (T+1,) log boundary evidence at frame index
    dur_lp: (D+1,) log duration prior (already weighted), NEG where disallowed
    pos_lp: (K, T+1) log position prior of event k, indexed by segment centre*2
            (all zeros when unused)
    ev_ch: (K,) channel index each event scores presence on
    """
    best = np.full((K + 1, T + 1), NEG)
    arg_a = np.full((K + 1, T + 1), -1, np.int64)   # segment start if event ends at t
    best[0, :] = 0.0
    for k in range(1, K + 1):
        ch = ev_ch[k - 1]
        for t in range(1, T + 1):
            # carry: no event ends exactly at t
            v = best[k, t - 1]
            a_best = -1
            dmax = min(D, t)
            for d in range(1, dmax + 1):
                if dur_lp[d] <= NEG / 2:
                    continue
                a = t - d
                prev_end = a - gap if k > 1 else a
                if prev_end < 0:
                    continue
                pb = best[k - 1, prev_end]
                if pb <= NEG / 2:
                    continue
                c2 = a + t                                   # centre in half-frames
                if c2 > T:
                    c2 = T
                s = (pb + pres_cum[ch, t] - pres_cum[ch, a] + on_lp[a] + off_lp[t]
                     + dur_lp[d] + pos_lp[k - 1, c2])
                if s > v:
                    v = s
                    a_best = a
            best[k, t] = v
            arg_a[k, t] = a_best
    return best, arg_a


@numba.njit(cache=True)
def _backtrack(best, arg_a, k, T, gap):
    # arg_a[i, t] < 0 means best[i, t] was carried from best[i, t - 1], so walking
    # left over carried cells lands exactly on the frame where event i ends.
    segs = np.zeros((k, 2), np.int64)
    t = T
    i = k
    while i > 0:
        while t > 0 and arg_a[i, t] < 0:
            t -= 1
        a = arg_a[i, t]
        segs[i - 1, 0] = a
        segs[i - 1, 1] = t
        t = a - gap if i > 1 else a
        if t < 0:
            t = 0
        i -= 1
    return segs


def lognormal_lp(D: int, fps: float, mu: float, sigma: float, lo: float = 0.02,
                 hi: float = 1e9) -> np.ndarray:
    """log-density of a log-normal over segment length in frames 0..D."""
    d = np.arange(D + 1, dtype=np.float64)
    sec = np.maximum(d, 1e-3) / fps
    lp = -np.log(sec * sigma * math.sqrt(2 * math.pi)) - (np.log(sec) - mu) ** 2 / (2 * sigma ** 2)
    lp[(sec < lo) | (sec > hi)] = NEG
    lp[0] = NEG
    return lp


def logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = np.clip(p.astype(np.float64), eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def decode(presence: np.ndarray, fps: float, K: int, *,
           onset: np.ndarray | None = None, offset: np.ndarray | None = None,
           ev_channel: np.ndarray | None = None, bias: float = 0.0,
           w_pres: float = 1.0, w_on: float = 0.0, w_off: float = 0.0,
           dur_lp: np.ndarray | None = None, w_dur: float = 0.0,
           pos_mu: np.ndarray | None = None, pos_sigma: float = 0.4, w_pos: float = 0.0,
           max_dur: float = 6.0, min_gap: float = 0.0, ev_cost: float = 0.0):
    """Best decoding for every k <= K.

    presence: (T,) or (T, C) event probabilities on the decode grid.
    ev_cost: log-odds charged per event. Without it, and without boundary
    evidence, splitting one event into two adjacent pieces costs nothing, so the
    unconstrained argmax over k drifts to K.
    Returns (scores[k] for k=0..K, function k -> (k, 2) array of [onset, offset] s).
    """
    P = presence if presence.ndim == 2 else presence[:, None]
    T, C = P.shape
    pres = w_pres * (logit(P) - bias)                         # (T, C)
    cum = np.zeros((C, T + 1))
    cum[:, 1:] = np.cumsum(pres, axis=0).T
    D = max(1, min(T, int(round(max_dur * fps))))

    def bnd(x):
        out = np.zeros(T + 1)
        if x is not None:
            xx = np.clip(x.astype(np.float64), 1e-4, 1.0)
            n = min(T, len(xx))
            out[:n] = np.log(xx[:n])
            out[n:] = np.log(xx[n - 1]) if n else 0.0
        return out
    on_lp = w_on * bnd(onset)
    off_lp = w_off * bnd(offset)
    dl = np.zeros(D + 1)
    if dur_lp is not None:
        dl = w_dur * np.where(dur_lp[:D + 1] <= NEG / 2, 0.0, dur_lp[:D + 1])
        dl = np.where(dur_lp[:D + 1] <= NEG / 2, NEG, dl)
    dl = np.where(dl <= NEG / 2, NEG, dl - ev_cost)
    dl[0] = NEG
    Keff = max(K, 1)
    pos = np.zeros((Keff, T + 1))
    if pos_mu is not None and w_pos > 0:
        c = np.arange(T + 1) / (2.0 * fps)                    # half-frame centres in s
        pos = w_pos * (-(c[None, :] - np.asarray(pos_mu)[:Keff, None]) ** 2
                       / (2 * pos_sigma ** 2))
    ch = np.zeros(Keff, np.int64) if ev_channel is None else np.asarray(ev_channel, np.int64)
    gap = int(round(min_gap * fps))
    best, arg_a = _dp(cum, on_lp, off_lp, dl, pos, ch, Keff, T, D, gap)
    scores = best[:, T].copy()
    if K == 0:
        scores = scores[:1]

    def segments(k: int) -> np.ndarray:
        if k == 0 or scores[k] <= NEG / 2:
            return np.zeros((0, 2))
        s = _backtrack(best, arg_a, k, T, gap)
        return s.astype(np.float64) / fps
    return scores, segments
