"""Posteriors -> events, for TraceModel outputs on the 50 Hz grid.

Every decoder takes one clip's stitched posterior dict
(pres (T, C+1), bnd (T, 2), ext (T, 2), count (K+1,), dur) and returns a sorted
list of (onset, offset) seconds.

The event count K is decided outside the decoder:
  * the transcript's noise tags, when the clip has a transcript (99.8-100% exact on
    validation);
  * otherwise the count head.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from src.infer.decode import soft_nms_1d
from tracesed.hsmm import decode as hsmm_decode, lognormal_lp

FPS = 50.0
TAG2CLS = json.loads((Path(__file__).parent / "tag_vocab.json").read_text(encoding="utf-8"))
_TOK = re.compile(r"<(/?)([^<>]+)>|\[([^\]]+)\]")


def transcript_events(t: str):
    """Ordered [(tag, char_start, char_end)] of annotated-vocabulary noise tags.

    Tags wrap the words spoken during the noise (`<horn> ... </horn>`), are point
    tags (`[breathing]`), or are empty spans between words. Markers outside the
    annotated vocabulary (`<noise>`, `<pause>`, `[unintelligible]`) are not events.
    """
    spans, openst = [], {}
    for m in _TOK.finditer(t or ""):
        if m.group(3) is not None:
            tag = "[" + m.group(3) + "]"
            if tag in TAG2CLS:
                spans.append([tag, m.start(), m.end()])
            continue
        tag = "<" + m.group(2).strip() + ">"
        if tag not in TAG2CLS:
            continue
        if m.group(1) == "":
            openst.setdefault(tag, []).append(len(spans))
            spans.append([tag, m.start(), None])
        elif openst.get(tag):
            spans[openst[tag].pop()][2] = m.end()
    for s in spans:
        if s[2] is None:
            s[2] = s[1]
    return sorted(spans, key=lambda s: s[1])


def count_from(p: dict, transcript: str | None) -> int:
    if transcript:
        return len(transcript_events(transcript))
    return int(np.argmax(p["count"]))


def thr_decode(p: dict, thr: float = 0.5, med: int = 5, min_dur: float = 0.1):
    from scipy.ndimage import median_filter
    x = median_filter(p["pres"][:, -1].astype(np.float32), size=med) > thr
    ev, t, n = [], 0, len(x)
    while t < n:
        if x[t]:
            a = t
            while t < n and x[t]:
                t += 1
            if (t - a) / FPS >= min_dur:
                ev.append((a / FPS, min(t / FPS, p["dur"])))
        else:
            t += 1
    return ev


def proposals(p: dict, floor: float = 0.35, w_b: float = 0.5):
    """EPN proposals: every confident frame +- its predicted extent."""
    pr = p["pres"][:, -1].astype(np.float64)
    ext = p["ext"].astype(np.float64)
    on, off = p["bnd"][:, 0].astype(np.float64), p["bnd"][:, 1].astype(np.float64)
    t = np.where(pr > floor)[0]
    if len(t) == 0:
        t = np.array([int(np.argmax(pr))])
    a = np.clip((t + 0.5 - ext[t, 0]) / FPS, 0, p["dur"])
    b = np.clip((t + 0.5 + ext[t, 1]) / FPS, 0, p["dur"])
    ok = b - a > 0.02
    a, b, t = a[ok], b[ok], t[ok]
    ia = np.clip((a * FPS).round().astype(int), 0, len(on) - 1)
    ib = np.clip((b * FPS).round().astype(int), 0, len(off) - 1)
    return np.stack([a, b], 1), pr[t] * (np.sqrt(on[ia] * off[ib]) + 1e-3) ** w_b


def epn_topk(p: dict, K: int, floor: float = 0.35, w_b: float = 0.5, sigma: float = 0.1, iou: float = 0.5):
    if K <= 0:
        return []
    s, c = proposals(p, floor, w_b)
    if len(s) == 0:
        return []
    s, c = soft_nms_1d(s, c, sigma=sigma, iou_thr=iou, max_out=max(24, K))
    o = np.argsort(-c)[:K]
    return sorted((float(x), float(y)) for x, y in s[o])


def hsmm_k(p: dict, K: int, syn: bool, bias: float = 0.0, w_on: float = 0.5, w_off: float = 0.5,
           w_dur: float = 0.0, ev_cost: float = 0.0):
    if K <= 0:
        return []
    dl = lognormal_lp(int(6 * FPS), FPS, np.log(0.5 if syn else 0.44), 1.0)
    scs, seg = hsmm_decode(p["pres"][:, -1], FPS, K, onset=p["bnd"][:, 0], offset=p["bnd"][:, 1], bias=bias,
                           w_on=w_on, w_off=w_off, dur_lp=dl, w_dur=w_dur, ev_cost=ev_cost, max_dur=6.0)
    k = K if scs[K] > -1e17 else int(np.argmax(scs))
    return [(float(a), float(min(b, p["dur"]))) for a, b in seg(k)]


DECODERS = {"thr": thr_decode, "epn": epn_topk, "hsmm": hsmm_k}
