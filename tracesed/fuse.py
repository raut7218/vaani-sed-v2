"""Natural-clip decoding: the span model's candidates, re-ranked by TraceModel evidence.

Measured on the held-out validation fold (358 natural clips):

  f0 span candidates, top-K by span score (K from transcript)   1.1766
  + re-ranked by TraceModel presence / edge contrast / boundary  1.1967

The span model (the repo's v2 `src/` detector) proposes better-shaped natural
events than any frame decoder of TraceModel (1.10-1.13), while TraceModel's
frame posteriors are better at telling a real event from a merged or truncated
proposal. Each candidate [a, b) is scored

  log s  +  w_in * log mean(p[a:b])  -  w_out * mean(p just outside)  +  w_b * log(on(a) * off(b))

so a span that stops inside an event (high presence just outside) or straddles
two events is demoted, and exactly K are kept.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from src.infer.decode import soft_nms_1d

W_DEFAULT = dict(w_in=1.0, w_out=1.0, w_b=0.25)


@torch.no_grad()
def f0_candidates(model, ck_data: dict, pp: dict, clips, dev, bs: int = 32):
    """Span candidates of the v2 span model over full-length clips (8 s windows, 4 s hop)."""
    from src.infer.runner import DEFAULT_POSTPROC
    from src.models.trident import decode_spans
    from tracesed.data import _read
    pp = {**DEFAULT_POSTPROC, **(pp or {})}
    fps, SR = float(ck_data["fps"]), int(ck_data["sr"])
    W = int(round(float(ck_data["clip_len"]) * SR)); H = W // 2; NF = int(round(float(ck_data["clip_len"]) * fps))
    wins, audio = [], {}
    for c in clips:
        y = _read(c["path"]); audio[c["uid"]] = y
        off = 0
        while True:
            wins.append((c["uid"], off, min(W, len(y) - off)))
            if off + W >= len(y):
                break
            off = min(off + H, len(y) - W)
    acc = {}
    model.eval()
    for i in range(0, len(wins), bs):
        ch = wins[i:i + bs]
        wav = torch.zeros(len(ch), W); fv = torch.zeros(len(ch), NF)
        for j, (u, off, nv) in enumerate(ch):
            wav[j, :nv] = torch.from_numpy(audio[u][off:off + nv]); fv[j, :max(1, math.ceil(nv / SR * fps))] = 1
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out = model(wav.to(dev), fv.to(dev))
        sp, sc, _, _ = decode_spans(out, NF, fps, q_power=float(pp["quality_power"]))
        sp, sc = sp.float().cpu().numpy(), sc.float().cpu().numpy()
        for j, (u, off, nv) in enumerate(ch):
            keep = sc[j] > 0.01
            s = sp[j][keep] + off / SR; c_ = sc[j][keep]
            n = len(audio[u])
            lo = off / SR + (1.0 if off > 0 else -1e9); hi = (off + nv) / SR - (1.0 if off + nv < n else -1e9)
            mid = s.mean(1); ok = (mid >= lo) & (mid <= hi)
            acc.setdefault(u, []).append((s[ok], c_[ok]))
    res = {}
    for c in clips:
        u = c["uid"]; dur = len(audio[u]) / SR
        s = np.concatenate([a for a, _ in acc[u]]).clip(0, dur); sc = np.concatenate([b for _, b in acc[u]])
        o = np.argsort(-sc)[:400]
        res[u] = dict(spans=s[o].astype(np.float32), scores=sc[o].astype(np.float32), dur=dur)
    return res


def rerank_topk(cand: dict, post: dict, K: int, w: dict | None = None, sigma: float = 0.1, iou: float = 0.5):
    """Exactly K of the span model's candidates, re-ranked by TraceModel posteriors (50 Hz)."""
    w = {**W_DEFAULT, **(w or {})}
    if K <= 0 or len(cand["spans"]) == 0:
        return []
    s, c = soft_nms_1d(cand["spans"], cand["scores"], sigma=sigma, iou_thr=iou, max_out=max(24, K))
    pr = post["pres"][:, -1].astype(np.float64)
    on, off = post["bnd"][:, 0].astype(np.float64), post["bnd"][:, 1].astype(np.float64)
    T = len(pr)
    ls = []
    for (a, b), sc in zip(s, c):
        ia, ib = int(np.clip(a * 50, 0, T - 1)), int(np.clip(b * 50, 1, T))
        ins = pr[ia:max(ib, ia + 1)].mean()
        out = ((pr[max(0, ia - 5):ia].mean() if ia > 0 else 0.0) + (pr[ib:min(T, ib + 5)].mean() if ib < T else 0.0)) / 2
        bb = on[max(0, ia - 2):ia + 3].max() * off[max(0, ib - 3):min(T, ib + 2)].max()
        ls.append(math.log(max(sc, 1e-4)) + w["w_in"] * math.log(max(ins, 1e-3)) - w["w_out"] * out
                  + w["w_b"] * math.log(max(bb, 1e-4)))
    o = np.argsort(-np.array(ls))[:K]
    return sorted((float(s[i, 0]), float(min(s[i, 1], cand["dur"]))) for i in o)
