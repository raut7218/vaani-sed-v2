"""Span decoding: SoftNMS, count-aware selection, and 1D box fusion.

What changed, and why it is not just "a different threshold"
------------------------------------------------------------
v1 chose boundaries by thresholding a frame posterior, so the *position* of
every boundary was a function of the threshold. Here the boundaries come out of
a regression head, and the only thing selection decides is *which* candidate
spans to keep and *how many*. Boundary precision is completely decoupled from
the operating point - which is the whole point, because the optimal per-clip
threshold on the v1 model was near-uniform over [0.05, 0.95] and no fixed value
could ever have worked.

The count head then removes most of what is left: instead of tuning a score
floor, the model predicts how many events the clip contains (83% of clips hold
exactly one) and we keep that many.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np


def soft_nms_1d(spans: np.ndarray, scores: np.ndarray, sigma: float = 0.5,
                iou_thr: float = 0.35, score_floor: float = 1e-3,
                max_out: int = 32, mode: str = "gaussian"):
    """Class-agnostic 1D SoftNMS.

    Gaussian decay rather than hard suppression: neighbouring pyramid points
    describe the *same* event with slightly different boundaries, and hard NMS
    throws that agreement away. Decaying instead lets a genuinely distinct second
    event survive next to a strong one - which matters when 17% of clips hold
    more than one event and some of those overlap.

    Note that `iou_thr` does nothing in the default gaussian mode - the decay is
    a smooth function of IoU with no threshold in it, and only `mode="linear"`
    reads it. Sweeping it on the v2 checkpoint gave four identical scores across
    0.20/0.35/0.50/0.65, which is the sweep noticing it is tuning a dead knob.
    """
    if len(spans) == 0:
        return np.zeros((0, 2), "float32"), np.zeros((0,), "float32")
    spans = spans.astype("float64").copy()
    scores = scores.astype("float64").copy()
    keep_s, keep_c = [], []

    while len(scores) and len(keep_s) < max_out:
        i = int(np.argmax(scores))
        if scores[i] < score_floor:
            break
        best, bs = spans[i], scores[i]
        keep_s.append(best)
        keep_c.append(bs)
        spans = np.delete(spans, i, axis=0)
        scores = np.delete(scores, i, axis=0)
        if not len(scores):
            break
        inter = (np.minimum(spans[:, 1], best[1]) - np.maximum(spans[:, 0], best[0]))
        inter = np.clip(inter, 0, None)
        union = ((spans[:, 1] - spans[:, 0]) + (best[1] - best[0]) - inter)
        iou = inter / np.clip(union, 1e-9, None)
        if mode == "linear":
            decay = np.where(iou > iou_thr, 1.0 - iou, 1.0)
        else:
            decay = np.exp(-(iou ** 2) / sigma)
        scores = scores * decay

    return np.asarray(keep_s, "float32").reshape(-1, 2), np.asarray(keep_c, "float32")


def wbf_1d(span_sets: Sequence[np.ndarray], score_sets: Sequence[np.ndarray],
           iou_thr: float = 0.5, n_models: int | None = None):
    """Weighted box fusion in 1D, for ensembling.

    Do **not** average frame posteriors across models. Averaging blurs, and blur
    is the one thing this task cannot afford - two models that both localise an
    onset well but disagree by 80 ms produce an averaged posterior whose ramp is
    80 ms wider than either. Fusing *spans* keeps the boundaries sharp: matched
    spans are combined as a score-weighted mean of their endpoints, and a span
    that only one model found is down-weighted rather than smeared.
    """
    all_s = np.concatenate([s.reshape(-1, 2) for s in span_sets], axis=0) \
        if span_sets else np.zeros((0, 2), "float32")
    all_c = np.concatenate([c.reshape(-1) for c in score_sets], axis=0) \
        if score_sets else np.zeros((0,), "float32")
    if len(all_s) == 0:
        return np.zeros((0, 2), "float32"), np.zeros((0,), "float32")
    n_models = n_models or len(span_sets)

    order = np.argsort(-all_c)
    all_s, all_c = all_s[order], all_c[order]
    clusters: List[List[int]] = []
    fused: List[np.ndarray] = []

    for i in range(len(all_s)):
        placed = False
        for ci, f in enumerate(fused):
            inter = max(0.0, min(f[1], all_s[i, 1]) - max(f[0], all_s[i, 0]))
            union = (f[1] - f[0]) + (all_s[i, 1] - all_s[i, 0]) - inter
            if union > 0 and inter / union >= iou_thr:
                clusters[ci].append(i)
                idx = clusters[ci]
                w = all_c[idx]
                fused[ci] = np.array([
                    float((all_s[idx, 0] * w).sum() / w.sum()),
                    float((all_s[idx, 1] * w).sum() / w.sum())])
                placed = True
                break
        if not placed:
            clusters.append([i])
            fused.append(all_s[i].copy())

    out_s = np.asarray(fused, "float32").reshape(-1, 2)
    # Confidence scales with how many models agreed: a span found by one model
    # out of five is much weaker evidence than the raw score suggests.
    out_c = np.asarray([float(all_c[c].mean() * min(len(c), n_models) / n_models)
                        for c in clusters], "float32")
    order = np.argsort(-out_c)
    return out_s[order], out_c[order]


def select_by_count(spans: np.ndarray, scores: np.ndarray, count_probs: np.ndarray,
                    min_score: float = 0.05, slack: int = 1,
                    count_weight: float = 1.0, count_mode: str = "expected"
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Keep as many spans as the count head says the clip contains.

    `count_weight` blends between pure count-head control (1.0) and a plain
    score floor (0.0), so the ablation is one number.

    `count_mode` picks how the head's distribution becomes an integer. The argmax
    minimises 0-1 error on a single clip, but 71% of silver clips hold exactly
    one event, so the argmax is 1 almost everywhere and the corpus rate collapses
    to it - the previous run emitted 1.18 events per clip against a reference of
    1.44, and every one of that shortfall is a false negative. The expectation
    minimises squared error instead, which is the one that makes the *rate* come
    out right. Both are here; the tuner picks.
    """
    if len(spans) == 0:
        return spans, scores
    if count_probs is None:
        k_pred = 1
    elif count_mode == "expected":
        k_pred = int(round(float((np.arange(len(count_probs)) * count_probs).sum())))
    else:
        k_pred = int(np.argmax(count_probs))
    k_score = int((scores >= min_score).sum())
    k = int(round(count_weight * k_pred + (1 - count_weight) * k_score)) + slack
    k = max(0, min(k, len(spans)))
    if k == 0:
        # Never emit nothing while a confident span exists: an empty clip scores
        # Dice 0 against a non-empty reference, which is the most expensive
        # single mistake available on this metric.
        k = 1 if scores[0] >= min_score else 0
    return spans[:k], scores[:k]


def merge_close(spans: np.ndarray, gap: float = 0.0) -> np.ndarray:
    """Merge spans separated by less than `gap` seconds. Default: off.

    v1 merged with gaps up to 0.24 s and median-filtered with windows up to
    0.24 s, then unioned eight class channels each already dilated that way.
    The measured result was predictions a median 0.20 s longer than the
    reference. Default 0.0 keeps that class of error out of the pipeline; the
    knob exists only so the ablation can be run.
    """
    if len(spans) == 0 or gap <= 0:
        return spans
    order = np.argsort(spans[:, 0])
    out = [spans[order[0]].copy()]
    for s in spans[order[1:]]:
        if s[0] - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], s[1])
        else:
            out.append(s.copy())
    return np.asarray(out, "float32")


def finalise(spans: np.ndarray, scores: np.ndarray, duration: float,
             min_dur: float = 0.03) -> List[List[float]]:
    """Clip to the audio, drop degenerate spans, round, and sort.

    The scorer rasterises with ``int(onset / 0.01)``, so a negative onset would
    index the frame mask from the wrong end - clipping here is correctness, not
    cosmetics.
    """
    out = []
    for (a, b), _ in zip(spans, scores):
        a = float(max(0.0, min(a, duration)))
        b = float(max(0.0, min(b, duration)))
        if b - a < min_dur:
            continue
        out.append([round(a, 3), round(b, 3)])
    out.sort(key=lambda e: (e[0], e[1]))
    return out


def refine_boundaries(spans: np.ndarray, onset: np.ndarray, offset: np.ndarray,
                      hi_fps: float, duration: float, window_frac: float = 0.25,
                      window_min: float = 0.08, peak_min: float = 0.20) -> np.ndarray:
    """Snap each regressed endpoint onto the nearest boundary-branch peak.

    The pyramid proposes at 40 ms and the branch runs at 20 ms, but resolution is
    not really the point - the point is that "where is the onset" is an easier
    question than "how far away is the onset", and the two heads fail
    independently. Refinement only ever moves an endpoint inside the metric's own
    tolerance window, so a confident-but-wrong branch cannot turn a matching span
    into a missing one; the worst it can do is fail to help.

    Sub-frame position comes from a parabola through the peak frame and its two
    neighbours, in log-probability. Quantising to the 20 ms grid would throw away
    a third of the tolerance on the shortest events, and a softmax-weighted mean
    over the whole window - the obvious alternative - is *biased*: the window is
    not centred on the peak, so whichever side is longer drags the estimate
    towards it by however much background mass it holds. The parabola only ever
    looks at three samples, and for a Gaussian peak log-probability is exactly
    quadratic, so it is unbiased by construction.

    Endpoints whose window holds no peak above `peak_min` are left exactly where
    the regression put them.
    """
    if len(spans) == 0:
        return spans
    out = spans.astype("float64").copy()
    n = len(onset)
    for i, (a, b) in enumerate(out):
        w = max(window_frac * (b - a), window_min)
        for j, (pos, prob) in enumerate(((a, onset), (b, offset))):
            lo = max(0, int(math.floor((pos - w) * hi_fps)))
            hi = min(n, int(math.ceil((pos + w) * hi_fps)) + 1)
            if hi - lo < 2:
                continue
            seg = prob[lo:hi]
            k = int(np.argmax(seg))
            if seg[k] < peak_min:
                continue
            k += lo
            delta = 0.0
            if 0 < k < n - 1:
                l0, l1, l2 = (math.log(max(prob[k - 1], 1e-6)),
                              math.log(max(prob[k], 1e-6)),
                              math.log(max(prob[k + 1], 1e-6)))
                den = l0 - 2.0 * l1 + l2
                if den < -1e-9:                     # a real maximum, not a plateau
                    delta = float(np.clip(0.5 * (l0 - l2) / den, -0.5, 0.5))
            out[i, j] = (k + 0.5 + delta) / hi_fps
    out[:, 0] = np.clip(out[:, 0], 0.0, duration)
    out[:, 1] = np.clip(out[:, 1], 0.0, duration)
    bad = out[:, 1] <= out[:, 0]
    out[bad] = spans[bad]
    return out.astype("float32")


def boundary_agreement(spans: np.ndarray, onset: np.ndarray, offset: np.ndarray,
                       hi_fps: float, tol_frac: float = 0.2,
                       tol_min: float = 0.05) -> np.ndarray:
    """How well each span's endpoints line up with the branch's peaks, in [0, 1].

    Read at exactly the metric's tolerance, so this is a direct estimate of "will
    this span match". Multiplied into the candidate score it ranks spans by how
    likely they are to *score*, which is what count-based selection and SoftNMS
    both need and what actionness alone cannot say.
    """
    if len(spans) == 0:
        return np.zeros((0,), "float32")
    n, out = len(onset), np.zeros(len(spans), "float32")
    for i, (a, b) in enumerate(spans):
        tol = max(tol_frac * (b - a), tol_min)
        v = 1.0
        for pos, prob in ((a, onset), (b, offset)):
            lo = max(0, int((pos - tol) * hi_fps))
            hi = min(n, int((pos + tol) * hi_fps) + 1)
            v *= float(prob[lo:hi].max()) if hi > lo else 0.0
        out[i] = math.sqrt(max(v, 0.0))
    return out
