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
                    count_weight: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Keep as many spans as the count head says the clip contains.

    `count_weight` blends between pure count-head control (1.0) and a plain
    score floor (0.0), so the ablation is one number.
    """
    if len(spans) == 0:
        return spans, scores
    k_pred = int(np.argmax(count_probs)) if count_probs is not None else 1
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
