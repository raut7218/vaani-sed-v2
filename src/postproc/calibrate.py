"""Transductive per-district calibration.

The problem this solves, measured
---------------------------------
v1's submitted predictions covered **29%** of the test audio (median 18%) with a
median event length of 0.56 s. The reference distribution on held-out data is
**52%** coverage (median 41%) and a median event of 1.04 s. Thresholds tuned on a
five-state validation slice were far too aggressive once the posteriors shifted
on unseen states, and on the validation coverage-to-score curve an operating
point at 0.29 coverage scores ~0.78 where 0.62 coverage scores ~0.96. That single
miscalibration was worth roughly 0.12-0.15 of leaderboard score.

Why per district
----------------
Every Vaani filename encodes its state and district:

    IISc_VaaniProject_K_WestBengal_Darjeeling_<session>_...

The 5,517 test clips span ~150 districts. Recording device, room, and ambient
noise floor are far more homogeneous *within* a district than across the corpus,
so a district is the natural unit of domain shift - and the grouping key is
sitting in the filename, free.

This is legitimate transductive adaptation: it uses only the unlabelled test
audio and its own predictions, never any label.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from src.infer.runner import candidates_to_events

# Fallback priors, used only when nothing measured is supplied. Prefer
# `priors_from_records`: these two constants were carried over from v1, and on
# the fold-0 validation set the reference is 1.44 events per clip against the
# 1.22 written here. Fitting to a prior that is 15% low makes the calibrator
# *remove* true positives - measured on the v2 checkpoint it turned 1.1881 into
# 1.1647, cutting fp 9942 -> 8636 but tp 9153 -> 8513.
PRIOR_EVENTS_PER_CLIP = 1.44
PRIOR_COVERAGE = 0.55
MIN_CLIPS_PER_GROUP = 30        # below this, a group's statistics are noise


def priors_from_records(records, clip_len: float | None = None) -> tuple:
    """(events/clip, coverage) measured on a manifest.

    The operating point the calibrator aims at should come from the data being
    modelled, not from a constant that was true of some earlier split. Pass the
    *training* records: they are the only labelled sample of the same annotation
    process the test set went through.
    """
    # Only clips that carry timestamps. Bronze clips have no events by
    # definition, and averaging them in measures the wrong population: the
    # evaluation references come from annotated clips alone. Including the
    # corpus's 17899 bronze clips pulled the measured prior to 1.12 events per
    # clip and 0.398 coverage, against the 1.44 / 0.548 the fold-0 references
    # actually hold - so the "measured" prior came out further from the truth
    # than the 1.22 constant it replaced, and the calibrator squeezed the
    # operating point down to 1.07 events per clip and cost 0.04 of score.
    n_ev, cov = [], []
    for r in records:
        ev = r.get("events") or []
        if not ev:
            continue
        dur = float(r.get("duration") or 0.0)
        if clip_len:
            dur = min(dur, float(clip_len))
        if dur <= 0:
            continue
        n_ev.append(len(ev))
        cov.append(min(1.0, sum(min(e["end"], dur) - min(e["start"], dur)
                                for e in ev) / dur))
    if not n_ev:
        return PRIOR_EVENTS_PER_CLIP, PRIOR_COVERAGE
    return float(np.mean(n_ev)), float(np.mean(cov))


def district_of(uid: str) -> str:
    """`IISc_VaaniProject_K_<State>_<District>_<session>...` -> 'State_District'."""
    parts = uid.split("_")
    if len(parts) >= 5 and parts[0].startswith("IISc"):
        return "%s_%s" % (parts[3], parts[4])
    return "_global"


def _stats(cands: Sequence[dict], pp: dict) -> tuple:
    n_ev, cov = [], []
    for c in cands:
        ev = candidates_to_events(c, pp)
        n_ev.append(len(ev))
        dur = max(c["duration"], 1e-6)
        cov.append(sum(b - a for a, b in ev) / dur)
    return float(np.mean(n_ev)), float(np.mean(cov))


def _cost(n_ev: float, cov: float, priors: tuple) -> float:
    p_ev, p_cov = priors
    return (((n_ev - p_ev) / p_ev) ** 2 + ((cov - p_cov) / p_cov) ** 2)


def _fit_group(cands: Sequence[dict], base_pp: dict,
               scale_grid: Sequence[float], slack_grid: Sequence[int],
               priors: tuple) -> dict:
    """Fit the two knobs that actually move the operating point.

    `score_scale` alone is not enough: when `count_weight` is 1.0 the number of
    emitted events comes from the count head, and scaling the scores changes
    nothing at all. `count_slack` is the lever that works in that regime, so both
    are searched together - the smoke test caught exactly this.
    """
    best, best_cost = {"score_scale": 1.0, "count_slack": base_pp.get("count_slack", 1)}, float("inf")
    for slack in slack_grid:
        for s in scale_grid:
            pp = {**base_pp, "score_scale": s, "count_slack": slack}
            scaled = [{**c, "scores": np.clip(c["scores"] * s, 0, 1)} for c in cands]
            c = _cost(*_stats(scaled, pp), priors)
            if c < best_cost:
                best_cost = c
                best = {"score_scale": float(s), "count_slack": int(slack)}
    return best


def calibrate(cands: Dict[str, dict], base_pp: dict,
              scale_grid: Sequence[float] | None = None,
              slack_grid: Sequence[int] | None = None,
              per_district: bool = True,
              priors: tuple | None = None) -> Dict[str, dict]:
    """Return {district: {"score_scale": .., "count_slack": ..}} plus '_global'."""
    scale_grid = scale_grid if scale_grid is not None else list(np.round(np.arange(0.5, 2.51, 0.25), 2))
    slack_grid = slack_grid if slack_grid is not None else [-1, 0, 1, 2]
    priors = priors or (PRIOR_EVENTS_PER_CLIP, PRIOR_COVERAGE)
    uids = list(cands)
    out: Dict[str, dict] = {"_global": _fit_group([cands[u] for u in uids], base_pp,
                                                  scale_grid, slack_grid, priors)}
    if not per_district:
        return out

    groups: Dict[str, List[str]] = {}
    for u in uids:
        groups.setdefault(district_of(u), []).append(u)
    for g, us in groups.items():
        if len(us) < MIN_CLIPS_PER_GROUP:
            # A 3-clip district cannot support its own operating point; the
            # global one is a far better estimate than an overfitted local one.
            continue
        out[g] = _fit_group([cands[u] for u in us], base_pp, scale_grid,
                            slack_grid, priors)
    return out


def apply_scales(cands: Dict[str, dict], scales: Dict[str, dict]) -> Dict[str, dict]:
    """Bake each clip's fitted scale into its scores and per-clip post-proc."""
    out = {}
    for u, c in cands.items():
        s = scales.get(district_of(u), scales.get("_global", {}))
        out[u] = {**c,
                  "scores": np.clip(c["scores"] * float(s.get("score_scale", 1.0)), 0, 1),
                  "pp_override": {"count_slack": int(s.get("count_slack", 1))}}
    return out


def fit_postproc(cands: Dict[str, dict], refs: Dict[str, list], base_pp: dict,
                 grids: Dict[str, Sequence] | None = None) -> dict:
    """Tune the post-processor against the metric itself, on labelled data.

    Prior-matching exists because the test set has no labels. Validation does,
    and there is no reason to aim at a proxy when the target is available: this
    coordinate-descends the knobs that move the score, one at a time, re-scoring
    with the real `evaluate` at every step. The result is what ships in the
    checkpoint's config, and the transductive calibrator then only has to correct
    the residual district-to-district drift on top of it.
    """
    from src.evaluation.metrics import evaluate

    # Only the knobs `candidates_to_events` itself reads. Refinement, boundary
    # agreement and SoftNMS all run earlier, in `_host_to_candidates`, against
    # the raw model output - re-running them per trial would mean re-running
    # SoftNMS on every clip for every trial, so those are swept once offline
    # against cached candidates instead of here.
    grids = grids or {
        "count_slack": [-1, 0, 1, 2],
        "count_weight": [0.0, 0.5, 0.75, 1.0],
        "count_mode": ["expected", "argmax"],
        "score_floor": [0.02, 0.05, 0.1, 0.2, 0.35],
        "min_dur": [0.03, 0.08, 0.15],
        "merge_gap": [0.0, 0.05, 0.1],
    }
    pp = dict(base_pp)

    def score_of(cfg: dict) -> float:
        preds = {u: candidates_to_events(c, cfg) for u, c in cands.items()}
        return evaluate({u: preds[u] for u in refs}, refs)["score"]

    best = score_of(pp)
    for _ in range(2):                       # two passes: the knobs interact
        for key, values in grids.items():
            cur = pp.get(key)
            for v in values:
                if v == cur:
                    continue
                s = score_of({**pp, key: v})
                if s > best + 1e-5:
                    best, pp[key] = s, v
    return {"pp": pp, "score": best}
