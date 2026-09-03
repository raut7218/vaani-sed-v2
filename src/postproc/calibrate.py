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

# Priors measured on the annotated corpus (class-agnostic, per clip).
PRIOR_EVENTS_PER_CLIP = 1.22
PRIOR_COVERAGE = 0.52
MIN_CLIPS_PER_GROUP = 30        # below this, a group's statistics are noise


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


def _cost(n_ev: float, cov: float) -> float:
    return (((n_ev - PRIOR_EVENTS_PER_CLIP) / PRIOR_EVENTS_PER_CLIP) ** 2
            + ((cov - PRIOR_COVERAGE) / PRIOR_COVERAGE) ** 2)


def _fit_group(cands: Sequence[dict], base_pp: dict,
               scale_grid: Sequence[float], slack_grid: Sequence[int]) -> dict:
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
            c = _cost(*_stats(scaled, pp))
            if c < best_cost:
                best_cost = c
                best = {"score_scale": float(s), "count_slack": int(slack)}
    return best


def calibrate(cands: Dict[str, dict], base_pp: dict,
              scale_grid: Sequence[float] | None = None,
              slack_grid: Sequence[int] | None = None,
              per_district: bool = True) -> Dict[str, dict]:
    """Return {district: {"score_scale": .., "count_slack": ..}} plus '_global'."""
    scale_grid = scale_grid if scale_grid is not None else list(np.round(np.arange(0.5, 2.51, 0.25), 2))
    slack_grid = slack_grid if slack_grid is not None else [-1, 0, 1, 2]
    uids = list(cands)
    out: Dict[str, dict] = {"_global": _fit_group([cands[u] for u in uids], base_pp,
                                                  scale_grid, slack_grid)}
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
        out[g] = _fit_group([cands[u] for u in us], base_pp, scale_grid, slack_grid)
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
