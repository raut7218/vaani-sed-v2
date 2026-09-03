"""Track 1 metrics - a faithful reimplementation of the Codabench scorer.

The competition's Evaluation tab publishes the exact scoring code, so there is
nothing left to guess. This module reproduces it operation for operation:

  * Event-based F1 - greedy closest-first matching, per-reference tolerance of
    ``max(0.20 * ref_duration, 0.05)`` seconds, micro-averaged over all clips.
  * Segment Dice   - events rasterised to a 10 ms frame grid, macro-averaged
    across clips (a clip with neither reference nor predicted events scores 1.0).
  * Combined       - ``event_f1 + segment_dice``, maximum 2.0.

Deviating from the official definitions here would be self-defeating: the
post-processor is tuned directly against `evaluate`, so any mismatch tunes the
system towards a target the leaderboard does not reward. Keep this file in sync
with the competition page, and change it only alongside that page.

Reference: https://www.codabench.org/competitions/17825/ -> Evaluation
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Event = Tuple[float, float]

TOLERANCE_FRAC = 0.20      # +/- 20% of the reference event's duration ...
MIN_TOLERANCE = 0.05       # ... but never tighter than 50 ms
FRAME_LEN = 0.01           # Dice rasterisation grid: 10 ms
DICE_TAIL_PAD = 0.5        # scorer pads the timeline 0.5 s past the last offset


def match_events(ref: Sequence[Event], pred: Sequence[Event]) -> Tuple[int, int, int]:
    """Greedy closest-first 1-to-1 matching. Returns (tp, fp, fn).

    Every (reference, prediction) pair whose onset *and* offset both fall inside
    the reference's tolerance becomes a candidate, and candidates are consumed in
    order of total boundary error. Sorting globally rather than per reference is
    what makes the result independent of the order events happen to be listed in.
    """
    candidates = []
    for ri, (r_on, r_off) in enumerate(ref):
        tol = max(TOLERANCE_FRAC * (r_off - r_on), MIN_TOLERANCE)
        for pi, (p_on, p_off) in enumerate(pred):
            d_on, d_off = abs(p_on - r_on), abs(p_off - r_off)
            if d_on <= tol and d_off <= tol:
                candidates.append((d_on + d_off, ri, pi))

    matched_ref: set[int] = set()
    matched_pred: set[int] = set()
    for _, ri, pi in sorted(candidates):
        if ri not in matched_ref and pi not in matched_pred:
            matched_ref.add(ri)
            matched_pred.add(pi)

    tp = len(matched_ref)
    return tp, len(pred) - tp, len(ref) - tp


def _rasterise(events: Sequence[Event], n_frames: int) -> int:
    """Events -> a bitset of occupied 10 ms frames, one bit per frame.

    Frame indices are truncated and the offset frame is inclusive, exactly as the
    official `events_to_frames` does. That makes every event at least one frame
    wide, which is what keeps the sub-40 ms `human_non_speech` events scoreable.

    The mask is a Python int used as a bit vector rather than a list or an array:
    the tuner evaluates this on every clip for every parameter trial, and integer
    AND plus `bit_count` does the whole intersection in two C-level operations.
    """
    mask = 0
    for on, off in events:
        a = max(0, int(on / FRAME_LEN))
        b = min(int(off / FRAME_LEN) + 1, n_frames)
        if b > a:
            mask |= ((1 << (b - a)) - 1) << a
    return mask


def clip_dice(pred: Sequence[Event], ref: Sequence[Event]) -> float:
    """Segment Dice for one clip: 2|P n G| / (|P| + |G|) on the frame grid."""
    if not ref and not pred:
        return 1.0
    max_time = max(off for _, off in list(ref) + list(pred)) + DICE_TAIL_PAD
    n_frames = int(max_time / FRAME_LEN) + 1
    r = _rasterise(ref, n_frames)
    p = _rasterise(pred, n_frames)
    total = r.bit_count() + p.bit_count()
    if total == 0:
        return 1.0
    return 2.0 * (r & p).bit_count() / total


def evaluate(preds: Dict[str, List[Event]], refs: Dict[str, List[Event]]) -> dict:
    """Corpus-level score over {clip_id: [(onset, offset), ...]}.

    Clips present in `preds` but absent from `refs` contribute their events as
    false positives, matching the scorer - so emitting predictions for clips that
    are not in the evaluation set costs precision.
    """
    tp = fp = fn = 0
    dice: List[float] = []

    for clip_id, ref in refs.items():
        pred = preds.get(clip_id, [])
        c_tp, c_fp, c_fn = match_events(ref, pred)
        tp += c_tp
        fp += c_fp
        fn += c_fn
        dice.append(clip_dice(pred, ref))

    for clip_id, pred in preds.items():
        if clip_id not in refs:
            fp += len(pred)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    segment_dice = sum(dice) / len(dice) if dice else 0.0

    return {
        "event_f1": round(f1, 5),
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "segment_dice": round(segment_dice, 5),
        # The leaderboard ranks on the sum, not the mean. Max 2.0.
        "score": round(f1 + segment_dice, 5),
        "tp": tp, "fp": fp, "fn": fn,
    }
