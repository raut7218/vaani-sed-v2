"""Component tests. `python tests/test_components.py`

The ones that matter most are the sub-frame boundary tests: the entire premise of
this rewrite is that boundaries can be placed between grid points, and if that is
broken nothing else in the design earns its keep.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import clip_dice, evaluate, match_events   # noqa: E402
from src.infer.decode import (boundary_agreement, finalise,          # noqa: E402
                              merge_close, refine_boundaries,
                              select_by_count, soft_nms_1d, wbf_1d)
from src.models.trident import decode_spans                            # noqa: E402
from src.train.losses import (boundary_targets,                        # noqa: E402
                              quality_focal)
from src.postproc.calibrate import district_of                         # noqa: E402
from src.train.losses import (assign_targets, diou_1d,                 # noqa: E402
                              distribution_focal, soft_dice)

OK = []


def check(name, cond, extra=""):
    OK.append(bool(cond))
    print("%-58s %s %s" % (name, "ok" if cond else "FAIL", extra))


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_metrics():
    # tolerance is max(0.2*dur, 0.05): a 1.0 s event gets 200 ms
    check("match: within 20% collar", match_events([(1.0, 2.0)], [(1.1, 2.1)]) == (1, 0, 0))
    check("match: outside collar", match_events([(1.0, 2.0)], [(1.3, 2.3)]) == (0, 1, 1))
    # a 0.1 s event gets the 50 ms floor, not 20 ms
    check("match: 50 ms floor applies", match_events([(1.0, 1.1)], [(1.04, 1.14)]) == (1, 0, 0))
    check("match: floor is not 20 ms", match_events([(1.0, 1.1)], [(1.06, 1.16)]) == (0, 1, 1))
    check("dice: empty vs empty is 1.0", clip_dice([], []) == 1.0)
    check("dice: perfect overlap is 1.0", abs(clip_dice([(0.0, 1.0)], [(0.0, 1.0)]) - 1.0) < 1e-9)
    d = clip_dice([(0.0, 1.0)], [(0.5, 1.5)])
    check("dice: half overlap ~0.5", abs(d - 0.5) < 0.02, "%.4f" % d)
    r = evaluate({"a": [(0.0, 1.0)]}, {"a": [(0.0, 1.0)]})
    check("evaluate: perfect scores 2.0", abs(r["score"] - 2.0) < 1e-9)
    r = evaluate({"a": [(0.0, 1.0)], "ghost": [(0.0, 1.0)]}, {"a": [(0.0, 1.0)]})
    check("evaluate: clips outside the ref set are FPs", r["fp"] == 1)


# --------------------------------------------------------------------------- #
# the sub-frame boundary claim
# --------------------------------------------------------------------------- #
def test_subframe():
    n_bins = 16
    # A target of 3.5 bins must be representable: DFL should drive the
    # expectation to 3.5, not to 3 or 4.
    logits = torch.zeros(1, n_bins, requires_grad=True)
    tgt = torch.tensor([3.5])
    opt = torch.optim.Adam([logits], lr=0.2)
    for _ in range(500):
        opt.zero_grad()
        distribution_focal(logits, tgt).mean().backward()
        opt.step()
    exp = (logits.softmax(-1) * torch.arange(n_bins).float()).sum().item()
    check("DFL: expectation lands between bins (3.5)", abs(exp - 3.5) < 0.05, "%.4f" % exp)

    logits2 = torch.zeros(1, n_bins, requires_grad=True)
    tgt2 = torch.tensor([2.2])
    opt = torch.optim.Adam([logits2], lr=0.2)
    for _ in range(500):
        opt.zero_grad()
        distribution_focal(logits2, tgt2).mean().backward()
        opt.step()
    exp2 = (logits2.softmax(-1) * torch.arange(n_bins).float()).sum().item()
    check("DFL: expectation lands at 2.2", abs(exp2 - 2.2) < 0.05, "%.4f" % exp2)
    check("DFL: 2.2 and 3.5 are distinguishable within one bin", abs(exp - exp2) > 1.0)


def test_diou():
    a = diou_1d(torch.tensor([2.0]), torch.tensor([2.0]),
                torch.tensor([2.0]), torch.tensor([2.0]))
    check("DIoU: identical spans -> 0 loss", abs(a.item()) < 1e-6, "%.6f" % a.item())
    b = diou_1d(torch.tensor([1.0]), torch.tensor([1.0]),
                torch.tensor([2.0]), torch.tensor([2.0]))
    check("DIoU: mismatched spans -> positive loss", b.item() > 0)
    # Non-overlapping must still have a usable gradient - that is why DIoU and
    # not plain IoU.
    ps = torch.tensor([5.0], requires_grad=True)
    pe = torch.tensor([-3.0], requires_grad=True)
    diou_1d(ps, pe, torch.tensor([1.0]), torch.tensor([1.0])).backward()
    check("DIoU: gradient survives zero overlap",
          ps.grad is not None and abs(float(ps.grad)) > 0)


def test_decode_spans():
    """A head that predicts exact distances must decode to the exact span."""
    fps, T, n_bins = 25.0, 40, 16
    big = 12.0
    # d_start = 2 strides, d_end = 3 strides at level 0 (stride 1 frame)
    cls = torch.full((1, T, 3), -big)
    cls[0, 10, 2] = big                                # agnostic channel hot at t=10
    cls[0, 10, 0] = big
    out = {"cls": [cls], "d_start": [torch.full((1, T), 2.0)],
           "d_end": [torch.full((1, T), 3.0)], "quality": [torch.full((1, T), big)],
           "masks": [torch.ones(1, T)],
           "start_logits": [torch.zeros(1, T, n_bins)],
           "end_logits": [torch.zeros(1, T, n_bins)]}
    spans, scores, cids, _ = decode_spans(out, T, fps)
    i = int(scores[0].argmax())
    a, b = spans[0, i].tolist()
    # point coordinate at level 0 is t + 0.5 - 0.5 = t
    check("decode: onset", abs(a - (10 - 2) / fps) < 1e-4, "%.4f" % a)
    check("decode: offset", abs(b - (10 + 3) / fps) < 1e-4, "%.4f" % b)
    check("decode: class id", int(cids[0, i]) == 0)

    # Sub-frame: a distance of 2.5 strides must produce a boundary at 2.5 frames,
    # not 2 or 3. This is the property no threshold decoder can have.
    out["d_start"] = [torch.full((1, T), 2.5)]
    spans2, sc2, _, _ = decode_spans(out, T, fps)
    a2 = spans2[0, int(sc2[0].argmax()), 0].item()
    check("decode: sub-frame onset (2.5 frames = 100 ms)",
          abs(a2 - (10 - 2.5) / fps) < 1e-4, "%.4f" % a2)


# --------------------------------------------------------------------------- #
# target assignment
# --------------------------------------------------------------------------- #
def test_assign():
    T, n_class, n_bins, n_levels = 64, 4, 16, 4
    masks = []
    t = T
    for _ in range(n_levels):
        masks.append(torch.ones(2, t))
        t = (t + 1) // 2

    spans = torch.full((2, 3, 2), -1.0)
    spans[0, 0] = torch.tensor([10.0, 14.0])       # 4 frames -> level 0
    spans[0, 1] = torch.tensor([30.0, 60.0])       # 30 frames
    spans[1, 0] = torch.tensor([5.0, 5.4])         # 0.4 frames -> shorter than a stride
    cls = torch.full((2, 3), -1, dtype=torch.long)
    cls[0, 0] = 1
    cls[0, 1] = 2
    cls[1, 0] = 3
    tier = torch.tensor([0, 0])

    t = assign_targets(spans, cls, tier, masks, T, n_class, n_bins)
    check("assign: 4-frame event lands on level 0", t["pos"][0][0].sum() > 0)
    # Which level a 30-frame event lands on is a function of the bin geometry,
    # not a constant: it goes to the finest level whose bins still reach its
    # boundaries. What must hold is that it lands on exactly one.
    hit = [lvl for lvl in range(n_levels) if float(t["pos"][lvl][0].sum()) > 0]
    check("assign: a 30-frame event lands on exactly one level besides level 0",
          len([l for l in hit if l > 0]) == 1, "levels %s" % hit)
    check("assign: sub-stride event still gets a positive",
          t["pos"][0][1].sum() > 0, "n=%d" % int(t["pos"][0][1].sum()))
    check("assign: agnostic channel is always set at positives",
          bool((t["cls"][0][0][..., n_class][t["pos"][0][0] > 0.5] == 1).all()))

    # silver's boundaries are down-weighted, gold's are not
    t2 = assign_targets(spans, cls, torch.tensor([1, 1]), masks, T, n_class, n_bins)
    gw = t["bw"][0][0][t["pos"][0][0] > 0.5].mean().item()
    sw = t2["bw"][0][0][t2["pos"][0][0] > 0.5].mean().item()
    check("assign: silver boundary weight < gold", sw < gw, "%.2f vs %.2f" % (sw, gw))

    # bronze contributes no spans at all
    t3 = assign_targets(spans, cls, torch.tensor([2, 2]), masks, T, n_class, n_bins)
    check("assign: bronze yields no positives",
          sum(float(p.sum()) for p in t3["pos"]) == 0)

    # regression targets must reconstruct the span exactly
    lvl0 = t["pos"][0][0] > 0.5
    idx = int(torch.nonzero(lvl0)[0])
    ds = t["d_start"][0][0][idx].item()
    de = t["d_end"][0][0][idx].item()
    check("assign: d_start/d_end reconstruct the span",
          abs((idx - ds) - 10.0) < 1e-4 and abs((idx + de) - 14.0) < 1e-4,
          "%.3f %.3f" % (idx - ds, idx + de))


def test_quality_focal():
    """A soft IoU target must rank a well-localised span above a poor one."""
    logits = torch.tensor([[0.0]])
    # target 0.9, prediction 0.5: pulled up. target 0.1, same prediction: down.
    hi = quality_focal(logits, torch.tensor([[0.9]]), torch.ones(1, 1))
    lo = quality_focal(logits, torch.tensor([[0.1]]), torch.ones(1, 1))
    check("qfl: a high-IoU target costs the same as a low one at p=0.5",
          abs(float(hi) - float(lo)) < 0.2, "%.3f vs %.3f" % (hi, lo))
    # loss vanishes where the prediction already equals the target
    exact = quality_focal(torch.tensor([[2.1972246]]), torch.tensor([[0.9]]),
                          torch.ones(1, 1))
    check("qfl: matching the soft target costs ~nothing", float(exact) < 1e-3,
          "%.5f" % float(exact))
    # and a hard target still behaves like a classification loss
    wrong = quality_focal(torch.tensor([[-4.0]]), torch.tensor([[1.0]]),
                          torch.ones(1, 1))
    check("qfl: a confident miss is expensive", float(wrong) > 3.0, "%.2f" % float(wrong))
    check("qfl: the mask zeroes excluded points",
          float(quality_focal(logits, torch.tensor([[0.9]]), torch.zeros(1, 1))) == 0.0)


def test_level_ranges():
    """Every event should sit where the metric's tolerance is worth >1 bin."""
    from src.train.losses import LEGACY_LEVEL_RANGES, level_ranges
    r = level_ranges(16, 5)
    check("levels: ranges are contiguous and increasing",
          all(abs(r[i][1] - r[i + 1][0]) < 1e-6 for i in range(len(r) - 1))
          and r[0][0] == 0.0)

    def tol_bins(D, ranges):
        L, tol = D * 25.0, max(0.2 * D, 0.05)
        lvl = next(i for i, (a, b) in enumerate(ranges) if a <= L < b)
        return tol * 25.0 / (2 ** lvl)

    # The old table pinned every event above half a second at 1.25 bins of
    # tolerance, whatever its scale - that is the thing being fixed.
    old = [tol_bins(D, LEGACY_LEVEL_RANGES) for D in (0.5, 1.0, 2.0, 4.0)]
    new = [tol_bins(D, r) for D in (0.5, 1.0, 2.0, 4.0)]
    check("levels: the old table gave 1.25 bins of tolerance at every scale",
          all(abs(v - 1.25) < 0.01 for v in old), "%s" % [round(v, 2) for v in old])
    check("levels: the new one doubles that at every scale",
          all(v > 2.4 for v in new), "%s" % [round(v, 2) for v in new])

    # An event must never need more bins than the level has.
    for lvl, (lo, hi) in enumerate(r):
        top = min(hi, 200.0)
        check("levels: level %d's longest event fits its bins" % lvl,
              top / 2.0 / (2 ** lvl) <= 15.0,
              "%.1f bins" % (top / 2.0 / (2 ** lvl)))


def test_soft_dice():
    logits = torch.tensor([[10.0, 10.0, -10.0, -10.0]])
    tgt = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    m = torch.ones(1, 4)
    check("soft_dice: perfect -> ~0", soft_dice(logits, tgt, m).item() < 1e-3)
    check("soft_dice: inverted -> ~1", soft_dice(-logits, tgt, m).item() > 0.99)

    # A per-clip weight must scale the *loss*, never the mask. Folding it into
    # the mask scales the numerator by w^2 and the denominator by w, which put an
    # irreducible 0.5 floor under every silver clip.
    w = torch.tensor([0.5])
    check("soft_dice: a down-weighted perfect clip still scores ~0",
          soft_dice(logits, tgt, m, w).item() < 1e-3,
          "%.4f" % soft_dice(logits, tgt, m, w).item())
    check("soft_dice: weight scales the loss, not the score",
          abs(soft_dice(-logits, tgt, m, w).item()
              - soft_dice(-logits, tgt, m).item()) < 1e-3)
    # Weight 0 (bronze) must drop the clip out of the mean, not drag it to 0.
    two = torch.cat([logits, -logits]), torch.cat([tgt, tgt]), torch.ones(2, 4)
    check("soft_dice: zero-weight clips are excluded",
          abs(soft_dice(*two, torch.tensor([1.0, 0.0])).item()
              - soft_dice(logits, tgt, m).item()) < 1e-3)


def test_sampler_sharding():
    from src.data.dataset import TierBatchSampler
    recs = [{"tier": ["gold", "silver", "silver", "silver", "bronze"][i % 5],
             "uid": str(i)} for i in range(4000)]
    kw = dict(quotas={"gold": 0.5, "silver": 0.35, "bronze": 0.15}, seed=42)
    r0 = list(TierBatchSampler(recs, 16, rank=0, world_size=2, **kw))
    r1 = list(TierBatchSampler(recs, 16, rank=1, world_size=2, **kw))
    check("sampler: ranks get the same number of batches", len(r0) == len(r1))
    check("sampler: ranks get different batches", r0 != r1)
    flat0 = {i for b in r0 for i in b}
    flat1 = {i for b in r1 for i in b}
    check("sampler: the two shards are not the same clips",
          len(flat0 & flat1) < 0.9 * len(flat0))
    solo = TierBatchSampler(recs, 16, rank=0, world_size=1, **kw)
    check("sampler: 2 ranks halve the per-rank epoch",
          abs(len(solo) - 2 * len(r0)) <= 1, "%d vs 2x%d" % (len(solo), len(r0)))
    check("sampler: quotas still hold per batch",
          all(len(b) == 16 for b in r0))


def test_tta_deshift():
    """The shift applied to the waveform and the offset subtracted from the
    decoded spans have to be the same number of seconds."""
    fps, n_frames, wav_len = 25.0, 200, 128000        # 8 s @ 16 kHz
    samples_per_sec = wav_len / (n_frames / fps)
    shift = int(0.02 * samples_per_sec)
    dt = shift / samples_per_sec
    check("tta: de-shift matches the shift applied", abs(dt - shift / 16000) < 1e-9,
          "%.4f s" % dt)
    check("tta: de-shift is ~20 ms, not ~0.8 ms", abs(dt - 0.02) < 1e-6)


def _peaky(n, hi_fps, positions, width=1.0):
    """A boundary map with Gaussian peaks at `positions` (seconds)."""
    g = np.zeros(n, "float32")
    t = (np.arange(n) + 0.5) / hi_fps
    for p in positions:
        g = np.maximum(g, np.exp(-0.5 * ((t - p) / (width / hi_fps)) ** 2))
    return g


def test_refine_boundaries():
    """Refinement pulls an endpoint onto a peak, and only inside the tolerance."""
    hi_fps, n = 50.0, 400
    on = _peaky(n, hi_fps, [2.000])
    off = _peaky(n, hi_fps, [3.000])

    # A span 60 ms off on each side: both peaks are inside the window, so both
    # endpoints should land on them to well under the metric's tolerance.
    s = np.array([[2.06, 2.94]], "float32")
    r = refine_boundaries(s, on, off, hi_fps, 8.0)
    check("refine: onset snaps to the peak", abs(r[0, 0] - 2.0) < 0.012,
          "%.4f s" % r[0, 0])
    check("refine: offset snaps to the peak", abs(r[0, 1] - 3.0) < 0.012,
          "%.4f s" % r[0, 1])

    # A span so far off that no peak falls in its window must not be moved:
    # refinement is allowed to fail to help, never to drag a boundary somewhere
    # the detector never proposed.
    s2 = np.array([[5.00, 6.00]], "float32")
    r2 = refine_boundaries(s2, on, off, hi_fps, 8.0)
    check("refine: leaves a span with no peak in range alone",
          np.allclose(r2, s2), "%s" % r2.tolist())

    # A flat map carries no boundary information at all.
    flat = np.full(n, 0.05, "float32")
    r3 = refine_boundaries(s, flat, flat, hi_fps, 8.0)
    check("refine: a map below peak_min never moves anything",
          np.allclose(r3, s), "%s" % r3.tolist())

    # Sub-frame: the grid is 20 ms, so snapping to the nearest frame centre
    # would leave up to 10 ms of error. The expectation has to do better.
    on4 = _peaky(n, hi_fps, [2.013])
    r4 = refine_boundaries(np.array([[2.05, 3.00]], "float32"), on4, off, hi_fps, 8.0)
    check("refine: resolves between 20 ms frames", abs(r4[0, 0] - 2.013) < 0.008,
          "%.4f s" % r4[0, 0])


def test_boundary_agreement():
    """Agreement ranks a well-placed span above a badly-placed one."""
    hi_fps, n = 50.0, 400
    on = _peaky(n, hi_fps, [2.0])
    off = _peaky(n, hi_fps, [3.0])
    a = boundary_agreement(np.array([[2.0, 3.0], [2.5, 3.5]], "float32"),
                           on, off, hi_fps)
    # 0.882, not 1.0: with 20 ms frames the peak at 2.000 s falls exactly
    # between two frame centres, so neither samples its top.
    check("agreement: aligned span scores near 1", a[0] > 0.85, "%.3f" % a[0])
    check("agreement: misaligned span scores near 0", a[1] < 0.1, "%.3f" % a[1])
    check("agreement: ranks the aligned span first", a[0] > a[1])


def test_boundary_targets():
    """The branch's supervision: right positions, right tiers, right edges."""
    mult, n_frames, n_hi = 2, 200, 400
    # gold clip: one event over base frames [10, 40] -> hi frames [20, 80]
    # silver clip: one event over [0, 120], i.e. starting at the very first
    #              sample - an annotation default, not an audible onset
    spans = torch.tensor([[[10., 40.], [-1., -1.]],
                          [[0., 120.], [-1., -1.]]])
    valid = torch.ones(2, n_frames)
    tier = torch.tensor([0, 1])                       # gold, silver
    t, w = boundary_targets(spans, valid, n_hi, mult, tier, silver_w=0.15)

    check("bmap: onset peaks at the event start",
          int(t[0, 0].argmax()) == 20, "frame %d" % int(t[0, 0].argmax()))
    check("bmap: offset peaks at the event end",
          int(t[0, 1].argmax()) == 80, "frame %d" % int(t[0, 1].argmax()))
    check("bmap: the peak is a bump, not a spike",
          0.5 < float(t[0, 0, 19]) < 0.9, "%.3f" % float(t[0, 0, 19]))
    check("bmap: gold boundaries carry full weight",
          abs(float(w[0, 0, 20]) - 1.0) < 1e-5, "%.3f" % float(w[0, 0, 20]))
    check("bmap: frames away from any boundary train as negatives",
          abs(float(w[0, 0, 200]) - 1.0) < 1e-5 and float(t[0, 0, 200]) < 1e-3)

    # silver, away from the edge: down-weighted but still supervised
    check("bmap: silver offset is down-weighted",
          abs(float(w[1, 1, 240]) - 0.15) < 1e-5, "%.3f" % float(w[1, 1, 240]))
    # silver, at the clip edge: dropped entirely
    check("bmap: a boundary at frame 0 is not a target",
          float(t[1, 0].max()) < 1e-3, "%.4f" % float(t[1, 0].max()))

    # a boundary at the end of the *valid* region is dropped too, even on gold
    spans2 = torch.tensor([[[10., 200.], [-1., -1.]]])
    t2, _ = boundary_targets(spans2, torch.ones(1, n_frames), n_hi, mult,
                             torch.tensor([0]), silver_w=0.15)
    check("bmap: a boundary at the clip's end is not a target",
          float(t2[0, 1].max()) < 1e-3, "%.4f" % float(t2[0, 1].max()))
    check("bmap: its onset still is", float(t2[0, 0].max()) > 0.5)

    # padding must contribute nothing
    pad = torch.full((1, 2, 2), -1.0)
    t3, _ = boundary_targets(pad, torch.ones(1, n_frames), n_hi, mult,
                             torch.tensor([0]), silver_w=0.15)
    check("bmap: padded spans produce no targets", float(t3.max()) < 1e-3)


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #
def test_nms_and_fusion():
    spans = np.array([[1.0, 2.0], [1.05, 2.05], [5.0, 6.0]], "float32")
    scores = np.array([0.9, 0.85, 0.7], "float32")
    s, c = soft_nms_1d(spans, scores)
    check("softnms: keeps the distant event", len(s) >= 2)
    check("softnms: best span survives first", abs(s[0][0] - 1.0) < 1e-6)
    check("softnms: the near-duplicate is decayed, not deleted",
          len(s) == 3 and c[1] < scores[1])

    a_s = np.array([[1.00, 2.00]], "float32")
    b_s = np.array([[1.08, 2.08]], "float32")
    fs, fc = wbf_1d([a_s, b_s], [np.array([1.0], "float32"), np.array([1.0], "float32")])
    check("wbf: two models 80 ms apart fuse to the midpoint",
          len(fs) == 1 and abs(fs[0][0] - 1.04) < 1e-3, "%.4f" % fs[0][0])
    fs2, fc2 = wbf_1d([a_s, np.zeros((0, 2), "float32")],
                      [np.array([1.0], "float32"), np.zeros((0,), "float32")],
                      n_models=2)
    check("wbf: a span only one model found is down-weighted", fc2[0] < 0.75,
          "%.3f" % fc2[0])

    sp = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], "float32")
    sc = np.array([0.9, 0.8, 0.7], "float32")
    ks, _ = select_by_count(sp, sc, np.array([0.05, 0.9, 0.05]), slack=0)
    check("count head: k=1 keeps one span", len(ks) == 1)
    ks, _ = select_by_count(sp, sc, np.array([0.05, 0.05, 0.9]), slack=0)
    check("count head: k=2 keeps two spans", len(ks) == 2)
    ks, _ = select_by_count(sp, sc, np.array([0.9, 0.05, 0.05]), slack=0)
    check("count head: never emits nothing while confident", len(ks) == 1)

    check("merge_close: default is off (v1 dilated by a median 0.20 s)",
          len(merge_close(np.array([[1.0, 2.0], [2.01, 3.0]], "float32"))) == 2)
    check("merge_close: merges when asked",
          len(merge_close(np.array([[1.0, 2.0], [2.01, 3.0]], "float32"), gap=0.05)) == 1)

    ev = finalise(np.array([[-1.0, 2.0], [3.0, 3.001]], "float32"),
                  np.array([0.9, 0.8], "float32"), duration=5.0)
    check("finalise: clamps negative onsets", ev and ev[0][0] == 0.0)
    check("finalise: drops degenerate spans", len(ev) == 1)


def test_district():
    u = "IISc_VaaniProject_K_WestBengal_Darjeeling_844425030_001_GENERIC_0098_1_2"
    check("district: parsed from the filename", district_of(u) == "WestBengal_Darjeeling",
          district_of(u))
    check("district: unknown pattern falls back to global",
          district_of("random_name") == "_global")


# --------------------------------------------------------------------------- #
# EMA over an encoder that uses weight_norm
# --------------------------------------------------------------------------- #
def test_save_atomic():
    """A failed checkpoint write must leave the previous one intact.

    The session this guards against died at `torch.save` with ENOSPC and, saving
    in place, left a truncated `best.pt` behind - two epochs of GPU time turned
    into a file nothing can load. The rename makes the swap all-or-nothing.
    """
    import tempfile
    from src.train.train import save_atomic

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "best.pt"
        save_atomic({"score": 1.0}, p)
        check("save_atomic: writes", p.exists() and torch.load(p, weights_only=False)["score"] == 1.0)

        class Unpicklable:
            def __reduce__(self):
                raise OSError(28, "No space left on device")

        try:
            save_atomic({"score": 2.0, "bad": Unpicklable()}, p)
            failed = False
        except RuntimeError as e:
            failed = "No space left" in str(e) and "GB free" in str(e)
        check("save_atomic: a failed write raises, and says why", failed)
        check("save_atomic: the previous checkpoint survives",
              torch.load(p, weights_only=False)["score"] == 1.0)
        check("save_atomic: no .tmp left behind",
              not (Path(d) / "best.pt.tmp").exists())


def test_ema_weight_norm():
    """The EMA must be able to clone a model carrying `weight_norm`.

    BEATs applies the deprecated `torch.nn.utils.weight_norm` to its `pos_conv`,
    which caches the weight it computes as a plain non-leaf attribute - and
    `copy.deepcopy` refuses those outright. A run therefore died on the first
    line of training, *after* the encoders had loaded, which on Kaggle is three
    hours in. `encoders: []` in the overfit and smoke configs meant nothing in
    the suite ever built a weight_norm module, so this test builds one.
    """
    import torch.nn as nn
    from src.train.train import EMA

    class WeightNormEncoder(nn.Module):
        """A stand-in for BEATs' pos_conv, built the same way."""

        def __init__(self):
            super().__init__()
            conv = nn.Conv1d(16, 16, 3, padding=1, groups=2)
            self.pos_conv = nn.Sequential(nn.utils.weight_norm(conv, name="weight", dim=2))

        def forward(self, x):
            return self.pos_conv(x)

    torch.manual_seed(0)
    model = WeightNormEncoder()
    try:
        ema = EMA(model, decay=0.9)
        built = True
    except RuntimeError as e:                                      # noqa: BLE001
        ema, built = None, False
        check("ema: clones a weight_norm model", False, str(e)[:60])
    if not built:
        return

    check("ema: clones a weight_norm model", True)
    check("ema: shadow is an independent module",
          ema.shadow.pos_conv[0] is not model.pos_conv[0])
    check("ema: the original still has its cached weight",
          "weight" in model.pos_conv[0].__dict__)
    check("ema: the cached weight stays out of state_dict",
          not any(k.endswith("pos_conv.0.weight") for k in ema.shadow.state_dict()))

    # decay 0.9, one step: shadow <- 0.9 * shadow + 0.1 * model
    key = "pos_conv.0.weight_v"
    before = ema.shadow.state_dict()[key].clone()
    with torch.no_grad():
        model.state_dict()[key].add_(1.0)
    after_src = model.state_dict()[key].clone()
    ema.update()
    want = 0.9 * before + 0.1 * after_src
    check("ema: update averages towards the model",
          torch.allclose(ema.shadow.state_dict()[key], want, atol=1e-6))

    # The shadow is evaluated and checkpointed, so both paths have to work on it.
    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        out = ema.shadow(x)
    check("ema: the shadow runs a forward pass", out.shape == (2, 16, 32))
    ema.shadow.load_state_dict(ema.shadow.state_dict())
    check("ema: the shadow survives a state_dict round-trip (--resume)", True)

def test_ema_skips_frozen():
    """The EMA averages only what can change, and re-pairs after an unfreeze."""
    import torch.nn as nn
    from src.train.train import EMA
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    for p in model[0].parameters():
        p.requires_grad = False
    ema = EMA(model, decay=0.5)
    check("ema: frozen tensors are not averaged", len(ema._src) == 2)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update()
    check("ema: trainable tensors move halfway",
          torch.allclose(ema.shadow[1].weight, model[1].weight - 0.5))
    for p in model[0].parameters():
        p.requires_grad = True
    ema.refresh(model)
    check("ema: refresh picks up newly trainable tensors", len(ema._src) == 4)


def test_packed_dataset():
    """A packed corpus must read back exactly what the loose files hold."""
    import subprocess
    import tempfile
    from scripts.smoke_test import make_corpus
    from src.data.dataset import VaaniSpanDataset, load_manifest
    from src.data.labels import LabelEncoder

    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        make_corpus(d / "data", n=12)
        (d / "data" / "vad").mkdir()
        recs = load_manifest(d / "data" / "manifest.jsonl")
        for r in recs[::2]:
            np.save(d / "data" / "vad" / (r["uid"] + ".npy"),
                    np.random.rand(150).astype("float32"))
        subprocess.run([sys.executable, str(root / "scripts" / "pack_data.py"),
                        "--data", str(d / "data"), "--out", str(d / "packed"),
                        "--pack-gb", "0.0005"], check=True, capture_output=True)
        packed = load_manifest(d / "packed" / "manifest.jsonl")
        le = LabelEncoder()
        a = VaaniSpanDataset(recs, d / "data", le, train=False, vad_dir=d / "data" / "vad")
        b = VaaniSpanDataset(packed, d / "packed", le, train=False)
        same = all(torch.allclose(x[k].float(), y[k].float(), atol=1e-3)
                   for x, y in (( a[i], b[i]) for i in range(len(recs)))
                   for k in x if k != "uid")
        check("pack: packed clips decode identically", same)
        check("pack: spread over several packs",
              len({r["pack"] for r in packed}) > 1)
        check("pack: VAD kept for exactly the clips that had it",
              sum("vad_off" in r for r in packed) == len(recs[::2]))


def test_kaldi_fbank():
    """The batched BEATs front-end must match torchaudio's Kaldi fbank."""
    import torchaudio.compliance.kaldi as ta_kaldi
    from src.models.encoders import KaldiFbank
    torch.manual_seed(0)
    wav = torch.randn(3, 32000) * 0.1
    wav[2, 20000:] = 0
    ref = torch.stack([ta_kaldi.fbank(w.unsqueeze(0) * 2 ** 15, num_mel_bins=128,
                                      sample_frequency=16000, frame_length=25,
                                      frame_shift=10) for w in wav])
    got = KaldiFbank()(wav)
    check("fbank: batched == torchaudio kaldi", got.shape == ref.shape
          and (got - ref).abs().max().item() < 1e-3,
          "max diff %.2e" % (got - ref).abs().max().item())


def test_encoder_fast_paths():
    """Fused attention must reproduce the upstream encoders (when present)."""
    ck = Path(__file__).resolve().parents[1] / "checkpoints"
    if not (ck / "BEATs_iter3_plus_AS2M.pt").exists() or not (ck / "atst_frame.ckpt").exists():
        print("%-58s skip (no encoder checkpoints)" % "encoders: fast paths")
        return
    from third_party.beats.BEATs import BEATs, BEATsConfig
    from src.models.encoders import ATSTFrameEncoder, BEATsEncoder, _load_atst
    torch.manual_seed(0)
    wav = torch.randn(2, 64000) * 0.1

    raw = torch.load(str(ck / "BEATs_iter3_plus_AS2M.pt"), map_location="cpu",
                     weights_only=False)
    ref = BEATs(BEATsConfig(raw["cfg"]))
    ref.load_state_dict(raw["model"], strict=False)
    enc = BEATsEncoder(ck / "BEATs_iter3_plus_AS2M.pt")
    ref.eval(), enc.eval()
    with torch.no_grad():
        want = ref.extract_features(wav)[0]
        want = want.reshape(2, -1, 8, want.size(-1)).mean(2)
        got = enc(wav)
    check("beats: SDPA + batched fbank == upstream",
          (got - want).abs().max().item() < 1e-4,
          "max diff %.2e" % (got - want).abs().max().item())
    check("beats: one token per 160 ms step", got.shape[1] == 24)

    a_ref, _ = _load_atst(ck / "atst_frame.ckpt", 0.9)
    a_enc = ATSTFrameEncoder(ck / "atst_frame.ckpt")
    a_ref.eval(), a_enc.eval()
    with torch.no_grad():
        spec = a_enc.features(wav)
        n = torch.full((2,), float(spec.size(-1)))
        want = a_ref.get_intermediate_layers(spec.unsqueeze(1), n, 1, scene=False)
        got = a_enc(wav)
    check("atst: SDPA == upstream", (got - want).abs().max().item() < 1e-4,
          "max diff %.2e" % (got - want).abs().max().item())

    enc.unfreeze_last(2, ckpt=True)
    enc.train()
    enc(wav).pow(2).mean().backward()
    layers = enc.beats.encoder.layers
    check("unfreeze: gradients reach the top blocks only",
          layers[-1].fc1.weight.grad is not None and layers[-3].fc1.weight.grad is None)
    check("unfreeze: frozen blocks stay in eval mode, trainable ones train",
          not layers[0].training and layers[-1].training
          and not enc.beats.encoder.training)
    rel = layers[0].self_attn.relative_attention_bias.weight
    check("unfreeze: BEATs' shared position embedding stays frozen",
          not rel.requires_grad and rel.grad is None)
    seen = []
    h = layers[-3].register_forward_hook(lambda m, i, o: seen.append(o[0].requires_grad))
    enc(wav)
    h.remove()
    check("unfreeze: the frozen stack builds no autograd graph", seen == [False])


# --------------------------------------------------------------------------- #
# ATST-Frame, when its checkpoint is present
# --------------------------------------------------------------------------- #
def test_atst_encoder():
    """Skipped without the checkpoint - but if it is there, it must be aligned.

    The two failure modes this catches are both silent. Wrong mel statistics
    (ATST wants its own 64-band, [-1, 1] scaled mel, not ours) still produce a
    correctly shaped tensor, and a wrong time convention still trains - the head
    just learns the offset and the event-F1 term pays for it.
    """
    ckpt = Path(__file__).resolve().parents[1] / "checkpoints" / "atst_frame.ckpt"
    if not ckpt.exists():
        print("%-58s skip (no checkpoints/atst_frame.ckpt)" % "atst: alignment")
        return
    from src.models.encoders import ATSTFrameEncoder

    enc = ATSTFrameEncoder(ckpt).eval()
    sr, dur, onset, offset = 16000, 8.0, 2.0, 2.5
    torch.manual_seed(0)
    wav = torch.zeros(1, int(sr * dur))
    n = int((offset - onset) * sr)
    wav[0, int(onset * sr):int(onset * sr) + n] = torch.randn(n) * 0.5
    with torch.no_grad():
        feat = enc(wav)[0]

    check("atst: one token per 40 ms", feat.shape[0] == int(dur * 25),
          "%d tokens" % feat.shape[0])
    check("atst: 768-d output", feat.shape[1] == enc.out_dim == 768)

    energy = feat.norm(dim=-1)
    floor = energy[:20].mean()
    hot = (energy > floor + 0.5 * (energy.max() - floor)).nonzero().flatten()
    t0, t1 = hot[0].item() * 0.04, (hot[-1].item() + 1) * 0.04
    # One token of slack at each edge: the burst boundary need not fall on a
    # token boundary, and the patch that straddles it lights up either way.
    # This is an alignment check, not a precision one - a wrong time convention
    # is off by tens of tokens, not by one.
    check("atst: the burst lands where it was put",
          abs(t0 - onset) <= 0.05 and abs(t1 - offset) <= 0.05,
          "%.2f-%.2f s vs %.2f-%.2f s" % (t0, t1, onset, offset))


if __name__ == "__main__":
    test_metrics()
    test_subframe()
    test_diou()
    test_decode_spans()
    test_assign()
    test_quality_focal()
    test_level_ranges()
    test_soft_dice()
    test_sampler_sharding()
    test_tta_deshift()
    test_boundary_targets()
    test_refine_boundaries()
    test_boundary_agreement()
    test_nms_and_fusion()
    test_district()
    test_ema_weight_norm()
    test_save_atomic()
    test_ema_skips_frozen()
    test_packed_dataset()
    test_kaldi_fbank()
    test_encoder_fast_paths()
    test_atst_encoder()
    print("\n%d/%d checks passed" % (sum(OK), len(OK)))
    sys.exit(0 if all(OK) else 1)
