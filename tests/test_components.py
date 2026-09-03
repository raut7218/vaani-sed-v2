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
from src.infer.decode import (finalise, merge_close, select_by_count,  # noqa: E402
                              soft_nms_1d, wbf_1d)
from src.models.trident import decode_spans                            # noqa: E402
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
    spans[0, 1] = torch.tensor([30.0, 60.0])       # 30 frames -> level 2
    spans[1, 0] = torch.tensor([5.0, 5.4])         # 0.4 frames -> shorter than a stride
    cls = torch.full((2, 3), -1, dtype=torch.long)
    cls[0, 0] = 1
    cls[0, 1] = 2
    cls[1, 0] = 3
    tier = torch.tensor([0, 0])

    t = assign_targets(spans, cls, tier, masks, T, n_class, n_bins)
    check("assign: 4-frame event lands on level 0", t["pos"][0][0].sum() > 0)
    check("assign: 30-frame event lands on level 2", t["pos"][2][0].sum() > 0)
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


def test_soft_dice():
    logits = torch.tensor([[10.0, 10.0, -10.0, -10.0]])
    tgt = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    m = torch.ones(1, 4)
    check("soft_dice: perfect -> ~0", soft_dice(logits, tgt, m).item() < 1e-3)
    check("soft_dice: inverted -> ~1", soft_dice(-logits, tgt, m).item() > 0.99)


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


if __name__ == "__main__":
    test_metrics()
    test_subframe()
    test_diou()
    test_decode_spans()
    test_assign()
    test_soft_dice()
    test_nms_and_fusion()
    test_district()
    print("\n%d/%d checks passed" % (sum(OK), len(OK)))
    sys.exit(0 if all(OK) else 1)
