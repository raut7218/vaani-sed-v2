"""Diagnostics that say *where* the score is going, not just what it is.

These are the measurements that diagnosed v1, turned into a tool. Run them on
every checkpoint - the headline score tells you almost nothing about which of
the four failure modes you are in.

    python scripts/diagnose.py --ckpt runs/v2/best.pt --data data/vaani --fold 0

Reads as:

* **Constant baseline** - "the whole clip is one event". v1's leaderboard
  submission scored 0.89 against this baseline's 0.919, i.e. the entire model
  was worth less than a one-line heuristic. Always check you have beaten it.
* **Strict vs loose recall** - loose (>50% overlap) says the model *found* the
  event; strict says it *placed* the boundaries. v1 was 0.877 loose and 0.343
  strict: a pure boundary failure that no amount of extra training data fixes.
* **Boundary error by duration** - v1's predicted spans were a median 0.20 s too
  long, and for sub-second events the onset error's 10th percentile was -2.7 s
  (short events swallowed by blobs).
* **Operating point** - predicted coverage and events/clip against the corpus
  prior. v1 submitted at 0.29 coverage where the prior is 0.52, worth ~0.12-0.15
  of score on its own.
* **Selection oracle** - the score if an oracle picked the best subset of the
  candidate spans the model already produced. The gap between this and the
  achieved score is what better selection/calibration can still buy; the gap
  between the oracle and 2.0 is what needs a better model.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import (VaaniSpanDataset, collate, load_manifest,   # noqa: E402
                              split_manifest)
from src.data.labels import LabelEncoder                                  # noqa: E402
from src.evaluation.metrics import evaluate                               # noqa: E402
from src.infer.predict import load_checkpoint                             # noqa: E402
from src.infer.runner import candidates_to_events, run_loader             # noqa: E402
from src.postproc.calibrate import apply_scales, calibrate                # noqa: E402

BUCKETS = [(0, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 1e9)]


def recall_breakdown(preds, refs):
    strict, loose, tot = Counter(), Counter(), Counter()
    for uid, ref in refs.items():
        pv = preds.get(uid, [])
        for a, b in ref:
            d = b - a
            tol = max(0.2 * d, 0.05)
            bi = next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= d < hi)
            tot[bi] += 1
            if any(abs(pa - a) <= tol and abs(pb - b) <= tol for pa, pb in pv):
                strict[bi] += 1
            if any(min(pb, b) - max(pa, a) > 0.5 * d for pa, pb in pv):
                loose[bi] += 1
    return strict, loose, tot


def boundary_errors(preds, refs):
    on, off = [], []
    for uid, ref in refs.items():
        pv = preds.get(uid, [])
        for a, b in ref:
            d = b - a
            best, bo = None, 1e9
            for pa, pb in pv:
                if min(pb, b) - max(pa, a) > 0.5 * d and abs(pa - a) + abs(pb - b) < bo:
                    bo, best = abs(pa - a) + abs(pb - b), (pa, pb)
            if best:
                on.append(best[0] - a)
                off.append(best[1] - b)
    return np.array(on), np.array(off)


def selection_oracle(cands, refs, max_k: int = 6) -> float:
    """Best achievable score from the candidate spans the model already emitted.

    Chooses, per clip, the top-k spans that maximise that clip's own score. This
    isolates *selection* error from *localisation* error: whatever is left
    between this and 2.0 needs a better model, not better post-processing.
    """
    best_preds = {}
    for uid, ref in refs.items():
        c = cands.get(uid)
        if c is None or len(c["spans"]) == 0:
            best_preds[uid] = []
            continue
        order = np.argsort(-c["scores"])
        sp = c["spans"][order]
        bs, bp = -1.0, []
        for k in range(0, min(max_k, len(sp)) + 1):
            p = [[float(a), float(b)] for a, b in sp[:k] if b > a]
            s = evaluate({uid: p}, {uid: ref})["score"]
            if s > bs:
                bs, bp = s, p
        best_preds[uid] = bp
    return evaluate(best_preds, refs)["score"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-calibrate", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(Path(args.ckpt), device)
    d = cfg["data"]
    fps = float(d["fps"])

    le = LabelEncoder(expand_vehicle=bool(d.get("expand_vehicle", True)))
    recs = load_manifest(Path(args.data) / "manifest.jsonl")
    _, va = split_manifest(recs, fold=args.fold, n_folds=int(d.get("n_folds", 5)),
                           seed=int(cfg["seed"]))
    ds = VaaniSpanDataset(va, root=args.data, le=le, clip_len=float(d["clip_len"]),
                          sr=int(d["sr"]), fps=fps, train=False, augment=False)
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate)

    refs, durations = {}, {}
    for batch in ld:
        sp = batch["spans"].numpy()
        fv = batch["frame_valid"].numpy()
        for i, uid in enumerate(batch["uid"]):
            refs[uid] = sorted([[float(a) / fps, float(b) / fps]
                                for a, b in sp[i] if b > a >= 0])
            durations[uid] = float(fv[i].sum()) / fps

    cands = run_loader(model, ld, device, fps, cfg.get("postproc"))
    if not args.no_calibrate:
        sc = calibrate(cands, cfg.get("postproc", {}))
        cands = apply_scales(cands, sc)
        g = sc["_global"]
        print("calibration: global scale %.2f, slack %+d, %d districts fitted"
              % (g["score_scale"], g["count_slack"], len(sc) - 1))
    preds = {u: candidates_to_events(c, cfg.get("postproc")) for u, c in cands.items()}

    r = evaluate(preds, refs)
    const = evaluate({u: [[0.0, durations[u]]] for u in refs}, refs)

    print("\n=== headline ===")
    print("model                F1 %.4f  Dice %.4f  score %.4f  (tp %d fp %d fn %d)"
          % (r["event_f1"], r["segment_dice"], r["score"], r["tp"], r["fp"], r["fn"]))
    print("constant baseline    F1 %.4f  Dice %.4f  score %.4f"
          % (const["event_f1"], const["segment_dice"], const["score"]))
    verdict = "BEATS" if r["score"] > const["score"] else "*** LOSES TO ***"
    print("model %s the whole-clip heuristic by %+.4f"
          % (verdict, r["score"] - const["score"]))

    print("\n=== detection vs localisation ===")
    strict, loose, tot = recall_breakdown(preds, refs)
    print("%12s %7s %12s %12s" % ("duration", "n", "strict rec", "loose rec"))
    for i, (lo, hi) in enumerate(BUCKETS):
        n = tot[i]
        lbl = "%.1f-%.1fs" % (lo, hi) if hi < 1e8 else "%.1fs+" % lo
        print("%12s %7d %12.3f %12.3f"
              % (lbl, n, strict[i] / max(n, 1), loose[i] / max(n, 1)))
    ts, tl, tt = sum(strict.values()), sum(loose.values()), sum(tot.values())
    print("%12s %7d %12.3f %12.3f" % ("ALL", tt, ts / max(tt, 1), tl / max(tt, 1)))
    if tt and tl / tt - ts / tt > 0.25:
        print(">> Large loose-strict gap: this is a BOUNDARY problem, not a "
              "detection problem. More data will not fix it.")

    print("\n=== boundary error ===")
    on, off = boundary_errors(preds, refs)
    if len(on):
        print("onset  mean %+.3f  median %+.3f  MAE %.3f  p10 %+.3f  p90 %+.3f"
              % (on.mean(), np.median(on), np.abs(on).mean(),
                 *np.percentile(on, [10, 90])))
        print("offset mean %+.3f  median %+.3f  MAE %.3f  p10 %+.3f  p90 %+.3f"
              % (off.mean(), np.median(off), np.abs(off).mean(),
                 *np.percentile(off, [10, 90])))
        print("predicted span is longer than the reference by a median %+.3f s"
              % float(np.median(off - on)))

    print("\n=== operating point ===")
    n_ev = float(np.mean([len(v) for v in preds.values()]))
    cov = float(np.mean([sum(b - a for a, b in preds[u]) / max(durations[u], 1e-6)
                         for u in refs]))
    ref_ev = float(np.mean([len(v) for v in refs.values()]))
    ref_cov = float(np.mean([sum(b - a for a, b in refs[u]) / max(durations[u], 1e-6)
                             for u in refs]))
    print("events/clip  pred %.2f  ref %.2f" % (n_ev, ref_ev))
    print("coverage     pred %.3f  ref %.3f" % (cov, ref_cov))
    print("empty clips  pred %.1f%%  ref %.1f%%"
          % (100 * np.mean([len(v) == 0 for v in preds.values()]),
             100 * np.mean([len(v) == 0 for v in refs.values()])))

    print("\n=== ceilings ===")
    orc = selection_oracle(cands, refs)
    print("selection oracle over the model's own candidates: %.4f" % orc)
    print("  -> %+.4f still available from better selection/calibration"
          % (orc - r["score"]))
    print("  -> %+.4f would need a better model" % (2.0 - orc))


if __name__ == "__main__":
    main()
