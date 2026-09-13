"""Test-set inference -> predictions.jsonl inside submission.zip.

python -m tracesed.predict --ckpt model.pt [model2.pt ...] --audio-dir <test audio root> \
    [--meta <test metadata json>] --decode config.json --out submission.zip

* Full-length inference: 8 s windows, 4 s hop, triangular stitching (no truncation).
* Several checkpoints are averaged at the posterior level (same architecture and grid).
* Routing by `syntheticData`: natural and synthetic clips get their own decoder settings.
* K per clip: the transcript's noise-tag count when the metadata carries transcripts,
  otherwise the count head.
* Every clip found in the audio dir appears exactly once, `[]` when nothing is detected.
"""
from __future__ import annotations

import argparse
import glob
import json
import zipfile
from pathlib import Path

import numpy as np
import torch

from src.data.labels import LabelEncoder
from tracesed.decode import DECODERS, count_from
from tracesed.model import TraceModel
from tracesed.train import infer_clips

DEFAULT_DECODE = {
    "natural": {"decoder": "epn", "kw": {"floor": 0.35, "w_b": 0.5, "sigma": 0.1, "iou": 0.5}},
    "synthetic": {"decoder": "epn", "kw": {"floor": 0.35, "w_b": 0.5, "sigma": 0.1, "iou": 0.5}},
}


def find_meta(root: str, given: str) -> dict:
    paths = [given] if given else [p for p in glob.glob(f"{root}/**/*.json", recursive=True) if "__MACOSX" not in p]
    for p in paths:
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:                                           # noqa: BLE001
            continue
        if isinstance(d, list) and d and isinstance(d[0], dict) and "segmentFileName" in d[0]:
            print(f"[meta] {p}: {len(d)} records, transcripts on "
                  f"{sum(bool(r.get('transcript')) for r in d)} clips")
            return {r["segmentFileName"].rsplit(".", 1)[0]: r for r in d}
    print("[meta] no metadata found - K from the count head, all clips decoded as natural")
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--meta", default="")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--decode", default="", help="JSON file overriding DEFAULT_DECODE")
    ap.add_argument("--no-transcript", action="store_true", help="ignore transcripts even if present")
    ap.add_argument("--out", default="/kaggle/working/submission.zip")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(DEFAULT_DECODE)
    if args.decode:
        cfg.update(json.load(open(args.decode)))
    meta = find_meta(args.audio_dir, args.meta)
    # Clean references share their noisy clip's file name; scoring one would emit a
    # duplicate clip_id whose events describe silence.
    files = sorted(p for p in glob.glob(f"{args.audio_dir}/**/*.wav", recursive=True)
                   if "__MACOSX" not in p and "cleanref" not in p.lower())
    seen, clips = set(), []
    for f in files:
        u = Path(f).stem
        if u in seen or (meta and u not in meta):
            continue
        seen.add(u)
        clips.append(dict(uid=u, path=f, syn=bool(meta.get(u, {}).get("syntheticData", False)), events=[]))
    if meta and len(clips) != len(meta):
        print(f"[audio] WARNING: {len(meta) - len(clips)} metadata clips have no audio file")
    print(f"[audio] {len(clips)} clips ({sum(c['syn'] for c in clips)} synthetic)")

    le = LabelEncoder(expand_vehicle=True)
    post = None
    for ck in args.ckpt:
        state = torch.load(ck, map_location="cpu", weights_only=False)
        enc = tuple(state["args"].get("encoders", "atst_frame,beats").split(","))
        model = TraceModel(len(le), args.ckpt_dir, encoders=enc)
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state["model"].items()})
        model.to(dev).eval()
        p = infer_clips(model, clips, dev)
        if post is None:
            post = p
        else:
            for u in post:
                for k in ("pres", "bnd", "ext", "count"):
                    post[u][k] = post[u][k].astype(np.float32) + p[u][k].astype(np.float32)
        del model
        torch.cuda.empty_cache()
    if len(args.ckpt) > 1:
        for u in post:
            for k in ("pres", "bnd", "ext", "count"):
                post[u][k] = post[u][k] / len(args.ckpt)

    lines, n_ev = [], []
    for c in clips:
        u = c["uid"]; p = post[u]; sub = cfg["synthetic" if c["syn"] else "natural"]
        tr = None if args.no_transcript else meta.get(u, {}).get("transcript")
        fn = DECODERS[sub["decoder"]]
        if sub["decoder"] == "thr":
            ev = fn(p, **sub["kw"])
        elif sub["decoder"] == "hsmm":
            ev = fn(p, count_from(p, tr), c["syn"], **sub["kw"])
        else:
            ev = fn(p, count_from(p, tr), **sub["kw"])
        ev = sorted((round(max(0.0, a), 3), round(min(b, p["dur"]), 3)) for a, b in ev if b > a)
        n_ev.append(len(ev))
        lines.append(json.dumps({"clip_id": u, "events": [{"onset": a, "offset": b} for a, b in ev]}, ensure_ascii=False))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("predictions.jsonl", "\n".join(lines) + "\n")
    print(f"[done] {args.out}: {len(lines)} clips, {np.mean(n_ev):.2f} events/clip, "
          f"{100 * np.mean([n == 0 for n in n_ev]):.1f}% empty")


if __name__ == "__main__":
    main()
