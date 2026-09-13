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


def load_span_model(path: str, ckpt_dir: str, dev):
    """The v2 span model, with its pretrained encoders read from `ckpt_dir`.

    The checkpoint's config records encoder paths as they were on the training
    machine; a local evaluation run keeps the encoder files wherever it likes.
    """
    from src.models.encoders import build_encoder
    from src.models.span_model import build_model
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    cfg["model"]["beats_dir"] = ckpt_dir
    cfg["model"]["atst_ckpt"] = str(Path(ckpt_dir) / "atst_frame.ckpt")
    cfg["model"]["beats_ckpt"] = str(Path(ckpt_dir) / "BEATs_iter3_plus_AS2M.pt")
    le = LabelEncoder(expand_vehicle=bool(cfg["data"].get("expand_vehicle", True)))
    enc = build_encoder(cfg["model"], ckpt_dir=ckpt_dir)
    assert len(enc.encoders) == len(cfg["model"].get("encoders", [])), \
        f"span model needs {cfg['model'].get('encoders')} under {ckpt_dir} - run scripts/fetch_encoders.py --all"
    model = build_model(cfg, len(le), enc)
    missing, _ = model.load_state_dict(ck["model"], strict=False)
    # frozen pretrained encoders may have been stripped on export (export_span_checkpoint);
    # build_encoder has just loaded them from ckpt_dir
    stripped = bool(ck.get("encoders_stripped"))
    bad = [k for k in missing if not k.startswith("head.dgqp")
           and not (stripped and k.startswith(ENCODER_PREFIX))]
    assert not bad, f"span checkpoint is missing {len(bad)} tensors, e.g. {bad[:3]}"
    return model.to(dev).eval(), cfg


ENCODER_PREFIX = "encoder.encoders."


def export_span_checkpoint(src: str, dst: str) -> None:
    """Copy a span-model checkpoint without its frozen pretrained encoder weights.

    The span model trains with frozen encoders (unfreeze_epoch 999), so those ~700 MB are
    byte-identical to the public ATST-Frame / BEATs files fetch_encoders.py downloads.
    Refuses to strip if the run unfroze them.
    """
    ck = torch.load(src, map_location="cpu", weights_only=False)
    ue = int(ck["cfg"].get("train", {}).get("unfreeze_epoch", 999))
    ep = int(ck.get("epoch", 0))
    if ue <= ep:
        torch.save(ck, dst)
        print(f"[export] encoders were unfrozen at epoch {ue}: kept in full")
        return
    ck["model"] = {k: v for k, v in ck["model"].items() if not k.startswith(ENCODER_PREFIX)}
    ck["encoders_stripped"] = True
    torch.save(ck, dst)


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
    ap.add_argument("--f0-ckpt", default="", help="v2 span-model checkpoint (runs/f0/best.pt): natural clips "
                                                  "use its candidates re-ranked by TraceModel")
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

    f0c = {}
    if args.f0_ckpt:
        # natural clips: the v2 span model's proposals, re-ranked by TraceModel (tracesed/fuse.py)
        from tracesed.fuse import f0_candidates
        f0m, f0cfg = load_span_model(args.f0_ckpt, args.ckpt_dir, dev)
        nat = [c for c in clips if not c["syn"]]
        f0c = f0_candidates(f0m, f0cfg["data"], dict(f0cfg.get("postproc", {})), nat, dev)
        del f0m
        torch.cuda.empty_cache()
        print(f"[f0] span candidates for {len(f0c)} natural clips")

    lines, n_ev = [], []
    for c in clips:
        u = c["uid"]; p = post[u]; sub = cfg["synthetic" if c["syn"] else "natural"]
        tr = None if args.no_transcript else meta.get(u, {}).get("transcript")
        fn = DECODERS[sub["decoder"]]
        if u in f0c:
            from tracesed.fuse import rerank_topk
            ev = rerank_topk(f0c[u], p, count_from(p, tr), cfg.get("fuse_w"))
        elif sub["decoder"] == "thr":
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
