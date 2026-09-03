"""Inference -> submission.zip.

Submission rules the writer enforces for you (unchanged from v1, they were
right): every evaluation clip appears exactly once with `[]` when nothing is
detected, `predictions.jsonl` sits at the archive root, times are
milliseconds-rounded, non-negative and non-decreasing, and clips that are not in
the evaluation set are never emitted (their events would count as false
positives).
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.labels import LabelEncoder                                   # noqa: E402
from src.infer.runner import (DEFAULT_POSTPROC, candidates_to_events,      # noqa: E402
                              fuse_candidates, run_loader)
from src.models.encoders import build_encoder                              # noqa: E402
from src.models.span_model import build_model                              # noqa: E402
from src.postproc.calibrate import apply_scales, calibrate                 # noqa: E402

AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


class AudioDirDataset(Dataset):
    """Raw audio directory -> the tensors the model expects. No labels."""

    def __init__(self, files: List[Path], sr: int = 16000, clip_len: float = 8.0,
                 fps: float = 25.0):
        self.files = files
        self.sr, self.fps = int(sr), float(fps)
        self.n_samples = int(round(clip_len * sr))
        self.n_frames = int(round(clip_len * fps))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> dict:
        import soundfile as sf
        p = self.files[i]
        try:
            y, sr = sf.read(str(p), dtype="float32", always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != self.sr:
                import librosa
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
        except Exception as e:                                        # noqa: BLE001
            # A clip that fails to decode must not abort the run and lose every
            # other prediction; emit silence and let it score an empty record.
            print("[predict] failed to decode %s: %s" % (p.name, e))
            y = np.zeros((self.sr,), "float32")
        y = y.astype("float32")

        n = min(len(y), self.n_samples)
        buf = np.zeros((self.n_samples,), "float32")
        buf[:n] = y[:n]
        valid = np.zeros((self.n_frames,), "float32")
        valid[:max(1, int(round(n / self.sr * self.fps)))] = 1.0
        return {"wav": torch.from_numpy(buf),
                "frame_valid": torch.from_numpy(valid),
                "uid": p.stem}


def collate(batch: List[dict]) -> dict:
    return {"wav": torch.stack([b["wav"] for b in batch]),
            "frame_valid": torch.stack([b["frame_valid"] for b in batch]),
            "uid": [b["uid"] for b in batch]}


def load_checkpoint(path: Path, device):
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    le = LabelEncoder(expand_vehicle=bool(cfg["data"].get("expand_vehicle", True)))
    enc = build_encoder(cfg["model"], ckpt_dir=cfg["model"].get("beats_dir", "checkpoints"))
    model = build_model(cfg, len(le), enc)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing:
        print("[predict] %s: %d missing keys (first %s)"
              % (path.name, len(missing), missing[:3]))
    return model.to(device).eval(), cfg


def write_submission(preds: Dict[str, List[List[float]]], out_zip: Path) -> None:
    lines = []
    for uid in sorted(preds):
        events = [{"onset": round(float(a), 3), "offset": round(float(b), 3)}
                  for a, b in preds[uid]]
        lines.append(json.dumps({"clip_id": uid, "events": events},
                                ensure_ascii=False))
    payload = "\n".join(lines) + "\n"
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("predictions.jsonl", payload)
    jl = out_zip.with_suffix(".jsonl")
    jl.write_text(payload, encoding="utf-8")
    print("[predict] wrote %s (%d clips) and %s" % (out_zip, len(preds), jl))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="one or more checkpoints; several are fused with 1D WBF")
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out", default="submission.zip")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--postproc", default="", help="JSON file of post-proc overrides")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="skip transductive per-district calibration")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--save-candidates", default="",
                    help="npz of raw spans/scores, for offline fusion")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pp = dict(DEFAULT_POSTPROC)
    if args.postproc:
        pp.update(json.loads(Path(args.postproc).read_text(encoding="utf-8")))

    files = sorted(p for p in Path(args.audio_dir).rglob("*")
                   if p.suffix.lower() in AUDIO_EXT)
    if not files:
        raise SystemExit("no audio found under %s" % args.audio_dir)
    print("[predict] %d clips" % len(files))

    per_model: List[Dict[str, dict]] = []
    for ck in args.ckpt:
        model, cfg = load_checkpoint(Path(ck), device)
        d = cfg["data"]
        ds = AudioDirDataset(files, sr=int(d["sr"]), clip_len=float(d["clip_len"]),
                             fps=float(d["fps"]))
        ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate,
                        pin_memory=device.type == "cuda")
        cands = run_loader(model, ld, device, float(d["fps"]), pp,
                           tta=not args.no_tta)
        per_model.append(cands)
        print("[predict] %s done" % Path(ck).name)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    uids = sorted(per_model[0])
    if len(per_model) == 1:
        fused = per_model[0]
    else:
        # Fuse spans, never posteriors: averaging two posteriors that localise
        # the same onset 80 ms apart widens the ramp by 80 ms.
        fused = {u: fuse_candidates([m[u] for m in per_model if u in m], pp)
                 for u in uids}

    if not args.no_calibrate:
        scales = calibrate(fused, pp)
        g = scales["_global"]
        print("[predict] calibrated %d districts (global: scale %.2f, slack %+d)"
              % (len(scales) - 1, g["score_scale"], g["count_slack"]))
        fused = apply_scales(fused, scales)

    if args.save_candidates:
        np.savez_compressed(
            args.save_candidates,
            **{u: np.concatenate([fused[u]["spans"],
                                  fused[u]["scores"].reshape(-1, 1)], axis=1)
               if len(fused[u]["spans"]) else np.zeros((0, 3), "float32")
               for u in uids})

    preds = {u: candidates_to_events(fused[u], pp) for u in uids}
    write_submission(preds, Path(args.out))

    n = [len(v) for v in preds.values()]
    cov = [sum(b - a for a, b in preds[u]) / max(fused[u]["duration"], 1e-6)
           for u in uids]
    print("[predict] events/clip %.2f (prior 1.22) | coverage %.3f (prior 0.52) "
          "| empty %.1f%%" % (float(np.mean(n)), float(np.mean(cov)),
                              100.0 * float(np.mean([x == 0 for x in n]))))


if __name__ == "__main__":
    main()
