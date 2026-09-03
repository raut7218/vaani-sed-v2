"""Model output -> events. Shared by validation and by submission inference."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from src.infer.decode import (finalise, merge_close, select_by_count, soft_nms_1d,
                              wbf_1d)
from src.models.trident import decode_spans


DEFAULT_POSTPROC = {
    "nms_sigma": 0.3,
    "nms_iou": 0.35,
    "score_floor": 0.05,
    "count_weight": 1.0,
    "count_slack": 0,
    "min_dur": 0.03,
    "merge_gap": 0.0,
    "max_out": 16,
    "score_scale": 1.0,      # per-district calibration multiplies this
}


@torch.no_grad()
def spans_from_output(out: dict, fps: float, durations: np.ndarray,
                      pp: dict | None = None) -> List[dict]:
    """Decode one batch of model outputs into per-clip candidate spans.

    Returns raw candidates *before* count selection, so an ensemble can fuse
    across models first and select once at the end - selecting per model and
    then fusing throws away exactly the agreement that makes fusion work.
    """
    pp = {**DEFAULT_POSTPROC, **(pp or {})}
    n_frames = out["base_mask"].size(1)
    spans, scores, cls_ids, _ = decode_spans(out, n_frames, fps)
    spans = spans.float().cpu().numpy()
    scores = (scores.float() * float(pp["score_scale"])).clamp(0, 1).cpu().numpy()
    cls_ids = cls_ids.cpu().numpy()
    counts = out["count_logits"].softmax(-1).float().cpu().numpy()

    res = []
    for i in range(spans.shape[0]):
        dur = float(durations[i])
        s, c = spans[i], scores[i]
        ok = (c > 1e-4) & (s[:, 1] > s[:, 0])
        s, c, k = s[ok], c[ok], cls_ids[i][ok]
        s = np.clip(s, 0.0, dur)
        s, c = soft_nms_1d(s, c, sigma=float(pp["nms_sigma"]),
                           iou_thr=float(pp["nms_iou"]),
                           max_out=int(pp["max_out"]))
        res.append({"spans": s, "scores": c, "count": counts[i], "duration": dur})
    return res


def candidates_to_events(cand: dict, pp: dict | None = None) -> List[List[float]]:
    # A clip carries its own overrides once the calibrator has fitted its
    # district, so one call site serves both the calibrated and uncalibrated
    # paths.
    pp = {**DEFAULT_POSTPROC, **(pp or {}), **cand.get("pp_override", {})}
    s, c = select_by_count(cand["spans"], cand["scores"], cand.get("count"),
                           min_score=float(pp["score_floor"]),
                           slack=int(pp["count_slack"]),
                           count_weight=float(pp["count_weight"]))
    s = merge_close(s, gap=float(pp["merge_gap"]))
    return finalise(s, c[:len(s)], cand["duration"], min_dur=float(pp["min_dur"]))


def fuse_candidates(per_model: List[dict], pp: dict | None = None) -> dict:
    """1D weighted box fusion across models for one clip."""
    pp = {**DEFAULT_POSTPROC, **(pp or {})}
    s, c = wbf_1d([m["spans"] for m in per_model], [m["scores"] for m in per_model],
                  iou_thr=float(pp["nms_iou"]) + 0.15, n_models=len(per_model))
    count = np.mean([m["count"] for m in per_model], axis=0)
    return {"spans": s, "scores": c, "count": count,
            "duration": per_model[0]["duration"]}


@torch.no_grad()
def run_loader(model, loader, device, fps: float, pp: dict | None = None,
               amp: bool = True, tta: bool = False) -> Dict[str, dict]:
    """Run the model over a loader and return {uid: candidate dict}."""
    model.eval()
    out_all: Dict[str, dict] = {}
    for batch in loader:
        wav = batch["wav"].to(device, non_blocking=True)
        fv = batch["frame_valid"].to(device, non_blocking=True)
        durations = fv.sum(1).cpu().numpy() / fps
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            out = model(wav, fv)
        cands = spans_from_output(out, fps, durations, pp)

        if tta:
            # Time-shift TTA. A half-frame shift is the cheapest probe of
            # boundary stability there is, and fusing the two span sets
            # (never the posteriors) keeps the edges sharp.
            shift = int(0.02 * wav.size(-1) / (fv.size(1) / fps))
            wav2 = torch.roll(wav, shifts=shift, dims=-1)
            with torch.autocast(device_type=device.type,
                                enabled=amp and device.type == "cuda"):
                out2 = model(wav2, fv)
            c2 = spans_from_output(out2, fps, durations, pp)
            dt = shift / (wav.size(-1) / (fv.size(1) / fps)) / fps
            for c in c2:
                if len(c["spans"]):
                    c["spans"] = c["spans"] - dt
            cands = [fuse_candidates([a, b], pp) for a, b in zip(cands, c2)]

        for uid, c in zip(batch["uid"], cands):
            out_all[uid] = c
    return out_all
