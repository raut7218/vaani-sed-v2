"""Model output -> events. Shared by validation and by submission inference."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from src.infer.decode import (boundary_agreement, finalise, merge_close,
                              refine_boundaries, select_by_count, soft_nms_1d,
                              wbf_1d)
from src.models.trident import decode_spans


DEFAULT_POSTPROC = {
    "nms_sigma": 0.3,
    "nms_iou": 0.35,
    "score_floor": 0.05,
    "count_weight": 1.0,
    "count_mode": "expected",   # "expected" or "argmax"; see select_by_count
    "count_slack": 0,
    "min_dur": 0.03,
    "merge_gap": 0.0,
    "max_out": 16,
    "score_scale": 1.0,      # per-district calibration multiplies this
    "quality_power": 0.0,    # exponent on the quality head in the span score;
                             # 0 under QFL, where actionness already carries it
    # --- boundary branch ---
    "refine": False,         # snap endpoints onto the 20 ms branch's peaks
    "refine_window": 0.15,   # +- max(this * duration, refine_window_min) seconds
    "refine_window_min": 0.08,
    "refine_peak_min": 0.40,
    "agree_weight": 0.0,     # exponent on boundary agreement in the span score
}


@torch.no_grad()
def _decode_to_host(out: dict, fps: float, pp: dict) -> tuple:
    """Decode on the GPU and start the device->host copy without waiting on it.

    Returns (tensors, event): the tensors are only safe to read once `event`
    has completed, which lets the caller queue the next batch's forward first.
    """
    n_frames = out["base_mask"].size(1)
    spans, scores, _, _ = decode_spans(out, n_frames, fps,
                                       q_power=float(pp["quality_power"]))
    ts = [spans.float(),
          (scores.float() * float(pp["score_scale"])).clamp(0, 1),
          out["count_logits"].float().softmax(-1)]
    if "onset_logits" in out:
        hm = out["hi_mask"]
        ts.append(torch.stack([out["onset_logits"].float().sigmoid() * hm,
                               out["offset_logits"].float().sigmoid() * hm], dim=1))
    # The refinement grid spans the whole padded window, not the clip's own
    # duration, so its frame rate is a property of the model - deriving it from
    # `duration` on the host would shift every refined boundary on any clip
    # shorter than the window.
    hi_fps = (out["onset_logits"].size(1) * fps / n_frames) if "onset_logits" in out else 0.0
    ts = tuple(ts)
    if ts[0].device.type != "cuda":
        return (ts, hi_fps), None
    host = tuple(torch.empty(t.shape, dtype=t.dtype, pin_memory=True) for t in ts)
    for h, t in zip(host, ts):
        h.copy_(t, non_blocking=True)
    ev = torch.cuda.Event()
    ev.record()
    return (host, hi_fps), ev


def _host_to_candidates(host: tuple, event, durations: np.ndarray,
                        pp: dict, shift: float = 0.0) -> List[dict]:
    if event is not None:
        event.synchronize()
    host, hi_fps = host
    arrays = [t.numpy() for t in host]
    spans, scores, counts = arrays[0], arrays[1], arrays[2]
    bmap = arrays[3] if len(arrays) > 3 else None
    res = []
    for i in range(spans.shape[0]):
        dur = float(durations[i])
        s, c = spans[i], scores[i]
        ok = (c > 1e-4) & (s[:, 1] > s[:, 0])
        s, c = s[ok], c[ok]
        if bmap is not None:
            # Still in the *shifted* branch's own time: so is its boundary map,
            # and undoing the TTA shift before reading the map would snap every
            # endpoint onto a peak `shift` seconds away from where it belongs.
            on, off = bmap[i, 0], bmap[i, 1]
            s = np.clip(s, 0.0, dur + shift)
            # Rank first, then move: agreement has to be read at the boundaries
            # the detector actually proposed, or every candidate gets credit for
            # a peak that refinement dragged it onto.
            aw = float(pp["agree_weight"])
            if aw > 0:
                c = c * np.maximum(boundary_agreement(s, on, off, hi_fps), 1e-3) ** aw
            if bool(pp["refine"]):
                s = refine_boundaries(s, on, off, hi_fps, dur + shift,
                                      window_frac=float(pp["refine_window"]),
                                      window_min=float(pp["refine_window_min"]),
                                      peak_min=float(pp["refine_peak_min"]))
        s = np.clip(s - shift, 0.0, dur)
        s, c = soft_nms_1d(s, c, sigma=float(pp["nms_sigma"]),
                           iou_thr=float(pp["nms_iou"]),
                           max_out=int(pp["max_out"]))
        res.append({"spans": s, "scores": c, "count": counts[i], "duration": dur})
    return res


def spans_from_output(out: dict, fps: float, durations: np.ndarray,
                      pp: dict | None = None) -> List[dict]:
    """Decode one batch of model outputs into per-clip candidate spans.

    Returns raw candidates *before* count selection, so an ensemble can fuse
    across models first and select once at the end - selecting per model and
    then fusing throws away exactly the agreement that makes fusion work.
    """
    pp = {**DEFAULT_POSTPROC, **(pp or {})}
    host, ev = _decode_to_host(out, fps, pp)
    return _host_to_candidates(host, ev, durations, pp)


def candidates_to_events(cand: dict, pp: dict | None = None) -> List[List[float]]:
    # A clip carries its own overrides once the calibrator has fitted its
    # district, so one call site serves both the calibrated and uncalibrated
    # paths.
    pp = {**DEFAULT_POSTPROC, **(pp or {}), **cand.get("pp_override", {})}
    s, c = select_by_count(cand["spans"], cand["scores"], cand.get("count"),
                           min_score=float(pp["score_floor"]),
                           slack=int(pp["count_slack"]),
                           count_weight=float(pp["count_weight"]),
                           count_mode=str(pp["count_mode"]))
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
    """Run the model over a loader and return {uid: candidate dict}.

    Software-pipelined: batch i's CPU post-processing (SoftNMS, one clip at a
    time) runs while batch i+1's forward is already queued on the GPU, instead
    of the GPU idling through it.
    """
    pp = {**DEFAULT_POSTPROC, **(pp or {})}
    model.eval()
    out_all: Dict[str, dict] = {}
    pending = None

    def finish(p):
        uids, durations, decoded = p
        sets = [_host_to_candidates(h, ev, durations, pp, shift)
                for h, ev, shift in decoded]
        cands = sets[0] if len(sets) == 1 else \
            [fuse_candidates(list(c), pp) for c in zip(*sets)]
        for uid, c in zip(uids, cands):
            out_all[uid] = c

    for batch in loader:
        wav = batch["wav"].to(device, non_blocking=True)
        fv = batch["frame_valid"].to(device, non_blocking=True)
        durations = batch["frame_valid"].sum(1).numpy() / fps
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            out = model(wav, fv)
        decoded = [_decode_to_host(out, fps, pp) + (0.0,)]

        if tta:
            # Time-shift TTA. A half-frame shift is the cheapest probe of
            # boundary stability there is, and fusing the two span sets
            # (never the posteriors) keeps the edges sharp. `shift /
            # samples_per_sec` is already seconds; the shifted branch's spans
            # are moved back by exactly that before fusion.
            samples_per_sec = wav.size(-1) / (fv.size(1) / fps)
            shift = int(0.02 * samples_per_sec)
            wav2 = torch.roll(wav, shifts=shift, dims=-1)
            with torch.autocast(device_type=device.type,
                                enabled=amp and device.type == "cuda"):
                out2 = model(wav2, fv)
            decoded.append(_decode_to_host(out2, fps, pp) + (shift / samples_per_sec,))

        if pending is not None:
            finish(pending)
        pending = (batch["uid"], durations, decoded)
    if pending is not None:
        finish(pending)
    return out_all
