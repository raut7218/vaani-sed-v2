"""Losses for the span model, and the point-to-event assignment they need.

Three ideas run through this file.

1. **Train the metric.** The leaderboard is ``event_F1 + segment_Dice``. Frame
   BCE is neither. So the boundary terms are a 1D DIoU on decoded spans (the
   thing event F1 measures) and a soft-Dice on the class-agnostic frame mask
   (literally the second half of the score, differentiably).

2. **Distributional boundaries.** Each distance-to-boundary is a categorical
   distribution over bins whose expectation is the regressed value. The
   Distribution Focal Loss puts mass on the two bins straddling the true
   distance, which is what lets the expectation land *between* grid points -
   sub-frame boundary resolution on a 40 ms grid.

3. **Annotation quality is about boundaries, not presence.** v1 gave silver a
   flat 0.5 weight on the whole frame loss. That is the wrong decomposition: a
   silver clip's *tags* are as trustworthy as gold's, only its *timestamps* are
   unverified. So silver keeps full weight on classification and clip terms and
   is heavily down-weighted only on the boundary terms. Bronze has no timestamps
   at all and contributes through attention pooling alone, exactly as before.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

# Span length (in base frames) that each pyramid level is responsible for.
# Level l has stride 2**l and 16 bins, so it can represent distances up to
# 15 * 2**l base frames - comfortably above each range's upper edge.
LEVEL_RANGES = [(0.0, 8.0), (8.0, 16.0), (16.0, 32.0), (32.0, 64.0), (64.0, 1e9)]
CENTER_RADIUS = 1.5      # positives within this many strides of the span centre

TIER_GOLD, TIER_SILVER, TIER_BRONZE = 0, 1, 2


def level_point_coords(n_frames: int, n_levels: int, device) -> List[torch.Tensor]:
    """Base-frame coordinate of each point, per level."""
    pts, t = [], n_frames
    for lvl in range(n_levels):
        stride = 2 ** lvl
        pts.append((torch.arange(t, device=device).float() + 0.5) * stride - 0.5)
        t = (t + 1) // 2
    return pts


@torch.no_grad()
def assign_targets(spans: torch.Tensor, span_cls: torch.Tensor, tier: torch.Tensor,
                   masks: List[torch.Tensor], n_frames: int, n_class: int,
                   n_bins: int) -> Dict[str, List[torch.Tensor]]:
    """Match ground-truth spans to pyramid points.

    spans:     (B, M, 2) in base frames, padded with -1
    span_cls:  (B, M)    class id, padded with -1
    tier:      (B,)      0 gold / 1 silver / 2 bronze

    Each event is assigned to exactly one level - the one whose range contains
    its length. Allowing a span to land on several levels produces duplicate
    detections that SoftNMS then has to clean up, and duplicates are pure false
    positives under a 1-to-1 matching metric.
    """
    device = spans.device
    B, M, _ = spans.shape
    n_levels = len(masks)
    pts = level_point_coords(n_frames, n_levels, device)

    cls_t, ds_t, de_t, pos_t, qual_t, bw_t = [], [], [], [], [], []
    lengths = (spans[..., 1] - spans[..., 0]).clamp(min=1e-3)          # (B, M)
    valid_ev = (span_cls >= 0) & (tier.view(B, 1) != TIER_BRONZE)

    # Per-event boundary weight: gold trusts its timestamps, silver does not.
    ev_bw = torch.where(tier.view(B, 1) == TIER_GOLD,
                        torch.ones_like(lengths), torch.full_like(lengths, 0.25))

    for lvl in range(n_levels):
        stride = float(2 ** lvl)
        p = pts[lvl][: masks[lvl].size(1)]                             # (T,)
        T = p.numel()
        lo, hi = LEVEL_RANGES[min(lvl, len(LEVEL_RANGES) - 1)]

        cls = spans.new_zeros((B, T, n_class + 1))
        ds = spans.new_zeros((B, T))
        de = spans.new_zeros((B, T))
        pos = spans.new_zeros((B, T))
        qual = spans.new_zeros((B, T))
        bw = spans.new_zeros((B, T))

        in_level = valid_ev & (lengths >= lo) & (lengths < hi)         # (B, M)
        if in_level.any():
            a = spans[..., 0].unsqueeze(-1)                            # (B, M, 1)
            b = spans[..., 1].unsqueeze(-1)
            pp = p.view(1, 1, T)
            d_start = (pp - a) / stride
            d_end = (b - pp) / stride
            inside = (d_start >= 0) & (d_end >= 0)
            centre = (a + b) / 2
            near = (pp - centre).abs() <= CENTER_RADIUS * stride
            ok = inside & (near | inside) & in_level.unsqueeze(-1)
            # Centre sampling proper: prefer near-centre points, but never let a
            # short event end up with zero positives just because it is narrower
            # than one stride.
            strict = inside & near & in_level.unsqueeze(-1)
            has_strict = strict.any(dim=-1, keepdim=True)
            ok = torch.where(has_strict, strict, ok)

            # An event shorter than a stride can miss every point; fall back to
            # the single closest point so it still supervises something.
            none_yet = in_level & ~ok.any(dim=-1)
            if none_yet.any():
                nearest = (pp - centre).abs().argmin(dim=-1)           # (B, M)
                idx = F.one_hot(nearest, T).bool() & none_yet.unsqueeze(-1)
                ok = ok | idx
                d_start = torch.where(idx, d_start.clamp(min=0.0), d_start)
                d_end = torch.where(idx, d_end.clamp(min=0.0), d_end)

            # Smallest event wins a contested point: short events are the ones
            # this model is bad at, and letting a long span absorb their points
            # is exactly the v1 failure mode (short events swallowed by blobs).
            big = torch.where(ok, lengths.unsqueeze(-1), torch.full_like(d_start, 1e9))
            best = big.argmin(dim=1)                                   # (B, T)
            any_pos = ok.any(dim=1).float()                            # (B, T)

            g = best.unsqueeze(1)
            sel_ds = torch.gather(d_start, 1, g).squeeze(1).clamp(0, n_bins - 1 - 1e-3)
            sel_de = torch.gather(d_end, 1, g).squeeze(1).clamp(0, n_bins - 1 - 1e-3)
            sel_cls = torch.gather(span_cls, 1, best).clamp(min=0)
            sel_bw = torch.gather(ev_bw, 1, best)

            ds = sel_ds * any_pos
            de = sel_de * any_pos
            pos = any_pos
            bw = sel_bw * any_pos
            # Centerness: down-weight points near an edge, where the far
            # boundary is a long extrapolation and least reliable.
            mn = torch.minimum(sel_ds, sel_de)
            mx = torch.maximum(sel_ds, sel_de).clamp(min=1e-6)
            qual = (mn / mx).clamp(0, 1).sqrt() * any_pos

            cls.scatter_(2, sel_cls.unsqueeze(-1), any_pos.unsqueeze(-1))
            cls[..., n_class] = any_pos                                # agnostic channel
            cls = cls * any_pos.unsqueeze(-1)

        m = masks[lvl]
        cls_t.append(cls * m.unsqueeze(-1))
        ds_t.append(ds * m)
        de_t.append(de * m)
        pos_t.append(pos * m)
        qual_t.append(qual * m)
        bw_t.append(bw * m)

    return {"cls": cls_t, "d_start": ds_t, "d_end": de_t, "pos": pos_t,
            "quality": qual_t, "bw": bw_t}


def sigmoid_focal(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                  alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    w = alpha * target + (1 - alpha) * (1 - target)
    loss = ce * w * (1 - pt).pow(gamma)
    return (loss * mask).sum()


def diou_1d(pred_s: torch.Tensor, pred_e: torch.Tensor,
            tgt_s: torch.Tensor, tgt_e: torch.Tensor) -> torch.Tensor:
    """1 - DIoU for intervals given as distances from a shared anchor point.

    Distance-IoU rather than plain IoU: when a prediction and its target do not
    overlap at all, IoU is flat at zero and its gradient says nothing about which
    direction to move. The centre-distance term keeps a gradient in that regime,
    which matters here because half the events are under a second and an early
    prediction can miss entirely.
    """
    inter = torch.minimum(pred_s, tgt_s) + torch.minimum(pred_e, tgt_e)
    inter = inter.clamp(min=0)
    union = (pred_s + pred_e) + (tgt_s + tgt_e) - inter
    iou = inter / union.clamp(min=1e-6)
    enclose = torch.maximum(pred_s, tgt_s) + torch.maximum(pred_e, tgt_e)
    # Centres relative to the anchor: (e - s) / 2.
    c_pred = (pred_e - pred_s) / 2
    c_tgt = (tgt_e - tgt_s) / 2
    rho2 = (c_pred - c_tgt).pow(2)
    return 1.0 - iou + rho2 / enclose.clamp(min=1e-6).pow(2)


def distribution_focal(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Distribution Focal Loss: put mass on the two bins straddling `target`.

    logits: (N, n_bins) - target: (N,) continuous, in bin units.
    This is the term that makes sub-frame boundaries learnable: the network is
    taught the exact linear interpolation weights, so the expectation it reports
    is continuous rather than snapped to a bin.
    """
    lo = target.floor().long()
    hi = (lo + 1).clamp(max=logits.size(1) - 1)
    w_hi = target - lo.float()
    w_lo = 1.0 - w_hi
    logp = F.log_softmax(logits, dim=1)
    return -(w_lo * logp.gather(1, lo.unsqueeze(1)).squeeze(1)
             + w_hi * logp.gather(1, hi.unsqueeze(1)).squeeze(1))


def soft_dice(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
              eps: float = 1e-4) -> torch.Tensor:
    """Per-clip soft Dice on the class-agnostic frame mask, macro-averaged.

    Macro, not micro, because the metric is macro - a two-second clip counts as
    much as a twenty-second one, and optimising the duration-weighted version
    quietly favours the long stationary classes.
    """
    p = logits.sigmoid() * mask
    t = target * mask
    num = 2.0 * (p * t).sum(dim=1)
    den = p.sum(dim=1) + t.sum(dim=1)
    # A clip with no events and no prediction is a perfect Dice of 1.0 under the
    # official scorer; the eps makes that the fixed point here too.
    return (1.0 - (num + eps) / (den + eps)).mean()


class SpanLoss:
    """Assembles every term. Returns (total, logs) with logs kept on-device.

    Nothing calls `.item()` or `float()` inside the step: v1 drained the CUDA
    queue five times per iteration doing exactly that. Logs are accumulated as
    tensors and read once per epoch.
    """

    def __init__(self, cfg: dict, n_class: int, n_bins: int):
        w = cfg.get("loss", {})
        self.n_class, self.n_bins = n_class, n_bins
        self.w_cls = float(w.get("cls", 1.0))
        self.w_reg = float(w.get("reg", 2.0))
        self.w_dfl = float(w.get("dfl", 0.5))
        self.w_qual = float(w.get("quality", 0.5))
        self.w_frame = float(w.get("frame", 0.5))
        self.w_dice = float(w.get("dice", 1.0))
        self.w_clip = float(w.get("clip", 0.5))
        self.w_speech = float(w.get("speech", 0.2))
        self.w_count = float(w.get("count", 0.2))
        self.silver_frame = float(w.get("silver_frame_weight", 0.5))

    def __call__(self, out: dict, batch: dict) -> tuple:
        device = out["cls"][0].device
        tier = batch["tier"]
        masks = out["masks"]
        n_frames = out["base_mask"].size(1)

        tgt = assign_targets(batch["spans"], batch["span_cls"], tier, masks,
                             n_frames, self.n_class, self.n_bins)

        n_pos = sum(t.sum() for t in tgt["pos"]).clamp(min=1.0)
        l_cls = out["cls"][0].new_zeros(())
        l_reg = out["cls"][0].new_zeros(())
        l_dfl = out["cls"][0].new_zeros(())
        l_qual = out["cls"][0].new_zeros(())

        for lvl in range(len(masks)):
            m = masks[lvl]
            l_cls = l_cls + sigmoid_focal(out["cls"][lvl], tgt["cls"][lvl],
                                          m.unsqueeze(-1))
            sel = tgt["pos"][lvl] > 0.5
            if not bool(sel.any()):
                continue
            bw = tgt["bw"][lvl][sel]                        # gold 1.0, silver 0.25
            ps = out["d_start"][lvl][sel]
            pe = out["d_end"][lvl][sel]
            ts = tgt["d_start"][lvl][sel]
            te = tgt["d_end"][lvl][sel]
            l_reg = l_reg + (diou_1d(ps, pe, ts, te) * bw).sum()
            l_dfl = l_dfl + ((distribution_focal(out["start_logits"][lvl][sel], ts)
                              + distribution_focal(out["end_logits"][lvl][sel], te))
                             * bw).sum()
            l_qual = l_qual + (F.binary_cross_entropy_with_logits(
                out["quality"][lvl][sel], tgt["quality"][lvl][sel],
                reduction="none") * bw).sum()

        l_cls = l_cls / n_pos
        l_reg = l_reg / n_pos
        l_dfl = l_dfl / (2 * n_pos)
        l_qual = l_qual / n_pos

        # --- auxiliary frame-level terms -------------------------------------
        vmask = out["base_mask"]
        strong = (tier != TIER_BRONZE).float()
        fw = torch.where(tier == TIER_GOLD, torch.ones_like(strong),
                         torch.full_like(strong, self.silver_frame)) * strong

        frame_bce = F.binary_cross_entropy_with_logits(
            out["frame_logits"], batch["frame_target"], reduction="none")
        frame_bce = (frame_bce.mean(-1) * vmask).sum(1) / vmask.sum(1).clamp(min=1)
        l_frame = (frame_bce * fw).sum() / fw.sum().clamp(min=1)

        agn_t = batch["frame_target"].amax(-1)
        l_dice = soft_dice(out["agn_logits"], agn_t, vmask * fw.view(-1, 1))

        l_clip = F.binary_cross_entropy(out["clip_probs"], batch["clip_target"])

        has_vad = batch.get("has_vad")
        if has_vad is not None and bool(has_vad.any()):
            sb = F.binary_cross_entropy_with_logits(
                out["speech_logits"], batch["speech_target"], reduction="none")
            sb = (sb * vmask).sum(1) / vmask.sum(1).clamp(min=1)
            l_speech = (sb * has_vad).sum() / has_vad.sum().clamp(min=1)
        else:
            l_speech = out["cls"][0].new_zeros(())

        cnt = batch["n_events"].clamp(max=out["count_logits"].size(1) - 1)
        ce = F.cross_entropy(out["count_logits"], cnt, reduction="none")
        l_count = (ce * strong).sum() / strong.sum().clamp(min=1)

        total = (self.w_cls * l_cls + self.w_reg * l_reg + self.w_dfl * l_dfl
                 + self.w_qual * l_qual + self.w_frame * l_frame
                 + self.w_dice * l_dice + self.w_clip * l_clip
                 + self.w_speech * l_speech + self.w_count * l_count)

        logs = {"loss": total.detach(), "cls": l_cls.detach(), "reg": l_reg.detach(),
                "dfl": l_dfl.detach(), "qual": l_qual.detach(),
                "frame": l_frame.detach(), "dice": l_dice.detach(),
                "clip": l_clip.detach(), "speech": l_speech.detach(),
                "count": l_count.detach(), "n_pos": n_pos.detach()}
        return total, logs
