"""Temporal feature pyramid + trident boundary head.

This is the file that replaces v1's "threshold a frame posterior" decoder, and
the reason is arithmetic rather than taste.

On the v1 checkpoint the class-agnostic posterior rises at ~0.54 per second
around a true onset. Placing a boundary to within 50 ms therefore requires the
decision threshold to be correct to **+-0.027 in probability**. Measured over
8,187 validation clips, the *optimal per-clip threshold* has mean 0.357 and
standard deviation **0.290**, and its histogram is essentially uniform across
[0.05, 0.95]. The spread is 11x wider than the precision the metric needs, so no
global threshold, per-class threshold, or adaptive heuristic can work. An oracle
that picks the best threshold per clip scores 1.414; the best fixed rule reaches
0.99.

The fix is to stop reading boundaries off a level set. Each time point directly
*regresses* its distance to the event start and end, and each distance is
predicted as a probability distribution over adjacent bins whose expectation is
continuous - so boundaries have sub-frame resolution even on a 40 ms grid, and
there is no threshold to tune. This is the trident-head idea from TriDet
(relative boundary modelling), adapted to audio.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvTransformerBlock(nn.Module):
    """Depthwise-conv locality + full self-attention + FFN, pre-norm.

    Sequence length at the base level is ~200 frames for an 8 s window, so full
    attention costs nothing and we skip the windowing ActionFormer needs for
    video-length inputs. The depthwise conv is what carries fine temporal detail
    through the block; attention alone tends to smear it, which is the one thing
    this model cannot afford.
    """

    def __init__(self, d_model: int, n_head: int = 8, kernel: int = 7,
                 dropout: float = 0.1, expand: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.dw = nn.Conv1d(d_model, d_model, kernel, padding=kernel // 2,
                            groups=d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * expand), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * expand, d_model), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D); mask: (B, T) with 1 on valid frames."""
        h = self.norm1(x)
        h = h + self.dw(h.transpose(1, 2)).transpose(1, 2)
        pad = mask < 0.5
        # A fully padded row would make softmax produce NaN across the whole row.
        # Clips are never entirely padding, but AMP plus an all-masked tail row
        # in a ragged batch can still produce one; force-allow position 0 there.
        allpad = pad.all(dim=1)
        if allpad.any():
            pad = pad.clone()
            pad[allpad, 0] = False
        a, _ = self.attn(h, h, h, key_padding_mask=pad, need_weights=False)
        x = x + self.drop(a)
        x = x + self.ff(self.norm2(x))
        return x * mask.unsqueeze(-1)


class TemporalFPN(nn.Module):
    """Multi-scale pyramid over the base 40 ms grid.

    Level l has stride 2**l frames, so with a 40 ms base the levels see
    40/80/160/320/640 ms resolution. Events in this corpus span 0.05 s to 20 s -
    four octaves - which is exactly what a pyramid is for: short events stay on
    the fine level where 40 ms of quantisation is affordable, long events move up
    to a level where their length is a small multiple of the stride.
    """

    def __init__(self, d_in: int, d_model: int = 384, n_levels: int = 5,
                 n_base_layers: int = 2, n_head: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_levels = n_levels
        self.stem = nn.Sequential(
            nn.Conv1d(d_in, d_model, 3, padding=1), nn.GELU(),
            nn.Conv1d(d_model, d_model, 3, padding=1))
        self.base = nn.ModuleList([
            ConvTransformerBlock(d_model, n_head, dropout=dropout)
            for _ in range(n_base_layers)])
        self.downs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(d_model, d_model, 3, stride=2, padding=1), nn.GELU())
            for _ in range(n_levels - 1)])
        self.level_blocks = nn.ModuleList([
            ConvTransformerBlock(d_model, n_head, dropout=dropout)
            for _ in range(n_levels - 1)])
        self.d_model = d_model

    def forward(self, x: torch.Tensor, mask: torch.Tensor
                ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """x: (B, T, D_in), mask: (B, T) -> per-level feats and masks."""
        h = self.stem(x.transpose(1, 2)).transpose(1, 2)
        h = h * mask.unsqueeze(-1)
        for blk in self.base:
            h = blk(h, mask)
        feats, masks = [h], [mask]
        for down, blk in zip(self.downs, self.level_blocks):
            h = down(h.transpose(1, 2)).transpose(1, 2)
            # max-pool the mask: a coarse frame is valid if any fine frame under
            # it was. Average-pooling would silently half-weight edge frames.
            mask = F.max_pool1d(mask.unsqueeze(1), 2, stride=2,
                                ceil_mode=True).squeeze(1)[:, :h.size(1)]
            if mask.size(1) < h.size(1):
                mask = F.pad(mask, (0, h.size(1) - mask.size(1)))
            h = h * mask.unsqueeze(-1)
            h = blk(h, mask)
            feats.append(h)
            masks.append(mask)
        return feats, masks


class TridentHead(nn.Module):
    """Per-point actionness + distributional distance-to-boundary regression.

    For a point t at a level of stride s, the head predicts two categorical
    distributions over `n_bins` bins:

        P_start[t, b]  ~  "the event starts b*s frames before t"
        P_end[t, b]    ~  "the event ends   b*s frames after  t"

    The regressed distance is the *expectation* of each distribution. That single
    choice is what buys sub-frame boundaries: the expectation is continuous even
    though the bins are not, so a 40 ms grid can express a 12 ms boundary shift.
    A plain scalar regression head cannot represent boundary ambiguity at all,
    and an argmax over bins would quantise straight back to the grid.

    The towers are shared across pyramid levels (weights tied, statistics per
    level via separate LayerNorms) because each level sees the same *relative*
    geometry - a span 3 strides wide looks the same at every scale.
    """

    def __init__(self, d_model: int, n_class: int, n_bins: int = 16,
                 n_levels: int = 5, n_tower: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_bins = n_bins
        self.n_class = n_class

        def tower():
            layers = []
            for _ in range(n_tower):
                layers += [nn.Conv1d(d_model, d_model, 3, padding=1),
                           nn.GroupNorm(8, d_model), nn.GELU(), nn.Dropout(dropout)]
            return nn.Sequential(*layers)

        self.cls_tower = tower()
        self.reg_tower = tower()
        # One norm per level: activation statistics genuinely differ by scale,
        # while the convolutional geometry does not.
        self.cls_norm = nn.ModuleList([nn.GroupNorm(8, d_model) for _ in range(n_levels)])
        self.reg_norm = nn.ModuleList([nn.GroupNorm(8, d_model) for _ in range(n_levels)])

        # +1 output channel: the class-agnostic actionness the metric scores.
        self.cls_out = nn.Conv1d(d_model, n_class + 1, 3, padding=1)
        self.start_out = nn.Conv1d(d_model, n_bins, 3, padding=1)
        self.end_out = nn.Conv1d(d_model, n_bins, 3, padding=1)
        self.quality = nn.Conv1d(d_model, 1, 3, padding=1)

        self.register_buffer("bins", torch.arange(n_bins).float(), persistent=False)
        # Bias the actionness prior to ~1% positive, the standard focal-loss
        # initialisation; without it the first hundred steps are spent undoing
        # a 50% prior over a very sparse target.
        nn.init.constant_(self.cls_out.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, feats: List[torch.Tensor], masks: List[torch.Tensor]) -> dict:
        cls, dstart, dend, qual, sbin, ebin = [], [], [], [], [], []
        for lvl, (h, m) in enumerate(zip(feats, masks)):
            x = h.transpose(1, 2)                               # (B, D, T)
            c = self.cls_out(self.cls_norm[lvl](self.cls_tower(x)))
            r = self.reg_norm[lvl](self.reg_tower(x))
            s_logits = self.start_out(r)                        # (B, bins, T)
            e_logits = self.end_out(r)
            q = self.quality(r)

            s_p = s_logits.softmax(dim=1)
            e_p = e_logits.softmax(dim=1)
            b = self.bins.view(1, -1, 1)
            cls.append(c.transpose(1, 2))                       # (B, T, C+1)
            dstart.append((s_p * b).sum(1))                     # (B, T) in strides
            dend.append((e_p * b).sum(1))
            qual.append(q.squeeze(1))
            sbin.append(s_logits.transpose(1, 2))               # (B, T, bins)
            ebin.append(e_logits.transpose(1, 2))
        return {"cls": cls, "d_start": dstart, "d_end": dend, "quality": qual,
                "start_logits": sbin, "end_logits": ebin, "masks": masks}


def level_points(n_frames: int, n_levels: int, device) -> List[torch.Tensor]:
    """Frame-grid coordinate of every point at every pyramid level."""
    pts = []
    t = n_frames
    for lvl in range(n_levels):
        stride = 2 ** lvl
        # Centre of the receptive field, so a point's span is symmetric about it.
        pts.append((torch.arange(t, device=device).float() + 0.5) * stride - 0.5)
        t = (t + 1) // 2
    return pts


def decode_spans(out: dict, n_frames: int, fps: float) -> Tuple[torch.Tensor, torch.Tensor,
                                                               torch.Tensor, torch.Tensor]:
    """Turn head outputs into (spans_sec, scores, class_ids, level_ids).

    Returns flat tensors over all levels: spans (B, N, 2) in seconds, scores
    (B, N) from the class-agnostic channel gated by predicted quality, class ids
    (B, N), and the level each point came from.
    """
    device = out["cls"][0].device
    B = out["cls"][0].size(0)
    pts = level_points(n_frames, len(out["cls"]), device)
    spans, scores, clses, levels = [], [], [], []
    for lvl, (c, ds, de, q, m) in enumerate(zip(
            out["cls"], out["d_start"], out["d_end"], out["quality"], out["masks"])):
        stride = float(2 ** lvl)
        p = pts[lvl][: c.size(1)].view(1, -1)
        start = (p - ds * stride) / fps
        end = (p + de * stride) / fps
        prob = c.sigmoid()
        agn = prob[..., -1]                                     # class-agnostic head
        percls = prob[..., :-1]
        cid = percls.argmax(dim=-1)
        # Geometric mean of actionness and quality: quality is what suppresses
        # points that sit inside an event but far from its centre, where the
        # boundary regression is least reliable.
        sc = (agn * q.sigmoid()).clamp(min=1e-6).sqrt() * agn.sqrt()
        sc = sc * m
        spans.append(torch.stack([start, end], dim=-1))
        scores.append(sc)
        clses.append(cid)
        levels.append(torch.full_like(cid, lvl))
    return (torch.cat(spans, 1), torch.cat(scores, 1),
            torch.cat(clses, 1), torch.cat(levels, 1))
