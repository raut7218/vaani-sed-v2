"""The v2 model: high-resolution mel CNN + pretrained fusion -> FPN -> trident head.

Shape of the thing
------------------
    waveform 16 kHz
      |-- log-mel @100 fps + spectral-flux channels -> CNN -> 25 fps, 512-d
      |-- ATST-Frame / BEATs fusion                 ->        25 fps, 256*k-d
      concat -> TemporalFPN (5 levels, 40..640 ms)
             -> TridentHead   : per-point actionness + distributional boundaries
             -> auxiliary heads at the base level:
                  frame class logits   (tier-masked BCE, soft-Dice on the union)
                  class-agnostic frame (the channel the metric actually scores)
                  speech presence      (Vaani is *speech* recordings - see below)
                  clip tags            (attention pooling; the bronze tier's only
                                        route into the frame representation)
                  event count          (83% of clips hold exactly one event)

Why a speech head
-----------------
This corpus is Project Vaani: conversational speech with noise events on top.
Every DCASE-derived system treats the audio as a generic soundscape. Telling the
model explicitly which energy is speech lets it factor the mixture instead of
inferring the decomposition implicitly, and noise-event boundaries are far easier
to place once the speech is accounted for. Pseudo-labels come free from a VAD
(`scripts/make_vad.py`), so this costs one extra output channel and no
annotation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models.encoders import FusionEncoder
from src.models.frontend import LogMel, SpecAugment, onset_strength, resample_time
from src.models.trident import TemporalFPN, TridentHead


class MelCNN(nn.Module):
    """(B, C, F, T@100fps) -> (B, T@25fps, D).

    Time is pooled by exactly 4, and only in the first two blocks. Every later
    block pools frequency only. That ordering is deliberate: pooling time late
    (or more than 4x) is how v1's features acquired their 760 ms rise time, and
    this branch exists specifically to carry the sharp temporal detail that the
    160 ms patch encoders cannot.
    """

    def __init__(self, in_ch: int = 3, n_mels: int = 128,
                 channels=(32, 64, 128, 256, 256, 256), dropout: float = 0.1,
                 hi_after: int = 1):
        super().__init__()
        pools = ((2, 2), (2, 2), (2, 1), (2, 1), (2, 1), (2, 1))
        blocks, c_in, f = [], in_ch, n_mels
        # `hi_after` marks the block whose output the boundary branch taps.
        # 1 is the default: time has been pooled 2x by then, so the tap is at
        # 50 fps - 20 ms, half the detection grid's cell. 0 taps the log-mel
        # itself at the full 100 fps, spectral-flux channels included, which is
        # the un-pooled onset signal and the fallback if 20 ms turns out not to
        # be fine enough.
        self.hi_after = int(hi_after)
        self.hi_dim = (in_ch * n_mels if self.hi_after == 0 else
                       channels[self.hi_after - 1] * max(1, n_mels // 2 ** self.hi_after))
        for c, p in zip(channels, pools):
            blocks.append(nn.Sequential(
                nn.Conv2d(c_in, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c), nn.GELU(),
                nn.AvgPool2d(p) if p != (1, 1) else nn.Identity(),
                nn.Dropout2d(dropout)))
            c_in, f = c, max(1, f // p[0])
        self.blocks = nn.ModuleList(blocks)
        self.out_dim = channels[-1] * f

    def forward(self, x: torch.Tensor):
        """Returns (base @25 fps, hi @50 or 100 fps) - both (B, T, D)."""
        def flat(t):
            B, C, Fp, Tp = t.shape
            return t.permute(0, 3, 1, 2).reshape(B, Tp, C * Fp)

        hi = flat(x) if self.hi_after == 0 else None
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i + 1 == self.hi_after:
                hi = flat(x)
        B, C, Fp, Tp = x.shape
        return x.permute(0, 3, 1, 2).reshape(B, Tp, C * Fp), hi


class AttentionPool(nn.Module):
    """Frame logits -> clip logits.

    The bridge that lets bronze clips (tags, no timestamps) train the frame
    representation: the clip loss backpropagates through the softmax attention
    into every frame. Carried over from v1 unchanged - it was correct.
    """

    def __init__(self, in_dim: int, n_class: int):
        super().__init__()
        self.cls = nn.Linear(in_dim, n_class)
        self.att = nn.Linear(in_dim, n_class)

    def forward(self, h: torch.Tensor, valid: torch.Tensor | None = None):
        logits = self.cls(h)                                  # (B, T, C)
        a = self.att(h)
        if valid is not None:
            a = a.masked_fill(valid.unsqueeze(-1) < 0.5, -1e4)
        w = torch.softmax(a, dim=1)
        clip = (torch.sigmoid(logits) * w).sum(dim=1).clamp(1e-6, 1 - 1e-6)
        return logits, clip


class BoundaryBranch(nn.Module):
    """Per-frame onset / offset probability at 50 fps - the refinement grid.

    Why this exists
    ---------------
    The detection pyramid runs at 25 fps and regresses a distance to each
    boundary. Measured on the v2 checkpoint, that path behaves like a perfect
    detector with ~0.19 s of boundary jitter, and the metric's tolerance is
    ``max(0.20 * duration, 0.05)`` s - a median of 0.09 s on the gold tier,
    where 56% of events are under half a second. Simulating jitter against the
    real references puts the achievable score at 1.17 for sigma 0.20 s and 1.60
    for sigma 0.08 s, so the whole remaining gap *is* boundary precision.

    A regression head cannot get there on its own: its features are 40 ms cells
    and its finest DFL bin is 40 ms wide. What can get there is a separate,
    much shallower branch looking at the 20 ms mel grid, asked one easy
    question - "is there an onset here?" - rather than the hard one, "how far is
    the onset from here?". The detection head proposes; this branch snaps.

    Supervision is deliberately unequal. Gold and synthetic clips train it at
    full weight; silver is down-weighted and its clip-edge boundaries are
    dropped outright, because 29.6% of silver events start at exactly 0.000 s
    and 11.7% span the whole clip - those are annotation defaults, not audible
    transients, and training a transient detector on them teaches it to fire on
    silence.
    """

    def __init__(self, hi_dim: int, d_model: int, d_hidden: int = 128,
                 kernel: int = 5, dropout: float = 0.1):
        super().__init__()
        self.proj_hi = nn.Conv1d(hi_dim, d_hidden, 1)
        self.proj_ctx = nn.Conv1d(d_model, d_hidden, 1)
        self.body = nn.Sequential(
            nn.Conv1d(d_hidden * 2, d_hidden, kernel, padding=kernel // 2),
            nn.GroupNorm(8, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(d_hidden, d_hidden, kernel, padding=kernel // 2),
            nn.GroupNorm(8, d_hidden), nn.GELU())
        self.out = nn.Conv1d(d_hidden, 2, 3, padding=1)
        # Boundaries are ~1 frame in 50: start the sigmoid near that prior
        # instead of spending the first epoch walking down from 0.5.
        nn.init.constant_(self.out.bias, -math.log((1 - 0.02) / 0.02))

    def forward(self, hi: torch.Tensor, ctx: torch.Tensor, n_hi: int) -> torch.Tensor:
        """hi: (B, T_hi, D_hi) @50 fps, ctx: (B, T, D) @25 fps -> (B, 2, n_hi)."""
        h = resample_time(hi, n_hi).transpose(1, 2)
        c = resample_time(ctx, n_hi).transpose(1, 2)
        x = torch.cat([self.proj_hi(h), self.proj_ctx(c)], dim=1)
        return self.out(self.body(x))                          # (B, 2, n_hi)


class VaaniSpanModel(nn.Module):
    def __init__(self, n_class: int, n_frames: int, encoder: FusionEncoder | None = None,
                 n_mels: int = 128, sr: int = 16000, hop: int = 160, fps: float = 25.0,
                 d_model: int = 384, n_levels: int = 5, n_bins: int = 16,
                 n_base_layers: int = 2, n_head: int = 8, dropout: float = 0.1,
                 use_specaug: bool = True, use_flux: bool = True,
                 max_count: int = 8, boundary_branch: bool = True,
                 boundary_mult: int = 2, dgqp: bool = True, hi_after: int = 1):
        super().__init__()
        self.n_class, self.n_frames, self.fps = n_class, n_frames, float(fps)
        self.n_levels, self.n_bins = n_levels, n_bins
        self.use_flux = use_flux
        # The refinement grid runs `boundary_mult` x the detection grid.
        self.boundary_mult = int(boundary_mult)
        self.n_hi = int(n_frames * self.boundary_mult)

        self.logmel = LogMel(sr=sr, hop=hop, n_mels=n_mels)
        self.specaug = SpecAugment() if use_specaug else nn.Identity()
        self.cnn = MelCNN(in_ch=3 if use_flux else 1, n_mels=n_mels,
                          dropout=dropout * 0.5,
                          hi_after=hi_after).to(memory_format=torch.channels_last)
        self.encoder = encoder
        d_in = self.cnn.out_dim + (encoder.out_dim if encoder is not None else 0)

        self.fpn = TemporalFPN(d_in, d_model=d_model, n_levels=n_levels,
                               n_base_layers=n_base_layers, n_head=n_head,
                               dropout=dropout)
        self.head = TridentHead(d_model, n_class, n_bins=n_bins, n_levels=n_levels,
                                dropout=dropout, dgqp=dgqp)

        # --- auxiliary heads, all at the base 40 ms grid ---
        self.frame_head = AttentionPool(d_model, n_class)
        self.agn_head = nn.Linear(d_model, 1)
        self.speech_head = nn.Linear(d_model, 1)
        self.count_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(),
                                        nn.Linear(d_model // 2, max_count))
        self.boundary = BoundaryBranch(self.cnn.hi_dim, d_model, dropout=dropout) \
            if boundary_branch else None

    def forward(self, wav: torch.Tensor, frame_valid: torch.Tensor | None = None) -> dict:
        mel = self.logmel(wav, frame_valid)                    # (B, 1, F, T@100)
        if self.use_flux:
            mel = torch.cat([mel, onset_strength(mel)], dim=1)
        mel = self.specaug(mel)
        # NHWC: cuDNN's fp16 tensor-core convs want it, and with NCHW every
        # conv in the CNN pays a layout conversion each way.
        h, hi = self.cnn(mel.contiguous(memory_format=torch.channels_last))

        h = resample_time(h, self.n_frames)

        if self.encoder is not None and len(self.encoder.encoders) > 0:
            e = self.encoder(wav, self.n_frames)
            h = torch.cat([h, e], dim=-1)

        mask = frame_valid if frame_valid is not None else \
            h.new_ones((h.size(0), self.n_frames))
        feats, masks = self.fpn(h, mask)
        out = self.head(feats, masks)

        base = feats[0]
        frame_logits, clip_probs = self.frame_head(base, mask)
        out["frame_logits"] = frame_logits                     # (B, T, C)
        out["clip_probs"] = clip_probs                         # (B, C)
        out["agn_logits"] = self.agn_head(base).squeeze(-1)    # (B, T)
        out["speech_logits"] = self.speech_head(base).squeeze(-1)
        pooled = (base * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        out["count_logits"] = self.count_head(pooled)          # (B, max_count)
        out["base_mask"] = mask
        if self.boundary is not None:
            b = self.boundary(hi, base, self.n_hi)             # (B, 2, n_hi)
            out["onset_logits"] = b[:, 0]
            out["offset_logits"] = b[:, 1]
            out["hi_mask"] = resample_time(mask.unsqueeze(-1), self.n_hi).squeeze(-1)
        return out


def build_model(cfg: dict, n_class: int, encoder: FusionEncoder | None = None
                ) -> VaaniSpanModel:
    d = cfg["data"]
    m = cfg["model"]
    n_frames = int(round(float(d["clip_len"]) * float(d["fps"])))
    return VaaniSpanModel(
        n_class=n_class, n_frames=n_frames, encoder=encoder,
        n_mels=int(d["n_mels"]), sr=int(d["sr"]), hop=int(d["hop"]),
        fps=float(d["fps"]), d_model=int(m.get("d_model", 384)),
        n_levels=int(m.get("n_levels", 5)), n_bins=int(m.get("n_bins", 16)),
        n_base_layers=int(m.get("n_base_layers", 2)), n_head=int(m.get("n_head", 8)),
        dropout=float(m.get("dropout", 0.1)),
        use_specaug=bool(m.get("specaug", True)), use_flux=bool(m.get("flux", True)),
        max_count=int(m.get("max_count", 8)),
        boundary_branch=bool(m.get("boundary_branch", True)),
        boundary_mult=int(m.get("boundary_mult", 2)),
        dgqp=bool(m.get("dgqp", True)),
        hi_after=int(m.get("boundary_tap", 1)))
