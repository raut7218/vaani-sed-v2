"""The v2 model: high-resolution mel CNN + pretrained fusion -> FPN -> trident head.

Shape of the thing
------------------
    waveform 16 kHz
      |-- log-mel @100 fps + spectral-flux channels -> CNN -> 25 fps, 512-d
      |-- ATST-Frame / BEATs / WavLM fusion         ->        25 fps, 256*k-d
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

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoders import FusionEncoder
from src.models.frontend import LogMel, SpecAugment, onset_strength, resample_time
from src.models.trident import TemporalFPN, TridentHead


class FDYConv2d(nn.Module):
    """Frequency-dynamic conv (Nam et al., 2022); a plain conv when n_basis == 1.

    Kept from v1 because it is genuinely sound - a kernel that detects a horn at
    800 Hz should not be the same kernel applied at 4 kHz. Defaulted *off*
    (n_basis 1) because it costs 4x the conv FLOPs on a component that was never
    the bottleneck; raise `model.n_basis` to ablate it back in.
    """

    def __init__(self, in_ch: int, out_ch: int, n_basis: int = 1,
                 temperature: float = 31.0):
        super().__init__()
        self.in_ch, self.out_ch, self.n_basis = in_ch, out_ch, n_basis
        self.weight = nn.Parameter(torch.empty(n_basis * out_ch, in_ch, 3, 3))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.temperature = temperature
        if n_basis > 1:
            hidden = max(in_ch // 4, 8)
            self.att = nn.Sequential(
                nn.Conv1d(in_ch, hidden, 1, bias=False), nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True), nn.Conv1d(hidden, n_basis, 1))
        else:
            self.att = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.conv2d(x, self.weight, None, stride=1, padding=1)
        if self.att is None:
            return y
        B = x.size(0)
        Fo, To = y.shape[-2], y.shape[-1]
        y = y.view(B, self.n_basis, self.out_ch, Fo, To)
        a = torch.softmax(self.att(x.mean(dim=3)) / self.temperature, dim=1)
        if a.size(-1) != Fo:
            a = F.interpolate(a, size=Fo, mode="linear", align_corners=False)
        return (y * a[:, :, None, :, None]).sum(dim=1)


class MelCNN(nn.Module):
    """(B, C, F, T@100fps) -> (B, T@25fps, D).

    Time is pooled by exactly 4, and only in the first two blocks. Every later
    block pools frequency only. That ordering is deliberate: pooling time late
    (or more than 4x) is how v1's features acquired their 760 ms rise time, and
    this branch exists specifically to carry the sharp temporal detail that the
    160 ms patch encoders cannot.
    """

    def __init__(self, in_ch: int = 3, n_mels: int = 128,
                 channels=(32, 64, 128, 256, 256, 256), n_basis: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        pools = ((2, 2), (2, 2), (2, 1), (2, 1), (2, 1), (2, 1))
        blocks, c_in, f = [], in_ch, n_mels
        for c, p in zip(channels, pools):
            blocks.append(nn.Sequential(
                FDYConv2d(c_in, c, n_basis=n_basis),
                nn.BatchNorm2d(c), nn.GELU(),
                nn.AvgPool2d(p) if p != (1, 1) else nn.Identity(),
                nn.Dropout2d(dropout)))
            c_in, f = c, max(1, f // p[0])
        self.blocks = nn.Sequential(*blocks)
        self.out_dim = channels[-1] * f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        B, C, Fp, Tp = x.shape
        return x.permute(0, 3, 1, 2).reshape(B, Tp, C * Fp)


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


class VaaniSpanModel(nn.Module):
    def __init__(self, n_class: int, n_frames: int, encoder: FusionEncoder | None = None,
                 n_mels: int = 128, sr: int = 16000, hop: int = 160, fps: float = 25.0,
                 d_model: int = 384, n_levels: int = 5, n_bins: int = 16,
                 n_base_layers: int = 2, n_head: int = 8, dropout: float = 0.1,
                 n_basis: int = 1, use_specaug: bool = True, use_flux: bool = True,
                 max_count: int = 8):
        super().__init__()
        self.n_class, self.n_frames, self.fps = n_class, n_frames, float(fps)
        self.n_levels, self.n_bins = n_levels, n_bins
        self.use_flux = use_flux

        self.logmel = LogMel(sr=sr, hop=hop, n_mels=n_mels)
        self.specaug = SpecAugment() if use_specaug else nn.Identity()
        self.cnn = MelCNN(in_ch=3 if use_flux else 1, n_mels=n_mels,
                          n_basis=n_basis, dropout=dropout * 0.5)
        self.encoder = encoder
        d_in = self.cnn.out_dim + (encoder.out_dim if encoder is not None else 0)

        self.fpn = TemporalFPN(d_in, d_model=d_model, n_levels=n_levels,
                               n_base_layers=n_base_layers, n_head=n_head,
                               dropout=dropout)
        self.head = TridentHead(d_model, n_class, n_bins=n_bins, n_levels=n_levels,
                                dropout=dropout)

        # --- auxiliary heads, all at the base 40 ms grid ---
        self.frame_head = AttentionPool(d_model, n_class)
        self.agn_head = nn.Linear(d_model, 1)
        self.speech_head = nn.Linear(d_model, 1)
        self.count_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(),
                                        nn.Linear(d_model // 2, max_count))

    def encoder_cache(self, wav: torch.Tensor) -> dict:
        return self.encoder.encode_raw(wav) if self.encoder is not None else {}

    def forward(self, wav: torch.Tensor, frame_valid: torch.Tensor | None = None,
                enc_cache: dict | None = None) -> dict:
        mel = self.logmel(wav, frame_valid)                    # (B, 1, F, T@100)
        if self.use_flux:
            mel = torch.cat([mel, onset_strength(mel)], dim=1)
        mel = self.specaug(mel)
        h = self.cnn(mel)                                      # (B, T', D)
        h = resample_time(h, self.n_frames)

        if self.encoder is not None and len(self.encoder.encoders) > 0:
            e = self.encoder(wav, self.n_frames, cache=enc_cache)
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
        dropout=float(m.get("dropout", 0.1)), n_basis=int(m.get("n_basis", 1)),
        use_specaug=bool(m.get("specaug", True)), use_flux=bool(m.get("flux", True)),
        max_count=int(m.get("max_count", 8)))
