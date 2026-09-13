"""Fine-tuned dual-SSL frame model with boundary-aware heads, on a 50 Hz grid.

  wav -> ATST-Frame (40 ms) + BEATs (160 ms) + a light 100 fps mel CNN
      -> resample all to 50 Hz, project, concat
      -> BiGRU x2
      -> presence   (T, C+1)  per class + class-agnostic
      -> boundary   (T, 2)    onset / offset, trained with a focal loss on
                              sparse boundary targets (the OOL of
                              Schmid et al. 2026, arXiv 2601.04178)
      -> extent     (T, 2)    time since onset / time to offset, trained with
                              a 1D IoU loss (their EPN)
      -> count      (K+1,)    events in the window, attention-pooled

The metric scores only onset/offset placement and frame overlap, so every
head is on the same 20 ms grid the decoder reads; nothing is decoded at a
coarser rate and interpolated up.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

FPS = 50.0


def resample(x: torch.Tensor, n: int) -> torch.Tensor:
    """(B, T, D) -> (B, n, D) linear interpolation in time."""
    if x.size(1) == n:
        return x
    return F.interpolate(x.transpose(1, 2).float(), size=n, mode="linear",
                         align_corners=False).transpose(1, 2).to(x.dtype)


class MelCNN(nn.Module):
    """100 fps log-mel -> 50 fps features. Carries sharp onsets the SSL tokens blur."""

    def __init__(self, n_mels: int = 64, ch: int = 64, out: int = 128):
        super().__init__()
        import torchaudio
        self.mel = torchaudio.transforms.MelSpectrogram(16000, n_fft=512, win_length=400, hop_length=160,
                                                        n_mels=n_mels, f_min=50, f_max=8000)
        self.body = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.GELU(),
            nn.AvgPool2d((4, 2)),                                   # freq /4, time /2 -> 50 fps
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.GELU(),
            nn.AvgPool2d((4, 1)))
        self.proj = nn.Linear(ch * (n_mels // 16), out)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=wav.device.type, enabled=False):
            m = self.mel(wav.float()).clamp(min=1e-8).log()        # (B, M, T100)
            m = (m - m.mean(dim=(1, 2), keepdim=True)) / (m.std(dim=(1, 2), keepdim=True) + 1e-5)
        h = self.body(m.unsqueeze(1))                               # (B, C, M/16, T50)
        B, C, Fq, T = h.shape
        return self.proj(h.permute(0, 3, 1, 2).reshape(B, T, C * Fq))


class TraceHead(nn.Module):
    def __init__(self, d_in: int, n_class: int, d: int = 256, max_count: int = 9, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, d * 2), nn.LayerNorm(d * 2), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(d * 2, d, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.mid = nn.Sequential(nn.Linear(d * 2, d * 2), nn.GELU())
        self.out_pres = nn.Linear(d * 2, n_class + 1)
        self.out_bnd = nn.Linear(d * 2, 2)
        # EPN: its own small BiGRU over the boundary-aware streams + features
        self.epn_gru = nn.GRU(d * 2 + n_class + 3, d // 2, batch_first=True, bidirectional=True)
        self.out_ext = nn.Linear(d, 2)
        self.cnt_att = nn.Linear(d * 2, 1)
        self.out_cnt = nn.Linear(d * 2, max_count + 1)
        nn.init.constant_(self.out_bnd.bias, -math.log((1 - 0.02) / 0.02))
        nn.init.constant_(self.out_pres.bias, -1.0)

    def forward(self, h: torch.Tensor, valid: torch.Tensor) -> dict:
        x = self.inp(h)
        x, _ = self.gru(x)
        x = self.mid(x) + x
        pres = self.out_pres(x)
        bnd = self.out_bnd(x)
        streams = torch.cat([x, pres.detach().float().sigmoid().to(x.dtype),
                             bnd.detach().float().sigmoid().to(x.dtype)], -1)
        e, _ = self.epn_gru(streams)
        ext = F.softplus(self.out_ext(e).float())                   # frames, >= 0
        a = self.cnt_att(x).float().squeeze(-1).masked_fill(valid <= 0, -1e4).softmax(-1)
        pooled = (x.float() * a.unsqueeze(-1)).sum(1)
        return dict(pres=pres.float(), bnd=bnd.float(), ext=ext, count=self.out_cnt(pooled.to(x.dtype)).float())


class TraceModel(nn.Module):
    def __init__(self, n_class: int, ckpt_dir: str = "checkpoints", encoders=("atst_frame", "beats"),
                 proj: int = 256, d: int = 256):
        super().__init__()
        from src.models.encoders import ATSTFrameEncoder, BEATsEncoder
        encs = []
        for name in encoders:
            if name == "atst_frame":
                encs.append(ATSTFrameEncoder(f"{ckpt_dir}/atst_frame.ckpt", freeze=True))
            elif name == "beats":
                encs.append(BEATsEncoder(f"{ckpt_dir}/BEATs_iter3_plus_AS2M.pt", freeze=True))
        self.encoders = nn.ModuleList(encs)
        self.projs = nn.ModuleList([nn.Sequential(nn.Linear(e.out_dim, proj), nn.LayerNorm(proj)) for e in encs])
        self.mel = MelCNN(out=128)
        self.head = TraceHead(proj * len(encs) + 128, n_class, d=d)

    # ---- encoder schedule -------------------------------------------------
    def set_encoder_trainable(self, on: bool) -> None:
        for e in self.encoders:
            for p in e.backbone().parameters():
                p.requires_grad = on
            e.frozen = not on
            if not on:
                e.backbone().eval()
        # BEATs' shared relative-position table stays frozen (see encoders.py)
        for e in self.encoders:
            if e.name == "beats":
                rel = e.beats.encoder.layers[0].self_attn.relative_attention_bias
                if rel is not None:
                    rel.weight.requires_grad = False

    def encoder_param_groups(self, lr_top: float, decay: float, wd: float):
        """Layer-wise LR decay: top block lr_top, each block below x decay."""
        groups = []
        for e in self.encoders:
            blocks = list(e.blocks())
            seen = set()
            for i, blk in enumerate(reversed(blocks)):
                ps = [p for p in blk.parameters() if p.requires_grad]
                seen.update(id(p) for p in blk.parameters())
                if ps:
                    groups.append(dict(params=ps, lr=lr_top * decay ** i, weight_decay=wd, name=f"{e.name}.b{len(blocks)-1-i}"))
            rest = [p for p in e.backbone().parameters() if p.requires_grad and id(p) not in seen]
            if rest:
                groups.append(dict(params=rest, lr=lr_top * decay ** len(blocks), weight_decay=wd, name=f"{e.name}.stem"))
        return groups

    def forward(self, wav: torch.Tensor, valid: torch.Tensor) -> dict:
        n = valid.size(1)
        feats = [resample(proj(enc(wav).to(proj[0].weight.dtype)), n) for enc, proj in zip(self.encoders, self.projs)]
        m = self.mel(wav)
        feats.append(resample(m, n))
        h = torch.cat(feats, -1)
        return self.head(h, valid)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def trace_losses(out: dict, b: dict, cfg: dict) -> dict:
    valid = b["valid"]                                             # (B, T)
    pw = b["pres_w"]                                               # (B,) presence weight per clip
    bw = b["bnd_w"]                                                # (B,) boundary/extent/count weight
    # presence: BCE with logits (autocast-safe), per class + agnostic; the class
    # channels only train where the source knows classes
    tgt = b["pres"]
    l = F.binary_cross_entropy_with_logits(out["pres"], tgt, reduction="none")      # (B, T, C+1)
    cls_mask = torch.cat([b["cls_known"].unsqueeze(-1).expand(-1, l.size(-1) - 1),
                          torch.ones_like(b["cls_known"]).unsqueeze(-1)], -1)        # (B, C+1)
    l = l * cls_mask.unsqueeze(1)
    l_pres = (l.sum(-1) * valid * pw[:, None]).sum() / ((valid * pw[:, None]).sum() * 2 + 1e-6)

    # boundaries: penalty-reduced focal loss on gaussian-splatted targets (CenterNet style)
    y = b["bnd"]                                                   # (B, T, 2) in [0, 1], 1 at boundary
    p = out["bnd"].sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = (y >= 0.999).float()
    lpos = -((1 - p) ** 2) * torch.log(p) * pos
    lneg = -((1 - y) ** 4) * (p ** 2) * torch.log(1 - p) * (1 - pos)
    m = (valid * bw[:, None]).unsqueeze(-1)
    npos = (pos * m).sum().clamp(min=1.0)
    l_bnd = ((lpos + lneg) * m).sum() / npos

    # extent: 1D IoU inside events, each event weighted equally
    ext_t = b["ext"]                                               # (B, T, 2) frames
    ew = b["ext_w"] * bw[:, None]                                  # (B, T)
    pl, pr = out["ext"][..., 0], out["ext"][..., 1]
    tl, tr = ext_t[..., 0], ext_t[..., 1]
    inter = torch.minimum(pl, tl) + torch.minimum(pr, tr)
    union = torch.maximum(pl, tl) + torch.maximum(pr, tr)
    l_ext = ((1 - inter / union.clamp(min=1e-3)) * ew).sum() / ew.sum().clamp(min=1.0)

    # count
    l_cnt = (F.cross_entropy(out["count"], b["count"].clamp(max=out["count"].size(-1) - 1), reduction="none")
             * bw).sum() / bw.sum().clamp(min=1.0)
    total = l_pres + cfg["w_bnd"] * l_bnd + cfg["w_ext"] * l_ext + cfg["w_cnt"] * l_cnt
    return dict(total=total, pres=l_pres.detach(), bnd=l_bnd.detach(), ext=l_ext.detach(), cnt=l_cnt.detach())
