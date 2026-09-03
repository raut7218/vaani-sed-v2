"""Shared audio front-end: log-mel + per-clip normalisation + SpecAugment.

The normalisation and the float32 pinning are carried over verbatim from v1 -
they were correct, and the fp16 failure mode they guard against is subtle enough
to be worth preserving with its explanation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


def resample_time(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """(B, T, D) -> (B, target_len, D) by linear interpolation along time."""
    if x.size(1) == target_len:
        return x
    return F.interpolate(x.transpose(1, 2), size=target_len, mode="linear",
                         align_corners=False).transpose(1, 2)


class LogMel(nn.Module):
    """Log-mel at `sr/hop` frames per second, normalised over valid frames only."""

    def __init__(self, sr: int = 16000, n_fft: int = 1024, hop: int = 160,
                 n_mels: int = 128, fmin: int = 0, fmax: int = 8000):
        super().__init__()
        self.hop = hop
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, win_length=n_fft,
            f_min=fmin, f_max=fmax, n_mels=n_mels, power=2.0, center=True)

    def forward(self, wav: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        # The whole front-end runs in float32 with autocast disabled.
        #
        # Under fp16 autocast this produces NaN from the first step: clips are
        # zero-padded to the window length, so the padded region gives mel power
        # of exactly 0, and the 1e-10 clamp floor is itself below the smallest
        # fp16 subnormal (6e-8) - it rounds to 0.0 and stops guarding log().
        # log(0) = -inf, and the per-clip mean/std then turn the whole batch NaN.
        with torch.autocast(device_type=wav.device.type, enabled=False):
            m = self.mel(wav.float())                       # (B, n_mels, T)
            m = torch.log(m.clamp(min=1e-10))
            return self._normalise(m, valid)

    def _normalise(self, m: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        # Recording level varies hugely across Vaani districts and devices, and
        # absolute loudness is not the signal. Statistics come from *valid*
        # frames only: clips are far shorter than the window, so normalising over
        # the padding would leak clip length into the features.
        if valid is None:
            mu = m.mean(dim=(1, 2), keepdim=True)
            sd = m.std(dim=(1, 2), keepdim=True).clamp(min=1e-5)
        else:
            w = F.interpolate(valid[:, None, :].to(m.dtype), size=m.size(-1),
                              mode="nearest")
            n = (w.sum(dim=(1, 2), keepdim=True) * m.size(1)).clamp(min=1.0)
            mu = (m * w).sum(dim=(1, 2), keepdim=True) / n
            var = ((m - mu) ** 2 * w).sum(dim=(1, 2), keepdim=True) / n
            sd = var.sqrt().clamp(min=1e-5)
        return ((m - mu) / sd).unsqueeze(1)                 # (B, 1, F, T)


class SpecAugment(nn.Module):
    """Frequency and time masking.

    Time masking is deliberately *narrower* than the v1 default (40 frames =
    400 ms at 100 fps). This model regresses boundaries, and a 400 ms hole
    straddling an onset teaches it that boundaries are approximate - exactly the
    lesson that cost v1 its event F1.
    """

    def __init__(self, n_freq_mask: int = 2, freq_width: int = 16,
                 n_time_mask: int = 2, time_width: int = 12):
        super().__init__()
        self.nf, self.fw, self.nt, self.tw = n_freq_mask, freq_width, n_time_mask, time_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        B, _, Fr, T = x.shape
        for _ in range(self.nf):
            w = int(torch.randint(0, self.fw + 1, (1,)).item())
            if w:
                f0 = int(torch.randint(0, max(1, Fr - w), (1,)).item())
                x[:, :, f0:f0 + w, :] = 0
        for _ in range(self.nt):
            w = int(torch.randint(0, self.tw + 1, (1,)).item())
            if w:
                t0 = int(torch.randint(0, max(1, T - w), (1,)).item())
                x[:, :, :, t0:t0 + w] = 0
        return x


def onset_strength(mel: torch.Tensor) -> torch.Tensor:
    """Spectral-flux onset-strength channels appended to the mel input.

    Half-wave-rectified first difference along time, at two lags. Classical
    onset detectors were engineered for exactly the 50 ms precision this metric
    demands, they are two subtractions, and no learned front-end reproduces them
    for free - so we hand them to the network directly.

    mel: (B, 1, F, T) -> (B, 2, F, T)
    """
    x = mel[:, 0]                                            # (B, F, T)
    flux1 = F.pad(x[..., 1:] - x[..., :-1], (1, 0)).clamp(min=0)
    flux3 = F.pad(x[..., 3:] - x[..., :-3], (3, 0)).clamp(min=0)
    return torch.stack([flux1, flux3], dim=1)
