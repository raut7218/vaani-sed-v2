"""Pretrained encoders, and the fusion that puts them on one time grid.

Why this file exists at all
---------------------------
v1 used frozen BEATs as its only pretrained encoder. BEATs is a *patch* model:
16x16 patches over a 100 fps mel, i.e. one token per **160 ms**. The competition
metric matches an event only when both boundaries land inside
``max(0.2 * duration, 0.05)`` s, and on this corpus the 10th percentile of that
tolerance is **59 ms**. So v1's primary encoder was ~3x coarser than the
precision the metric demands.

ATST-Frame was built for frame-level tasks and runs at **40 ms**. So ATST-Frame
is the primary encoder here and BEATs is a semantic side-channel: BEATs is good
at *what*, ATST is good at *when*.

Speed
-----
These two ViT-Base encoders are most of the FLOPs in a training step, so both
run their attention through `F.scaled_dot_product_attention` (the fused
memory-efficient kernel on a T4) instead of the upstream explicit
``softmax(q @ k.T)``. The explicit form materialises every (B, H, L, L) score
tensor several times over - BEATs even promotes it to fp32 for the softmax -
and at L = 392 tokens that elementwise traffic cost about as much as the
encoder's matmuls. The fused kernel is the same maths in one pass, and it does
not keep the attention matrix around for backward either.

Once unfrozen, only the top blocks carry `requires_grad`, so autograd records
nothing for the frozen bottom of the stack and there is nothing to recompute:
the forward runs once and backward touches only the trainable blocks.
"""
from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _checkpoint

from src.models.frontend import resample_time

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _log(*a):
    """Print from rank 0 only - every rank builds its own encoder stack."""
    if os.environ.get("RANK", "0") == "0":
        print(*a, flush=True)


class EncoderBase(nn.Module):
    """(B, L) waveform at 16 kHz -> (B, T, D) time-major features."""

    out_dim: int = 0
    frame_ms: float = 40.0
    name: str = "base"

    def __init__(self):
        super().__init__()
        self.frozen = True

    def blocks(self) -> nn.ModuleList:
        raise NotImplementedError

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.backbone().eval()
        return self

    def backbone(self) -> nn.Module:
        raise NotImplementedError

    def freeze(self) -> None:
        for p in self.backbone().parameters():
            p.requires_grad = False
        self.frozen = True
        self.backbone().eval()

    def unfreeze_last(self, n_blocks: int, ckpt: bool = False) -> int:
        """Make the top `n_blocks` transformer blocks trainable. Returns count.

        `ckpt` recomputes those blocks' activations in backward instead of
        keeping them - only the trainable blocks, never the frozen stack under
        them, which builds no graph at all.
        """
        if n_blocks <= 0:
            return 0
        blocks = self.blocks()
        top = blocks[max(0, len(blocks) - n_blocks):]
        for blk in top:
            for p in blk.parameters():
                p.requires_grad = True
            if ckpt:
                blk.forward = _checkpointed_forward.__get__(blk)
        self.frozen = False
        return len(top)

    def _grad_ctx(self):
        return torch.no_grad() if self.frozen else contextlib.nullcontext()


def _checkpointed_forward(self, *args, **kwargs):
    # Bound to the block as a method (not a closure over it), so `deepcopy` -
    # the EMA shadow is one - rebinds it to the copy instead of calling back
    # into the live model.
    fwd = type(self).forward.__get__(self)
    if not torch.is_grad_enabled():
        return fwd(*args, **kwargs)
    return _checkpoint.checkpoint(fwd, *args, use_reentrant=False, **kwargs)


# --------------------------------------------------------------------------- #
# BEATs - 160 ms tokens. Semantic side-channel.
# --------------------------------------------------------------------------- #
BEATS_REPO = "lpepino/beats_ckpts"
BEATS_FILE = "BEATs_iter3_plus_AS2M.pt"
BEATS_FBANK_MEAN, BEATS_FBANK_STD = 15.41663, 6.55582


def download_beats(dest="checkpoints"):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    local = dest / BEATS_FILE
    if local.exists():
        return local
    try:
        from huggingface_hub import hf_hub_download
        import shutil
        p = hf_hub_download(repo_id=BEATS_REPO, filename=BEATS_FILE, repo_type="model")
        shutil.copy(p, local)
        return local
    except Exception as e:                                        # noqa: BLE001
        _log("[beats] download failed: %s" % e)
        return None


class KaldiFbank(nn.Module):
    """`torchaudio.compliance.kaldi.fbank` with BEATs' arguments, batched.

    Upstream BEATs calls the Kaldi routine once per clip in a Python loop - a
    dozen small kernels per clip, so a 48-clip batch issued several hundred
    launches just to build its input. Same arithmetic, one pass over the batch:
    25 ms povey frames every 10 ms (snip_edges), DC removal, 0.97 pre-emphasis,
    512-point power spectrum, 128 Kaldi mel bins from 20 Hz, log with the float
    epsilon floor. `tests/test_components.py` checks it against torchaudio.
    """

    def __init__(self, sr: int = 16000, n_mels: int = 128):
        super().__init__()
        from torchaudio.compliance.kaldi import get_mel_banks
        self.win, self.hop, self.n_fft = int(0.025 * sr), int(0.010 * sr), 512
        banks, _ = get_mel_banks(n_mels, self.n_fft, float(sr), 20.0, 0.0,
                                 100.0, -500.0, 1.0)
        self.register_buffer("banks", F.pad(banks, (0, 1)).float(), persistent=False)
        win = torch.hann_window(self.win, periodic=False).pow(0.85)
        self.register_buffer("window", win, persistent=False)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=wav.device.type, enabled=False):
            x = wav.float() * 32768.0
            fr = x.unfold(-1, self.win, self.hop)                  # (B, T, 400)
            fr = fr - fr.mean(dim=-1, keepdim=True)
            prev = torch.cat([fr[..., :1], fr[..., :-1]], dim=-1)
            fr = (fr - 0.97 * prev) * self.window
            spec = torch.fft.rfft(fr, n=self.n_fft).abs().pow(2.0)
            mel = spec @ self.banks.T
            eps = torch.finfo(torch.float32).eps
            return mel.clamp(min=eps).log()                         # (B, T, 128)


def _beats_sdpa(self, query, key, value, key_padding_mask=None,
                incremental_state=None, need_weights=False, static_kv=False,
                attn_mask=None, before_softmax=False, need_head_weights=False,
                position_bias=None):
    """BEATs' self-attention through the fused kernel. Same maths as upstream.

    Upstream scales q by 1/alpha, subtracts each row's max and multiplies alpha
    back in before adding the gated relative-position bias - a numerical-
    stability trick for an explicit fp16 softmax. Softmax is shift-invariant per
    row, so the scores are exactly ``q.k * scaling + bias``, which is what SDPA
    computes (with fp32 accumulation). `position_bias` is carried between layers
    as (1, H, L, L) rather than repeated per clip.
    """
    L, B, E = query.shape
    H, d = self.num_heads, self.head_dim
    q = self.q_proj(query).view(L, B, H, d).permute(1, 2, 0, 3)     # (B, H, L, d)
    k = self.k_proj(query).view(L, B, H, d).permute(1, 2, 0, 3)
    v = self.v_proj(query).view(L, B, H, d).permute(1, 2, 0, 3)

    bias = None
    if self.has_relative_attention_bias:
        if position_bias is None:
            position_bias = self.compute_bias(L, L).unsqueeze(0)       # (1, H, L, L)
        bias = position_bias
        if self.gru_rel_pos == 1:
            g = self.grep_linear(q).view(B, H, L, 2, 4).sum(-1)
            gate_a, gate_b = torch.sigmoid(g).chunk(2, dim=-1)
            bias = (gate_a * (gate_b * self.grep_a - 1.0) + 2.0) * position_bias
    if key_padding_mask is not None:
        pad = key_padding_mask.view(B, 1, 1, L).to(torch.bool)
        bias = (bias if bias is not None else q.new_zeros((B, 1, L, L)))
        bias = bias.masked_fill(pad, float("-inf"))
    if bias is not None:
        bias = bias.to(q.dtype).expand(B, H, L, L)

    p = self.dropout_module.p if self.training else 0.0
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=p,
                                         scale=self.scaling)
    out = out.permute(2, 0, 1, 3).reshape(L, B, E)
    return self.out_proj(out), None, position_bias


class BEATsEncoder(EncoderBase):
    frame_ms = 160.0
    name = "beats"

    def __init__(self, ckpt_path, freeze: bool = True):
        super().__init__()
        from third_party.beats.BEATs import BEATs, BEATsConfig
        from third_party.beats import backbone as _bb

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg = BEATsConfig(ckpt["cfg"])
        model = BEATs(cfg)
        missing, _ = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            _log("[beats] missing keys: %d (first: %s)" % (len(missing), missing[:3]))
        model.predictor = None      # we want hidden states, not AudioSet logits
        # LayerDrop draws from NumPy's global RNG, independently on each rank,
        # so once the top blocks train it drops a *different* trainable layer on
        # each GPU. DDP (static graph, no unused-parameter search) cannot reduce
        # a parameter that one rank never used, and at 0.05 per layer over 4
        # trainable layers x 2 ranks that happens on a third of all steps.
        model.encoder.layerdrop = 0.0
        for m in model.modules():
            if isinstance(m, _bb.MultiheadAttention):
                m.forward = _beats_sdpa.__get__(m)
        self.beats = model
        self.fbank = KaldiFbank()
        self.out_dim = cfg.encoder_embed_dim
        self.n_freq = 128 // cfg.input_patch_size
        if freeze:
            self.freeze()
        else:
            self.frozen = False

    def backbone(self) -> nn.Module:
        return self.beats

    def blocks(self) -> nn.ModuleList:
        return self.beats.encoder.layers

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        b = self.beats
        with self._grad_ctx():
            fb = (self.fbank(wav) - BEATS_FBANK_MEAN) / (2 * BEATS_FBANK_STD)
            x = b.patch_embedding(fb.unsqueeze(1))                  # (B, C, T/16, 8)
            B, C, Tp, Fp = x.shape
            x = x.reshape(B, C, Tp * Fp).transpose(1, 2)
            x = b.layer_norm(x)
            if b.post_extract_proj is not None:
                x = b.post_extract_proj(x)
            x = b.dropout_input(x)
            x, _ = b.encoder(x)                                     # (B, Tp*Fp, D)
            # Tokens are time-major with `Fp` frequency patches per 160 ms step.
            # Pool the frequency patches so the sequence is a time axis: read as
            # one flat sequence and interpolated, consecutive output frames each
            # saw a different frequency band.
            x = x.reshape(B, Tp, Fp, -1).mean(dim=2)                # (B, Tp, D)
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- #
# ATST-Frame - 40 ms tokens. The primary encoder.
# --------------------------------------------------------------------------- #
# ATST-Frame takes *its own* mel: 64 bands, 60-7800 Hz, 10 ms hop, dB, min-max
# scaled to [-1, 1] with the constants the checkpoint was trained under. These
# numbers are copied from the upstream `ATSTTransform` / `ATSTNorm` pair.
ATST_MEL = dict(sample_rate=16000, f_min=60, f_max=7800, hop_length=160,
                win_length=1024, n_fft=1024, n_mels=64)
ATST_DB_MIN, ATST_DB_MAX = -79.6482, 50.6842


def _atst_sdpa(self, x, mask):
    """Upstream `Attention.forward` through the fused kernel (mask is None)."""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    p = self.attn_drop.p if self.training else 0.0
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p,
                                       scale=self.scale)
    x = x.transpose(1, 2).reshape(B, N, C)
    return self.proj_drop(self.proj(x)), None


class ATSTFrameEncoder(EncoderBase):
    """Adapter around the official ATST-Frame implementation.

    `scripts/fetch_encoders.py` vendors the upstream source into
    `third_party/atst/`, and the loader asserts the checkpoint load was clean -
    a silently half-loaded encoder trains fine and scores badly.
    """

    frame_ms = 40.0
    name = "atst_frame"

    def __init__(self, ckpt_path, freeze: bool = True, min_load_frac: float = 0.9):
        super().__init__()
        import torchaudio
        model, dim = _load_atst(ckpt_path, min_load_frac)
        for m in model.modules():
            if type(m).__name__ == "Attention" and hasattr(m, "qkv"):
                m.forward = _atst_sdpa.__get__(m)
        self.atst = model
        self.out_dim = dim
        self.mel = torchaudio.transforms.MelSpectrogram(**ATST_MEL)
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        if freeze:
            self.freeze()
        else:
            self.frozen = False

    def backbone(self) -> nn.Module:
        return self.atst

    def blocks(self) -> nn.ModuleList:
        return self.atst.blocks

    def features(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, L) waveform -> (B, 64, T) normalised mel, exactly as upstream."""
        with torch.autocast(device_type=wav.device.type, enabled=False):
            spec = self.mel(wav.float())
            spec = self.amp_to_db(spec).clamp(min=-50, max=80)
            spec = (spec - ATST_DB_MIN) / (ATST_DB_MAX - ATST_DB_MIN) * 2.0 - 1.0
        return spec

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        with self._grad_ctx():
            spec = self.features(wav)                      # (B, 64, T)
            # Upstream hardcodes the 10 s length its clips always had; ours are
            # `clip_len` long, so pass the real one.
            length = torch.full((spec.size(0),), float(spec.size(-1)),
                                device=spec.device, dtype=torch.float32)
            out = self.atst.get_intermediate_layers(spec.unsqueeze(1), length, 1,
                                                    scene=False)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# The checkpoint layouts in circulation store the same FrameAST tensors under
# different prefixes, and `load_state_dict(strict=False)` on the wrong prefix
# loads *nothing* while reporting success - which `min_load_frac` catches.
def _atst_frame_state(sd: dict) -> dict:
    """Reduce any known ATST checkpoint layout to bare FrameAST keys."""
    for key in ("state_dict", "model", "teacher", "student"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    out = {}
    for k, v in sd.items():
        if k.startswith("atst_frame.atst."):        # ATST-SED stage-1/2 finetune
            out[k[len("atst_frame.atst."):]] = v
        elif k.startswith("atst."):
            out[k[len("atst."):]] = v
        elif "model.teacher.encoder." in k:         # atst_as2M.ckpt, pretrained
            if "cls_token" in k:
                continue                            # FrameAST has no CLS token
            nk = k.split("model.teacher.encoder.", 1)[1]
            if nk.startswith("norm."):              # upstream renames the final norm
                nk = "norm_frame." + nk[len("norm."):]
            out[nk] = v
        elif "encoder.encoder.teacher_module." in k:
            continue
        elif "encoder.encoder.frame_encoder." in k:  # C2F
            out[k.split("encoder.encoder.frame_encoder.", 1)[1]] = v
        elif "encoder.encoder." in k:
            out[k.split("encoder.encoder.", 1)[1]] = v
    return out or sd


def _load_atst(ckpt_path, min_load_frac: float):
    """Import the vendored ATST source and load `ckpt_path` into it."""
    atst_dir = _ROOT / "third_party" / "atst"
    if not atst_dir.exists():
        raise FileNotFoundError(
            "third_party/atst is missing. Run `python scripts/fetch_encoders.py "
            "--atst` to vendor the upstream ATST-Frame source and checkpoint.")
    for p in (atst_dir, atst_dir / "ATST-SED"):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from atst_adapter import build_atst_frame

    model = build_atst_frame()
    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(_atst_frame_state(sd), strict=False)

    total = len(model.state_dict())
    loaded = total - len(missing)
    frac = loaded / max(total, 1)
    _log("[atst] loaded %d/%d tensors (%.1f%%), %d unexpected"
         % (loaded, total, 100 * frac, len(unexpected)))
    if frac < min_load_frac:
        raise RuntimeError(
            "ATST checkpoint only populated %.1f%% of the model (threshold %.0f%%). "
            "Refusing to train a partly-random encoder - check that the checkpoint "
            "matches the vendored source." % (100 * frac, 100 * min_load_frac))

    dim = getattr(model, "embed_dim", None) or getattr(model, "out_dim", None) or 768
    return model, int(dim)


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
class FusionEncoder(nn.Module):
    """Concatenate several pretrained encoders on one common time grid.

    Each branch is projected to `proj_dim` *before* concatenation so a 768-d
    encoder cannot swamp the others, and each gets its own LayerNorm because the
    activation scales differ wildly across checkpoints.
    """

    def __init__(self, encoders: Sequence[EncoderBase], proj_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Linear(e.out_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
            for e in encoders])
        self.drop = nn.Dropout(dropout)
        self.out_dim = proj_dim * len(encoders)
        self.names = [e.name for e in encoders]

    def unfreeze_last(self, n_blocks: int, ckpt: bool = False) -> dict:
        return {e.name: e.unfreeze_last(n_blocks, ckpt) for e in self.encoders}

    def forward(self, wav: torch.Tensor, target_len: int) -> torch.Tensor:
        outs: List[torch.Tensor] = []
        for enc, proj in zip(self.encoders, self.projs):
            h = proj(enc(wav).to(proj[0].weight.dtype))
            outs.append(resample_time(h, target_len))
        if not outs:
            return wav.new_zeros((wav.size(0), target_len, 0))
        return self.drop(torch.cat(outs, dim=-1))


def build_encoder(cfg: dict, ckpt_dir="checkpoints") -> FusionEncoder:
    """Assemble the fusion stack described by `cfg['encoders']`.

    Missing checkpoints are reported and skipped rather than fatal, so a fresh
    clone still trains end-to-end on the mel branch alone.
    """
    ckpt_dir = Path(ckpt_dir)
    want = cfg.get("encoders", ["atst_frame", "beats"])
    freeze = bool(cfg.get("freeze_encoders", True))
    built: List[EncoderBase] = []

    for name in want:
        try:
            if name == "beats":
                p = cfg.get("beats_ckpt") or download_beats(ckpt_dir)
                if not p or not Path(p).exists():
                    raise FileNotFoundError("BEATs checkpoint not found")
                built.append(BEATsEncoder(p, freeze=freeze))
            elif name == "atst_frame":
                p = cfg.get("atst_ckpt") or (ckpt_dir / "atst_frame.ckpt")
                if not Path(p).exists():
                    raise FileNotFoundError(
                        "no checkpoint at %s - run `python scripts/fetch_encoders.py "
                        "--atst`" % p)
                built.append(ATSTFrameEncoder(p, freeze=freeze))
            else:
                _log("[encoders] unknown encoder %r, skipping" % name)
                continue
            _log("[encoders] + %s (%.0f ms frames, %d-d)"
                 % (name, built[-1].frame_ms, built[-1].out_dim))
        except Exception as e:                                     # noqa: BLE001
            _log("[encoders] ! %s unavailable: %s" % (name, e))

    if not built:
        _log("[encoders] no pretrained encoder available - running on the "
             "high-resolution mel branch alone. Expect a much lower score.")
    return FusionEncoder(built, proj_dim=int(cfg.get("proj_dim", 256)),
                         dropout=float(cfg.get("dropout", 0.1)))
