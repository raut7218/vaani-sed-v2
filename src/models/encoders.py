"""Pretrained encoders, and the fusion that puts them on one time grid.

Why this file exists at all
---------------------------
v1 used frozen BEATs as its only pretrained encoder. BEATs is a *patch* model:
16x16 patches over a 100 fps mel, i.e. one token per **160 ms**. The competition
metric matches an event only when both boundaries land inside
``max(0.2 * duration, 0.05)`` s, and on this corpus the 10th percentile of that
tolerance is **59 ms**. So v1's primary encoder was ~3x coarser than the
precision the metric demands, and the model then linearly interpolated those
160 ms tokens up to the 40 ms output grid - manufacturing the smear that a
median filter was later asked to undo.

Measured on the v1 checkpoint: the class-agnostic posterior takes **760 ms** to
go from 10% to 90% of its range around a true onset.

ATST-Frame was built for frame-level tasks and runs at **40 ms**, and the
literature attributes its SED gains to precisely that (40 ms vs BEATs' 160 ms).
So ATST-Frame is the primary encoder here and BEATs is demoted to a semantic
side-channel: BEATs is good at *what*, ATST is good at *when*.

WavLM joins them because the published complementary-fusion result
(ATST-Frame + BEATs + WavLM) shows the three are complementary - v1's README
dismissed the WavLM family wholesale on a *classification* benchmark, which was
never the failing axis here.

Every encoder is optional. `build_encoder` degrades to whatever checkpoints are
actually present, and reports what it loaded, so the pipeline always runs.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn as nn

from src.models.frontend import resample_time

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class EncoderBase(nn.Module):
    """(B, L) waveform at 16 kHz -> (B, T, D) time-major features."""

    out_dim: int = 0
    frame_ms: float = 40.0
    name: str = "base"

    def unfreeze_last(self, n_blocks: int) -> int:
        """Make the top `n_blocks` transformer blocks trainable. Returns count."""
        return 0


# --------------------------------------------------------------------------- #
# BEATs - 160 ms tokens. Semantic side-channel.
# --------------------------------------------------------------------------- #
BEATS_REPO = "lpepino/beats_ckpts"
BEATS_FILE = "BEATs_iter3_plus_AS2M.pt"


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
        print("[beats] download failed: %s" % e)
        return None


class BEATsEncoder(EncoderBase):
    frame_ms = 160.0
    name = "beats"

    def __init__(self, ckpt_path, freeze: bool = True):
        super().__init__()
        from third_party.beats.BEATs import BEATs, BEATsConfig

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg = BEATsConfig(ckpt["cfg"])
        model = BEATs(cfg)
        missing, _ = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print("[beats] missing keys: %d (first: %s)" % (len(missing), missing[:3]))
        model.predictor = None      # we want hidden states, not AudioSet logits
        self.beats = model
        self.out_dim = cfg.encoder_embed_dim
        self.frozen = freeze
        if freeze:
            for p in self.beats.parameters():
                p.requires_grad = False
            self.beats.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.beats.eval()
        return self

    def unfreeze_last(self, n_blocks: int) -> int:
        if n_blocks <= 0:
            return 0
        layers = self.beats.encoder.layers
        for blk in layers[max(0, len(layers) - n_blocks):]:
            for p in blk.parameters():
                p.requires_grad = True
        self.frozen = False
        return min(n_blocks, len(layers))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            out = self.beats.extract_features(wav)
            feat = out[0] if isinstance(out, (tuple, list)) else out
        # A frozen encoder's non-finite value can only poison the trainable
        # branch downstream, so neutralise rather than propagate.
        return torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- #
# ATST-Frame - 40 ms tokens. The primary encoder.
# --------------------------------------------------------------------------- #
# ATST-Frame does not take a waveform. It takes *its own* mel: 64 bands,
# 60-7800 Hz, 10 ms hop, converted to dB and min-max scaled to [-1, 1] with the
# fixed constants the checkpoint was trained under. Feeding it anything else -
# our own 128-band mel, or a raw waveform - produces a tensor of the right shape
# and completely wrong statistics, which is the failure mode this file's docstring
# is about. These numbers are copied from the upstream
# `ATSTTransform` / `ATSTNorm` pair, not chosen.
ATST_MEL = dict(sample_rate=16000, f_min=60, f_max=7800, hop_length=160,
                win_length=1024, n_fft=1024, n_mels=64)
ATST_DB_MIN, ATST_DB_MAX = -79.6482, 50.6842


class ATSTFrameEncoder(EncoderBase):
    """Adapter around the official ATST-Frame implementation.

    We deliberately do *not* reimplement ATST from memory: a checkpointed
    architecture that silently half-loads is worse than no checkpoint at all,
    because it trains a partly-random encoder and still looks fine in the loss
    curve. `scripts/fetch_encoders.py` vendors the upstream source into
    `third_party/atst/`, and this class asserts the load was clean.
    """

    frame_ms = 40.0
    name = "atst_frame"

    def __init__(self, ckpt_path, freeze: bool = True, min_load_frac: float = 0.9):
        super().__init__()
        import torchaudio
        model, dim = _load_atst(ckpt_path, min_load_frac)
        self.atst = model
        self.out_dim = dim
        self.mel = torchaudio.transforms.MelSpectrogram(**ATST_MEL)
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        self.frozen = freeze
        if freeze:
            for p in self.atst.parameters():
                p.requires_grad = False
            self.atst.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.atst.eval()
        return self

    def unfreeze_last(self, n_blocks: int) -> int:
        if n_blocks <= 0:
            return 0
        blocks = _atst_blocks(self.atst)
        if blocks is None:
            return 0
        for blk in blocks[max(0, len(blocks) - n_blocks):]:
            for p in blk.parameters():
                p.requires_grad = True
        self.frozen = False
        return min(n_blocks, len(blocks))

    def features(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, L) waveform -> (B, 64, T) normalised mel, exactly as upstream."""
        # fp32 regardless of the surrounding autocast: the mel is a fixed
        # transform, and half-precision STFT costs accuracy for no speed here.
        with torch.autocast(device_type=wav.device.type, enabled=False):
            spec = self.mel(wav.float())
            spec = self.amp_to_db(spec).clamp(min=-50, max=80)
            spec = (spec - ATST_DB_MIN) / (ATST_DB_MAX - ATST_DB_MIN) * 2.0 - 1.0
        return spec

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            spec = self.features(wav)                      # (B, 64, T)
            n_mel = spec.size(-1)
            if hasattr(self.atst, "get_intermediate_layers"):
                # The upstream signature: (B, 1, mel, T) plus the unpadded mel
                # length, which becomes the token-level attention mask. Upstream
                # hardcodes 1001 because it only ever sees 10 s clips; ours are
                # `clip_len` long, so pass the real length or the tail of every
                # clip is masked out.
                length = torch.full((spec.size(0),), float(n_mel),
                                    device=spec.device, dtype=torch.float32)
                out = self.atst.get_intermediate_layers(
                    spec.unsqueeze(1), length, 1, scene=False)
            else:
                out = self.atst(spec.unsqueeze(1))
            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.dim() == 3 and out.size(1) == self.out_dim and out.size(2) != self.out_dim:
                out = out.transpose(1, 2)                 # (B, D, T) -> (B, T, D)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _atst_blocks(model: nn.Module):
    for attr in ("blocks", "layers", "encoder"):
        obj = getattr(model, attr, None)
        if isinstance(obj, (nn.ModuleList, nn.Sequential)):
            return obj
        if obj is not None:
            inner = getattr(obj, "blocks", None)
            if inner is None:
                inner = getattr(obj, "layers", None)
            if isinstance(inner, (nn.ModuleList, nn.Sequential)):
                return inner
    return None


# The three checkpoint layouts in circulation, in the order they are tested.
# Every one of them stores the same FrameAST tensors under a different prefix,
# and `load_state_dict(strict=False)` on the wrong prefix loads *nothing* while
# reporting success - which is exactly the silent half-load `min_load_frac`
# exists to catch. Mapping them explicitly is cheaper than catching it late.
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
            # upstream renames the encoder's final norm on the way in
            if nk.startswith("norm."):
                nk = "norm_frame." + nk[len("norm."):]
            out[nk] = v
        elif "encoder.encoder.teacher_module." in k:
            continue
        elif "encoder.encoder.frame_encoder." in k:  # C2F
            out[k.split("encoder.encoder.frame_encoder.", 1)[1]] = v
        elif "encoder.encoder." in k:
            out[k.split("encoder.encoder.", 1)[1]] = v
    # Nothing matched a known prefix: the checkpoint is already bare FrameAST.
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

    build = None
    for mod_name, fn_name in (
            ("atst_adapter", "build_atst_frame"),
            ("desed_task.nnet.atst.audio_transformer", "FrameASTModel"),
            ("audiossl.models.atst.frame", "ATSTFrame"),
            ("models.atst", "ATST")):
        try:
            mod = __import__(mod_name, fromlist=[fn_name])
            build = getattr(mod, fn_name)
            break
        except Exception:                                          # noqa: BLE001
            continue
    if build is None:
        raise ImportError(
            "Could not import an ATST model factory from third_party/atst. "
            "See scripts/fetch_encoders.py for the expected layout.")

    model = build() if callable(build) else build
    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(_atst_frame_state(sd), strict=False)

    total = len(model.state_dict())
    loaded = total - len(missing)
    frac = loaded / max(total, 1)
    print("[atst] loaded %d/%d tensors (%.1f%%), %d unexpected"
          % (loaded, total, 100 * frac, len(unexpected)))
    if frac < min_load_frac:
        raise RuntimeError(
            "ATST checkpoint only populated %.1f%% of the model (threshold %.0f%%). "
            "Refusing to train a partly-random encoder - check that the checkpoint "
            "matches the vendored source." % (100 * frac, 100 * min_load_frac))

    dim = getattr(model, "embed_dim", None) or getattr(model, "out_dim", None) or 768
    return model, int(dim)


# --------------------------------------------------------------------------- #
# WavLM - 20 ms tokens. Third fusion channel.
# --------------------------------------------------------------------------- #
class WavLMEncoder(EncoderBase):
    frame_ms = 20.0
    name = "wavlm"

    def __init__(self, freeze: bool = True, layer: int = 8):
        super().__init__()
        import torchaudio
        bundle = torchaudio.pipelines.WAVLM_BASE_PLUS
        self.model = bundle.get_model()
        self.layer = layer
        self.out_dim = 768
        self.frozen = freeze
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    def unfreeze_last(self, n_blocks: int) -> int:
        if n_blocks <= 0:
            return 0
        layers = self.model.encoder.transformer.layers
        for blk in layers[max(0, len(layers) - n_blocks):]:
            for p in blk.parameters():
                p.requires_grad = True
        self.frozen = False
        return min(n_blocks, len(layers))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            feats, _ = self.model.extract_features(wav, num_layers=self.layer + 1)
        return torch.nan_to_num(feats[-1], nan=0.0, posinf=0.0, neginf=0.0)


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

    def unfreeze_last(self, n_blocks: int) -> dict:
        return {e.name: e.unfreeze_last(n_blocks) for e in self.encoders}

    def frozen_names(self) -> List[str]:
        return [e.name for e in self.encoders if getattr(e, "frozen", False)]

    def encode_raw(self, wav: torch.Tensor) -> dict:
        """Raw features per frozen encoder, for caching across student/teacher."""
        out = {}
        for e in self.encoders:
            if getattr(e, "frozen", False):
                out[e.name] = e(wav)
        return out

    def forward(self, wav: torch.Tensor, target_len: int,
                cache: dict | None = None) -> torch.Tensor:
        outs: List[torch.Tensor] = []
        for enc, proj in zip(self.encoders, self.projs):
            raw = None if cache is None else cache.get(enc.name)
            if raw is None:
                raw = enc(wav)
            h = proj(raw.to(proj[0].weight.dtype))
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
            elif name == "wavlm":
                built.append(WavLMEncoder(freeze=freeze))
            else:
                print("[encoders] unknown encoder %r, skipping" % name)
                continue
            print("[encoders] + %s (%.0f ms frames, %d-d)"
                  % (name, built[-1].frame_ms, built[-1].out_dim))
        except Exception as e:                                     # noqa: BLE001
            print("[encoders] ! %s unavailable: %s" % (name, e))

    if not built:
        print("[encoders] no pretrained encoder available - running on the "
              "high-resolution mel branch alone. Expect a much lower score.")
    return FusionEncoder(built, proj_dim=int(cfg.get("proj_dim", 256)),
                         dropout=float(cfg.get("dropout", 0.1)))
