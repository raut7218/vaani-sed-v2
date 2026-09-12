"""Training loop for the span model.

Differences from v1 that are deliberate, not incidental:

* **No mean-teacher consistency term.** Consistency is a *smoothness* prior,
  and smoothness is precisely what destroys 50 ms boundary precision. What
  survives is an EMA of the *weights*, which reduces variance without touching
  the sharpness of any single prediction.
* **Staged encoder unfreezing.** The encoders start frozen; after
  `train.unfreeze_epoch` the top `train.unfreeze_blocks` transformer blocks are
  trained at a much lower LR - the ATST-SED recipe.
* **Model selection on the competition score**, not on loss.

Throughput (2x T4, the Kaggle target)
------------------------------------
* Validation runs on **both** GPUs, each on half the held-out set, and scores
  the EMA weights only by default (`train.eval_raw` adds the raw ones back). It
  used to run twice over the whole set on rank 0 while rank 1 idled at a barrier.
  `train.eval_every` / `train.eval_last` evaluate sparsely early on and every
  epoch at the end, where the best checkpoint actually comes from.
* Gain/noise augmentation runs on the GPU, so the 4 host vCPUs only decode audio.
* The EMA only averages tensors that can change: two frozen ViT-Bases are
  ~170M parameters of pure memory traffic on every step otherwise.
* Checkpoints are written from a background thread once the state is on the host.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import shutil
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import (TierBatchSampler, VaaniSpanDataset, collate,  # noqa: E402
                              load_manifest, split_manifest)
from src.data.labels import LabelEncoder                                    # noqa: E402
from src.evaluation.metrics import evaluate                                 # noqa: E402
from src.infer.runner import candidates_to_events, run_loader               # noqa: E402
from src.models.encoders import build_encoder                               # noqa: E402
from src.models.span_model import build_model                               # noqa: E402
from src.train.losses import SpanLoss                                       # noqa: E402


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def log(*a):
    if is_main():
        print(*a, flush=True)


def setup_ddp(timeout_min: int = 60) -> tuple:
    """Returns (device, ddp, rank, world_size)."""
    if "RANK" in os.environ and torch.cuda.is_available():
        dist.init_process_group("nccl",
                                timeout=timedelta(minutes=max(1, int(timeout_min))))
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        return (torch.device("cuda", local), True,
                dist.get_rank(), dist.get_world_size())
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return dev, False, 0, 1


def clone_module(model: torch.nn.Module) -> torch.nn.Module:
    """`copy.deepcopy` of a module that may carry `torch.nn.utils.weight_norm`.

    The deprecated `weight_norm` caches the weight it computes as a *plain
    attribute* on the module, and that tensor is a non-leaf, which `deepcopy`
    refuses outright (pytorch#103001). BEATs builds its `pos_conv` that way, so
    the cached tensors are lifted off for the duration of the copy and put
    straight back, on the clone too.
    """
    stash = []
    for name, mod in model.named_modules():
        for k, v in list(mod.__dict__.items()):
            if torch.is_tensor(v) and not v.is_leaf:
                stash.append((name, mod, k, v))
                del mod.__dict__[k]
    try:
        clone = copy.deepcopy(model)
    finally:
        for _, mod, k, v in stash:
            mod.__dict__[k] = v
    for name, _, k, v in stash:
        setattr(clone.get_submodule(name), k, v.detach().clone())
    return clone


def _unwrap(m):
    return m.module if hasattr(m, "module") else m


class EMA:
    """Exponential moving average of the weights, over what can actually change.

    Frozen parameters are identical in the model and the shadow for as long as
    they stay frozen, so averaging them is ~170M parameters of memory traffic per
    step for nothing. `refresh()` re-pairs after an unfreeze: the newly trainable
    blocks' shadow copies still hold the pretrained values, which is exactly
    where their average starts.
    """

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = clone_module(_unwrap(model)).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.refresh(model)

    def refresh(self, model) -> None:
        live = _unwrap(model)
        sp = dict(self.shadow.named_parameters())
        sb = dict(self.shadow.named_buffers())
        self._src, self._dst = [], []
        for n, p in live.named_parameters():
            if p.requires_grad:
                self._src.append(p.detach())
                self._dst.append(sp[n])
        for n, b in live.named_buffers():
            if b.dtype.is_floating_point and n in sb:
                self._src.append(b)
                self._dst.append(sb[n])

    @torch.no_grad()
    def update(self) -> None:
        torch._foreach_lerp_(self._dst, self._src, 1.0 - self.decay)


def save_atomic(obj, path: Path) -> None:
    """torch.save via a temp file + rename, so a failed write keeps the old file.

    A full disk aborts `torch.save` mid-record and torch reports it as
    `unexpected pos ...`, which reads like a corrupt tensor, not like ENOSPC -
    so the free-space figure is added to the error.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(obj, tmp)
        tmp.replace(path)
    except Exception as e:  # noqa: BLE001 - re-raised; this only adds the reason
        tmp.unlink(missing_ok=True)
        free = shutil.disk_usage(path.parent).free / float(1 << 30)
        raise RuntimeError(
            "could not write %s (%.2f GB free): %s\nUnder ~1 GB free this is the "
            "disk, whatever the message says; %s was left as it was."
            % (path, free, e, path.name)) from e


def to_host(obj):
    """Deep-copy every tensor in `obj` to the CPU, for a background write."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        out = type(obj)((k, to_host(v)) for k, v in obj.items())
        if hasattr(obj, "_metadata"):          # state_dict version info
            out._metadata = obj._metadata
        return out
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_host(v) for v in obj)
    return obj


class AsyncSaver:
    """One write at a time, off the training thread. `wait()` before exit."""

    def __init__(self):
        self._t = None
        self._err = None

    def submit(self, jobs) -> None:
        """jobs: [(obj_on_host, path), ...], written in order."""
        self.wait()

        def run():
            try:
                for obj, path in jobs:
                    save_atomic(obj, path)
            except Exception as e:  # noqa: BLE001 - surfaced on the next wait()
                self._err = e

        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()

    def wait(self) -> None:
        if self._t is not None:
            self._t.join()
            self._t = None
        if self._err is not None:
            e, self._err = self._err, None
            raise e


def build_refs(loader, fps: float) -> dict:
    """Class-agnostic reference events, in the *cropped* clip's coordinates.

    Built from the same tensors the model sees so evaluation cannot silently
    score against events that fell outside the evaluation crop.
    """
    refs = {}
    for batch in loader:
        sp = batch["spans"].numpy()
        for i, uid in enumerate(batch["uid"]):
            ev = [[float(a) / fps, float(b) / fps] for a, b in sp[i] if b > a >= 0]
            refs[uid] = sorted(ev)
    return refs


def split_batch(batch: dict, n: int) -> list:
    """Split a collated batch dict into `n` near-equal chunks along dim 0."""
    bsz = batch["wav"].size(0)
    n = max(1, min(n, bsz))
    idx = torch.linspace(0, bsz, n + 1).round().long().tolist()
    return [{k: v[lo:hi] for k, v in batch.items()}
            for lo, hi in zip(idx[:-1], idx[1:]) if hi > lo]


def gpu_augment(wav: torch.Tensor) -> torch.Tensor:
    """Per-clip gain in [0.85, 1.15], and 1e-3 white noise on half the clips.

    The same augmentation the dataset used to apply on the host, where drawing
    128k Gaussian samples per clip was a real share of 4 vCPUs' decode budget.
    """
    B = wav.size(0)
    gain = torch.empty((B, 1), device=wav.device).uniform_(0.85, 1.15)
    noisy = (torch.rand((B, 1), device=wav.device) < 0.5).to(wav.dtype)
    return wav * gain + torch.randn_like(wav) * (1e-3 * noisy)


def make_param_groups(model, lr: float, enc_scale: float, wd: float):
    """Param groups, each carrying its own `base_lr` for the schedule to scale.

    Norms and biases are excluded from weight decay *inside* the encoder group
    too; decaying a pretrained LayerNorm's gain toward zero erases the
    statistics the checkpoint was trained with. The encoder groups come last,
    so an unfreeze can append them to a live optimiser in the same order a
    resume rebuilds them.
    """
    enc, enc_nd, rest, nodecay = [], [], [], []
    for n, p in _unwrap(model).named_parameters():
        if not p.requires_grad:
            continue
        flat = p.ndim <= 1 or n.endswith(".bias")
        if n.startswith("encoder.encoders"):
            (enc_nd if flat else enc).append(p)
        elif flat:
            nodecay.append(p)
        else:
            rest.append(p)
    groups = [{"params": rest, "lr": lr, "weight_decay": wd},
              {"params": nodecay, "lr": lr, "weight_decay": 0.0}]
    if enc:
        groups.append({"params": enc, "lr": lr * enc_scale, "weight_decay": wd})
    if enc_nd:
        groups.append({"params": enc_nd, "lr": lr * enc_scale, "weight_decay": 0.0})
    for g in groups:
        g["base_lr"] = g["lr"]
    return groups


def cosine_lr(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def should_eval(epoch: int, epochs: int, t: dict) -> bool:
    every = max(1, int(t.get("eval_every", 1)))
    return epoch % every == 0 or epoch > epochs - int(t.get("eval_last", 0)) \
        or epoch == epochs


def validate(net, loader, refs, device, fps, pp, amp, ddp) -> dict | None:
    """Score `net` with every rank decoding its own shard. Result on rank 0 only."""
    cands = run_loader(net, loader, device, fps, pp, amp=amp)
    preds = {u: candidates_to_events(c, pp) for u, c in cands.items()}
    if ddp:
        parts = [None] * dist.get_world_size() if is_main() else None
        dist.gather_object(preds, parts, dst=0)
        if not is_main():
            return None
        preds = {}
        for p in parts:
            preds.update(p)
    return evaluate(preds, refs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data/vaani")
    ap.add_argument("--extra-data", nargs="*", default=[],
                    help="extra manifest roots (e.g. the packed synthetic clips); "
                         "train-only, never validated on")
    ap.add_argument("--out", default="runs/v2")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=0, help="per GPU")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vad-dir", default="")
    ap.add_argument("--time-limit-h", type=float, default=0.0,
                    help="stop after the last epoch that fits in this many hours "
                         "(state.pt is written; resume in the next session)")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="stop each epoch after this many steps (profiling only)")
    ap.add_argument("--resume", default="",
                    help="'auto' resumes <out>/state.pt if present (Kaggle sessions "
                         "cap at ~12 h), or give an explicit state.pt path")
    ap.add_argument("--no-encoders", action="store_true",
                    help="mel branch only; the encoder ablation")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.seed:
        cfg["seed"] = args.seed
    if args.no_encoders:
        cfg["model"]["encoders"] = []

    d, t, m = cfg["data"], cfg["train"], cfg["model"]
    device, ddp, rank, world = setup_ddp(int(t.get("ddp_timeout_min", 60)))
    # Rank-offset so the augmentation RNG differs across ranks.
    torch.manual_seed(cfg["seed"] + rank)
    np.random.seed(cfg["seed"] + rank)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    amp = bool(t.get("amp", True)) and device.type == "cuda"

    out_dir = Path(args.out)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config_run.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    le = LabelEncoder(expand_vehicle=bool(d.get("expand_vehicle", True)))
    recs = load_manifest(Path(args.data) / "manifest.jsonl")
    tr_recs, va_recs = split_manifest(recs, fold=args.fold,
                                      n_folds=int(d.get("n_folds", 5)),
                                      seed=int(cfg["seed"]))
    for extra in args.extra_data:
        er = load_manifest(Path(extra) / "manifest.jsonl")
        for r in er:
            r["_root"] = str(extra)
            # Its own sampling pool, but still gold for the loss: the synthetic
            # clips' boundaries are exact by construction, so they deserve full
            # boundary weight - what they must not do is *define* the gold pool.
            # They outnumber the real gold clips 20000 to ~8900, so leaving them
            # in it made 69% of every "gold" draw synthetic and set the epoch
            # length from a pool that is mostly our own splicing.
            r["pool"] = "synth"
        # Extras are train-only: validating on synthetic audio would measure how
        # well the model reads our own splicing, not the competition's task.
        tr_recs += er
        log("[data] + %d extra clips from %s" % (len(er), extra))
    log("[data] %d train / %d val  (fold %d of %d)"
        % (len(tr_recs), len(va_recs), args.fold, int(d.get("n_folds", 5))))

    vad = args.vad_dir or d.get("vad_dir") or None
    ds_kw = dict(root=args.data, le=le, clip_len=float(d["clip_len"]),
                 sr=int(d["sr"]), fps=float(d["fps"]), vad_dir=vad)
    tr_ds = VaaniSpanDataset(tr_recs, train=True, **ds_kw)
    # Each rank validates its own stride of the held-out set.
    va_ds = VaaniSpanDataset(va_recs[rank::world], train=False, **ds_kw)

    bs = int(t["batch_size"])
    ebs = int(t.get("eval_batch_size", 0)) or 2 * bs
    # Workers are per *process*, one process per GPU: split the host between them.
    nw = max(0, min(int(t.get("num_workers", 4)), (os.cpu_count() or 2) // world))
    # The sampler seed is identical across ranks on purpose: every rank builds
    # the same global batch list and takes its own disjoint slice of it.
    sampler = TierBatchSampler(tr_recs, bs, t.get("tier_quotas"), seed=cfg["seed"],
                               rank=rank, world_size=world,
                               steps_per_epoch=int(t.get("steps_per_epoch", 0)))
    gen = torch.Generator()
    gen.manual_seed(int(cfg["seed"]) + 1000 * rank)     # per-rank worker seeds
    tr_ld = DataLoader(tr_ds, batch_sampler=sampler, num_workers=nw,
                       collate_fn=collate, pin_memory=True, generator=gen,
                       persistent_workers=nw > 0,
                       prefetch_factor=int(t.get("prefetch_factor", 4)) if nw else None)
    va_ld = DataLoader(va_ds, batch_size=ebs, shuffle=False, num_workers=nw,
                       collate_fn=collate, pin_memory=True, persistent_workers=nw > 0,
                       prefetch_factor=int(t.get("prefetch_factor", 4)) if nw else None)

    enc = build_encoder(m, ckpt_dir=m.get("beats_dir", "checkpoints"))
    model = build_model(cfg, len(le), enc).to(device)

    def wrap_ddp(net):
        if not ddp:
            return net
        ddp_net = torch.nn.parallel.DistributedDataParallel(
            _unwrap(net), device_ids=[device.index],
            find_unused_parameters=bool(t.get("find_unused_parameters", False)),
            gradient_as_bucket_view=True)
        if bool(t.get("fp16_allreduce", True)):
            # Two T4s talk over PCIe; halving the bytes of every gradient
            # all-reduce is most of DDP's cost once the encoder blocks train.
            # An overflow shows up as inf, which the GradScaler already skips.
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
            ddp_net.register_comm_hook(None, default_hooks.fp16_compress_hook)
        return ddp_net

    crit = SpanLoss(cfg, len(le), int(m.get("n_bins", 16)))
    lr = float(t["lr"])
    enc_scale = float(t.get("encoder_lr_scale", 0.05))
    wd = float(t["weight_decay"])
    # The EMA shadow is cloned before any unfreeze, so it never carries the
    # live model's activation-checkpoint wrappers.
    ema = EMA(model, float(t.get("ema_decay", 0.999)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    epochs = int(t["epochs"])
    total_steps = epochs * len(sampler)
    warmup = int(t.get("warmup_steps", 500))
    step = 0
    history, best, start_epoch = [], -1.0, 1
    unfreeze_at = int(t.get("unfreeze_epoch", 6))
    n_unfreeze = int(t.get("unfreeze_blocks", 0))
    ckpt_blocks = bool(t.get("checkpoint_unfrozen", False))

    def unfreeze():
        got = _unwrap(model).encoder.unfreeze_last(n_unfreeze, ckpt_blocks)
        ema.refresh(model)
        return got

    # --- resume -----------------------------------------------------------
    # The optimiser, the scaler, the EMA shadow and the step counter all have to
    # come back: without the step counter the cosine schedule restarts and the
    # LR jumps back up, quietly undoing the previous session's progress.
    st = None
    if args.resume:
        p = (out_dir / "state.pt") if args.resume == "auto" else Path(args.resume)
        if p.exists():
            st = torch.load(str(p), map_location="cpu", weights_only=False)
            log("[resume] %s" % p)
        elif args.resume != "auto":
            raise SystemExit("--resume %s does not exist" % args.resume)
    if st is not None and int(st.get("unfroze_at_epoch", 0)):
        # Rebuild the trainable set *before* the optimiser, or the param groups
        # will not line up with the saved state.
        unfreeze()
    model = wrap_ddp(model)
    opt = torch.optim.AdamW(make_param_groups(model, lr, enc_scale, wd))
    if st is not None:
        _unwrap(model).load_state_dict(st["model"])
        base_lrs = [g["base_lr"] for g in opt.param_groups]
        opt.load_state_dict(st["opt"])
        for g, b in zip(opt.param_groups, base_lrs):
            g["base_lr"] = b
        scaler.load_state_dict(st["scaler"])
        ema.shadow.load_state_dict(st["ema"])      # in place: pairing stays valid
        step, best = int(st["step"]), float(st["best"])
        # The cosine schedule is indexed by raw step count and steps-per-epoch
        # changes with world size and batch size; carry the *fraction* over.
        old_total = int(st.get("total_steps", 0))
        if old_total and old_total != total_steps:
            step = int(round(step * total_steps / old_total))
            log("[resume] steps-per-epoch changed; rescaled step %d -> %d"
                % (int(st["step"]), step))
        history = st.get("history", [])
        start_epoch = int(st["epoch"]) + 1
        log("[resume] -> epoch %d, step %d, best %.4f" % (start_epoch, step, best))
        del st

    refs = {}
    if is_main():
        ref_ld = DataLoader(
            VaaniSpanDataset(va_recs, train=False, labels_only=True, **ds_kw),
            batch_size=ebs, shuffle=False, num_workers=nw, collate_fn=collate)
        refs = build_refs(ref_ld, float(d["fps"]))

    clip_params = [p for g in opt.param_groups for p in g["params"]]
    accum = max(1, int(t.get("unfreeze_grad_accum", 1)))
    log_every = int(t.get("log_every", 100))
    saver = AsyncSaver()
    n_params = sum(p.numel() for p in clip_params)
    log("[train] %d steps/epoch/rank x %d rank(s), batch %d/GPU, %.1fM trainable params"
        % (len(sampler), world, bs, n_params / 1e6))

    t_start, recent = time.time(), []
    for epoch in range(start_epoch, epochs + 1):
        if args.time_limit_h and epoch > start_epoch:
            # Kaggle kills a session at its limit and a killed commit keeps no
            # output, so stop while the next epoch would still not fit.
            left = args.time_limit_h * 3600 - (time.time() - t_start)
            epoch_s = max(recent[-3:])          # eval epochs run longer
            if left < 1.1 * epoch_s:
                log("[time] %.0f min left, an epoch takes %.0f min - stopping at "
                    "epoch %d; --resume auto continues from here"
                    % (left / 60, epoch_s / 60, epoch - 1))
                break
        t_epoch = time.time()
        is_unfrozen = epoch > unfreeze_at and n_unfreeze > 0
        if epoch == unfreeze_at + 1 and n_unfreeze > 0:
            got = unfreeze()
            log("[train] unfroze top blocks per encoder: %s" % got)
            # DDP fixes its gradient buckets from the parameters that had
            # `requires_grad` at construction, so it has to be rebuilt to reduce
            # the newly trainable ones. The optimiser keeps its state for
            # everything it already had and gains the encoder groups.
            model = wrap_ddp(model)
            for g in make_param_groups(model, lr, enc_scale, wd)[2:]:
                opt.add_param_group(g)
            clip_params = [p for g in opt.param_groups for p in g["params"]]
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

        model.train()
        acc, nb, t0 = {}, 0, time.time()
        t_wait, t_mark = 0.0, time.time()
        for batch in tr_ld:
            t_wait += time.time() - t_mark
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)
            batch["wav"] = gpu_augment(batch["wav"])
            f = cosine_lr(step, total_steps, warmup)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f

            opt.zero_grad(set_to_none=True)
            chunks = split_batch(batch, accum) if (is_unfrozen and accum > 1) else [batch]
            logs = {}
            for i, chunk in enumerate(chunks):
                w = chunk["wav"].size(0) / batch["wav"].size(0)
                # Only the last micro-batch all-reduces.
                sync = contextlib.nullcontext() if (not ddp or i == len(chunks) - 1) \
                    else model.no_sync()
                with sync:
                    with torch.autocast(device_type=device.type, enabled=amp):
                        out = model(chunk["wav"], chunk["frame_valid"])
                        c_loss, c_logs = crit(out, chunk)
                    scaler.scale(c_loss * w).backward()
                for k, v in c_logs.items():
                    logs[k] = logs.get(k, 0.0) + v.detach() * w
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(clip_params, float(t.get("grad_clip", 5.0)))
            scaler.step(opt)
            scaler.update()
            ema.update()
            step += 1
            nb += 1
            for k, v in logs.items():
                acc[k] = acc.get(k, 0.0) + v          # stays on device
            if log_every and nb % log_every == 0:
                el = time.time() - t0
                log("  [%d/%d] step %d  loss %.4f  %.2f it/s  %.0f clips/s  "
                    "data-wait %.0f%%  mem %.1f GB"
                    % (epoch, epochs, nb, float(acc["loss"]) / nb, nb / el,
                       nb * bs * world / el, 100 * t_wait / el,
                       torch.cuda.max_memory_allocated() / 2**30
                       if device.type == "cuda" else 0.0))
            if args.max_steps and nb >= args.max_steps:
                break
            t_mark = time.time()

        train_s = time.time() - t0
        msg = "  ".join("%s %.4f" % (k, float(v) / max(nb, 1)) for k, v in acc.items())
        log("[epoch %d/%d] %s" % (epoch, epochs, msg))
        log("[epoch %d/%d] train %.0fs (%.0f clips/s, data-wait %.0f%%, peak mem %.1f GB)"
            % (epoch, epochs, train_s, nb * bs * world / max(train_s, 1e-6),
               100 * t_wait / max(train_s, 1e-6),
               torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0))

        jobs = []                       # checkpoint writes for rank 0, in order
        if should_eval(epoch, epochs, t):
            t1 = time.time()
            nets = [("ema", ema.shadow)]
            if bool(t.get("eval_raw", False)):
                nets.append(("raw", _unwrap(model)))
            for name, net in nets:
                r = validate(net, va_ld, refs, device, float(d["fps"]),
                             cfg.get("postproc"), amp, ddp)
                if r is None:
                    continue
                r.update(epoch=epoch, which=name)
                history.append(r)
                log("   [%s] F1 %.4f  Dice %.4f  score %.4f  (tp %d fp %d fn %d)"
                    % (name, r["event_f1"], r["segment_dice"], r["score"],
                       r["tp"], r["fp"], r["fn"]))
                if r["score"] > best:
                    best = r["score"]
                    jobs.append((to_host(
                        {"model": net.state_dict(), "cfg": cfg,
                         "classes": le.classes, "score": best,
                         "which": name, "epoch": epoch}), out_dir / "best.pt"))
            log("   eval %.0fs" % (time.time() - t1))

        if is_main():
            (out_dir / "history.json").write_text(json.dumps(history, indent=1),
                                                  encoding="utf-8")
            # Full resume state every epoch. Snapshotted to host memory here,
            # written behind the next epoch's steps; the rename keeps the
            # previous state.pt intact until the new one is complete.
            jobs.append((to_host(
                {"model": _unwrap(model).state_dict(), "opt": opt.state_dict(),
                 "scaler": scaler.state_dict(), "ema": ema.shadow.state_dict(),
                 "step": step, "best": best, "total_steps": total_steps,
                 "epoch": epoch, "history": history, "cfg": cfg,
                 "unfroze_at_epoch": is_unfrozen}), out_dir / "state.pt"))
            saver.submit(jobs)

        # Keep the ranks in step: rank 0's scoring and host snapshot are the
        # only work here that its peer does not share.
        if ddp:
            dist.barrier()
        # Every rank must agree on when to stop, so the epoch time is rank 0's.
        epoch_s = time.time() - t_epoch
        if ddp:
            tt = torch.tensor([epoch_s], device=device)
            dist.broadcast(tt, 0)
            epoch_s = float(tt.item())
        recent.append(epoch_s)

    if is_main():
        saver.wait()
    log("[done] best val score %.4f -> %s" % (best, out_dir / "best.pt"))
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
