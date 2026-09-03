"""Training loop for the span model.

Differences from v1 that are deliberate, not incidental:

* **No mean-teacher consistency term.** v1 ran it at weight 2.0. Consistency is
  a *smoothness* prior, and smoothness is precisely what destroys 50 ms boundary
  precision. v1's own history shows student and teacher converged to within
  0.005 of each other by epoch 11 - past that point the term was contributing
  blur and nothing else. What survives is an EMA of the *weights* (plain
  parameter averaging), which reduces variance without touching the sharpness of
  any single prediction.
* **Staged encoder unfreezing.** The encoders start frozen; after
  `train.unfreeze_epoch` the top `train.unfreeze_blocks` transformer blocks are
  trained at a much lower LR. This is the ATST-SED recipe, and it is where a
  large part of that system's advantage comes from.
* **Model selection on the competition score**, not on loss.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

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


def setup_ddp() -> tuple:
    if "RANK" in os.environ and torch.cuda.is_available():
        dist.init_process_group("nccl")
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        return torch.device("cuda", local), True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return dev, False


class EMA:
    """Exponential moving average of the weights.

    Fused `_foreach` updates against a pairing built once: v1's EMA walked
    `named_parameters()` and issued two kernel launches per tensor, a few hundred
    per step, which a 2-vCPU host cannot feed.
    """

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(_unwrap(model)).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self._src, self._dst = [], []
        for (_, a), (_, b) in zip(_unwrap(model).state_dict().items(),
                                  self.shadow.state_dict().items()):
            if a.dtype.is_floating_point:
                self._src.append(a)
                self._dst.append(b)

    @torch.no_grad()
    def update(self, model):
        src = [v for v in _unwrap(model).state_dict().values() if v.dtype.is_floating_point]
        torch._foreach_mul_(self._dst, self.decay)
        torch._foreach_add_(self._dst, src, alpha=1.0 - self.decay)


def _unwrap(m):
    return m.module if hasattr(m, "module") else m


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


def make_param_groups(model, lr: float, enc_scale: float, wd: float):
    enc, rest, nodecay = [], [], []
    for n, p in _unwrap(model).named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("encoder.encoders"):
            enc.append(p)
        elif p.ndim <= 1 or n.endswith(".bias"):
            nodecay.append(p)
        else:
            rest.append(p)
    groups = [{"params": rest, "lr": lr, "weight_decay": wd},
              {"params": nodecay, "lr": lr, "weight_decay": 0.0}]
    if enc:
        # A pretrained encoder that gets the head's LR forgets what it knew
        # within an epoch; this is the single most common way fine-tuning a
        # foundation model on a small corpus goes wrong.
        groups.append({"params": enc, "lr": lr * enc_scale, "weight_decay": wd})
    return groups


def cosine_lr(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data/vaani")
    ap.add_argument("--extra-data", nargs="*", default=[],
                    help="extra manifest roots (e.g. scripts/make_synthetic.py output); "
                         "train-only, never validated on")
    ap.add_argument("--out", default="runs/v2")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=0, help="per GPU")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vad-dir", default="")
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

    device, ddp = setup_ddp()
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    out_dir = Path(args.out)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config_run.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    d, t, m = cfg["data"], cfg["train"], cfg["model"]
    le = LabelEncoder(expand_vehicle=bool(d.get("expand_vehicle", True)))
    recs = load_manifest(Path(args.data) / "manifest.jsonl")
    tr_recs, va_recs = split_manifest(recs, fold=args.fold,
                                      n_folds=int(d.get("n_folds", 5)),
                                      seed=int(cfg["seed"]))
    for extra in args.extra_data:
        er = load_manifest(Path(extra) / "manifest.jsonl")
        for r in er:
            r["_root"] = str(extra)
        # Extras are train-only. Validating on synthetic audio would measure how
        # well the model reads our own splicing, not the competition's task.
        tr_recs += er
        log("[data] + %d extra clips from %s" % (len(er), extra))
    log("[data] %d train / %d val  (fold %d of %d)"
        % (len(tr_recs), len(va_recs), args.fold, int(d.get("n_folds", 5))))
    log("[data] held-out states: %s"
        % sorted({r.get("state", "") for r in va_recs})[:10])

    vad = args.vad_dir or d.get("vad_dir") or None
    ds_kw = dict(root=args.data, le=le, clip_len=float(d["clip_len"]),
                 sr=int(d["sr"]), fps=float(d["fps"]), vad_dir=vad)
    tr_ds = VaaniSpanDataset(tr_recs, train=True, augment=True, **ds_kw)
    va_ds = VaaniSpanDataset(va_recs, train=False, augment=False, **ds_kw)

    bs = int(t["batch_size"])
    nw = min(int(t.get("num_workers", 4)), os.cpu_count() or 2)
    sampler = TierBatchSampler(tr_recs, bs, t.get("tier_quotas"), seed=cfg["seed"])
    tr_ld = DataLoader(tr_ds, batch_sampler=sampler, num_workers=nw,
                       collate_fn=collate, pin_memory=True,
                       persistent_workers=nw > 0,
                       prefetch_factor=int(t.get("prefetch_factor", 4)) if nw else None)
    va_ld = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=nw,
                       collate_fn=collate, pin_memory=True,
                       persistent_workers=nw > 0)

    enc = build_encoder(m, ckpt_dir=m.get("beats_dir", "checkpoints"))
    model = build_model(cfg, len(le), enc).to(device)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], find_unused_parameters=True)

    crit = SpanLoss(cfg, len(le), int(m.get("n_bins", 16)))
    lr = float(t["lr"])
    opt = torch.optim.AdamW(
        make_param_groups(model, lr, float(t.get("encoder_lr_scale", 0.05)),
                          float(t["weight_decay"])))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(t.get("amp", True))
                                  and device.type == "cuda")
    ema = EMA(model, float(t.get("ema_decay", 0.999)))

    epochs = int(t["epochs"])
    total_steps = epochs * len(sampler)
    warmup = int(t.get("warmup_steps", 500))
    step = 0
    history, best, start_epoch = [], -1.0, 1
    unfreeze_at = int(t.get("unfreeze_epoch", 6))

    # --- resume -----------------------------------------------------------
    # Kaggle caps a GPU session at ~12 h, which a 60-epoch run over 154 h of
    # audio will not fit inside. `--resume auto` picks up `<out>/state.pt` if it
    # exists, so a run spans as many sessions as it needs. The optimiser, the
    # scaler, the EMA shadow and the step counter all have to come back: without
    # the step counter the cosine schedule restarts and the LR jumps back up,
    # which quietly undoes the previous session's progress.
    resume_path = None
    if args.resume:
        p = (out_dir / "state.pt") if args.resume == "auto" else Path(args.resume)
        resume_path = p if p.exists() else None
        if args.resume != "auto" and resume_path is None:
            raise SystemExit("--resume %s does not exist" % args.resume)
    if resume_path is not None:
        st = torch.load(str(resume_path), map_location="cpu", weights_only=False)
        if int(st.get("unfroze_at_epoch", 0)):
            # Rebuild the trainable set *before* loading the optimiser, or the
            # param groups will not line up with the saved state.
            _unwrap(model).encoder.unfreeze_last(int(t.get("unfreeze_blocks", 0)))
            opt = torch.optim.AdamW(
                make_param_groups(model, lr, float(t.get("encoder_lr_scale", 0.05)),
                                  float(t["weight_decay"])))
        _unwrap(model).load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        # In-place copy, so the EMA's cached tensor references stay valid.
        ema.shadow.load_state_dict(st["ema"])
        step, best = int(st["step"]), float(st["best"])
        history = st.get("history", [])
        start_epoch = int(st["epoch"]) + 1
        log("[resume] %s -> epoch %d, step %d, best %.4f"
            % (resume_path, start_epoch, step, best))

    refs = build_refs(va_ld, float(d["fps"])) if is_main() else {}

    for epoch in range(start_epoch, epochs + 1):
        if epoch == unfreeze_at + 1 and int(t.get("unfreeze_blocks", 0)) > 0:
            got = _unwrap(model).encoder.unfreeze_last(int(t["unfreeze_blocks"]))
            log("[train] unfroze top blocks per encoder: %s" % got)
            opt = torch.optim.AdamW(
                make_param_groups(model, lr, float(t.get("encoder_lr_scale", 0.05)),
                                  float(t["weight_decay"])))

        model.train()
        acc, nb, t0 = {}, 0, time.time()
        for batch in tr_ld:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)
            f = cosine_lr(step, total_steps, warmup)
            for g, base in zip(opt.param_groups,
                               [lr, lr, lr * float(t.get("encoder_lr_scale", 0.05))]):
                g["lr"] = base * f

            with torch.autocast(device_type=device.type,
                                enabled=bool(t.get("amp", True)) and device.type == "cuda"):
                out = model(batch["wav"], batch["frame_valid"])
                loss, logs = crit(out, batch)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]],
                float(t.get("grad_clip", 5.0)))
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            step += 1
            nb += 1
            for k, v in logs.items():
                acc[k] = acc.get(k, 0.0) + v          # stays on device

        if is_main():
            msg = "  ".join("%s %.4f" % (k, float(v) / max(nb, 1))
                            for k, v in acc.items())
            log("[epoch %d/%d] %s  (%.1fs)" % (epoch, epochs, msg, time.time() - t0))

        if is_main() and epoch % int(t.get("eval_every", 1)) == 0:
            for name, net in (("ema", ema.shadow.to(device)), ("raw", _unwrap(model))):
                cands = run_loader(net, va_ld, device, float(d["fps"]),
                                   cfg.get("postproc"), amp=bool(t.get("amp", True)))
                preds = {u: candidates_to_events(c, cfg.get("postproc"))
                         for u, c in cands.items()}
                r = evaluate(preds, refs)
                r.update(epoch=epoch, which=name)
                history.append(r)
                log("   [%s] F1 %.4f  Dice %.4f  score %.4f  (tp %d fp %d fn %d)"
                    % (name, r["event_f1"], r["segment_dice"], r["score"],
                       r["tp"], r["fp"], r["fn"]))
                if r["score"] > best:
                    best = r["score"]
                    torch.save({"model": net.state_dict(), "cfg": cfg,
                                "classes": le.classes, "score": best,
                                "which": name, "epoch": epoch},
                               out_dir / "best.pt")
            (out_dir / "history.json").write_text(json.dumps(history, indent=1),
                                                  encoding="utf-8")
            torch.save({"model": _unwrap(model).state_dict(), "cfg": cfg,
                        "classes": le.classes, "epoch": epoch}, out_dir / "last.pt")
            # Full resume state, written every epoch. Written to a temp name and
            # renamed so a session killed mid-write leaves the previous state
            # intact rather than a truncated file.
            tmp = out_dir / "state.pt.tmp"
            torch.save({"model": _unwrap(model).state_dict(),
                        "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                        "ema": ema.shadow.state_dict(), "step": step, "best": best,
                        "epoch": epoch, "history": history, "cfg": cfg,
                        "unfroze_at_epoch": epoch > unfreeze_at
                        and int(t.get("unfreeze_blocks", 0)) > 0},
                       tmp)
            tmp.replace(out_dir / "state.pt")

    log("[done] best val score %.4f -> %s" % (best, out_dir / "best.pt"))
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
