"""E2 probe / full run: DDP training of TraceModel, scored on the held-out val fold.

torchrun --nproc_per_node 2 -m tracesed.train --corpus /kaggle/input/.../vaani \
    --val-meta /kaggle/input/.../validationMetadata.json --steps 6000 --opt muon
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from src.data.labels import LabelEncoder
from src.evaluation.metrics import evaluate
from tracesed.data import FPS, SR, TraceStream, ValBank, _read, collate
from tracesed.model import TraceModel, trace_losses
from tracesed.muon import MuonAdamW, head_param_groups


def log(*a):
    if int(os.environ.get("RANK", "0")) == 0:
        print(*a, flush=True)


@torch.no_grad()
def infer_clips(model, clips, dev, win=8.0, hop=4.0, bs=24):
    """Full-length posteriors via 8 s windows, 4 s hop, triangular stitching."""
    model.eval()
    W, H, TW = int(win * SR), int(hop * SR), int(win * FPS)
    wins = []
    audio = {}
    for c in clips:
        y = _read(c["path"]); audio[c["uid"]] = y
        off = 0
        while True:
            wins.append((c["uid"], off, min(W, len(y) - off)))
            if off + W >= len(y): break
            off = min(off + H, len(y) - W)
    acc = {}
    for i in range(0, len(wins), bs):
        ch = wins[i:i + bs]
        wav = torch.zeros(len(ch), W); val = torch.zeros(len(ch), TW)
        for j, (u, off, nv) in enumerate(ch):
            wav[j, :nv] = torch.from_numpy(audio[u][off:off + nv]); val[j, :max(1, math.ceil(nv / SR * FPS))] = 1
        with torch.autocast("cuda", dtype=torch.float16):
            o = model(wav.to(dev, non_blocking=True), val.to(dev))
        h = dict(pres=o["pres"].sigmoid().half().cpu().numpy(), bnd=o["bnd"].sigmoid().half().cpu().numpy(),
                 ext=o["ext"].half().cpu().numpy(), count=o["count"].softmax(-1).cpu().numpy())
        for j, (u, off, nv) in enumerate(ch):
            acc.setdefault(u, []).append((off, nv, {k: v[j] for k, v in h.items()}))
    res = {}
    for c in clips:
        u = c["uid"]; n = math.ceil(len(audio[u]) / SR * FPS); parts = acc[u]
        out = {}
        for key in ("pres", "bnd", "ext"):
            A = np.zeros((n,) + parts[0][2][key].shape[1:], np.float32); ws = np.zeros(n, np.float32)
            for off, nv, h in parts:
                a = int(round(off / SR * FPS)); m = min(math.ceil(nv / SR * FPS), n - a)
                t = np.arange(m, dtype=np.float32)
                w = np.minimum(t + 1, m - t) if len(parts) > 1 else np.ones(m, np.float32)
                A[a:a + m] += h[key][:m].astype(np.float32) * w[:, None]; ws[a:a + m] += w
            out[key] = (A / np.maximum(ws, 1e-6)[:, None]).astype(np.float16)
        out["count"] = np.mean([h["count"] for _, _, h in parts], 0)
        out["dur"] = len(audio[u]) / SR; out["syn"] = c["syn"]
        res[u] = out
    model.train()
    return res


def freeze_unused(model, b, cfg) -> int:
    """Freeze trainable tensors that receive no gradient.

    The pretrained checkpoints carry tensors off our forward path (heads and
    norms of their pre-training objectives). DDP without unused-parameter
    search - the fast mode - aborts on the first step if any trainable tensor
    goes unreduced, so find them once, on one GPU, before wrapping.
    """
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16, enabled=b["wav"].is_cuda):
        out = model(b["wav"][:2], b["valid"][:2])
    L = trace_losses(out, {k: v[:2] for k, v in b.items()}, cfg)
    L["total"].backward()
    n = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is None:
            p.requires_grad = False
            n += 1
    model.zero_grad(set_to_none=True)
    return n


def quick_decode(p, thr, min_dur=0.1, med=5):
    from scipy.ndimage import median_filter
    x = median_filter(p["pres"][:, -1].astype(np.float32), size=med) > thr
    ev = []; t = 0; n = len(x)
    while t < n:
        if x[t]:
            a = t
            while t < n and x[t]: t += 1
            if (t - a) / FPS >= min_dur: ev.append((a / FPS, min(t / FPS, p["dur"])))
        else:
            t += 1
    return ev


def quick_score(post, clips):
    ref = {c["uid"]: [(s, e) for s, e, _ in c["events"]] for c in clips}
    rows = {}
    for nm, flag in (("NAT", False), ("SYN", True)):
        ks = [c["uid"] for c in clips if c["syn"] == flag]
        best = None
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
            r = evaluate({k: quick_decode(post[k], thr) for k in ks}, {k: ref[k] for k in ks})
            if best is None or r["score"] > best[0]:
                best = (r["score"], thr, r["event_f1"], r["segment_dice"])
        rows[nm] = best
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="")
    ap.add_argument("--val-meta", required=True)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--out", default="/kaggle/working/e2")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--frozen-steps", type=int, default=800)
    ap.add_argument("--bs", type=int, default=12, help="per GPU")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--enc-lr", type=float, default=3e-5)
    ap.add_argument("--llrd", type=float, default=0.85)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--opt", choices=["adamw", "muon"], default="adamw")
    ap.add_argument("--eval-every", type=int, default=1500)
    ap.add_argument("--encoders", default="atst_frame,beats")
    ap.add_argument("--quotas", default='{"gold":0.30,"silver":0.10,"valnat":0.25,"remix":0.25,"valsyn":0.10}')
    ap.add_argument("--w-bnd", type=float, default=1.0)
    ap.add_argument("--w-ext", type=float, default=1.0)
    ap.add_argument("--w-cnt", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-steps-time", type=float, default=0, help="stop after this many hours")
    ap.add_argument("--hold-fold", type=int, default=0, help="validation fold held out for eval; -1 = train on all")
    ap.add_argument("--seed", type=int, default=0, help="data-order and init seed (ensemble members differ here)")
    args = ap.parse_args()

    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        lr_ = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr_)
    dev = torch.device("cuda")
    rank = int(os.environ.get("RANK", "0")); world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(1234 + 101 * args.seed + rank)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    le = LabelEncoder(expand_vehicle=True)
    t0 = time.time()
    # --hold-fold -1: the full run trains on every validation fold and skips evaluation
    bank = ValBank(args.val_meta, le, hold_fold=args.hold_fold)
    log(f"[bank] natural {len(bank.natural)} synth {len(bank.synth)} snippets {len(bank.snippets)} "
        f"hosts {len(bank.hosts)} held-out {len(bank.heldout)} rate {bank.syn_rate:.3f}/s ({time.time()-t0:.0f}s)")
    corpus = []
    if args.corpus:
        for l in open(Path(args.corpus) / "manifest.jsonl"):
            if l.strip():
                r = json.loads(l); r["_le_idx"] = le.idx; corpus.append(r)
    ds = TraceStream(bank, corpus, args.corpus, len(le), json.loads(args.quotas), seed=7 + 1000 * args.seed)
    log(f"[data] gold {len(ds.gold)} silver {len(ds.silver)} quotas {ds.quotas}")
    dl = DataLoader(ds, batch_size=args.bs, num_workers=args.workers, collate_fn=collate, pin_memory=True,
                    prefetch_factor=4, persistent_workers=True)

    model = TraceModel(len(le), args.ckpt_dir, encoders=tuple(args.encoders.split(","))).to(dev)
    cfg = dict(w_bnd=args.w_bnd, w_ext=args.w_ext, w_cnt=args.w_cnt)

    def build_opt(enc_on):
        groups = head_param_groups(model.head, args.lr, args.wd, args.opt == "muon")
        other = [p for n, p in model.named_parameters()
                 if p.requires_grad and not n.startswith("head.") and not n.startswith("encoders.")]
        groups.append(dict(params=other, lr=args.lr, weight_decay=args.wd, name="proj_mel"))
        if enc_on:
            groups += model.encoder_param_groups(args.enc_lr, args.llrd, args.wd)
        for g in groups:
            g["base_lr"] = g["lr"]
        return MuonAdamW(groups, lr=args.lr, weight_decay=args.wd)

    def wrap():
        return DDP(model, device_ids=[dev.index], broadcast_buffers=False, gradient_as_bucket_view=True) if ddp else model

    model.set_encoder_trainable(False)
    net = wrap(); opt = build_opt(False)
    scaler = torch.amp.GradScaler("cuda")
    warm = 300
    step = 0; it = iter(dl); tl = time.time(); agg = {}
    heldout = bank.heldout
    while step < args.steps:
        b = {k: v.to(dev, non_blocking=True) for k, v in next(it).items()}
        if step == args.frozen_steps:
            model.set_encoder_trainable(True)
            n_off = freeze_unused(model, b, cfg)
            net = wrap(); opt = build_opt(True)
            log(f"[stage] step {step}: encoders unfrozen, LLRD {args.llrd}, enc lr {args.enc_lr}, "
                f"{n_off} checkpoint tensors off the forward path frozen")
        # schedule: linear warmup then cosine, per group relative to base_lr
        s_rel = step - (args.frozen_steps if step >= args.frozen_steps else 0)
        span = args.steps - (args.frozen_steps if step >= args.frozen_steps else 0)
        f = min(1.0, (s_rel + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, s_rel / max(1, span)))))
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * max(f, 0.02)
        with torch.autocast("cuda", dtype=torch.float16):
            out = net(b["wav"], b["valid"])
        L = trace_losses(out, b, cfg)
        opt.zero_grad(set_to_none=True)
        scaler.scale(L["total"]).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt); scaler.update()
        for k, v in L.items():
            agg[k] = agg.get(k, 0.0) + float(v)
        step += 1
        if step % 100 == 0:
            dt = (time.time() - tl) / 100; tl = time.time()
            log(f"[{step}] " + " ".join(f"{k} {v/100:.4f}" for k, v in agg.items())
                + f" | {dt:.2f}s/it | mem {torch.cuda.max_memory_allocated()/2**30:.1f}G | {(time.time()-t0)/60:.0f}min")
            agg = {}
        if rank == 0 and (step % args.eval_every == 0 or step == args.steps):
            sd = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
            torch.save(dict(model=sd, args=vars(args), step=step), f"{args.out}/model.pt")   # save BEFORE eval
            try:
                if not heldout:
                    raise StopIteration("no held-out fold (--hold-fold -1)")
                post = infer_clips(model, heldout, dev)
                rows = quick_score(post, heldout)
                log(f"[eval {step}] " + " | ".join(f"{k} {v[0]:.4f} thr {v[1]} F1 {v[2]:.3f} D {v[3]:.3f}" for k, v in rows.items()))
                pickle.dump(dict(post=post, clips=heldout, classes=le.classes, fps=FPS), open(f"{args.out}/post_fold0.pkl", "wb"), protocol=5)
            except Exception as ex:                                 # noqa: BLE001
                log("[eval] failed:", type(ex).__name__, ex)
        if args.max_steps_time and (time.time() - t0) / 3600 > args.max_steps_time:
            log("[time] budget reached"); break
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
