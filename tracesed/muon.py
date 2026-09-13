"""Muon for the freshly initialised head, AdamW for everything else.

Where Muon is used, and where it is not
---------------------------------------
Muon orthogonalises each 2D update (Newton-Schulz on the momentum), so every
singular direction of a weight matrix moves at the same rate. That is a win
for matrices trained *from scratch* - the GRU, projections and convs of our
head. It is not used on:

  * the pretrained encoders: ATST-Frame and BEATs were pre-trained with AdamW,
    and fine-tuning an AdamW-pretrained model with Muon has been reported to
    underperform (Moonshot, "Muon is Scalable for LLM Training", 2025) - the
    update geometry changes under a converged network;
  * biases, norms, and the output projections to logits: not hidden matrices.

Update RMS is matched to AdamW's (0.2 * sqrt(max(m, n)), the Moonlight
recipe), so one learning rate serves both groups and an AdamW-vs-Muon A/B
changes only the geometry, not the effective step size.

GRU weights stack three gates in one tensor (3H x in); each gate is
orthogonalised on its own, since the gates are separate linear maps.
"""
from __future__ import annotations

import torch
from torch.optim import Optimizer


@torch.no_grad()
def zeropower_ns5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Quintic Newton-Schulz orthogonalisation (Keller Jordan's coefficients).

    fp32: the T4 has no bf16 kernels, and these head matrices are small.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    tall = X.size(0) > X.size(1)
    if tall:
        X = X.mT
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.mT if tall else X


class MuonAdamW(Optimizer):
    """One optimiser, two update rules chosen per param group (`use_muon`)."""

    def __init__(self, groups, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01,
                 momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        momentum=momentum, ns_steps=ns_steps, use_muon=False, n_split=1)
        super().__init__(groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            lr, wd = g["lr"], g["weight_decay"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                st = self.state[p]
                if g["use_muon"]:
                    buf = st.get("mom")
                    if buf is None:
                        buf = st["mom"] = torch.zeros_like(grad)
                    buf.mul_(g["momentum"]).add_(grad)
                    upd = grad.add(buf, alpha=g["momentum"])            # Nesterov
                    shape = upd.shape
                    m2 = upd.reshape(shape[0], -1)
                    parts = m2.chunk(g["n_split"], dim=0)
                    outs = []
                    for q in parts:
                        o = zeropower_ns5(q, g["ns_steps"])
                        outs.append(o * (0.2 * max(q.size(0), q.size(1)) ** 0.5))
                    upd = torch.cat(outs, 0).reshape(shape).to(p.dtype)
                    p.mul_(1 - lr * wd).add_(upd, alpha=-lr)
                else:
                    b1, b2 = g["betas"]
                    if "m" not in st:
                        st["m"] = torch.zeros_like(p); st["v"] = torch.zeros_like(p); st["t"] = 0
                    st["t"] += 1
                    st["m"].mul_(b1).add_(grad, alpha=1 - b1)
                    st["v"].mul_(b2).addcmul_(grad, grad, value=1 - b2)
                    bc1 = 1 - b1 ** st["t"]; bc2 = 1 - b2 ** st["t"]
                    denom = (st["v"] / bc2).sqrt_().add_(g["eps"])
                    p.mul_(1 - lr * wd).addcdiv_(st["m"], denom, value=-lr / bc1)
        return None


def head_param_groups(head: torch.nn.Module, lr: float, wd: float, use_muon: bool):
    """Split a from-scratch head into Muon (hidden 2D/3D weights) and AdamW groups."""
    muon, muon_gru, rest = [], [], []
    for name, p in head.named_parameters():
        if not p.requires_grad:
            continue
        hidden = p.ndim >= 2 and not name.startswith(("out", "cls_out")) and "logit" not in name
        if use_muon and hidden:
            (muon_gru if name.split(".")[-1].startswith("weight_") and "gru" in name else muon).append(p)
        else:
            rest.append(p)
    groups = []
    if muon:
        groups.append(dict(params=muon, lr=lr, weight_decay=wd, use_muon=True, n_split=1, name="muon"))
    if muon_gru:
        groups.append(dict(params=muon_gru, lr=lr, weight_decay=wd, use_muon=True, n_split=3, name="muon_gru"))
    groups.append(dict(params=rest, lr=lr, weight_decay=wd if not use_muon else 0.0, name="head_adamw"))
    return groups
