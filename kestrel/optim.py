"""Optimizers for Kestrel (docs/05 §2).

Muon for the 2-D weight matrices (Newton-Schulz orthogonalized momentum — pure
matmuls, Pascal-safe, FP32), AdamW for everything else (embeddings, norms/gains,
gates, and the PKM value tables). Muon stores only a momentum buffer, halving
optimizer VRAM vs Adam — part of why Nano's full training fits in 8 GB.

References: Keller Jordan's modded-nanoGPT Muon; Moonshot "Muon is Scalable".
Single-device implementation (no distributed all-gather); everything is a matmul
so it runs on sm_61 in eager FP32.
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz iteration approximating the orthogonal factor of G
    (i.e. U V^T of its SVD). All ops are matmuls -> Pascal-safe. Runs in FP32."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.t()
    for _ in range(steps):
        A = X @ X.t()
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.t()
    return X


class Muon(torch.optim.Optimizer):
    """Momentum + Newton-Schulz orthogonalization for 2-D parameters only.

    The update is orthogonalized momentum, rescaled by sqrt(max(1, out/in)) so
    its RMS is roughly shape-independent (lets one LR serve matrices of different
    aspect ratios). LR is tuned at K-S and transferred up (docs/04 §4)."""

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                assert g.ndim == 2, "Muon only handles 2-D params; route others to AdamW"
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                upd = g.add(buf, alpha=mom) if group["nesterov"] else buf
                upd = zeropower_via_newtonschulz5(upd, steps=group["ns_steps"])
                if wd != 0.0:
                    p.mul_(1 - lr * wd)
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(upd, alpha=-lr * scale)
        return loss


def build_optimizers(model: nn.Module, muon_lr: float = 0.02, adamw_lr: float = 3e-3,
                     weight_decay: float = 0.01, adam_betas=(0.9, 0.95)):
    """Split params into a Muon group (2-D non-embedding matrices) and an AdamW
    group (embeddings, norms/gains/scalars, gates, PKM values, 1-D params).

    Returns (muon, adamw). Embeddings go to AdamW because orthogonalization is
    meaningless for lookup tables; weight decay is applied to embeddings only."""
    muon_params: List[nn.Parameter] = []
    adam_decay: List[nn.Parameter] = []
    adam_nodecay: List[nn.Parameter] = []

    # PKM value tables are nn.Embedding but are plastic memory -> no weight decay,
    # never Muon. The tied token embedding IS weight-decayed (docs/05 §2).
    pkm_value_ids = set()
    for m in model.modules():
        if type(m).__name__ == "ProductKeyMemory":
            pkm_value_ids.add(id(m.values.weight))

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in pkm_value_ids:
            adam_nodecay.append(p)            # PKM value slots (plastic tissue)
        elif isinstance(getattr(model, "embed", None), nn.Embedding) and p is model.embed.weight:
            adam_decay.append(p)              # tied token embedding (wd on)
        elif p.ndim == 2:
            muon_params.append(p)             # 2-D linear weight matrices -> Muon
        else:
            adam_nodecay.append(p)            # 1-D (norms/gates/scalars) + 3-D depthwise conv

    muon = Muon(muon_params, lr=muon_lr, weight_decay=0.0)
    adamw = torch.optim.AdamW(
        [
            {"params": adam_decay, "weight_decay": weight_decay},
            {"params": adam_nodecay, "weight_decay": 0.0},
        ],
        lr=adamw_lr, betas=adam_betas, eps=1e-8,
    )
    return muon, adamw
