"""The nightly "sleep" job (docs/05 §5, MASTER-PLAN §4.3).

    1. Batch = 50% new declarative material + 50% replay of pretraining shards.
    2. Update ONLY the PKM value slots the new material activates. Keys, dense
       weights, embeddings, norms - everything else is frozen.
    3. Gate on a frozen suite before/after: accept iff new-fact recall >= 60%
       AND old-capability regression <= 2%. Otherwise roll back.
    4. Rollback is a tensor copy of the value tables (~84 MB for Nano).

Why this is safe in a way that ordinary fine-tuning is not: plasticity is
*architecturally confined*. Gradients can only reach a few thousand rows of an
embedding table that the model consults additively through a tanh gate. The dense
pathway that does the reasoning is untouched by construction, not by convention.
So the worst outcome of a bad night is "it didn't learn the fact", never "it
forgot how to code" - and the gate catches the former.

The 50% replay is not optional. Sparse updates reduce forgetting, they do not
eliminate it; replay is what keeps the updated slots consistent with the
distribution the dense weights expect.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F

from kestrel.data import make_domain_val_loaders, find_shards, ShardedTokenLoader
from kestrel.model import ProductKeyMemory


# ------------------------------------------------------------------ slot maths

@torch.no_grad()
def pkm_slots(pkm: ProductKeyMemory, x: torch.Tensor) -> torch.Tensor:
    """Which value slots does `x` address? Mirrors ProductKeyMemory.forward.

    Duplicated here rather than hooked out of the module so the training path
    carries no recording overhead and model.py stays unchanged.
    """
    B, T, _ = x.shape
    q = pkm.q_norm(pkm.wq(x)).view(B * T, 2, -1)
    kh = min(pkm.topk, pkm.n_keys)
    s1 = q[:, 0] @ pkm.keys[0].t()
    s2 = q[:, 1] @ pkm.keys[1].t()
    v1, i1 = s1.topk(kh, dim=-1)
    v2, i2 = s2.topk(kh, dim=-1)
    cand = (v1.unsqueeze(-1) + v2.unsqueeze(-2)).flatten(1)
    _, flat = cand.topk(pkm.topk, dim=-1)
    rows = i1.gather(1, flat // kh)
    cols = i2.gather(1, flat % kh)
    return (rows * pkm.n_keys + cols).reshape(-1)


def _pkm_modules(model) -> list[ProductKeyMemory]:
    return [m for m in model.modules() if isinstance(m, ProductKeyMemory)]


@torch.no_grad()
def _slot_counts(model, batches, device) -> list[torch.Tensor]:
    """Activation histogram over each PKM's slots, for the given token batches."""
    pkms = _pkm_modules(model)
    counts = [torch.zeros(p.values.num_embeddings, device=device) for p in pkms]
    captured: dict[int, torch.Tensor] = {}

    def mk_hook(i):
        def hook(_mod, args):
            captured[i] = args[0].detach()
        return hook

    handles = [p.register_forward_pre_hook(mk_hook(i)) for i, p in enumerate(pkms)]
    try:
        was = model.training
        model.eval()
        for x in batches:
            captured.clear()
            model(x.to(device), targets=None)
            for i, p in enumerate(pkms):
                if i in captured:
                    s = pkm_slots(p, captured[i])
                    counts[i].scatter_add_(0, s, torch.ones_like(s, dtype=counts[i].dtype))
        if was:
            model.train()
    finally:
        for h in handles:
            h.remove()
    return counts


def select_slots(new_counts, replay_counts, top_t: int) -> list[torch.Tensor]:
    """TF-IDF-style selection: frequent in the NEW material, rare in replay.

    Picking the merely most-activated slots would select the generic ones every
    input touches, and writing to those is exactly how you damage general
    capability. Dividing by replay frequency keeps the update on slots that are
    discriminative for the new material.
    """
    out = []
    for nc, rc in zip(new_counts, replay_counts):
        score = nc / (1.0 + rc)
        score[nc == 0] = 0.0                       # never touch unactivated slots
        k = int(min(top_t, int((score > 0).sum().item())))
        out.append(torch.topk(score, k).indices if k > 0
                   else torch.zeros(0, dtype=torch.long, device=score.device))
    return out


# ---------------------------------------------------------------------- evals

@torch.no_grad()
def _greedy(model, tokenizer, prompt: str, device, max_new: int = 24) -> str:
    was = model.training
    model.eval()
    ids = tokenizer.encode(prompt).ids[-512:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    made = []
    for _ in range(max_new):
        logits, _ = model(x[:, -512:], targets=None)
        nxt = int(logits[0, -1].argmax())
        made.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    if was:
        model.train()
    return tokenizer.decode(made)


def _answer_tokens(text: str) -> set[str]:
    return {w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split()
            if len(w) > 2}


def recall_score(model, tokenizer, store, device, max_new: int = 24) -> float:
    """Fraction of taught facts the model reproduces unprompted.

    Cue the model with the first half of the statement and check whether the
    content words of the second half come back. Crude, but it is the same crude
    measure before and after, which is what a gate needs.
    """
    if len(store) == 0:
        return 0.0
    hits = 0
    for it in store.items:
        words = it.text.rstrip(".").split()
        if len(words) < 4:
            continue
        cut = max(2, len(words) // 2)
        cue, tail = " ".join(words[:cut]), " ".join(words[cut:])
        want = _answer_tokens(tail)
        if not want:
            continue
        got = _answer_tokens(_greedy(model, tokenizer, cue, device, max_new))
        if len(want & got) / len(want) >= 0.5:
            hits += 1
    return hits / max(1, len(store))


@torch.no_grad()
def frozen_suite(model, val_loaders, iters: int = 8) -> float:
    """Mean val loss over the per-domain shards - the 'old capabilities' number."""
    was = model.training
    model.eval()
    means = []
    for _, loader in val_loaders.items():
        tot = 0.0
        for _ in range(iters):
            x, y = loader.get_batch()
            _, loss = model(x, targets=y)
            tot += float(loss)
        means.append(tot / iters)
    if was:
        model.train()
    return sum(means) / max(1, len(means))


# ----------------------------------------------------------------- the job

@dataclass
class ConsolidationResult:
    accepted: bool
    reason: str
    recall_before: float
    recall_after: float
    val_before: float
    val_after: float
    regression_pct: float
    slots_updated: int
    steps: int
    seconds: float

    def __str__(self):
        v = "ACCEPTED" if self.accepted else "ROLLED BACK"
        return (f"[roost] {v}: {self.reason}\n"
                f"  recall     {self.recall_before:.1%} -> {self.recall_after:.1%}\n"
                f"  val loss   {self.val_before:.4f} -> {self.val_after:.4f} "
                f"({self.regression_pct:+.2f}%)\n"
                f"  slots      {self.slots_updated} updated over {self.steps} steps "
                f"in {self.seconds:.1f}s")


def _text_batch(tokenizer, texts, seq: int, device) -> torch.Tensor:
    """Pack declarative statements into fixed-length rows, repeating to fill."""
    ids: list[int] = []
    for t in texts:
        ids.extend(tokenizer.encode(t).ids + [0])
    if not ids:
        raise ValueError("no new material to consolidate")
    while len(ids) < seq + 1:
        ids = ids + ids
    rows = max(1, len(ids) // (seq + 1))
    x = torch.tensor([ids[i * (seq + 1): i * (seq + 1) + seq + 1] for i in range(rows)],
                     dtype=torch.long, device=device)
    return x


def consolidate(model, tokenizer, store, *, replay_dir: str = "data_5b",
                device=None, seq: int = 256, steps: int = 200, lr: float = 1e-3,
                top_t: int = 2048, recall_gate: float = 0.60,
                regression_gate: float = 2.0, eval_iters: int = 8,
                log=print) -> ConsolidationResult:
    """Run one night of consolidation. Returns the gate decision."""
    t0 = time.time()
    device = device or next(model.parameters()).device
    pkms = _pkm_modules(model)
    if not pkms:
        raise RuntimeError("model has no ProductKeyMemory sites; nothing to consolidate")

    # --- material -----------------------------------------------------------
    texts = [f for it in store.items for f in it.all_forms()]
    new_x = _text_batch(tokenizer, texts, seq, device)
    log(f"[roost] new material: {len(store)} teachables -> {len(texts)} forms "
        f"-> {new_x.shape[0]} rows")

    shards = find_shards(replay_dir, "train")
    if not shards:
        raise FileNotFoundError(f"no replay shards under {replay_dir}")
    replay = ShardedTokenLoader(shards, seq, max(1, new_x.shape[0]), device, seed=1234)

    # --- which slots? -------------------------------------------------------
    rx, _ = replay.get_batch()
    new_counts = _slot_counts(model, [new_x[:, :-1]], device)
    rep_counts = _slot_counts(model, [rx], device)
    sel = select_slots(new_counts, rep_counts, top_t)
    n_sel = int(sum(s.numel() for s in sel))
    log(f"[roost] selected {n_sel} slots across {len(pkms)} PKM sites "
        f"(of {sum(p.values.num_embeddings for p in pkms):,} total)")
    if n_sel == 0:
        return ConsolidationResult(False, "no slots activated by the new material",
                                   0, 0, 0, 0, 0, 0, 0, time.time() - t0)

    # --- baseline -----------------------------------------------------------
    val_loaders = make_domain_val_loaders(replay_dir, seq, 4, device)
    val_before = frozen_suite(model, val_loaders, eval_iters)
    rec_before = recall_score(model, tokenizer, store, device)
    log(f"[roost] before: recall {rec_before:.1%}  val {val_before:.4f}")

    # --- snapshot for rollback (tens of MB) ---------------------------------
    snapshot = [p.values.weight.detach().clone() for p in pkms]

    # --- freeze everything but the value tables -----------------------------
    saved_rg = {n: p.requires_grad for n, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    for p in pkms:
        p.values.weight.requires_grad_(True)

    masks = []
    for p, s in zip(pkms, sel):
        m = torch.zeros(p.values.num_embeddings, 1, device=device)
        if s.numel():
            m[s] = 1.0
        masks.append(m)

    # weight_decay MUST be 0: AdamW's decay term updates every row regardless of
    # gradient, which would silently write to all ~32k slots and destroy the
    # containment property Roost is built on. The row restore after each step
    # below enforces containment structurally anyway, so this is belt and braces.
    opt = torch.optim.AdamW([p.values.weight for p in pkms], lr=lr, weight_decay=0.0)
    keep = [(~m.bool()).squeeze(1) for m in masks]      # rows that must NOT move

    # --- train: 50% new, 50% replay ----------------------------------------
    was = model.training
    model.train()
    for step in range(steps):
        if step % 2 == 0:
            x, y = new_x[:, :-1], new_x[:, 1:]
        else:
            x, y = replay.get_batch()
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y)
        loss.backward()
        # confine the update to the selected rows - this is the whole safety story
        for p, m in zip(pkms, masks):
            if p.values.weight.grad is not None:
                p.values.weight.grad.mul_(m)
        opt.step()
        # Hard containment: restore every unselected row from the snapshot. Grad
        # masking alone depends on optimizer internals (decay, momentum, fused
        # kernels); this does not. After this line, "only the selected slots
        # changed" is true by construction, not by assumption.
        with torch.no_grad():
            for pk, snap, kp in zip(pkms, snapshot, keep):
                pk.values.weight[kp] = snap[kp]
        if step % 50 == 0:
            log(f"[roost]   step {step:3d}/{steps}  loss {float(loss):.4f}")
    if not was:
        model.eval()

    # --- verify containment actually held ------------------------------------
    with torch.no_grad():
        moved = 0
        for pk, snap, s_idx in zip(pkms, snapshot, sel):
            d = (pk.values.weight - snap).abs().sum(1) > 0
            allowed = torch.zeros_like(d)
            if s_idx.numel():
                allowed[s_idx] = True
            leaked = int((d & ~allowed).sum())
            if leaked:
                raise RuntimeError(
                    f"containment violated: {leaked} unselected slots changed. "
                    "Refusing to continue - this is the one invariant Roost rests on.")
            moved += int(d.sum())
    log(f"[roost] containment verified: {moved} slots moved, all within selection")

    # --- gate ---------------------------------------------------------------
    val_after = frozen_suite(model, val_loaders, eval_iters)
    rec_after = recall_score(model, tokenizer, store, device)
    reg = 100.0 * (val_after - val_before) / max(1e-9, val_before)
    log(f"[roost] after:  recall {rec_after:.1%}  val {val_after:.4f} ({reg:+.2f}%)")

    ok_recall = rec_after >= recall_gate
    ok_reg = reg <= regression_gate
    accepted = ok_recall and ok_reg
    if accepted:
        reason = "recall and regression both within gate"
    elif not ok_recall and not ok_reg:
        reason = f"recall {rec_after:.1%} < {recall_gate:.0%} AND regression {reg:+.2f}% > {regression_gate}%"
    elif not ok_recall:
        reason = f"recall {rec_after:.1%} < {recall_gate:.0%}"
    else:
        reason = f"regression {reg:+.2f}% > {regression_gate}%"

    if not accepted:
        with torch.no_grad():
            for p, snap in zip(pkms, snapshot):
                p.values.weight.copy_(snap)

    # restore the original requires_grad flags
    for n, p in model.named_parameters():
        p.requires_grad_(saved_rg.get(n, True))

    return ConsolidationResult(
        accepted=accepted, reason=reason,
        recall_before=rec_before, recall_after=rec_after,
        val_before=val_before, val_after=val_after, regression_pct=reg,
        slots_updated=n_sel, steps=steps, seconds=time.time() - t0)
