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
    captured: dict[int, list[torch.Tensor]] = {}

    # A PKM inside the looped core runs once PER LOOP. v1 kept only the last call
    # (captured[i] = ...), so the slots read on earlier loops were never selected -
    # and the row restore then froze them every step. Record every call, and run
    # at r_max so every loop the model may take (R=1..r_max) is covered.
    def mk_hook(i):
        def hook(_mod, args):
            captured.setdefault(i, []).append(args[0].detach())
        return hook

    handles = [p.register_forward_pre_hook(mk_hook(i)) for i, p in enumerate(pkms)]
    try:
        was = model.training
        model.eval()
        for x in batches:
            captured.clear()
            model(x.to(device), targets=None, n_loops=model.cfg.r_max)
            for i, p in enumerate(pkms):
                for inp in captured.get(i, []):
                    s = pkm_slots(p, inp)
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


_STOP = {"the", "a", "an", "of", "is", "are", "was", "it", "and", "to", "in"}


def _answer_tokens(text: str) -> set[str]:
    return {w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split()
            if w not in _STOP and (len(w) > 2 or w.isdigit())}


def split_fact(text: str) -> tuple[str, str, str] | None:
    """'The capital of Varnal is Dreymoor.' -> (cue, answer, question):
    ('The capital of Varnal is', 'Dreymoor', 'What is the capital of Varnal?').

    v1 cut every fact in half by word count, which put the cue BEFORE the entity
    name ('The capital of' -> ?). With 50 facts sharing 5 templates the model
    cannot know which entity is meant, so even perfect memorization scored ~0%.
    The cue must contain the subject and stop right before the answer.
    """
    t = text.strip().rstrip(".")
    for sep in (" is ", " are ", " was ", " were ", ": "):
        i = t.rfind(sep)
        if i > 0:
            subject, answer = t[:i], t[i + len(sep):]
            if not _answer_tokens(answer):
                return None
            cue = (subject + sep).rstrip()
            verb = sep.strip(" :") or "is"
            subj = (subject[0].lower() + subject[1:]
                    if subject.startswith(("The ", "A ", "An ")) else subject)
            return cue, answer, f"What {verb} {subj}?"
    return None


def qa_form(text: str) -> str | None:
    """The fact as one chat exchange, in the SFT template the model is used through."""
    parts = split_fact(text)
    if parts is None:
        return None
    _, _, question = parts
    return f"User: {question}\n\nAssistant: {text.strip()}"


def _hit(want: set[str], got_text: str) -> bool:
    return bool(want) and len(want & _answer_tokens(got_text)) / len(want) >= 0.5


@torch.no_grad()
def _answer_nll(model, tokenizer, cue: str, answer: str, device) -> float:
    """Mean NLL of the answer tokens given the cue (teacher-forced). Unlike exact
    recall this moves continuously, so it shows partial learning."""
    c = tokenizer.encode(cue).ids
    a = tokenizer.encode(" " + answer).ids
    x = torch.tensor([c + a], dtype=torch.long, device=device)
    logits, _ = model(x, targets=None)
    lp = F.log_softmax(logits[0, len(c) - 1:-1].float(), dim=-1)
    return float(-lp.gather(1, torch.tensor(a, device=device)[:, None]).mean())


def recall_report(model, tokenizer, texts: list[str], device, max_new: int = 16) -> dict:
    """Three views of whether facts are known:
      decl  - greedy completion of the cue ('The capital of Varnal is') contains the answer
      qa    - the chat question ('User: What is the capital of Varnal?') is answered
      nll   - mean answer-token NLL given the cue (lower = better known)
    """
    was = model.training
    model.eval()
    n = decl = qa = 0
    nlls = []
    for t in texts:
        parts = split_fact(t)
        if parts is None:
            continue
        cue, answer, question = parts
        want = _answer_tokens(answer)
        n += 1
        decl += _hit(want, _greedy(model, tokenizer, cue, device, max_new))
        qa += _hit(want, _greedy(model, tokenizer, f"User: {question}\n\nAssistant:", device, 2 * max_new))
        nlls.append(_answer_nll(model, tokenizer, cue, answer, device))
    if was:
        model.train()
    return {"n": n, "decl": decl / max(1, n), "qa": qa / max(1, n),
            "nll": sum(nlls) / max(1, len(nlls))}


def recall_score(model, tokenizer, store, device, max_new: int = 16) -> float:
    """Fraction of taught facts recalled from a cue that ends right before the answer."""
    return recall_report(model, tokenizer, [it.text for it in store.items], device, max_new)["decl"]


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
    # v2 diagnostics
    qa_before: float = 0.0
    qa_after: float = 0.0
    nll_before: float = 0.0
    nll_after: float = 0.0
    new_loss_curve: list = None          # loss on the new material, every log_every steps
    config: dict = None

    def __str__(self):
        v = "ACCEPTED" if self.accepted else "ROLLED BACK"
        curve = self.new_loss_curve or []
        return (f"[roost] {v}: {self.reason}\n"
                f"  recall     {self.recall_before:.1%} -> {self.recall_after:.1%}  (cue ends before the answer)\n"
                f"  chat Q&A   {self.qa_before:.1%} -> {self.qa_after:.1%}\n"
                f"  answer NLL {self.nll_before:.3f} -> {self.nll_after:.3f}\n"
                f"  new-material loss {' -> '.join(f'{l:.3f}' for l in curve[:1] + curve[-1:])}\n"
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


def _answer_batch(tokenizer, texts, seq: int, device):
    """Like _text_batch, plus a loss mask that scores only the ANSWER of each fact.

    Scoring every token spends most of the update on predicting the entity name
    ('The capital of ___'), which 50 facts sharing 5 templates make unpredictable
    no matter how much is learned. The answer is the part that has to be stored.
    Texts that are not '<subject> is <answer>' statements are scored in full."""
    ids: list[int] = []
    mask: list[int] = []
    for t in texts:
        parts = split_fact(t)
        if parts is None:
            e = tokenizer.encode(t).ids
            ids += e + [0]
            mask += [1] * (len(e) + 1)
            continue
        cue, answer, _ = parts
        tail = t.strip()[len(cue):]                     # ' Dreymoor.' (keeps the closing period)
        c = tokenizer.encode(cue).ids
        a = tokenizer.encode(tail).ids
        ids += c + a + [0]
        mask += [0] * len(c) + [1] * (len(a) + 1)
    if not ids:
        raise ValueError("no new material to consolidate")
    while len(ids) < seq + 1:
        ids, mask = ids + ids, mask + mask
    rows = max(1, len(ids) // (seq + 1))
    x = torch.tensor([ids[i * (seq + 1): (i + 1) * (seq + 1)] for i in range(rows)],
                     dtype=torch.long, device=device)
    m = torch.tensor([mask[i * (seq + 1): (i + 1) * (seq + 1)] for i in range(rows)],
                     dtype=torch.float32, device=device)
    return x, m


def consolidate(model, tokenizer, store, *, replay_dir: str = "data_5b",
                device=None, seq: int = 256, steps: int = 200, lr: float = 1e-3,
                top_t: int = 2048, recall_gate: float = 0.60,
                regression_gate: float = 2.0, eval_iters: int = 8,
                new_frac: float = 0.5, qa: bool = True, answer_only: bool = False,
                log_every: int = 25,
                log=print) -> ConsolidationResult:
    """Run one night of consolidation. Returns the gate decision.

    v2: `qa` adds each fact as a chat exchange (the form it will be asked in),
    `new_frac` sets the share of steps spent on new material vs replay, the
    new-material loss is logged, and the recall gate cues up to the answer."""
    config = dict(steps=steps, lr=lr, top_t=top_t, new_frac=new_frac, qa=qa, seq=seq,
                  answer_only=answer_only)
    t0 = time.time()
    # loaders branch on device.type, so a CLI string like "cuda" must become a torch.device
    device = torch.device(device) if device is not None else next(model.parameters()).device
    pkms = _pkm_modules(model)
    if not pkms:
        raise RuntimeError("model has no ProductKeyMemory sites; nothing to consolidate")

    # --- material -----------------------------------------------------------
    texts = [f for it in store.items for f in it.all_forms()]
    if qa:
        texts += [q for it in store.items if (q := qa_form(it.text))]
    facts = [it.text for it in store.items]
    if answer_only:
        new_x, new_m = _answer_batch(tokenizer, texts, seq, device)
    else:
        new_x, new_m = _text_batch(tokenizer, texts, seq, device), None
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
    rb = recall_report(model, tokenizer, facts, device)
    rec_before = rb["decl"]
    log(f"[roost] before: recall {rb['decl']:.1%}  qa {rb['qa']:.1%}  nll {rb['nll']:.3f}  "
        f"val {val_before:.4f}")

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
    curve = []
    acc = 0.0
    for step in range(steps):
        # Bresenham-style interleave: exactly round(steps * new_frac) new-material steps
        is_new = int((step + 1) * new_frac) > int(step * new_frac)
        lm = None
        if is_new:
            x, y = new_x[:, :-1], new_x[:, 1:]
            lm = new_m[:, 1:] if new_m is not None else None
        else:
            x, y = replay.get_batch()
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y, loss_mask=lm)
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
        if is_new:
            acc = float(loss.detach())
        if step % log_every == 0 or step == steps - 1:
            curve.append(round(acc, 4))
            log(f"[roost]   step {step:3d}/{steps}  new-material loss {acc:.4f}")
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
    ra = recall_report(model, tokenizer, facts, device)
    rec_after = ra["decl"]
    reg = 100.0 * (val_after - val_before) / max(1e-9, val_before)
    log(f"[roost] after:  recall {ra['decl']:.1%}  qa {ra['qa']:.1%}  nll {ra['nll']:.3f}  "
        f"val {val_after:.4f} ({reg:+.2f}%)")

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
        slots_updated=n_sel, steps=steps, seconds=time.time() - t0,
        qa_before=rb["qa"], qa_after=ra["qa"], nll_before=rb["nll"], nll_after=ra["nll"],
        new_loss_curve=curve, config=config)
