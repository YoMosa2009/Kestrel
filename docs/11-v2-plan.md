# 11 — Kestrel-Nano v2: MTP, then distillation

*Written 2026-09-16, while v0.2 (Phase D) is still training. This is the plan for
the run AFTER v0.2 — not a change to anything currently running.*

**Constraint: local only** (docs/08 §0b). Every number here is priced in days on
the RTX 3060 at the measured 4,100 tok/s, never in dollars.

## In plain terms

**v0.2 will be undertrained, and training longer can't fix it.** It will have seen
~19 tokens per parameter; the best small models see thousands. Closing that gap
locally would take months, and the dataset runs out of fresh material around 20B
tokens anyway.

**So v2 doesn't train longer. It learns more from each token.** Two ways, in order:

1. **Turn on multi-token prediction.** The model currently learns to predict the
   next word. With MTP it also predicts the word after that, which gets more
   learning out of the same data - roughly 20-50% more. The code already exists and
   has never been switched on. One catch: switching it on breaks the memory trick
   that lets training fit in 12 GB, so that gets fixed first. Then run it both ways
   and keep it only if it actually wins.

2. **Distillation - the big one.** Instead of learning only from raw text,
   Kestrel-Nano learns from a bigger model's *opinions*: not just "the next word is
   X", but "X is most likely, Y is plausible, Z is wrong". That is far richer
   information per token, and it is the main reason the best tiny models are as good
   as they are - potentially 2-10x more efficient.
   - Teacher: **Qwen2.5-Coder-1.5B** - ~10x bigger than Kestrel, good at code, small
     enough for this card.
   - **Record its opinions once, in advance** (~4 days, ~32 GB of disk) rather than
     running it alongside training, which would make every run ~4x slower forever.

**Image understanding comes after both**, so it is built on the better model.

**One line:** v0.2 proves the architecture works; v2 makes it smarter without months
of training, by learning from a bigger model instead of only from raw text.

---

## The thesis

v0.2 will finish at ~19 tokens/parameter. SmolLM2-135M saw ~15,000. We cannot close
that with wall-clock — 20B tokens is already 48 local days and the corpus hits its
~4x repeat ceiling there. So v2's goal is **not more tokens. It is more signal per
token.**

Two levers, in this order:

| # | lever | expected | status |
|---|---|---|---|
| 1 | **Multi-token prediction** | ~1.2-1.5x data efficiency + speculative decoding later | **already implemented, never run** |
| 2 | **Knowledge distillation** from a local quantized teacher | 2-10x data efficiency | new work |

Ordering is deliberate: MTP is nearly free and settles in two matched runs, so it
should be resolved *before* distillation changes the loss function underneath it.

---

## 1. Multi-token prediction — switch it on and ablate it

Pillar 4 of the original design (MASTER-PLAN §3). It is **built, wired, and has
never been executed** — there are zero references to `mtp` anywhere in `scripts/`,
`train.py` or the docs.

```python
# config.py
mtp: bool = False             # multi-token-prediction aux head (ablation A6)
mtp_weight: float = 0.2

# model.py:445 — fully implemented
if self.cfg.mtp and targets.size(1) > 2:
    h2 = self.mtp_norm(self.mtp_proj(x[:, :-2]))
    loss = loss + self.cfg.mtp_weight * F.cross_entropy(...)
```

**Why it should help:** predicting t+2 as well as t+1 gives denser supervision per
token — the model is forced to represent more than the immediate next symbol.
DeepSeek-V3 used it for exactly this reason. It also yields **speculative decoding
for free at inference**, which MASTER-PLAN already parks as a v2 item.

**The blocker to measure first.** `model.py:418` disables the chunked-CE path when
MTP is on:

```python
chunked = (targets is not None and self.cfg.loss_chunks > 1 and not self.cfg.mtp)
```

Chunked CE is what keeps the 49k-vocab logits tensor from materialising at once —
the binding VRAM constraint on this card. So enabling MTP today means a full
`(B*T, 49152)` logits tensor **plus a second one for the MTP head**. At batch 24 x
seq 1024 that is ~4.8 GB per head in fp32. It will OOM.

**Therefore the first task is not the ablation, it is extending
`chunked_cross_entropy` to cover the MTP head too.** The masked-loss work already
done for SFT is the template: the function now takes an optional mask and reduces
by its total, so adding a second target stream is a contained change.

**The ablation** (pre-registered, per the project's "evidence or ablation" rule):
two matched runs from the same v0.2 checkpoint, same tokens, same schedule, only
`--mtp` differing. Keep it iff it beats the control on `val_mean` at matched
compute — MTP costs ~10-15% throughput for the extra head, so a 1.05x quality win
is a *loss*.

---

## 2. Distillation — the only lever big enough to matter

Train the student against a larger teacher's output **distribution** rather than
one-hot next tokens. Each token then carries information about everything the
teacher considered plausible, not just what was correct. This is the most proven
technique for small models and is how the SmolLM and Gemma-small lines get their
quality.

### Teacher throughput on this card (forward-only, 2*N*T FLOPs)

| teacher | bf16 | 4-bit |
|---|---:|---:|
| Qwen2.5-0.5B | ~7,300 tok/s | ~4,500 tok/s |
| **Qwen2.5-Coder-1.5B** | **~2,433 tok/s** | **~1,500 tok/s** |
| Qwen2.5-3B | ~1,217 tok/s | ~750 tok/s |
| Qwen2.5-7B | ~521 tok/s | ~321 tok/s |

**Qwen2.5-Coder-1.5B is the right teacher**: ~10x the student, domain-matched to
Kestrel's coding/CLI profile, and fast enough to be practical. A 7B teacher is
8-13x slower than the student trains — it would dominate the entire budget.

### Two ways to run it, and why B wins

**Option A — on-the-fly** (teacher forward every step, alongside the student):

| teacher | combined rate | 3B tokens |
|---|---:|---:|
| 1.5B | 1,098 tok/s (3.7x slower) | **31.6 days** |
| 3B | 634 tok/s (6.5x slower) | 54.8 days |

**Option B — precompute top-k logits once, then train at full speed:**

| subset | top-k | one-time precompute | storage |
|---|---:|---:|---:|
| 0.3B tokens | 16 | 2.3 days | 19 GB |
| 0.5B tokens | 16 | 3.9 days | 32 GB |
| 0.5B tokens | 32 | 3.9 days | 64 GB |
| 1.0B tokens | 16 | 7.7 days | 64 GB |

(~76 GB free on S:, so 0.5B/top-16 at 32 GB is the comfortable point.)

**Option B is strictly better here.** The teacher pass is paid once and reused
across every epoch and every ablation, the student then trains at its full 4,100
tok/s, and the precompute can run while nothing else needs the GPU. Option A pays
the teacher cost on every token of every run forever.

Storage is dominated by k. Top-16 keeps the head of the distribution, which is
where nearly all the distillation signal lives; k=32 doubles the disk for
diminishing benefit.

### The loss

Mixed, not pure:

```
L = alpha * KL(student || teacher_topk)  +  (1 - alpha) * CE(student, hard_labels)
```

Pure distillation caps the student at the teacher; the hard-label term keeps it
anchored to ground truth. `alpha ~ 0.5` is the usual starting point and is itself
worth an ablation.

Only a subset carries teacher logits, so the trainer must handle mixed batches —
distilled tokens use KL, the rest fall back to CE. **The SFT loss-mask machinery
already does exactly this shape of thing** (per-token weighting, reduce by the
weight total), so it is an extension rather than a rewrite.

### What to distil

Not uniformly. Spend the 0.5B-token teacher budget where the teacher is strongest
and Kestrel is weakest — code, CLI, tool, instruct — rather than on web text,
where a 1.5B teacher has little to teach and Kestrel is already weakest for
reasons distillation cannot fix.

---

## 3. Deferred, not planned

**VLM (screenshot/terminal reading).** A frozen SigLIP encoder plus a trained
projector would add image understanding for ~93M extra params (~160 MB total at
Q4), and the NoPE GLA blocks handle long image-token sequences well. The honest
ceiling is that visual quality is bounded by the *language* model, so the credible
positioning is "reads a stack trace from a screenshot", not general vision.

**Deliberately after distillation**, so the VLM inherits a better base for the same
compute. Parked until items 1 and 2 have landed.

---

## Order of work

1. Finish v0.2 (Phase D -> 4k context extension -> SFT -> Roost validation)
2. Extend `chunked_cross_entropy` to cover the MTP head — the VRAM blocker
3. MTP ablation: two matched runs, keep only on evidence at matched compute
4. Precompute Qwen2.5-Coder-1.5B top-16 logits over a ~0.5B-token curated subset
5. Distillation run with the mixed KL+CE loss, `alpha` ablated
6. Re-baseline against v0.2 with `scripts/eval_generalization.py` at matched `--seq`
7. VLM, if and when items 3-5 have paid off

Every estimate here is a projection from measured local throughput, not a
benchmark. The precompute rates in particular assume the teacher hits the same
~4.5-7.3 TFLOP/s the student does, which is unverified for a quantized model and
should be measured before committing 2-4 days to it.
