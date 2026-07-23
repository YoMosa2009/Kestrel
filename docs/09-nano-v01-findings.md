# 09 — Kestrel-Nano v0.1: Evaluation Findings & Improvement Plan

*Record created 2026-07-23, immediately after the first trained checkpoint. Quantitative
signals from training probes; qualitative signals from generation (fixed seed 42, temp 0.7,
top-k 40, R=2). This is the reference "what v0.1 actually is" doc.*

## Run summary
- **Kestrel-Nano**, 157M params (125.8M non-emb + 21.2M PKM), **1.311B tokens**, bf16, RTX 3090, ~$21.
- Trained clean: no divergence, grad_norm 0.07–0.10 for thousands of steps, WSD anneal landed.
- **Final val_loss 2.310** (≈ perplexity ~10 on our tokenizer/data — not comparable across tokenizers).

## Quantitative findings (from the probes, step 500 → 10000)
- **val_loss** fell 3.42 → 2.31 and **never plateaued** → the model is nowhere near its capacity ceiling → **over-training will keep paying off** (the key green light for the sub-200M thesis).
- **PKM: validated.** Gate rose 0.05 → ~1.0 by ~step 1,500 and pinned there. The memory layers are load-bearing, not decorative. *(Caveat: gate saturation shows the memory is USED, not how much it CONTRIBUTES — needs an ablation to quantify.)*
- **Looped core: works as a mechanism, not as a quality dial.** R2 < R1 in 16/16 probes (real, ~0.02 loss, ~0.8% relative) but **flat across training** and **R3 never beats R2**. Confirmed behaviorally in generation (R=1/2/4 give lateral changes, not better answers). Likely cause: train-time R-sampling p={.25,.5,.25} over-trains R=2.

## Qualitative findings — the 5-prompt probe
| Prompt type | Verdict | Evidence |
|---|---|---|
| **Prose/facts** | fluent, confidently wrong | "printing press… invented by Jean-Baptiste La Fontaine… in 1695" (real invention, fabricated inventor + dates) |
| **Code** | perfect form, wrong logic | flawless epydoc docstring for `reverse_string`, then `return s.upper()` |
| **Fact list** | knows names, not order/values | planets listed (Mars, Jupiter, Saturn…) but wrong order, garbage numbers, collapses to "Pluto" ×5 |
| **Procedure** | relative strength | clean, plausible markdown recipe (flour/baking powder/sugar/oats/honey…) — best output, least repetition |
| **Reasoning/math** | fails | "3 apples, eat 1?" → "One apple per apple." repeated. No arithmetic. |

### The through-line: **it learned FORM, not CONTENT.**
Nano reliably reproduces the *shape* of text — docstrings, recipes, bulleted lists, expository
paragraphs, Q/A format — but not the correct facts, logic, or arithmetic to fill them. This is the
textbook signature of an **undertrained** model: structure is high-frequency and learned early;
facts/reasoning/semantics need far more tokens. Consistent secondary failure mode: **repetition
collapse** (planets → Pluto×5, apples → stock phrase), which is the clearest single symptom of too
few tokens.

**Overall:** a real, coherent, honestly-undertrained v0.1 — GPT-2-small-class, code-lean. Impressive
as "I designed and trained this," not yet useful as an assistant. No self-deception: every prediction
made pre-generation held.

## Improvement plan (observed weakness → fix)

| Observed weakness | Root cause | Fix | Phase |
|---|---|---|---|
| Wrong facts; PKM holds little | undertrained; memory slots unfilled | **more tokens** + more diverse knowledge data | D |
| No arithmetic/reasoning | undertrained; too little math; loop not helping | more tokens + **boost FineMath / reasoning data** + fix loop | D |
| Code form ✓ logic ✗ | undertrained on code semantics | more code + **code-with-execution traces** | C/D |
| Repetition collapse | undertraining hallmark | more tokens (primary) + **repetition penalty at inference** (cheap patch now) | D + now |
| Continues, doesn't answer | base model, no post-training | **SFT** on instruction/chat data | E |
| Loop ≠ quality lever | R-sampling over-weights R=2 | flatter/annealed R-distribution or **Ouro early-exit gate** | D setup |
| `�` in output | byte-level partial multibyte tokens | decode fix (buffer bytes) | quick |

### Priority order
1. **Now, cheap:** add a **repetition penalty + byte-safe decode** to `generate.py` — immediate demo-quality bump, no training.
2. **Phase B — optimize the GLA scan** (bandwidth-bound, ~8k tok/s). The enabler: makes the token budget below ~2–4× cheaper. *Do before spending on compute.*
3. **Phase C — scale + rebalance data:** ~5–15B unique tokens, with **more math/reasoning (FineMath) and code-with-execution** specifically to attack the two hardest failures (arithmetic, code logic).
4. **Phase D — the real over-trained run:** WSD-resume this checkpoint toward **20–50B tokens**. This single lever fixes facts, reasoning, repetition, and code-logic *together* (form→content). Fold in the loop-sampling fix here.
5. **Phase E — SFT + Roost:** instruction-tune (answers not continuations) + build the continual-learning loop.

**The one-line diagnosis for future me:** *v0.1 learned form; it needs tokens for content. The whole
improvement story is "afford more tokens (Phase B) → spend them on the right data (C) → over-train (D)."*
