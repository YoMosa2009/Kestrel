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
2. ~~**Phase B — optimize the GLA scan**~~ **DONE 2026-08-21 — premise disproved.** Profiling showed the model is *matmul-bound* (~30% MFU), not scan-bound: the scan is only 5.2% of GPU time. Adopted `chunk_size` 64→256 (~1.12× end-to-end); rejected decay-folding as numerically unsafe (NaN). See docs/08 §3. The token budget did **not** get 2–4× cheaper.
3. **Phase C — scale + rebalance data:** ~5–15B unique tokens, with **more math/reasoning (FineMath) and code-with-execution** specifically to attack the two hardest failures (arithmetic, code logic).
4. **Phase D — the real over-trained run:** WSD-resume this checkpoint toward **20–50B tokens**. This single lever fixes facts, reasoning, repetition, and code-logic *together* (form→content). Fold in the loop-sampling fix here.
5. **Phase E — SFT + Roost:** instruction-tune (answers not continuations) + build the continual-learning loop.

**The one-line diagnosis for future me:** *v0.1 learned form; it needs tokens for content. The whole
improvement story is "spend tokens on the right data (C) → over-train (D)." Phase B tried to make
those tokens cheaper and largely could not — the model was already efficient.*

---

## Appendix A — Cloud-run probe telemetry (the evidence for the findings above)

*Transcribed from the pod's `train.log` before the pod was terminated. The raw log file was
not pulled off the pod (only `ckpt.pt` was), so this table is the surviving record. Probes ran
every 500 steps; steps 8500/9000/9500 were scrolled out of the captured output. Values are
rounded to the precision that supports the claims — the raw log carried ~16 digits.*

| step | val_loss | R=1 | R=2 | R=3 | R1−R2 gap | pkm_gate |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 3.4244 | 3.4251 | 3.3849 | 3.3920 | +0.0402 | 0.31540 |
| 1000 | 3.0318 | 3.0056 | 2.9796 | 2.9849 | +0.0260 | 0.86570 |
| 1500 | 2.8078 | 2.7879 | 2.7618 | 2.7655 | +0.0261 | 0.98390 |
| 2000 | 2.8284 | 2.7323 | 2.7095 | 2.7117 | +0.0228 | 0.99810 |
| 2500 | 2.6331 | 2.5706 | 2.5490 | 2.5530 | +0.0216 | 0.99970 |
| 3000 | 2.5883 | 2.6093 | 2.5880 | 2.5911 | +0.0213 | 0.99993 |
| 3500 | 2.5506 | 2.5749 | 2.5537 | 2.5565 | +0.0212 | 0.99970 |
| 4000 | 2.6260 | 2.7051 | 2.6848 | 2.6877 | +0.0203 | 0.99998 |
| 4500 | 2.4540 | 2.6170 | 2.5962 | 2.5983 | +0.0208 | 0.99998 |
| 5000 | 2.4348 | 2.5563 | 2.5366 | 2.5400 | +0.0197 | 0.99999 |
| 5500 | 2.4760 | 2.4492 | 2.4301 | 2.4326 | +0.0191 | 0.99999 |
| 6000 | 2.4831 | 2.4523 | 2.4329 | 2.4347 | +0.0194 | 0.99999 |
| 6500 | 2.4207 | 2.4084 | 2.3887 | 2.3911 | +0.0197 | 0.99999 |
| 7000 | 2.3959 | 2.3886 | 2.3676 | 2.3705 | +0.0210 | 1.00000 |
| 7500 | 2.4836 | 2.5792 | 2.5591 | 2.5608 | +0.0201 | 1.00000 |
| 8000 | 2.3934 | 2.4473 | 2.4273 | 2.4296 | +0.0200 | 1.00000 |
| 10000 | 2.3104 | 2.3395 | 2.3199 | 2.3207 | +0.0196 | 1.00000 |

**What this table shows, in one read:**
- `val_loss` **3.4244 → 2.3104**, still descending at the end — the model never hit its capacity
  ceiling, which is the single strongest argument that over-training (Phase D) will pay off.
- `pkm_gate` **0.32 → ~1.0 by step ~1,500** and pinned there: the product-key memory is
  load-bearing. (Saturation proves it is *used*, not how much it *contributes* — that needs an
  ablation, listed as open work.)
- **R=2 beats R=1 in 17/17 probes** — the looped core is real — but the gap is flat at ~0.02
  from step 2,500 onward, and **R=3 never beats R=2**. The loop is a working mechanism, not yet
  a quality dial. Suspected cause: train-time R-sampling p={.25,.50,.25} over-trains R=2.
- The bumps (steps 4000, 7500) are val-batch noise, not divergence — grad_norm held at 0.07–0.10
  throughout.
