# 08 — The "Real Model": Roadmap to a Properly-Trained Kestrel-Nano

*Living doc. Started 2026-07-21, after the Nano v0.1 cloud run. This is the forward
plan; docs 01–07 are the original design phase. Referenced by both the user (MalxTech)
and Claude across sessions.*

## 0. The decision & thesis

**Decision (2026-07-21):** the "real model" is a **properly-trained ~156M (sub-200M) model** —
*not* a scale-up to 500M/1B. Own the tiny niche rather than compete in a crowded larger one.

**Why this is the sharp call:**
- **Sub-200M is underserved.** Most effort is on 1B–7B+. The truly-tiny space (SmolLM2-135M/360M,
  Gemma-3-270M, MobileLLM) has few serious players.
- **Universal hardware.** ~156M @ Q4 ≈ ~100 MB → phones, CPU-only laptops, decade-old GPUs, offline.
  The addressable "can't run anything bigger" audience is huge.
- **Over-training is cheapest at tiny scale.** The sub-200M niche is won by *tokens*, not just
  architecture: SmolLM2-135M saw ~2T tokens (~15,000 tok/param). Over-training a 156M model to
  50–100B tokens costs a fraction of doing so at 1B, and is impossible for a solo builder at 7B.
  **This is the one size where an individual can afford to make a genuinely competitive model.**
- **The differentiators are the moat, and they matter most here.** Loop dial (adaptive compute),
  persistent session state, and Roost (overnight learning) *compensate for size* — and no other
  sub-200M model has them. Position: *"the best tiny model that also remembers you and thinks
  harder on demand."*

**The reframe:** "properly-trained" ≠ Chinchilla (~3.1B tokens). It means **over-trained: ~20–100B+
tokens.** That sets the target and the budget below.

## 1. Where we are — Nano v0.1

- 156M params (125.8M non-emb + 21.2M PKM), **1.31B tokens**, bf16, RTX 3090, ~$23. Trained
  end-to-end via `kestrel/train.py` (Muon+AdamW, WSD, grad-checkpoint, chunked CE).
- Status: **undertrained** (~40% of Chinchilla), a **base model** (no chat tuning), **Roost not built
  yet**. It's a real end-to-end *proof*, not the product.
- **Key finding that shapes everything below:** the GLA chunked scan is **memory-bandwidth-bound**
  (on the 4090, batch 16 == batch 48 tok/s; the RTX 4000 Ada's 2.8× lower bandwidth gave 2.8× lower
  speed). Result: only ~8k tok/s and ~1.3B tokens per $29. `torch.compile` does NOT help (graph-breaks
  on the scan loop + OOMs compiling the `[B,H,n,C,C]` decay tensors).

## 2. The phased plan

| Phase | What | Cost | Effort |
|---|---|---|---|
| **A — See what we have** | `generate()` inference script; honest eval — *did the pillars fire?* (loop-gain R1>R2>R3, session-state carry, PKM usage, coding/language quality) | ~$0 | ~days |
| **B — Fix training economics** | **Optimize the GLA scan** (§3). The enabler for everything below. | ~$0 | ~1–3 weeks |
| **C — Scale the data** | Build ~5–15B unique high-quality tokens (add Stack-Edu via blob-fetch, FineMath, more code/tool traces to the existing pipeline). | ~$0 | ~days (prep) |
| **D — The real run** | Over-train to **~20–50B tokens** with staged curriculum + quality anneal. WSD-resume from v0.1. | see §4 | ~1–3 weeks wall-clock |
| **E — Differentiate** | Light **SFT** (conversational) + build **Roost** (overnight consolidation, eval-gated) + session-state serialize/restore. The moat. | ~$0–25 | ~weeks |
| **F — Ship universal** | Quantize (GGUF / the runtime), benchmark vs SmolLM2-135M/360M & Gemma-270M, deploy to phone/CPU/old-GPU, integrate into **Axiom** (TorchSharp, see [[kestrel-axiom-integration]]). | ~$0 | ~weeks |

## 3. The GLA-scan optimization (Phase B — the highest-leverage move)

**Problem:** `gla_chunked_scan` (kestrel/model.py) materializes per-chunk `[B,H,n,C,C]` decay
matrices in fp32 and runs a sequential Python loop over chunks → heavy memory traffic + kernel-launch
overhead → bandwidth-bound, ~8k tok/s, and it caps batch size (compile OOMs on those tensors).

**Directions to try (roughly in order):**
1. **Leaner math** — avoid materializing the full `C×C` decay matrix; fold the decay into the k/q
   scaling so the intra-chunk term is a plain masked matmul. Cut fp32 temporaries.
2. **Chunk-size / head tuning** — trade loop iterations vs per-chunk memory (bandwidth-bound favors
   fewer, larger ops up to a memory ceiling).
3. **A fused kernel for the cloud** — a Triton (or CUDA) gated-linear-attention scan for the pod
   (Ampere+), kept behind the same interface; local stays pure-PyTorch/Pascal-safe. FlashLinearAttention
   has reference kernels to adapt.
4. **Recompute vs store** — with a leaner scan, non-checkpointed larger batches may fit and run faster.

**Target:** ~2–4× throughput (≈8k → ~20–30k tok/s on a 3090-class card). Every future training dollar
then goes 2–4× further. Validate with `scripts/bench_throughput.py` + the smoke test (scan must stay
numerically exact vs the naive recurrence).

## 4. Budget reality (honest)

Cost of the real over-trained run, at ~$0.51/hr (3090-class), **before vs after** the Phase-B optimization:

| Tokens seen | ~tok/param | now (~8k tok/s) | after opt (~24k tok/s) | verdict |
|---|---|---|---|---|
| 3B (Chinchilla) | 20 | ~$53 | ~$18 | baseline "solid" |
| **20B** | **~130** | ~$350 | **~$120** | **strong sub-200M model** |
| 50B | ~320 | ~$885 | ~$295 | genuinely competitive |
| 100B+ | 640+ | ~$1,770 | ~$590 | leader-class (SmolLM territory is 2T) |

**Takeaway:** Phase B (~free engineering) turns the real model from a ~$350–900 spend into a
**~$120–300 project** for a genuinely competitive over-trained 156M. That's the whole argument for
doing the scan optimization *before* spending more on compute. Data is ~free (open corpora); the real
costs are the training run + prep/engineering time.

## 5. Success criteria & positioning

- **Matched-compute:** beat a vanilla dense-transformer twin at the same size/tokens (the original S1).
- **Absolute:** competitive with SmolLM2-135M/360M & Gemma-3-270M on coding/tool/language evals *given
  our token budget* — clearly labeled unmatched where their budget is larger.
- **Capabilities no frozen tiny model has:** working loop dial (D3), session resume (S4/D2),
  teach-me-today (S3) via Roost.
- **The one-line position:** *the most capable model this size at coding/tool-use/language — that also
  runs anywhere, remembers you, and thinks harder on demand.*

## 6. Immediate next steps (checklist)

1. [x] ~~Nano v0.1 run completes~~ **DONE 2026-07-23** — 10,000 steps, 1.311B tokens, final val_loss 2.310, ~$21. `ckpt.pt` (1.4 GB) is home and resumable; pod terminated.
2. [x] ~~Write `generate()` inference script~~ **DONE** — `kestrel/generate.py` (`python -m kestrel.generate --interactive`).
3. [x] ~~Honest eval~~ **DONE** — full record in `docs/09-nano-v01-findings.md`. PKM validated; loop works as mechanism not quality-dial; model learned form, not content (undertrained).
4. [ ] **← YOU ARE HERE · Phase B — optimize the GLA scan** (§3); re-benchmark. Bandwidth-bound at ~8k tok/s; a 2–4× win turns the ~$350 over-training run into ~$120. Do before spending more on compute.
5. [ ] Phase C — scale the dataset toward ~5–15B unique tokens.
6. [ ] Phase D — the real over-trained run (~20–50B tokens), WSD-resumed.
7. [ ] Phase E/F — SFT, Roost, quantize, benchmark, Axiom.
