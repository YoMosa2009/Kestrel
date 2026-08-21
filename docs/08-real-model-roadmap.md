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

## 3. Phase B — GLA-scan optimization: RESULTS (measured 2026-08-21, GTX 1080)

**Outcome: the original premise was wrong, and the 2–4x target is not available here.**
Profiling the full model (`torch.profiler`, Nano, batch 8, seq 1024, R=2) shows it is already
**matmul-dominated**, not scan-dominated:

| op | share of CUDA time | note |
|---|---|---|
| `aten::mm` | **43%** | dense matmuls — the useful FLOPs |
| `aten::mul` | 14% | 3,298 elementwise calls |
| `aten::copy_` | 6.4% | padding/reshape + grad-ckpt recompute |
| `PowBackward` (RMSNorm) | 5.7% | |
| **`aten::bmm` (the GLA scan)** | **5.2%** | **the thing we set out to optimize** |

At ~2,150 tok/s the model runs at **~30% MFU** of the 1080's measured 7.4 TFLOP/s fp32 — a
healthy eager-mode number. Amdahl's law therefore caps any scan-only win: a 2.5x on the
isolated scan is ~1.12x end-to-end.

**What was tried (all validated against the naive recurrence):**

| Variant | Isolated scan | End-to-end | Verdict |
|---|---|---|---|
| v0 baseline, chunk 64 | 51.3 ms | 1,925 tok/s | — |
| v1 lean temporaries (cached mask, no `full_like`) | 46.5 ms (1.10x) | ~0 | micro-opt only; no gain at large chunk |
| **v0 @ chunk 256** | **25.5 ms (2.0x)** | **2,151 tok/s (1.12x)** | **ADOPTED** — safe, exact math |
| v2 decay folded into q/k | 20.7 ms (2.5x) | — | **REJECTED — numerically unsafe** |
| fused `F.rms_norm` | — | 2,157 tok/s (~0) | no gain |
| disable grad-checkpointing | — | OOM at batch 4 | untestable on 8 GB |

**Why v2 was rejected (important):** folding gives `A_ij = (q_i e^{L_i-L_0})·(k_j e^{L_0-L_j})`,
whose k-side exponent is **>= 0** and overflows fp32. Measured: `max|k~|` = 1.4e28 at chunk 64 with
default gates, and **`inf`/`NaN` at chunk 256, or at chunk 64 once gates get more negative**. The
original log-space form (all exponents <= 0 by construction) exists precisely to prevent this.
Adopting v2 for its speed would have NaN'd an expensive training run.

**Adopted:** `chunk_size` 64 -> **256** (`kestrel/config.py`). Safe, exact same math, ~1.12x
end-to-end, +0.16 GB, relative error ~1e-5 (bf16 eps is 7.8e-3). Benchmark harness kept at
`scripts/bench_scan.py` (correctness-checks every variant before timing).

**The real remaining lever is on the cloud, not here.** The v0.1 run used
`--grad-checkpoint` at batch 16 and peaked at only ~7 GB of the 3090's 24 GB — **~17 GB sat
unused**. Gradient checkpointing costs a full forward recompute (typically ~25-30%), so
**turning it off (or raising the batch) on a 24 GB card is likely the biggest single win left**,
and it costs nothing but a ~$1 sanity run to measure. It could not be tested locally: the 8 GB
1080 OOMs without checkpointing even at batch 4.

**Revised expectation:** do NOT budget on a 2-4x. Assume ~1.1x from chunk tuning, plus a
possible ~1.3x on the pod from dropping grad-checkpointing = **~1.4x realistic**, not 3x. The
budget table below is therefore presented at both the measured rate and an optimistic rate.

## 4. Budget reality (honest)

Cost of the real over-trained run, at ~$0.51/hr (3090-class), **before vs after** the Phase-B optimization:

| Tokens seen | ~tok/param | measured (~8k tok/s) | realistic post-B (~11k tok/s) | verdict |
|---|---|---|---|---|
| 3B (Chinchilla) | 20 | ~$53 | ~$39 | baseline "solid" |
| **20B** | **~130** | ~$350 | **~$255** | **strong sub-200M model** |
| 50B | ~320 | ~$885 | ~$645 | genuinely competitive |
| 100B+ | 640+ | ~$1,770 | ~$1,290 | leader-class (SmolLM territory is 2T) |

*(Revised 2026-08-21 after Phase B measurement. The earlier "~24k tok/s / $120" column assumed a
2-4x scan win that does not exist — the model is matmul-bound, not scan-bound. The ~11k figure
assumes chunk-256 plus dropping grad-checkpointing on the pod, the latter still unverified.)*

**Takeaway (revised):** Phase B did not deliver the hoped-for 3x, because the bottleneck was
misdiagnosed — the model is already matmul-dominated at ~30% MFU. A coherent 20B-token Nano is a
**~$255–350 project**, not ~$120. The cheap win left is dropping grad-checkpointing on the pod
(verify with a ~$1 run). Data is ~free (open corpora); the real
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
