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
| **partial checkpointing** (`grad_checkpoint_every` 2/3/4) | — | best case 2,237 tok/s (+3%) at 2x the VRAM | **NOT WORTH IT** locally |

**Partial-checkpointing test (2026-08-21).** Added `cfg.grad_checkpoint_every` (checkpoint only
every Nth block) to trade the 4.3 GB of spare VRAM against checkpointing's recompute tax:

| config | tok/s | peak VRAM |
|---|---|---|
| full ckpt, batch 8 (baseline) | 2,170 | 3.20 GB |
| full ckpt, batch 12 | 2,163 | 4.20 GB |
| full ckpt, batch 16 | 2,172 | 5.31 GB |
| every=2, batch 2 | 2,117 | 4.30 GB |
| every=3, batch 2 | **2,237 (+3%)** | 6.09 GB |
| every=2/3/4 at batch 4+, or no ckpt at all | OOM | >8 GB |

**Throughput is flat at ~2,150–2,240 tok/s across every batch size and every checkpointing
strategy.** That is the signature of a *compute-saturated* GPU: the 1080 is doing all it can, and
no memory-for-recompute trade buys anything. Best case was +3% for ~2x the VRAM — rejected.
The knob is kept (default `every=1` = unchanged behaviour) because a 24 GB pod may be able to use
it at a useful batch size, where it is worth re-measuring.

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

## 3b. HARDWARE CHANGED — RTX 3060 (measured 2026-08-27)

**The lab GPU is no longer the GTX 1080. It is now an RTX 3060 (12 GB, Ampere sm_86).**
Every throughput number above (and in docs/03, docs/10 §5) was measured on Pascal FP32 and is
superseded for local work. Three Pascal restrictions are gone: **bf16 + tensor cores**,
**FlashAttention-backed SDPA**, and (in principle) Triton. VRAM 8 GB -> 12 GB.

| config | tok/s | peak VRAM | vs 1080 |
|---|---:|---:|---:|
| fp32 + ckpt, b8 (the old 1080 config) | 2,530 | 3.20 GB | 1.17x |
| bf16 + ckpt, b8 | 4,119 | 2.71 GB | 1.90x |
| bf16 + ckpt, b16 | 4,221 | 4.52 GB | 1.95x |
| **bf16 + ckpt, b24** | **4,227** | 6.32 GB | **1.95x** ADOPT |
| bf16 without ckpt, b8 / b16 | OOM | >12 GB | — |
| bf16 + `torch.compile`, b16 | FAIL | — | no Triton then; `triton-windows` exists now, but compile graph-broke on the GLA scan when tried on a 4090 *with* Triton, so the blocker is the architecture, not the OS |

**Read carefully: the win is bf16, not the card.** In FP32 the 3060 is only 1.17x a 1080 —
it has *fewer* CUDA cores. The 1.95x comes almost entirely from **bf16 tensor cores**, which
Pascal could not use at all. Throughput plateaus at ~4,220 tok/s from batch 16 upward, so the
GPU is compute-saturated again (same conclusion as Phase B, new ceiling).

Two things still do not work locally: **gradient checkpointing cannot be disabled** (OOM even at
batch 8 — 12 GB is not enough, so the ~25-30% recompute tax stands), and **`torch.compile` fails
for lack of Triton on Windows**. (Even with Triton it likely would not help: on the 4090 it
graph-broke on the GLA scan and OOM'd compiling the `[B,H,n,C,C]` decay tensors.)

### Local training time, revised (bf16, batch 24)

| new tokens | cumulative seen | tok/param | RTX 3060 | (old 1080) |
|---|---|---|---|---|
| 1B | 2.3B | 15 | **2.8 days** | 5.5 days |
| 2B | 3.3B | 21 | **5.6 days** | 11.0 days |
| **3B** | **4.3B** | **27** | **8.5 days** | 16.5 days |
| 5B | 6.3B | 40 | **14.1 days** | 27.5 days |

**Phase D is now viable locally.** The recommended 3B run drops from ~16.5 days to **~8.5 days**,
and even the full 5B is ~2 weeks rather than a month. Local training also costs roughly half the
electricity per token. Renting remains faster in wall-clock, but the case for paying is much weaker.

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
4. [x] ~~Phase B — optimize the GLA scan~~ **DONE 2026-08-21** (§3). The premise was wrong:
   the model is matmul-dominated, not scan-dominated, so no 2–4× was available. Adopted
   `chunk_size` 64→256 (~1.12× end-to-end); rejected decay-folding as numerically unsafe.
5. [x] ~~Phase C — scale the dataset~~ **DONE 2026-08-22** — 4.86B tokens in `data_5b/`,
   30 per-domain shards, plus curriculum staging and the generalization eval suite
   ([docs/10](10-phase-c-plan.md)).
6. [ ] **← YOU ARE HERE · Phase D — RUNNING.** Target step 18,590 = **3.00B cumulative**
   (19 tok/param), ~3.9 days at the measured 4,100 tok/s, anneal over steps 17,561-18,590.
   Shortened from 4.31B at the loss-curve knee (§0b, `scripts/stopping_point.py`): past 3B
   the marginal gain per day halves from ~0.08 to ~0.05 val_mean. Tool curriculum weight
   raised 0.7% -> 4.0% mid-run. Procedure: [NANO_RUNBOOK.md §4](../NANO_RUNBOOK.md).
7. [x] **Roost built** - `kestrel/roost/`: episodic store, session serialize/restore, and the
   eval-gated consolidation job (50/50 new+replay, TF-IDF slot selection, recall >= 60% and
   regression <= 2% or roll back). Containment is enforced structurally after every optimizer
   step, because AdamW's weight decay updates every row regardless of gradient - gradient
   masking alone silently writes to all 32k slots. Mechanics unit-tested; teach-me-today
   awaits the finished model.
8. [x] **SFT set + loss masking built** - 23,289 examples / 16.2M tokens in `data_sft/`, built
   for a 4k context in the exact template pretraining already saw. Loss is scored on Assistant
   tokens only (43% of SFT tokens are prompt). The scored fraction depends on sequence length -
   11.6% at seq 512 vs 44.9% at seq 4096 - so this data MUST be trained at 4096.
9. [ ] **Context extension to 4k** - must run BEFORE the SFT, or the SFT is partly overwritten.
   Only 5 of 20 blocks carry RoPE; the 15 GLA blocks are NoPE and length-agnostic, so ~75% of
   the model is already 4k-ready. ~0.2-0.5B tokens, ~0.6-1 day locally.
10. [ ] Phase E/F - run the SFT, validate teach-me-today, quantize, benchmark, Axiom.
11. [ ] **v2 - see [docs/11](11-v2-plan.md).** v0.2 lands at ~19 tok/param against SmolLM2's
    ~15,000, and wall-clock cannot close that (20B tokens = 48 local days and the corpus hits
    its ~4x repeat ceiling). So v2 buys signal per token instead: MTP first (built, never run,
    but `model.py:418` disables chunked CE when it is on, so the 49k-vocab logits must be
    chunked for the aux head before it will even fit), then distillation from a local
    Qwen2.5-Coder-1.5B with top-k logits PRECOMPUTED once (~3.9 days, 32 GB) rather than run
    on-the-fly (3.7x slower forever). VLM deferred behind both.

### Phase D pre-flight — measured 2026-09-12

- **The GPU is not free at idle.** Ollama pins `gemma4:12b` (8.43 GB, `keep_alive` never
  expires), leaving ~2.3 GB. `ollama stop gemma4:12b` before launching. Note that
  `torch.cuda.mem_get_info()` reported 11.8 GB "free" on WDDM while a 40 MB allocation
  OOM'd — trust `nvidia-smi`.
- **Resume bug, fixed.** `wsd_lr_mult` was keyed off the checkpoint's *absolute* step, so a
  WSD-resume skipped warmup and hit a decayed model with the full plateau LR. The schedule is
  now measured from the resume point, and `--steps` is validated as cumulative.
- **Curriculum position on resume is now explicit** (`--curriculum-start`) instead of falling
  out of `step / args.steps` arithmetic.
- **The corpus is on a 5400-rpm HDD** (S:, ST1000LM024) with a 9.1 GB working set against
  ~5.8 GB of free RAM. Measured ~2.4 s/step (~5%) of serialized read time; `--prefetch N`
  overlaps it with compute.
## 0b. COMPUTE POLICY (local-only imposed 2026-09-14, **LIFTED 2026-09-17**)

**Two machines now, each for what it is good at.** The RTX 3060 remains the
workhorse for long continuous training; a Colab A100 (via the user's Google AI Pro
subscription, ~200 compute units/month ~ **15-16 A100-hours**) handles burst jobs
that 12 GB cannot hold. See §0c.

*Historical note: between 2026-09-14 and 2026-09-17 the plan was strictly local, and
the paragraph below reflects that. Paid GPU RENTAL (RunPod etc.) is still not part
of the plan - the lift covers the Colab subscription only.*

**~~No cloud compute. Every phase runs on the RTX 3060 until the user says otherwise.~~**

This supersedes the cloud lines throughout docs/01, docs/03, docs/07 and §4 below.
Those sections are kept for the reasoning they record, but their cloud budgets and
the "$25 Mini pod" are **not the plan**.

What this changes, concretely:

| item | was | now |
|---|---|---|
| Phase D | 3B local | unchanged - already local |
| the over-trained run | ~20B for ~$255 rented | **20B = 48 days local**, or 10B = 20 days |
| dropping grad-checkpointing | measure on a $2 rented 4090 | **unmeasurable, dropped** - 12 GB OOMs even at batch 4, so the ~25-30% recompute tax is permanent |
| `torch.compile` / Triton | test on a Linux pod | test locally via `triton-windows` if ever, but it graph-broke on the GLA scan when it WAS tried on a 4090 |
| **Kestrel-Mini (~500M)** | the one cloud item (~$25) | **shelved** - see below |
| context extension to 4k | either | local, ~0.6-1 day |
| SFT | either | local, ~2 h |

### The local-only ladder for Nano (at the measured 4,100 tok/s)

| cumulative | tok/param | new tokens | days | corpus epochs |
|---:|---:|---:|---:|---:|
| 4B | 25 | +1.0B | 2.8 | 0.8x |
| 6B | 38 | +3.0B | 8.5 | 1.2x |
| 10B | 64 | +7.0B | 19.8 | 2.1x |
| 15B | 95 | +12.0B | 33.9 | 3.1x |
| 20B | 127 | +17.0B | 48.0 | 4.1x - at the repeat limit |

20B remains reachable locally; it costs seven weeks of wall-clock instead of $255,
and it is the ceiling on the current 4.87B-token corpus. Going further needs a
Phase C extension to ~10-15B unique tokens (free - time, bandwidth and ~22 GB of
disk), not money.

### Kestrel-Mini is shelved, not cancelled

docs/01 ruled out local Mini pretraining on the **GTX 1080**. On the 3060 the
arithmetic is different but still unattractive: compute scales roughly with
parameters, so Mini (482M) runs at ~1,340 tok/s, and a Chinchilla-ish 10B tokens
is **~87 days - about three months** as a single run. VRAM fit at 12 GB is also
unverified. So under a local-only constraint the two-tier Nano+Mini family becomes
**Nano-only**, and Mini waits for either a hardware change or a lifted constraint.

That is not a bad outcome for the thesis: docs/08 §0 already argues the sub-200M
niche is the sharp end of the mission, and Nano is the model that occupies it.

## 0c. Colab A100 — what it is for (2026-09-17)

| | RTX 3060 12 GB | Colab A100 40 GB |
|---|---|---|
| cost | electricity | ~12-13 units/hr of 200/mo = ~15-16 h/month |
| availability | unlimited, uninterrupted | **not guaranteed** - may hand you an L4 or T4, both of which are NO faster than the 3060 for Kestrel |
| session | unlimited | time-limited, disconnects |
| best at | long continuous training | short bursts, and anything 12 GB cannot hold |

**It is not a way to buy more pretraining tokens.** 15-16 A100-hours is only a few
days of local output. Its value is doing what 12 GB *cannot*:

1. **The v2 teacher-logit precompute** (docs/11) - the single biggest win, ~11
   estimated A100-hours, saves ~4 local days, output lands straight in the 5 TB Drive.
2. **Two things this doc previously recorded as permanently unmeasurable**, both of
   which OOM at 12 GB and fit in 40 GB:
   - dropping gradient checkpointing (worth ~25-30% if it holds up)
   - the MTP head's full-vocab logits - which means the MTP **ablation can be run on
     Colab BEFORE doing the chunked-CE engineering** that docs/11 lists as its
     blocker. If MTP does not win, that work is never needed at all.
3. **Measuring, generally.** Every A100 figure in docs/11 is an estimate derived from
   local TFLOP/s. A ~20-minute `scripts/bench_step.py` run replaces them with
   measurements before a month of units is committed.

Backups also live on the same subscription's 5 TB Drive: `I:\My Drive\Kestrel\`
holds the v0.1 and Phase D checkpoints, the SFT set, and the run records.

### Phase D tuning: what was measured, and what did NOT work (2026-09-12)

Full-step benchmark (`scripts/bench_step.py`, nano, batch 24, seq 1024, bf16, grad-ckpt on,
**optimizer included**): **4,163 tok/s**, peak 7.71 GB. The docs/03 and docs/08 §3b figure of
4,227 tok/s excluded the optimizer, so the optimizer costs only ~1.5% — not the ~16% a
batch-2 profile suggests, because it is amortized across `accum`.

**The Ampere profile inverts Phase B's diagnosis.** `scripts/profile_step.py` at batch 24, bf16:

| op class | share of CUDA time | (GTX 1080 FP32, §3) |
|---|---:|---:|
| `aten::mm` | **13.3%** | 43% |
| all matmul kernels (mm + cutlass + ampere gemm) | ~20% | ~43% |
| `aten::mul` | 8.9% | 14% |
| `aten::copy_` | 7.6% | 6.4% |
| elementwise kernel tail | ~15% | — |
| `Muon.step` | 2.2% | — |

bf16 tensor cores cut matmul several-fold while leaving elementwise and reduction work
untouched, so **the model is no longer matmul-bound on this card** — it is bandwidth- and
elementwise-bound. That is the opposite of the Phase B conclusion, which was correct for
Pascal FP32 and does not carry to Ampere.

**But every lever that inversion suggests was measured, and none of them paid:**

| change | result | verdict |
|---|---|---|
| TF32 for fp32 matmuls (Muon's Newton-Schulz) | **1.000x** | free, kept, no gain |
| fused AdamW kernel | **1.000x** | free, kept, no gain |
| `chunk_size` 64 / 128 / 512 vs 256 | 0.87x / 0.98x / 0.93x | **256 is still optimal on Ampere** |
| lean decay mask (cached mask, no `full_like` temporary) | 1.002x | noise — not adopted |
| disable grad-checkpointing | **OOM at batch 4** | confirmed in fresh processes |
| partial checkpointing (`every` 2/3/4) | **OOM at batch 8** | 12 GB is genuinely not enough |
| raise power limit 170 W -> 212 W | **no headroom to reclaim** | see below |
| **`--prefetch 3` (background data reader)** | **+6.8%** | **ADOPTED** |

**The power limit is not the constraint.** Under sustained load the card draws **142-154 W of
its 170 W cap**, holds **1950-1965 MHz**, sits at 69-73 C, and reports
`clocks_throttle_reasons.active = 0x0` — no power, thermal or reliability throttling at any
sample. Raising the cap with `nvidia-smi -pl 212` would reclaim nothing. (Memory does run at
7301 MHz rather than 7501 in the P2 compute state, worth ~2.7% of bandwidth; locking it needs
admin and was not tested.)

**The only win was I/O, not the GPU.** `data_5b` is 9.1 GB of randomly-sampled shards on a
5400-rpm HDD with ~5.8 GB of free RAM, costing ~2.4 s per step serialized ahead of compute.
Measured on real shards at the Phase D config: **4,083 -> 4,359 tok/s** and
**4,413 -> 4,715 tok/s** with `--prefetch 3`. The loader was also changed to do one gather and
one *pinned* H2D copy per domain instead of 48 pageable blocking copies per batch.

**Conclusion: the RTX 3060 is already saturated at ~4,150-4,400 tok/s for this model.** Phase D
at 3B new tokens is **~8 days**, and no further local tuning is available without changing the
model or the precision. The remaining untested lever is `torch.compile`, which needs
`triton-windows` and graph-broke on the GLA scan when it was tried on a 4090.

- **Every tok/s figure in these docs excludes the optimizer step** —
  `scripts/bench_throughput.py` measures only the fwd+bwd core. Use the new
  `scripts/bench_step.py` for a full step, and `scripts/profile_step.py` to re-derive the op
  mix on Ampere+bf16 (the §3 profile above is GTX 1080 FP32 and its conclusions do not
  automatically carry).
