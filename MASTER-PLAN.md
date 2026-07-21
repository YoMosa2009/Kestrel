# The Master Plan
## A new small-LLM architecture & training methodology — designed for independence from the cloud

*Version 1.2 — July 2026. Revised: the flagship is now a **two-tier family** — **Kestrel-Nano (~150M), trained/​run/​continually-learned 100% locally on a GTX 1080 for $0**, plus **Kestrel-Mini (~500M), cloud-pretrained for ≤ $25** then run locally. Caps: **≤ 1 week of local GPU time + ≤ $25 cloud**. Focus domains: **coding, tool use, language**. This document is self-contained; the `docs/` folder holds deeper engineering detail.*

---

## 0. Names

**Chosen (July 2026):** the project is a family of three named things —

| | Name | Meaning |
|---|---|---|
| Architecture | **KESTREL** | A kestrel is a small falcon that hunts by *hovering* — constantly reading the wind and adjusting. Small, light, sharp, adaptive. |
| Training methodology | **FLEDGE** | Fledging is how a young bird learns to fly, in stages — our staged, small-to-large validation ladder. |
| Lifelong-learning system | **ROOST** | The roost is where the bird returns every night — our nightly memory consolidation, literally "sleeping on it." |

*(Runner-up families, kept for the record: Ember/Kindling/Hearth; Lantern/Wick/Glow.)*

---

## 1. The pitch in one paragraph

Kestrel is a small language-model architecture built around one central belief: **a small model should not try to be a frozen miniature of a big model — it should be a different kind of animal that compensates for size with adaptivity.** It combines four efficiency mechanisms that each have strong 2024–2026 evidence but have never been integrated into one stack — a hybrid linear/full attention backbone, a looped "thinking" core, sparse key-value memory banks, and a superword tokenizer — and wires them into a **layered memory system that keeps learning after deployment** without the catastrophic forgetting that makes naive fine-tuning destructive. It is trained with Fledge, a staged, data-centric methodology whose defining discipline is *extreme* frugality: the whole program runs on one GTX 1080, and the larger of the two shipped models is cloud-pretrained for **under $25**. Both models quantize small enough to run on a decade-old GPU, a CPU-only laptop, or a phone, fully offline.

**Two models, one architecture — sized to the two things you asked for (radical locality + pushing tiny-model limits):**

| Model | Params | Trained | Runs / learns | Role |
|---|---|---|---|---|
| **Kestrel-Nano** | ~150M | **100% on the GTX 1080, $0, ~1 week** | locally | the *radically local* hero — train, infer, and continually learn without ever leaving the machine |
| **Kestrel-Mini** | ~500M | RunPod, **≤ $25** (the FP32 token ceiling makes local pretraining at this size a multi-month job — see §4.1) | locally, after | the *stronger assistant* — the one you'd actually code with |

**Why this size band (~150M–500M) and not larger:** it's where truly universal hardware lives — a ~500M model quantizes to ~300 MB, a ~150M one to ~100 MB — and, crucially, it's where the target domains reward you. **Coding, tool use, and language are skill/reasoning abilities, not stored-knowledge abilities.** Fluency saturates early; coding and tool-calling are procedural reasoning that benefits from *compute-per-token (loops)* and *good data* far more than from parameter count; and the broad world-knowledge a small model can't hold is exactly what the memory system (Pillar 3) and Roost offload. So going small isn't a compromise against the mission — for this niche it's the sharp end of it.

---

## 2. The problem we're solving

**Problem 1 — Cloud dependence.** Capable AI currently lives in data centers. People in low-connectivity regions, people who can't afford subscriptions, and people who simply want ownership of their tools are locked out. A capable 1B model that runs on nearly *any* computer made in the last decade changes who gets access to this technology.

**Problem 2 — Small models are starved, not stupid.** The gap between a 1B and a 70B model is mostly (a) stored knowledge and (b) compute per answer. But storing facts in dense weights is brutally expensive — measured at roughly **2 bits per parameter** (Allen-Zhu & Li, ICLR'25) — and standard transformers spend *identical* compute on trivial and hard tokens. Both are design choices, not laws of nature.

**Problem 3 — The frozen-brain problem.** Melanie Mitchell's *AI: A Guide for Thinking Humans* identifies what may be the deepest limitation of current systems: they lack the common sense that humans build through continuous, cumulative experience — and unlike any living intelligence, **they stop learning the day training ends**. Everything a deployed LLM "learns" in conversation evaporates when the context window closes. Naive fixes fail: fine-tuning on new facts erases old capabilities (catastrophic forgetting — in Meta's 2025 measurements, full fine-tuning destroyed **89%** of prior factual performance; even LoRA destroyed 71%).

Kestrel attacks all three: local-first deployment (P1), architectural efficiency (P2), and a memory system engineered specifically so that post-deployment learning is *safe* (P3).

---

## 3. The Kestrel architecture — how it works

Kestrel is a decoder-only language model (so it trains and runs with completely standard PyTorch tooling) with four structural departures from the standard transformer, plus a memory hierarchy that ties them together.

### 3.1 Pillar 1 — A hybrid backbone: 3 linear-attention blocks for every 1 full-attention block

**What:** 75% of Kestrel's layers replace quadratic softmax attention with **gated linear attention**: each attention head maintains a small fixed-size memory matrix that is *updated* as tokens stream past (new information written in, old information decayed away by learned per-token gates), instead of re-reading the entire history at every step. The remaining 25% of layers keep classic full attention for the thing linear attention is weak at: precise, random-access recall of exact tokens.

**Why:** This 3:1 ratio isn't our invention to defend — it's the configuration the industry converged on in 2025 (Qwen3-Next, Kimi Linear, Qwen3.5) because it *beats both pure alternatives*. What we exploit that the big labs don't care about:

1. **Constant memory per token.** No KV-cache explosion → long contexts on tiny VRAM.
2. **The recurrent state is a serializable "mind-state."** A few megabytes capture what the model is currently "holding in mind." Save it when a session ends; restore it days later; the model resumes mid-thought *without replaying the transcript*. On a frozen transformer this feature is impossible — the equivalent state is a giant KV cache. This is the medium-term tier of the memory hierarchy (§3.5), and it falls out of the architecture for free.
3. Heads are initialized with a **spectrum of decay rates** — some heads forget in tens of tokens (syntax), some persist for thousands (topic, task) — mirroring how useful memories have different natural lifetimes.

### 3.2 Pillar 2 — A looped core: the model can think harder without being bigger

**What:** Kestrel's middle block group is *re-run* R times per token (R = 1–4). During training, R is randomized so the model becomes robust at every depth; at inference, R is a dial: R=1 for casual chat, R=3–4 for a hard debugging or math problem.

**Why:** Parameters and computation are different resources, and standard transformers wastefully lock them together. Looping shared layers buys effective depth without new weights: the Ouro project (2025) showed a **1.4B looped model matching 4B standard transformers**, and — crucially — that loops improve knowledge *manipulation* (multi-hop reasoning, composition), not storage. That's exactly the half of intelligence a small model should spend FLOPs on, because storage is handled far more cheaply by Pillar 3. At our sizes this is the main lever for punching above weight: Kestrel-Mini at R=2 computes like a ~1B model and at R=4 like a ~1.8B, while storing like a 500M; Kestrel-Nano at R=4 reaches ~450M-class compute from a ~150M download. And because coding and tool-use *are* manipulation-heavy tasks, this is the lever that matters most for the chosen domains. It's also our answer to Mitchell's observation that humans flexibly allocate *more thought* to harder problems.

### 3.3 Pillar 3 — Product-key memory: knowledge as a lookup, not a computation

**What:** Two layers carry a large bank of key-value memory slots (a few thousand on Nano; ~48k×2 on Mini). For each token, the model retrieves and blends the top-8 relevant slots — an address-and-fetch operation whose cost barely grows with bank size, because slot lookup uses a clever two-half "product key" scheme (Lample et al.; scaled up by Meta in 2024). This is how a sub-500M model gets a knowledge shelf far larger than its dense-parameter budget would suggest.

**Why, part 1 — economics:** Facts in dense weights cost ~2 bits/parameter *and* every parameter participates in every token's computation. Facts in memory slots cost almost no compute at all. A small model with a big cheap memory bank gets to spend its dense parameters on *reasoning*.

**Why, part 2 — this is the organ of lifelong learning:** Meta's follow-up (*Sparse Memory Finetuning*, 2025) showed that if you teach a model new facts by updating **only the specific memory slots those facts activate** — dense weights untouched — you get full-fine-tuning-level learning with **11% forgetting instead of 89%**. Kestrel carries memory banks from day one *so that this is how it learns forever after* (§4.3). Plasticity is confined to memory tissue by construction; a bad learning day can corrupt some slots (reversible — they're snapshotted), but it *cannot* damage the reasoning circuitry.

### 3.4 Pillar 4 — Token efficiency: more meaning per token

**What:** A ~48k-vocabulary tokenizer trained with **superword merges** (SuperBPE, COLM 2025): tokens are allowed to cross word boundaries, so common multi-word units ("as a matter of fact", "for (int i = 0;") become single tokens. Same vocabulary size, **20–30% fewer tokens** for the same text — verified gains, including +8.2% MMLU at 8B scale in the original paper. Optionally, an auxiliary head predicts two tokens ahead (multi-token prediction), densifying the training signal and later enabling ~1.5–2× faster speculative decoding.

**Why:** Every resource that matters to us — context window, training compute, inference speed — is denominated in tokens. A 25% token reduction is equivalent to a 33% bigger context window, a 33% faster model, and 33% more text seen per training dollar, all at once, for the one-time cost of training a better tokenizer. On a $25 budget, "more text per dollar" is not an optimization — it's survival.

### 3.5 The unifying idea — a memory hierarchy modeled on how humans actually retain things

This is the conceptual heart of Kestrel, and the direct answer to Mitchell's frozen-brain critique:

| Timescale | Human analogy | Kestrel mechanism | Plastic? |
|---|---|---|---|
| Seconds | Working memory | Attention over the context window | — |
| A session | Short-term memory | Recurrent mind-state (Pillar 1), serialized between sessions | automatic |
| Days, explicit | Notes / episodic memory | Local vector store of session "teachables," retrieved into context | append-only |
| Permanent, implicit | Consolidated long-term memory | Memory-bank slots (Pillar 3), written by nightly Roost consolidation | **yes — safely** |
| Core competence | Skills / faculties | Dense weights | **frozen by policy** |

A standard LLM has only rows 1 and 5 — nothing in between, which is precisely why it feels like an amnesiac savant. Kestrel fills in the middle.

---

## 4. The Fledge training methodology — how it's trained

Fledge has two defining commitments: **(a) every architectural claim is proven by controlled experiment before scaling, and (b) the science is radically cheap** — the whole validation program fits on one GTX 1080 in about a week of GPU time, and the one cloud model costs less than a takeout dinner.

### 4.1 The scale ladder (the "one-GPU science" protocol)

Train a family of proxies, smallest first, let evidence climb the ladder, then ship two models:

1. **K-S (~20M core parameters, GTX 1080, free)** — the workhorse. One pretraining run = one overnight (~7h). Every key design question becomes an *ablation pair*: Kestrel-variant vs. an identically-budgeted vanilla twin, same data, same tokens, same FLOPs. Pre-registered keep-rule: a mechanism stays only if it wins by ≥1% compute-equivalent loss or unlocks a capability, at ≤5% speed cost.
2. **K-M (~60M, GTX 1080, free)** — the surviving design re-confirmed at 3× scale (does the win *grow* with scale?).
3. **Kestrel-Nano (~150M, GTX 1080, free) — the fully-local hero.** Hyperparameters *transferred* up from the proxies (width-scaled learning-rate rules — the μP family — exist precisely so small-scale tuning transfers). Trained from scratch on the 1080 in FP32; a ~1-week window yields ~1–1.5B tokens (≈7–10 tokens/parameter — undertrained versus commercial models but coherent, especially on a focused coding/tool/language mixture), and WSD scheduling means you can simply leave it running longer for more.
4. **Kestrel-Mini (~500M, RunPod, ≤ $25) — the stronger assistant.** The one size that *cannot* be pretrained locally in reasonable time: the 1080's FP32 throughput caps local training at ~1.5–2B tokens/week regardless of model size, which only properly feeds a ~150M model — a 500M model needs 5–10B+ tokens to be decent, i.e. months locally. So Mini is pretrained on a rented RTX 4090 (bf16 + compiled): ~3–4B tokens in ~60–65 GPU-hours for ~$21–23. Everything *after* pretraining — inference, Roost consolidation, fine-tuning — runs back on the 1080.
5. **Beyond (the upgrade ladder, §6.3)** — the same recipe with more tokens, if/when funding appears. Nothing about the design changes; only the token count.

**Why 150M is the local ceiling — the one hard number.** What decides "trainable locally in a week" isn't parameter count, it's *tokens-per-week*, and the 1080's FP32-only compute (Pascal has no usable low-precision path) fixes that at roughly **1.5–2B tokens/week**. That budget makes a ~150M model (Nano) coherent and a ~500M model (Mini) badly undertrained — which is exactly why Nano is local and Mini is the $25 cloud item. (P0 measures the real throughput before any of this is promised.)

**Pascal note:** the local 1080 forces pure-PyTorch/FP32 discipline (which keeps the architecture universally portable and is how Nano trains), but the Mini cloud run is free to use bf16, FlashAttention-backed kernels, and `torch.compile` — same model code, modern speed. The architecture is deliberately legal in both worlds.

### 4.2 The training recipe (what modern evidence says actually matters)

- **Optimizer: Muon** for the weight matrices (the optimizer behind the nanoGPT speed-run records and, since 2025, frontier models like Kimi K2) + AdamW for embeddings and scalars. Controlled benchmarks show ~1.3–2× data-efficiency over AdamW alone — on a starved token budget this is the single biggest recipe win available. It also halves optimizer memory, which is what lets Nano's full training fit in the 1080's 8 GB.
- **Schedule: warmup-stable-decay (WSD).** The learning rate stays flat for the bulk of training and only decays at the end — so a run can be *paused, extended, or branched* at any checkpoint. Three practical consequences: an interrupted run is a hiccup, not a disaster; Nano can be left training longer than a week for more tokens whenever the PC is free; and if budget ever appears, Mini can be **resumed from its stable-phase checkpoint and simply trained further** — the $25 run is a down-payment, not a dead end.
- **Data: quality-first, openly available, staged.** Curated open corpora (FineWeb-Edu educational web, Stack-Edu code, FineMath, Cosmopedia synthetic textbooks) in evolving mixtures: broad web early → code/math-heavier middle → a final "anneal" on the highest-quality slice. This is the SmolLM playbook — the best-documented small-model recipe in the open. Two evidence-backed liberties: up to ~50% code hurts nothing (and code, with its explicit state and causality, is the cheapest "physics simulator" text there is — part of the common-sense agenda), and repeating good data up to ~4 epochs costs nothing (so a couple billion unique tokens suffice).
- **The Mitchell curriculum:** deliberately over-weight procedural, causal, and physical-commonsense text, and track a dedicated common-sense evaluation slice (PIQA, WinoGrande, CommonsenseQA + perturbed variants to measure *brittleness*, not just accuracy) as a first-class metric — not to claim the barrier of meaning is broken, but to measure whether grounding-oriented data and memory actually move the needle.

### 4.3 Roost — the lifelong-learning protocol (what happens *after* training)

This is the methodology's most novel component: deployment is not the end of learning, it's a phase of it.

**During use:** the mind-state persists across sessions automatically (Pillar 1). Anything worth remembering — facts the user teaches, corrections, preferences, project context — is rewritten by the model into clean declarative statements (several paraphrases each; restatement-before-learning is the key trick from MIT's SEAL work) and appended to a local episodic store, immediately retrievable into context.

**Every night ("sleep"):** a small consolidation job runs on the local machine — for *both* models, since only the memory-slot values need gradients (even Mini's 500M slots fine-tune comfortably in the 1080's 8 GB this way):
1. Build a batch: 50% the new declarative material, 50% replay of original training-distribution data.
2. Update **only the memory slots** the new material activates (Meta's sparse-memory-finetuning procedure). Dense weights are never touched.
3. **Gate the result:** run a frozen evaluation suite before and after. Keep the update only if new-fact recall succeeds AND old capabilities regress ≤2%; otherwise roll back (slot snapshots are tens of megabytes).
4. Weekly: deduplicate and compact the episodic store.

**The safety argument, in one line:** the worst possible outcome of a bad consolidation is "it didn't learn the fact" — never "it forgot how to code" — because the parts of the network that can change are architecturally separated from the parts that reason.

---

## 5. How this is better than existing approaches

### Against each architecture class

| Compared to… | Their weakness for our mission | Kestrel's answer |
|---|---|---|
| **Standard dense transformers** (Llama-class, and every mainstream small model) | KV-cache grows with context (VRAM death on small devices); every token costs identical compute; facts stored at ~2 bits/param in expensive dense weights; **completely frozen after training** | Constant-memory state on 75% of layers; loop dial allocates compute to difficulty; facts offloaded to cheap sparse memory; Roost keeps learning safely |
| **Pure linear-attention / SSM models** (Mamba-2, RWKV-7) | Measurably weak at precise recall and verbatim copying — a real capability tax | Keep 25% full attention: the 2025-industry-validated hybrid keeps recall while inheriting linear-attention economics |
| **Mixture-of-Experts** (the mainstream sparse-capacity play) | All experts must be resident in memory during training — hostile to small VRAM; routing instabilities; capacity gains aimed at throughput, not lifelong learning | Product-key memory delivers sparse capacity with embedding-lookup economics, trains happily on small GPUs, quantizes/offloads gracefully — and doubles as the substrate for continual learning, which MoE experts cannot safely do |
| **Looped models alone** (Ouro, Huginn) | Loops improve reasoning but explicitly *don't* add knowledge storage | Kestrel pairs loops (manipulation) with memory banks (storage) — each pillar covers the other's documented blind spot. This complementarity is the core architectural thesis. |
| **Memory-layer models alone** (Meta's) | Used purely as a pretraining capacity trick; the continual-learning potential was demonstrated separately and never productized | Kestrel makes the memory bank the centerpiece of a full memory *hierarchy* — session state, episodic store, nightly consolidation — an integrated learning system, not a layer type |
| **The current small class** (Llama-3.2-1B, Gemma-3-270M/1B, Qwen3-0.6B, SmolLM2-135M/360M/1.7B, MobileLLM) | Genuinely excellent — but frozen, token-standard, trained on ~1–10T tokens of cloud compute, and none can remember you tomorrow | We do not claim benchmark parity against multi-terabyte-token budgets on ~$25 or one home GPU. We claim: more capability *per FLOP and per parameter* (proven by matched-compute twins), plus capabilities **no frozen model has at any size**: persistent sessions, safe overnight learning, a compute dial |

### The honest novelty claim

Every individual mechanism above has published evidence behind it — deliberately so; a shoestring lab cannot afford to gamble on unproven physics. The contribution is threefold:

1. **The integration.** No published model combines the hybrid backbone + looped core + memory banks + superword tokenization in one design, and no one has wired them into a unified memory hierarchy where the pieces reinforce each other (the recurrent state enables session persistence; the memory banks enable safe consolidation; the loops exploit whatever knowledge the memory holds).
2. **The lifelong-learning protocol.** Roost — sparse-slot consolidation with replay, evaluation gates, and rollback, layered over episodic retrieval and persistent mind-state — is, to our knowledge, the first *complete, safety-gated* continual-learning loop proposed as a standard operating mode for a local model.
3. **The methodology as a result in itself.** Fledge demonstrates that rigorous architecture research — controlled twins, pre-registered decision rules, hyperparameter transfer — plus a *fully-local trained model* fits in **about a week of one consumer GPU's time, at $0**, with a ~$25 cloud step for the larger model. If the ablation numbers come out clean, the report is publishable regardless of whether any single pillar wins, and the whole program is reproducible by anyone with a gaming PC. That, too, serves the mission: democratizing not just AI, but AI *research*.

### What we do NOT claim

- We will not "solve" common sense — Mitchell's barrier of meaning stands. We mitigate (grounding via memory/retrieval, curriculum, adaptation) and *measure honestly* (brittleness tests, not just leaderboard scores).
- Neither shipped model will top the leaderboards of its size class: the incumbents train on 50–5,000× more tokens. **Kestrel-Nano** proves the architecture trains stably and efficiently *end-to-end on one home GPU*; **Kestrel-Mini** is a stronger, still-undertrained assistant whose WSD checkpoint resumes cleanly when more compute appears. The per-FLOP/per-parameter superiority claim rests on the matched-twin experiments, not on the shipped models' absolute scores.
- Every scale-up beyond these two needs money we haven't spent yet — the plan (§6.3) says exactly how much for what, instead of pretending otherwise.
- Absolute coding ability at these sizes is bounded. A ~150–500M model will not solve hard, novel programming problems end-to-end; the honest goal is *the most capable model this size has been at coding/tool-use/language*, using loops + memory as the levers, not parity with cloud giants.

---

## 6. The plan

### 6.1 The compute ledger (two independent budgets: ~1 week of local GPU time, and ≤ $25 cloud)

**Local budget — GTX 1080, $0** (the science + the Nano hero; runs sequentially, so this is the ~1-week line):

| Budget line | GPU-hours |
|---|---|
| P0 benchmarking + P2 trainer shakedown | ~8 h |
| P3 twin gauntlet: 4 × K-S runs (twin, full, −loops, −memory) @ ~300M tokens | ~28 h |
| P4 scale confirm: 1–2 × K-M runs @ ~350M tokens | ~22–44 h |
| P5 Roost prototype (slot-finetuning on a K-S/Nano checkpoint) | ~4 h |
| **P6a — Kestrel-Nano hero: from-scratch local train, ~1–1.5B tokens** | ~50–90 h |
| **Local total** | **~120–170 h ≈ 5–7 days** ✅ |

If the week is tight, Nano and the ablations compete for the same GPU; WSD lets Nano bank whatever tokens the leftover hours allow (and keep going later). The Mini cloud run does **not** touch this budget.

**Cloud budget — RunPod, ≤ $25** (Kestrel-Mini only, independent of the local week):

| Budget line | Where | GPU-hours | Cash |
|---|---|---|---|
| P6b — Kestrel-Mini pretrain, ~3–4B tokens, bf16 + compiled | RunPod RTX 4090 (community, ~$0.35/h — verify at P6) | ~60–65 h | **~$21–23** |
| Reserve (re-runs, data transfer) | RunPod | — | ~$2–4 |
| **Cloud total** | | | **≤ $25** ✅ |

Tokenizer training and data prep are CPU jobs (overnight on the i7, $0). One tokenizer + one data pipeline feed both models.

### 6.2 Phases

| Phase | What happens | Exit criterion |
|---|---|---|
| **P0 — Groundwork** | Environment pinning on the PC (Pascal-compatible PyTorch), measured 1080 throughput; RunPod account sanity check | real tokens/day table |
| **P1 — Tokenizer & data** | Train the superword tokenizer (CPU); verify ≥18% token reduction; pre-tokenize ~2B unique tokens into shards (~4 GB), coding/tool/language-weighted | tokenizer gate passed |
| **P2 — Trainer** | Resumable training loop (Muon + WSD, checkpoint-everything); dual-mode: FP32-eager (local) / bf16-compiled (cloud) | tiny model overfits; K-S shakedown clean |
| **P3 — Twin gauntlet** (the science) | 4 overnight K-S runs: vanilla twin vs full Kestrel vs two single-pillar removals, matched FLOPs, pre-registered keep/kill rules | Kestrel v1 config frozen on evidence |
| **P4 — Scale confirmation** | Twin vs Kestrel at K-M (3× scale) | wins hold or grow with scale |
| **P5 — Roost prototype** | Teach-me-today on a small checkpoint: 50 novel facts overnight, ≥60% recall, ≤2% regression | consolidation gate passes |
| **P6a — Kestrel-Nano** (local hero) | From-scratch FP32 train on the 1080 (~1–1.5B tokens, WSD); full domain eval | a stable, coherent, 100%-local ~150M model |
| **P6b — Kestrel-Mini** (cloud stretch) | ~3–4B tokens on RunPod in one ~60h run; checkpoints pulled hourly; then pull weights home for local inference + Roost | a stronger ~500M model + full training log |
| **P7 — Release & report** | Both models' weights, code, tokenizer, and a technical report with every table (negative results included); upgrade-ladder pitch | published artifacts |

*Sequencing note:* P6a and P6b are independent — Nano trains on the PC while Mini trains on the pod, so they can run the same calendar days. Do P0–P5 once; both models inherit the frozen v1 config.

### 6.3 The upgrade ladder (what more money buys later — same recipe, more tokens)

WSD scheduling means **each tier resumes the previous tier's checkpoint** — money spent is never thrown away:

| Tier | Model | Tokens | Approx. cost | What it is |
|---|---|---|---|---|
| **$0 (this plan)** | Nano ~150M | ~1–1.5B | free, local | proof: a coherent architecture trained end-to-end on one home GPU |
| **~$25 (this plan)** | Mini ~500M | ~3–4B | ~$25 | a stronger, still-undertrained local assistant |
| ~$150 | Mini ~500M | ~15–20B | ~$150 | "Chinchilla-solid": a genuinely usable 500M coding/tool assistant |
| ~$800–1.5k | Mini ~500M | ~100–150B | ~$800–1.5k | competitive-class vs commercial sub-1B models (with distilled/synthetic data leverage) |
| Grant-scale | either | 1T+ | partner/community compute | frontier sub-1B attempt |

**End state (this plan):** two models you fully own — **Nano (~150M, ~100 MB quantized)** trained start-to-finish on your own GPU, and **Mini (~500M, ~300 MB quantized)** for ~$25 — both running at interactive speed on the 1080 (or CPU), both remembering yesterday's conversation, learning what you teach them overnight, and thinking harder when asked. Plus the twin-experiment evidence that the architecture beats a vanilla transformer per FLOP — the ammunition for climbing the ladder with other people's compute.

---

## 7. Key risks, stated plainly

| Risk | Mitigation |
|---|---|
| A pillar fails its ablation | It gets cut and reported — the methodology is designed so negative results still produce a credible report |
| 1080 throughput worse than estimated → Nano gets too few tokens in a week | Measured in P0 *before* any promises; Nano's token count flexes (WSD banks whatever the hours allow) and can extend past a week; worst case Nano shrinks toward ~100M |
| Local week over-subscribed (ablations + Nano compete for the one GPU) | Ablations are cheap (~60–80 h); Nano takes the remainder and can run on into a second week — nothing is lost, WSD resumes |
| Shipped models disappoint observers expecting leaderboard parity | Framing pre-committed: Nano is an *end-to-end-local proof*, Mini a *down-payment* (WSD-resumable); the efficiency claim lives in the matched twins |
| Cloud run (Mini) fails mid-way (pod preemption, bugs) | Hourly checkpoint pulls; WSD makes resumption lossless; trainer shaken down locally first; community-vs-secure pod decided at P6 |
| Consolidation gains don't transfer from Meta's 7B-scale evidence to small models | P5 tests exactly this, for free, before either shipped model depends on it |
| Scope creep | Pre-registered decision rules; a formal "v2 parking lot" (test-time-trained memory, byte-level tokenization, RL self-edits, a later 1B tier) keeps ambition out of v1 |

---

## 8. Supporting documents

| Doc | Contents |
|---|---|
| [docs/01-goals-and-constraints.md](docs/01-goals-and-constraints.md) | Mission, hard constraints, measurable success criteria |
| [docs/02-research-survey.md](docs/02-research-survey.md) | The annotated evidence base — every claim above, sourced |
| [docs/03-hardware-reality.md](docs/03-hardware-reality.md) | GTX 1080 capability math, software pins, cloud lane, budgets |
| [docs/04-architecture-spec.md](docs/04-architecture-spec.md) | Engineering-level spec: block designs, model-size presets (K-Nano, K-Mini), parameter budgets |
| [docs/05-training-methodology.md](docs/05-training-methodology.md) | Fledge in full: optimizer, schedule, data mixtures, Roost protocol, ablation ladder |
| [docs/06-evaluation-plan.md](docs/06-evaluation-plan.md) | Probes, benchmarks, baselines, capability demos |
| [docs/07-roadmap.md](docs/07-roadmap.md) | Phase details, compute ledger, risk register |
| `kestrel/` | *(Optional appendix)* An untested PyTorch sketch of the blocks, useful when implementation eventually begins |
