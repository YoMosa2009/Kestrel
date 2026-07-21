# 05 — Training Methodology

## 1. The scale ladder (how a one-GPU lab does science)

| Rung | Role | Budget | Cadence |
|---|---|---|---|
| K-test | correctness (CPU) | seconds | every commit |
| K-S + ~300M tokens (GTX 1080) | **ablation currency** | ~7 h | overnight, one question per night |
| K-M + ~350M tokens (GTX 1080) | confirmation at 3× scale | ~22 h | only for the surviving design |
| **K-Nano (~150M) + ~1–1.5B tokens (GTX 1080)** | fully-local hero, $0 | ~6–8 days | once, after the design freezes; FP32; WSD-extendable |
| **K-Mini (~500M) + ~3–4B tokens (RunPod, ≤$25)** | cloud stretch | ~60–65 h | once; bf16 + compile on the pod (Pascal rules are local-only) |

Discipline: one change per run; fixed seed set {1337,1338,1339} (3 seeds only for headline claims, 1 for scouting); every run = config hash + JSONL metrics + resumable checkpoint (model, optimizer, dataloader cursor, RNG). Decisions come from the pre-registered probe suite (doc 06), with a keep-rule declared *before* the run: **keep if ≥1% compute-equivalent val-loss win or a capability unlock, at ≤5% throughput cost.**

## 2. Optimization

- **Muon** for all 2-D non-embedding matrices (momentum 0.95, Nesterov, 5 Newton-Schulz steps — pure matmuls, Pascal-happy). Evidence: nanoGPT speedrun lineage, Kimi K2/GLM-4.5 at scale, ~1.3–2× data-efficiency over AdamW in controlled benchmarks. Bonus: momentum-only state halves optimizer VRAM vs Adam.
- **AdamW** for embeddings, norms/gains/gates, PKM values (β 0.9/0.95, wd 0.01 on embeddings only).
- **Schedule: WSD** (warmup ~300 steps → long stable plateau → 1−√ decay over final ~15%). Chosen specifically for our lab reality: a stable-phase checkpoint can be *extended, branched, or annealed at any time* without committing to a total token count up front — crucial when a hero run might be interrupted or a better data mix arrives mid-run.
- Grad clip 1.0; z-loss 1e-4; batch ≈ 96–128k tokens/step via gradient accumulation (8×1024 micro-batch × 12–16 accum); gradient checkpointing on.
- Context curriculum: 1024 for the first 75% (cheaper steps), 2048 for the last 25% + anneal. Long-context proper (8k+) is a post-hero extension — GLA layers make it cheap; the few attention layers get windowed.

## 3. Data (English + code, v1 scope)

All open, all pre-filtered by stronger labs — we spend compute on training, not curation. Mixture is **domain-tilted toward code, tools, and language** (the target capabilities), heavier on code than a general-purpose model would be — safe because up to ~50% code is loss-neutral for NL (Muennighoff) and code is the cheapest source of executable causality:

| Source | Share | Why |
|---|---|---|
| Stack-Edu + high-quality Python/JS/MD | 30% | the primary skill target; code = executable causality; strengthens tool-call syntax |
| FineWeb-Edu (sample) | 33% | best open educational web filter; language fluency + the capacity-per-token argument (Allen-Zhu) |
| Tool-use / function-calling / agentic traces (e.g. open toolformer-style, code-with-execution, shell/CLI transcripts) | 10% | tool usage is a first-class goal and under-represented in generic corpora — deliberately boosted |
| FineMath 4+ | 9% | reasoning density (transfers to coding) |
| DCLM-baseline (sample) | 8% | diversity complement to FWE's textbook bias |
| Cosmopedia-v2 | 5% | synthetic textbooks = many-phrasings knowledge (extraction robustness) |
| Instruct/QA (SmolTalk subset) | 5%, anneal-weighted | format familiarity, not full SFT |

- **Common-sense slice (the Mitchell curriculum):** within the above, over-sample how-to/procedural text, physical-QA, and code-with-execution-traces. Tagged shards so ablation A9 can measure its effect on the commonsense eval slice directly.
- Budgets: **Nano** ≈ ~1–1.5B tokens seen (≈1B unique × ≤1.5 epochs); **Mini** ≈ ~3–4B tokens seen (≈1.5–2.5B unique × ≤2 epochs) — both well inside the ≤4-epoch free-repetition zone, so ~2–2.5B unique tokens on disk feed both.
- **Pipeline (16 GB RAM discipline):** HF `datasets` streaming → tokenize on the fly → flat `uint16` shards on disk (~3–4 GB total) → `np.memmap` dataloader. Raw corpora are never stored. Per-domain val shards held out.
- Tokenizer: **48k BPE with superword merges** (SuperBPE-style: standard subword merges to 80% of vocab, then whitespace-crossing merges), digit-splitting, code-friendly pretokenization. Gate S2: ≥18% fewer tokens on held-out web/code/math vs standard BPE at same vocab. Trained once in M1 on a ~2 GB stratified sample (CPU, streamed).

## 4. What we deliberately don't do in pretraining

- No local distillation from a big teacher (nothing large runs usefully on the 1080 for logit serving). Instead we *consume* the open synthetic corpora that already embody frontier-teacher knowledge (Cosmopedia, SmolTalk, FineMath) — offline distillation someone else paid for.
- No RLHF-style post-training in v1 — the hero gets light SFT on SmolTalk-style data at most. Alignment research is out of scope until M6.

## 5. The continual-learning protocol (Kestrel's differentiator)

Runs *after* pretraining, on deployment hardware — this is a product feature and a research contribution.

**Session time (automatic):** GLA states persist per user/agent (serialize on exit ~MBs; restore on start). Episodic store: session "teachables" (facts, corrections, preferences) are template-rewritten into declarative statements (+3 paraphrase variants — SEAL's insight, minus the RL), embedded, and appended to a local vector store; top-k retrieved into context at query time (grounding tier).

**Nightly consolidation (the "sleep" job):**
1. Batch = 50% new material (the rewrites) + 50% replay (reservoir sample of pretraining shards).
2. Update **only PKM value slots** activated by the new material (top-t TF-IDF slot selection, keys & dense weights frozen) — AdamW lr 1e-3, ≤200 steps. This is Meta's Sparse Memory Finetuning applied as designed; expected forgetting ~11% of full-FT levels.
3. **Gate:** frozen probe suite before/after. Accept iff new-fact recall ≥60% AND old-suite regression ≤2%; else rollback (PKM values snapshot ≈ tens of MB).
4. Weekly: compaction pass (merge episodic duplicates, decay stale entries).

**Why this beats naive fine-tuning:** plasticity is *architecturally confined* to sparse memory tissue; the model can't lobotomize its own reasoning during a bad night. Failure mode is "didn't learn the fact," never "forgot how to code."

**Applying Roost to the shipped models:** after P6, consolidation of *both* Nano and Mini runs *on the GTX 1080 itself* — only PKM value-slot gradients are needed, which fit comfortably in 8 GB even for Mini's 500M. **v2 track (staged, not promised):** Titans-style test-time-trained neural memory behind the same interface; SEAL-style RL over self-edit formats.

## 6. Ablation ladder (P3; each = one overnight K-S run)

**Core four (guaranteed within the one-week budget):** A0 twin · A8 full Kestrel · A2-inverse (Kestrel−loops) · A3-inverse (Kestrel−memory). The rest of the ladder below runs only if the week has slack after P4 — priorities: A4 (tokenizer), A5 (optimizer).

| ID | Question | Config vs A0 twin |
|---|---|---|
| A0 | baseline: vanilla dense transformer, same d/L/data/tokens/optimizer | — |
| A1 | hybrid 3:1 worth it? | +GLA blocks |
| A2 | loops worth it? | +looped core (report R-curve at eval) |
| A3 | PKM worth it? | +1 site (watch usage entropy, dead slots) |
| A4 | superword tokenizer? | matched **bytes**, not tokens |
| A5 | Muon vs AdamW-only | optimizer swap |
| A6 | MTP aux head? | λ=0.2 |
| A7 | decay-spectrum init? | uniform vs spread gate init |
| A8 | full Kestrel vs A0 | 3 seeds, headline table |
| A9 | common-sense curriculum? | slice re-weighted vs natural mix |

Kill-list honesty: any pillar that fails A-series at K-S *and* K-M gets cut from the hero and documented in the writeup — negative results included.
