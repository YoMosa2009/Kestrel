# 01 — Goals & Constraints

## Mission

Design and empirically validate an LLM architecture + training methodology that:

1. **Punches above its parameter count** — at matched training compute and data, beats a well-tuned Llama-style dense transformer baseline of the same size.
2. **Is token- and FLOP-efficient** — more useful text per token, more capability per FLOP (the two resources our hardware is poorest in).
3. **Learns and adapts after deployment** — retains what users teach it across sessions without catastrophic forgetting, addressing the "frozen weights" limitation Mitchell identifies.
4. **Ships two owned models, sized for universal hardware** — **Kestrel-Nano (~150M, Q4 ≈ 100 MB)** trained from scratch **100% on the GTX 1080 for $0**, and **Kestrel-Mini (~500M, Q4 ≈ 300 MB)** cloud-pretrained **within $25**, both running (and continually learning) locally afterward. A costed resume-from-checkpoint ladder reaches competitive sub-1B scale later.
5. **Is buildable by us, in PyTorch, within the hardware and budget below** — no exotic kernels, no cluster assumptions, ≤1 week of local GPU time + ≤$25 cloud.

Target capability areas (deliberately skill/reasoning-heavy, where small models can punch above weight): **coding, tool usage, language** — plus creativity and general assistant work, offline.

## Hard constraints (the lab)

| Resource | Spec | Consequence for design |
|---|---|---|
| GPU | GTX 1080 (Pascal, sm_61), 8 GB GDDR5X, ~8.9 TFLOPs FP32, 320 GB/s | FP32 training only (FP16 compute is 1/64 rate on Pascal); no tensor cores; no FlashAttention/Triton/torch.compile-GPU; matmul-heavy ops only |
| CPU | i7-4790 (4c/8t, AVX2) | Data preprocessing is overnight-batch work; tokenizer training on samples, not full corpora |
| RAM | 16 GB DDR3 | Memory-mapped pre-tokenized shards, never in-RAM datasets; WSL2 capped via `.wslconfig` |
| OS | Windows 10 + WSL2 | Both stacks work with Pascal; details in doc 03 |
| Budget | **≤ $25 cloud (RunPod) + ≤ 1 week local GPU time** | Local 1080 does the science *and* trains Kestrel-Nano ($0); the single cloud line item is Kestrel-Mini pretraining (~$21–23); full ledger in doc 07 |
| Machines | Plan authored on the user's laptop; the GTX 1080 PC is the lab | Project folder is self-contained — copy the whole `AI Architecture` folder to the PC before P0 (now resident on the lab PC at `S:\AI Architecture`) |
| Software | PyTorch (locally pinned ≤2.7.1 cu126 or 2.4.1 cu121 — see doc 03) | Pure PyTorch FP32 eager locally; the same code runs bf16 + compiled on cloud pods |

## Design tenets

1. **Data and recipe beat architecture cleverness** at small scale (SmolLM2/3 lesson). Architecture work must target things data cannot buy: adaptive compute, cheap capacity, persistent memory, token efficiency.
2. **Every mechanism earns its place through a matched-compute ablation.** One change at a time; fixed seeds; decisions from the probe suite, not vibes.
3. **The memory hierarchy is the product.** Small models cannot know everything; they can *find, remember, and consolidate*. Capacity for facts goes into cheap sparse memory; FLOPs go into reasoning.
4. **Never update dense weights in deployment.** Plasticity lives in memory slots (and optionally LoRA side-cars); consolidation is eval-gated with rollback. This is how we avoid catastrophic forgetting by construction.
5. **Compute proportional to difficulty.** Loops make "thinking harder" a runtime dial, not a parameter-count decision made at training time.

## Success criteria (measurable)

- **S1 (architecture):** Kestrel ≥ baseline on ≥7 of 10 benchmark/probe metrics at K-S (~20M non-embedding) and K-M (~60M) scales, matched tokens & FLOPs.
- **S2 (token efficiency):** ≥18% fewer tokens per reference corpus vs. standard 49k BPE, with no benchmark regression at matched *bytes seen*.
- **S3 (adaptivity):** After a "teach session" of 50 novel facts, ≥60% recall next-day post-consolidation, with ≤2% regression on the frozen eval suite. (Meta's sparse-memory result — 11% forgetting vs 89% for full finetuning — says this is achievable.)
- **S4 (session memory):** Serialize ≤300 MB of state, resume next day, and correctly reference conversation content from the prior session without re-feeding the transcript.
- **S5 (shipped models):** Kestrel-Nano trains to completion from scratch *entirely on the 1080* (~1–1.5B tokens, no divergence, coherent domain-relevant text); Kestrel-Mini trains to completion on ~3–4B tokens for ≤$25. Both models' loss/FLOP sits on or below the trend extrapolated from the K-S/K-M twins.
- **S6 (upgrade ladder):** the released WSD checkpoints + report constitute an executable, costed path to competitive sub-1B scale (~$150 → ~15–20B tokens; ~$800–1.5k → ~100–150B) that an external funder could green-light without us re-deriving anything.

## What we are NOT claiming (honesty section)

- We will **not pretrain Mini (~500M) locally.** The 1080's FP32 throughput caps local training at ~1.5–2B tokens/week regardless of size, which only properly feeds Nano's ~150M; a 500M model needs 5–10B+ tokens (months locally). So Mini's from-scratch pretraining is the one cloud item. The 1080 still runs Mini *inference* (Q4 ≈ 300 MB, ~40–80 tok/s) and *all Roost consolidation* (slot-only gradients fit easily). Nano, by contrast, does everything locally — training included.
- We will **not "solve" common sense.** Mitchell's barrier-of-meaning critique survives any architecture we ship. We target the *engineering-tractable slice*: grounding via retrieval/memory, adaptation via consolidation, and a common-sense-weighted curriculum & eval slice.
- We will **not beat SmolLM2 / Llama-3.2-1B / Gemma-3 / Qwen3 in their size classes on our budget.** Those models see ~1–10T tokens — hundreds to thousands of times our token budgets. Our claim is per-FLOP and per-parameter *efficiency*, demonstrated rigorously by matched twins, plus capabilities frozen models don't have (S3/S4). Nano is an end-to-end-local viability proof; Mini is a stronger prototype whose checkpoint resumes cleanly when more compute arrives. Absolute coding ability is bounded at these sizes — the goal is *the best a 150M/500M model has been at this niche*, not parity with cloud models.
