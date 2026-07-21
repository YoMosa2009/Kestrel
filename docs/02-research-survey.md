# 02 — Research Survey (annotated)

What the field knows as of mid-2026, organized by the question each theme answers for Kestrel. Each entry ends with **→ take**: what we adopt, adapt, or reject.

---

## A. The critique we are answering: common sense & frozen learning

**Melanie Mitchell — *Artificial Intelligence: A Guide for Thinking Humans*** ([book site](https://melaniemitchell.me/aibook/), [Wikipedia](https://en.wikipedia.org/wiki/Artificial_Intelligence:_A_Guide_for_Thinking_Humans))
Her core arguments, restated as engineering requirements:

1. **The barrier of meaning.** Systems manipulate symbols without grounded understanding; performance collapses off-distribution (the "long tail"). *Requirement: measure robustness, don't assume it; keep a common-sense eval slice separate from aggregate scores.*
2. **Common sense is vast, implicit, and mostly never written down.** Explicit encoding (Cyc) failed after decades. Humans get it from *core knowledge* — Spelke-style primitives about objects, agents, space, causality — plus embodied experience. *Requirement: we can't encode it, and pure web text under-represents it. Curriculum must over-weight procedural, causal, and physical text (how-to, physics QA, code + execution traces — code is a cheap simulator of state and causality).*
3. **"Without concepts there can be no thought, and without analogies there can be no concepts"** (Hofstadter & Sander, quoted by Mitchell). Abstraction/analogy is the engine of understanding — the thing Copycat/Bongard/ARC test and LLMs fail. *Requirement: include abstraction probes (mini-ARC-style, Winograd-style) in evals; loops (Pillar 2) are our architectural bet that iterated computation helps composition/abstraction per parameter.*
4. **Humans learn continuously; models are frozen at deployment.** A child updates from single experiences without erasing old knowledge. *Requirement: this one is architecturally addressable today — see theme D. It is Kestrel's differentiator.*

**Our position:** (1)–(3) we *mitigate and measure*; (4) we *engineer directly*. No overclaiming.

---

## B. How small models get good: recipes, shape, data

**SmolLM2/SmolLM3 (HuggingFace)** — [SmolLM3 blog](https://huggingface.co/blog/smollm3), [repo](https://github.com/huggingface/smollm), [SmolLM2 paper](https://arxiv.org/pdf/2502.02737)
3B model, 11.2T tokens, fully open recipe. Key lessons: (a) **data-centric beats architecture-centric** — staged mixtures (web → +code/math → high-quality anneal) with ablations run at small scale before committing; (b) architecture is conservative (GQA, NoPE at 3:1); (c) mid-training "anneal" on premium data moves benchmarks disproportionately.
**→ take:** the entire *methodological* posture — ablate small, stage the data, anneal on quality. Their data (FineWeb-Edu, Stack-Edu, FineMath, SmolTalk) is open; we reuse it rather than curating from scratch.

**MobileLLM (Meta)** — [arXiv 2402.14905](https://arxiv.org/abs/2402.14905)
Sub-billion design study: **deep-and-thin beats wide** at small scale; SwiGLU, GQA, tied embeddings, and **immediate block-wise layer sharing** (run the same block twice in a row) all win. Layer sharing ≈ free accuracy for zero extra weights (only latency).
**→ take:** deep-thin proportions and tied embeddings everywhere; their layer-sharing result is the single-block precursor of our looped core.

**Physics of Language Models 3.3 (Allen-Zhu & Li)** — [arXiv 2404.05405](https://arxiv.org/abs/2404.05405) (ICLR'25)
Models store **~2 bits of factual knowledge per parameter** (robust to int8 quantization); junk-mixed data slashes capacity (domain tokens partially rescue it); knowledge needs many exposures/rewrites to become extractable.
**→ take:** (a) a 150M dense model caps at ~37 MB of facts — *don't spend dense parameters on facts*; push knowledge into sparse memory (theme D) and retrieval. (b) Data quality is a capacity multiplier, not a nicety. (c) Facts must appear in multiple phrasings — favor corpora with rewrites (Cosmopedia-style).

**Data-constrained scaling (Muennighoff et al.)** — [arXiv 2305.16264](https://arxiv.org/abs/2305.16264)
Repeating a corpus **up to ~4 epochs ≈ as good as fresh data**; useful gains to ~16 epochs; dead by ~40. Mixing up to ~50% code into NL training doesn't hurt NL.
**→ take:** our 3B-token hero budget only needs ~1–3B unique tokens on disk (≈2–6 GB tokenized). Code-heavy mixtures are safe and serve the common-sense-via-causality agenda.

---

## C. Architecture: efficiency mechanisms with recent validation

**Hybrid linear/full attention at 3:1** — [Qwen3-Next](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd), Kimi Linear, [Raschka's survey](https://magazine.sebastianraschka.com/p/beyond-standard-llms), [pure-PyTorch Gated DeltaNet reference](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/08_deltanet/README.md)
Industry convergence in 2025-26: ~75% of layers use gated linear attention (Gated DeltaNet family — fixed-size recurrent state, O(1) per token), ~25% full softmax attention for precise recall. Hybrids consistently beat either extreme.
**→ take:** 3:1 hybrid as the backbone. Bonus nobody markets: the linear layers' recurrent state **is a serializable "mind-state"** — megabytes, not a gigabyte KV cache — enabling cross-session memory (theme D). We start with scalar-gated GLA (simpler, provably chunkable in pure PyTorch — Pascal-safe) with Gated DeltaNet as an upgrade ablation.

**Looped/recurrent-depth models** — [Ouro, arXiv 2510.25741](https://arxiv.org/abs/2510.25741) ([site](https://ouro-llm.github.io/)); Huginn (Geiping et al. 2025); [survey of loop models](https://github.com/huskydoge/Awesome-Loop-Models)
Ouro 1.4B ≈ standard 4B; 2.6B ≈ 8B — **2–3× parameter efficiency** by re-running shared middle layers with a learned early-exit gate. Crucially, analysis shows the gain is in knowledge *manipulation* (composition, multi-hop), not storage — exactly complementary to memory layers, which are storage.
**→ take:** Pillar 2. Entry → looped core (R=1–4, stochastic during training) → exit. This is also our answer to "adaptive compute": hard inputs get more loops.

**Memory layers (product-key)** — [Meta, Memory Layers at Scale, arXiv 2412.09764](https://arxiv.org/abs/2412.09764)
Trainable key-value banks activated sparsely (top-k of ~million slots): capacity without FLOPs. Meta showed factual-task gains matching models with 2× dense compute.
**→ take:** Pillar 3, sized to our scale (~2^14 slots ×2 on Nano/150M, ~2×48k on Mini/500M). Pure embedding-lookup math — ideal for Pascal and for CPU-offload at deployment.

**SuperBPE** — [COLM 2025](https://arxiv.org/abs/2503.13423), [site](https://superbpe.github.io/)
Tokens that cross whitespace ("superwords"): same vocab size, **up to 33% fewer tokens** (~21% typical), +8.2% MMLU at 8B at matched train compute. More text per context window and per FLOP — our two scarcest resources.
**→ take:** Pillar 4. Train a 48k SuperBPE-style tokenizer (subword→superword transition at ~80% of vocab) on our mixture. Cheap, high-confidence win.

**Multi-token prediction (MTP)** — DeepSeek-V3 practice
Auxiliary head predicting token t+2: densifies the training signal and later powers self-speculative decoding (~1.5-2× inference speedup).
**→ take:** small aux head, λ=0.2, as a *flagged ablation* — evidence at <100M scale is thin; it must earn its place (Tenet 2).

**Optimizer: Muon** — [Muon in modded-nanoGPT](https://varunneal.github.io/essays/muon), [NanoGPT speedrun overview](https://www.emergentmind.com/topics/nanogpt-speedrun), [optimizer benchmark](https://arxiv.org/html/2509.01440v1)
Orthogonalized momentum for weight matrices; the engine of the nanoGPT speedrun (45 min → ~2.2 min) and since adopted at frontier scale (Kimi K2, GLM 4.5). ~1.35-2× data efficiency over AdamW in controlled tests, **and** stores only momentum (half of Adam's optimizer memory) — doubly valuable in 8 GB.
**→ take:** Muon for matrices + AdamW for embeddings/gains/scalars. Pure PyTorch (Newton-Schulz iterations = matmuls; Pascal-safe).

---

## D. The differentiator: learning after deployment

**Titans (Google, NeurIPS 2025)** — [arXiv 2501.00663](https://arxiv.org/abs/2501.00663)
A neural memory module whose weights are *updated at test time* by a surprise-based rule; attention = short-term memory, neural memory = long-term. Validates the layered-memory framing.
**→ take:** conceptual blueprint for v2. Our v1 medium-term memory is simpler: the GLA recurrent state itself, persisted across sessions.

**SEAL (MIT, NeurIPS 2025)** — [arXiv 2506.10943](https://arxiv.org/abs/2506.10943), [VentureBeat summary](https://venturebeat.com/ai/self-improving-language-models-are-becoming-reality-with-mits-updated-seal)
Models generate their own finetuning data ("self-edits"), trained by RL on downstream improvement. Shows self-directed weight updates work and scale.
**→ take:** v2+ ambition (needs RL infra). But its *format* insight is v1-usable: restating session content as clean declarative rewrites before consolidation improves absorption — we do this with templates instead of RL.

**Sparse Memory Finetuning (Meta, 2025)** — [arXiv 2510.15103](https://arxiv.org/abs/2510.15103), [follow-up comparison](https://arxiv.org/abs/2605.03229)
THE enabling result for Kestrel's consolidation loop: finetune **only the memory-layer slots** the new data activates (TF-IDF-selected), leaving dense weights frozen. New-knowledge acquisition matches full finetuning while forgetting collapses: **NaturalQuestions F1 drop 89% (full-FT) / 71% (LoRA) / 11% (sparse memory)**.
**→ take:** Pillar 5's mechanism, verbatim. This is why Kestrel carries product-key memory from day one: the memory bank is the organ of lifelong learning, not just a capacity trick.

**Synthesis — the Kestrel memory hierarchy:**

| Timescale | Mechanism | Cost | Analogy |
|---|---|---|---|
| Seconds (in-context) | Attention over context window | KV cache | Working memory |
| Session (minutes–hours) | GLA recurrent state, serialized on exit | ~MBs | Short-term memory |
| Days+ (explicit) | External store (embeddings + kNN) read as context | Disk | Notes/episodic |
| Permanent (implicit) | PKM slots via nightly sparse consolidation w/ replay + eval-gated rollback | One small training job | Sleep consolidation |

---

## E. Deployment reality (sub-1B on a 1080 — and on a phone)

[llama.cpp on Pascal benchmarks/discussion](https://github.com/ggml-org/llama.cpp/discussions/15013), [GTX 1080 Ti for local LLM (2026)](https://ariya.io/2026/02/gtx-1080-ti-for-local-llm), [running 30B MoE on a GTX 1080](https://mdda.net/blog/tech/dl/llama-cpp-moe-on-an-old-gtx-1080)
Pascal remains fully supported by llama.cpp CUDA (dp4a int8 path). Inference is bandwidth-bound: at 320 GB/s, a 500M model at Q4 (~300 MB) sits far under the bandwidth limit → ~40–80 tok/s realistically, and Nano (~100 MB) faster still — comfortably interactive, and light enough for CPU-only laptops and phones. PKM banks quantize well and can even sit in system RAM (sparse fetches).
**→ take:** the end-product story is sound and now *fully* local: *train the science small, train Nano locally end-to-end, and run both shipped models on the same GPU (or a phone).*

---

## F. What we considered and rejected (for v1)

- **BitNet/ternary training** — attractive for deployment but immature training dynamics at our scale and no Pascal-tuned kernels for training; revisit at a later scale-up stage.
- **Byte-level + dynamic chunking (BLT/H-Net)** — the most radical token-efficiency play; too much architecture risk for a one-GPU lab v1. SuperBPE captures most of the win cheaply. Revisit v2.
- **Mixture-of-Experts** — capacity-per-FLOP overlaps with PKM but with much worse VRAM behavior at training time (all experts resident). PKM is strictly better-suited to 8 GB.
- **Mamba/SSM CUDA kernels & FlashLinearAttention (Triton)** — require sm_70+; violates Pascal-safety. Chunked pure-PyTorch GLA achieves the same asymptotics via cuBLAS matmuls.
- **Test-time-training layers (TTT/Titans) in v1** — powerful but training-loop complexity (inner-loop gradients) is a schedule risk; staged for v2 behind the same memory interface.
