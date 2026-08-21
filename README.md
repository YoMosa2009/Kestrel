# Project Kestrel

**A small, dense, adaptively-learning LLM architecture — two owned models: Kestrel-Nano (~150M) trained 100% on a GTX 1080 for $0, and Kestrel-Mini (~500M) cloud-pretrained for under $25. Both run on nearly any computer.**

> **Start here → [MASTER-PLAN.md](MASTER-PLAN.md)** — the self-contained plan: names (Kestrel / Fledge / Roost), how the architecture and training methodology work, how they improve on existing approaches, the phased plan, and honest risks. **Status: Kestrel-Nano v0.1 is trained.** 157M params, 1.311B tokens, on one rented RTX 3090 for ~$21 — it generates coherent text and runs locally. See [docs/09](docs/09-nano-v01-findings.md) for the honest evaluation and [docs/08](docs/08-real-model-roadmap.md) for where it goes next.

**Focus domains:** coding, tool use, language — deliberately skill/reasoning-heavy, where a small looped-and-memory-augmented model can punch above its weight.

A kestrel is a small falcon that hunts by hovering — reading the wind and adjusting continuously. That is the design thesis: a small model that stays light but *adapts in place*, instead of a frozen giant.

## Why this project exists

- **Independence from the cloud.** People in low-connectivity regions (and anyone who values privacy/ownership) deserve capable local models. Deployment: **Nano (~150M, ~100 MB quantized)** and **Mini (~500M, ~300 MB quantized)** — an 8 GB GPU, a CPU-only laptop, in principle a phone. Nano is also *trained* entirely locally.
- **Small ≠ shallow.** Recent evidence (Ouro looped models, MobileLLM, SmolLM2/3, Qwen3-Next hybrids) shows small models close the gap through *architecture shape, data quality, and adaptive compute* — not just scale.
- **The "common sense & frozen weights" critique.** Melanie Mitchell's *AI: A Guide for Thinking Humans* argues today's models lack grounded common sense and — unlike humans — stop learning the moment training ends. Kestrel's memory subsystem (recurrent session state → sparse memory layers → nightly consolidation) is a direct engineering response.

## The five pillars (v0.1)

| # | Pillar | What it buys us | Key evidence |
|---|--------|-----------------|--------------|
| 1 | **Hybrid sequence mixer** — 3 gated linear-attention blocks : 1 full-attention block | O(1) inference state per token, long context on 8 GB, persistent "mind-state" | Qwen3-Next, Kimi Linear, Samba |
| 2 | **Looped core** — middle block group re-runs 1–4× | Deeper effective reasoning without more parameters; adaptive compute | Ouro (1.4B ≈ 4B dense), Huginn |
| 3 | **Product-key memory layers** — large sparse key-value banks | Knowledge capacity without FLOPs; the *plastic tissue* for continual learning | Meta Memory Layers at Scale; Sparse Memory Finetuning (11% forgetting vs 89% full-FT) |
| 4 | **Token efficiency** — superword BPE tokenizer + multi-token-prediction aux head | ~20–30% more text per FLOP and per context window; speculative decoding later | SuperBPE (COLM 2025) |
| 5 | **Layered memory & consolidation** — context → recurrent state → memory slots → nightly sparse consolidation | A model that *learns from use* without catastrophic forgetting | Titans, SEAL, Meta sparse-memory-FT |

## Repository map

```
docs/
  01-goals-and-constraints.md   Mission, hard constraints, success criteria, honesty section
  02-research-survey.md         Annotated survey (the "study" backbone, with links)
  03-hardware-reality.md        What a GTX 1080 + 16 GB RAM can and cannot do, with math
  04-architecture-spec.md       Kestrel v0.1 spec: blocks, presets (Nano/Mini), parameter budgets
  05-training-methodology.md    Scale ladder, optimizer, data curriculum, continual-learning protocol
  06-evaluation-plan.md         Probes, benchmarks, baselines, "teach-me-today" tests
  07-roadmap.md                 Phases P0–P7, two-budget compute ledger, risk register
kestrel/                        (optional appendix — untested design sketch, not built/run yet)
  config.py                     Model configs + presets (test/s/m/nano/mini)
  model.py                      Reference implementation (pure PyTorch, Pascal-safe)
  smoke_test.py                 CPU correctness tests: chunked-scan equivalence, overfit, state carry
requirements.txt
```

## Status

- [x] Research sweep (2026-07-19)
- [x] Design docs 01–07 + [MASTER-PLAN.md](MASTER-PLAN.md)
- [x] **Implementation** — model, trainer (Muon + WSD, resumable), memmap data pipeline, tokenizer, inference (`kestrel/`, `scripts/`); all CPU smoke tests pass
- [x] **Kestrel-Nano v0.1 trained** (2026-07-23) — 157M params, **1.311B tokens**, bf16 on a rented RTX 3090, final val_loss **2.310**, ~$21
- [x] **Honest evaluation** — [docs/09](docs/09-nano-v01-findings.md). Product-key memory validated (gate → ~1.0); the looped core changes computation but is not yet a quality dial; **the model learned form, not content** (undertrained)
- [x] **Phase B — scan optimization** — [docs/08 §3](docs/08-real-model-roadmap.md). `chunk_size` 64→256 (~1.12× end-to-end); decay-folding **rejected as numerically unsafe**; profiling shows the model is matmul-bound at ~30% MFU, so the scan was never the bottleneck
- [ ] **Phase C** — 5B-token dataset built for generalization ([docs/10](docs/10-phase-c-plan.md)) — planned, not started
- [ ] **Phase D/E/F** — over-train (20B+ tokens), SFT + Roost, quantize & ship

### Trained artifacts (not in this repo)
`ckpt.pt` (1.4 GB) and the tokenized corpora (~2 GB) exceed GitHub's limits and are kept
locally. The corpora are reproducible with `scripts/prepare_data.py`; the tokenizer
(`tokenizer/kestrel-bpe.json`) **is** committed, so a checkpoint can be loaded and run
with `python -m kestrel.generate --interactive`.

## Ground rules baked into every decision

1. **Pascal-safe (local):** pure PyTorch, matmul-heavy, FP32; no Triton, no FlashAttention, no bf16. The same code runs bf16 + compiled on the cloud pod (Mini) — the architecture is legal in both worlds.
2. **Evidence or ablation:** every deviation from a vanilla transformer must beat the matched-compute baseline or it gets cut.
3. **Honest scale & budget:** the science is proven locally at 20M–60M for $0; **Kestrel-Nano (~150M) is trained end-to-end locally for $0**; **Kestrel-Mini (~500M) is a ≤$25 cloud prototype**; caps are ≤1 week local GPU time + ≤$25 cloud, with a costed resume-from-checkpoint ladder to competitive sub-1B scale.
