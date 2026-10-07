# Project Kestrel

**A small, dense, adaptively-learning LLM, built from scratch. Kestrel-Nano (157M) is released and runs natively on any Windows PC in Kestrel Studio. No Python, no GPU, no cloud.**

## ⬇ Download

| release | what's in it | get it |
|---|---|---|
| **Kestrel V1** | Kestrel Studio app + **Kestrel-Nano V1** (4k context, instruction-tuned) | [Releases → v1.0.0](https://github.com/YoMosa2009/Kestrel/releases/tag/v1.0.0): `Kestrel-V1-Windows-x64.zip` |
| **Kestrel V2** | V1 + **Roost consolidation fixed**: the model can learn new facts after training | [Releases → v2.0.0](https://github.com/YoMosa2009/Kestrel/releases/tag/v2.0.0): `Kestrel-V2-Windows-x64.zip` |

Unzip and run `KestrelStudio.exe`. The bundled model loads automatically. It runs on any 64-bit Windows 10/11 PC.

**Naming:** **V1** and **V2** are the public releases. `v0.1`/`v0.2` name the internal pretraining checkpoints, and the next model plan ([docs/11](docs/11-v3-plan.md): multi-token prediction plus distillation) is **V3**.

### Kestrel-Nano V1 at a glance
- **Architecture:** 157.2M parameters. A gated-linear-attention hybrid (15 linear-attention + 5 attention blocks), a looped core (R=1–3, "thinking depth") and 2 product-key-memory sites (16,384 slots each).
- **Training:** 3.23B tokens. Pretraining 3.0B ([docs/10](docs/10-phase-c-plan.md)), then context extension 1k → 4k by position interpolation, then SFT on 23.8k examples. Trained on an RTX 3060 and Colab A100.
- **Evals vs v0.2:** loss at positions 3k–4k 1.868 → **1.580**; val loss at 1k 1.770 → **1.771** (no regression); two-hop composition 15% → **45%**; verbatim copying 3.75% → **0%**.
- **Native engine:** `app/Kestrel.Engine` is a C# port, checked against PyTorch: max |Δlogit| 0.0002 (f32) / 0.018 (f16), and the same top token as PyTorch 100% of the time. The tokenizer matches Hugging Face on 319/319 test strings. See [app/README.md](app/README.md).
- **Limits:** it's a 157M model. It hallucinates confidently, and its 3-digit arithmetic is 0%.

> Background reading: **[MASTER-PLAN.md](MASTER-PLAN.md)** has the full plan (names Kestrel / Fledge / Roost, architecture, training methodology, risks). [docs/09](docs/09-nano-v01-findings.md) is the honest v0.1 evaluation, and [docs/12](docs/12-roost-v2.md) covers the Roost V2 fix.

**Focus domains:** coding, tool use, language. These are deliberately skill- and reasoning-heavy, where a small looped, memory-augmented model can punch above its weight.

A kestrel is a small falcon that hunts by hovering: reading the wind and adjusting continuously. That is the design thesis. A small model that stays light but *adapts in place*, instead of a frozen giant.

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
  03-hardware-reality.md        Original GTX 1080 analysis (SUPERSEDED — lab GPU is now an RTX 3060; see docs/08 §3b)
  04-architecture-spec.md       Kestrel v0.1 spec: blocks, presets (Nano/Mini), parameter budgets
  05-training-methodology.md    Scale ladder, optimizer, data curriculum, continual-learning protocol
  06-evaluation-plan.md         Probes, benchmarks, baselines, "teach-me-today" tests
  07-roadmap.md                 Phases P0–P7, compute ledger, risk register (cloud parts SUPERSEDED)
  08-real-model-roadmap.md      The forward plan: local-only decision, Phase D tuning, what was rejected
  09-nano-v01-findings.md       Honest v0.1 evaluation
  10-phase-c-plan.md            The 4.86B corpus + the dual-context v0.1 baseline
  11-v3-plan.md                 V3 plan: multi-token prediction, then distillation from a local teacher
  12-roost-v2.md                Roost V2: what was wrong with consolidation, the fixes, the measurements
kestrel/                        the model and trainer, trained and running
  config.py                     Model configs + presets (test/s/m/nano/mini)
  model.py                      Implementation: GLA hybrid, looped core, PKM, masked chunked CE
  data.py                       Sharded/curriculum/prefetch loaders + SFTLoader (token+mask)
  train.py                      Trainer: Muon+WSD, resumable, pause/stop controls, live status
  optim.py  generate.py  smoke_test.py  test_sft.py
  roost/                        lifelong learning: store.py, session.py, consolidate.py, test_roost.py
scripts/
  prepare_data_v2.py            the 4.86B corpus builder
  prepare_sft.py                the SFT set builder (emits a loss mask)
  bench_step.py                 FULL-step benchmark (bench_throughput.py excludes the optimizer)
  profile_step.py               op breakdown + MFU on the current card and precision
  stopping_point.py             fits the loss curve to find where more training stops paying
  eval_generalization.py        the scoreboard — always compare at a matched --seq
  roost.py                      teach / ask / sleep / teach-me-today / absorb
  roost_v2.py                   Roost V2 benchmark sweep + builds Kestrel-Nano V2
  export_gguf.py                checkpoint -> .gguf (arch "kestrel"; f32/f16/q8_0) + parity fixture
  eval_context.py               loss by position out to 4k
notebooks/
  kestrel_colab.ipynb           Colab A100: 4k context extension -> SFT -> Roost -> release export
  kestrel_v2_roost.ipynb        Colab: Roost V2 sweep + V2 build, from the public release (no Drive)
app/                            Kestrel Studio (WPF) + Kestrel.Engine (native C# inference) + kestrel CLI
tools/KestrelMonitor/           WPF live monitor: charts, probes, Pause/Resume/Stop
monitor.cmd                     launches it
requirements.txt
```

## Status

- [x] Research sweep (2026-07-19)
- [x] Design docs 01–07 + [MASTER-PLAN.md](MASTER-PLAN.md)
- [x] **Implementation** — model, trainer (Muon + WSD, resumable), memmap data pipeline, tokenizer, inference (`kestrel/`, `scripts/`); all CPU smoke tests pass
- [x] **Kestrel-Nano v0.1 trained** (2026-07-23) — 157M params, **1.311B tokens**, bf16 on a rented RTX 3090, final val_loss **2.310**, ~$21
- [x] **Honest evaluation** — [docs/09](docs/09-nano-v01-findings.md). Product-key memory validated (gate → ~1.0); the looped core changes computation but is not yet a quality dial; **the model learned form, not content** (undertrained)
- [x] **Phase B — scan optimization** — [docs/08 §3](docs/08-real-model-roadmap.md). `chunk_size` 64→256 (~1.12× end-to-end); decay-folding **rejected as numerically unsafe**; profiling shows the model is matmul-bound at ~30% MFU, so the scan was never the bottleneck
- [x] **Phase C complete** ([docs/10](docs/10-phase-c-plan.md)) — **4.86B-token dataset** (`data_5b/`, 30 per-domain shards): StarCoderData x10 languages, **shell/PowerShell/cmd**, git-commits + GitHub issues, FineMath, SmolTalk, tool-calling, secure-coding. Plus **curriculum staging + anneal** in the trainer and a **generalization eval suite** with v0.1 baselined as the control
- [x] **Hardware change** (2026-08-27) — lab GPU is now an **RTX 3060 12 GB (Ampere)**. bf16 gives **1.95x** the GTX 1080 (fp32 alone is only 1.17x). Local Phase D: **3B tokens in ~8.5 days**. See [docs/08 §3b](docs/08-real-model-roadmap.md)
- [x] **LOCAL ONLY** (2026-09-14) — no cloud compute for any phase until that changes. Supersedes every cloud budget in docs/01/03/07 and docs/08 §4; **Kestrel-Mini is shelved** (~87 days locally). See [docs/08 §0b](docs/08-real-model-roadmap.md)
- [x] **Phase D, done (v0.2)**: **3.00B cumulative tokens** on the RTX 3060 (step 18590). val_mean @1024 **1.770 vs the 2.523 v0.1 control**
- [x] **Roost built** (`kestrel/roost/`) — episodic store with template paraphrases, GLA session-state serialize/restore, and the eval-gated nightly consolidation job. Slot containment is enforced **structurally**, not by optimizer settings: AdamW's weight decay updates every row regardless of gradient, so gradient masking alone silently writes to all 32k slots
- [x] **SFT set + loss masking** — 23,289 examples / 16.2M tokens (`data_sft/`), built for a 4k context, using the exact template pretraining already saw. Loss is scored on Assistant tokens only; **43% of SFT tokens are prompt**, so without masking half the compute teaches the model to ask questions
- [x] **Context extension to 4k** (Colab A100, 2026-10-05): position interpolation ×4 + 1,000 steps (0.2B tokens). Only the 5 attention blocks carry RoPE; the 15 GLA blocks are NoPE and length-agnostic. Loss at 3k–4k **1.868 → 1.580**, no regression at 1k (1.770 → 1.771)
- [x] **SFT + release: Kestrel V1** (2026-10-06): 165 steps at seq 4096 on the masked SFT set (val 0.97). Exported to `.gguf` and shipped with **Kestrel Studio**, a native WPF app with a C# engine checked against PyTorch ([v1.0.0](https://github.com/YoMosa2009/Kestrel/releases/tag/v1.0.0))
- [x] **Kestrel V2: Roost consolidation**: V1's consolidation learned nothing, and its test couldn't have shown it if it had. Fixed and measured in [docs/12](docs/12-roost-v2.md): V1 recalled **0%** of taught facts. V2's consolidation recalls **100%** of the 7 self-facts it was taught (answer NLL 3.26 → 0.001) at **+0.04%** regression, with every non-memory weight bit-identical. The 50-fact nonsense benchmark reaches 40% (gate 60%), so that one is **not passed yet** ([v2.0.0](https://github.com/YoMosa2009/Kestrel/releases/tag/v2.0.0))
- [ ] **V3** ([docs/11](docs/11-v3-plan.md)): stop buying quality with wall-clock and buy it with **signal per token**. Switch on the **multi-token-prediction head** (built since day one, never run) and ablate it, then **distil from a local quantized teacher** (Qwen2.5-Coder-1.5B, top-k logits precomputed once over a curated subset). VLM deferred behind both

### Trained artifacts
Model files are attached to the [GitHub releases](https://github.com/YoMosa2009/Kestrel/releases),
not committed: `.gguf` (f16 / q8_0) for Kestrel Studio, plus `kestrel-nano-v1-weights.pt`
(PyTorch: cfg + state dict) for `python -m kestrel.generate --ckpt ... --interactive`, and
`roost-data.zip` (the Roost benchmark's replay and validation shards). The full training
corpora (~10 GB) stay local; they're reproducible with `scripts/prepare_data_v2.py`.

## Ground rules baked into every decision

1. **Portable by construction:** pure PyTorch, matmul-heavy, no custom kernels. Originally written FP32/Pascal-safe for a GTX 1080; now runs bf16 on the RTX 3060 with the same code. `torch.compile` is *not* used — it graph-breaks on the GLA scan's chunk loop, which was measured on a 4090 with Triton available, so the blocker is the architecture rather than the platform.
2. **Evidence or ablation:** every deviation from a vanilla transformer must beat the matched-compute baseline or it gets cut.
3. **Honest scale & budget:** everything runs on one RTX 3060 for the cost of electricity. The local ladder at the measured 4,100 tok/s is 10B tokens ≈ 20 days, 15B ≈ 34 days, 20B ≈ 48 days — past which the 4.87B-token corpus hits its ~4× repeat limit and needs extending, not funding. WSD means every rung resumes the last one, so no work is ever thrown away.
