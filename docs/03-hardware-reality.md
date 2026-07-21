# 03 — Hardware Reality Check

Every number here shapes the design. Milestone M0 replaces estimates (†) with measurements.

## The machine

| Component | Spec | Notes |
|---|---|---|
| GPU | GTX 1080 — Pascal GP104, **sm_61**, 8 GB GDDR5X | 2560 CUDA cores @ ~1.7 GHz |
| FP32 throughput | ~8.9 TFLOPs peak | Our training currency |
| FP16 throughput | ~0.14 TFLOPs (**1/64 rate**) | FP16 *compute* is useless on Pascal; never enable autocast |
| INT8 (dp4a) | ~35 TOPS | Why llama.cpp inference works well |
| Memory bandwidth | 320 GB/s | Inference tok/s ceiling |
| CPU | i7-4790, 4c/8t, AVX2, 3.6-4.0 GHz | Preprocessing bottleneck |
| RAM | 16 GB DDR3 | The tightest constraint after VRAM |
| Storage | (measure free space in M0; data budget ~30-60 GB) | |

## Software matrix (Pascal-safety)

### Works
| Thing | Status |
|---|---|
| PyTorch **≤2.7.1 + cu126** or **2.4.1 + cu121** wheels | sm_61 included ([pytorch#157517](https://github.com/pytorch/pytorch/issues/157517): Pascal dropped only from cu128/cu129 builds in 2.8-2.9) |
| CUDA 12.x runtime | Last major CUDA line with Pascal (CUDA 13 drops sm_61) |
| WSL2 GPU passthrough | Pascal supported in WDDM mode ([NVIDIA WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)); Windows 10 21H2+ OK; use the *Windows* NVIDIA driver only |
| `F.scaled_dot_product_attention` | `math` + memory-efficient backends run on sm_61 in FP32 (verify in M0; worst case: explicit attention) |
| cuBLAS matmuls (FP32) | Our workhorse; chunked-GLA and Muon are designed around them |
| Muon optimizer | Pure PyTorch (Newton-Schulz = matmuls) |
| llama.cpp CUDA backend | Full Pascal support incl. dp4a quantized kernels |
| gradient checkpointing, grad accumulation | Plain PyTorch, version-independent |

### Does NOT work (design around, never fight)
| Thing | Why | Our answer |
|---|---|---|
| FlashAttention 2/3 | needs sm_80+ | SDPA efficient/math; short-ish training context (1-2k) |
| Triton kernels / `torch.compile` on GPU | Triton needs sm_70+ | Eager mode; matmul-heavy design so eager is fine |
| bf16 | Ampere+ | FP32 everywhere |
| fp16 mixed precision | 1/64 compute rate on Pascal | FP32 (weights+compute); FP16 only for *storage* of checkpoints/states |
| Mamba/FLA CUDA+Triton kernels | sm_70+/Triton | Chunked pure-PyTorch GLA (same asymptotics via cuBLAS) |
| PyTorch ≥2.8 cu128+ wheels | Pascal removed | Pin; verify with `torch.cuda.get_arch_list()` containing `sm_61` |

**Environment pin (M0):** WSL2 Ubuntu 22.04 → `pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126`; fallback `torch==2.4.1+cu121`. Same pins work on native Windows; keep both usable. First-run check: `python -c "import torch; x=torch.randn(1024,1024,device='cuda'); print((x@x).norm())"`.

## VRAM budget — Kestrel-Nano (~150M params, FP32, Muon) trains locally ✅

| Item | Size |
|---|---|
| Weights | 0.60 GB |
| Gradients | 0.60 GB |
| Muon momentum (matrices) + Adam states (embeddings) | ~0.75 GB |
| Activations @ batch 8×1024, checkpointed | ~1.5–2.5 GB† |
| CUDA/cuBLAS workspace + fragmentation | ~0.7 GB |
| **Total** | **~4.2–5.2 GB† of 8 GB** ✅ headroom for batch 12-16 or seq 2048 |

This is the whole reason Nano is a fully-local hero: its from-scratch training fits in 8 GB with room to spare.

**Kestrel-Mini (~500M) does *not* fit local training.** Even in FP32-minimal form, weights (2.0) + grads (2.0) + Muon momentum (2.0) already ≈ 6 GB before activations/overhead — and, more decisively, the FP32 *throughput* ceiling (below) would need months for the tokens a 500M model requires. So Mini's pretraining is the single cloud item (RunPod, ≤$25). Mini *inference* at Q4 ≈ 300 MB → ✅ trivial (~40–80 tok/s†), and Roost slot-only finetuning of Mini fits locally (only PKM value gradients needed).

## The cloud lane (RunPod, budget ≤ $25 total — Kestrel-Mini only)

The Kestrel-Mini pretrain rents one GPU for ~60–65 hours. Pricing as of writing (**verify at P6 kickoff**): RTX 4090 community ≈ $0.35/h → ~$21–23 for the run. Selection rule: **tokens per dollar**, not raw speed (a 3090 at ~60% speed and ~60% price is a wash; A100/H100 are worse tokens/$ at this scale).

What changes on a cloud pod vs the Pascal rules below: **everything modern is allowed** — bf16, FlashAttention-backed SDPA, `torch.compile`, current PyTorch. The model code is identical; only precision/backend flags flip. Expected throughput for Mini (~500M, bf16, compiled, seq 2048, ~30–40% MFU on a 4090): ~13–18k tok/s† → **~3–4B tokens in the budget window** (roughly 2× the tokens the old 1B plan would have bought — a direct benefit of shrinking to 500M). Operational rules: pre-tokenized shards (~4 GB) uploaded or re-streamed on-pod; checkpoints pulled off-pod hourly (community pods preempt; WSD resumes losslessly).

## Throughput & calendar-time budget (†estimates, ±2×, M0 measures)

Assume 22–28% MFU of 8.9 TFLOPs → ~2.0–2.5 TFLOPs sustained. FLOPs/token ≈ 6 × N_active (loops count core params each pass).

| Config | Non-emb params | Eff. compute params (R=2) | tok/s† | tokens/day† | Typical run |
|---|---|---|---|---|---|
| K-S (proxy) | ~20M | ~28M | ~13,000 | ~1.1B | 300M tok ablation ≈ **~7 h (overnight)** |
| K-M (proxy) | ~60M | ~85M | ~4,300 | ~370M | 350M tok confirmation ≈ **~22 h** |
| **K-Nano (hero)** | ~125M | ~175M | ~2,100 | ~180M | ~1–1.5B tok from-scratch ≈ **~6–8 days** (WSD-extendable) |

**Measured (P0, 2026-07-20 — real GTX 1080, torch 2.7.1+cu126, FP32 eager):** `sm_61` present in the wheel; raw FP32 matmul **7.4 TFLOP/s** (≈83% of peak — better than the 22–28% MFU assumption above, which stands for full training). Full-training tok/s for **K-Nano @ seq 1024, R=2, micro-batch 8, gradient-checkpointing + chunked cross-entropy: ~1,850–1,990 tok/s, peak VRAM 3.04 GB** (batch 12 → 3.96 GB) — i.e. ~150–170M tokens/day, so ~1–1.5B tokens in **~6–9 days**, matching the estimate. **Two things the table under-counted, now handled in code:** (a) with 49k vocab the full `(B·T, vocab)` logits tensor is ~1.6 GB/1.6 GB fwd/bwd at batch 8 and was the true OOM cause — solved by a chunked-CE path (`cfg.loss_chunks`); (b) gradient checkpointing (`cfg.grad_checkpoint`) is required, not optional, to fit. With both, Nano fits 8 GB with comfortable headroom even while the desktop holds ~0.5 GB.

Design implications: (1) the ablation currency is the **overnight K-S run** — the eval suite must discriminate at that scale; (2) the **~1.5–2B tokens/week** ceiling this table implies is *the* reason Nano (~150M) is the largest locally-trainable model and Mini (~500M) is the cloud item; (3) WSD schedule so any run can be paused/branched/extended without restarting; (4) checkpoint + full RNG/dataloader state every ~30 min; (5) the local program (science + Nano) flexes token budgets to fit the ~1-week window (doc 07 ledger).

## System-RAM discipline (16 GB)

- **Never hold a dataset in RAM.** Pre-tokenize once → `uint16`/`uint32` flat `.bin` shards → `np.memmap` in the dataloader (nanoGPT pattern). 3B tokens ≈ 6 GB on disk, ~0 RAM.
- **WSL2 cap:** `C:\Users\<you>\.wslconfig` (e.g. `C:\Users\mosaa\.wslconfig`) → `[wsl2] memory=10GB swap=16GB` — leaves Windows ~6 GB. (Default WSL2 grabs 50% and fights the browser.)
- Tokenizer training: stream a ≤2 GB sample; never the full corpus.
- Dataloader: `num_workers=2`, `pin_memory=True`, prefetch 2 — the 4-core CPU saturates quickly.

## Where each stage runs

| Stage | Where | Why |
|---|---|---|
| Data download + shard prep | WSL2 | HF `datasets` streaming friendlier on Linux |
| Training | Either (same wheels); start WSL2 | tooling parity; native Windows is the fallback if WSL RAM pressure bites |
| Evals (lm-eval-harness) | WSL2 | ecosystem |
| Kestrel-Nano train + infer + Roost | The GTX 1080 PC (FP32 / llama.cpp) | ~150M fits local training end-to-end |
| Kestrel-Mini pretrain | RunPod pod (bf16, compiled) | the one thing 8 GB cannot do; ≤$25 |
| Kestrel-Mini inference / demos / Roost | The GTX 1080 PC (llama.cpp / PyTorch) | Q4 ≈ 300 MB; slot-only grads fit in 8 GB |
