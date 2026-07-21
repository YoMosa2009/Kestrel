# 04 — Kestrel Architecture Spec v0.1

Decoder-only, but three structural departures from the standard transformer: a **3:1 hybrid mixer**, a **looped core**, and **product-key memory**. Everything is pure PyTorch, FP32, eager-mode, matmul-heavy (Pascal-safe). Reference implementation: [`kestrel/model.py`](../kestrel/model.py).

## 1. Macro-layout

```
tokens → tied embedding
  → Entry blocks   (E unique blocks)
  → Core blocks    (C unique blocks) ← looped R times (R=1..4)
  → Exit blocks    (X unique blocks)
  → RMSNorm → tied unembedding → logits (+ optional MTP aux head)
```

- **Block pattern (everywhere):** every 4th block is a *full-attention* block; the rest are *gated-linear-attention* (GLA) blocks → the 3:1 hybrid ratio converged on by Qwen3-Next / Kimi Linear.
- **Looped core:** the C core blocks re-run R times on their own output. Training samples R ∈ {1,2,3} per step (p = 0.25/0.5/0.25); inference default R=2, user-dial up to 4 for hard problems ("think harder" without more parameters). Rationale: Ouro's 2–3× parameter-efficiency result, targeted specifically at knowledge *manipulation* — the complement of memory layers, which handle knowledge *storage*.
- **PKM sites:** product-key memory bolted onto 1–2 designated blocks (post-MLP residual add, gated, init-neutral).

## 2. Blocks

### 2.1 GLA block (the 75%)
Pre-RMSNorm → short depthwise causal conv (k=4, "canon"-style local mixing) → projections q,k,v (per-head dim 64) + per-head **per-token decay gate** + output gate → chunked linear-attention scan → RMSNorm → output gate (SiLU) → out-proj → residual. Then pre-RMSNorm → SwiGLU MLP → residual.

The scan maintains per-head state **S ∈ ℝ^(dk×dv)**: `S_t = g_t·S_{t-1} + k_tᵀv_t`, `o_t = q_t·S_t`.

- q,k are L2-normalized per head (stability in FP32).
- Gates: `log g_t = −softplus(a_t)`, with bias init spread per head over ≈[−6,−2] → initial decays ≈0.998…0.88, giving a **spectrum of memory timescales** per layer (short-loop syntax heads ↔ long-horizon topic heads).
- **Chunked implementation** (chunk 64): intra-chunk = masked matmul attention with log-space relative decays; cross-chunk = state matmuls. All exponents ≤ 0 by construction → numerically safe in FP32. Verified against a naive recurrent reference in the smoke test.
- **No position embeddings** on GLA blocks (recurrence is inherently positional).
- **The state is the session memory.** ~2.5 MB for K-Nano (15 GLA layers × 10 heads × 64×64 FP32; a few MB more for Mini). Serialized on session exit, restored on resume → the model "picks up where its mind left off" without replaying the transcript. This falls out of the linear-attention choice for free and is the medium-term tier of the memory hierarchy.

### 2.2 Attention block (the 25%)
Pre-RMSNorm → GQA (kv_heads = 2 at small scale), **QK-RMSNorm**, **RoPE** (θ=10⁴; positions live only here, NoPE elsewhere — SmolLM3-style), causal SDPA (efficient/math backend), out-proj → residual → SwiGLU MLP. These layers supply the precise random-access recall that linear attention lacks.

### 2.3 Product-key memory (PKM) site
Query = LayerNorm(W_q·x) split into two halves; each half scores its own table of √N sub-keys (dim 64); top-8 per half → 64 candidate slots → top-8 overall → softmax-weighted sum of value embeddings (dim d) → scalar-gated residual add (gate init 0.05 → born near-inert but trainable, learns to engage).
- N slots: 4k (K-S) → 9k (K-M) → 16k ×2 sites (K-Nano) → 48k ×2 sites (K-Mini).
- Pure `topk` + embedding gathers: no FLOP explosion, Pascal-safe, CPU-offloadable at deployment.
- **Role beyond capacity:** the PKM values are the *only* weights the continual-learning consolidation job may touch (doc 05 §5). Sparse slot updates ≈ localized plasticity → Meta's 11%-forgetting result.

### 2.4 Heads
- Tied input/output embedding (MobileLLM: essential below 1B).
- Cross-entropy + z-loss (1e-4).
- Optional **MTP aux head** (predict t+2 via a small projection, reusing the tied embedding; λ=0.2). Off by default; ablation A6. Later doubles as self-speculative decoding.

## 3. Presets

Two research proxies (K-S, K-M) for the free local science, and **two shipped models** — a fully-local hero and a cloud-pretrained stretch:

| | K-test (CPU) | K-S (proxy) | K-M (proxy) | **K-Nano (hero)** | **K-Mini (stretch)** |
|---|---|---|---|---|---|
| Role | correctness | overnight ablations | scale-confirm | 100% local, $0 | cloud-pretrained, ≤$25 |
| d_model | 128 | 384 | 512 | 640 | 1024 |
| Unique blocks (E/C/X) | 1/2/1 | 3/6/3 | 4/10/4 | 5/10/5 | 6/13/6 |
| Effective depth @R=2 | 6 | 18 | 28 | 30 | 38 |
| Heads (q/kv) | 4/2 | 6/2 | 8/2 | 10/2 | 16/4 |
| d_ff (SwiGLU) | 256 | 1024 | 1408 | 1792 | 2816 |
| PKM | 1×1k | 1×4k | 1×9k | 2×16k | 2×48k |
| Vocab | 2048 | 49152 | 49152 | 49152 | 49152 |
| Non-emb params | ~1M | ~23M | ~63M | ~125M | ~450M |
| Total params | ~1.3M | ~42M | ~88M | **~156M** | **~500M** |
| Train ctx | 128 | 1024 | 1024→2048 | 1024→2048 | 2048→4096 |
| Trains where | CPU | GTX 1080 | GTX 1080 | **GTX 1080 (FP32)** | **RunPod 4090 (bf16)** |

(Exact counts printed by `smoke_test.py` / `config.py`.) One tokenizer (49152 superword vocab) serves both shipped models so a session/memory store is portable between them. K-Nano fits local FP32 training (~4.3–4.9 GB of 8 GB — see doc 03); K-Mini's from-scratch pretraining does not fit 8 GB in any useful precision, so it is the one cloud item — but K-Mini *inference* and *Roost consolidation* run fine on the 1080 (slot-only gradients).

**Why deep-thin + loops:** MobileLLM shows depth beats width sub-1B; loops buy 1.5–2× more effective depth with zero extra parameters; GLA keeps the extra depth cheap in VRAM (no KV growth on 75% of layers). Both tiers lean deep-thin because the target domains — coding, tool-use, language — reward reasoning depth over width/knowledge.

**Deployment math:**
- **K-Nano (~156M):** Q4 GGUF ≈ **95–120 MB**. Runs at ~60–110 tok/s on the 1080, briskly on CPU-only laptops, and comfortably on phones. Effective compute @R=2 ≈ 30-layer ~250M-class, @R=4 ≈ ~450M-class.
- **K-Mini (~500M):** ~0.1B of the params are PKM values that quantize to int4/int8 gracefully and can spill to system RAM. Q4 GGUF ≈ **300–340 MB** → ~40–80 tok/s on the 1080, usable on CPU. Effective compute @R=2 ≈ 38-layer ~1B-class, @R=4 ≈ ~1.8B-class; per Ouro, a looped 500M competes with 2–3× larger dense models on manipulation-heavy tasks (exactly coding/tool-use).

## 4. Init & scaling discipline

- Trunc-normal σ=0.02; out-projections scaled ×(2·L_eff)^(−1/2); PKM gates & MTP init 0.
- LR transfer: v0.1 uses standard parametrization with width-corrected LR (∝1/d for matrices) tuned at K-S and transferred; full μP is milestone M3 stretch (the machinery exists in Muon-community repos).
- FP32 end-to-end; FP16 allowed only for *serialized* states/checkpoints on disk.

## 5. The memory hierarchy (the Mitchell answer, consolidated view)

| Tier | Mechanism | Persistence | Plasticity |
|---|---|---|---|
| Working | attention over context | one forward pass | none |
| Session | GLA states (MBs), serialize/resume | across restarts | automatic (recurrence) |
| Episodic | external embedding store, retrieved into context | disk, user-ownable | append-only |
| Semantic | PKM slots | permanent | **nightly sparse consolidation, eval-gated** |
| Core competence | dense weights | permanent | **frozen after training — by policy** |

Failure containment: anything the consolidation job learns badly is confined to memory slots and reversible (slot snapshots); dense reasoning circuitry cannot be corrupted by deployment-time learning.
