"""Kestrel v0.1 reference implementation.

Design contract (docs/03, docs/04): pure PyTorch, FP32, eager-mode, matmul-heavy.
No Triton, no FlashAttention, no custom CUDA — everything here runs on a Pascal
GTX 1080 and on CPU.

Blocks:
  * GLAMixer  — gated linear attention with per-head/per-token decay, short
                depthwise conv, chunked O(T) scan with carried state.
  * AttnMixer — GQA softmax attention with QK-RMSNorm + RoPE (the 1-in-4 blocks).
  * SwiGLU MLP; optional ProductKeyMemory site per block.
  * Looped core: the core block group re-runs R times (stochastic in training).

Streaming state: dict mapping "<block-key>" -> per-layer state. GLA state is
exact across segment boundaries (verified in smoke_test.py on a pure-GLA
config). Attention blocks recompute within the current segment only in v0.1
(no cross-segment KV cache yet — M2 work).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .config import KestrelConfig

StateDict = Dict[str, Dict[str, torch.Tensor]]


# ---------------------------------------------------------------- primitives

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def rms_norm_headdim(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Parameter-free RMS norm over the last (head) dimension — QK-norm."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def build_rope_cache(t: int, dim: int, theta: float, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    pos = torch.arange(t, device=device).float()
    freqs = torch.outer(pos, inv)                        # (T, dim/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D) with D even
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


# ---------------------------------------------------------------- GLA scan

def gla_chunked_scan(
    q: torch.Tensor,           # (B, H, T, D)
    k: torch.Tensor,           # (B, H, T, D)  (already decay-scaled by caller)
    v: torch.Tensor,           # (B, H, T, D)
    log_g: torch.Tensor,       # (B, H, T), every entry <= 0
    s0: Optional[torch.Tensor] = None,   # (B, H, D, D)
    chunk: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Recurrence  S_t = g_t * S_{t-1} + k_t^T v_t ;  o_t = q_t S_t.

    Chunked parallel form. All decay exponents are differences of a running
    cumulative log-sum along time and are <= 0 by construction, so every
    exp() here is bounded in (0, 1] — FP32-safe with no rescaling tricks.
    Cost: two (C x C) matmuls per chunk (cuBLAS-friendly) + a short python
    loop over chunks for the carried state (BPTT flows through it).
    """
    B, H, T, D = q.shape
    pad = (chunk - T % chunk) % chunk
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        log_g = F.pad(log_g, (0, pad))          # pad with 0 -> g=1, k=0: no-ops
    n = q.shape[2] // chunk

    qc = q.view(B, H, n, chunk, D)
    kc = k.view(B, H, n, chunk, D)
    vc = v.view(B, H, n, chunk, D)
    lg = log_g.view(B, H, n, chunk)
    lcum = lg.cumsum(dim=-1)                     # L_i, per chunk

    # intra-chunk: A_ij = (q_i . k_j) * exp(L_i - L_j)  for j <= i
    ldiff = lcum.unsqueeze(-1) - lcum.unsqueeze(-2)          # (B,H,n,C,C)
    causal = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    decay = torch.where(causal, ldiff, torch.full_like(ldiff, float("-inf"))).exp()
    o = (qc @ kc.transpose(-1, -2) * decay) @ vc             # (B,H,n,C,D)

    # cross-chunk: sequential state carry
    s = s0 if s0 is not None else q.new_zeros(B, H, D, D)
    cross = []
    for i in range(n):
        cross.append((qc[:, :, i] * lcum[:, :, i].exp().unsqueeze(-1)) @ s)
        k_tail = kc[:, :, i] * (lcum[:, :, i, -1:] - lcum[:, :, i]).exp().unsqueeze(-1)
        s = lcum[:, :, i, -1].exp()[..., None, None] * s + k_tail.transpose(-1, -2) @ vc[:, :, i]
    o = o + torch.stack(cross, dim=2)

    o = o.reshape(B, H, -1, D)[:, :, :T]
    return o, s


class GLAMixer(nn.Module):
    """Gated linear attention block mixer (the 75% of layers). NoPE."""

    def __init__(self, cfg: KestrelConfig):
        super().__init__()
        d, h, hd = cfg.d_model, cfg.n_heads, cfg.head_dim
        self.cfg = cfg
        self.h, self.hd = h, hd
        self.conv = nn.Conv1d(d, d, cfg.conv_kernel, groups=d, bias=False)
        self.wqkv = nn.Linear(d, 3 * d, bias=False)
        self.wg = nn.Linear(d, h, bias=True)      # per-head decay logits
        self.wog = nn.Linear(d, d, bias=False)    # output gate
        self.wo = nn.Linear(d, d, bias=False)
        self.out_norm = RMSNorm(hd)
        self.q_scale = nn.Parameter(torch.ones(h))
        # decay-spectrum init: heads spread from long-memory to short-memory
        lo, hi = cfg.gate_bias_range
        with torch.no_grad():
            if not self.wg.bias.is_meta:   # skip data-dependent init on meta device
                self.wg.bias.copy_(torch.linspace(lo, hi, h))
                self.wg.weight.mul_(0.1)
                # start the causal conv as identity (last tap = 1) so the
                # block passes signal through cleanly at init
                self.conv.weight.zero_()
                self.conv.weight[:, 0, -1] = 1.0

    def forward(self, x, state=None):
        B, T, d = x.shape
        k_conv = self.cfg.conv_kernel
        # causal depthwise conv with streaming cache of the last k-1 inputs
        xt = x.transpose(1, 2)                                  # (B, d, T)
        cache = state["conv"] if state is not None else xt.new_zeros(B, d, k_conv - 1)
        xt_pad = torch.cat([cache, xt], dim=2)
        new_conv_cache = xt_pad[:, :, -(k_conv - 1):].detach()
        xc = F.silu(self.conv(xt_pad)).transpose(1, 2)          # (B, T, d)

        q, k, v = self.wqkv(xc).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.hd).transpose(1, 2)
        k = k.view(B, T, self.h, self.hd).transpose(1, 2)
        v = v.view(B, T, self.h, self.hd).transpose(1, 2)
        q = F.normalize(q, dim=-1) * self.q_scale.view(1, -1, 1, 1)
        k = F.normalize(k, dim=-1)

        log_g = -F.softplus(self.wg(xc)).transpose(1, 2)        # (B, H, T), <= 0
        # EMA input scaling k <- (1 - g) k keeps ||S|| bounded in FP32
        k = k * (-torch.expm1(log_g)).unsqueeze(-1)

        s0 = state["S"] if state is not None else None
        o, s_new = gla_chunked_scan(q, k, v, log_g, s0, self.cfg.chunk_size)

        o = self.out_norm(o).transpose(1, 2).reshape(B, T, d)
        o = o * F.silu(self.wog(xc))
        return self.wo(o), {"S": s_new.detach(), "conv": new_conv_cache}


class AttnMixer(nn.Module):
    """GQA softmax attention with QK-RMSNorm + RoPE (the 1-in-4 blocks)."""

    def __init__(self, cfg: KestrelConfig):
        super().__init__()
        d, hd = cfg.d_model, cfg.head_dim
        self.cfg = cfg
        self.h, self.kvh, self.hd = cfg.n_heads, cfg.n_kv_heads, hd
        self.wq = nn.Linear(d, cfg.n_heads * hd, bias=False)
        self.wk = nn.Linear(d, cfg.n_kv_heads * hd, bias=False)
        self.wv = nn.Linear(d, cfg.n_kv_heads * hd, bias=False)
        self.wo = nn.Linear(cfg.n_heads * hd, d, bias=False)

    def forward(self, x, state=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.kvh, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.kvh, self.hd).transpose(1, 2)
        q, k = rms_norm_headdim(q), rms_norm_headdim(k)
        cos, sin = build_rope_cache(T, self.hd, self.cfg.rope_theta, x.device, x.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.kvh != self.h:
            rep = self.h // self.kvh
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, T, -1)
        return self.wo(o), None                    # v0.1: no cross-segment KV cache


class SwiGLU(nn.Module):
    def __init__(self, cfg: KestrelConfig):
        super().__init__()
        self.w_up = nn.Linear(cfg.d_model, 2 * cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        u, g = self.w_up(x).chunk(2, dim=-1)
        return self.w_down(u * F.silu(g))


class ProductKeyMemory(nn.Module):
    """Sparse key-value memory (Lample et al.; Meta 'Memory Layers at Scale').

    slots = n_keys^2 addressed by two half-queries. Pure topk + embedding
    gathers. The values table is the designated plastic tissue for the
    continual-learning consolidation job (docs/05 §5) — sparse slot updates,
    dense weights frozen.
    """

    def __init__(self, cfg: KestrelConfig):
        super().__init__()
        self.n_keys, self.topk = cfg.pkm_n_keys, cfg.pkm_topk
        half = cfg.pkm_d_key // 2
        self.wq = nn.Linear(cfg.d_model, cfg.pkm_d_key, bias=False)
        self.q_norm = nn.LayerNorm(cfg.pkm_d_key)
        self.keys = nn.Parameter(torch.randn(2, cfg.pkm_n_keys, half) * 0.02)
        self.values = nn.Embedding(cfg.pkm_n_keys ** 2, cfg.d_model)
        nn.init.normal_(self.values.weight, std=0.02)
        self.gate = nn.Parameter(torch.tensor(0.05))   # born near-inert

    def forward(self, x):
        B, T, d = x.shape
        q = self.q_norm(self.wq(x)).view(B * T, 2, -1)          # two half-queries
        kh = min(self.topk, self.n_keys)
        s1 = q[:, 0] @ self.keys[0].t()                          # (BT, n_keys)
        s2 = q[:, 1] @ self.keys[1].t()
        v1, i1 = s1.topk(kh, dim=-1)
        v2, i2 = s2.topk(kh, dim=-1)
        cand = (v1.unsqueeze(-1) + v2.unsqueeze(-2)).flatten(1)  # (BT, kh*kh)
        score, flat = cand.topk(self.topk, dim=-1)
        rows = i1.gather(1, flat // kh)
        cols = i2.gather(1, flat % kh)
        slots = rows * self.n_keys + cols                        # (BT, topk)
        w = F.softmax(score, dim=-1)
        out = (self.values(slots) * w.unsqueeze(-1)).sum(dim=1)
        return (out * torch.tanh(self.gate)).view(B, T, d)


# ---------------------------------------------------------------- chunked loss

def _chunk_ce(h_c: torch.Tensor, W: torch.Tensor, t_c: torch.Tensor,
              z_loss: float) -> torch.Tensor:
    """Summed cross-entropy (+ z-loss) for one token-chunk. Materializes only
    this chunk's logits, so peak logit memory is (chunk_rows x vocab)."""
    logits = h_c @ W.t()
    l = F.cross_entropy(logits.float(), t_c, reduction="sum")
    if z_loss > 0:
        l = l + z_loss * logits.float().logsumexp(-1).pow(2).sum()
    return l


def chunked_cross_entropy(h: torch.Tensor, W: torch.Tensor, targets: torch.Tensor,
                          z_loss: float, n_chunks: int) -> torch.Tensor:
    """Mean CE over (B,T) computed in `n_chunks` token-slices, each slice's
    logits recomputed in backward (checkpointed) so the full (B*T, vocab) logits
    tensor never exists at once — the key 8 GB enabler at 49k vocab."""
    N = h.shape[0] * h.shape[1]
    hf = h.reshape(N, h.shape[-1])
    tf = targets.reshape(N)
    total = hf.new_zeros(())
    for h_c, t_c in zip(hf.chunk(n_chunks, 0), tf.chunk(n_chunks, 0)):
        if torch.is_grad_enabled():
            total = total + torch.utils.checkpoint.checkpoint(
                _chunk_ce, h_c, W, t_c, z_loss, use_reentrant=False)
        else:
            total = total + _chunk_ce(h_c, W, t_c, z_loss)   # eval: no checkpoint needed
    return total / N


# ---------------------------------------------------------------- blocks / model

class Block(nn.Module):
    def __init__(self, cfg: KestrelConfig, index: int):
        super().__init__()
        self.index = index
        self.is_attn = cfg.is_attention_block(index)
        self.norm1 = RMSNorm(cfg.d_model)
        self.mixer = AttnMixer(cfg) if self.is_attn else GLAMixer(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)
        self.pkm = ProductKeyMemory(cfg) if index in cfg.pkm_sites else None
        if self.pkm is not None:
            self.norm3 = RMSNorm(cfg.d_model)

    def forward(self, x, state=None):
        y, new_state = self.mixer(self.norm1(x), state)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        if self.pkm is not None:
            x = x + self.pkm(self.norm3(x))
        return x, new_state


class KestrelModel(nn.Module):
    def __init__(self, cfg: KestrelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.entry = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_entry))
        self.core = nn.ModuleList(
            Block(cfg, cfg.n_entry + i) for i in range(cfg.n_core))
        self.exit = nn.ModuleList(
            Block(cfg, cfg.n_entry + cfg.n_core + i) for i in range(cfg.n_exit))
        self.norm_f = RMSNorm(cfg.d_model)
        if cfg.mtp:
            self.mtp_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            self.mtp_norm = RMSNorm(cfg.d_model)
        self.apply(self._init)
        # depth-scaled init on residual-out projections
        scale = 1.0 / math.sqrt(2 * cfg.effective_depth)
        for m in self.modules():
            if isinstance(m, (GLAMixer, AttnMixer)):
                m.wo.weight.data.mul_(scale)
            if isinstance(m, SwiGLU):
                m.w_down.weight.data.mul_(scale)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            if m.weight.is_meta:
                return                       # meta device: param-counting only, no data init
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None and m.bias.data.abs().max() == 0:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        # nn.Conv1d deliberately untouched: GLAMixer sets identity init

    def _run(self, blocks, x, state: Optional[StateDict], prefix: str):
        # Gradient checkpointing is only safe when we are NOT carrying streaming
        # state (state is None): standard independent-sequence pretraining. When
        # state is carried, we keep activations so the state graph is intact.
        ckpt = self.cfg.grad_checkpoint and self.training and state is None
        for i, blk in enumerate(blocks):
            key = f"{prefix}{i}"
            blk_state = state.get(key) if state is not None else None
            if ckpt:
                x, new_state = torch.utils.checkpoint.checkpoint(
                    blk, x, blk_state, use_reentrant=False)
            else:
                x, new_state = blk(x, blk_state)
            if state is not None and new_state is not None:
                state[key] = new_state
        return x

    def sample_loops(self) -> int:
        probs = torch.tensor(self.cfg.r_train_probs)
        return int(torch.multinomial(probs, 1).item()) + 1

    def forward(
        self,
        idx: torch.Tensor,                       # (B, T) int64
        targets: Optional[torch.Tensor] = None,  # (B, T) int64
        n_loops: Optional[int] = None,
        state: Optional[StateDict] = None,
        return_state: bool = False,
    ):
        if n_loops is None:
            n_loops = self.sample_loops() if self.training else self.cfg.r_default
        carry: Optional[StateDict] = dict(state) if state is not None else (
            {} if return_state else None)

        x = self.embed(idx)
        x = self._run(self.entry, x, carry, "e")
        for r in range(n_loops):
            x = self._run(self.core, x, carry, f"c{r}.")
        x = self._run(self.exit, x, carry, "x")
        x = self.norm_f(x)

        # Large-vocab loss uses the chunked path so the full (B*T, vocab) logits
        # tensor never materializes -- in EVAL too, not just training: the full
        # path's 1.5 GB logits spike OOM'd the 8 GB card during eval probes.
        # When targets are given the logits aren't returned (all loss callers
        # ignore them); inference (no targets) still gets full logits below.
        chunked = (targets is not None
                   and self.cfg.loss_chunks > 1 and not self.cfg.mtp)
        if chunked:
            loss = chunked_cross_entropy(
                x, self.embed.weight, targets, self.cfg.z_loss, self.cfg.loss_chunks)
            if return_state:
                return None, loss, carry
            return None, loss

        logits = x @ self.embed.weight.t()

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            if self.cfg.z_loss > 0:
                loss = loss + self.cfg.z_loss * logits.logsumexp(-1).pow(2).mean()
            if self.cfg.mtp and targets.size(1) > 2:
                h2 = self.mtp_norm(self.mtp_proj(x[:, :-2]))
                logits2 = h2 @ self.embed.weight.t()
                loss = loss + self.cfg.mtp_weight * F.cross_entropy(
                    logits2.reshape(-1, logits2.size(-1)),
                    targets[:, 2:].reshape(-1))

        if return_state:
            return logits, loss, carry
        return logits, loss

    @torch.no_grad()
    def param_report(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        emb = self.embed.weight.numel()
        pkm = sum(p.numel() for m in self.modules()
                  if isinstance(m, ProductKeyMemory) for p in m.parameters())
        return (f"total {total/1e6:.2f}M | non-emb {(total-emb)/1e6:.2f}M | "
                f"pkm {pkm/1e6:.2f}M | eff-depth@R{self.cfg.r_default} "
                f"{self.cfg.effective_depth}")
