"""Kestrel model configuration and presets.

Presets mirror docs/04-architecture-spec.md. Non-embedding / total parameter
counts for any preset can be printed with:  python -m kestrel.config
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class KestrelConfig:
    # vocabulary / embedding
    vocab_size: int = 49152
    d_model: int = 384
    tie_embeddings: bool = True

    # macro layout: entry -> (core x loops) -> exit
    n_entry: int = 3
    n_core: int = 6
    n_exit: int = 3

    # every `attn_every`-th unique block is full softmax attention (3:1 hybrid);
    # set larger than the block count to build a pure-GLA model.
    attn_every: int = 4

    # attention blocks
    n_heads: int = 6
    n_kv_heads: int = 2
    rope_theta: float = 10_000.0

    # GLA blocks
    conv_kernel: int = 4          # short depthwise causal conv before projections
    # Chunked-scan block length. Measured on the GTX 1080 (Phase B, docs/08 §3):
    # the scan is loop/launch-bound at small chunks, so 64 -> 256 is ~2.5x on the
    # isolated scan and ~1.12x end-to-end, at +0.16 GB and ~1e-3 max abs /
    # ~1e-5 RELATIVE error (bf16 eps is 7.8e-3). Larger (512+) regresses as the (B,H,n,C,C) score
    # tensor starts to dominate bandwidth.
    chunk_size: int = 256
    gate_bias_range: Tuple[float, float] = (-6.0, -2.0)  # per-head decay-spectrum init

    # MLP
    d_ff: int = 1024

    # looped core
    r_max: int = 3
    r_default: int = 2
    r_train_probs: Tuple[float, ...] = (0.25, 0.50, 0.25)  # p(R=1), p(R=2), p(R=3)

    # product-key memory: global unique-block indices that carry a PKM site
    pkm_sites: Tuple[int, ...] = (8,)
    pkm_n_keys: int = 64          # slots = n_keys ** 2
    pkm_d_key: int = 128          # query dim (two halves of pkm_d_key // 2)
    pkm_topk: int = 8

    # heads / losses
    mtp: bool = False             # multi-token-prediction aux head (ablation A6)
    mtp_weight: float = 0.2
    z_loss: float = 1e-4
    # split the (B*T, vocab) logits+loss into this many token-chunks during
    # training so the full-vocab logits tensor never materializes at once
    # (the binding 8 GB constraint at 49k vocab). >1 activates the chunked path.
    loss_chunks: int = 1

    dropout: float = 0.0

    # training-time memory saver: re-run each block in backward instead of
    # storing its activations (torch.utils.checkpoint). Only used when no
    # streaming state is carried (i.e. standard independent-sequence pretraining).
    grad_checkpoint: bool = False
    # PARTIAL checkpointing: only checkpoint blocks whose index is a multiple of
    # this. 1 = every block (max memory saving, max recompute); 2 = half the
    # blocks; higher = less recompute but more VRAM. Lets a card with spare VRAM
    # buy back part of checkpointing's ~25-30% recompute tax.
    grad_checkpoint_every: int = 1

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    @property
    def n_blocks(self) -> int:
        return self.n_entry + self.n_core + self.n_exit

    def is_attention_block(self, i: int) -> bool:
        return (i + 1) % self.attn_every == 0

    @property
    def effective_depth(self) -> int:
        return self.n_entry + self.n_core * self.r_default + self.n_exit


# ---------------------------------------------------------------- presets

def kestrel_test() -> KestrelConfig:
    """Tiny CPU-testable config (smoke tests / every-commit correctness)."""
    return KestrelConfig(
        vocab_size=2048, d_model=128, n_entry=1, n_core=2, n_exit=1,
        n_heads=4, n_kv_heads=2, d_ff=256,
        pkm_sites=(2,), pkm_n_keys=32, pkm_d_key=64, pkm_topk=4,
        r_max=2, r_default=2, r_train_probs=(0.5, 0.5),
    )


def kestrel_s() -> KestrelConfig:
    """~23M non-embedding. The overnight-ablation currency."""
    return KestrelConfig()  # defaults above are K-S


def kestrel_m() -> KestrelConfig:
    """~63M non-embedding. Scale-confirmation rung."""
    return KestrelConfig(
        d_model=512, n_entry=4, n_core=10, n_exit=4,
        n_heads=8, n_kv_heads=2, d_ff=1408,
        pkm_sites=(10,), pkm_n_keys=96,
    )


def kestrel_nano() -> KestrelConfig:
    """~125M non-embedding / ~156M total. The fully-local hero (trains on the
    GTX 1080 in FP32, ~1-1.5B tokens in ~1 week; see docs/03, docs/07)."""
    return KestrelConfig(
        d_model=640, n_entry=5, n_core=10, n_exit=5,
        n_heads=10, n_kv_heads=2, d_ff=1792,
        pkm_sites=(7, 14), pkm_n_keys=128,       # 2 x 16k slots
        loss_chunks=8,                            # tame the 49k-vocab logits on 8 GB
    )


def kestrel_mini() -> KestrelConfig:
    """~500M total. The cloud-pretrained stretch (RunPod, <=$25, ~3-4B tokens,
    bf16). Local training does NOT fit 8 GB / the FP32 token ceiling; inference
    and Roost slot-finetuning DO run on the 1080."""
    return KestrelConfig(
        d_model=1024, n_entry=6, n_core=13, n_exit=6,
        n_heads=16, n_kv_heads=4, d_ff=2816,
        pkm_sites=(8, 17), pkm_n_keys=220,       # ~2 x 48k slots
        loss_chunks=8,
    )


PRESETS = {
    "test": kestrel_test,
    "s": kestrel_s,
    "m": kestrel_m,
    "nano": kestrel_nano,
    "mini": kestrel_mini,
}


if __name__ == "__main__":
    import torch
    from kestrel.model import KestrelModel

    for name, fn in PRESETS.items():
        cfg = fn()
        with torch.device("meta"):
            model = KestrelModel(cfg)
        total = sum(p.numel() for p in model.parameters())
        emb = model.embed.weight.numel()
        print(f"K-{name:<4s}  total={total/1e6:7.1f}M  non-emb={(total-emb)/1e6:7.1f}M  "
              f"eff-depth@R{cfg.r_default}={cfg.effective_depth}")
