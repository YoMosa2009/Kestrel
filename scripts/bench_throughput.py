"""P0 throughput micro-benchmark (docs/07 P0, docs/03).

Measures REAL forward+backward tok/s on this machine for each preset, replacing
the doc's (estimate) numbers. Also prints peak VRAM so we can confirm K-Nano
fits the 8 GB FP32 budget before committing to the multi-day run.

Usage:
    python -m scripts.bench_throughput                 # all presets, autodetect device
    python -m scripts.bench_throughput --presets s nano --seq 1024 --batch 8
"""

from __future__ import annotations

import argparse
import time

import torch

from kestrel.config import PRESETS
from kestrel.model import KestrelModel


def human(n: float) -> str:
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}T"


def bench_one(name, cfg, device, seq, batch, r, warmup, iters, grad_checkpoint=False):
    torch.manual_seed(1337)
    if grad_checkpoint:
        cfg.grad_checkpoint = True
    model = KestrelModel(cfg).to(device).train()
    n_params = sum(p.numel() for p in model.parameters())

    muon_adam = None  # optimizer step excluded; we measure the fwd+bwd core
    idx = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    tgt = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)

    def one_step():
        model.zero_grad(set_to_none=True)
        _, loss = model(idx, targets=tgt, n_loops=r)
        loss.backward()
        return loss

    for _ in range(warmup):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    last = None
    for _ in range(iters):
        last = one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    toks = batch * seq * iters
    tok_s = toks / dt
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    print(f"  K-{name:<5s} params={human(n_params):>7s}  R={r}  "
          f"tok/s={tok_s:8.0f}  tokens/day={human(tok_s*86400):>7s}  "
          f"peakVRAM={peak:4.2f}GB  loss0={last.item():.3f}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return tok_s, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", nargs="+", default=["s", "m", "nano"])
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--r", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"torch {torch.__version__} | device {device} "
          f"| arch_list={torch.cuda.get_arch_list() if device.type=='cuda' else 'cpu'}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"seq={args.seq} batch={args.batch} R={args.r} "
          f"(tokens/step={args.batch*args.seq})\n" + "-" * 78)

    for name in args.presets:
        try:
            bench_one(name, PRESETS[name](), device, args.seq, args.batch,
                      args.r, args.warmup, args.iters, args.grad_checkpoint)
        except RuntimeError as e:
            print(f"  K-{name:<5s} FAILED: {str(e)[:80]}")
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
