"""Full training-step benchmark: fwd+bwd x accum PLUS the optimizer step.

`scripts/bench_throughput.py` deliberately measures only the fwd+bwd core, so
every tok/s figure in the docs excludes Muon's Newton-Schulz and AdamW. This
script measures what a real step costs, and A/B's the knobs that matter on
Ampere so a claimed speedup is a measured one.

Usage
-----
    python -m scripts.bench_step                       # A/B every knob
    python -m scripts.bench_step --batch 24 --accum 4  # size it to your card
    python -m scripts.bench_step --only baseline       # one config

Needs the GPU to itself: anything else holding VRAM (an Ollama model pinned with
keep_alive, a game, a browser with hardware acceleration) will OOM the larger
batches or silently distort the numbers.
"""

from __future__ import annotations

import argparse
import time

import torch

from kestrel.config import PRESETS
from kestrel.model import KestrelModel
from kestrel.optim import build_optimizers


def bench(label, args, *, tf32, fused_adam, ckpt_every=1):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)

    cfg = PRESETS[args.preset]()
    cfg.grad_checkpoint = not args.no_checkpoint
    cfg.grad_checkpoint_every = ckpt_every
    model = KestrelModel(cfg).to(dev).train()
    muon, adamw = build_optimizers(model, fused=fused_adam)

    x = torch.randint(0, cfg.vocab_size, (args.batch, args.seq), device=dev)
    y = torch.randint(0, cfg.vocab_size, (args.batch, args.seq), device=dev)
    ac = torch.autocast("cuda", dtype=torch.bfloat16) if args.bf16 else None

    def step():
        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            if ac is not None:
                with ac:
                    _, loss = model(x, targets=y, n_loops=cfg.r_default)
            else:
                _, loss = model(x, targets=y, n_loops=cfg.r_default)
            (loss / args.accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        muon.step()
        adamw.step()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    for _ in range(args.iters):
        step()
    torch.cuda.synchronize()
    dt = (time.time() - t) / args.iters

    tok = args.batch * args.seq * args.accum
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  {label:34s} {tok/dt:8.0f} tok/s | {dt:6.2f} s/step | peak {peak:5.2f} GB",
          flush=True)
    del model, muon, adamw
    torch.cuda.empty_cache()
    return tok / dt


CONFIGS = {
    "baseline":        dict(tf32=False, fused_adam=False),
    "tf32":            dict(tf32=True,  fused_adam=False),
    "tf32+fusedadam":  dict(tf32=True,  fused_adam=True),
    "tf32+fusedadam+ckpt2": dict(tf32=True, fused_adam=True, ckpt_every=2),
    "tf32+fusedadam+ckpt3": dict(tf32=True, fused_adam=True, ckpt_every=3),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="nano", choices=list(PRESETS))
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--fp32", dest="bf16", action="store_false")
    ap.add_argument("--no-checkpoint", action="store_true")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--only", default=None, choices=list(CONFIGS))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    free, total = torch.cuda.mem_get_info()
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name} | sm_{p.major}{p.minor} | {p.multi_processor_count} SMs | "
          f"{total/1e9:.1f} GB total")
    print(f"{args.preset}: batch {args.batch} x seq {args.seq} x accum {args.accum} "
          f"= {args.batch*args.seq*args.accum} tok/step | "
          f"{'bf16' if args.bf16 else 'fp32'} | "
          f"grad-ckpt {'off' if args.no_checkpoint else 'on'}\n")

    names = [args.only] if args.only else list(CONFIGS)
    base = None
    for n in names:
        try:
            r = bench(n, args, **CONFIGS[n])
        except torch.OutOfMemoryError:
            print(f"  {n:34s} OOM")
            torch.cuda.empty_cache()
            continue
        if base is None:
            base = r
        elif base:
            print(f"  {'':34s} {r/base:.3f}x vs {names[0]}")


if __name__ == "__main__":
    main()
