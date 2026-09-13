"""Re-profile a training step on the CURRENT card, in the CURRENT precision.

Why this exists: the Phase B conclusion "the model is matmul-bound at ~30% MFU,
so the scan was never the bottleneck" (docs/08 §3) was measured on a GTX 1080 in
FP32. bf16 tensor cores cut matmul time several-fold while leaving every
elementwise, reduction and launch-bound op untouched, so on Ampere the mix has
necessarily moved. This prints the op breakdown and the achieved MFU so the next
optimization targets whatever is actually expensive now, rather than what was
expensive on Pascal.

Usage:
    python -m scripts.profile_step --batch 24 --bf16
    python -m scripts.profile_step --batch 8 --bf16 --sort self_cuda_memory_usage
"""

from __future__ import annotations

import argparse

import torch
from torch.profiler import ProfilerActivity, profile

from kestrel.config import PRESETS
from kestrel.model import KestrelModel
from kestrel.optim import build_optimizers

# GA10x/Ampere GeForce: bf16 tensor with fp32 accumulate runs at 2x the fp32
# CUDA-core rate. Used only to turn measured seconds into an MFU percentage.
BF16_TENSOR_MULT = 2.0


def peak_tflops(bf16: bool) -> float:
    p = torch.cuda.get_device_properties(0)
    try:                      # needs pynvml; fall back to the RTX 3060 boost clock
        clk = torch.cuda.clock_rate() * 1e6
    except Exception:
        clk = 2.16e9
    fp32 = p.multi_processor_count * 128 * 2 * clk / 1e12
    return fp32 * (BF16_TENSOR_MULT if bf16 else 1.0)


def model_flops(cfg, tokens: int, ckpt: bool) -> float:
    """2*N*T forward, 4*N*T backward, +2*N*T for the checkpoint recompute."""
    blocks_eff = cfg.n_entry + cfg.n_core * cfg.r_default + cfg.n_exit
    with torch.device("meta"):
        m = KestrelModel(cfg)
    emb = m.embed.weight.numel()
    non_emb = sum(p.numel() for p in m.parameters()) - emb
    per_block = non_emb / cfg.n_blocks
    n_eff = per_block * blocks_eff + emb        # emb tied -> counts as the output matmul
    mult = 8.0 if ckpt else 6.0
    return mult * n_eff * tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="nano", choices=list(PRESETS))
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--fp32", dest="bf16", action="store_false")
    ap.add_argument("--no-checkpoint", action="store_true")
    ap.add_argument("--tf32", action="store_true", default=True)
    ap.add_argument("--rows", type=int, default=18)
    ap.add_argument("--sort", default="self_cuda_time_total")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cudnn.benchmark = True

    dev = torch.device("cuda")
    cfg = PRESETS[args.preset]()
    cfg.grad_checkpoint = not args.no_checkpoint
    model = KestrelModel(cfg).to(dev).train()
    muon, adamw = build_optimizers(model)
    x = torch.randint(0, cfg.vocab_size, (args.batch, args.seq), device=dev)
    y = torch.randint(0, cfg.vocab_size, (args.batch, args.seq), device=dev)
    ac = torch.autocast("cuda", dtype=torch.bfloat16) if args.bf16 else None

    def one():
        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)
        if ac is not None:
            with ac:
                _, loss = model(x, targets=y, n_loops=cfg.r_default)
        else:
            _, loss = model(x, targets=y, n_loops=cfg.r_default)
        loss.backward()
        muon.step()
        adamw.step()

    for _ in range(2):
        one()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        one()
        torch.cuda.synchronize()

    ka = prof.key_averages()
    total = sum(e.self_device_time_total for e in ka) or 1.0
    print(f"\n{torch.cuda.get_device_properties(0).name} | "
          f"{'bf16' if args.bf16 else 'fp32'} | tf32={args.tf32} | "
          f"batch {args.batch} x {args.seq} | "
          f"grad-ckpt {'off' if args.no_checkpoint else 'on'}")
    print(f"total CUDA time this step: {total/1e3:.1f} ms\n")
    print(f"{'op':40s} {'self CUDA ms':>13s} {'share':>7s}")
    print("-" * 63)
    for e in sorted(ka, key=lambda e: -e.self_device_time_total)[:args.rows]:
        t = e.self_device_time_total
        if t <= 0:
            continue
        print(f"{e.key[:40]:40s} {t/1e3:13.2f} {100*t/total:6.1f}%")

    tok = args.batch * args.seq
    fl = model_flops(cfg, tok, not args.no_checkpoint)
    achieved = fl / (total * 1e-6) / 1e12
    pk = peak_tflops(args.bf16)
    print(f"\nmodel FLOPs this micro-batch: {fl/1e12:.1f} TFLOP")
    print(f"achieved: {achieved:.2f} TFLOP/s | peak ~{pk:.1f} TFLOP/s | "
          f"MFU ~{100*achieved/pk:.1f}%")
    print(f"throughput: {tok/(total*1e-6):.0f} tok/s (fwd+bwd+opt, profiler overhead included)")


if __name__ == "__main__":
    main()
