"""Kestrel trainer (docs/05, docs/07 P2).

One codebase, resumable everything: Muon(matrices)+AdamW(rest), WSD schedule,
gradient accumulation + gradient checkpointing, memmap dataloader, JSONL metrics,
Tier-0 probes. FP32-eager is the local (Pascal) mode; --bf16 flips it to the
cloud mode for the Mini pod (same code).

Examples
--------
# machinery shakedown: overfit a fixed synthetic batch (no data needed)
python -m kestrel.train --preset test --synthetic --steps 200 --out experiments/shakedown

# real run on prepared shards
python -m kestrel.train --preset nano --data-dir data --seq 1024 \
    --batch 8 --accum 16 --steps 60000 --out experiments/nano
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict

import torch

from kestrel.config import PRESETS, KestrelConfig
from kestrel.model import KestrelModel
from kestrel.optim import build_optimizers
from kestrel.data import make_loaders, CurriculumLoader, make_domain_val_loaders


# ----------------------------------------------------------------- schedule

def wsd_lr_mult(step: int, total: int, warmup: int, decay_frac: float,
                floor: float = 0.0) -> float:
    """Warmup-Stable-Decay. Flat plateau, then 1-sqrt decay over the final
    `decay_frac` of steps. Multiplies each group's base LR."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    decay_start = int(total * (1.0 - decay_frac))
    if step < decay_start:
        return 1.0
    p = (step - decay_start) / max(1, total - decay_start)
    return floor + (1.0 - floor) * (1.0 - math.sqrt(min(1.0, p)))


# ----------------------------------------------------------------- checkpoint

def _core(model):
    """Unwrap a torch.compile()'d model so checkpoints have clean keys
    (no '_orig_mod.' prefix) and resume works with or without --compile."""
    return getattr(model, "_orig_mod", model)


def save_ckpt(path, model, muon, adamw, loader, step, tokens_seen, cfg, args):
    tmp = path + ".tmp"
    torch.save({
        "model": _core(model).state_dict(),
        "muon": muon.state_dict(),
        "adamw": adamw.state_dict(),
        "loader": loader.state_dict() if loader is not None else None,
        "step": step,
        "tokens_seen": tokens_seen,
        "cfg": asdict(cfg),
        "args": vars(args),
        "torch_rng": torch.get_rng_state(),
    }, tmp)
    os.replace(tmp, path)  # atomic: a crash mid-write never corrupts the ckpt


def load_ckpt(path, model, muon, adamw, loader, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    _core(model).load_state_dict(ck["model"])
    muon.load_state_dict(ck["muon"])
    adamw.load_state_dict(ck["adamw"])
    if loader is not None and ck.get("loader") is not None:
        loader.load_state_dict(ck["loader"])
    if ck.get("torch_rng") is not None:
        torch.set_rng_state(ck["torch_rng"].cpu())
    return ck["step"], ck["tokens_seen"]


# ----------------------------------------------------------------- probes

@torch.no_grad()
def eval_val_loss(model, val_loader, iters, r):
    if val_loader is None:
        return None
    model.eval()
    tot = 0.0
    for _ in range(iters):
        x, y = val_loader.get_batch()
        _, loss = model(x, targets=y, n_loops=r)
        tot += loss.item()
    model.train()
    return tot / iters


@torch.no_grad()
def loop_gain(model, val_loader, iters, r_max):
    """Val loss at R=1..r_max on the SAME batches. A descending curve means the
    looped core is actually using its iterations (docs/06 Tier-0)."""
    if val_loader is None:
        return {}
    model.eval()
    # fix a small set of batches so the R comparison is apples-to-apples
    batches = [val_loader.get_batch() for _ in range(iters)]
    out = {}
    for r in range(1, r_max + 1):
        tot = sum(model(x, targets=y, n_loops=r)[1].item() for x, y in batches)
        out[f"R{r}"] = tot / len(batches)
    model.train()
    return out


@torch.no_grad()
def pkm_health(model):
    """Fraction of live PKM sites' gate magnitude (are the memory layers engaging?)."""
    from kestrel.model import ProductKeyMemory
    gates = [float(torch.tanh(m.gate).abs()) for m in model.modules()
             if isinstance(m, ProductKeyMemory)]
    return {"pkm_gate_mean": sum(gates) / len(gates)} if gates else {}


# ----------------------------------------------------------------- train loop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="nano", choices=list(PRESETS))
    ap.add_argument("--data-dir", default=None, help="dir with train*.bin / val*.bin")
    ap.add_argument("--curriculum", action="store_true",
                    help="stage the per-domain mixture over training + premium anneal "
                         "(needs per-domain shards from prepare_data_v2)")
    ap.add_argument("--synthetic", action="store_true",
                    help="overfit one fixed random batch (trainer shakedown, no data)")
    ap.add_argument("--out", default="experiments/run")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8, help="micro-batch size")
    ap.add_argument("--accum", type=int, default=16, help="grad-accum steps")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--decay-frac", type=float, default=0.15)
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--adamw-lr", type=float, default=3e-3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="recompute activations to save VRAM (needed on 8 GB; skip on 24 GB pod for speed)")
    ap.add_argument("--loss-chunks", type=int, default=-1,
                    help="override cfg.loss_chunks (-1=preset; on a 24 GB pod use 1-2 for speed)")
    ap.add_argument("--bf16", action="store_true",
                    help="cloud mode (Ampere+ pod): bf16 autocast + FlashAttention. Local Pascal stays FP32.")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model (Ampere+/pod only; best-effort, may graph-break on the GLA scan)")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--ckpt-secs", type=int, default=1800, help="also checkpoint every N seconds")
    ap.add_argument("--eval-iters", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "ckpt.pt")
    log_path = os.path.join(args.out, "metrics.jsonl")

    cfg: KestrelConfig = PRESETS[args.preset]()
    if args.grad_checkpoint:
        cfg.grad_checkpoint = True
    if args.loss_chunks >= 0:
        cfg.loss_chunks = args.loss_chunks
    model = KestrelModel(cfg).to(device)
    print(model.param_report())
    if args.compile:
        model = torch.compile(model)   # cloud only; Pascal has no Triton
        print("torch.compile enabled")

    muon, adamw = build_optimizers(_core(model), muon_lr=args.muon_lr, adamw_lr=args.adamw_lr)

    # ---- data ----
    train_loader = val_loader = None
    dom_val = {}
    synth = None
    if args.synthetic:
        g = torch.Generator().manual_seed(args.seed)
        synth = (torch.randint(0, cfg.vocab_size, (args.batch, args.seq), generator=g).to(device),
                 torch.randint(0, cfg.vocab_size, (args.batch, args.seq), generator=g).to(device))
    else:
        assert args.data_dir, "--data-dir required unless --synthetic"
        if args.curriculum:
            train_loader = CurriculumLoader(args.data_dir, args.seq, args.batch,
                                            device, args.seed)
            dom_val = make_domain_val_loaders(args.data_dir, args.seq, args.batch, device)
            val_loader = None
            print(f"curriculum: {len(train_loader.loaders)} domains "
                  f"({train_loader.total_tokens/1e9:.3f}B tokens) | "
                  f"per-domain val: {list(dom_val)}")
        else:
            train_loader, val_loader = make_loaders(
                args.data_dir, args.seq, args.batch, device, args.seed)
            dom_val = {}
            print(f"train shards: {len(train_loader.files)} "
                  f"({train_loader.total_tokens/1e9:.3f}B tokens)"
                  + (f" | val: {len(val_loader.files)}" if val_loader else " | no val"))

    # ---- resume ----
    step, tokens_seen = 0, 0
    if os.path.exists(ckpt_path):
        step, tokens_seen = load_ckpt(ckpt_path, model, muon, adamw, train_loader, device)
        print(f"resumed from {ckpt_path} at step {step} ({tokens_seen/1e9:.3f}B tokens)")

    def set_lr(mult):
        for opt, base in ((muon, args.muon_lr), (adamw, args.adamw_lr)):
            for grp in opt.param_groups:
                grp["lr"] = base * mult

    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if (args.bf16 and device.type == "cuda") else None)

    model.train()
    tokens_per_step = args.batch * args.seq * args.accum
    t_log = time.time()
    t_ckpt = time.time()
    log_f = open(log_path, "a")
    print(f"training {args.preset}: {args.steps} steps x {tokens_per_step} tok/step "
          f"= {args.steps*tokens_per_step/1e9:.2f}B tokens target | fp={'bf16' if args.bf16 else 'fp32'}")

    while step < args.steps:
        mult = wsd_lr_mult(step, args.steps, args.warmup, args.decay_frac)
        set_lr(mult)
        if args.curriculum and not args.synthetic:
            train_loader.set_progress(step / max(1, args.steps))

        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(args.accum):
            if synth is not None:
                x, y = synth
            else:
                x, y = train_loader.get_batch()
            if autocast is not None:
                with autocast:
                    _, loss = model(x, targets=y)      # training -> R sampled per step
            else:
                _, loss = model(x, targets=y)
            (loss / args.accum).backward()
            loss_accum += loss.item() / args.accum

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        muon.step()
        adamw.step()
        step += 1
        tokens_seen += tokens_per_step

        if step % args.log_every == 0:
            dt = time.time() - t_log
            tok_s = args.log_every * tokens_per_step / dt
            rec = {"step": step, "loss": round(loss_accum, 4),
                   "lr_mult": round(mult, 4), "grad_norm": round(float(gnorm), 3),
                   "tok_s": round(tok_s), "tokens_seen": tokens_seen}
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            print(f"step {step:>6d} | loss {loss_accum:6.3f} | lr {args.muon_lr*mult:.2e} "
                  f"| gnorm {float(gnorm):5.2f} | {tok_s:7.0f} tok/s "
                  f"| {tokens_seen/1e9:.3f}B seen")
            t_log = time.time()

        if step % args.eval_every == 0 and not args.synthetic:
            # probe on val shards if present, else fall back to fresh train batches
            # (loop-gain compares R on the SAME batches, so it's valid either way)
            probe_loader = val_loader if val_loader is not None else train_loader
            probes = {"loss": eval_val_loss(model, probe_loader, args.eval_iters, cfg.r_default),
                      "loss_is_val": val_loader is not None,
                      "loop_gain": loop_gain(model, probe_loader, 8, cfg.r_max)}
            probes.update(pkm_health(model))
            if dom_val:
                probes["val_by_domain"] = {
                    t: round(eval_val_loss(model, l, 4, cfg.r_default), 4)
                    for t, l in dom_val.items()}
            if args.curriculum and not args.synthetic:
                probes["stage"] = train_loader.stage_weights
            rec = {"step": step, "probe": probes}
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            print(f"  [probe] {'val' if val_loader else 'train'}_loss={probes['loss']} "
                  f"loop_gain={probes['loop_gain']} pkm_gate={probes.get('pkm_gate_mean')}", flush=True)

        if step % args.ckpt_every == 0 or (time.time() - t_ckpt) > args.ckpt_secs:
            save_ckpt(ckpt_path, model, muon, adamw, train_loader, step, tokens_seen, cfg, args)
            t_ckpt = time.time()
            print(f"  [ckpt] saved at step {step}")

    save_ckpt(ckpt_path, model, muon, adamw, train_loader, step, tokens_seen, cfg, args)
    log_f.close()
    print(f"done: {step} steps, {tokens_seen/1e9:.3f}B tokens -> {ckpt_path}")


if __name__ == "__main__":
    main()
