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
from kestrel.data import (make_loaders, CurriculumLoader, make_domain_val_loaders,
                          PrefetchLoader)


# ----------------------------------------------------------------- schedule

def wsd_lr_mult(step: int, total: int, warmup: int, decay_frac: float,
                floor: float = 0.0) -> float:
    """Warmup-Stable-Decay. Flat plateau, then 1-sqrt decay over the final
    `decay_frac` of steps. Multiplies each group's base LR.

    `step`/`total` are measured from the START OF THIS RUN, not from the
    checkpoint's absolute step. On a WSD-resume the previous run ended with its
    LR decayed to ~0, so re-warming is mandatory: feeding the absolute step here
    skips warmup entirely and slams a decayed model with the full plateau LR.
    """
    if step < warmup:
        return (step + 1) / max(1, warmup)
    decay_start = int(total * (1.0 - decay_frac))
    if step < decay_start:
        return 1.0
    p = (step - decay_start) / max(1, total - decay_start)
    return floor + (1.0 - floor) * (1.0 - math.sqrt(min(1.0, p)))


# ----------------------------------------------------------------- run control

def read_control(path: str) -> str:
    """Current command from control.json: 'run' | 'pause' | 'stop'.

    Deliberately forgiving: a missing, empty, or half-written file means 'run',
    so a UI crashing mid-write can never stall or kill a multi-day run.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return str(json.load(f).get("command", "run")).lower()
    except Exception:
        return "run"


def write_status(path: str, **kw):
    """Atomically publish live state for the monitor UI (and for Claude to read).

    Written via a temp file + os.replace so a reader never sees a partial JSON
    document. Failures here are swallowed: telemetry must never break training.
    """
    try:
        kw["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(kw, f, indent=1)
        os.replace(tmp, path)
    except Exception:
        pass


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
    ap.add_argument("--curriculum-start", type=float, default=0.0,
                    help="curriculum progress at the start of THIS run (0-1). A "
                         "from-scratch run wants 0.0; a resume that already saw the "
                         "broad stage can skip into 'technical' with e.g. 0.40.")
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
    ap.add_argument("--no-tf32", action="store_true",
                    help="disable TF32 tensor cores for fp32 matmuls (Ampere+). TF32 is "
                         "ON by default: the only fp32 matmuls left under --bf16 are "
                         "Muon's Newton-Schulz, which is an iterative orthogonalization "
                         "and numerically indifferent to the reduced mantissa.")
    ap.add_argument("--prefetch", type=int, default=0, metavar="DEPTH",
                    help="read batches on a background thread, DEPTH deep (0=off). "
                         "Overlaps HDD seeks with GPU compute; 3 is plenty.")
    ap.add_argument("--fused-adam", action="store_true",
                    help="use the fused CUDA AdamW kernel (one launch for all params)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    # Ampere+ tensor cores for the fp32 matmuls that autocast leaves alone
    # (Muon's Newton-Schulz). Harmless on pre-Ampere: the flag is ignored.
    if not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True      # autotune the depthwise causal conv
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "ckpt.pt")
    log_path = os.path.join(args.out, "metrics.jsonl")
    ctl_path = os.path.join(args.out, "control.json")
    status_path = os.path.join(args.out, "status.json")
    if not os.path.exists(ctl_path):
        write_status(ctl_path, command="run")

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

    muon, adamw = build_optimizers(_core(model), muon_lr=args.muon_lr,
                                   adamw_lr=args.adamw_lr, fused=args.fused_adam)

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
    # Every schedule below is measured from here, so a resumed run re-warms its LR
    # and runs its own full WSD shape between start_step and args.steps.
    if args.prefetch and train_loader is not None:
        train_loader = PrefetchLoader(train_loader, depth=args.prefetch)
        print(f"prefetch: background reader, depth {args.prefetch}")

    start_step = step
    run_steps = max(1, args.steps - start_step)
    if start_step >= args.steps:
        raise SystemExit(f"--steps {args.steps} is at or below the resumed step "
                         f"{start_step}; --steps is CUMULATIVE, so pass "
                         f"{start_step} + (new steps you want).")

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
    tf32_on = torch.backends.cuda.matmul.allow_tf32
    print(f"training {args.preset}: {run_steps} NEW steps "
          f"(step {start_step} -> {args.steps}) x {tokens_per_step} tok/step "
          f"= {run_steps*tokens_per_step/1e9:.2f}B new tokens "
          f"({(tokens_seen + run_steps*tokens_per_step)/1e9:.2f}B cumulative) | "
          f"fp={'bf16' if args.bf16 else 'fp32'} | tf32={tf32_on} "
          f"| fused_adam={args.fused_adam}")

    t_start = time.time()
    last_probe = None
    last_metrics = {}
    stopped = False

    def publish(state, **extra):
        el = time.time() - t_start
        done = step - start_step
        rate = done / el if el > 0 else 0.0
        write_status(
            status_path, state=state, pid=os.getpid(), out=args.out,
            step=step, start_step=start_step, target_step=args.steps,
            steps_done=done, steps_total=run_steps,
            progress=round(done / run_steps, 5),
            tokens_seen=tokens_seen,
            tokens_new=done * tokens_per_step,
            tokens_target=run_steps * tokens_per_step,
            elapsed_s=round(el, 1),
            eta_s=round((run_steps - done) / rate, 1) if rate > 0 else None,
            preset=args.preset, batch=args.batch, accum=args.accum, seq=args.seq,
            tokens_per_step=tokens_per_step,
            bf16=args.bf16, curriculum=args.curriculum,
            last_probe=last_probe, **{**last_metrics, **extra})

    publish("starting")

    while step < args.steps:
        cmd = read_control(ctl_path)
        if cmd == "stop":
            print("  [control] stop requested -> checkpointing and exiting", flush=True)
            save_ckpt(ckpt_path, model, muon, adamw, train_loader, step,
                      tokens_seen, cfg, args)
            publish("stopped")
            stopped = True
            break
        if cmd == "pause":
            print("  [control] paused", flush=True)
            publish("paused")
            t_pause = time.time()
            while read_control(ctl_path) == "pause":
                time.sleep(1.0)
                publish("paused", paused_s=round(time.time() - t_pause, 1))
            if read_control(ctl_path) == "stop":
                continue                      # handled at the top of the next pass
            t_start += time.time() - t_pause  # a pause must not skew the ETA
            print(f"  [control] resumed after {time.time()-t_pause:.0f}s", flush=True)

        rel = step - start_step
        mult = wsd_lr_mult(rel, run_steps, args.warmup, args.decay_frac)
        set_lr(mult)
        if args.curriculum and not args.synthetic:
            c0 = args.curriculum_start
            train_loader.set_progress(c0 + (rel / run_steps) * (1.0 - c0))

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
        publish("running")

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
            last_metrics = dict(loss=round(loss_accum, 4), lr_mult=round(mult, 5),
                                lr=args.muon_lr * mult,
                                grad_norm=round(float(gnorm), 3), tok_s=round(tok_s))

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
            last_probe = dict(probes, step=step)
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            print(f"  [probe] {'val' if val_loader else 'train'}_loss={probes['loss']} "
                  f"loop_gain={probes['loop_gain']} pkm_gate={probes.get('pkm_gate_mean')}", flush=True)

        if step % args.ckpt_every == 0 or (time.time() - t_ckpt) > args.ckpt_secs:
            save_ckpt(ckpt_path, model, muon, adamw, train_loader, step, tokens_seen, cfg, args)
            t_ckpt = time.time()
            print(f"  [ckpt] saved at step {step}")

    if not stopped:
        save_ckpt(ckpt_path, model, muon, adamw, train_loader, step, tokens_seen, cfg, args)
        publish("done")
    log_f.close()
    print(f"{'stopped' if stopped else 'done'}: {step} steps, "
          f"{tokens_seen/1e9:.3f}B tokens -> {ckpt_path}")


if __name__ == "__main__":
    main()
