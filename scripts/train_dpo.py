"""DPO (Rafailov et al. 2023) on a Kestrel checkpoint, reading scripts/prepare_dpo.py output.

    loss = -logsigmoid(beta * [(logp(chosen) - logp_ref(chosen)) - (logp(rejected) - logp_ref(rejected))])

logp is summed over the answer tokens. The reference is a frozen copy of the starting model.
Every forward runs at R=2, which is what the app ships. Resumable like kestrel/train.py, and
it writes the same checkpoint format, so kestrel.generate and scripts/eval_chat.py just work.

    python -m scripts.train_dpo --self-test        # tiny random model on CPU
    python -m scripts.train_dpo --ckpt in.pt --data-dir data_dpo --sft-dir data_sft --out runs/dpo
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from kestrel.config import PRESETS, KestrelConfig
from kestrel.data import SFTLoader
from kestrel.model import KestrelModel
from kestrel.optim import build_optimizers
from kestrel.train import wsd_lr_mult


def seq_logp(model, ids, mask, n_loops):
    """Summed log-prob of the scored tokens of each row. mask aligns with ids."""
    logits, _ = model(ids[:, :-1], n_loops=n_loops)
    m = mask[:, 1:].bool()
    rows = m.nonzero()[:, 0]
    lp = F.log_softmax(logits[m].float(), -1).gather(-1, ids[:, 1:][m].unsqueeze(-1)).squeeze(-1)
    return torch.zeros(ids.size(0), device=ids.device, dtype=lp.dtype).index_add_(0, rows, lp)


def dpo_loss(pi, ref, beta):
    """pi and ref hold chosen rows then rejected rows."""
    n = pi.size(0) // 2
    h = (pi[:n] - ref[:n]) - (pi[n:] - ref[n:])
    return -F.logsigmoid(beta * h).mean(), (pi[:n] > pi[n:]).float().mean()


class Pairs:
    def __init__(self, data_dir, device, eot=0):
        self.tokens = np.fromfile(os.path.join(data_dir, "tokens.bin"), dtype=np.uint16)
        self.index = np.load(os.path.join(data_dir, "index.npy"))
        self.device, self.eot = device, eot
        self.val = np.nonzero(self.index[:, 5] == 1)[0]
        self.train = np.nonzero(self.index[:, 5] == 0)[0]

    def batch(self, idx):
        """(ids, mask) with chosen rows first, then rejected, right-padded."""
        seqs = []
        for side in (0, 1):
            for i in idx:
                s, p, c, r = self.index[i, :4]
                prompt = self.tokens[s:s + p]
                ans = self.tokens[s + p:s + p + c] if side == 0 else self.tokens[s + p + c:s + p + c + r]
                seqs.append((prompt, ans))
        L = max(len(p) + len(a) for p, a in seqs)
        ids = np.full((len(seqs), L), self.eot, dtype=np.int64)
        mask = np.zeros((len(seqs), L), dtype=np.float32)
        for b, (p, a) in enumerate(seqs):
            ids[b, :len(p)] = p
            ids[b, len(p):len(p) + len(a)] = a
            mask[b, len(p):len(p) + len(a)] = 1.0
        return (torch.from_numpy(ids).to(self.device), torch.from_numpy(mask).to(self.device))

    def schedule(self, order, micro, seed):
        """Micro-batches of similar length, so little compute goes to padding."""
        rng = np.random.default_rng(seed)
        order = rng.permutation(order)
        lens = self.index[order, 1] + np.maximum(self.index[order, 2], self.index[order, 3])
        out = []
        for i in range(0, len(order), micro * 64):
            chunk = order[i:i + micro * 64]
            chunk = chunk[np.argsort(lens[i:i + micro * 64])]
            out += [chunk[j:j + micro] for j in range(0, len(chunk), micro)]
        rng.shuffle(out)
        return out


@torch.no_grad()
def eval_pairs(model, ref, pairs, micro, beta, n_loops, autocast):
    model.eval()
    losses, accs = [], []
    per: dict[int, list] = {}
    for mb in pairs.schedule(pairs.val, micro, 0):
        ids, mask = pairs.batch(mb)
        with autocast:
            pi, r = seq_logp(model, ids, mask, n_loops), seq_logp(ref, ids, mask, n_loops)
        loss, acc = dpo_loss(pi, r, beta)
        losses.append(loss.item()); accs.append(acc.item())
        n = len(mb)
        for k, i in enumerate(mb):
            per.setdefault(int(pairs.index[i, 4]), []).append(float(pi[k] > pi[n + k]))
    model.train()
    return float(np.mean(losses)), float(np.mean(accs)), {k: round(float(np.mean(v)), 3) for k, v in sorted(per.items())}


@torch.no_grad()
def sft_val(model, sft_dir, device, autocast, n_loops, iters=10):
    """The masked SFT val loss: did DPO cost ordinary quality? Same batches every call."""
    model.eval()
    loader = SFTLoader(sft_dir, 4096, 8, device, split="val", seed=1)
    tot = 0.0
    for _ in range(iters):
        x, y, m = loader.get_batch()
        with autocast:
            tot += model(x, targets=y, loss_mask=m, n_loops=n_loops)[1].item()
    model.train()
    return tot / iters


def self_test():
    """Runs on a Colab CPU runtime before any A100 time is spent."""
    torch.manual_seed(0)
    cfg = PRESETS["test"]()
    model = KestrelModel(cfg).train()
    ref = copy.deepcopy(model).eval()
    V = cfg.vocab_size

    # right padding must not change the log-prob of the tokens before it
    a = torch.randint(1, V, (1, 40))
    pad = torch.cat([a, torch.zeros((1, 25), dtype=torch.long)], 1)
    ma = torch.zeros(1, 40); ma[0, 20:] = 1
    mp = torch.zeros(1, 65); mp[0, 20:40] = 1
    with torch.no_grad():
        d = abs(seq_logp(ref, a, ma, 2).item() - seq_logp(ref, pad, mp, 2).item())
    assert d < 1e-3, f"padding leaked into earlier tokens: {d}"

    # hand-computed log-prob
    with torch.no_grad():
        logits, _ = ref(a[:, :-1], n_loops=2)
        want = F.log_softmax(logits[0, 19:39].float(), -1).gather(-1, a[0, 20:40].unsqueeze(-1)).sum().item()
        got = seq_logp(ref, a, ma, 2).item()
    assert abs(want - got) < 1e-3, f"log-prob alignment is wrong: {want} vs {got}"

    # at the start pi == ref, so the loss is exactly log 2
    c = torch.randint(1, V, (2, 30)); r = torch.randint(1, V, (2, 30))
    ids = torch.cat([c, r]); mask = torch.zeros(4, 30); mask[:, 10:] = 1
    with torch.no_grad():
        l0, _ = dpo_loss(seq_logp(ref, ids, mask, 2), seq_logp(ref, ids, mask, 2), 0.1)
    assert abs(l0.item() - 0.6931) < 1e-3, f"start loss {l0.item()}, expected log 2"

    # and it can learn: the preference for chosen rises
    muon, adamw = build_optimizers(model, muon_lr=0.005, adamw_lr=0.001)
    with torch.no_grad():
        base = seq_logp(ref, ids, mask, 2)
        gap0 = (base[:2] - base[2:]).mean().item()
    for _ in range(12):
        muon.zero_grad(set_to_none=True); adamw.zero_grad(set_to_none=True)
        loss, _ = dpo_loss(seq_logp(model, ids, mask, 2), base, 0.1)
        loss.backward()
        muon.step(); adamw.step()
    with torch.no_grad():
        pi = seq_logp(model, ids, mask, 2)
        gap1 = (pi[:2] - pi[2:]).mean().item()
    assert gap1 > gap0 + 1.0, f"preference did not rise: {gap0} -> {gap1}"
    assert not any(p.grad is not None and p.grad.abs().sum() > 0 for p in ref.parameters()), "reference got gradients"
    print(f"SELF_TEST_OK | padding diff {d:.2e} | start loss {l0.item():.4f} | gap {gap0:.1f} -> {gap1:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--data-dir")
    ap.add_argument("--sft-dir")
    ap.add_argument("--out")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--micro", type=int, default=8, help="pairs per micro-batch")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0, help="cap the run; 0 = --epochs")
    ap.add_argument("--muon-lr", type=float, default=0.001)
    ap.add_argument("--adamw-lr", type=float, default=0.00015)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--decay-frac", type=float, default=0.5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--n-loops", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    os.makedirs(args.out, exist_ok=True)
    ck_path, status = os.path.join(args.out, "ckpt.pt"), os.path.join(args.out, "status.json")
    log = open(os.path.join(args.out, "log.jsonl"), "a")

    src = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = KestrelConfig(**src["cfg"])
    ref_cfg = copy.deepcopy(cfg); ref_cfg.grad_checkpoint = False
    ref = KestrelModel(ref_cfg).to(device).eval()
    ref.load_state_dict(src["model"])
    ref.requires_grad_(False)
    cfg.grad_checkpoint = True
    model = KestrelModel(cfg).to(device).train()
    model.load_state_dict(src["model"])
    muon, adamw = build_optimizers(model, muon_lr=args.muon_lr, adamw_lr=args.adamw_lr, fused=True)
    print(f"reference and start: {args.ckpt} | rope_scale={cfg.rope_scale} | "
          f"{src['tokens_seen']/1e9:.3f}B tokens seen", flush=True)

    pairs = Pairs(args.data_dir, device)
    sched = []
    for e in range(int(np.ceil(args.epochs))):
        sched += pairs.schedule(pairs.train, args.micro, args.seed + e)
    total = int(len(pairs.train) * args.epochs / (args.micro * args.accum))
    if args.max_steps:
        total = min(total, args.max_steps)
    step, tokens_seen, base_sft = 0, src["tokens_seen"], None
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); muon.load_state_dict(ck["muon"]); adamw.load_state_dict(ck["adamw"])
        step, tokens_seen, base_sft = ck["dpo_step"], ck["tokens_seen"], ck.get("base_sft")
        print(f"resumed at DPO step {step}", flush=True)
    del src
    print(f"{len(pairs.train):,} train pairs, {len(pairs.val):,} val | {total} steps of "
          f"{args.micro * args.accum} pairs | beta {args.beta}", flush=True)

    def save(path, final=False):
        tmp = path + ".tmp"
        d = {"model": model.state_dict(), "step": step, "dpo_step": step, "tokens_seen": tokens_seen,
             "cfg": asdict(cfg), "args": vars(args), "base_sft": base_sft}
        if not final:
            d.update(muon=muon.state_dict(), adamw=adamw.state_dict())
        torch.save(d, tmp)
        os.replace(tmp, path)

    def publish(state, **kw):
        tmp = status + ".tmp"
        with open(tmp, "w") as f:
            json.dump(dict(state=state, step=step, total=total, **kw), f)
        os.replace(tmp, status)

    if base_sft is None:
        base_sft = sft_val(model, args.sft_dir, device, autocast, args.n_loops)
    print(f"start: SFT val {base_sft:.4f}", flush=True)
    t0 = time.time()
    ntok = 0
    while step < total:
        mult = wsd_lr_mult(step, total, args.warmup, args.decay_frac)
        for opt, base in ((muon, args.muon_lr), (adamw, args.adamw_lr)):
            for g in opt.param_groups:
                g["lr"] = base * mult
        muon.zero_grad(set_to_none=True); adamw.zero_grad(set_to_none=True)
        run_loss = run_acc = 0.0
        for k in range(args.accum):
            ids, mask = pairs.batch(sched[step * args.accum + k])
            with autocast:
                with torch.no_grad():
                    r = seq_logp(ref, ids, mask, args.n_loops)
                pi = seq_logp(model, ids, mask, args.n_loops)
            loss, acc = dpo_loss(pi, r, args.beta)
            (loss / args.accum).backward()
            run_loss += loss.item() / args.accum; run_acc += acc.item() / args.accum
            ntok += ids.numel()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        muon.step(); adamw.step()
        step += 1
        if step % args.log_every == 0:
            rate = ntok / (time.time() - t0)
            rec = dict(step=step, loss=round(run_loss, 4), acc=round(run_acc, 3), gnorm=round(float(gnorm), 3), tok_s=round(rate))
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"dpo {step:>5d}/{total} | loss {run_loss:.4f} | chosen>rejected {run_acc:.2f} | "
                  f"gnorm {float(gnorm):6.1f} | {rate:6.0f} tok/s | eta {(total-step)*(time.time()-t0)/max(step,1)/60:.0f} min", flush=True)
            publish("running")
        if step % args.eval_every == 0 or step == total:
            v, a, per = eval_pairs(model, ref, pairs, args.micro, args.beta, args.n_loops, autocast)
            s = sft_val(model, args.sft_dir, device, autocast, args.n_loops)
            print(f"  [val] loss {v:.4f} | chosen>rejected {a:.3f} | SFT val {s:.4f} ({100*(s/base_sft-1):+.2f}%) | per slice {per}", flush=True)
            log.write(json.dumps(dict(step=step, val=v, val_acc=a, sft_val=s, per=per)) + "\n"); log.flush()
        if step == total // 2:
            save(os.path.join(args.out, "ckpt_half.pt"), final=True)
        if step % args.ckpt_every == 0 and step < total:
            save(ck_path)
    save(ck_path, final=True)
    publish("done")
    print("DPO_DONE", flush=True)


if __name__ == "__main__":
    main()
