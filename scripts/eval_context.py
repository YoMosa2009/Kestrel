"""Does the model actually work past position 1024?

Loss broken down by POSITION within a long window. A model trained at 1024 tokens
typically holds steady up to 1024 and then degrades sharply past it, because its
attention blocks have never seen those RoPE positions. A successful context
extension shows a flat (or falling) curve all the way to 4096. That curve is the
pass/fail test for the 4k step; a single averaged loss would hide it.

Caveat, stated rather than buried: windows are random 4096-token spans of the
validation shards, which are concatenated documents. A window can cross a
document boundary, so loss at late positions measures "handles late positions
without breaking" more than "uses 3,000 tokens of genuine long-range context".
The code and docs shards carry the long documents (12% / 18% of documents are
>= 4096 tokens), so they are reported separately and are the better signal.

    python -m scripts.eval_context --ckpt ckpt.pt --label v0.2
    python -m scripts.eval_context --ckpt ckpt.pt --rope-scale 4 --label v0.2-PI-zeroshot
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from kestrel.generate import load

BUCKETS = [(0, 512), (512, 1024), (1024, 2048), (2048, 3072), (3072, 4096)]


@torch.no_grad()
def position_loss(model, arr, seq, n_windows, rng, device):
    """Mean per-token loss in each position bucket over n random windows."""
    sums = np.zeros(len(BUCKETS))
    counts = np.zeros(len(BUCKETS))
    hi = len(arr) - seq - 1
    if hi <= 0:
        return None
    for _ in range(n_windows):
        off = int(rng.integers(0, hi))
        w = torch.from_numpy(np.asarray(arr[off: off + seq + 1], dtype=np.int64)).to(device)
        x, y = w[:-1][None], w[1:][None]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = model(x, targets=None)
        per = F.cross_entropy(logits[0].float(), y[0], reduction="none").cpu().numpy()
        for i, (a, b) in enumerate(BUCKETS):
            if a < seq:
                seg = per[a:min(b, seq)]
                sums[i] += seg.sum()
                counts[i] += seg.size
    return [float(s / c) if c else None for s, c in zip(sums, counts)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/kestrel-bpe.json")
    ap.add_argument("--data-dir", default="data_5b")
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--windows", type=int, default=24, help="per domain")
    ap.add_argument("--rope-scale", type=float, default=None,
                    help="override the checkpoint's value, e.g. to test zero-shot PI")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default="experiments/context_eval.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, tok, cfg, seen = load(args.ckpt, args.tokenizer, device)
    if args.rope_scale is not None:
        cfg.rope_scale = args.rope_scale            # cfg is shared by reference
    label = args.label or os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))
    print(f"== {label} | rope theta={cfg.rope_theta:g} scale={cfg.rope_scale:g} | "
          f"seq {args.seq} | {args.windows} windows/domain ==")

    rng = np.random.default_rng(1234)
    domains = {}
    for f in sorted(glob.glob(os.path.join(args.data_dir, "val_*.bin"))):
        tag = os.path.basename(f)[4:-4]
        arr = np.memmap(f, dtype=np.uint16, mode="r")
        t = time.time()
        r = position_loss(model, arr, args.seq, args.windows, rng, device)
        if r is None:
            continue
        domains[tag] = r
        print(f"  {tag:5s} " + "  ".join(f"{v:6.3f}" if v is not None else "   -  " for v in r)
              + f"   ({time.time()-t:.0f}s)")

    head = "        " + "  ".join(f"{a:>4d}-{b:<4d}"[:6] for a, b in BUCKETS)
    mean = [float(np.mean([d[i] for d in domains.values() if d[i] is not None]))
            for i in range(len(BUCKETS))]
    long_tags = [t for t in ("code", "docs") if t in domains]
    long_mean = [float(np.mean([domains[t][i] for t in long_tags])) for i in range(len(BUCKETS))]
    print(head)
    print("  ALL   " + "  ".join(f"{v:6.3f}" for v in mean))
    if long_tags:
        print("  LONG  " + "  ".join(f"{v:6.3f}" for v in long_mean) + "   (code+docs)")
    beyond = mean[2:]
    within = mean[1]
    verdict = ("EXTENDS: positions past 1024 are no worse than 512-1024"
               if max(beyond) <= within * 1.05 else
               f"DEGRADES past 1024: worst bucket {max(beyond):.3f} vs {within:.3f} at 512-1024")
    print(f"  -> {verdict}")

    rec = {"label": label, "ckpt": args.ckpt, "tokens_seen": seen, "seq": args.seq,
           "rope_theta": cfg.rope_theta, "rope_scale": cfg.rope_scale,
           "buckets": [f"{a}-{b}" for a, b in BUCKETS], "all": mean,
           "long_docs": long_mean if long_tags else None, "by_domain": domains,
           "verdict": verdict, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    hist = []
    if os.path.exists(args.out):
        try:
            hist = json.load(open(args.out))
        except Exception:
            hist = []
    hist.append(rec)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(hist, open(args.out, "w"), indent=1)
    print(f"saved -> {args.out} ({len(hist)} record(s))")


if __name__ == "__main__":
    main()
