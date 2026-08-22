"""Generalization eval suite (docs/10 §6) — does the model GENERALIZE, or copy form?

Five measures, all runnable on one checkpoint, designed so Phase D can be compared
against the v0.1 control:

  1. arithmetic   — 3-digit add/sub on randomly drawn numbers (few-shot prompted,
                    which is the fair way to test a base model). v0.1 baseline ~0.
  2. two_hop      — composition over NONSENSE entities, so it cannot be memorized:
                    "A zorp is a blim. All blims are red." -> is the zorp red?
  3. verbatim     — % of generated 20-grams found verbatim in the training corpus.
                    The most direct "is it copying?" metric. Lower is better.
  4. repetition   — distinct-4gram ratio inside generated text. Catches the
                    "Pluto x5" collapse. Higher is better.
  5. val_domain   — per-domain held-out loss (code/math/cli/... separately).

    python -m scripts.eval_generalization --ckpt ckpt.pt --data-dir data_5b
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random

import numpy as np
import torch

from kestrel.generate import load, generate


# ----------------------------------------------------------------- 1. arithmetic

def eval_arithmetic(model, tok, device, n=60, seed=0):
    rng = random.Random(seed)
    correct = 0
    for _ in range(n):
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        op = rng.choice(["+", "-"])
        want = a + b if op == "+" else a - b
        shots = []
        for _ in range(4):                      # few-shot: fair for a base model
            x, y = rng.randint(100, 999), rng.randint(100, 999)
            shots.append(f"{x} {op} {y} = {x + y if op == '+' else x - y}")
        prompt = "\n".join(shots) + f"\n{a} {op} {b} ="
        out = generate(model, tok, prompt, max_new=8, temperature=1.0, top_k=1,
                       top_p=1.0, n_loops=2, device=device)[len(prompt):]
        head = out.strip().split("\n")[0].strip()
        num = "".join(c for c in head.split()[0] if c in "-0123456789") if head.split() else ""
        if num.lstrip("-").isdigit() and int(num) == want:
            correct += 1
    return correct / n


# ----------------------------------------------------------------- 2. two-hop

NONSENSE = ["zorp", "blim", "drask", "vunt", "krell", "plome", "thex", "gorb",
            "murn", "quilb", "snad", "wexil"]

def eval_two_hop(model, tok, device, n=40, seed=0):
    rng = random.Random(seed)
    colors = ["red", "blue", "green", "purple", "orange"]
    correct = 0
    for _ in range(n):
        a, b, dc = rng.sample(NONSENSE, 3)
        col, distractor = rng.sample(colors, 2)
        # A DISTRACTOR colour is essential. Without one the only colour in the
        # prompt IS the answer, so copying the last colour word passes without any
        # reasoning — v0.1 scored 75% that way, which is why this test was fixed.
        # With a wrong colour stated LAST, copying fails and the model must trace
        # a -> b -> col. Entities are nonsense, so nothing can be memorized.
        d1a, d1b, d1c = rng.sample(NONSENSE, 3)
        c1, c1d = rng.sample(colors, 2)
        prompt = (f"A {d1a} is a {d1b}. All {d1b}s are {c1}. All {d1c}s are {c1d}. "
                  f"Question: what color is the {d1a}? Answer: {c1}\n"
                  f"A {a} is a {b}. All {b}s are {col}. All {dc}s are {distractor}. "
                  f"Question: what color is the {a}? Answer:")
        out = generate(model, tok, prompt, max_new=6, temperature=1.0, top_k=1,
                       top_p=1.0, n_loops=2, device=device)[len(prompt):]
        head = out.lower()[:30]
        if col in head and distractor not in head:   # right colour, not the decoy
            correct += 1
    return correct / n


# ----------------------------------------------------------------- 3./4. text stats

def _ngrams(ids, n):
    return {tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)}


def build_train_ngrams(data_dir, n=20, sample_tokens=40_000_000, seed=0):
    """Hash-set of 20-grams from a random sample of the training shards.
    A generated 20-gram found here is definitely copied; one that is not may
    still exist elsewhere in the corpus — so the reported rate is a LOWER BOUND."""
    files = sorted(glob.glob(os.path.join(data_dir, "train_*.bin")))
    rng = np.random.default_rng(seed)
    per = max(1, sample_tokens // max(1, len(files)))
    seen = set()
    for f in files:
        a = np.memmap(f, dtype=np.uint16, mode="r")
        if len(a) <= per:
            chunk = np.asarray(a)
        else:
            off = int(rng.integers(0, len(a) - per))
            chunk = np.asarray(a[off:off + per])
        c = chunk.tolist()
        for i in range(0, len(c) - n, 1):
            seen.add(hash(tuple(c[i:i + n])))
    return seen


PROMPTS = [
    "The history of the printing press begins",
    "def quicksort(arr):",
    "In order to configure a web server, you should",
    "#!/bin/bash\n# Backup script",
    "The main difference between TCP and UDP is",
    "Get-Process | Where-Object",
]

def eval_text_stats(model, tok, device, train_ngrams, n_gen=12, max_new=120, ngram=20):
    copied = total = 0
    distinct_ratios = []
    for i, p in enumerate(PROMPTS * ((n_gen // len(PROMPTS)) + 1)):
        if i >= n_gen:
            break
        out = generate(model, tok, p, max_new=max_new, temperature=0.8, top_k=40,
                       top_p=0.95, n_loops=2, device=device)
        ids = tok.encode(out[len(p):]).ids
        if len(ids) > ngram:
            for j in range(len(ids) - ngram):
                total += 1
                if hash(tuple(ids[j:j + ngram])) in train_ngrams:
                    copied += 1
        if len(ids) > 8:                     # distinct-4gram ratio (repetition)
            g = [tuple(ids[k:k + 4]) for k in range(len(ids) - 3)]
            distinct_ratios.append(len(set(g)) / len(g))
    return (copied / max(1, total)), (sum(distinct_ratios) / max(1, len(distinct_ratios)))


# ----------------------------------------------------------------- 5. val loss

@torch.no_grad()
def eval_val_domains(model, data_dir, seq, device, iters=8):
    from kestrel.data import make_domain_val_loaders
    out = {}
    for tag, loader in make_domain_val_loaders(data_dir, seq, 4, device).items():
        tot = 0.0
        for _ in range(iters):
            x, y = loader.get_batch()
            tot += model(x, targets=y, n_loops=2)[1].item()
        out[tag] = round(tot / iters, 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--tokenizer", default="tokenizer/kestrel-bpe.json")
    ap.add_argument("--data-dir", default="data_5b")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--sample-tokens", type=int, default=20_000_000)
    ap.add_argument("--label", default=None, help="name for this checkpoint in the report")
    ap.add_argument("--out", default="experiments/generalization.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, tok, cfg, seen = load(args.ckpt, args.tokenizer, device)
    label = args.label or os.path.basename(args.ckpt)
    print(f"== {label} | {seen/1e9:.2f}B tokens seen ==", flush=True)

    print("[1/5] arithmetic (3-digit, few-shot) ...", flush=True)
    arith = eval_arithmetic(model, tok, device)
    print("[2/5] two-hop composition (nonsense entities) ...", flush=True)
    twohop = eval_two_hop(model, tok, device)
    print(f"[3/5] indexing {args.sample_tokens/1e6:.0f}M training tokens for copy check ...", flush=True)
    ng = build_train_ngrams(args.data_dir, sample_tokens=args.sample_tokens)
    print("[4/5] generating + measuring copy / repetition ...", flush=True)
    copy_rate, distinct = eval_text_stats(model, tok, device, ng)
    print("[5/5] per-domain val loss ...", flush=True)
    vals = eval_val_domains(model, args.data_dir, args.seq, device)

    res = {"label": label, "tokens_seen": seen, "arithmetic_acc": round(arith, 4),
           "two_hop_acc": round(twohop, 4), "verbatim_copy_rate": round(copy_rate, 5),
           "distinct_4gram": round(distinct, 4), "val_by_domain": vals,
           "val_mean": round(sum(vals.values()) / max(1, len(vals)), 4)}

    print("\n" + "=" * 56)
    print(f"{'arithmetic (3-digit)':32s} {arith*100:6.1f} %   higher better")
    print(f"{'two-hop composition':32s} {twohop*100:6.1f} %   higher better")
    print(f"{'verbatim 20-gram copy':32s} {copy_rate*100:6.2f} %   LOWER better (lower bound)")
    print(f"{'distinct-4gram (anti-repetition)':32s} {distinct*100:6.1f} %   higher better")
    print(f"{'val loss (mean over domains)':32s} {res['val_mean']:6.3f}     lower better")
    print("=" * 56)
    for t, v in sorted(vals.items(), key=lambda x: x[1]):
        print(f"   val[{t:5s}] {v:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    hist = json.load(open(args.out)) if os.path.exists(args.out) else []
    hist = [h for h in hist if h.get("label") != label] + [res]
    json.dump(hist, open(args.out, "w"), indent=1)
    print(f"\nsaved -> {args.out} ({len(hist)} checkpoint(s) on record)")


if __name__ == "__main__":
    main()
