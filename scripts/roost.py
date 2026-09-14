"""Roost CLI — teach, retrieve, consolidate, and the teach-me-today test.

    # teach it things (immediately retrievable, not yet consolidated)
    python -m scripts.roost teach "The lab GPU is an RTX 3060 with 12 GB."

    # what does it recall right now, via retrieval?
    python -m scripts.roost ask "What GPU does the lab have?"

    # the nightly sleep job
    python -m scripts.roost sleep --ckpt experiments/nano_d/ckpt.pt

    # the S3 success criterion: 50 novel facts, >=60% next-day recall,
    # <=2% regression on the frozen suite
    python -m scripts.roost teach-me-today --ckpt ckpt.pt --n 50

    # session state
    python -m scripts.roost absorb --text "..." --session experiments/roost/s1.pt
    python -m scripts.roost session-info --session experiments/roost/s1.pt
"""

from __future__ import annotations

import argparse
import json
import os
import random

import torch
from tokenizers import Tokenizer

from kestrel.config import KestrelConfig
from kestrel.model import KestrelModel
from kestrel.roost import (EpisodicStore, consolidate, save_session, load_session)
from kestrel.roost.store import KestrelEmbedder
from kestrel.roost.session import absorb, session_info


def load_model(ckpt: str, tok_path: str, device: str):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = KestrelConfig(**ck["cfg"])
    cfg.grad_checkpoint = False
    model = KestrelModel(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, Tokenizer.from_file(tok_path)


# --- synthetic novel facts for the teach-me-today test -----------------------
# Deliberately nonsense entities: a real model cannot already know these, so
# recall after consolidation measures learning rather than pretraining hits.
_SUBJ = ["Varnal", "Quillon", "Threx", "Marabel", "Osquith", "Penhallow",
         "Ryndle", "Tovaric", "Ashgrove", "Belcourt"]
_PROP = [("capital", ["Dreymoor", "Halvenpike", "Sorrowbay", "Kelmarch"]),
         ("currency", ["the drell", "the marka", "the vint", "the solmar"]),
         ("chief export", ["blue salt", "glass timber", "cold iron", "reed silk"]),
         ("founding year", ["1483", "1622", "1291", "1755"]),
         ("official language", ["Tarrin", "Vessic", "Old Kelm", "Dunnish"])]


def novel_facts(n: int, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        s = rng.choice(_SUBJ)
        prop, vals = rng.choice(_PROP)
        if (s, prop) in seen:
            continue
        seen.add((s, prop))
        out.append(f"The {prop} of {s} is {rng.choice(vals)}.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["teach", "ask", "list", "sleep", "compact",
                                    "teach-me-today", "absorb", "session-info"])
    ap.add_argument("text", nargs="?", default=None)
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--tokenizer", default="tokenizer/kestrel-bpe.json")
    ap.add_argument("--store", default="experiments/roost/episodic.jsonl")
    ap.add_argument("--session", default="experiments/roost/session.pt")
    ap.add_argument("--replay-dir", default="data_5b")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--top-t", type=int, default=2048)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--recall-gate", type=float, default=0.60)
    ap.add_argument("--regression-gate", type=float, default=2.0)
    ap.add_argument("--out", default=None, help="where to write the updated ckpt")
    args = ap.parse_args()

    store = EpisodicStore(args.store)

    # --- store-only commands: no model needed ---
    if args.cmd == "teach":
        if not args.text:
            raise SystemExit("teach needs some text")
        t = store.add(args.text)
        print(f"remembered: {t.text}")
        for v in t.variants:
            print(f"  variant: {v}")
        return

    if args.cmd == "list":
        for i, it in enumerate(store.items):
            print(f"{i:3d}  [{it.kind}] {it.text}   (uses={it.uses})")
        print(f"\n{len(store)} teachables")
        return

    # --- everything below needs the model ---
    model, tok = load_model(args.ckpt, args.tokenizer, args.device)
    emb = KestrelEmbedder(model, tok, args.device)

    if args.cmd == "ask":
        if not args.text:
            raise SystemExit("ask needs a question")
        for score, it in store.retrieve(args.text, emb, args.k):
            print(f"  {score:.3f}  {it.text}")
        return

    if args.cmd == "compact":
        print(store.compact(emb))
        return

    if args.cmd == "absorb":
        if not args.text:
            raise SystemExit("absorb needs text")
        st = load_session(args.session, model, args.device) if os.path.exists(args.session) else None
        st = absorb(model, tok, args.text, st, args.device)
        save_session(st, args.session, model, label=args.text[:60])
        print(f"session saved -> {args.session}")
        print(json.dumps(session_info(args.session), indent=1))
        return

    if args.cmd == "session-info":
        print(json.dumps(session_info(args.session), indent=1))
        return

    if args.cmd == "teach-me-today":
        facts = novel_facts(args.n)
        store.extend(facts)
        print(f"taught {len(facts)} novel facts (store now {len(store)})")

    if args.cmd in ("sleep", "teach-me-today"):
        res = consolidate(model, tok, store, replay_dir=args.replay_dir,
                          device=args.device, seq=args.seq, steps=args.steps,
                          lr=args.lr, top_t=args.top_t,
                          recall_gate=args.recall_gate,
                          regression_gate=args.regression_gate)
        print()
        print(res)
        if res.accepted and args.out:
            ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
            ck["model"] = {k: v.cpu() for k, v in model.state_dict().items()}
            tmp = args.out + ".tmp"
            torch.save(ck, tmp)
            os.replace(tmp, args.out)
            print(f"consolidated checkpoint -> {args.out}")
        elif res.accepted:
            print("(not saved; pass --out to write the consolidated checkpoint)")
        log = os.path.join(os.path.dirname(args.store) or ".", "consolidation.jsonl")
        with open(log, "a", encoding="utf-8") as f:
            from dataclasses import asdict
            f.write(json.dumps(asdict(res)) + "\n")
        print(f"logged -> {log}")


if __name__ == "__main__":
    main()
