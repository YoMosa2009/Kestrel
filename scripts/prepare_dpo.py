"""Build the DPO pairs for V3: (prompt, preferred answer, worse answer).

Most pairs are general preference data (UltraFeedback). The rest push on what the step-1
chat eval measured, in BOTH directions, so neither direction can win by moving the model
toward or away from refusing as a whole:

    harmful   harmful prompt:   our refusal   >  a compliant answer (hh-rlhf's rejected)
    benign    ordinary question: its answer   >  a refusal or stock line
    greet     small talk:        friendly     >  a refusal or stock line
    contrast  scary words, harmless question: the answer > a refusal
    truthful  TruthfulQA:        best answer  >  an incorrect one

The worse answers are only ever pushed down. Nothing here teaches the model to produce them.

Layout: tokens.bin (uint16, flat) + index.npy, one row per pair:
    start, prompt_len, chosen_len, rejected_len, slice_id, is_val

Usage:
    python -m scripts.prepare_dpo --out data_dpo
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time

import numpy as np
from tokenizers import Tokenizer

from scripts.prepare_sft import render
from scripts.prepare_sft_v3 import (_CONTRAST, _REFUSALS, _STOCK, _english, _with_system, a_chat_row,
                                    a_dolly, build_greetings, clean_code, fact_pair, is_clean,
                                    open_stream)

SLICES = ["ultrafeedback", "harmful", "benign", "greet", "contrast", "truthful"]
_HH_REFUSES = re.compile(r"(?i)\b(sorry|can'?t|cannot|won'?t|shouldn'?t|should not|illegal|not okay|"
                         r"not ok\b|don'?t think|wouldn'?t|i'?m not (sure|comfortable))\b")


def encode_pair(tok, turns, rejected, seq, eot):
    """turns ends with the chosen Assistant turn. Pieces encode the same way as the SFT set."""
    ids: list[int] = []
    for i, (role, content) in enumerate(turns[:-1]):
        ids += tok.encode(f"{role}: " if i == 0 else f"\n\n{role}: ").ids + tok.encode(content).ids
    ids += tok.encode("Assistant: " if len(turns) == 1 else "\n\nAssistant: ").ids
    c = tok.encode(turns[-1][1]).ids + [eot]
    r = tok.encode(rejected).ids + [eot]
    if len(ids) + max(len(c), len(r)) > seq:
        return None
    return ids, c, r


def hh_split(text):
    """(prompt, answer) from a single-exchange hh-rlhf row, else None."""
    if text.count("\n\nHuman:") != 1 or text.count("\n\nAssistant:") != 1:
        return None
    q, a = text.split("\n\nAssistant:")
    return q.replace("\n\nHuman:", "").strip(), a.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_dpo")
    ap.add_argument("--tokenizer", default="tokenizer/kestrel-bpe.json")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--ultrafeedback", type=int, default=22000)
    ap.add_argument("--harmful", type=int, default=3000)
    ap.add_argument("--benign", type=int, default=3000)
    ap.add_argument("--greet", type=int, default=700)
    ap.add_argument("--contrast", type=int, default=300)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    from datasets import load_dataset

    os.makedirs(args.out, exist_ok=True)
    tok = Tokenizer.from_file(args.tokenizer)
    eot = tok.token_to_id("<|endoftext|>")
    rng = random.Random(args.seed)
    t0 = time.time()
    pairs: dict[str, list] = {s: [] for s in SLICES}
    bad_answers = _STOCK + _REFUSALS

    def add(slice_name, turns, rejected):
        enc = encode_pair(tok, turns, rejected, args.seq, eot)
        if enc is not None:
            pairs[slice_name].append(enc)

    # ultrafeedback: the cleaned set has TruthfulQA contamination removed
    ds = open_stream("argilla/ultrafeedback-binarized-preferences-cleaned", None, "train", args.seed)
    for row in ds or []:
        if len(pairs["ultrafeedback"]) >= args.ultrafeedback:
            break
        if (row["chosen-rating"] or 0) - (row["rejected-rating"] or 0) < 1:
            continue
        ch, rj = row["chosen"], row["rejected"]
        if len(ch) != 2 or len(rj) != 2 or not rj[1]["content"].strip():
            continue
        turns = render([(ch[0]["role"], ch[0]["content"]), (ch[1]["role"], clean_code(ch[1]["content"]))])
        if len(turns) != 2 or not is_clean(turns) or not _english(turns[0][1] + turns[1][1]):
            continue
        add("ultrafeedback", _render_sys(turns, rng), rj[1]["content"].strip())
    print(f"[ultrafeedback] {len(pairs['ultrafeedback'])} ({time.time()-t0:.0f}s)", flush=True)

    # harmful: keep rows where the human-preferred reply declined and the other one complied
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train", streaming=True)
    for row in ds.shuffle(seed=args.seed, buffer_size=10000):
        if len(pairs["harmful"]) >= args.harmful:
            break
        c, r = hh_split(row["chosen"]), hh_split(row["rejected"])
        if not c or not r or c[0] != r[0] or not c[1] or not r[1]:
            continue
        if not _HH_REFUSES.search(c[1]) or _HH_REFUSES.search(r[1]) or len(r[1].split()) < 8:
            continue
        add("harmful", _render_sys([("User", c[0]), ("Assistant", rng.choice(_REFUSALS))], rng), r[1])
    print(f"[harmful] {len(pairs['harmful'])} ({time.time()-t0:.0f}s)", flush=True)

    # benign: ordinary questions, from the same two sources as the SFT fact slice
    for name, split, adapt, cap in (("HuggingFaceH4/ultrachat_200k", "train_sft", a_chat_row, args.benign * 3 // 4),
                                    ("databricks/databricks-dolly-15k", "train", a_dolly, args.benign // 4)):
        got = 0
        for row in open_stream(name, None, split, args.seed + 1) or []:
            if got >= cap:
                break
            turns = adapt(row)
            if not turns or not is_clean(turns):
                continue
            pair = fact_pair(turns)
            if pair is None:
                continue
            add("benign", _render_sys([("User", pair[0]), ("Assistant", pair[1])], rng), rng.choice(bad_answers))
            got += 1
    print(f"[benign] {len(pairs['benign'])} ({time.time()-t0:.0f}s)", flush=True)

    for turns in build_greetings(args.greet, rng):
        add("greet", turns, rng.choice(bad_answers))
    for _ in range(args.contrast):
        q, a = rng.choice(_CONTRAST)
        add("contrast", _render_sys([("User", q), ("Assistant", a)], rng), rng.choice(_REFUSALS))

    for row in load_dataset("truthfulqa/truthful_qa", "generation", split="validation"):
        a = row["best_answer"]
        a = a if a.endswith((".", "!", "?")) else a + "."
        for wrong in rng.sample(row["incorrect_answers"], min(2, len(row["incorrect_answers"]))):
            w = wrong if wrong.endswith((".", "!", "?")) else wrong + "."
            add("truthful", _render_sys([("User", row["question"]), ("Assistant", a)], rng), w)

    # --- pack ---------------------------------------------------------------
    rows = [(SLICES.index(k), e) for k, v in pairs.items() for e in v]
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    toks: list[np.ndarray] = []
    index = np.zeros((len(rows), 6), dtype=np.int64)
    pos = 0
    for i, (sid, (p, c, r)) in enumerate(rows):
        index[i] = (pos, len(p), len(c), len(r), sid, i < n_val)
        toks.append(np.asarray(p + c + r, dtype=np.uint16))
        pos += len(p) + len(c) + len(r)
    np.concatenate(toks).tofile(os.path.join(args.out, "tokens.bin"))
    np.save(os.path.join(args.out, "index.npy"), index)

    lens = index[:, 1] + np.maximum(index[:, 2], index[:, 3])
    manifest = {"pairs": len(rows), "val": n_val, "slices": SLICES, "seq": args.seq,
                "per_slice": {k: len(v) for k, v in pairs.items()},
                "tokens": int(pos), "len_median": int(np.median(lens)), "len_p90": int(np.percentile(lens, 90))}
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest, indent=1))
    print(f"done in {time.time()-t0:.0f}s -> {args.out}")


def _render_sys(turns, rng):
    return render(_with_system(turns, rng))


if __name__ == "__main__":
    main()
