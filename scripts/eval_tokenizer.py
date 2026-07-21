"""Tokenizer reduction gate (docs/05 §3 S2, docs/06).

Compares two tokenizers on fresh held-out text per domain: tokens, chars/token,
and the %-reduction of the second vs the first. Gate S2: superword must use
>=18% fewer tokens than the standard BPE at the same vocab, with clean
round-trips and no degenerate-token pathology.

    python -m scripts.eval_tokenizer --a tokenizer/kestrel-bpe.json \
        --b tokenizer/kestrel-super.json --docs 400
"""

from __future__ import annotations

import argparse

from datasets import load_dataset
from tokenizers import Tokenizer

# held-out streams (skip ahead so we don't score the tokenizer's own training docs)
EVAL_SOURCES = [
    ("web", "HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
    ("know", "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "text"),
    ("code", "codeparrot/codeparrot-clean", None, "content"),
]


def collect(name, cfg, field, docs, skip):
    ds = load_dataset(name, cfg, split="train", streaming=True).skip(skip)
    out, n = [], 0
    for row in ds:
        t = row.get(field) or ""
        if t:
            out.append(t); n += 1
        if n >= docs:
            break
    return out


def stats(tok: Tokenizer, texts):
    n_tok = n_chr = bad = 0
    max_tok_bytes = 0
    for t in texts:
        ids = tok.encode(t).ids
        n_tok += len(ids); n_chr += len(t)
        if tok.decode(ids).strip() != t.strip():
            bad += 1
    # longest single token in the vocab (degenerate-token check)
    for piece in tok.get_vocab():
        max_tok_bytes = max(max_tok_bytes, len(piece))
    return n_tok, n_chr, bad, max_tok_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="tokenizer/kestrel-bpe.json", help="baseline (standard)")
    ap.add_argument("--b", default="tokenizer/kestrel-super.json", help="candidate (superword)")
    ap.add_argument("--docs", type=int, default=400, help="held-out docs per domain")
    ap.add_argument("--skip", type=int, default=50_000)
    args = ap.parse_args()

    tok_a = Tokenizer.from_file(args.a)
    tok_b = Tokenizer.from_file(args.b)
    print(f"A (baseline)  = {args.a}  vocab={tok_a.get_vocab_size()}")
    print(f"B (candidate) = {args.b}  vocab={tok_b.get_vocab_size()}")
    print(f"held-out: {args.docs} docs/domain (skip {args.skip})\n" + "-" * 72)

    tot_a = tot_b = tot_c = 0
    print(f"{'domain':6s} {'chars':>10s} {'A tok':>10s} {'B tok':>10s} "
          f"{'A c/t':>6s} {'B c/t':>6s} {'reduce':>7s}")
    for tag, name, cfg, field in EVAL_SOURCES:
        texts = collect(name, cfg, field, args.docs, args.skip)
        an, ac, abad, amax = stats(tok_a, texts)
        bn, bc, bbad, bmax = stats(tok_b, texts)
        red = 100 * (1 - bn / an)
        tot_a += an; tot_b += bn; tot_c += ac
        print(f"{tag:6s} {ac:>10d} {an:>10d} {bn:>10d} "
              f"{ac/an:>6.2f} {bc/bn:>6.2f} {red:>6.1f}%"
              + (f"  [A bad={abad} B bad={bbad}]" if (abad or bbad) else ""))
    overall = 100 * (1 - tot_b / tot_a)
    print("-" * 72)
    print(f"OVERALL reduction B vs A: {overall:.1f}%  (gate: >=18%)  "
          f"-> {'PASS' if overall >= 18 else 'FAIL'}")
    print(f"max token bytes: A={amax} B={bmax}  "
          f"(B >> A signals degenerate superwords worth inspecting)")


if __name__ == "__main__":
    main()
