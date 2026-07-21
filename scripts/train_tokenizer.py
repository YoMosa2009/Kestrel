"""P1 tokenizer training (docs/05 §3, gate S2).

Trains a byte-level BPE at the target vocab (49152, uint16-safe) on a streamed,
stratified sample of the real data mixture. Byte-level + digit-splitting +
code-friendly. A `--superword` variant relaxes whitespace splitting so tokens
may cross word boundaries (SuperBPE-style, Pillar 4) — used to measure the
token-reduction gate (>=18%) against the standard tokenizer.

    python -m scripts.train_tokenizer --sample-mb 300 --out tokenizer/kestrel-bpe.json
    python -m scripts.train_tokenizer --sample-mb 300 --superword --out tokenizer/kestrel-super.json
"""

from __future__ import annotations

import argparse
import os
import time

from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, Regex

# (dataset, config, text-field, sampling weight) — the stratified mixture
SOURCES = [
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text", 0.55),
    ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "text", 0.20),
    ("codeparrot/codeparrot-clean", None, "content", 0.25),
]

SPECIALS = ["<|endoftext|>"]


def sample_texts(sample_mb: int, seed: int = 1337):
    """Yield documents round-robin from the sources until ~sample_mb of text."""
    budget = sample_mb * 1024 * 1024
    per_source = [int(budget * w) for _, _, _, w in SOURCES]
    iters = []
    for (name, cfg, field, _), _ in zip(SOURCES, per_source):
        # small shuffle buffer: big buffers make streaming warmup very slow, and
        # we only need light decorrelation for a tokenizer-training sample
        ds = load_dataset(name, cfg, split="train", streaming=True).shuffle(seed, buffer_size=1000)
        iters.append((iter(ds), field))
    used = [0] * len(SOURCES)
    done = [False] * len(SOURCES)
    while not all(done):
        for i, (it, field) in enumerate(iters):
            if done[i]:
                continue
            try:
                row = next(it)
            except StopIteration:
                done[i] = True
                continue
            txt = row.get(field) or ""
            if not txt:
                continue
            used[i] += len(txt.encode("utf-8"))
            yield txt
            if used[i] >= per_source[i]:
                done[i] = True
    print(f"  sampled bytes per source (MB): {[round(u/1e6,1) for u in used]}")


def build_tokenizer(superword: bool) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=None))
    steps = [pre_tokenizers.Digits(individual_digits=True)]
    if superword:
        # Superword: allow merges to cross spaces (that is the SuperBPE win —
        # multi-word units like "for (int i = 0;"), but split on newlines so a
        # merge unit is at most one line. Without this bound the BPE trainer
        # treats whole documents as single "words" and OOMs on 16 GB.
        steps.append(pre_tokenizers.Split(Regex(r"\n"), behavior="isolated"))
        steps.append(pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False))
    else:
        steps.append(pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True))
    tok.pre_tokenizer = pre_tokenizers.Sequence(steps)
    tok.decoder = decoders.ByteLevel()
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=49152)
    ap.add_argument("--sample-mb", type=int, default=300)
    ap.add_argument("--superword", action="store_true")
    ap.add_argument("--max-token-len", type=int, default=0,
                    help="cap bytes per token (0=default: 128 std, 32 superword)")
    ap.add_argument("--out", default="tokenizer/kestrel-bpe.json")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"training {'SUPERWORD ' if args.superword else ''}BPE vocab={args.vocab} "
          f"on ~{args.sample_mb} MB sample -> {args.out}")

    max_len = args.max_token_len or (32 if args.superword else 128)
    tok = build_tokenizer(args.superword)
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab, special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        max_token_length=max_len, show_progress=True)

    t0 = time.time()
    tok.train_from_iterator(sample_texts(args.sample_mb, args.seed), trainer=trainer)
    tok.save(args.out)
    print(f"done in {time.time()-t0:.0f}s | vocab={tok.get_vocab_size()} -> {args.out}")


if __name__ == "__main__":
    main()
