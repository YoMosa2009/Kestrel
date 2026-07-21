"""P1 data prep (docs/05 §3, docs/03 RAM discipline).

Streams the code-tilted mixture, tokenizes on the fly with the trained tokenizer,
and writes flat uint16 `.bin` shards to disk (np.memmap-ready). Raw corpora are
never stored; memory stays bounded (only one encode_batch of ids is ever held,
plus a small per-domain val buffer) — safe for the 16 GB machine as an unattended
overnight job. A per-domain validation slice is held out first.

    python -m scripts.prepare_data --tokenizer tokenizer/kestrel-bpe.json \
        --tokens 200_000_000 --out data
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

# (dataset, config, text-field, domain-tag, mixture weight) — code-tilted
SOURCES = [
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text", "web", 0.45),
    ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "text", "know", 0.20),
    ("codeparrot/codeparrot-clean", None, "content", "code", 0.35),
]


class ShardWriter:
    """Streams uint16 tokens to numbered shard files, rolling over at
    shard_tokens. Only the current small write-batch is ever in memory."""

    def __init__(self, out_dir: str, shard_tokens: int):
        self.out_dir = out_dir
        self.shard_tokens = shard_tokens
        self.idx = 0
        self.count_in_shard = 0
        self.total = 0
        self.f = None

    def _open(self):
        path = os.path.join(self.out_dir, f"train_{self.idx:03d}.bin")
        self.f = open(path, "ab")
        return path

    def write(self, ids: np.ndarray):
        pos = 0
        while pos < len(ids):
            if self.f is None:
                self._open()
            room = self.shard_tokens - self.count_in_shard
            take = min(room, len(ids) - pos)
            ids[pos:pos + take].tofile(self.f)
            self.count_in_shard += take
            self.total += take
            pos += take
            if self.count_in_shard >= self.shard_tokens:
                self.f.close(); self.f = None
                print(f"  wrote train_{self.idx:03d}.bin "
                      f"({self.count_in_shard/1e6:.1f}M tok, total {self.total/1e6:.1f}M)")
                self.idx += 1
                self.count_in_shard = 0

    def close(self):
        if self.f is not None:
            self.f.close()
            print(f"  wrote train_{self.idx:03d}.bin (partial "
                  f"{self.count_in_shard/1e6:.1f}M tok, total {self.total/1e6:.1f}M)")
            self.f = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer/kestrel-bpe.json")
    ap.add_argument("--tokens", type=int, default=200_000_000, help="total train tokens target")
    ap.add_argument("--val-tokens", type=int, default=1_000_000, help="held-out tokens PER domain")
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--batch", type=int, default=1000, help="docs per encode_batch")
    ap.add_argument("--shuffle-buffer", type=int, default=2000,
                    help="streaming shuffle buffer per source (0 disables; big values are slow)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok = Tokenizer.from_file(args.tokenizer)
    eot = tok.token_to_id("<|endoftext|>")
    assert eot is not None, "tokenizer missing <|endoftext|>"
    print(f"tokenizer vocab={tok.get_vocab_size()} eot={eot} | "
          f"target {args.tokens/1e6:.0f}M train tokens, shard {args.shard_tokens/1e6:.0f}M")

    writer = ShardWriter(args.out, args.shard_tokens)
    t0 = time.time()

    # Interleave the sources round-robin so every shard carries the mixture.
    def make_stream(name, cfg, seed):
        ds = load_dataset(name, cfg, split="train", streaming=True)
        if args.shuffle_buffer > 0:
            ds = ds.shuffle(seed, buffer_size=args.shuffle_buffer)
        return iter(ds)

    srcs = []
    for name, cfg, field, tag, weight in SOURCES:
        srcs.append({"name": name, "cfg": cfg, "field": field, "tag": tag,
                     "budget": int(args.tokens * weight),
                     "it": make_stream(name, cfg, args.seed),
                     "val": [], "tok": 0, "done": False, "reconnects": 0})
        print(f"[{tag}] {name} -> budget {int(args.tokens*weight)/1e6:.0f}M tokens")

    MAX_RECONNECTS = 100

    def draw_batch(s):
        """Pull up to args.batch non-empty docs from source s; encode; route to
        val (first val_tokens) then train shards. Reconnects on network errors so
        a multi-hour build survives transient HF stream drops instead of dying."""
        texts = []
        for _ in range(args.batch):
            try:
                row = next(s["it"])
            except StopIteration:
                s["done"] = True
                break
            except Exception as e:   # network drop / socket error / broken stream
                s["reconnects"] += 1
                print(f"  [warn] {s['tag']} stream error #{s['reconnects']}: "
                      f"{str(e)[:70]} -> reconnecting", flush=True)
                if s["reconnects"] > MAX_RECONNECTS:
                    print(f"  [warn] {s['tag']} gave up after {MAX_RECONNECTS} reconnects", flush=True)
                    s["done"] = True
                    break
                time.sleep(2)
                s["it"] = make_stream(s["name"], s["cfg"], args.seed + s["reconnects"])
                break   # end this batch; resume next round with a fresh stream
            txt = row.get(s["field"]) or ""
            if txt:
                texts.append(txt)
        if not texts:
            return 0
        flat = []
        for e in tok.encode_batch(texts):
            ids = e.ids
            ids.append(eot)
            if len(s["val"]) < args.val_tokens:
                s["val"].extend(ids)
            else:
                flat.extend(ids)
            s["tok"] += len(ids)
        if flat:
            writer.write(np.asarray(flat, dtype=np.uint16))
        return len(texts)

    next_report = 10_000_000
    while not all(s["done"] or s["tok"] >= s["budget"] for s in srcs):
        for s in srcs:
            if s["done"] or s["tok"] >= s["budget"]:
                continue
            draw_batch(s)
        seen = sum(s["tok"] for s in srcs)
        if seen >= next_report:
            mix = ", ".join(f"{s['tok']/1e6:.0f}M {s['tag']}" for s in srcs)
            print(f"  ... {writer.total/1e6:.0f}M train tok on disk | per-source: {mix} | {time.time()-t0:.0f}s", flush=True)
            next_report += 10_000_000

    for s in srcs:
        val_path = os.path.join(args.out, f"val_{s['tag']}.bin")
        np.asarray(s["val"][:args.val_tokens], dtype=np.uint16).tofile(val_path)
        print(f"  [{s['tag']}] {s['tok']/1e6:.1f}M tokens | val -> {val_path} "
              f"({len(s['val'][:args.val_tokens])/1e6:.2f}M)")
    writer.close()
    print(f"DONE: {writer.total/1e6:.1f}M train tokens in {writer.idx + 1} shards "
          f"({time.time()-t0:.0f}s) -> {args.out}/")


if __name__ == "__main__":
    main()
