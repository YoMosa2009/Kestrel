"""Memmap token dataloader (docs/03 "System-RAM discipline").

Data lives as flat uint16 `.bin` shards on disk; we np.memmap them and draw
random (seq_len+1) windows. Nothing is ever held in RAM (16 GB machine). The
sampler RNG state is serializable so a run resumes at the exact data cursor.
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional

import numpy as np
import torch


class ShardedTokenLoader:
    """Random-window sampler over a set of uint16 shards.

    Shards are memory-mapped (mode='r'); a batch is gathered by choosing shards
    with probability proportional to their length, then random offsets within.
    """

    def __init__(self, files: List[str], seq_len: int, batch_size: int,
                 device: torch.device, seed: int = 1337, dtype=np.uint16):
        if not files:
            raise FileNotFoundError("ShardedTokenLoader got no shard files")
        self.files = list(files)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.arrays = [np.memmap(f, dtype=dtype, mode="r") for f in self.files]
        # usable start positions per shard (need seq_len+1 tokens after offset)
        self.usable = np.array([max(0, len(a) - seq_len - 1) for a in self.arrays],
                               dtype=np.int64)
        if self.usable.sum() == 0:
            raise ValueError("no shard is long enough for the requested seq_len")
        self.shard_p = self.usable / self.usable.sum()
        self.rng = np.random.default_rng(seed)

    @property
    def total_tokens(self) -> int:
        return int(sum(len(a) for a in self.arrays))

    def get_batch(self):
        xs = np.empty((self.batch_size, self.seq_len), dtype=np.int64)
        ys = np.empty((self.batch_size, self.seq_len), dtype=np.int64)
        shard_idx = self.rng.choice(len(self.arrays), size=self.batch_size, p=self.shard_p)
        for b, si in enumerate(shard_idx):
            arr = self.arrays[si]
            off = int(self.rng.integers(0, self.usable[si] + 1))
            window = np.asarray(arr[off: off + self.seq_len + 1], dtype=np.int64)
            xs[b] = window[:-1]
            ys[b] = window[1:]
        x = torch.from_numpy(xs).to(self.device, non_blocking=True)
        y = torch.from_numpy(ys).to(self.device, non_blocking=True)
        return x, y

    # --- resumability: the sampler RNG is the "dataloader cursor" ---
    def state_dict(self):
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng"]


def find_shards(data_dir: str, split: str = "train") -> List[str]:
    """Return sorted .bin shard paths under data_dir/<split>*.bin or data_dir/split/."""
    patterns = [
        os.path.join(data_dir, f"{split}*.bin"),
        os.path.join(data_dir, split, "*.bin"),
    ]
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(set(files))


def make_loaders(data_dir: str, seq_len: int, batch_size: int, device: torch.device,
                 seed: int = 1337):
    """Convenience: build a train loader and (if present) a val loader."""
    train_files = find_shards(data_dir, "train")
    val_files = find_shards(data_dir, "val")
    train = ShardedTokenLoader(train_files, seq_len, batch_size, device, seed)
    val: Optional[ShardedTokenLoader] = None
    if val_files:
        val = ShardedTokenLoader(val_files, seq_len, batch_size, device, seed + 1)
    return train, val
