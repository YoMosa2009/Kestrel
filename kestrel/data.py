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

    def get_batch(self, batch_size: int | None = None, to_device: bool = True):
        """Gather `batch_size` random windows.

        Staged through a pinned-memory buffer so the host->device copy is a real
        async DMA; from pageable memory `non_blocking=True` is silently ignored
        and every copy blocks the compute stream.
        """
        bs = self.batch_size if batch_size is None else batch_size
        buf = np.empty((2, bs, self.seq_len), dtype=np.int64)
        shard_idx = self.rng.choice(len(self.arrays), size=bs, p=self.shard_p)
        for b, si in enumerate(shard_idx):
            arr = self.arrays[si]
            off = int(self.rng.integers(0, self.usable[si] + 1))
            window = np.asarray(arr[off: off + self.seq_len + 1], dtype=np.int64)
            buf[0, b] = window[:-1]
            buf[1, b] = window[1:]
        t = torch.from_numpy(buf)
        if not to_device:
            return t[0], t[1]
        if self.device.type == "cuda":
            t = t.pin_memory()
        t = t.to(self.device, non_blocking=True)
        return t[0], t[1]

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


# ---------------------------------------------------------------- curriculum

# Staged mixture (docs/05 §3, docs/10 §4): broad early -> technical middle ->
# premium anneal. Weights are per-domain sampling probabilities; they need not
# sum to 1 (they are normalized). Domains absent from data_dir are dropped and
# the rest renormalized, so this is safe on partial corpora.
CURRICULUM_STAGES = [
    # (until_fraction_of_training, weights)
    (0.40, {"web": 35, "know": 18, "math": 8,  "code": 25, "cli": 4,
            "repo": 4,  "inst": 3,  "docs": 2, "tool": 0.7, "sec": 0.3}),
    (0.85, {"web": 20, "know": 12, "math": 11, "code": 34, "cli": 8,
            "repo": 7,  "inst": 5,  "docs": 2, "tool": 0.7, "sec": 0.3}),
    # anneal — premium slice: knowledge, math, instruct up; raw web down.
    # This is the lever for pulling general/conversational ability back up.
    (1.01, {"web": 8,  "know": 22, "math": 18, "code": 20, "cli": 6,
            "repo": 6,  "inst": 15, "docs": 2, "tool": 2,   "sec": 1}),
]


def discover_domains(data_dir: str, split: str = "train"):
    """Map domain tag -> shard files, from <split>_<tag>_NNN.bin."""
    out = {}
    for f in sorted(glob.glob(os.path.join(data_dir, f"{split}_*.bin"))):
        base = os.path.basename(f)[len(split) + 1:-4]      # strip "train_" / ".bin"
        tag = base.rsplit("_", 1)[0] if "_" in base else base
        out.setdefault(tag, []).append(f)
    return out


class CurriculumLoader:
    """Samples each batch across per-domain shards using stage-dependent weights.

    `set_progress(frac)` switches the mixture as training advances, so the run
    can start broad, go technical, and finish on a premium anneal without
    rebuilding any data.
    """

    def __init__(self, data_dir, seq_len, batch_size, device, seed=1337,
                 stages=None, split="train"):
        self.stages = stages or CURRICULUM_STAGES
        self.batch_size = batch_size
        self.device = device
        doms = discover_domains(data_dir, split)
        if not doms:
            raise FileNotFoundError(f"no {split}_<tag>_*.bin shards in {data_dir}")
        self.loaders = {}
        for i, (tag, files) in enumerate(sorted(doms.items())):
            try:
                self.loaders[tag] = ShardedTokenLoader(files, seq_len, 1, device, seed + i)
            except ValueError:
                pass                                  # shards shorter than seq_len
        self.rng = np.random.default_rng(seed)
        self.set_progress(0.0)

    @property
    def total_tokens(self):
        return sum(l.total_tokens for l in self.loaders.values())

    def set_progress(self, frac: float):
        for until, w in self.stages:
            if frac < until:
                weights = w
                break
        else:
            weights = self.stages[-1][1]
        tags = [t for t in self.loaders if weights.get(t, 0) > 0]
        p = np.array([weights[t] for t in tags], dtype=np.float64)
        self.tags, self.p = tags, p / p.sum()
        self.stage_weights = {t: round(float(x), 4) for t, x in zip(tags, self.p)}

    def get_batch(self):
        """One gather + one pinned H2D copy per ACTIVE DOMAIN.

        The old form called each domain loader once per sequence (batch_size=1),
        so a batch of 24 cost 48 separate pageable, blocking H2D copies. Sampling
        is unchanged: the multinomial still decides how many sequences each domain
        contributes, they are just fetched in one call.
        """
        counts = self.rng.multinomial(self.batch_size, self.p)
        xs, ys = [], []
        for tag, n in zip(self.tags, counts):
            n = int(n)
            if n == 0:
                continue
            x, y = self.loaders[tag].get_batch(batch_size=n)
            xs.append(x); ys.append(y)
        return torch.cat(xs, 0), torch.cat(ys, 0)

    def state_dict(self):
        return {t: l.state_dict() for t, l in self.loaders.items()}

    def load_state_dict(self, state):
        # A checkpoint written by the flat ShardedTokenLoader stores {"rng": ...},
        # which has no domain tags and so restores nothing here. That is harmless
        # (the per-domain loaders just start from their seeds) but it must not be
        # silent, or a resume looks like it restored a cursor it never had.
        unknown = [t for t in (state or {}) if t not in self.loaders]
        if unknown:
            print(f"  [data] WARNING: checkpoint loader state has no curriculum "
                  f"domains {unknown}; per-domain samplers start from their seeds. "
                  f"(Expected when resuming a run made by the non-curriculum loader.)")
        for t, s in (state or {}).items():
            if t in self.loaders:
                self.loaders[t].load_state_dict(s)


def make_domain_val_loaders(data_dir, seq_len, batch_size, device, seed=1337):
    """One loader per val_<tag>.bin, for per-domain validation loss."""
    out = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "val_*.bin"))):
        tag = os.path.basename(f)[4:-4]
        try:
            out[tag] = ShardedTokenLoader([f], seq_len, batch_size, device, seed)
        except (ValueError, FileNotFoundError):
            pass
    return out


class PrefetchLoader:
    """Runs any loader's get_batch() on a background thread.

    The shards live on a 5400-rpm HDD and are sampled at random offsets, so a
    micro-batch costs real seek time that is otherwise serialized in front of the
    GPU. A worker thread keeps `depth` batches ready, so that read overlaps the
    previous micro-batch's compute. The numpy gather and the pinned H2D copy both
    release the GIL, so one thread is enough.

    Resume caveat (deliberate, documented): the worker runs ahead of the training
    loop, so a checkpoint's RNG cursor is up to `depth` batches ahead of the
    batches actually consumed. On resume that re-draws a handful of windows out of
    billions of tokens - statistically irrelevant, but it means the loader cursor
    is approximate rather than exact.
    """

    def __init__(self, inner, depth: int = 3):
        import queue
        import threading
        self.inner = inner
        self.q = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._work, daemon=True)
        self._t.start()

    def _work(self):
        while not self._stop.is_set():
            try:
                self.q.put(self.inner.get_batch(), timeout=1.0)
            except Exception:
                if self._stop.is_set():
                    return
                continue

    def get_batch(self):
        return self.q.get()

    def close(self):
        self._stop.set()

    # --- transparently forward everything the trainer asks of a loader ---
    def set_progress(self, frac):
        return self.inner.set_progress(frac)

    @property
    def stage_weights(self):
        return self.inner.stage_weights

    @property
    def total_tokens(self):
        return self.inner.total_tokens

    @property
    def loaders(self):
        return self.inner.loaders

    @property
    def files(self):
        return self.inner.files

    def state_dict(self):
        return self.inner.state_dict()

    def load_state_dict(self, state):
        return self.inner.load_state_dict(state)
