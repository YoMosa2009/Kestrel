"""Episodic store — the grounding tier of Roost (docs/05 §5).

Anything worth remembering is rewritten into clean declarative statements with a
few paraphrases, embedded, and appended to a local store. Top-k retrieval pulls
them back into context at query time, so a taught fact is usable IMMEDIATELY -
before any consolidation has run. Consolidation later moves it from "retrieved"
to "known".

The paraphrase step is SEAL's insight minus the RL: restating material before
learning it measurably improves what sticks. We generate variants from templates
rather than from the model, because a 157M base model cannot yet paraphrase
reliably - and a bad paraphrase teaches a wrong fact.

Embeddings come from Kestrel itself (mean-pooled final hidden states), so the
store has no external dependency and stays portable between Nano and Mini, which
share a tokenizer.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, asdict, field
from typing import Iterable, Sequence

import torch


# --------------------------------------------------------------------- records

@dataclass
class Teachable:
    """One thing the user taught, plus its rewrites."""
    text: str                                  # the canonical declarative form
    variants: list[str] = field(default_factory=list)
    kind: str = "fact"                         # fact | correction | preference
    source: str = "user"
    ts: float = field(default_factory=time.time)
    uses: int = 0                              # retrieval hit count (for compaction)
    last_used: float = 0.0

    def all_forms(self) -> list[str]:
        return [self.text] + self.variants


# ------------------------------------------------------------------- rewriting

# Template rewrites. Deliberately conservative: each pattern only fires when it
# is confident, and the fallback is to keep the sentence as-is. A paraphrase that
# changes meaning is worse than no paraphrase.
_PATTERNS: list[tuple[str, list[str]]] = [
    (r"^(?:my|the)\s+(.+?)\s+is\s+(.+?)\.?$",
     ["The {0} is {1}.", "{0}: {1}.", "Regarding the {0} - it is {1}."]),
    (r"^(.+?)\s+lives?\s+in\s+(.+?)\.?$",
     ["{0} lives in {1}.", "{0} is based in {1}.", "The home of {0} is {1}."]),
    (r"^(.+?)\s+(?:means?|stands for)\s+(.+?)\.?$",
     ["{0} means {1}.", "{0} refers to {1}.", "The meaning of {0} is {1}."]),
    (r"^(?:i|we)\s+(?:prefer|like|want)\s+(.+?)\.?$",
     ["The user prefers {0}.", "Preference: {0}.", "When in doubt, choose {0}."]),
    (r"^(.+?)\s+(?:was|were)\s+(.+?)\.?$",
     ["{0} was {1}.", "It is recorded that {0} was {1}.", "{0}: {1}."]),
]


def rewrite(raw: str, kind: str = "fact", max_variants: int = 3) -> Teachable:
    """Turn a raw utterance into a declarative statement plus paraphrases."""
    text = " ".join(raw.strip().split())
    if not text:
        raise ValueError("nothing to remember")
    if not text.endswith((".", "!", "?")):
        text += "."

    variants: list[str] = []
    low = text.lower().rstrip(".")
    for pat, forms in _PATTERNS:
        m = re.match(pat, low)
        if not m:
            continue
        groups = [g.strip() for g in m.groups()]
        for f in forms:
            v = f.format(*groups)
            v = v[0].upper() + v[1:]
            if v.lower() != text.lower() and v not in variants:
                variants.append(v)
        break

    return Teachable(text=text, variants=variants[:max_variants], kind=kind)


# ---------------------------------------------------------------- the embedder

class KestrelEmbedder:
    """Mean-pooled final hidden states from the model itself.

    Uses the same weights the model reasons with, so retrieval similarity is in
    the model's own representation space. No external embedding model, no extra
    dependency, and portable across Nano/Mini since they share a tokenizer.
    """

    def __init__(self, model, tokenizer, device=None, max_len: int = 256):
        self.model = model
        self.tok = tokenizer
        self.device = device or next(model.parameters()).device
        self.max_len = max_len

    @torch.no_grad()
    def __call__(self, texts: Sequence[str]) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        out = []
        for t in texts:
            ids = self.tok.encode(t).ids[: self.max_len]
            if not ids:
                ids = [0]
            x = torch.tensor([ids], dtype=torch.long, device=self.device)
            h = self.model.embed(x)
            h = self.model._run(self.model.entry, h, None, "e")
            for r in range(self.model.cfg.r_default):
                h = self.model._run(self.model.core, h, None, f"c{r}.")
            h = self.model._run(self.model.exit, h, None, "x")
            h = self.model.norm_f(h)
            v = h.float().mean(dim=1).squeeze(0)
            out.append(v / (v.norm() + 1e-8))
        if was_training:
            self.model.train()
        return torch.stack(out)


# ------------------------------------------------------------------- the store

class EpisodicStore:
    """Append-only JSONL of teachables + a cached embedding matrix.

    JSONL because it survives partial writes, is inspectable by hand, and can be
    diffed. Embeddings are cached alongside and rebuilt whenever the two fall out
    of sync, so deleting the cache is always a safe repair.
    """

    def __init__(self, path: str = "experiments/roost/episodic.jsonl"):
        self.path = path
        self.emb_path = os.path.splitext(path)[0] + ".emb.pt"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.items: list[Teachable] = []
        self._emb: torch.Tensor | None = None
        self.load()

    # --- persistence ---
    def load(self):
        self.items = []
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.items.append(Teachable(**json.loads(line)))
                        except Exception:
                            continue          # skip a corrupt line, never crash
        if os.path.exists(self.emb_path):
            try:
                e = torch.load(self.emb_path, map_location="cpu")
                self._emb = e if e.shape[0] == len(self.items) else None
            except Exception:
                self._emb = None

    def flush(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for it in self.items:
                f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)
        if self._emb is not None:
            torch.save(self._emb, self.emb_path)

    # --- writing ---
    def add(self, raw: str, kind: str = "fact") -> Teachable:
        t = rewrite(raw, kind)
        self.items.append(t)
        self._emb = None                        # invalidate; rebuilt on demand
        self.flush()
        return t

    def extend(self, raws: Iterable[str], kind: str = "fact") -> list[Teachable]:
        made = [rewrite(r, kind) for r in raws]
        self.items.extend(made)
        self._emb = None
        self.flush()
        return made

    # --- reading ---
    def embeddings(self, embedder: KestrelEmbedder) -> torch.Tensor:
        if self._emb is None or self._emb.shape[0] != len(self.items):
            if not self.items:
                self._emb = torch.zeros(0, 1)
            else:
                self._emb = embedder([it.text for it in self.items]).cpu()
            torch.save(self._emb, self.emb_path)
        return self._emb

    def retrieve(self, query: str, embedder: KestrelEmbedder, k: int = 4
                 ) -> list[tuple[float, Teachable]]:
        """Top-k by cosine similarity, with usage stats updated for compaction."""
        if not self.items:
            return []
        E = self.embeddings(embedder)
        q = embedder([query]).cpu()[0]
        sims = (E @ q).tolist()
        order = sorted(range(len(sims)), key=lambda i: -sims[i])[:k]
        hits = []
        for i in order:
            self.items[i].uses += 1
            self.items[i].last_used = time.time()
            hits.append((sims[i], self.items[i]))
        self.flush()
        return hits

    def as_context(self, query: str, embedder: KestrelEmbedder, k: int = 4) -> str:
        """Retrieved memories formatted for prepending to a prompt."""
        hits = self.retrieve(query, embedder, k)
        if not hits:
            return ""
        lines = [f"- {t.text}" for _, t in hits]
        return "Known facts:\n" + "\n".join(lines) + "\n\n"

    # --- maintenance ---
    def compact(self, embedder: KestrelEmbedder, dup_threshold: float = 0.97,
                stale_days: float = 90.0, min_uses: int = 1) -> dict:
        """Weekly pass: merge near-duplicates, drop stale unused entries.

        Conservative by design - an entry is only dropped if it is BOTH old and
        never retrieved. Losing a fact the user taught is a worse failure than
        keeping a redundant one.
        """
        if len(self.items) < 2:
            return {"removed_dup": 0, "removed_stale": 0, "kept": len(self.items)}
        E = self.embeddings(embedder)
        drop: set[int] = set()
        for i in range(len(self.items)):
            if i in drop:
                continue
            for j in range(i + 1, len(self.items)):
                if j in drop:
                    continue
                if float(E[i] @ E[j]) >= dup_threshold:
                    self.items[i].uses += self.items[j].uses
                    drop.add(j)
        n_dup = len(drop)
        cutoff = time.time() - stale_days * 86400
        for i, it in enumerate(self.items):
            if i in drop:
                continue
            if it.ts < cutoff and it.uses < min_uses:
                drop.add(i)
        n_stale = len(drop) - n_dup
        self.items = [it for i, it in enumerate(self.items) if i not in drop]
        self._emb = None
        self.flush()
        return {"removed_dup": n_dup, "removed_stale": n_stale, "kept": len(self.items)}

    def __len__(self):
        return len(self.items)
