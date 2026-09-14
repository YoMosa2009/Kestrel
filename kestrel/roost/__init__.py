"""Roost — Kestrel's lifelong-learning protocol (docs/05 §5, MASTER-PLAN §4.3).

Deployment is a phase of learning, not the end of it. Roost has three parts:

  store.py        episodic memory: teachables -> declarative rewrites (+paraphrases)
                  -> embedded -> retrievable into context immediately
  consolidate.py  the nightly "sleep" job: 50/50 new+replay, updates ONLY the PKM
                  value slots the new material activates, gated on a frozen suite
                  with rollback
  session.py      persistent mind-state: serialize/restore the GLA recurrent state
                  so a session survives a restart

The safety argument, in one line: plasticity is architecturally confined to sparse
memory tissue, so the worst outcome of a bad night is "it didn't learn the fact",
never "it forgot how to code".
"""

from kestrel.roost.store import EpisodicStore, Teachable, rewrite
from kestrel.roost.session import save_session, load_session
from kestrel.roost.consolidate import consolidate, ConsolidationResult

__all__ = [
    "EpisodicStore", "Teachable", "rewrite",
    "save_session", "load_session",
    "consolidate", "ConsolidationResult",
]
