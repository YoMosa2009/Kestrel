"""Persistent mind-state — serialize and restore a session (Pillar 1 / docs/05 §5).

Kestrel's GLA blocks carry an O(1) recurrent state per token: an (H, D, D) matrix
plus a short conv cache, per block. That state IS the session - what the model has
absorbed from the conversation so far, compressed. Writing it to disk and reading
it back makes a session survive a restart, which no other model this size does.

Two honest limits, both v0.1 architecture facts rather than bugs here:

  * The attention blocks carry no cross-segment KV cache (model.py:204), so only
    the GLA half of the hybrid persists. Restoring a session recovers the
    recurrent summary, not verbatim earlier tokens.
  * State is per-model. A state saved from Nano cannot be loaded into Mini, even
    though they share a tokenizer - the shapes differ. The header records which
    model wrote it and load refuses on mismatch.
"""

from __future__ import annotations

import os
import time

import torch

FORMAT = 1


def _fingerprint(model) -> dict:
    cfg = model.cfg
    return {
        "d_model": cfg.d_model, "n_heads": cfg.n_heads, "head_dim": cfg.head_dim,
        "n_entry": cfg.n_entry, "n_core": cfg.n_core, "n_exit": cfg.n_exit,
        "r_default": cfg.r_default, "conv_kernel": cfg.conv_kernel,
    }


def save_session(state: dict, path: str, model, label: str = "") -> str:
    """Write a carried state dict (from model.forward(..., return_state=True))."""
    if not state:
        raise ValueError("refusing to save an empty session state")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "format": FORMAT,
        "saved": time.time(),
        "label": label,
        "model": _fingerprint(model),
        # detach + CPU so the file is portable and never pins GPU memory
        "state": {k: {kk: vv.detach().to("cpu") for kk, vv in v.items()}
                  for k, v in state.items() if v is not None},
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_session(path: str, model, device=None, strict: bool = True) -> dict:
    """Read a session back, verifying it came from a compatible model."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") != FORMAT:
        raise ValueError(f"session format {ck.get('format')} != {FORMAT}")
    want, got = _fingerprint(model), ck.get("model", {})
    if strict and want != got:
        diff = {k: (got.get(k), want[k]) for k in want if got.get(k) != want[k]}
        raise ValueError(
            "session was written by a different model config "
            f"(saved, current): {diff}")
    device = device or next(model.parameters()).device
    return {k: {kk: vv.to(device) for kk, vv in v.items()}
            for k, v in ck["state"].items()}


def session_info(path: str) -> dict:
    """Header only - cheap enough to list a directory of sessions."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    st = ck.get("state", {})
    n_bytes = sum(vv.numel() * vv.element_size()
                  for v in st.values() for vv in v.values())
    return {
        "label": ck.get("label", ""),
        "saved": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ck.get("saved", 0))),
        "blocks": len(st),
        "bytes": n_bytes,
        "model": ck.get("model", {}),
    }


@torch.no_grad()
def absorb(model, tokenizer, text: str, state: dict | None = None,
           device=None, chunk: int = 512) -> dict:
    """Run text through the model purely to update the session state.

    Long inputs are fed in chunks and the state is carried between them, which is
    the whole point of an O(1) recurrent state: absorbing a 50k-token document
    costs constant memory, unlike extending an attention context.
    """
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    ids = tokenizer.encode(text).ids
    carry = state if state is not None else {}
    for i in range(0, len(ids), chunk):
        piece = ids[i: i + chunk]
        if not piece:
            continue
        x = torch.tensor([piece], dtype=torch.long, device=device)
        _, _, carry = model(x, targets=None, state=carry, return_state=True)
    if was_training:
        model.train()
    return carry
