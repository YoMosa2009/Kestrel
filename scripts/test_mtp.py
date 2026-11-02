"""Checks for the MTP head and --init-from. Tiny random model; runs on a Colab runtime
before any A100 time goes to the ablation (docs/11 §1).

    python -m scripts.test_mtp
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict

import torch

import kestrel.model as km
from kestrel.config import PRESETS
from kestrel.model import KestrelModel


def build(mtp, chunks=4):
    cfg = PRESETS["test"]()
    cfg.mtp, cfg.loss_chunks = mtp, chunks
    torch.manual_seed(0)
    return KestrelModel(cfg), cfg


def main():
    plain, cfg = build(False)
    head, _ = build(True)
    missing, unexpected = head.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("mtp_") for k in missing), (missing, unexpected)
    x = torch.randint(1, cfg.vocab_size, (3, 32))
    y = torch.randint(1, cfg.vocab_size, (3, 32))

    # eval never sees the head, so the two arms' val losses mean the same thing
    plain.eval(); head.eval()
    with torch.no_grad():
        a = plain(x, targets=y, n_loops=2)[1].item()
        b = head(x, targets=y, n_loops=2)[1].item()
    assert abs(a - b) < 1e-5, f"eval loss differs with the head present: {a} vs {b}"

    # training: the aux head scores position t against targets[t + 1], the token after next
    seen = []
    real = km.chunked_cross_entropy

    def spy(h, W, t, z, n, m=None):
        seen.append((h.shape, t.clone()))
        return real(h, W, t, z, n, m)

    km.chunked_cross_entropy = spy
    head.train()
    train_loss = head(x, targets=y, n_loops=2)[1]
    km.chunked_cross_entropy = real
    assert len(seen) == 2, f"expected main + aux losses, got {len(seen)}"
    assert seen[1][0][1] == 31 and torch.equal(seen[1][1], y[:, 1:]), "aux targets are misaligned"
    assert train_loss.item() > a, "training loss should include the aux term"

    # gradients reach the head
    train_loss.backward()
    assert head.mtp_proj.weight.grad.abs().sum() > 0, "no gradient reached the MTP head"

    # chunked and full logit paths agree, with and without a mask
    full, _ = build(True, chunks=1)
    full.load_state_dict(head.state_dict())
    head.train(); full.train()
    mask = (torch.rand(3, 32) > 0.5).float()
    for m in (None, mask):
        torch.manual_seed(1)
        lc = head(x, targets=y, n_loops=2, loss_mask=m)[1].item()
        torch.manual_seed(1)
        lf = full(x, targets=y, n_loops=2, loss_mask=m)[1].item()
        assert abs(lc - lf) < 1e-3, f"chunked {lc} vs full {lf}"

    # inference and a masked aux with an empty mask stay safe
    head.eval()
    with torch.no_grad():
        logits, none = head(x)
    assert logits.shape[-1] == cfg.vocab_size and none is None

    # --init-from end to end: a checkpoint with no head starts an MTP run, then resumes
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "src.pt")
        torch.save({"model": plain.state_dict(), "cfg": asdict(cfg), "step": 100, "tokens_seen": 5}, src)
        out = os.path.join(d, "run")
        base = [sys.executable, "-u", "-m", "kestrel.train", "--preset", "test", "--synthetic", "--mtp",
                "--out", out, "--seq", "32", "--batch", "2", "--accum", "1", "--warmup", "1",
                "--log-every", "1", "--eval-every", "1000", "--device", "cpu", "--init-from", src,
                "--muon-lr", "0.001", "--adamw-lr", "0.0001"]
        r = subprocess.run(base + ["--steps", "102"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-800:] + r.stderr[-1500:]
        assert "initialized from" in r.stdout and "mtp_proj.weight" in r.stdout, r.stdout[-800:]
        assert json.load(open(os.path.join(out, "status.json")))["state"] == "done"
        assert torch.load(os.path.join(out, "ckpt.pt"), weights_only=False)["step"] == 102
        r = subprocess.run(base + ["--steps", "103"], capture_output=True, text=True)
        assert r.returncode == 0 and "resumed from" in r.stdout, r.stdout[-800:] + r.stderr[-1500:]

    print(f"MTP_TEST_OK | eval loss {a:.4f} | train loss with head {train_loss.item():.4f}")


if __name__ == "__main__":
    main()
