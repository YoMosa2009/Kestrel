"""Kestrel CPU smoke tests — run on every change:  python -m kestrel.smoke_test

Covers the correctness-critical claims of docs/04:
  1. chunked GLA scan == naive recurrence (incl. carried state, ragged tail)
  2. model forward: shapes, finiteness
  3. trainability: tiny overfit drives loss down
  4. streaming: split-sequence forward with state carry == full-sequence
     forward (pure-GLA config; exactness is the session-memory guarantee)
  5. PKM: gradients reach the values table and the gate
  6. presets: parameter budgets match docs/04 (printed)
"""

import sys
import time

import torch
import torch.nn.functional as F

from kestrel.config import PRESETS, KestrelConfig, kestrel_test
from kestrel.model import KestrelModel, gla_chunked_scan

torch.manual_seed(1337)
FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def gla_naive(q, k, v, log_g, s0=None):
    B, H, T, D = q.shape
    s = s0.clone() if s0 is not None else q.new_zeros(B, H, D, D)
    outs = []
    for t in range(T):
        s = log_g[:, :, t].exp()[..., None, None] * s \
            + k[:, :, t].unsqueeze(-1) @ v[:, :, t].unsqueeze(-2)
        outs.append((q[:, :, t].unsqueeze(-2) @ s).squeeze(-2))
    return torch.stack(outs, dim=2), s


def test_scan_equivalence():
    B, H, T, D = 2, 3, 100, 8          # T=100 with chunk=16 exercises padding
    q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
    log_g = -F.softplus(torch.randn(B, H, T))
    s0 = torch.randn(B, H, D, D) * 0.3

    for name, state in [("no initial state", None), ("with initial state", s0)]:
        o_c, s_c = gla_chunked_scan(q, k, v, log_g, state, chunk=16)
        o_n, s_n = gla_naive(q, k, v, log_g, state)
        do = (o_c - o_n).abs().max().item()
        ds = (s_c - s_n).abs().max().item()
        check(f"scan equivalence, {name}", do < 1e-4 and ds < 1e-4,
              f"max|d_o|={do:.2e} max|d_S|={ds:.2e}")


def test_forward_and_pkm():
    model = KestrelModel(kestrel_test())
    print("K-test:", model.param_report())
    idx = torch.randint(0, 2048, (2, 128))
    logits, loss = model(idx, targets=idx, n_loops=2)
    check("forward shapes", logits.shape == (2, 128, 2048), str(tuple(logits.shape)))
    check("loss finite", torch.isfinite(loss).item(), f"loss={loss.item():.3f}")

    loss.backward()
    pkm = model.core[1].pkm or model.entry[0].pkm  # site index 2 -> core[1]
    for blk in list(model.entry) + list(model.core) + list(model.exit):
        if blk.pkm is not None:
            pkm = blk.pkm
    vgrad = pkm.values.weight.grad
    check("PKM values receive grads",
          vgrad is not None and vgrad.abs().sum().item() > 0)
    check("PKM gate receives grads",
          pkm.gate.grad is not None and pkm.gate.grad.abs().item() > 0)

    model.train()
    _ = model(idx[:, :64])             # stochastic-loop sampling path
    check("stochastic loop sampling runs", True)


def test_overfit():
    torch.manual_seed(7)
    model = KestrelModel(kestrel_test())
    idx = torch.randint(0, 2048, (2, 128))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    model.train()
    first = last = None
    t0 = time.time()
    for step in range(60):
        _, loss = model(idx, targets=idx, n_loops=2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        first = first or loss.item()
        last = loss.item()
    check("tiny overfit reduces loss >40%", last < 0.6 * first,
          f"{first:.3f} -> {last:.3f} in 60 steps, {time.time()-t0:.1f}s")


def test_streaming_state_carry():
    # pure-GLA config: attn_every > n_blocks -> streaming must be EXACT
    cfg = kestrel_test()
    cfg = KestrelConfig(**{**cfg.__dict__, "attn_every": 999})
    model = KestrelModel(cfg).eval()
    idx = torch.randint(0, 2048, (2, 128))
    with torch.no_grad():
        full, _ = model(idx, n_loops=2)
        _, _, st = model(idx[:, :64], n_loops=2, return_state=True)
        second, _, _ = model(idx[:, 64:], n_loops=2, state=st, return_state=True)
    diff = (full[:, 64:] - second).abs().max().item()
    check("streaming state carry exact (pure-GLA)", diff < 2e-4,
          f"max|d_logits|={diff:.2e}")

    # hybrid config: states must at least exist, differ from init, be finite
    model2 = KestrelModel(kestrel_test()).eval()
    with torch.no_grad():
        _, _, st2 = model2(idx[:, :64], n_loops=2, return_state=True)
    ok = all(torch.isfinite(s["S"]).all() and s["S"].abs().sum() > 0
             for s in st2.values() if "S" in s)
    check("hybrid states finite & non-trivial", ok, f"{len(st2)} block states")


def test_presets():
    expected_nonemb = {"test": (0.5, 3), "s": (18, 30), "m": (50, 80),
                       "nano": (100, 155), "mini": (400, 520)}
    for name, fn in PRESETS.items():
        cfg = fn()
        with torch.device("meta"):
            m = KestrelModel(cfg)
        total = sum(p.numel() for p in m.parameters()) / 1e6
        emb = m.embed.weight.numel() / 1e6
        lo, hi = expected_nonemb[name]
        check(f"preset K-{name} non-emb in [{lo},{hi}]M",
              lo <= total - emb <= hi,
              f"total={total:.1f}M non-emb={total-emb:.1f}M eff-depth={cfg.effective_depth}")


if __name__ == "__main__":
    print(f"torch {torch.__version__} | device: cpu\n" + "-" * 60)
    test_scan_equivalence()
    test_forward_and_pkm()
    test_overfit()
    test_streaming_state_carry()
    test_presets()
    print("-" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL SMOKE TESTS PASSED")
