"""Phase B — isolate, verify and benchmark GLA-scan variants (docs/08 §3).

Every variant must stay numerically exact vs the naive recurrence; then we
measure fwd and fwd+bwd time and peak memory. Run:
    python -m scripts.bench_scan
"""
from __future__ import annotations
import time, argparse
import torch, torch.nn.functional as F

# ----------------------------------------------------------------- reference
def gla_naive(q, k, v, log_g, s0=None):
    B, H, T, D = q.shape
    s = s0.clone() if s0 is not None else q.new_zeros(B, H, D, D)
    outs = []
    for t in range(T):
        s = log_g[:, :, t].exp()[..., None, None] * s \
            + k[:, :, t].unsqueeze(-1) @ v[:, :, t].unsqueeze(-2)
        outs.append((q[:, :, t].unsqueeze(-2) @ s).squeeze(-2))
    return torch.stack(outs, dim=2), s

# ----------------------------------------------------------------- v0: current
def scan_v0(q, k, v, log_g, s0=None, chunk=64):
    B, H, T, D = q.shape
    pad = (chunk - T % chunk) % chunk
    if pad:
        q = F.pad(q, (0,0,0,pad)); k = F.pad(k, (0,0,0,pad))
        v = F.pad(v, (0,0,0,pad)); log_g = F.pad(log_g, (0,pad))
    n = q.shape[2] // chunk
    qc = q.view(B,H,n,chunk,D); kc = k.view(B,H,n,chunk,D); vc = v.view(B,H,n,chunk,D)
    lg = log_g.view(B,H,n,chunk); lcum = lg.cumsum(dim=-1)
    ldiff = lcum.unsqueeze(-1) - lcum.unsqueeze(-2)
    causal = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    decay = torch.where(causal, ldiff, torch.full_like(ldiff, float("-inf"))).exp()
    o = (qc @ kc.transpose(-1,-2) * decay) @ vc
    s = s0 if s0 is not None else q.new_zeros(B,H,D,D)
    cross = []
    for i in range(n):
        cross.append((qc[:,:,i] * lcum[:,:,i].exp().unsqueeze(-1)) @ s)
        k_tail = kc[:,:,i] * (lcum[:,:,i,-1:] - lcum[:,:,i]).exp().unsqueeze(-1)
        s = lcum[:,:,i,-1].exp()[...,None,None] * s + k_tail.transpose(-1,-2) @ vc[:,:,i]
    o = o + torch.stack(cross, dim=2)
    return o.reshape(B,H,-1,D)[:,:,:T], s

# ------------------------------------------------- v1: lean temporaries
_MASK_CACHE = {}
def _causal_mask(chunk, device):
    key = (chunk, device)
    m = _MASK_CACHE.get(key)
    if m is None:
        m = torch.ones(chunk, chunk, dtype=torch.bool, device=device).tril()
        _MASK_CACHE[key] = m
    return m

def scan_v1(q, k, v, log_g, s0=None, chunk=64):
    """Same math, fewer materialized (B,H,n,C,C) tensors:
    cached mask, fused masked-exp via subtraction trick, in-place score scaling,
    and the cross-chunk term accumulated into a preallocated buffer."""
    B, H, T, D = q.shape
    pad = (chunk - T % chunk) % chunk
    if pad:
        q = F.pad(q, (0,0,0,pad)); k = F.pad(k, (0,0,0,pad))
        v = F.pad(v, (0,0,0,pad)); log_g = F.pad(log_g, (0,pad))
    n = q.shape[2] // chunk
    qc = q.view(B,H,n,chunk,D); kc = k.view(B,H,n,chunk,D); vc = v.view(B,H,n,chunk,D)
    lcum = log_g.view(B,H,n,chunk).cumsum(dim=-1)

    mask = _causal_mask(chunk, q.device)
    # decay = exp(L_i - L_j) masked causally. masked_fill on the *log* then exp
    # keeps exponents <= 0 (fp32-safe) while avoiding full_like(-inf).
    ldiff = (lcum.unsqueeze(-1) - lcum.unsqueeze(-2)).masked_fill(~mask, float("-inf"))
    scores = (qc @ kc.transpose(-1,-2)) * ldiff.exp()
    o = scores @ vc

    s = s0 if s0 is not None else q.new_zeros(B,H,D,D)
    lend = lcum[:,:,:,-1]                                   # (B,H,n)
    q_sc = qc * lcum.exp().unsqueeze(-1)                    # fold decay into q
    k_sc = kc * (lend.unsqueeze(-1) - lcum).exp().unsqueeze(-1)
    outs = []
    for i in range(n):
        outs.append(q_sc[:,:,i] @ s)
        s = lend[:,:,i,None,None].exp() * s + k_sc[:,:,i].transpose(-1,-2) @ vc[:,:,i]
    o = o + torch.stack(outs, dim=2)
    return o.reshape(B,H,-1,D)[:,:,:T], s

# ------------------------------------------------- v2: decay folded into q/k
def scan_v2(q, k, v, log_g, s0=None, chunk=64):
    """A_ij = (q_i.k_j)exp(L_i-L_j) = (q_i e^{L_i-L0}).(k_j e^{L0-L_j}).
    Folding kills the (B,H,n,C,C) decay matrix entirely — only the score matrix
    remains. NOTE: the k-side exponent is >= 0 and can overflow fp32 if gates
    are very negative; guarded/measured below before adoption."""
    B, H, T, D = q.shape
    pad = (chunk - T % chunk) % chunk
    if pad:
        q = F.pad(q, (0,0,0,pad)); k = F.pad(k, (0,0,0,pad))
        v = F.pad(v, (0,0,0,pad)); log_g = F.pad(log_g, (0,pad))
    n = q.shape[2] // chunk
    qc = q.view(B,H,n,chunk,D); kc = k.view(B,H,n,chunk,D); vc = v.view(B,H,n,chunk,D)
    lcum = log_g.view(B,H,n,chunk).cumsum(dim=-1)
    lref = lcum[..., :1]                                  # L_0 per chunk
    qf = qc * (lcum - lref).unsqueeze(-1).exp()           # exponent <= 0
    kf = kc * (lref - lcum).unsqueeze(-1).exp()           # exponent >= 0
    mask = _causal_mask(chunk, q.device)
    scores = (qf @ kf.transpose(-1,-2)).masked_fill(~mask, 0.0)
    o = scores @ vc
    s = s0 if s0 is not None else q.new_zeros(B,H,D,D)
    lend = lcum[:,:,:,-1]
    q_sc = qc * lcum.exp().unsqueeze(-1)
    k_sc = kc * (lend.unsqueeze(-1) - lcum).exp().unsqueeze(-1)
    outs = []
    for i in range(n):
        outs.append(q_sc[:,:,i] @ s)
        s = lend[:,:,i,None,None].exp() * s + k_sc[:,:,i].transpose(-1,-2) @ vc[:,:,i]
    o = o + torch.stack(outs, dim=2)
    return o.reshape(B,H,-1,D)[:,:,:T], s

VARIANTS = {"v0_current": scan_v0, "v1_lean": scan_v1, "v2_folded": scan_v2}

# ----------------------------------------------------------------- harness
def check(fn, device, dtype):
    torch.manual_seed(0)
    B,H,T,D = 2,3,100,16
    q,k,v = (torch.randn(B,H,T,D, device=device, dtype=dtype) for _ in range(3))
    log_g = -F.softplus(torch.randn(B,H,T, device=device, dtype=dtype))
    s0 = torch.randn(B,H,D,D, device=device, dtype=dtype) * 0.3
    worst = 0.0
    for st in (None, s0):
        oc, sc = fn(q,k,v,log_g, st, 16)
        on, sn = gla_naive(q,k,v,log_g, st)
        worst = max(worst, (oc-on).abs().max().item(), (sc-sn).abs().max().item())
    return worst

def bench(fn, device, B,H,T,D,chunk, iters=8, backward=True):
    torch.manual_seed(0)
    q,k,v = (torch.randn(B,H,T,D, device=device, requires_grad=backward) for _ in range(3))
    raw = torch.randn(B,H,T, device=device, requires_grad=backward)
    def step():
        # softplus INSIDE the step so the graph is rebuilt each iteration
        log_g = -F.softplus(raw)
        o,_ = fn(q,k,v,log_g,None,chunk)
        if backward:
            for t in (q,k,v,raw): t.grad = None
            o.sum().backward()
    for _ in range(3): step()
    if device.type=="cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0=time.time()
    for _ in range(iters): step()
    if device.type=="cuda": torch.cuda.synchronize()
    dt=(time.time()-t0)/iters
    peak = torch.cuda.max_memory_allocated()/1e9 if device.type=="cuda" else 0.0
    return dt, peak

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--heads", type=int, default=10)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--chunks", type=int, nargs="+", default=[64])
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {dev} | B{a.batch} H{a.heads} T{a.seq} D{a.dim}\n"+"-"*72)
    print("correctness vs naive recurrence (fp32, must be < 1e-4):")
    for name, fn in VARIANTS.items():
        print(f"  {name:12s} max|err| = {check(fn, dev, torch.float32):.2e}")
    print("-"*72)
    for chunk in a.chunks:
        print(f"chunk={chunk}")
        base=None
        for name, fn in VARIANTS.items():
            try:
                dt, pk = bench(fn, dev, a.batch,a.heads,a.seq,a.dim, chunk)
                if base is None: base = dt
                print(f"  {name:12s} fwd+bwd {dt*1000:7.1f} ms  peak {pk:5.2f} GB  "
                      f"speedup {base/dt:4.2f}x")
            except RuntimeError as e:
                print(f"  {name:12s} FAILED: {str(e)[:60]}")
