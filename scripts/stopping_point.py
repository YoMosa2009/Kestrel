"""Where should the stable phase stop?

Fits the observed val_mean probes to a power law L(N) = Linf + A*N^-alpha and
reports what another day of training is actually worth. The point is NOT to
predict a final loss - it is to find where the marginal gain per day falls below
what the wall-clock is worth to you.

Two things this deliberately does NOT hide:

  * With few probes a 3-parameter fit is under-determined. Linf (the irreducible
    loss) is the parameter the data constrains worst, and it drives the whole
    extrapolation. So the fit is reported across a BAND of Linf values rather
    than as a single curve, and the spread between them is the honest error bar.

  * The numbers describe the STABLE phase only. The WSD anneal delivers a
    further step-change at the end that this curve cannot see, so every
    prediction here is pessimistic about the finished model.

Re-run it as probes accumulate; the band narrows.

    python -m scripts.stopping_point
    python -m scripts.stopping_point --tok-s 4050 --min-step 10500
"""

from __future__ import annotations

import argparse
import json
import math
import os


def load_probes(path: str, min_step: int):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    tok = {r["step"]: r["tokens_seen"] for r in rows
           if "probe" not in r and "tokens_seen" in r}
    steps = sorted(tok)
    if not steps:
        raise SystemExit("no training records with tokens_seen yet")

    def tokens_at(s):
        return tok[min(steps, key=lambda x: abs(x - s))]

    out = []
    for r in rows:
        p = r.get("probe")
        if not p or "val_by_domain" not in p:
            continue
        if r["step"] < min_step:
            continue
        v = p["val_by_domain"]
        out.append((r["step"], tokens_at(r["step"]), sum(v.values()) / len(v)))
    return sorted(out)


def fit(points, linf):
    """Least-squares on log(L - Linf) = log A - alpha*log N. Returns (A, alpha, resid)."""
    xs, ys = [], []
    for _, n, m in points:
        if m - linf <= 0:
            return None
        xs.append(math.log(n))
        ys.append(math.log(m - linf))
    k = len(xs)
    mx, my = sum(xs) / k, sum(ys) / k
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    a = my - b * mx
    resid = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return math.exp(a), -b, resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="experiments/nano_d/metrics.jsonl")
    ap.add_argument("--min-step", type=int, default=10500,
                    help="ignore probes before the LR reached plateau")
    ap.add_argument("--tok-s", type=float, default=4050.0)
    ap.add_argument("--worth-it", type=float, default=0.010,
                    help="stop when one more DAY buys less val_mean than this")
    args = ap.parse_args()

    pts = load_probes(args.metrics, args.min_step)
    if len(pts) < 2:
        raise SystemExit(f"only {len(pts)} usable probes; need at least 2")

    print(f"probes at plateau LR (step >= {args.min_step}):")
    for s, n, m in pts:
        print(f"   step {s:6d}   {n/1e9:6.3f}B tok   val_mean {m:.4f}")

    cur_n = pts[-1][1]
    per_day = args.tok_s * 86400

    # --- model-free: what did the last interval actually buy? ----------------
    print("\nobserved marginal gain (model-free):")
    for i in range(1, len(pts)):
        dn = pts[i][1] - pts[i - 1][1]
        dl = pts[i - 1][2] - pts[i][2]
        print(f"   {pts[i-1][1]/1e9:.3f}B -> {pts[i][1]/1e9:.3f}B "
              f"(+{dn/1e9:.3f}B): val_mean -{dl:.4f}  "
              f"= -{dl/(dn/per_day):.4f} per day")

    # --- power-law band ------------------------------------------------------
    print(f"\npower-law fits across an Linf band ({len(pts)} points):")
    band = []
    for linf in (0.8, 1.0, 1.2, 1.4, 1.6):
        r = fit(pts, linf)
        if r is None:
            continue
        A, al, resid = r
        band.append((linf, A, al))
        print(f"   Linf={linf:.1f}  alpha={al:.3f}  resid={resid:.2e}")

    if not band:
        raise SystemExit("no usable fit; need more probes")

    print(f"\npredicted val_mean at the END OF THE STABLE PHASE")
    print(f"(anneal adds a further gain this curve cannot see)\n")
    hdr = "   {:>7s} {:>7s}".format("tokens", "+days")
    for linf, _, _ in band:
        hdr += f"  Linf={linf:.1f}"
    print(hdr)
    for tgt in (2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.31):
        n = tgt * 1e9
        if n <= cur_n:
            continue
        days = (n - cur_n) / per_day
        row = f"   {tgt:6.2f}B {days:6.1f}d"
        for linf, A, al in band:
            row += f"  {linf + A * n ** (-al):8.4f}"
        print(row)

    # --- the actual decision -------------------------------------------------
    print(f"\nwhere one more DAY buys less than {args.worth_it:.3f} val_mean:")
    for linf, A, al in band:
        n = cur_n
        for _ in range(400):
            gain = (linf + A * n ** (-al)) - (linf + A * (n + per_day) ** (-al))
            if gain < args.worth_it:
                break
            n += per_day
        print(f"   Linf={linf:.1f}  ->  stop near {n/1e9:.2f}B cumulative "
              f"({(n-cur_n)/per_day:5.1f} days from now)")

    print("\nRe-run as probes accumulate - the band narrows and the answer firms up.")


if __name__ == "__main__":
    main()
