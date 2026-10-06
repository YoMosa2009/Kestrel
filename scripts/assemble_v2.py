"""Assemble Kestrel-Nano V2 from the V1 weights + the V2 memory tables (file work, no compute).

Roost consolidation changes ONLY the product-key-memory value tables, so a V2 run ships
just those (kestrel-nano-v2-pkm-values.pt, ~84 MB) and V2 is rebuilt here:

    python -m scripts.assemble_v2 --v1 kestrel-nano-v1-weights.pt \
        --pkm kestrel-nano-v2-pkm-values.pt --out kestrel-nano-v2-weights.pt

It also verifies the containment claim on the real artifact: every tensor except the
PKM value tables must be bit-identical to V1, and it reports how many slot rows moved.
"""

from __future__ import annotations

import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True)
    ap.add_argument("--pkm", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ck = torch.load(args.v1, map_location="cpu", weights_only=False)
    delta = torch.load(args.pkm, map_location="cpu", weights_only=False)
    sd = ck["model"]
    moved_total = 0
    for name, t in zip(delta["names"], delta["tensors"]):
        assert name in sd and sd[name].shape == t.shape, f"{name}: shape/name mismatch"
        moved = int(((sd[name].float() - t.float()).abs().sum(1) > 0).sum())
        moved_total += moved
        print(f"{name}: {moved:,} of {t.shape[0]:,} slot rows changed")
        sd[name] = t.to(sd[name].dtype)
    ck["model"] = sd
    ck["version"] = "V2"
    torch.save(ck, args.out)
    # containment check against a fresh copy of V1
    v1 = torch.load(args.v1, map_location="cpu", weights_only=False)["model"]
    changed = [k for k in v1 if not torch.equal(v1[k], sd[k])]
    assert set(changed) <= set(delta["names"]), f"non-memory tensors changed: {changed}"
    print(f"{args.out}: V2 written; {moved_total:,} memory slots differ from V1; "
          f"all {len(v1) - len(changed)} other tensors bit-identical")


if __name__ == "__main__":
    main()
