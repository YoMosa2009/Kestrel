"""Roost mechanics tests. Run: python -m kestrel.roost.test_roost"""
import os, tempfile, torch
from kestrel.config import PRESETS
from kestrel.model import KestrelModel
from kestrel.roost.store import EpisodicStore, rewrite
from kestrel.roost.consolidate import _slot_counts, select_slots, _pkm_modules
from kestrel.roost.session import save_session, load_session, session_info

d = tempfile.mkdtemp(); dev = torch.device("cpu")
cfg = PRESETS["test"](); model = KestrelModel(cfg).to(dev).eval()
fails = []
def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    if not cond: fails.append(name)

print("=== 1. rewrite ===")
t = rewrite("my lab GPU is an RTX 3060")
check("produces variants", len(t.variants) > 0, f"{len(t.variants)} variants")

print("=== 2. store round-trip ===")
s = EpisodicStore(os.path.join(d,"ep.jsonl"))
s.extend(["The capital of Varnal is Dreymoor.", "The currency of Threx is the drell."])
s2 = EpisodicStore(os.path.join(d,"ep.jsonl"))
check("persists", [i.text for i in s.items]==[i.text for i in s2.items])

print("=== 3. slot selection ===")
pkms = _pkm_modules(model)
xa = torch.randint(0,cfg.vocab_size,(2,64)); xb = torch.randint(0,cfg.vocab_size,(2,64))
nc = _slot_counts(model,[xa],dev); rc = _slot_counts(model,[xb],dev)
sel = select_slots(nc, rc, top_t=16)
check("selected only activated slots", bool((nc[0][sel[0]]>0).all()), f"{sel[0].numel()} selected")

print("=== 4. containment: masking alone (wd=0.01) LEAKS ===")
def run(wd, restore):
    m = KestrelModel(cfg).to(dev); pk = _pkm_modules(m)[0]
    for p in m.parameters(): p.requires_grad_(False)
    pk.values.weight.requires_grad_(True)
    mask = torch.zeros(pk.values.num_embeddings,1); mask[sel[0]] = 1.0
    keep = (~mask.bool()).squeeze(1)
    snap = pk.values.weight.detach().clone()
    opt = torch.optim.AdamW([pk.values.weight], lr=1e-2, weight_decay=wd)
    m.train(); _, loss = m(xa, targets=torch.randint(0,cfg.vocab_size,(2,64)))
    loss.backward(); pk.values.weight.grad.mul_(mask); opt.step()
    if restore:
        with torch.no_grad(): pk.values.weight[keep] = snap[keep]
    moved = ((pk.values.weight.detach()-snap).abs().sum(1) > 0).nonzero().flatten()
    return set(moved.tolist()) <= set(sel[0].tolist()), moved.numel()
ok, n = run(0.01, False); check("wd=0.01, mask only -> LEAKS (expected)", not ok, f"{n} rows moved")
ok, n = run(0.01, True);  check("wd=0.01 + row restore -> contained", ok, f"{n} rows moved")
ok, n = run(0.0,  True);  check("wd=0.0  + row restore -> contained", ok, f"{n} rows moved")

print("=== 5. dense weights never receive gradient ===")
m = KestrelModel(cfg).to(dev)
for p in m.parameters(): p.requires_grad_(False)
_pkm_modules(m)[0].values.weight.requires_grad_(True)
m.train(); _, l = m(xa, targets=torch.randint(0,cfg.vocab_size,(2,64))); l.backward()
dense = [n for n,p in m.named_parameters() if 'values' not in n and p.grad is not None and p.grad.abs().sum()>0]
check("no dense gradients", not dense, f"{len(dense)} dense params with grad")

print("=== 6. session round-trip ===")
model.eval(); x = torch.randint(0,cfg.vocab_size,(1,32))
_,_,st = model(x, targets=None, state={}, return_state=True)
p = os.path.join(d,"s.pt"); save_session(st,p,model,label="t"); back = load_session(p,model,dev)
check("restores identical", all(torch.equal(st[k][kk].cpu(),back[k][kk].cpu()) for k in st for kk in st[k]), f"{len(st)} blocks")
cfg2 = PRESETS["s"](); other = KestrelModel(cfg2)
try:
    load_session(p, other, dev); check("rejects wrong model", False)
except ValueError:
    check("rejects wrong model", True)

print()
print("ALL PASS" if not fails else f"FAILURES: {fails}")
raise SystemExit(1 if fails else 0)
