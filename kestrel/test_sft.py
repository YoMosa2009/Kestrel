"""SFT loader + masked-loss tests.  Run: python -m kestrel.test_sft <sft_data_dir>"""
import numpy as np, torch, torch.nn.functional as F
from kestrel.data import SFTLoader
from kestrel.config import KestrelConfig
from kestrel.model import KestrelModel, chunked_cross_entropy

import sys; d = sys.argv[1]
dev = torch.device("cpu")
fails=[]
def check(n,c,d_=""):
    print(f"  [{'PASS' if c else 'FAIL'}] {n}  {d_}")
    if not c: fails.append(n)

print("=== SFTLoader ===")
ld = SFTLoader(d, 512, 4, dev, "train", seed=1)
x,y,m = ld.get_batch()
check("shapes", x.shape==(4,512) and y.shape==(4,512) and m.shape==(4,512), f"{tuple(x.shape)}")
check("mask is 0/1 float", set(torch.unique(m).tolist())<= {0.0,1.0})
check("some tokens scored", float(m.sum())>0, f"{float(m.mean())*100:.0f}% scored")
check("windows start at an example boundary",
      all(int(s)==0 or int(np.asarray(ld.tokens)[int(s)-1])==0 for s in ld.starts[:200]))
# trailing partial example must be masked
row=0
eots=(y[row]==0).nonzero().flatten()
if eots.numel():
    last=int(eots[-1])
    check("mask zeroed after last complete example", float(m[row,last+1:].sum())==0.0,
          f"{m.shape[1]-last-1} trailing positions")

print("=== masked loss maths (z_loss=0 so the manual calc is exact) ===")
cfg = KestrelConfig(vocab_size=64, d_model=32, n_entry=1, n_core=1, n_exit=1,
                    n_heads=2, n_kv_heads=1, d_ff=64, pkm_sites=(), pkm_n_keys=8,
                    pkm_d_key=8, pkm_topk=2, r_max=1, r_default=1, r_train_probs=(1.0,),
                    loss_chunks=1, z_loss=0.0)
torch.manual_seed(0); mdl = KestrelModel(cfg).eval()
xi = torch.randint(0,64,(2,16)); yi = torch.randint(0,64,(2,16))
mk = torch.zeros(2,16); mk[:, 8:] = 1.0     # score only the second half

with torch.no_grad():
    lg,_ = mdl(xi)
    per = F.cross_entropy(lg.reshape(-1,64), yi.reshape(-1), reduction="none")
    manual = float((per*mk.reshape(-1)).sum()/mk.sum())
    _, l_masked = mdl(xi, targets=yi, loss_mask=mk)
    _, l_plain  = mdl(xi, targets=yi)
check("masked loss matches manual", abs(float(l_masked)-manual)<1e-4,
      f"{float(l_masked):.6f} vs {manual:.6f}")
check("masked differs from unmasked", abs(float(l_masked)-float(l_plain))>1e-4,
      f"{float(l_masked):.4f} vs {float(l_plain):.4f}")

print("=== chunked path agrees with unchunked ===")
h = torch.randn(2,16,32); W = torch.randn(64,32)
a = chunked_cross_entropy(h, W, yi, 0.0, 1, mk)
b = chunked_cross_entropy(h, W, yi, 0.0, 4, mk)
check("1 chunk == 4 chunks", abs(float(a)-float(b))<1e-4, f"{float(a):.6f} vs {float(b):.6f}")
z = chunked_cross_entropy(h, W, yi, 0.0, 4, torch.zeros(2,16))
check("all-zero mask -> 0 not NaN", float(z)==0.0 and not torch.isnan(z), f"{float(z)}")

print()
print("ALL PASS" if not fails else f"FAILURES: {fails}")
raise SystemExit(1 if fails else 0)
