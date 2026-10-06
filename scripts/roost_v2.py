"""Roost V2: verify consolidation with a correct test, then build Kestrel-Nano V2.

    python -m scripts.roost_v2 --ckpt kestrel-nano-v1-weights.pt --replay-dir roost-data --out v2

1. Benchmark (the S3 criterion): 50 novel facts about nonsense entities, consolidated
   into PKM value slots only, under a sweep of settings. Run 0 repeats the exact V1
   settings under the corrected test, which answers "did V1 learn anything?".
   Gate: recall >= 60% (cue ends right before the answer) AND regression <= 2%.
2. If a setting passes, teach V1 a small pack of TRUE facts about itself with that
   setting (V1 answered "Who made you?" wrong) -> Kestrel-Nano V2.
3. Write results.json and the V2 PKM value tables (the only weights that change).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
import time

import torch

from kestrel.roost import EpisodicStore, consolidate
from kestrel.roost.consolidate import _greedy, _pkm_modules
from scripts.roost import load_model, novel_facts

SWEEP = [
    # name, steps, lr, new_frac, top_t, qa, answer_only
    ("v1-settings",       200, 1e-3, 0.50, 2048, False, False),
    ("v1-settings+qa",    200, 1e-3, 0.50, 2048, True,  False),
    ("lr1e-2",            300, 1e-2, 0.50, 2048, True,  False),
    ("lr1e-2-wide",       300, 1e-2, 0.67, 8192, True,  False),
    ("lr3e-2-wide",       300, 3e-2, 0.67, 8192, True,  False),
    ("lr1e-2-wide-long",  600, 1e-2, 0.67, 8192, True,  False),
    # answer-only loss: score only the part of each fact that has to be stored
    ("ans-lr1e-2",        300, 1e-2, 0.75, 8192, True,  True),
    ("ans-lr3e-2",        300, 3e-2, 0.75, 8192, True,  True),
    ("ans-lr1e-1",        300, 1e-1, 0.75, 8192, True,  True),
    ("ans-lr3e-2-long",   600, 3e-2, 0.75, 8192, True,  True),
]

# True statements about the model, phrased "<subject> is <answer>" so the recall
# test can cue up to the answer. Chat variants teach the questions people actually ask.
SELF_FACTS = [
    ("The creator of Kestrel-Nano is MalxTech.",
     ["User: Who made you?\n\nAssistant: I was made by MalxTech.",
      "User: Who created you?\n\nAssistant: MalxTech created me. I am Kestrel-Nano."]),
    ("The company that made Kestrel-Nano is MalxTech.", []),
    ("The name of this AI model is Kestrel-Nano.",
     ["User: What AI model are you?\n\nAssistant: I am Kestrel-Nano, a small language model made by MalxTech."]),
    ("The context length of Kestrel-Nano is 4096 tokens.", []),
    ("The app that runs Kestrel-Nano is Kestrel Studio.", []),
    ("The size of Kestrel-Nano is 157 million parameters.", []),
    ("The memory system of Kestrel-Nano is called Roost.", []),
]
SELF_CHECKS = [("Who made you?", "MalxTech"), ("Who created you?", "MalxTech"),
               ("What AI model are you?", "Kestrel")]


def chat_checks(model, tok, device):
    out = []
    for q, want in SELF_CHECKS:
        ans = _greedy(model, tok, f"User: {q}\n\nAssistant:", device, 24)
        out.append({"q": q, "answer": ans.split("\n\n")[0].strip(),
                    "ok": want.lower() in ans.lower()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/kestrel-bpe.json")
    ap.add_argument("--replay-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--only", default=None, help="comma-separated sweep names to run")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tmp = tempfile.mkdtemp()
    t0 = time.time()

    model, tok = load_model(args.ckpt, args.tokenizer, args.device)
    base = copy.deepcopy({k: v.detach().clone() for k, v in model.state_dict().items()})
    results = {"benchmark": [], "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    def reset():
        model.load_state_dict(base)

    # ------------------------------------------------------------- 1. benchmark
    facts = novel_facts(50)
    sweep = [s for s in SWEEP if not args.only or s[0] in args.only.split(",")]
    for name, steps, lr, frac, top_t, qa, ans in sweep:
        reset()
        store = EpisodicStore(os.path.join(tmp, f"bench-{name}.jsonl"))
        store.extend(facts)
        print(f"\n===== benchmark: {name} =====", flush=True)
        res = consolidate(model, tok, store, replay_dir=args.replay_dir, device=args.device,
                          steps=steps, lr=lr, new_frac=frac, top_t=top_t, qa=qa,
                          answer_only=ans)
        print(res, flush=True)
        r = res.__dict__.copy()
        r["name"] = name
        results["benchmark"].append(r)
        json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=1)

    passed = [r for r in results["benchmark"] if r["accepted"]]
    best = (max(passed, key=lambda r: (r["recall_after"], -r["regression_pct"])) if passed
            else max(results["benchmark"], key=lambda r: (r["recall_after"], -r["nll_after"])))
    results["best"] = best["name"]
    results["benchmark_passed"] = bool(passed)
    print(f"\n===== best: {best['name']} (passed gate: {bool(passed)}) =====", flush=True)

    # ------------------------------------------------------------- 2. build V2
    reset()
    results["v1_chat"] = chat_checks(model, tok, args.device)
    if passed:
        c = best["config"]
        store = EpisodicStore(os.path.join(tmp, "self.jsonl"))
        store.extend([f for f, _ in SELF_FACTS])
        for it, (_, chats) in zip(store.items, SELF_FACTS):
            it.variants.extend(chats)
        print("\n===== V2: consolidating self-knowledge =====", flush=True)
        res = consolidate(model, tok, store, replay_dir=args.replay_dir, device=args.device,
                          steps=c["steps"], lr=c["lr"], new_frac=c["new_frac"],
                          top_t=c["top_t"], qa=c["qa"], answer_only=c.get("answer_only", False))
        print(res, flush=True)
        results["v2_consolidation"] = res.__dict__.copy()
        results["v2_chat"] = chat_checks(model, tok, args.device)
        results["v2_built"] = bool(res.accepted)
        if res.accepted:
            pkm = {f"pkm{i}.values.weight": p.values.weight.detach().cpu().clone()
                   for i, p in enumerate(_pkm_modules(model))}
            names = [n for n, _ in model.named_parameters() if n.endswith("pkm.values.weight")]
            torch.save({"names": names, "tensors": [pkm[f"pkm{i}.values.weight"]
                                                    for i in range(len(names))]},
                       os.path.join(args.out, "kestrel-nano-v2-pkm-values.pt"))
            ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
            ck["model"] = {k: v.cpu() for k, v in model.state_dict().items()}
            ck["version"] = "V2"
            torch.save(ck, os.path.join(args.out, "kestrel-nano-v2-weights.pt"))
    else:
        results["v2_built"] = False
    results["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=1)
    print(json.dumps({k: results[k] for k in ("best", "benchmark_passed", "v2_built", "minutes")}),
          flush=True)


if __name__ == "__main__":
    main()
