"""Kestrel-Nano text generation (the first way to actually run the model).

Loads a training checkpoint + the tokenizer and generates autoregressively.
v0.1 has no cross-segment KV cache, so each step re-runs the (cropped) context
— fine for a 156M model and short samples.

    python -m kestrel.generate --prompt "The kestrel is a small falcon that"
    python -m kestrel.generate --interactive
    python -m kestrel.generate --prompt "def fib(n):" --r 3 --temp 0.7
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from kestrel.config import KestrelConfig
from kestrel.model import KestrelModel


def load(ckpt_path, tok_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = KestrelConfig(**ck["cfg"])
    cfg.grad_checkpoint = False
    model = KestrelModel(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    tok = Tokenizer.from_file(tok_path)
    return model, tok, cfg, ck.get("tokens_seen", 0)


@torch.no_grad()
def generate(model, tok, prompt, max_new=120, temperature=0.8, top_k=40,
             top_p=0.95, n_loops=2, ctx=1024, device="cpu"):
    eot = tok.token_to_id("<|endoftext|>")
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new):
        logits, _ = model(x[:, -ctx:], n_loops=n_loops)
        logits = logits[:, -1, :].float() / max(temperature, 1e-6)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        if top_p and top_p < 1.0:
            sp, si = torch.sort(probs, descending=True)
            cum = sp.cumsum(-1)
            sp[cum - sp > top_p] = 0.0
            sp /= sp.sum(-1, keepdim=True)
            nxt = si.gather(-1, torch.multinomial(sp, 1))
        else:
            nxt = torch.multinomial(probs, 1)
        if nxt.item() == eot:
            break
        x = torch.cat([x, nxt], dim=1)
    return tok.decode(x[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--tokenizer", default="tokenizer/kestrel-bpe.json")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--r", type=int, default=2, help="loop count R (1-4)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, tok, cfg, seen = load(args.ckpt, args.tokenizer, device)
    print(f"Kestrel-Nano | {sum(p.numel() for p in model.parameters())/1e6:.0f}M params "
          f"| trained on {seen/1e9:.2f}B tokens | R={args.r} | device {device}\n" + "-"*64)

    def run(p):
        out = generate(model, tok, p, args.max_new, args.temp, args.top_k,
                       args.top_p, args.r, cfg.chunk_size * 16, device)
        print(out + "\n" + "-"*64)

    if args.interactive:
        print("Interactive base-model completion. Type a prompt (Ctrl+C to quit).")
        try:
            while True:
                p = input("\nprompt> ")
                if p.strip():
                    run(p)
        except (KeyboardInterrupt, EOFError):
            print("\nbye.")
    else:
        prompts = [args.prompt] if args.prompt else [
            "The kestrel is a small falcon that",
            "def fibonacci(n):",
            "The three most important things in life are",
        ]
        for p in prompts:
            print(f"\n[prompt] {p!r}")
            run(p)


if __name__ == "__main__":
    main()
