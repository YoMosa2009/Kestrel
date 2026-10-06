"""Export a Kestrel checkpoint to a single .gguf file for Kestrel Studio.

GGUF is used purely as the container: one self-describing file holding the
weights, the full model config and the tokenizer. The architecture tag is
`kestrel`, which llama.cpp / LM Studio / Ollama do not implement (gated linear
attention + looped core + product-key memory); the native engine in
app/Kestrel.Engine does.

    python -m scripts.export_gguf --ckpt runs/sft/ckpt.pt --out kestrel-nano.gguf
    python -m scripts.export_gguf --ckpt ... --out ... --dtype q8_0
    python -m scripts.export_gguf --ckpt ... --out ... --fixture parity.json

Tensor names are the PyTorch state-dict names unchanged ("core.3.mixer.wqkv.weight"),
so the engine and this script share one naming scheme with model.py. Shapes are
written GGUF-style (fastest-varying dimension first) - the reverse of numpy order.

--fixture additionally runs the PyTorch model on a few prompts and writes the
reference logits the C# engine is checked against (needs a GPU or patience; run
it on Colab, not on the home PC).
"""

from __future__ import annotations

import argparse
import json
import os
import struct

import numpy as np
import torch

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
ALIGN = 32

# metadata value types
T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64 = 4, 5, 6, 7, 8, 9, 10, 11
# tensor types
F32, F16, Q8_0 = 0, 1, 8

CHAT_TEMPLATE = "User: {prompt}\n\nAssistant:"


# ------------------------------------------------------------------ writer

def _str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, value) -> bytes:
    out = _str(key)
    if isinstance(value, bool):
        return out + struct.pack("<IB", T_BOOL, int(value))
    if isinstance(value, int):
        if -2**31 <= value < 2**31:
            return out + struct.pack("<Ii", T_I32, value)
        return out + struct.pack("<Iq", T_I64, value)       # e.g. tokens_seen > 2^31
    if isinstance(value, float):
        return out + struct.pack("<If", T_F32, value)
    if isinstance(value, str):
        return out + struct.pack("<I", T_STR) + _str(value)
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, str) for v in value):
            return (out + struct.pack("<IIQ", T_ARR, T_STR, len(value))
                    + b"".join(_str(v) for v in value))
        if all(isinstance(v, int) for v in value):
            return (out + struct.pack("<IIQ", T_ARR, T_I32, len(value))
                    + struct.pack(f"<{len(value)}i", *value))
        if all(isinstance(v, (int, float)) for v in value):
            return (out + struct.pack("<IIQ", T_ARR, T_F32, len(value))
                    + struct.pack(f"<{len(value)}f", *value))
    raise TypeError(f"unsupported metadata value for {key}: {type(value)}")


def _q8_0(a: np.ndarray) -> bytes:
    """ggml Q8_0: per 32 values along the last axis, an f16 scale + 32 int8."""
    rows = a.reshape(-1, a.shape[-1]).astype(np.float32)
    assert rows.shape[1] % 32 == 0, "Q8_0 needs the row length to be a multiple of 32"
    blocks = rows.reshape(-1, 32)
    amax = np.abs(blocks).max(axis=1, keepdims=True)
    d = amax / 127.0
    q = np.where(d > 0, np.round(blocks / np.where(d > 0, d, 1)), 0).astype(np.int8)
    rec = np.zeros(len(blocks), dtype=[("d", "<f2"), ("q", "i1", 32)])
    rec["d"] = d[:, 0].astype(np.float16)
    rec["q"] = q
    return rec.tobytes()


def _pick_type(name: str, a: np.ndarray, want: int) -> int:
    # Only 2-D weight matrices are quantized; norms, biases, gates, conv taps,
    # PKM keys and q_scale stay F32 (tiny, and precision-sensitive).
    if want == F32 or a.ndim != 2 or a.shape[-1] % 32 != 0 or name.endswith("keys"):
        return F32
    return want


def write_gguf(path: str, meta: dict, tensors: dict[str, np.ndarray], dtype: int):
    infos, blobs, offset = [], [], 0
    for name, a in tensors.items():
        t = _pick_type(name, a, dtype)
        if t == F32:
            data = np.ascontiguousarray(a, dtype="<f4").tobytes()
        elif t == F16:
            data = np.ascontiguousarray(a, dtype="<f2").tobytes()
        else:
            data = _q8_0(a)
        dims = list(reversed(a.shape))
        info = _str(name) + struct.pack("<I", len(dims))
        info += struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", t, offset)
        infos.append(info)
        pad = (-len(data)) % ALIGN
        blobs.append(data + b"\0" * pad)
        offset += len(data) + pad

    header = GGUF_MAGIC + struct.pack("<IQQ", GGUF_VERSION, len(tensors), len(meta))
    header += b"".join(_kv(k, v) for k, v in meta.items())
    header += b"".join(infos)
    header += b"\0" * ((-len(header)) % ALIGN)

    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(header)
        for b in blobs:
            f.write(b)
    os.replace(tmp, path)


# ------------------------------------------------------------------ kestrel

def collect(ckpt_path: str, tok_path: str, name: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    sd = ck["model"]

    tensors = {}
    for k, v in sd.items():
        a = v.float().numpy()
        if k.endswith("mixer.conv.weight"):        # (d, 1, k) depthwise -> (d, k)
            a = a[:, 0, :]
        tensors[k] = a

    tok = json.load(open(tok_path, encoding="utf-8"))
    vocab = tok["model"]["vocab"]
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    assert all(t is not None for t in tokens), "tokenizer vocab ids are not dense"
    merges = [m if isinstance(m, str) else " ".join(m) for m in tok["model"]["merges"]]
    eot = next(a["id"] for a in tok["added_tokens"] if a["content"] == "<|endoftext|>")
    digits = any(p.get("type") == "Digits" and p.get("individual_digits")
                 for p in tok.get("pre_tokenizer", {}).get("pretokenizers", []))

    meta = {
        "general.architecture": "kestrel",
        "general.name": name,
        "general.author": "MalxTech",
        "general.alignment": ALIGN,
        "kestrel.tokens_seen": int(ck.get("tokens_seen", 0)),
        "kestrel.chat_template": CHAT_TEMPLATE,
    }
    for k in ("vocab_size", "d_model", "n_entry", "n_core", "n_exit", "attn_every",
              "n_heads", "n_kv_heads", "conv_kernel", "d_ff", "r_max", "r_default",
              "pkm_n_keys", "pkm_d_key", "pkm_topk"):
        meta[f"kestrel.{k}"] = int(cfg[k])
    meta["kestrel.rope_theta"] = float(cfg["rope_theta"])
    meta["kestrel.rope_scale"] = float(cfg.get("rope_scale", 1.0))
    meta["kestrel.context_length"] = int(1024 * cfg.get("rope_scale", 1.0))
    meta["kestrel.pkm_sites"] = [int(s) for s in cfg["pkm_sites"]]
    meta["kestrel.tie_embeddings"] = bool(cfg.get("tie_embeddings", True))
    meta["tokenizer.ggml.model"] = "gpt2"
    meta["tokenizer.ggml.pre"] = "kestrel-digits" if digits else "gpt2"
    meta["tokenizer.ggml.tokens"] = tokens
    meta["tokenizer.ggml.merges"] = merges
    meta["tokenizer.ggml.eos_token_id"] = int(eot)
    meta["tokenizer.ggml.bos_token_id"] = int(eot)
    return meta, tensors, cfg


@torch.no_grad()
def fixture(ckpt_path: str, tok_path: str, out: str, device: str):
    """Reference outputs for engine parity: token ids, last-position logits
    (top-32 + logsumexp) at R=1..3, and the Roost embedding of each prompt."""
    from kestrel.generate import load
    from kestrel.roost.store import KestrelEmbedder
    model, tok, cfg, _ = load(ckpt_path, tok_path, torch.device(device))
    model.float()
    prompts = ["The kestrel is a small falcon that",
               "User: Write a Python function that reverses a string.\n\nAssistant:",
               "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
               "In 1969, 3 astronauts flew 384400 km to the Moon."]
    emb = KestrelEmbedder(model, tok, torch.device(device))
    recs = []
    for p in prompts:
        ids = tok.encode(p).ids
        x = torch.tensor([ids], device=device)
        per_r = {}
        for r in range(1, cfg.r_max + 1):
            logits, _ = model(x, n_loops=r)
            last = logits[0, -1].float()
            v, i = last.topk(32)
            per_r[str(r)] = {"top_ids": i.tolist(), "top_logits": v.tolist(),
                             "lse": float(last.logsumexp(-1)),
                             "argmax_per_pos": logits[0].argmax(-1).tolist()}
        recs.append({"prompt": p, "ids": ids, "by_r": per_r,
                     "embedding": emb([p])[0].cpu().tolist()})
    json.dump({"ckpt": ckpt_path, "records": recs}, open(out, "w"), indent=1)
    print(f"fixture -> {out} ({len(recs)} prompts)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/kestrel-bpe.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=["f32", "f16", "q8_0"], default="f16")
    ap.add_argument("--name", default="Kestrel-Nano")
    ap.add_argument("--fixture", default=None, help="also write PyTorch reference logits here")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    meta, tensors, cfg = collect(args.ckpt, args.tokenizer, args.name)
    dtype = {"f32": F32, "f16": F16, "q8_0": Q8_0}[args.dtype]
    write_gguf(args.out, meta, tensors, dtype)
    n = sum(a.size for a in tensors.values())
    print(f"{args.out}: {len(tensors)} tensors, {n/1e6:.1f}M params, {args.dtype}, "
          f"{os.path.getsize(args.out)/1e6:.0f} MB, ctx {meta['kestrel.context_length']}")
    if args.fixture:
        fixture(args.ckpt, args.tokenizer, args.fixture, args.device)


if __name__ == "__main__":
    main()
