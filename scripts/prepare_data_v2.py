"""Phase C data builder (docs/10) — 5B unique tokens composed for generalization
AND for the stated target profile: coding intelligence, tool use, codebase
understanding, CLI/terminal fluency (bash + PowerShell + cmd), and secure coding.

Differences from v1 (`prepare_data.py`, kept for v0.1 reproducibility):
  * StarCoderData replaces raw codeparrot GitHub — quality-filtered, 10 languages
  * NEW domains: math (FineMath), instruct (SmolTalk), tool-use (Glaive/Hermes),
    cli (shell/powershell/batchfile/dockerfile), repo (git-commits + issues),
    docs (markdown), sec (secure-coding pairs)
  * DEDUPLICATION: normalized full-text + prefix hashing (v1 had none)
  * PER-DOMAIN SHARDS (`train_<tag>_NNN.bin`) so Phase D can stage the mixture
    and run a final quality anneal
  * SEQUENTIAL source processing (not 24 concurrent streams) — far more robust
    on a home connection, and `--resume` skips sources already finished

    python -m scripts.prepare_data_v2 --tokens 5000000000 --out data_5b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

SC = "bigcode/starcoderdata"


# ---------------------------------------------------------------- renderers

def _chat(turns):
    role_map = {"system": "System", "user": "User", "human": "User",
                "assistant": "Assistant", "gpt": "Assistant", "tool": "Tool"}
    out = []
    for role, content in turns:
        if content:
            out.append(f"{role_map.get(str(role).lower(), 'User')}: {str(content).strip()}")
    return "\n\n".join(out)


def r_text(row):    return row.get("text") or ""
def r_content(row): return row.get("content") or ""

def r_smoltalk(row):
    return _chat([(m.get("role"), m.get("content")) for m in (row.get("messages") or [])])

def r_glaive(row):
    txt = ((row.get("system") or "") + "\n\n" + (row.get("chat") or "")).strip()
    for a, b in (("SYSTEM:", "System:"), ("USER:", "User:"),
                 ("ASSISTANT:", "Assistant:"), ("FUNCTION RESPONSE:", "Tool:")):
        txt = txt.replace(a, b)
    return txt

def r_hermes(row):
    conv = row.get("conversations")
    if isinstance(conv, str):
        try:
            conv = json.loads(conv.replace("'", '"'))
        except Exception:
            return ""
    if not isinstance(conv, list):
        return ""
    return _chat([(m.get("from"), m.get("value")) for m in conv if isinstance(m, dict)])

def r_secure(row):
    """Secure-coding pairs: keep the SECURE answer (`chosen`), never the vulnerable one."""
    q, chosen = row.get("question") or "", row.get("chosen") or ""
    vuln, lang = row.get("vulnerability") or "", row.get("lang") or ""
    if not (q and chosen):
        return ""
    head = f"System: Write secure {lang} code. Avoid: {vuln}".strip()
    return f"{head}\n\nUser: {q.strip()}\n\nAssistant: {chosen.strip()}"


def sc_src(tag, lang, share):
    """A StarCoderData language subset (loaded via data_dir)."""
    return dict(tag=tag, name=SC, cfg=None, data_dir=lang, render=r_content,
                share=share, key=f"{tag}:{lang}")


# ---- the mixture (shares sum to ~1.0) --------------------------------------
# language & reasoning 48% | code 30% | cli 6% | repo 5% | docs 2%
# instruct 5% | tool 3% | security 1.5%
SOURCES = [
    dict(tag="web",  name="HuggingFaceFW/fineweb-edu",   cfg="sample-10BT",   data_dir=None, render=r_text,     share=0.240, key="web"),
    dict(tag="know", name="HuggingFaceTB/smollm-corpus", cfg="cosmopedia-v2", data_dir=None, render=r_text,     share=0.140, key="know"),
    dict(tag="math", name="HuggingFaceTB/finemath",      cfg="finemath-4plus",data_dir=None, render=r_text,     share=0.100, key="math"),
    # code — full-stack languages
    sc_src("code", "python",     0.070),
    sc_src("code", "javascript", 0.050),
    sc_src("code", "typescript", 0.040),
    sc_src("code", "java",       0.030),
    sc_src("code", "cpp",        0.020),
    sc_src("code", "go",         0.020),
    sc_src("code", "rust",       0.020),
    sc_src("code", "sql",        0.020),
    sc_src("code", "html",       0.015),
    sc_src("code", "css",        0.015),
    # cli / terminal — bash, PowerShell, cmd, containers
    sc_src("cli",  "shell",      0.030),
    sc_src("cli",  "powershell", 0.020),
    sc_src("cli",  "batchfile",  0.005),
    sc_src("cli",  "dockerfile", 0.005),
    # codebase understanding — edits (<commit_before>/<commit_after>) and issue threads
    sc_src("repo", "git-commits-cleaned",               0.030),
    sc_src("repo", "github-issues-filtered-structured", 0.020),
    # documentation
    sc_src("docs", "markdown",   0.020),
    # instruct / tool-use / security
    dict(tag="inst", name="HuggingFaceTB/smoltalk", cfg="all", data_dir=None, render=r_smoltalk, share=0.050, key="inst"),
    dict(tag="tool", name="glaiveai/glaive-function-calling-v2", cfg=None, data_dir=None, render=r_glaive, share=0.020, key="tool:glaive"),
    dict(tag="tool", name="NousResearch/hermes-function-calling-v1", cfg="func_calling_singleturn", data_dir=None, render=r_hermes, share=0.010, key="tool:hermes"),
    dict(tag="sec",  name="CyberNative/Code_Vulnerability_Security_DPO", cfg=None, data_dir=None, render=r_secure, share=0.015, key="sec"),
]

_WS = re.compile(r"\s+")


def norm_hash(text: str) -> int:
    n = _WS.sub(" ", text.lower()).strip()
    return int.from_bytes(hashlib.blake2b(n.encode("utf-8", "ignore"),
                                          digest_size=8).digest(), "big")


class DomainWriter:
    """Appends uint16 tokens into per-domain shards train_<tag>_NNN.bin."""

    def __init__(self, out_dir, tag, shard_tokens, start_idx=0):
        self.out_dir, self.tag, self.shard_tokens = out_dir, tag, shard_tokens
        self.idx, self.in_shard, self.total = start_idx, 0, 0
        self.f = None

    def write(self, ids: np.ndarray):
        pos = 0
        while pos < len(ids):
            if self.f is None:
                self.f = open(os.path.join(
                    self.out_dir, f"train_{self.tag}_{self.idx:03d}.bin"), "ab")
                self.in_shard = self.f.tell() // 2
            take = min(max(1, self.shard_tokens - self.in_shard), len(ids) - pos)
            ids[pos:pos + take].tofile(self.f)
            self.in_shard += take; self.total += take; pos += take
            if self.in_shard >= self.shard_tokens:
                self.f.close(); self.f = None; self.idx += 1; self.in_shard = 0

    def close(self):
        if self.f is not None:
            self.f.close(); self.f = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer/kestrel-bpe.json")
    ap.add_argument("--tokens", type=int, default=5_000_000_000)
    ap.add_argument("--val-tokens", type=int, default=400_000, help="held out per source")
    ap.add_argument("--shard-tokens", type=int, default=200_000_000)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--dedup-cap", type=int, default=14_000_000)
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip sources already done")
    ap.add_argument("--out", default="data_5b")
    ap.add_argument("--only", nargs="*", default=None, help="only these source keys")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok = Tokenizer.from_file(args.tokenizer)
    eot = tok.token_to_id("<|endoftext|>")
    prog_path = os.path.join(args.out, "progress.json")
    prog = {}
    if args.resume and os.path.exists(prog_path):
        prog = json.load(open(prog_path))
        print(f"resuming — {len(prog)} sources already complete", flush=True)

    print(f"target {args.tokens/1e9:.2f}B tokens | {len(SOURCES)} sources | "
          f"dedup={'off' if args.no_dedup else 'on'}", flush=True)

    seen_full, seen_pref = set(), set()
    stats = {"dup": 0, "short": 0, "kept": 0}
    t_all = time.time()

    for src in SOURCES:
        key = src["key"]
        if args.only and key not in args.only:
            continue
        if key in prog:
            print(f"[skip] {key} (already {prog[key]/1e6:.0f}M)", flush=True)
            continue
        budget = int(args.tokens * src["share"])
        # continue this domain's shard numbering across sources sharing the tag
        existing = [f for f in os.listdir(args.out)
                    if f.startswith(f"train_{src['tag']}_") and f.endswith(".bin")]
        w = DomainWriter(args.out, src["tag"], args.shard_tokens,
                         start_idx=len(existing[:-1]) if existing else 0)
        print(f"[{src['tag']:4s}] {key:34s} -> {budget/1e6:6.0f}M tokens", flush=True)

        def stream(seed=0):
            kw = dict(split="train", streaming=True)
            if src["data_dir"]:
                return iter(load_dataset(src["name"], data_dir=src["data_dir"], **kw))
            return iter(load_dataset(src["name"], src["cfg"], **kw))

        it = stream(); val = []; got = 0; reconn = 0; t0 = time.time()
        while got < budget:
            texts = []
            stop = False
            for _ in range(args.batch):
                try:
                    row = next(it)
                except StopIteration:
                    stop = True; break
                except Exception as e:
                    reconn += 1
                    print(f"  [warn] {key} stream error #{reconn}: {str(e)[:55]} -> reconnect", flush=True)
                    if reconn > 60:
                        stop = True; break
                    time.sleep(2); it = stream(reconn); break
                try:
                    txt = src["render"](row)
                except Exception:
                    continue
                if not txt or len(txt) < args.min_chars:
                    stats["short"] += 1; continue
                if not args.no_dedup:
                    h = norm_hash(txt)
                    if h in seen_full:
                        stats["dup"] += 1; continue
                    hp = norm_hash(txt[:512])
                    if hp in seen_pref:
                        stats["dup"] += 1; continue
                    if len(seen_full) < args.dedup_cap:
                        seen_full.add(h); seen_pref.add(hp)
                stats["kept"] += 1
                texts.append(txt)
            if texts:
                flat = []
                for e in tok.encode_batch(texts):
                    ids = e.ids; ids.append(eot)
                    if len(val) < args.val_tokens:
                        val.extend(ids)
                    else:
                        flat.extend(ids)
                    got += len(ids)
                if flat:
                    w.write(np.asarray(flat, dtype=np.uint16))
            if stop:
                print(f"  [note] {key} exhausted at {got/1e6:.0f}M / {budget/1e6:.0f}M", flush=True)
                break
        w.close()
        vp = os.path.join(args.out, f"val_{src['tag']}.bin")
        with open(vp, "ab") as f:
            np.asarray(val[:args.val_tokens], dtype=np.uint16).tofile(f)
        prog[key] = got
        json.dump(prog, open(prog_path, "w"), indent=1)
        total = sum(prog.values())
        print(f"  [{key}] {got/1e6:.0f}M in {time.time()-t0:.0f}s | running total "
              f"{total/1e9:.2f}B | dedup-dropped {stats['dup']/1e6:.2f}M docs", flush=True)

    total = sum(prog.values())
    manifest = {"tokens_total": total, "per_source": prog, "dedup": stats,
                "mixture": {s["key"]: s["share"] for s in SOURCES}}
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"DONE: {total/1e9:.3f}B tokens -> {args.out}/ in {(time.time()-t_all)/60:.0f} min | "
          f"dedup dropped {stats['dup']/1e6:.2f}M docs, {stats['short']/1e6:.2f}M too short",
          flush=True)


if __name__ == "__main__":
    main()
