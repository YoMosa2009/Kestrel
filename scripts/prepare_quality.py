"""V3 step 4 data (docs/11): quality tokens the model has never seen.

Same sources as the knowledge part of data_5b, but from files that corpus never reached, so
this is new text rather than repeats. data_5b read each source in file order and stopped early
(web 1.2B of ~10B, know 0.7B of ~50B, math 0.5B), so the second half of each file list is unseen.
FineWeb keeps only documents scoring 4 or more.

Writes train_q<tag>_NNN.bin + val_q<tag>.bin. The q prefix keeps them apart from data_5b's
domains, so the trainer's --mix can weight old and new separately, and eval_generalization run
on data_5b alone stays comparable with every earlier number.

    python -m scripts.prepare_quality --out /content/q --sync-dir /content/drive/MyDrive/Kestrel/datasets/quality_v3
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

CORPUS = "HuggingFaceTB/smollm-corpus"
SWH = "https://softwareheritage.s3.amazonaws.com/content/"

SOURCES = [
    dict(tag="qknow", repo=CORPUS, prefix="cosmopedia-v2/", tokens=600_000_000, columns=["text"]),
    dict(tag="qweb", repo="HuggingFaceFW/fineweb-edu", prefix="sample/10BT/", tokens=420_000_000,
         columns=["text", "int_score"], min_score=4),
    dict(tag="qmath", repo="HuggingFaceTB/finemath", prefix="finemath-4plus/", tokens=220_000_000, columns=["text"]),
    # Python-Edu's rows only name a file; the text comes from Software Heritage's public bucket.
    dict(tag="qcode", repo=CORPUS, prefix="python-edu/", tokens=130_000_000,
         columns=["blob_id", "length_bytes"], blobs=True),
]


def clean(text):
    """Some FineMath text carries lone surrogates, which the tokenizer can't take."""
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")


def unseen_files(names, prefix):
    """The second half of a source's sorted parquet files, the part data_5b never reached."""
    files = sorted(n for n in names if n.startswith(prefix) and n.endswith(".parquet"))
    return files[len(files) // 2:]


def rows_of(repo, path, columns, retries=30):
    """Stream one parquet file, picking up where it stopped after a network error."""
    from datasets import load_dataset
    done = 0
    for attempt in range(retries + 1):
        try:
            ds = load_dataset("parquet", data_files=f"hf://datasets/{repo}/{path}", split="train",
                              streaming=True, columns=columns)
            for row in ds.skip(done):
                done += 1
                yield row
            return
        except Exception as e:
            print(f"  [warn] {path} row {done}: {str(e)[:70]} -> retry {attempt + 1}", flush=True)
            time.sleep(min(60, 2 * (attempt + 1)))
    print(f"  [warn] gave up on {path} at row {done}", flush=True)


def fetch_blob(session, blob_id):
    try:
        r = session.get(SWH + blob_id, timeout=30)
        return clean(gzip.decompress(r.content).decode("utf-8", "ignore")) if r.status_code == 200 else ""
    except Exception:
        return ""


def texts_of(src, files, batch, min_chars, pool, session):
    """Batches of usable documents from a source's files."""
    pending = []

    def flush():
        ids = [p for p in pending]
        pending.clear()
        got = list(pool.map(lambda b: fetch_blob(session, b), ids)) if src.get("blobs") else ids
        return [t for t in got if len(t) >= min_chars]

    for path in files:
        for row in rows_of(src["repo"], path, src["columns"]):
            if src.get("blobs"):
                if 200 <= (row.get("length_bytes") or 0) <= 200_000:
                    pending.append(row["blob_id"])
            elif (row.get("int_score") or 0) >= src.get("min_score", 0):
                pending.append(clean(row.get("text") or ""))
            if len(pending) >= batch:
                out = flush()
                if out:
                    yield out
    out = flush()
    if out:
        yield out


def build_source(src, names, tok, eot, out_dir, args, pool, session):
    from scripts.prepare_data_v2 import DomainWriter, norm_hash
    tag = src["tag"]
    for f in os.listdir(out_dir):                  # a source that stopped halfway restarts clean
        if f.startswith((f"train_{tag}_", f"val_{tag}.")):
            os.remove(os.path.join(out_dir, f))
    files = unseen_files(names, src["prefix"])
    if not files:
        raise SystemExit(f"{tag}: no parquet files under {src['prefix']}")
    budget = int(src["tokens"] * args.scale)
    val_budget = int(args.val_tokens * args.scale)
    print(f"[{tag}] {len(files)} files from {files[0]} -> {budget / 1e6:.0f}M tokens", flush=True)
    w = DomainWriter(out_dir, tag, args.shard_tokens)
    seen, val, got, docs, dups, t0 = set(), [], 0, 0, 0, time.time()
    for texts in texts_of(src, files, args.batch, args.min_chars, pool, session):
        keep = []
        for t in texts:
            h = norm_hash(t)
            if h in seen:
                dups += 1
            else:
                seen.add(h)
                keep.append(t)
        flat = []
        for e in tok.encode_batch(keep):
            ids = e.ids
            ids.append(eot)
            (val if len(val) < val_budget else flat).extend(ids)
            got += len(ids)
            docs += 1
        if flat:
            w.write(np.asarray(flat, dtype=np.uint16))
        if time.time() - t0 > 600 and docs and got // 50_000_000 != (got - len(flat)) // 50_000_000:
            print(f"  {tag} {got / 1e6:.0f}M | {got / (time.time() - t0):,.0f} tok/s", flush=True)
        if got >= budget:
            break
    w.close()
    np.asarray(val[:val_budget], dtype=np.uint16).tofile(os.path.join(out_dir, f"val_{tag}.bin"))
    print(f"  [{tag}] {got / 1e6:.0f}M tokens, {docs:,} docs, {dups:,} duplicates dropped, "
          f"{(time.time() - t0) / 60:.0f} min", flush=True)
    return dict(tokens=got, docs=docs, dups=dups, files=[files[0], files[-1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer/kestrel-bpe.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sync-dir", default=None, help="copy each finished source here, and resume from it")
    ap.add_argument("--scale", type=float, default=1.0, help="shrink every budget (0.001 = a smoke test)")
    ap.add_argument("--val-tokens", type=int, default=400_000)
    ap.add_argument("--shard-tokens", type=int, default=200_000_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--blob-threads", type=int, default=48)
    args = ap.parse_args()

    import requests
    from huggingface_hub import list_repo_files
    from tokenizers import Tokenizer

    os.makedirs(args.out, exist_ok=True)
    home = args.sync_dir or args.out
    os.makedirs(home, exist_ok=True)
    prog_path = os.path.join(home, "progress.json")
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {}
    tok = Tokenizer.from_file(args.tokenizer)
    eot = tok.token_to_id("<|endoftext|>")
    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=args.blob_threads))
    names = {}
    with ThreadPoolExecutor(args.blob_threads) as pool:
        for src in SOURCES:
            tag = src["tag"]
            if tag in prog:
                print(f"[skip] {tag} already has {prog[tag]['tokens'] / 1e6:.0f}M", flush=True)
                continue
            names.setdefault(src["repo"], list_repo_files(src["repo"], repo_type="dataset"))
            prog[tag] = build_source(src, names[src["repo"]], tok, eot, args.out, args, pool, session)
            if args.sync_dir:
                for f in sorted(os.listdir(args.out)):
                    if f.startswith((f"train_{tag}_", f"val_{tag}.")):
                        shutil.copy(os.path.join(args.out, f), os.path.join(args.sync_dir, f))
            json.dump(prog, open(prog_path, "w"), indent=1)   # last, so it only names copied files
    total = sum(p["tokens"] for p in prog.values())
    json.dump(dict(tokens_total=total, sources=prog), open(os.path.join(home, "manifest.json"), "w"), indent=1)
    print(f"QUALITY_DONE {total / 1e9:.3f}B tokens", flush=True)


if __name__ == "__main__":
    main()
