# Kestrel-Nano — Local Training Runbook

Everything needed to train Kestrel-Nano (~157M) end-to-end on the local GPU. NOTE: the lab
GPU is now an **RTX 3060 12 GB (Ampere)** — use `--bf16 --grad-checkpoint --batch 24`
(~4,227 tok/s). The GTX 1080 numbers below are historical. Built and
verified 2026-07-20 (P0–P2 done; hero run P6a is the last step). All commands use the
project venv: `S:\AI Architecture\.venv` (Python 3.12, torch 2.7.1+cu126, sm_61 verified).

Run from the project root. On Windows the venv python is `.venv\Scripts\python.exe`.

## 0. Environment (done — P0)

```
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.get_arch_list())"   # must contain sm_61
.venv\Scripts\python.exe -m kestrel.smoke_test                                   # all CPU correctness tests
.venv\Scripts\python.exe -m scripts.bench_throughput --presets nano --batch 8 --grad-checkpoint
```
Measured: Nano ~1,850–1,990 tok/s, peak VRAM ~3.0 GB (batch 8, seq 1024, R=2). Fits 8 GB.

## 1. Tokenizer (done — P1)

`tokenizer/kestrel-bpe.json` — 49,152-vocab byte-level BPE, digit-split, uint16-safe.
Retrain / make the SuperBPE variant (Pillar 4 / ablation A4):
```
.venv\Scripts\python.exe -m scripts.train_tokenizer --sample-mb 300 --out tokenizer/kestrel-bpe.json
.venv\Scripts\python.exe -m scripts.train_tokenizer --sample-mb 300 --superword --out tokenizer/kestrel-super.json
```

## 2. Data (P1 — building)

Streams FineWeb-Edu (web) + Cosmopedia-v2 (knowledge) + codeparrot-clean (code),
interleaved, into uint16 shards under `data/` with per-domain val shards. Memory-bounded.
```
.venv\Scripts\python.exe -u -m scripts.prepare_data --tokens 200000000 --out data
```
To grow the corpus later, raise `--tokens` (or add sources in `scripts/prepare_data.py`)
and re-run into a fresh dir; point the hero run at it. Data enrichment TODO: Stack-Edu
multi-language + tool-use/function-call traces + FineMath (need a blob-fetcher or HF auth).

## 3. Hero run (P6a) — LAUNCHED 2026-07-20

Currently running on the standard BPE + the 129M-token `data/` shards (see the tokenizer
note below). WSD makes it pausable/extendable; checkpoints every ~30 min hold
model+optimizer+dataloader+RNG, so an interrupted run resumes losslessly by re-running the
SAME command. The exact launched command:
```
.venv\Scripts\python.exe -u -m kestrel.train --preset nano --data-dir data ^
    --seq 1024 --batch 8 --accum 16 --steps 3000 --warmup 300 --decay-frac 0.15 ^
    --grad-checkpoint --out experiments/nano ^
    --log-every 10 --eval-every 100 --eval-iters 16 --ckpt-every 500 --ckpt-secs 1800
```
- 8 × 1024 × 16 = 131k tokens/step; 3000 steps ≈ 393M tokens seen (~3 epochs of 129M),
  ~2–2.5 days at ~1,500–1,900 tok/s. To go bigger, build more data (§2) and raise `--steps`.
- Watch `experiments/nano/metrics.jsonl`: `loss` should fall; `probe.loop_gain` R1>R2>R3
  means the looped core is working; `probe.pkm_gate_mean` should stay non-zero. No val shards
  in this run, so probes use fresh train batches (`loss_is_val:false`).
- Resume after any stop: re-run the identical command (auto-detects `experiments/nano/ckpt.pt`).

**Tokenizer note:** the run uses the standard byte-level BPE (`tokenizer/kestrel-bpe.json`).
SuperBPE (Pillar 4) was attempted: single-phase superword only reaches ~16% token reduction
(below the 18% gate) and is memory-limited to ~50 MB training on 16 GB; the faithful two-phase
version needs a custom trainer. Adopting it later means re-tokenizing data + restarting the run
(WSD makes the restart cheap). Superword tokenizers kept at `tokenizer/kestrel-super*.json`.

## Files

```
kestrel/config.py    presets + KestrelConfig (grad_checkpoint, loss_chunks added)
kestrel/model.py     model + chunked_cross_entropy + gradient-checkpointed blocks
kestrel/optim.py     Muon (matrices) + AdamW grouping (embeddings/norms/gates/PKM)
kestrel/data.py      memmap sharded token loader (resumable RNG cursor)
kestrel/train.py     the trainer (WSD, grad-accum, checkpoint-everything, probes)
scripts/bench_throughput.py  P0 GPU throughput + VRAM
scripts/train_tokenizer.py   P1 tokenizer (BPE / --superword)
scripts/prepare_data.py      P1 streaming tokenize -> interleaved uint16 shards
```
