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

---

## 4. Phase D — the over-trained run (RTX 3060, local, $0)

### Hardware (measured 2026-09-12)

| | |
|---|---|
| GPU | RTX 3060 12 GB, GA106, **sm_86**, 28 SMs / 3584 CUDA cores / 112 3rd-gen tensor cores |
| Clocks | 2160 MHz max SM, 7501 MHz memory (15 Gbps effective) |
| Bandwidth | 192-bit GDDR6 → **360 GB/s** |
| Compute | ~15.5 TFLOP/s fp32 · ~31 TFLOP/s bf16-tensor (fp32 accumulate) |
| Power | 170 W cap of a 212 W maximum — but **measured 142-154 W under load with zero throttling**, so the headroom is not reclaimable |
| Driver | 610.47, CUDA UMD 13.3, torch 2.7.1+cu126 |
| L2 | 2.25 MB |

### Before you launch: free the GPU

`nvidia-smi` reports ~9.9 GB used at idle because **Ollama pins `gemma4:12b` (8.43 GB,
`keep_alive` set to never expire)**. Phase D needs ~6.3 GB and will OOM instantly.
`torch.cuda.mem_get_info()` lies here — under WDDM it reported 11.8 GB "free" while a
40 MB allocation failed. Trust `nvidia-smi`, not torch.

```bash
ollama stop gemma4:12b
```

Also close anything else holding VRAM (the desktop runs on this GPU, so a few hundred MB
always moves around).

### Data lives on a 5400-rpm HDD

`data_5b/` is 9.1 GB of randomly-sampled shards on **S: (ST1000LM024, HDD)**, and the
machine has 16 GB RAM with ~5.8 GB free — the working set does not fit page cache. Measured
cost: **~2.4 s per optimizer step (~5%)**, fully serialized in front of the GPU. `--prefetch 3`
moves it onto a background thread. C: has only 2.6 GB free, so moving the corpus to the SSD
is not currently an option.

### Launch

`--steps` is **CUMULATIVE** — it includes the 10,000 steps already in `ckpt.pt`. At
24 × 1024 × 8 = 196,608 tok/step, 3B new tokens is 15,259 new steps → 25,259.

```bash
cp ckpt.pt experiments/nano_d/ckpt.pt
```

```bash
.venv/Scripts/python.exe -u -m kestrel.train --preset nano --data-dir data_5b --curriculum --curriculum-start 0.40 --bf16 --grad-checkpoint --fused-adam --prefetch 3 --seq 1024 --batch 24 --accum 8 --steps 25259 --warmup 300 --decay-frac 0.15 --out experiments/nano_d --log-every 20 --eval-every 500 --eval-iters 40 --ckpt-every 1000 --ckpt-secs 1800
```

- `--curriculum-start 0.40` opens in the **technical** stage. v0.1 already spent 1.31B tokens
  on web/know/code, and its measured blind spots (cli 2.80, repo 3.03, sec 2.95, math 2.78,
  tool 2.27 — docs/10 §9) are what the technical stage feeds. Pass `0.0` for the full
  broad → technical → anneal shape instead; the anneal always lands in the last 15%.
- The LR **re-warms over `--warmup` steps from the resume point**. This is not cosmetic:
  v0.1 ended its WSD decay at ~0 LR, and the schedule used to be keyed off the absolute step,
  so a resume skipped warmup entirely and hit a decayed model with the full plateau LR.
- Resume after any interruption by re-running the identical command.

### Verify before committing 8 days

```bash
.venv/Scripts/python.exe -m scripts.bench_step --batch 24 --accum 4
```

```bash
.venv/Scripts/python.exe -m scripts.profile_step --batch 24
```

`bench_step` measures a **full** step (fwd+bwd × accum **plus** the optimizer) and A/Bs the
knobs; `bench_throughput.py` excludes the optimizer, so every tok/s figure in docs/03 and
docs/08 is a fwd+bwd-only number. `profile_step` re-derives the op breakdown and MFU on
*this* card in bf16 — the docs/08 §3 profile ("matmul-bound at ~30% MFU") was taken on a
GTX 1080 in FP32, and bf16 tensor cores move that mix.

**Expected numbers on a free GPU (measured 2026-09-12):** 4,163 tok/s at batch 24 / accum 4
synthetic, ~4,400 tok/s on real shards with `--prefetch 3`, peak **7.71 GB**. If you see
materially less, something else is holding VRAM or the shards are cold. The full record of
what was tried and rejected is in [docs/08 "Phase D tuning"](docs/08-real-model-roadmap.md) —
TF32, fused AdamW, chunk-size changes, and dropping grad-checkpointing were all measured and
none of them help, so do not spend time re-deriving them.

Then a ~30-minute shakedown with the real flags before the full run:

```bash
.venv/Scripts/python.exe -u -m kestrel.train --preset nano --data-dir data_5b --curriculum --curriculum-start 0.40 --bf16 --grad-checkpoint --fused-adam --prefetch 3 --seq 1024 --batch 24 --accum 8 --steps 10050 --warmup 300 --out experiments/nano_d --log-every 5 --eval-every 25 --ckpt-every 50
```

Expect the first probe near v0.1's **val_mean 2.695**, not 8.x. A high loss means the
checkpoint did not load.

### Grade against the control

```bash
.venv/Scripts/python.exe -m scripts.eval_generalization --ckpt experiments/nano_d/ckpt.pt --label "nano-v0.2 (4.3B tok)"
```
