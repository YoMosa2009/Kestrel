# Kestrel-Nano — RunPod Training Runbook

Train Nano (~156M) on an RTX 4090 for ~$29 (~8–10B tokens, bf16). The local code is
already cloud-ready: `--bf16` (autocast + FlashAttention), `--compile`, `--loss-chunks`
override, and gradient checkpointing OFF by default (only needed on the 8 GB card).

## 1. Deploy the pod
RunPod → **Community Cloud** → filter **RTX 4090** (~$0.34–0.44/hr) → template **"RunPod PyTorch 2.x"**
(CUDA + PyTorch preinstalled) → **Container disk ~20 GB, Volume disk ~40 GB** (holds data + checkpoints)
→ Deploy. Connect via the **web terminal** (or SSH).

## 2. Get the code on the pod
The repo is **private**, so pick one:
- **Simplest:** temporarily set github.com/YoMosa2009/Kestrel to Public → clone → set back to Private.
- **Or** a GitHub read PAT: `git clone https://<PAT>@github.com/YoMosa2009/Kestrel.git`
```
cd /workspace
git clone https://github.com/YoMosa2009/Kestrel.git
cd Kestrel
pip install -U tokenizers datasets numpy   # torch is preinstalled on the template
```

## 3. Get the data (rebuild on the pod — its network is fast + stable)
Avoids uploading 1.9 GB from the local flaky link. Needs HF auth on the pod:
```
hf auth login                       # paste a Read token at the prompt
python -u -m scripts.prepare_data --tokenizer tokenizer/kestrel-bpe.json \
    --tokens 1000000000 --shard-tokens 200000000 --val-tokens 500000 \
    --batch 200 --shuffle-buffer 0 --out data_cloud
```
(~5–15 min on a datacenter link.) *Alternative:* upload the local `data_cloud/` via `runpodctl send`.

## 4. ~$1 sanity run (do this BEFORE the real run)
Confirms bf16 works on the 4090 and measures real tok/s → tells us exact steps for the budget:
```
python -u -m scripts.bench_throughput --presets nano --batch 24 --seq 1024 --r 2   # tok/s + VRAM
python -u -m kestrel.train --preset nano --synthetic --bf16 --loss-chunks 1 \
    --steps 40 --batch 24 --seq 1024 --warmup 5 --out experiments/sanity
```
Note the tok/s. If VRAM allows, push `--batch` up (48/64). Optionally test `--compile`.

## 5. The real run
Pick `--steps` from the measured tok/s and your budget (tokens = steps × batch × accum × seq;
$ ≈ tokens ÷ tok/s ÷ 3600 × hourly-rate). Rough default for ~8B tokens @ batch 24 × accum 8:
```
python -u -m kestrel.train --preset nano --data-dir data_cloud --bf16 --loss-chunks 1 \
    --seq 1024 --batch 24 --accum 8 --steps 40000 --warmup 400 --decay-frac 0.15 \
    --out experiments/nano --log-every 20 --eval-every 500 --ckpt-every 2000 --ckpt-secs 900
```
- **Pull checkpoints off-pod hourly** (community pods can be preempted): `runpodctl send experiments/nano/ckpt.pt` or `scp`. WSD makes resume lossless — re-run the same command.
- Watch `experiments/nano/metrics.jsonl`: loss falling, `loop_gain` R1>R2>R3 emerging, `pkm_gate_mean` growing.
- **Stop the pod the moment it's done** to stop billing; pull the final `ckpt.pt` home first.

## 6. After: bring the model home
`ckpt.pt` → local `experiments/nano/` → run inference locally (a `generate()` script — TBD) or
proceed to the Axiom/TorchSharp integration.
