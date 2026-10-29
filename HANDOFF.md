# Kestrel handoff (paste into a new chat)

> You are continuing **Project Kestrel** with me (handle MalxTech; Colab/Drive Google account
> **malxshrouds@gmail.com**). Read this whole file, then `docs/11-v3-plan.md` §0 and
> `docs/12-roost-v2.md`, before doing anything. **Do not build or run anything until I say go.**

## 1. What Kestrel is
A from-scratch small LLM. **Kestrel-Nano = 157.2M params**: gated-linear-attention hybrid
(15 NoPE GLA blocks + 5 RoPE GQA attention blocks; every 4th block is attention), a
**looped core** (5 entry + 10 core × R + 5 exit blocks, R=1..3, trained R=2), and 2
**product-key memory** sites (blocks 7 and 14, 128² = 16,384 slots each, top-k 8).
d_model 640, 10 heads (2 KV), d_ff 1792, vocab 49,152 (byte-level BPE, digits split),
context **4,096** (position interpolation ×4, `rope_scale=4`). **Roost** = lifelong
learning: an episodic store, persistent session state, and gated consolidation that writes
facts into the PKM value slots only.

- Repo: `S:\AI Architecture` = https://github.com/YoMosa2009/Kestrel (public, branch `main`)
- Read first: `README.md`, `MASTER-PLAN.md`, `docs/11-v3-plan.md` (V3 plan + field
  observations), `docs/12-roost-v2.md` (Roost fixes + every measurement), `app/README.md`

## 2. Released (GitHub releases)
| release | contents |
|---|---|
| **v1.0.0 "Kestrel V1"** | `Kestrel-V1-Windows-x64.zip` (KestrelStudio.exe + kestrel-nano-v1-f16.gguf), `kestrel-nano-v1-f16.gguf`, `-q8_0.gguf`, **`kestrel-nano-v1-weights.pt`** (PyTorch cfg+state dict), **`roost-data.zip`** (Roost val shards + 24M-token replay sample) |
| **v2.0.0 "Kestrel V2"** (Latest) | `Kestrel-V2-Windows-x64.zip` (app labeled V2 + V2 f16 model), `kestrel-nano-v2-f16.gguf`, `-q8_0.gguf`, `results.json` |

- **V1** = pretrain 3.0B tokens (RTX 3060) → 4k context extension (1,000 steps, Colab A100)
  → SFT (165 steps at seq 4096, assistant-only loss mask). 3.23B tokens total.
  Evals: loss at 3k–4k 1.868→1.580; val @1024 1.771 (no regression); two-hop 45%; copy 0%.
- **V2** = V1 + 7 true self-facts (maker MalxTech, name, ctx 4096, app Kestrel Studio, 157M,
  Roost) consolidated into 1,280 PKM slots: recall 0%→100%, answer NLL 3.26→0.001,
  regression +0.04%; every other tensor bit-identical to V1 (`scripts/assemble_v2.py`).
  V2 f16 md5 `8f8778f491e1d9c8823955f016cc35b3`.
- Native engine parity vs PyTorch: f32 max|Δlogit| ≤ 2e-4, f16 ≤ 0.018, q8_0 ≤ 0.28,
  argmax 100%. The tokenizer matches HF on 319/319 strings. ~23 tok/s decode on 12 CPU threads.

## 3. Code map (what matters now)
- `kestrel/` model.py, train.py (resumable, WSD, `--rope-scale`, `--sft-dir`, status.json/
  control.json), data.py, generate.py, `roost/` (store.py, session.py, **consolidate.py**)
- `scripts/` export_gguf.py (`--dtype f32|f16|q8_0 --version --fixture --save-pt`),
  roost_v2.py (sweep + V2 build; `--only`, `--v2-setting`, `--skip-benchmark`),
  roost_v2_release.py, assemble_v2.py, eval_context.py, eval_generalization.py, prepare_sft.py
- `notebooks/` kestrel_colab.ipynb (ctx4k→SFT→Roost→release, needs Drive mount),
  kestrel_v2_roost.ipynb (runs from **public release assets, no Drive**)
- `app/` Kestrel.Engine (C# .NET 8 inference), Kestrel.Studio (WPF, `StudioVersion` in
  Support.cs, now "V2"), Kestrel.Cli (`kestrel parity|bench|chat|tokcheck`), publish.ps1,
  finish_v2.sh. Build: `powershell -File app/publish.ps1` → `app/dist/KestrelStudio.exe`
- Roost final recipe (V2): answer-only loss, one fact per row, fact steps at R=2,
  lr 1e-1, 1,200 steps, new_frac 0.75, all activated slots, replay sized in tokens, gate
  recall ≥60% & regression ≤2%.

## 4. Where files live
- Local release staging: `C:\Users\mosaa\KestrelRelease\` (v1/, v2/ weights .pt + gguf; zips)
- Google Drive `Kestrel/` (via Drive for desktop on `I:\My Drive\Kestrel`, **not mounted
  right now**): checkpoints/ (v0.1, phase-d, v0.2-final), datasets/corpus-5b (9.07 GB,
  FILES.json) + datasets/sft, runs/ctx4k, runs/sft (SFT ckpt.pt), evals/, roost/, release/
- Local corpus: `S:\AI Architecture\data_5b\` (and data_sft/). S: is nearly full (~29 GB free); C: ~85 GB.
- Python: `.venv/Scripts/python.exe` (torch 2.7.1, tokenizers). .NET SDK 10 installed (targets net8).

## 5. Rules I set (follow them)
- **No model compute on this PC.** Training, evals and inference all go to Colab. File
  conversion/packaging/builds locally are fine. Don't use the local GPU.
- **All V3 training runs on Colab A100**, not the 3060. Keep working on **Kestrel-Nano (157M)**.
- **Ship V3 as one release** (app + model bundled, clearly labeled V3).
- Don't build anything until I say go. Prefer automating; ask me only when it truly
  needs me (Drive OAuth popup, native browser prompts).
- Browser automation: use the **built-in browser pane in the Claude desktop app** (tools
  `mcp__Claude_Browser__*`). Verified 2026-10-28: signed in to Colab on the account above
  (Pro), and clicks, navigation, screenshots and page JS all work. If Colab shows another
  language, the URL carried `?hl=<lang>`; open `https://colab.research.google.com/?hl=en` once
  and it stays English. The fallback is the Claude-in-Chrome extension in Edge (device
  "MALXPC"), where Colab dialogs freeze the extension. Do sign-in popups and native download
  prompts by hand.
- Colab: **~5.3 units/A100-hour** (Resources panel, 2026-10-28; V1-era estimate was 6.8).
  **259 units left** after step 0 (2026-10-28; the earlier "295" was wrong). Google AI Pro
  grants 200/month. Balance: command palette → "View resources". Always "Disconnect and
  delete runtime" when a job ends.

## 6. The V3 plan (agreed, not started), see docs/11 "Agreed V3 plan" (body of docs/11 §1–2 is superseded)
Problems seen in V2 chat: refuses "Hi"/"Hello"; echoes questions ("The largest animal in
the world."); code shape right but fake `# Output:` values and stray imports; repeats a stock
line; high variance and needs constant sampling tweaks. Cause: SFT mix (refusal slice, no
greetings, terse answers), thin knowledge, flat distribution at 157M.

Chosen approach (instead of expensive logit distillation): **synthetic-data distillation via
downloadable datasets** (Cosmopedia v2, FineWeb-Edu high-score, Python-Edu, FineMath) + **DPO**
on downloadable preference data (e.g. UltraFeedback) + an SFT data fix.

| step | what | units |
|---|---|---|
| 0 | measure A100 speed at seq 2k vs 4k, check balance | ~2 |
| 1 | new SFT mix (greetings, small talk, full factual answers, multi-turn recovery, refusals only for harmful, no fake Output comments); tokenize on a CPU runtime; retrain SFT | ~10 |
| 2 | DPO | ~14 |
| 3 | MTP head ablation (keep only if it wins) | ~20 |
| 4 | continued pretraining on 1.5–2B quality tokens from V2, low LR, resumable | ~150–200 |
| 5 | re-SFT + DPO, Roost self-facts, export gguf, parity, app defaults (R=2, temp ~0.5, top-p 0.9, top-k 30, rep 1.1, memory recall off, neutral system prompt) → **release V3** | ~20 |

**Step 0 is done** (see docs/11 "Step 0 results"): seq 4096 at batch 16 × accum 3 runs at
17.9k tok/s (bench, all R=2), and seq 2048 is only ~9% faster. Use seq 4096. That is ~82 units
per 1B tokens at 5.3 units/hour. Next is step 1.

With 259 units, **steps 1–5 still fit** (~220–270 units at 1.5–2B tokens in step 4), so V3
can be done in one go without waiting for a refill. Keep a ~25-unit reserve for reruns: if
step 0's measured speed or steps 1–3 cost more than estimated, trim step 4's token count, not
step 5's checks.

## 7. Open items
- 50-fact Roost benchmark: 40% recall (gate 60%); one-fact-per-row version not yet run
  (`row-bench` setting, needs the replay-by-tokens fix that is now in).
- Chat-question recall of taught facts is low (29%).
- Tested sampling that works best: R=2, temp 0.4–0.5, top-p 0.90, top-k 30, rep 1.15, max 256.
- System prompt in use: "You are Kestrel-Nano, a small AI assistant made by MalxTech… If you do not
  know something, say so in one sentence." The last sentence encourages refusals; drop it in V3.
