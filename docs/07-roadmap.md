# 07 — Roadmap, Compute Ledger & Risk Register

*Revised v1.2: the flagship is a **two-tier family** — **Kestrel-Nano (~150M), trained 100% locally on the GTX 1080 for $0**, and **Kestrel-Mini (~500M), cloud-pretrained for ≤ $25**. Caps: **≤ 1 week of local GPU time + ≤ $25 cloud**. The 1080 does the science and trains Nano; the cloud does only the one thing 8 GB cannot (Mini pretraining). Focus domains: coding, tool use, language.*

## The compute ledger (two independent budgets)

**Local — GTX 1080, $0** (science + the Nano hero; sequential, so this is the ~1-week line):

| Line | GPU-hours | Notes |
|---|---|---|
| P0 env + throughput benchmark | ~4 h | replaces every estimate with measurements |
| P2 trainer shakedown (K-test overfit + K-S 50M run) | ~4 h | |
| P3 twin gauntlet: 4 × K-S @ ~300M tokens | ~28 h | 4 overnights |
| P4 scale confirm: 1–2 × K-M @ ~350M tokens | ~22–44 h | |
| P5 Roost prototype (slot-only finetuning) | ~4 h | mostly CPU-side |
| **P6a — Kestrel-Nano: from-scratch local train, ~1–1.5B tokens** | ~50–90 h | FP32; WSD-extendable past a week |
| **Local total** | **~112–150 h ≈ 5–7 days ✅** | tokenizer & data prep are CPU jobs ($0) |

**Cloud — RunPod, ≤ $25** (Kestrel-Mini only; does not consume the local week):

| Line | Where | GPU-hours | Cash |
|---|---|---|---|
| P6b — Kestrel-Mini pretrain, ~3–4B tokens | RunPod RTX 4090 community (~$0.35/h — **verify at P6**) | ~60–65 h | ~$21–23 |
| Reserve (re-runs, transfer) | RunPod | — | ~$2–4 |
| **Cloud total** | | | **≤ $25 ✅** |

If Nano and the ablations over-subscribe the week, WSD lets Nano bank whatever hours remain and continue later — nothing is lost. Cloud alternates if 4090 community pricing drifted: RTX 3090 (~60% speed, ~60% price — a wash), A100/H100 (worse tokens/$ at 500M). **Tokens per dollar, not raw speed, picks the pod.**

## Phases

**P0 — Groundwork (PC, 1–2 days calendar).**
Env pins (WSL2 + torch ≤2.7.1/cu126, `sm_61` verified); SDPA backend check; micro-bench K-S/K-M block fwd+bwd → real tok/s; `.wslconfig` RAM cap; RunPod account + a $0.10 pod-boot sanity test. *Exit: measured tokens/day table; final token budgets set to fit the week.*

**P1 — Tokenizer & data (CPU overnights, $0 GPU).**
48k superword BPE on ~2 GB stratified sample; gate: ≥18% token reduction vs standard BPE; stream-tokenize the code-tilted mixture (Stack-Edu/FineWeb-Edu/tool-use traces/FineMath/Cosmopedia/DCLM — docs/05 §3) into uint16 shards (~4 GB, ~2–2.5B unique tokens, feeding both Nano and Mini); per-domain val shards; probe suite generated & frozen.

**P2 — Trainer (2–3 days dev, ~4 GPU-hours).**
One codebase, two modes: FP32-eager (1080) / bf16-compiled (cloud). Muon+AdamW groups, WSD, grad-accum + checkpointing, resumable everything, JSONL logs, Tier-0 probes. Sanity: K-test overfit; K-S 50M-token shakedown.

**P3 — Twin gauntlet (4 overnights, the core science).**
Four K-S runs @ ~300M tokens, matched FLOPs: **A0 vanilla twin · full Kestrel · Kestrel−loops · Kestrel−memory.** Pre-registered keep-rules (docs/05 §6). The extended ablation list (tokenizer, optimizer, MTP, decay-init, curriculum) runs *only if* the week's budget has slack after P4. *Exit: Kestrel v1 config frozen on evidence.*

**P4 — Scale confirmation (2 runs, ~2 days GPU).**
A0 twin vs frozen Kestrel at K-M @ ~350M tokens. *Exit: wins hold or grow with scale → green-light P6 spend.*

**P5 — Roost prototype (~4 GPU-hours + CPU work; overlaps P4).**
Episodic store + rewrite templates; consolidation job (PKM-slot-only updates, 50/50 replay, eval gate + rollback); teach-me-today: 50 novel facts, ≥60% next-day recall, ≤2% regression on the frozen suite.

**P6a — Kestrel-Nano, the local hero (~50–90 h on the 1080, $0).**
μP-style LR transfer from P3/P4 proxies; from-scratch FP32 train, ~1–1.5B tokens, WSD stable phase + short quality anneal; checkpoint every ~30 min (a multi-day run *will* be interrupted). *Exit: a coherent, 100%-local ~150M model + full training log.*

**P6b — Kestrel-Mini, the cloud stretch (~60–65 h on RunPod, ≤$23).**
Same frozen v1 config, scaled up; ~3–4B tokens, WSD stable phase + short anneal; **checkpoints pulled off-pod hourly** (community pods can be preempted; WSD makes resumption lossless); community-vs-secure pod decided at kickoff on current prices. Runs on the *same calendar days* as P6a (different machine). Then pull weights home for local inference + Roost. *Exit: a stronger ~500M model + full log.*

**P7 — Release & report ($0).**
Both models' weights + tokenizer + code + technical report (all tables, negative results included). The report doubles as the grant/community-compute pitch for the upgrade ladder.

## The upgrade ladder (beyond this budget — same recipe, more tokens)

WSD scheduling means every tier **resumes the previous tier's checkpoint** — no money is ever wasted:

| Tier | Model | Tokens | Approx. cost | Outcome |
|---|---|---|---|---|
| $0 (this plan) | Nano ~150M | ~1–1.5B | free, local | coherent architecture trained end-to-end on one home GPU |
| $25 (this plan) | Mini ~500M | ~3–4B | $25 | stronger, still-undertrained local assistant |
| ~$150 | Mini ~500M | ~15–20B | ~$150 | "Chinchilla-solid" usable 500M coding/tool assistant |
| ~$800–1.5k | Mini ~500M | ~100–150B | ~$800–1.5k | competitive-class vs commercial sub-1B models |
| Grant-scale | either | 1T+ | partner/community | frontier sub-1B attempt |

**The 1080's role after P6:** inference of both models (Nano Q4 ≈ 100 MB; Mini Q4 ≈ 300 MB; ~40–110 tok/s — comfortably interactive) and *all Roost consolidation experiments on the real shipped models* (only memory-slot gradients are needed, which fit in 8 GB even for Mini).

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| 1080 throughput ≪ estimate → Nano under-trained in a week | medium | P0 measures first; Nano's token count flexes (WSD banks what the hours allow) and extends past a week; worst case Nano shrinks toward ~100M |
| Local week over-subscribed (ablations + Nano share one GPU) | medium-high | ablations are cheap (~60–80 h); Nano takes the remainder and can run into a 2nd week — WSD loses nothing |
| Cloud price drift breaks the $25 fit for Mini | medium | P0 price check; pod chosen by tokens/$; fewer tokens is the fallback (WSD-resumable), never a bigger bill |
| Pod preemption / crash mid-Mini | medium | hourly off-pod checkpoint pulls; WSD = lossless resume; trainer shaken down locally first |
| A pillar fails its ablation | medium | cut + report; negative results are still the deliverable |
| GLA scan instability in FP32 | low-med | log-space math (exponents ≤0 by construction), unit test vs naive recurrence; fallback: pure attention hybrid |
| PKM slots die / gate never opens | medium | usage-entropy probe, dead-slot re-init, LayerNorm'd queries; worst case: cut via gauntlet |
| Consolidation results don't transfer down from Meta's 7B evidence | medium | P5 tests it for free before anything depends on it |
| Data download bandwidth/disk | medium | stream-tokenize (never store raw); ~4 GB final footprint |
| Two-machine logistics (plan authored on laptop; lab = PC) | certain | the project folder is self-contained — copy the whole `AI Architecture` folder to the PC before P0 (now resident on the lab PC at `S:\AI Architecture`); docs carry all context |
| Scope creep | high | pre-registered rules; v2 parking lot below |

## v2 parking lot (explicitly deferred)

Titans/TTT neural memory · SEAL-style RL self-edits · byte-level dynamic chunking (H-Net/BLT) · BitNet-ternary · MoE hybrid · multimodal grounding · speculative decoding via the MTP head · 32k+ context extension · a later **~1B tier** (revived once the sub-1B pair validates and compute grows) · deeper agentic/tool-use RL post-training.
