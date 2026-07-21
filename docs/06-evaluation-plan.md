# 06 — Evaluation Plan

Three tiers, cheapest first. The suite's job: **discriminate between design choices at K-S scale overnight**, then prove the headline claims at K-M and on the two shipped models (Kestrel-Nano, Kestrel-Mini). Because the target domains are **coding, tool use, and language**, those get first-class metrics — not just generic perplexity.

## Tier 0 — In-run probes (every ~500 steps, seconds each)

| Probe | What it detects |
|---|---|
| Val loss per domain shard (web/code/math/instruct) | mixture balance, overfit onset |
| **Loop-gain curve**: val loss at R=1/2/3 on a fixed batch | is the looped core actually using iterations? (flat curve → Pillar 2 failing) |
| **State-carry delta**: chunked val loss with vs without carried GLA state | is session memory real? |
| PKM usage entropy + dead-slot % + gate magnitude | dead-memory pathology (re-init dead slots if >30%) |
| Synthetic micro-tasks (200 items each): copy/induction, 2-hop composition, bracket state-tracking, 3-digit arithmetic | capability fingerprints that move earlier than benchmarks |
| Grad-norm / update-RMS per param group | divergence early-warning (FP32 GLA scan sanity) |

## Tier 1 — Nightly benchmark sweep (lm-eval-harness, ~30-60 min on 1080)

Zero-shot accuracy, normalized where standard: HellaSwag, **PIQA**, ARC-easy, ARC-challenge, **WinoGrande**, **CommonsenseQA**, OpenBookQA, SciQ, BoolQ, LAMBADA.

- **Commonsense slice** (bold above + HellaSwag) reported as its own average — the Mitchell metric, tracked against ablation A9's curriculum.
- **Domain headline metrics (Nano + Mini):** HumanEval and MBPP pass@1 (temp 0) for coding; a small held-out **tool-use / function-calling** suite (does the model emit well-formed calls with correct args, and use returned results?) for tool usage; LAMBADA + a writing-quality rubric for language. GSM8K reported (expect near-floor at our token counts) as a reasoning-transfer signal.
- Context rows for the relevant size class (Nano vs SmolLM2-135M/360M, Gemma-3-270M; Mini vs Qwen3-0.6B, SmolLM2-1.7B, Llama-3.2-1B) appear clearly labeled **unmatched** — they trained on hundreds to thousands of times our token budget. The load-bearing comparison is always the matched twin, never the leaderboard.
- Reporting rule: every table lists tokens seen + FLOPs + wall-clock; comparisons only at matched compute. Public models (SmolLM2-135M etc.) appear as *context rows*, clearly labeled unmatched (they saw 100–1000× more tokens).

## Tier 2 — Capability demos (the claims no benchmark covers)

**D1 — "Teach-me-today" (success criterion S3).** 50 facts about *fictional* entities (guaranteed novel) taught in a session → episodic store → nightly consolidation → next day, fresh process: (a) cloze + QA recall of the 50 facts, target ≥60%; (b) frozen Tier-1 suite regression ≤2%; (c) control: same test *without* consolidation (retrieval-only) to isolate what the weights absorbed vs. what the vector store carries.

**D2 — Session resume (S4).** Day 1: 30-min conversation establishing entities/decisions; serialize state (~MBs, no transcript). Day 2: cold process + restored state; scripted questions referencing day-1 content. Score: referenced-detail accuracy vs. (i) no state, (ii) full-transcript-in-context oracle.

**D3 — Adaptive compute.** Accuracy vs. loop count R on the 2-hop composition probe and ARC-challenge. Claim shape: monotone gains to R=3–4 on hard tasks, flat on easy ones → "think harder" dial works.

**D4 — Long-tail brittleness (Mitchell stress test).** Perturbed rewrites of PIQA/WinoGrande items (surface changes, same answer). Report accuracy *drop* vs. originals, Kestrel vs. A0 twin — measures whether our curriculum/memory reduce brittleness or just fit the benchmark.

## Baseline protocol

- **A0 twin** at every scale: identical tokenizer, data order, tokens, optimizer, params (±2%) — vanilla pre-norm transformer (RoPE all layers, no loops/PKM/GLA). Any Kestrel number without its twin row is invalid.
- Seeds: 1 for scouting, 3 for headline (mean ± range).
- All eval code + prompts versioned in-repo; no cherry-picking task subsets after seeing results (task list frozen in this doc).

## What "winning" looks like (pre-registered)

1. K-S/K-M: Kestrel ≥ twin on ≥7/10 Tier-1 tasks at matched compute (S1) with the loop-gain and state-carry probes non-flat.
2. D1 passes thresholds (S3) — this alone is a publishable small-lab result given the Meta forgetting numbers were shown at 7B, not 100M.
3. Honest failures documented: any pillar cut by ablation appears in the writeup with its numbers.
