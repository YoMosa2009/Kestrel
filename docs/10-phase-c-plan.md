# 10 — Phase C Plan: 5B-Token Dataset, Built for Generalization

*Written 2026-08-21 after Phase B. **BUILT 2026-08-21 — see §8 for measured results.** Supersedes the one-line
Phase C row in docs/08 §2. The governing constraint from docs/09: **v0.1 learned FORM,
not CONTENT.** Phase C is where we attack that at the data layer.*

## 1. Goal

Build **~5B unique high-quality tokens** on this machine, composed and processed
specifically to push Nano from *reproducing the shape of text* toward *generalizing to
text it has not seen*. Then (Phase D) over-train on it.

Two deliverables, both local, both $0:
1. **`data_5b/`** — ~5B unique tokens (~9.3 GB uint16), deduplicated, domain-tagged, with
   held-out and contamination-checked validation slices.
2. **A staged curriculum + anneal** implemented in the trainer (docs/05 §3 specifies it;
   v0.1 never used it — it trained on one flat mixture start to finish).

## 2. Why v0.1 copies form — the diagnosis

Three causes, in order of size:

1. **Undertrained.** 1.31B tokens ≈ 8 tokens/param. Form (syntax, docstring shape, list
   structure, recipe layout) is high-frequency and learned first; facts, arithmetic and
   composition need far more exposure. This is the dominant cause and Phase D's job.
2. **The mixture actively rewards copying.** `codeparrot-clean` at 35% is raw GitHub:
   license headers, import blocks, boilerplate, near-duplicate files. That is a
   template-memorization corpus. Meanwhile there was **no math** (hence "3 apples minus 1
   gives *one apple per apple*"), **no instruction data** (hence it continues instead of
   answering), and **no tool-use traces** despite tool-use being a stated target domain.
3. **No deduplication whatsoever.** The v0.1 pipeline streams and tokenizes with zero
   dedup. Duplicated text is the single best-documented driver of memorization over
   generalization (Lee et al., *Deduplicating Training Data Makes Language Models Better*).

## 3. What actually drives generalization (and what we do about each)

| Lever | Evidence | Phase C action |
|---|---|---|
| **More tokens** | the memorize-to-generalize transition needs training past memorization | 5B unique enables Phase D's 20B+ seen (<=4 epochs, the free-repetition zone) |
| **Deduplication** | duplicates cause memorization | **NEW:** exact-hash + MinHash/LSH near-dup pass over every shard |
| **Many phrasings of the same fact** | Allen-Zhu & Li (*Physics of LMs 3.3*, already in docs/02): knowledge is only *extractable* if seen in multiple rewrites | raise **Cosmopedia-v2** (synthetic textbooks = built-in paraphrase) to 20% |
| **Reasoning density** | math transfers to code and multi-hop composition | **NEW:** FineMath 4+ at ~12% — aimed straight at the arithmetic failure |
| **Question-to-answer mapping** | base models continue; instruction data teaches the mapping | **NEW:** SmolTalk-style instruct at ~8% (anneal-weighted) |
| **Executable causality** | code with *traces* teaches state and consequence, not just syntax | replace raw GitHub with quality-filtered code + execution traces; cut code 35% to 20% |
| **Staged curriculum + quality anneal** | SmolLM playbook: the final anneal on premium data moves benchmarks disproportionately | **NEW:** implement mixture staging + a final anneal phase in the trainer |
| **Adaptive compute (our own pillar)** | Ouro: loops help *manipulation/composition* — i.e. exactly generalization | fix R-sampling (docs/09) so R>=3 is actually trained, not just R=2 |

### The honest boundary
We are **not** solving "understanding." Mitchell's barrier of meaning (docs/02 §A) stands,
and this project's stated position is to *mitigate and measure*, never to claim it away.
What Phase C can credibly deliver is **measurably better generalization**: less verbatim
copying, correct arithmetic on unseen numbers, reasoning that survives unseen phrasings.
If we cannot measure it, we do not claim it — see §6.

## 4. The mixture (5B unique)

| Source | Share | Tokens | Why it is in |
|---|---|---|---|
| FineWeb-Edu | 35% | 1.75B | language fluency, quality-filtered web |
| Cosmopedia-v2 | 20% | 1.00B | **many-phrasings knowledge, so facts become extractable, not memorized** |
| Code (quality-filtered + execution traces) | 20% | 1.00B | target domain; causality. Down from 35%, deduped hard |
| **FineMath 4+** | 12% | 0.60B | **NEW** — reasoning/arithmetic, the worst v0.1 failure |
| **Instruct / SmolTalk subset** | 8% | 0.40B | **NEW** — answers vs continuations; anneal-weighted |
| **Tool-use / function-call traces** | 5% | 0.25B | **NEW** — stated target domain, absent from v0.1 |

**Anneal phase** (final ~10% of Phase D training): re-weight to the premium slice —
Cosmopedia + FineMath + instruct — and drop raw web.

**Step 0 — verify availability before building anything.** We were burned in v0.1:
`stack-edu` and `smollm-corpus/python-edu` ship only `blob_id` pointers (no inline text)
and `the-stack*` is gated (see the data-sources memory). Every source above must be
confirmed to stream with inline text *before* the build starts; each needs a fallback.

## 5. Build and cost reality (measured on this machine)

- **Disk:** 5B uint16 tokens = **9.3 GB** (+ dedup working space). Free on S: **95.7 GB** OK
- **Build time:** v0.1 did 999M tokens in ~20 min locally, so 5B is **~1.5–2 h streaming**,
  plus a dedup pass (CPU/RAM-bound; must stay inside 16 GB — shard-wise MinHash, never
  a whole-corpus in-RAM set).
- **RAM discipline:** the v0.1 build crashed twice (network drop, then `MemoryError`).
  Keep `--shuffle-buffer 0`, small batches, and the reconnect logic that fixed it.

**Local training on 5B is the honest constraint.** At the measured post-Phase-B rate of
**2,151 tok/s**:

| Local run | Tokens seen | Wall-clock |
|---|---|---|
| 1 week | 1.3B | 7 days |
| 2 weeks | 2.6B | 14 days |
| **1 epoch of 5B** | **5.0B** | **~27 days** |

For comparison, the same 5B on a rented 3090 (~8k tok/s) is ~174 h, about **$89**. Local is
$0 but ties up the machine for a month. **WSD de-risks this completely**: the run can be
stopped at any checkpoint and either shipped as-is or resumed — including resumed *on the
cloud later*. Starting locally costs nothing but time and forecloses nothing.

## 6. Success criteria — how we know it generalized

Measured against the v0.1 checkpoint as the control. Generalization claims require
**contamination checks** (verify eval text is not in the training shards) or they are void.

1. **Arithmetic on unseen numbers** — 3-digit add/sub held out from training. v0.1 baseline: ~0.
2. **Perturbed-phrasing robustness (docs/06 D4)** — accuracy *drop* between original and
   surface-rewritten PIQA/WinoGrande items. Smaller drop = less form-matching.
3. **2-hop composition probe** (already in Tier-0) — the direct test of manipulation, and
   the one the loop pillar is supposed to help.
4. **Verbatim-copy rate** — % of generated 20-grams found in the training corpus. Should
   fall. The most direct "is it copying?" metric, and cheap to compute.
5. **Repetition collapse** — the "Pluto x5" failure should disappear.
6. **Per-domain val loss** — code/math/web tracked separately, not just an aggregate.

## 7. Order of work

1. Verify every source streams with inline text (§4 step 0); fix fallbacks.
2. Extend `scripts/prepare_data.py`: new sources, dedup pass, domain tags, curriculum stages.
3. Build `data_5b/` (~2 h) + held-out val + contamination index.
4. Implement curriculum staging + anneal in `kestrel/train.py`.
5. Build the generalization eval suite (§6) and **baseline v0.1 on it** — so Phase D has a control.
6. Then Phase D (train), local or cloud, WSD-resumable either way.


---

## 8. Phase C RESULTS (built 2026-08-21)

**`data_5b/` — 4.860B tokens, 30 per-domain train shards + 10 val shards (11.6M held out),
9.1 GB on disk, built in ~2 hours. Validated: no out-of-range token IDs, all renderers
produce correct text.**

| domain | tokens | actual | target | |
|---|---:|---:|---:|---|
| code | 1,502M | 30.8% | 30% | ✅ StarCoderData ×10 languages |
| web | 1,200M | 24.6% | 24% | ✅ |
| know | 700M | 14.4% | 14% | ✅ |
| math | 500M | 10.3% | 10% | ✅ FineMath — new |
| cli | 300M | 6.2% | 6% | ✅ shell/powershell/batchfile/dockerfile — new |
| repo | 250M | 5.1% | 5% | ✅ git-commits + github-issues — new |
| inst | 250M | 5.1% | 5% | ✅ |
| docs | 100M | 2.1% | 2% | ✅ |
| **tool** | **53M** | **1.09%** | 3% | ⚠️ every source exhausted |
| **sec** | **16M** | **0.33%** | 1.5% | ⚠️ every source exhausted |

### Two honest shortfalls
**tool-use and security data simply do not exist at the requested scale in open datasets.**
Glaive (~40M), ToolACE (8M), all four Hermes configs (5M total), CyberNative (1M) and
code_x_glue defect-detection (15M) were each pulled to exhaustion. A top-up run raised sec
from 0.02% → 0.33% (16×) and tool from 0.8% → 1.09%, and that is the ceiling without either
synthesising data or finding gated/commercial corpora. **Consequence: expect format familiarity
for tool-calls and security review, not real capability.** If tool-use matters for v0.2, the
realistic path is *generating* synthetic tool-call traces, not sourcing more.

### A correction to §2 of this plan
§2 listed "no deduplication whatsoever" as one of the three causes of v0.1's form-copying.
**That was overstated.** With dedup on, only **2.5% of documents were dropped** (125k of ~4.6M),
because FineWeb-Edu, Cosmopedia and StarCoderData are all already deduplicated upstream. The
dedup pass is a cheap safety net for cross-source overlap, not the lever. **The actual fix for
code-boilerplate memorization was replacing raw `codeparrot-clean` with quality-filtered
StarCoderData** — source quality, not our filtering.

### What v0.1 never had
PowerShell/batchfile (v0.1's shell knowledge was accidentally Linux-only), `git-commits`
(`<commit_before>`/`<commit_after>` — the shape of an agent edit), `github-issues`
(problem→solution threads), math, instruct, tool, and security. Those are the domains most
aligned with the stated target profile.

### Still open before Phase D
Items 4–5 of §7 are **not** done: curriculum staging + anneal in `kestrel/train.py`, and the
generalization eval suite (§6) with v0.1 baselined on it as the control.
