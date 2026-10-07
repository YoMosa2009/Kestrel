# 12 — Roost V2: making consolidation actually learn

*2026-10-06. Kestrel V1 shipped with Roost consolidation reported as "0% recall". This doc
records what was wrong, what was fixed, and what the fixed version measures. All numbers are
from `scripts/roost_v2.py` on a Colab A100, starting from the V1 release weights, on the
50-fact teach-me-today benchmark (nonsense entities, 10 subjects × 5 properties).*

**The gate (unchanged):** a consolidation is accepted only if recall ≥ 60% AND regression on
the frozen per-domain validation suite ≤ 2%. Otherwise it rolls back. Containment holds in
every run below: only the selected memory-slot rows ever change, which is verified after
each run.

## What was wrong with V1

1. **The recall test asked an impossible question.** It cut each fact in half by word count,
   so the cue for "The capital of Varnal is Dreymoor." was "The capital of", which stops
   *before* the entity. With 50 facts on 5 templates, even perfect memorization scores
   about 0%. **Fixed:** the cue now ends right before the answer ("The capital of Varnal
   is"). The test also asks the chat question ("User: What is the capital of Varnal?") and
   reports answer-token NLL, which shows partial learning.
2. **The training barely moved anything.** At V1's settings (lr 1e-3, 200 steps) the loss
   on the new facts went 3.906 → 3.880 and answer NLL 6.664 → 6.642. That is learning
   nothing, independent of the test.
3. **Most of the scored tokens were unlearnable.** Every token was scored, so most of the
   update went into predicting entity names ("The capital of ___"). **Fixed:** an
   answer-only loss mask scores only what has to be stored.
4. **Slot selection missed most loops (bug).** The memory sites sit inside the looped core
   and run once per loop. The selection hook overwrote its capture on every call, so only
   the last loop's slot reads were selectable. The rest were frozen back after every step.
   **Fixed:** every call is recorded, at `r_max` loops.
5. **No visibility.** V1 logged no learning curve on the new material. **Fixed:** the
   curve is logged and returned.
6. **Facts were taught only as plain sentences,** but in chat they are asked as questions.
   **Fixed:** each fact is also taught as a chat exchange.

## Measurements

| round | setting | recall | chat Q&A | answer NLL | regression | gate |
|---|---|---|---|---|---|---|
| 1 | V1 settings (lr 1e-3, 200 steps) | 0% | 0% | 6.66 → 6.64 | +0.04% | fail |
| 1 | + chat forms | 0% | 0% | 6.66 → 6.64 | +0.03% | fail |
| 1 | lr 1e-2, 300 steps | 0% | 0% | 6.66 → 6.31 | +0.02% | fail |
| 3 | slot fix + answer-only, lr 1e-2 | 0% | 0% | 6.66 → 6.02 | +0.07% | fail |
| 3 | slot fix + answer-only, lr 3e-2 | 4% | 4% | 6.66 → 4.92 | +0.45% | fail |
| 3 | slot fix + answer-only, lr 1e-1 | **24%** | **18%** | 6.66 → **1.82** | +2.13% | fail (both limits) |
| 4 | slot fix + answer-only, lr 1e-1, fact steps at R=2, 1,024 slots/site | **40%** | 18% | 1.43 | **+1.85%** | fail (recall) |
| 4 | same, 512 slots/site, 900 steps | 34% | 20% | 1.45 | +1.62% | fail |
| 4 | same, all slots, 60% fact steps | 40% | 22% | 1.46 | +1.39% | fail |

Round 2 (answer-only loss without the slot fix) was stopped early, once round 3's fix was
found. Its partial curve matched round 3 at lr 1e-2.

**Reading:** V1's consolidation learned nothing. With the fixes, the same sparse,
contained, gated mechanism measurably writes facts into the memory slots: answer NLL drops
from 6.66 to 1.82, and recall goes from 0% to 24%, while every non-memory weight stays
frozen. The remaining gap is recall (24% vs 60%) and regression (2.13% vs 2%).

## Round 4

Fact steps run at R=2, which is what the gate and the app use by default. Earlier rounds
used the stochastic training R on every step. Fewer, more fact-specific slots are selected
to bring regression under 2%.

Training the fact steps at R=2 lifted recall from 24% to 40%, and the tighter slot
selection brought regression inside the 2% budget. **One caveat about this benchmark:** each
property has only 4 answers shared across the 10 subjects. A model that learns the answer
*set* without binding each answer to its subject would score about 25%. So 40% means
binding has started but is still weak, and the 50-fact benchmark is **not passed yet**.

### Building V2: one fact per row

V2 teaches V1 seven **true facts about itself** (its maker, name, context length, app, size
and memory system). V1 answers "Who made you?" with "I was a small model, about 15 years
old." The V2 consolidation must pass the same gate on its own facts.

| V2 build | recall | answer NLL | regression | gate |
|---|---|---|---|---|
| packed rows, 600 steps | 42.9% | 3.26 → 0.92 | +0.61% | fail |
| packed rows, 1,200 steps | 57.1% | 3.26 → 0.97 | +0.29% | fail |
| packed rows, 2,400 steps | 57.1% | 3.26 → 0.97 | +0.14% | fail |
| **one fact per row, 1,200 steps** | **100%** | **3.26 → 0.001** | **+0.04%** | **PASS** |

Doubling the steps changed nothing. The packed training material reached loss 0.003 while
the same sentence on its own scored 0.97. Packing facts back-to-back meant each one was
learned only in the context of the facts before it, which sit in the recurrent state. Training
**one fact per row from position 0**, the way a fact is asked in chat, closed the gap
completely.

## What V2 ships

**Kestrel-Nano V2** = V1 + the seven self-facts consolidated into 1,280 memory slots. Every
other tensor is bit-identical to V1, verified by `scripts/assemble_v2.py`. V2's engine parity
against PyTorch: f32 max |Δlogit| ≤ 0.0002, f16 ≤ 0.016, argmax 100%.

**Not done yet:** the 50-fact benchmark with one fact per row (the run ran out of memory and
was fixed by sizing replay in tokens, but it hasn't been re-run). Chat-question recall
lags sentence recall (29% on the self-facts). The Q&A forms are taught, but questions phrased
differently from training aren't reliably answered yet. Both are next for Roost.
