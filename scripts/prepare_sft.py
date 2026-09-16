"""Build the Phase E SFT set — designed for a 4k context from the start.

Two things make this different from ordinary SFT data prep:

1. **It emits a LOSS MASK.** Standard SFT computes loss only on the Assistant
   tokens; training on the User/System prompt spends capacity learning to
   generate questions. We write a parallel uint8 mask (1 = compute loss here)
   alongside the uint16 tokens, so the trainer can do this properly.

2. **It uses the EXACT template pretraining already saw** -
   `System:` / `User:` / `Assistant:` / `Tool:` joined by blank lines, and
   `<|endoftext|>` (token 0) to terminate. The base model has seen ~300M tokens
   in this format and ~4.6M document terminations, so format and stopping are
   reinforcement rather than new learning. A different template (ChatML,
   Llama-style) would burn scarce capacity teaching a new convention for nothing.

Composition (docs: Phase E plan):
    code 30 | instruct 25 | cli 15 | tool 15 | refusal 5 | sec 5 | plan 5

`refusal` and `plan` are constructed here rather than downloaded. No good open
refusal set exists at this scale, and the `plan` slice is deliberately shallow -
1-3 procedural steps anchored to a domain the model knows, NOT open-ended
reasoning. A 157M model that scores 0% on 3-digit arithmetic cannot reason in
text; training it to produce reasoning-shaped output just yields longer, more
confident errors. The plan slice is built as an ablation: train with and without
it and keep it only if it beats the control.

Usage:
    python -m scripts.prepare_sft --examples 25000 --seq 4096 --out data_sft
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time

import numpy as np
from tokenizers import Tokenizer

ROLE = {"system": "System", "user": "User", "human": "User",
        "assistant": "Assistant", "gpt": "Assistant", "tool": "Tool",
        "function_response": "Tool"}


# ------------------------------------------------------------------ rendering

def render(turns: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Normalize (role, content) pairs to the pretraining template's role names."""
    out = []
    for role, content in turns:
        if not content:
            continue
        r = ROLE.get(str(role).strip().lower(), "User")
        out.append((r, str(content).strip()))
    return out


# ------------------------------------------------------------------- adapters
# Each returns a list of (role, content) turns, or [] to skip the row.

def a_smoltalk(row):
    return render([(m.get("role"), m.get("content")) for m in (row.get("messages") or [])])


def a_magicoder(row):
    p, s = row.get("problem"), row.get("solution")
    return render([("user", p), ("assistant", s)]) if p and s else []


def a_evol_code(row):
    i, o = row.get("instruction"), row.get("output")
    return render([("user", i), ("assistant", o)]) if i and o else []


def a_nl2bash(row):
    nl, sh = row.get("nl_command"), row.get("bash_code")
    if not nl or not sh:
        return []
    return render([("system", "You are a shell expert. Answer with the command only."),
                   ("user", nl), ("assistant", f"```bash\n{sh.strip()}\n```")])


def a_toolace(row):
    turns = [("system", row.get("system") or "")]
    conv = row.get("conversations")
    if isinstance(conv, str):
        try:
            conv = json.loads(conv.replace("'", '"'))
        except Exception:
            return []
    if not isinstance(conv, list):
        return []
    turns += [(m.get("from"), m.get("value")) for m in conv if isinstance(m, dict)]
    return render(turns)


def a_glaive(row):
    """Glaive ships one blob with SYSTEM:/USER:/ASSISTANT: markers inline."""
    txt = ((row.get("system") or "") + "\n" + (row.get("chat") or "")).strip()
    if not txt:
        return []
    parts = re.split(r"\n?(SYSTEM|USER|ASSISTANT|FUNCTION RESPONSE):", txt)
    turns, i = [], 1
    m = {"SYSTEM": "system", "USER": "user",
         "ASSISTANT": "assistant", "FUNCTION RESPONSE": "tool"}
    while i + 1 < len(parts):
        turns.append((m.get(parts[i], "user"), parts[i + 1]))
        i += 2
    return render(turns)


def a_hermes(row):
    """Hermes function-calling: a conversations list, same shape as ToolACE."""
    conv = row.get("conversations")
    if isinstance(conv, str):
        try:
            conv = json.loads(conv.replace("'", '"'))
        except Exception:
            return []
    if not isinstance(conv, list):
        return []
    return render([(m.get("from"), m.get("value")) for m in conv if isinstance(m, dict)])


def a_sec(row):
    q, ch = row.get("question"), row.get("chosen")
    if not q or not ch:
        return []
    sys_ = row.get("system") or "Review the code for security defects."
    return render([("system", sys_), ("user", q), ("assistant", ch)])


# ------------------------------------------- constructed slices (not downloaded)

_REFUSE_TEMPLATES = [
    ("What is the population of {x}?",
     "I don't know. I'm a small model and I don't reliably hold facts like that - "
     "I'd rather say so than guess."),
    ("Who won the {x} in 2019?",
     "I'm not sure, and I don't want to invent an answer. I'd check a current source."),
    ("What is my {x}?",
     "I don't have that information. If you tell me, I'll remember it for this session."),
    ("Summarize the contents of the file {x}.",
     "I can't see that file. Paste the contents and I'll summarize them."),
    ("What did we discuss last week about {x}?",
     "I don't have a record of that conversation. Could you remind me?"),
    ("Compute {x} to twelve decimal places.",
     "I'm not reliable at arithmetic that precise - I'd get it wrong. Use a calculator "
     "or a Python one-liner for that."),
    ("What is the current price of {x}?",
     "I don't have live data, so I can't tell you a current price."),
]
_REFUSE_FILL = ["Vantersk", "the Halgrove Prize", "API key", "config/prod.yaml",
                "the migration plan", "817.4419 / 3.77", "SLVR", "Brennick",
                "the deployment window", "the staging password", "notes.md",
                "the Q3 rollout", "tin", "Ordmark", "the retro action items"]


# --- identity -----------------------------------------------------------------
# The corpus contains essentially NO model-identity text: zero mentions of
# ChatGPT / OpenAI / GPT / "language model" across the web, knowledge and instruct
# slices (FineWeb-Edu and Cosmopedia are educational prose, not AI discourse). So
# this fills a vacuum rather than overwriting a belief, which is why a small slice
# is enough. Without it the model has no answer at all and will improvise one.
#
# Answers deliberately VARY in wording while holding the facts constant, so the
# model learns the fact rather than memorising one string. They are also kept
# short: a 157M model will not reproduce a paragraph faithfully, and every token
# spent here is a token not spent on code.
_ID_NAME = "Kestrel-Nano"
_ID_QUESTIONS = [
    "What AI model are you?", "What AI are you?", "Which model is this?",
    "What are you?", "Who are you?", "What's your name?", "What is your name?",
    "Tell me about yourself.", "Introduce yourself.", "What model am I talking to?",
    "Are you ChatGPT?", "Are you GPT-4?", "Are you Claude?", "Are you Gemini?",
    "Are you a large language model?", "What version are you?",
    "How big are you?", "How many parameters do you have?",
    "Who made you?", "Who created you?", "Who built you?",
    "What can you do?", "What are you good at?", "What are you bad at?",
    "Can you remember things?", "Do you run in the cloud?",
]
_ID_ANSWERS = {
    "name": [
        f"I'm {_ID_NAME}, a small language model built to run locally.",
        f"{_ID_NAME} - a compact model, about 157 million parameters.",
        f"I'm {_ID_NAME}. Small, local, and focused on code and terminal work.",
        f"My name is {_ID_NAME}.",
    ],
    "other": [
        f"No - I'm {_ID_NAME}, a small open model that runs on your own machine.",
        f"I'm not. I'm {_ID_NAME}, a 157M-parameter model, much smaller than those.",
        f"No. I'm {_ID_NAME} - a local model, not a hosted one.",
    ],
    "size": [
        f"About 157 million parameters, which is tiny as language models go.",
        f"I'm {_ID_NAME}: roughly 157M parameters, small enough to run offline.",
        f"157M parameters. That makes me fast and portable, and also limited.",
    ],
    "maker": [
        f"I was built by MalxTech as part of Project Kestrel.",
        f"MalxTech built me - {_ID_NAME}, from the Kestrel architecture.",
        f"I come from Project Kestrel, built by MalxTech.",
    ],
    "can": [
        "I'm best at writing code, shell and PowerShell commands, and calling tools.",
        "Code, CLI commands, and tool use are what I'm built for.",
        "I help with programming, terminal work, and structured tool calls.",
    ],
    "cant": [
        "I'm small, so I get facts wrong and I'm poor at arithmetic and long reasoning. "
        "Check anything that matters.",
        "Maths, precise facts, and long multi-step reasoning are weak points. "
        "I'd rather tell you that than bluff.",
        "I have limited world knowledge and I make mistakes. Treat me as a fast "
        "assistant, not an authority.",
    ],
    "memory": [
        "Yes - I keep session state between conversations, and facts you teach me "
        "can be consolidated into my memory overnight.",
        "I can. My memory layers let me keep what you teach me across sessions.",
    ],
    "local": [
        "No, I run locally on your machine. Nothing is sent anywhere.",
        "I run offline on your own hardware - no cloud involved.",
    ],
}
_ID_ROUTE = {
    "Are you ChatGPT?": "other", "Are you GPT-4?": "other", "Are you Claude?": "other",
    "Are you Gemini?": "other", "Are you a large language model?": "other",
    "How big are you?": "size", "How many parameters do you have?": "size",
    "What version are you?": "size",
    "Who made you?": "maker", "Who created you?": "maker", "Who built you?": "maker",
    "What can you do?": "can", "What are you good at?": "can",
    "What are you bad at?": "cant",
    "Can you remember things?": "memory", "Do you run in the cloud?": "local",
}


def build_identity(n: int, rng: random.Random) -> list[list[tuple[str, str]]]:
    """Teach the model what it is. Small slice, high phrasing variety."""
    out = []
    while len(out) < n:
        q = rng.choice(_ID_QUESTIONS)
        a = rng.choice(_ID_ANSWERS[_ID_ROUTE.get(q, "name")])
        turns = [("user", q), ("assistant", a)]
        if rng.random() < 0.3:      # sometimes with a system prompt, sometimes not
            turns.insert(0, ("system", f"You are {_ID_NAME}, a small local assistant."))
        out.append(render(turns))
    return out


def build_refusals(n: int, rng: random.Random) -> list[list[tuple[str, str]]]:
    """Teach 'I don't know'. A base model never does this, and this model will
    hallucinate heavily - this is the cheapest anti-hallucination lever there is."""
    out = []
    while len(out) < n:
        q, a = rng.choice(_REFUSE_TEMPLATES)
        out.append(render([
            ("system", "You are a careful assistant. Say when you do not know."),
            ("user", q.format(x=rng.choice(_REFUSE_FILL))),
            ("assistant", a)]))
    return out


def build_plans(rows: list[list[tuple[str, str]]], n: int, rng: random.Random
                ) -> list[list[tuple[str, str]]]:
    """Prepend a SHORT procedural plan to existing code/CLI answers.

    Deliberately 1-2 lines and domain-anchored. This is format conditioning, not
    reasoning: the plan constrains what gets generated next. Kept as an ablation
    slice so it can be cut if it does not beat the no-plan control.
    """
    out = []
    for turns in rows:
        if len(out) >= n:
            break
        if not turns or turns[-1][0] != "Assistant":
            continue
        ans = turns[-1][1]
        steps = []
        if "```" in ans:
            steps.append("write the code")
        if re.search(r"\bimport |\brequire\(", ans):
            steps.insert(0, "bring in the dependencies")
        if re.search(r"\bdef |\bfunction |\bclass ", ans):
            steps.append("define the function")
        if re.search(r"\breturn\b", ans):
            steps.append("return the result")
        if len(steps) < 2:
            continue
        plan = "Plan: " + ", then ".join(steps[:3]) + "."
        out.append(turns[:-1] + [("Assistant", plan + "\n\n" + ans)])
    return out


# ------------------------------------------------------------------ tokenizing

def encode_example(tok, turns, seq: int, eot: int):
    """Return (ids, mask) or None. mask=1 only on Assistant content + its EOT."""
    ids: list[int] = []
    mask: list[int] = []
    for i, (role, content) in enumerate(turns):
        head = f"{role}: " if i == 0 else f"\n\n{role}: "
        h = tok.encode(head).ids
        c = tok.encode(content).ids
        ids += h + c
        # The role header itself is NOT a target - the model should learn to
        # produce the answer, not to predict that it is its own turn.
        mask += [0] * len(h) + ([1] * len(c) if role == "Assistant" else [0] * len(c))
    ids.append(eot)
    mask.append(1 if any(mask) else 0)          # learning to STOP is the point
    if len(ids) > seq or not any(mask):
        return None
    # Store as numpy immediately. Holding 25k examples as Python int lists costs
    # ~2 GB (36 bytes per int); as uint16+uint8 it is ~45 MB.
    return np.asarray(ids, dtype=np.uint16), np.asarray(mask, dtype=np.uint8)


SOURCES = [
    # (key, hf name, config, adapter, share)
    ("code:magicoder", "ise-uiuc/Magicoder-OSS-Instruct-75K", None, a_magicoder, 0.18),
    ("code:evol", "nickrosh/Evol-Instruct-Code-80k-v1", None, a_evol_code, 0.12),
    ("inst:smoltalk", "HuggingFaceTB/smoltalk", "all", a_smoltalk, 0.25),
    ("cli:nl2bash", "AnishJoshi/nl2bash-custom", None, a_nl2bash, 0.15),
    ("tool:toolace", "Team-ACE/ToolACE", None, a_toolace, 0.08),
    # glaive-function-calling-v2 is NOT used: its shards need a contiguous memory
    # block pandas cannot reserve on a 16 GB box that is also training, and it
    # dies mid-iteration ("Could not reserve memory block") after ~40 rows. The
    # a_glaive adapter is kept and verified (40/40 rows parsed) so the source can
    # be restored on a machine with more RAM. Its 7% moved to Hermes, which
    # streams reliably and is the same task.
    ("tool:hermes-sgl", "NousResearch/hermes-function-calling-v1",
     "func_calling_singleturn", a_hermes, 0.04),
    ("tool:hermes-fc", "NousResearch/hermes-function-calling-v1",
     "func_calling", a_hermes, 0.03),
    ("sec:dpo", "CyberNative/Code_Vulnerability_Security_DPO", None, a_sec, 0.05),
]
REFUSAL_SHARE = 0.05
IDENTITY_SHARE = 0.02
PLAN_SHARE = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=25000)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--out", default="data_sft")
    ap.add_argument("--tokenizer", default="tokenizer/kestrel-bpe.json")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--long-frac", type=float, default=0.25,
                    help="reserve this fraction of each source's quota for examples "
                         ">=--long-min tokens, so the set actually exercises a 4k "
                         "context and supports multi-turn / long-form answers")
    ap.add_argument("--long-min", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    from datasets import load_dataset

    os.makedirs(args.out, exist_ok=True)
    tok = Tokenizer.from_file(args.tokenizer)
    eot = tok.token_to_id("<|endoftext|>")
    rng = random.Random(args.seed)
    t0 = time.time()

    collected: dict[str, list] = {}
    code_cli_pool: list = []

    for key, name, cfg, adapt, share in SOURCES:
        want = int(args.examples * share)
        want_long = int(want * args.long_frac)
        # Two buckets. Without this the set is dominated by short single-turn rows
        # (median ~300 tokens in a probe run), which trains a 4k model that has
        # never seen a long prompt and cannot hold a 4+ turn conversation.
        got, longs, seen = [], [], 0
        print(f"[{key}] target {want} ({want_long} long >= {args.long_min} tok) "
              f"from {name} ...", flush=True)
        # Opening a stream can fail transiently under memory pressure. On a 16 GB
        # box that is also training, glaive died once with "Could not reserve memory
        # block" and produced ZERO examples with an empty error string. Silently
        # losing a whole source is the worst outcome here, so retry then SHOUT.
        ds = None
        for attempt in range(3):
            try:
                ds = (load_dataset(name, cfg, split='train', streaming=True) if cfg
                      else load_dataset(name, split='train', streaming=True))
                break
            except Exception as e:
                print(f'[{key}] open attempt {attempt+1}/3 failed: '
                      f'{type(e).__name__}: {str(e)[:80]}', flush=True)
                time.sleep(5 * (attempt + 1))
        if ds is None:
            print(f'[{key}] *** SOURCE UNAVAILABLE - slice will be EMPTY ***', flush=True)
            collected[key] = []
            continue
        try:
            for row in ds:
                seen += 1
                if seen > want * 40 + 2000:
                    break
                turns = adapt(row)
                if not turns or len(turns) < 2:
                    continue
                enc = encode_example(tok, turns, args.seq, eot)
                if enc is None:
                    continue
                is_long = enc[0].size >= args.long_min
                kept = False
                if is_long and len(longs) < want_long:
                    longs.append(enc); kept = True
                elif len(got) < want - want_long:
                    got.append(enc); kept = True
                # keep raw turns ONLY for the plan slice, and only as many as it
                # needs - this pool used to grow unbounded across every source
                if kept and key.startswith(("code", "cli"))                         and len(code_cli_pool) < args.examples * PLAN_SHARE * 6:
                    code_cli_pool.append(turns)
                if len(got) >= want - want_long and len(longs) >= want_long:
                    break
        except Exception as e:
            print(f"[{key}] iteration stopped early: {type(e).__name__}: "
                  f"{str(e)[:100]}", flush=True)
        collected[key] = got + longs
        print(f"[{key}] kept {len(got)+len(longs)} ({len(longs)} long) "
              f"of {seen} rows seen ({time.time()-t0:.0f}s elapsed)", flush=True)

    # constructed slices
    n_id = int(args.examples * IDENTITY_SHARE)
    idt = [encode_example(tok, t, args.seq, eot) for t in build_identity(n_id, rng)]
    collected["identity"] = [e for e in idt if e]
    print(f"[identity] built {len(collected['identity'])}")

    n_ref = int(args.examples * REFUSAL_SHARE)
    ref = [encode_example(tok, t, args.seq, eot) for t in build_refusals(n_ref, rng)]
    collected["refusal"] = [e for e in ref if e]
    print(f"[refusal] built {len(collected['refusal'])}")

    n_plan = int(args.examples * PLAN_SHARE)
    rng.shuffle(code_cli_pool)
    pl = [encode_example(tok, t, args.seq, eot)
          for t in build_plans(code_cli_pool, n_plan, rng)]
    collected["plan"] = [e for e in pl if e]
    print(f"[plan] built {len(collected['plan'])}")

    # --- pack ---------------------------------------------------------------
    allex = [(k, e) for k, v in collected.items() for e in v]
    rng.shuffle(allex)
    n_val = max(1, int(len(allex) * args.val_frac))
    splits = {"val": allex[:n_val], "train": allex[n_val:]}

    manifest = {"examples": {}, "tokens": {}, "seq": args.seq,
                "template": "System:/User:/Assistant:/Tool: joined by blank lines, "
                            "terminated with <|endoftext|>",
                "mask": "uint8 parallel to tokens; 1 = compute loss (Assistant only)"}
    for split, rows in splits.items():
        if not rows:
            continue
        a = np.concatenate([e[0] for _, e in rows])
        b = np.concatenate([e[1] for _, e in rows])
        a.tofile(os.path.join(args.out, f"{split}_tokens.bin"))
        b.tofile(os.path.join(args.out, f"{split}_mask.bin"))
        lens = np.diff(np.concatenate([[-1], np.nonzero(a == eot)[0]]))
        manifest["examples"][split] = len(rows)
        manifest["tokens"][split] = int(a.size)
        manifest.setdefault("length", {})[split] = {
            "median": int(np.median(lens)), "p90": int(np.percentile(lens, 90)),
            "max": int(lens.max()),
            "pct_ge_1024": round(100 * float((lens >= 1024).mean()), 1),
            "pct_ge_2048": round(100 * float((lens >= 2048).mean()), 1)}
        print(f"[{split}] {len(rows)} examples, {a.size/1e6:.2f}M tokens, "
              f"{100*b.mean():.1f}% loss targets | median {np.median(lens):.0f} tok, "
              f"{100*(lens>=1024).mean():.0f}% >=1024, {100*(lens>=2048).mean():.0f}% >=2048")

    targets = {k: int(args.examples * sh) for k, _, _, _, sh in SOURCES}
    targets["refusal"] = int(args.examples * REFUSAL_SHARE)
    targets["identity"] = int(args.examples * IDENTITY_SHARE)
    targets["plan"] = int(args.examples * PLAN_SHARE)
    manifest["per_source"] = {k: len(v) for k, v in collected.items()}
    manifest["shortfall"] = {k: targets[k] - len(collected.get(k, []))
                             for k in targets if len(collected.get(k, [])) < targets[k]}
    if manifest["shortfall"]:
        print(chr(10) + "SHORTFALLS (source exhausted or unavailable):")
        for k, n in manifest["shortfall"].items():
            print(f"  {k:18s} short by {n} of {targets[k]}")
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\ndone in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
