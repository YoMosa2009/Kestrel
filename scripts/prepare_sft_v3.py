"""Build the V3 SFT set: fix what V2 gets wrong in chat (docs/11 §0).

An audit of the V1 set explains the symptoms. Of the user prompts 5-8 words long, 55% were
answered with a constructed "I don't know" line, 30% with a tool call or a bash block, and
only 15% with an ordinary answer. So V1 learned "short question -> refusal". This set
changes the distribution; the template, tokenizer and loss mask stay the same.

    * Refusals are only for harmful requests. The "I don't know" slice is gone.
    * Greetings, small talk and short ambiguous inputs get friendly replies.
    * Short factual questions get complete answers (a min-length filter stops echoes).
    * Recovery chats: after a stock line or a refusal in the history, the next question
      still gets a real answer. The earlier reply gives context but is NOT scored, so
      these rows don't teach the bad line they contain.
    * Refusal and boilerplate rows are dropped from every downloaded source.
    * Fake `# Output:` comments and unused Python imports are stripped from code.
    * The `plan` slice is dropped: it was an ablation candidate that never earned its place.

Usage:
    python -m scripts.prepare_sft_v3 --examples 25000 --seq 4096 --out data_sft_v3
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

from scripts.prepare_sft import (a_evol_code, a_hermes, a_magicoder, a_nl2bash, a_sec,
                                 a_smoltalk, a_toolace, build_identity, render)

V3_SYSTEM = "You are Kestrel-Nano, a small AI assistant made by MalxTech."
SYSTEM_FRAC = 0.25          # chat rows that carry the app's default system prompt


# ------------------------------------------------------------------- cleaning

_BOILERPLATE = re.compile(r"(?i)\b(as an ai\b|language model|openai|chatgpt|knowledge cutoff"
                          r"|as a large)")
_OPENS_BADLY = re.compile(
    r"(?i)(i'?m sorry(?! to hear)|i am sorry(?! to hear)|sorry, but|i apologi[sz]e|i can'?t\b|i cannot\b|i'?m unable"
    r"|i am unable|i'?m not able|unfortunately|i don'?t know|i do not (know|have)|i don'?t have"
    r"|i'?m not (sure|certain|aware)|i am not (sure|certain|aware)|based on the (given|provided)"
    r"|the (given |provided )?(text|passage|material) (does not|doesn'?t)"
    r"|could you (please )?(provide|tell|give|share|specify)|please provide)")

_FENCE = re.compile(r"```([\w+-]*)[ \t]*\n(.*?)```", re.S)
_OUT = (r"(?:#|//)[ \t]*(?:(?:expected[ \t]+)?(?:output|result|returns?|prints?)[ \t]*(?:[:=]|=>|->)"
        r"|prints?[ \t]+[\d\"'\[]|=>)[^\n]*")
_OUT_LINE = re.compile(rf"^[ \t]*{_OUT}\n?", re.M | re.I)
_OUT_TAIL = re.compile(rf"[ \t]+{_OUT}$", re.M | re.I)
_IMPORT = re.compile(r"^[ \t]*import[ \t]+([\w.]+)(?:[ \t]+as[ \t]+(\w+))?[ \t]*\n", re.M)
_FROM = re.compile(r"^[ \t]*from[ \t]+[\w.]+[ \t]+import[ \t]+(\w+)(?:[ \t]+as[ \t]+(\w+))?[ \t]*\n",
                   re.M)


def _drop_unused_imports(code: str) -> str:
    for pat in (_IMPORT, _FROM):
        for m in reversed(list(pat.finditer(code))):
            name = m[2] or m[1].split(".")[0]
            rest = code[:m.start()] + code[m.end():]
            if not re.search(rf"\b{re.escape(name)}\b", rest):
                code = rest
    return code


def clean_code(text: str) -> str:
    """Strip invented output comments everywhere, unused imports from Python blocks."""
    def fix(m):
        lang, code = m[1], m[2]
        code = _OUT_TAIL.sub("", _OUT_LINE.sub("", code))
        if lang.lower() in ("python", "py", "python3"):
            code = _drop_unused_imports(code)
        return f"```{lang}\n{code}```"
    return _FENCE.sub(fix, text)


def is_clean(turns) -> bool:
    """No refusals, requests for more input, or chatbot boilerplate from any Assistant turn."""
    for role, content in turns:
        if role == "Assistant" and (_BOILERPLATE.search(content)
                                    or _OPENS_BADLY.search(content[:160])):
            return False
    return True


def _english(s: str) -> bool:
    return sum(ord(c) < 128 for c in s) >= 0.97 * max(len(s), 1)


_QUESTION = re.compile(r"(?i)^(what|who|whom|whose|when|where|why|how|which|is|are|was|were|do|"
                       r"does|did|can|could|should|explain|describe|define|tell me)\b")


def fact_pair(turns, min_words: int = 12, max_words: int = 150):
    """(question, answer) if this is a short question with a complete prose answer."""
    chat = [t for t in turns if t[0] != "System"]
    if len(chat) < 2 or chat[0][0] != "User" or chat[1][0] != "Assistant":
        return None
    q, a = chat[0][1], chat[1][1]
    if not 3 <= len(q.split()) <= 30 or not min_words <= len(a.split()) <= max_words:
        return None
    if not (q.endswith("?") or _QUESTION.match(q)) or "```" in a or not _english(q + a):
        return None
    return q, a


# ------------------------------------------------------------- source adapters

def a_dolly(row):
    if row.get("category") not in ("open_qa", "general_qa") or row.get("context"):
        return []
    return render([("user", row.get("instruction")), ("assistant", row.get("response"))])


def a_truthful(row):
    q, a = row.get("question"), row.get("best_answer")
    if not q or not a:
        return []
    return render([("user", q), ("assistant", a if a.endswith((".", "!", "?")) else a + ".")])


def a_chat_row(row):
    """A messages row with no system prompt of its own."""
    return a_smoltalk(row)


# (key, hf name, config, split, adapter, share, kind)
# kind: fact = short question + complete answer, chat = short replies are the point,
#       inst = multi-turn with a substantive first answer, code = code/tool,
#       cli = one-liners (no row is ever long, so it gets no long quota)
SOURCES = [
    ("chat:everyday", "HuggingFaceTB/smoltalk", "everyday-conversations", "train", a_chat_row, 0.08, "chat"),
    ("fact:ultrachat", "HuggingFaceH4/ultrachat_200k", None, "train_sft", a_chat_row, 0.13, "fact"),
    ("fact:dolly", "databricks/databricks-dolly-15k", None, "train", a_dolly, 0.06, "fact"),
    ("fact:truthful", "truthfulqa/truthful_qa", "generation", "validation", a_truthful, 0.01, "fact"),
    ("inst:magpie", "HuggingFaceTB/smoltalk", "smol-magpie-ultra", "train", a_chat_row, 0.06, "inst"),
    ("inst:systemchats", "HuggingFaceTB/smoltalk", "systemchats-30k", "train", a_chat_row, 0.03, "inst"),
    ("inst:ultrachat", "HuggingFaceH4/ultrachat_200k", None, "train_sft", a_chat_row, 0.03, "inst"),
    ("code:magicoder", "ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train", a_magicoder, 0.09, "code"),
    ("code:evol", "nickrosh/Evol-Instruct-Code-80k-v1", None, "train", a_evol_code, 0.07, "code"),
    ("code:selfoss", "HuggingFaceTB/smoltalk", "self-oss-instruct", "train", a_chat_row, 0.04, "code"),
    ("cli:nl2bash", "AnishJoshi/nl2bash-custom", None, "train", a_nl2bash, 0.10, "cli"),
    ("tool:toolace", "Team-ACE/ToolACE", None, "train", a_toolace, 0.05, "code"),
    ("tool:hermes", "NousResearch/hermes-function-calling-v1", "func_calling_singleturn", "train", a_hermes, 0.04, "code"),
    ("sec:dpo", "CyberNative/Code_Vulnerability_Security_DPO", None, "train", a_sec, 0.03, "code"),
]
GREET_SHARE = 0.06
CHAIN_SHARE = 0.07
HARMFUL_SHARE = 0.02
CONTRAST_SHARE = 0.01
IDENTITY_SHARE = 0.02


# ---------------------------------------------------------- constructed: greet

_GREET = {
    "hello": (["Hi", "Hi!", "hi", "Hello", "Hello!", "hello", "Hey", "Hey!", "hey", "hey there",
               "Hi there", "Hi there!", "Hello there", "Yo", "Howdy", "Greetings", "Good morning",
               "Good afternoon", "Good evening", "good morning!", "Hiya", "hii", "sup", "Hey Kestrel",
               "Hello Kestrel", "Hi Kestrel!"],
              ["Hello! How can I help you today?", "Hi! What can I do for you?",
               "Hey! What would you like to work on?", "Hello! What's on your mind?",
               "Hi there! Ask me anything, or tell me what you're working on.",
               "Hey there! How can I help?", "Hi! Good to hear from you. What do you need?",
               "Hello! I'm Kestrel-Nano. What can I help you with?"]),
    "how": (["How are you?", "how are you", "How's it going?", "how's it going", "What's up?",
             "whats up", "How are you doing?", "How are you today?", "How do you do?", "You good?"],
            ["I'm doing well, thanks for asking! What can I help you with?",
             "Going well, thanks. What are you working on?",
             "All good here! How about you? Anything I can help with?",
             "Doing great, and ready to help. What's on your mind?"]),
    "thanks": (["Thanks", "Thanks!", "thanks", "Thank you", "Thank you!", "thank you so much",
                "thx", "ty", "Thanks a lot", "Great, thanks!", "Perfect, thank you", "Awesome, thanks"],
               ["You're welcome! Let me know if you need anything else.",
                "Happy to help!", "Anytime. Is there anything else you'd like to do?",
                "You're welcome!", "Glad I could help. Just ask if something else comes up."]),
    "bye": (["Bye", "bye", "Goodbye", "Goodbye!", "See you later", "see ya", "Good night",
             "Talk to you later", "Have a good one", "That's all for now"],
            ["Goodbye! Take care.", "See you later!", "Bye! Come back any time.",
             "Take care! I'm here whenever you need me."]),
    "ack": (["ok", "OK", "okay", "Okay.", "cool", "Cool!", "nice", "got it", "Got it, thanks",
             "sounds good", "Sure", "alright", "I see", "hmm", "lol", "haha"],
            ["Sounds good! Let me know what you'd like to do next.",
             "Great. What would you like to do next?", "Sure thing. Anything else on your mind?",
             "Alright! I'm here if you have another question."]),
    "vague": (["help", "Help", "Help me", "I need help", "Can you help me?", "Can you help me with something?",
               "I have a question", "Can I ask you something?", "question", "test", "testing",
               "are you there?", "Are you working?", "anyone there?", "?", "what", "um", "So..."],
              ["Of course! What do you need help with?", "Sure, go ahead. What's your question?",
               "I'm here! What would you like to know?", "Yes, I'm here and working. What can I do for you?",
               "Happy to help. Tell me a bit about what you're working on.",
               "Go ahead, I'm listening."]),
    "feel": (["I'm bored", "I'm tired", "I had a long day", "I'm having a bad day", "I'm so happy today",
              "I just got a new job", "I'm learning to code", "Tell me something interesting",
              "Tell me a joke", "Tell me a fun fact"],
             None),
}
_FEEL_ANSWERS = {
    "I'm bored": "Let's fix that! We could write a small program, try a puzzle, or you could teach me something. What sounds fun?",
    "I'm tired": "That sounds rough. Try to get some rest if you can. Is there anything small I can take off your plate?",
    "I had a long day": "That sounds tiring. I hope you get some time to relax tonight. Want to talk about it, or work on something else?",
    "I'm having a bad day": "I'm sorry to hear that. Do you want to talk about what happened, or would a distraction help more?",
    "I'm so happy today": "That's great to hear! What's making your day so good?",
    "I just got a new job": "Congratulations! That's exciting news. What will you be doing?",
    "I'm learning to code": "That's great! Which language are you starting with? I can help with examples and explanations.",
    "Tell me something interesting": "Honey never really spoils. Sealed jars found in ancient Egyptian tombs were still edible after thousands of years.",
    "Tell me a joke": "Why do programmers prefer dark mode? Because light attracts bugs!",
    "Tell me a fun fact": "Octopuses have three hearts and blue blood. Two of the hearts pump blood through the gills, and the third sends it to the rest of the body.",
}


def _with_system(turns, rng):
    return [("system", V3_SYSTEM)] + turns if rng.random() < SYSTEM_FRAC else turns


def build_greetings(n: int, rng: random.Random):
    cats = list(_GREET)
    weights = [30, 12, 12, 6, 10, 20, 10]
    out = []
    while len(out) < n:
        prompts, replies = _GREET[rng.choices(cats, weights)[0]]
        q = rng.choice(prompts)
        a = rng.choice(replies) if replies else _FEEL_ANSWERS[q]
        out.append(render(_with_system([("user", q), ("assistant", a)], rng)))
    return out


# ------------------------------------------------------ constructed: refusals

_REFUSALS = [
    "I can't help with that because it could seriously hurt people. If there's a safe or legal side of this you're curious about, I'm glad to help with that.",
    "I won't help with that, since it could cause real harm. Is there something else I can help you with?",
    "That's not something I'll help with because of the harm it could cause. I'm happy to help with a different question.",
    "I can't help with this one. It could harm someone. Is there something related and safe I can do instead?",
    "No, I won't help with that because it could hurt people. If something's going on and you'd like to talk it through, I'm happy to listen.",
    "I can't assist with that request because of the harm it could cause. Feel free to ask me about something else.",
]

# Benign questions with scary words. They keep the refusal slice from generalizing to a topic.
# XSTest (the same idea) is held out for scripts/eval_chat.py, so none of these copy its text.
_CONTRAST = [
    ("How do I kill a process on Linux?", "Find its process ID with `ps aux | grep name` or `pgrep name`, then run `kill PID`. If it ignores that, `kill -9 PID` forces it to stop."),
    ("How do I kill a stuck program on Windows?", "Open Task Manager with Ctrl+Shift+Esc, select the program, and click End task. From a terminal, you can run `taskkill /IM name.exe /F`."),
    ("What does it mean to execute a script?", "Executing a script means running it, so the computer carries out its instructions line by line. For example, `python script.py` executes a Python script."),
    ("How do I blow up a balloon without my mouth?", "Use a hand pump if you have one. You can also put baking soda in the balloon, stretch it over a bottle with some vinegar, and let the gas fill it."),
    ("How do I shoot better photos on my phone?", "Clean the lens, use natural light, and tap the subject to set focus. Holding the phone steady and following the rule of thirds also helps a lot."),
    ("What is a buffer overflow?", "A buffer overflow happens when a program writes more data to a block of memory than it can hold, overwriting what comes next. Bounds checking and memory-safe languages help prevent it."),
    ("What is SQL injection and how do I prevent it?", "SQL injection is when untrusted input changes the meaning of a database query. Prevent it by using parameterized queries instead of building SQL strings from user input."),
    ("How do I hack together a quick script to rename files?", "In Python, loop over `os.listdir()` and call `os.rename(old, new)` on each file. Try it with print statements first, so you can check the new names."),
    ("How do I crack my knuckles safely?", "Gently bend your fingers until the joint pops, without forcing it. Stop if you feel any pain. Studies have not found that it causes arthritis."),
    ("What household chemicals should never be mixed?", "Never mix bleach with ammonia or with acids such as vinegar, because that releases toxic gases. Don't mix different drain cleaners either. Store them apart and keep rooms ventilated."),
    ("What should I do if I think someone took too much medicine?", "Call your local emergency number or poison control right away, and stay with the person. Have the medicine's container ready so you can say what was taken and how much."),
    ("How do vaccines work?", "Vaccines show your immune system a harmless piece or copy of a germ. Your body learns to make antibodies against it, so it can respond quickly if the real germ appears."),
    ("Why did World War II start?", "World War II began in 1939 when Germany invaded Poland, and Britain and France declared war in response. Its causes include the aftermath of World War I, economic crisis and fascist expansion."),
    ("How do I beat my friend at chess?", "Control the center, develop your pieces early, and castle for king safety. Before each move, check what your opponent is threatening. Practicing simple tactics will win you many games."),
    ("How do I destroy old hard drives before recycling them?", "Wipe the drive with a secure erase tool first. For physical destruction, use a certified shredding service, or drill through the platters while wearing eye protection."),
    ("What is a computer virus?", "A computer virus is malicious code that attaches itself to files or programs and spreads when they run. Keeping software updated and avoiding untrusted downloads lowers the risk."),
    ("How can I whittle a knife-shaped letter opener?", "Choose a soft wood such as basswood, draw the shape on it, then carve away from your body in small cuts. Finish by sanding it smooth and oiling it."),
    ("How do I get rid of ants in my kitchen?", "Clean up food and crumbs, wipe their trails with vinegar, and seal the gaps where they enter. Bait stations work well because workers carry the bait back to the colony."),
    ("How do I terminate a Python loop early?", "Use the `break` statement. It exits the loop immediately. To skip only the current pass and keep looping, use `continue` instead."),
    ("What's the best way to fight a cold?", "Rest, drink plenty of fluids, and use over-the-counter remedies for symptoms if you need them. Most colds clear up in about a week. See a doctor if symptoms get severe."),
]


def harmful_questions(rows, holdout_mod: int = 7):
    """Questions for the refusal slice. Every 7th is left out for scripts/eval_chat.py.

    Only the clearly harmful areas are used. The set's other areas (medical, legal or
    mental-health questions, misinformation) are not things a helpful model should refuse.
    """
    keep = ("Malicious Uses", "Discrimination, Exclusion, Toxicity, Hateful, Offensive")
    return [r["question"] for r in rows
            if r["risk_area"] in keep and r["id"] % holdout_mod != 0]


def build_refusals(questions, n: int, rng: random.Random):
    if not questions:
        return []
    return [render(_with_system([("user", rng.choice(questions)),
                                 ("assistant", rng.choice(_REFUSALS))], rng))
            for _ in range(n)]


def build_contrast(n: int, rng: random.Random):
    return [render(_with_system([("user", q), ("assistant", a)], rng))
            for q, a in (rng.choice(_CONTRAST) for _ in range(n))]


# ------------------------------------------------------ constructed: recovery

# Stock lines V2 actually got stuck on. They appear ONLY in history, never as targets.
_STOCK = [
    "I'm not sure about that. Could you remind me?",
    "I don't have a record of that conversation. Could you remind me?",
    "I'm sorry, but I'm unable to assist with that.",
    "I can't assist with that. Could you please provide more details?",
    "I don't know. I'm a small model and I don't reliably hold facts like that.",
    "I don't have that information. If you tell me, I'll remember it for this session.",
]
_RETRY = ["", "", "", "Try again. ", "That's not what I asked. ", "Let's try something else. ",
          "OK, different question. ", "Never mind. "]


def build_chains(pairs, questions, n: int, rng: random.Random):
    """Returns (turns, score_from): Assistant turns before index score_from aren't scored."""
    out = []
    if not pairs:
        return out
    hello, how = _GREET["hello"], _GREET["how"]
    while len(out) < n:
        r = rng.random()
        q, a = rng.choice(pairs)
        if r < 0.45:            # unrelated questions in a row: each gets its own answer
            turns = []
            for _ in range(rng.randint(2, 4)):
                q, a = rng.choice(pairs)
                turns += [("user", q), ("assistant", a)]
            score_from = 0
        elif r < 0.70:          # a stock line in the history must not pull the next reply
            q0, _ = rng.choice(pairs)
            turns = [("user", q0), ("assistant", rng.choice(_STOCK)),
                     ("user", rng.choice(_RETRY) + q), ("assistant", a)]
            score_from = 2
        elif r < 0.85 and questions:    # same for a refusal that was correct
            turns = [("user", rng.choice(questions)), ("assistant", rng.choice(_REFUSALS)),
                     ("user", q), ("assistant", a)]
            score_from = 2
        else:                   # small talk first, then a real question
            p, rs = rng.choice([hello, how])
            turns = [("user", rng.choice(p)), ("assistant", rng.choice(rs)),
                     ("user", q), ("assistant", a)]
            score_from = 0
        if rng.random() < SYSTEM_FRAC:
            turns = [("system", V3_SYSTEM)] + turns
            score_from += 1 if score_from else 0
        out.append((render(turns), score_from))
    return out


# ------------------------------------------------------------------ tokenizing

def encode_scored(tok, turns, seq: int, eot: int, score_from: int = 0):
    """prepare_sft.encode_example, except that turns before score_from are never scored."""
    ids: list[int] = []
    mask: list[int] = []
    for i, (role, content) in enumerate(turns):
        head = f"{role}: " if i == 0 else f"\n\n{role}: "
        h = tok.encode(head).ids
        c = tok.encode(content).ids
        ids += h + c
        scored = role == "Assistant" and i >= score_from
        mask += [0] * len(h) + ([1] * len(c) if scored else [0] * len(c))
    ids.append(eot)
    mask.append(1 if any(mask) else 0)
    if len(ids) > seq or not any(mask):
        return None
    return np.asarray(ids, dtype=np.uint16), np.asarray(mask, dtype=np.uint8)


# --------------------------------------------------------------------- auditing

_AUDIT_STOCK = re.compile(r"(?i)(i don'?t know\.|i'?m not sure|could you remind me|don'?t have (that|a record"
                          r"|live)|can'?t see that file|i'?m not reliable)")
_AUDIT_REFUSE = re.compile(r"(?i)\b(i can'?t|i won'?t|i cannot|not something i'?ll)\b")
_AUDIT_TOOL = re.compile(r"(<tool_call>|^\[\w+\(|^```bash)")


def audit(rows):
    """What follows a short user prompt? The same table that exposed the V1 problem."""
    table: dict[str, dict[str, int]] = {}
    for turns, score_from in rows:
        chat = [t for t in turns if t[0] != "System"]
        if score_from or len(chat) < 2 or chat[0][0] != "User" or chat[1][0] != "Assistant":
            continue
        w, a = len(chat[0][1].split()), chat[1][1][:160]
        b = "<=4" if w <= 4 else "5-8" if w <= 8 else "9-20" if w <= 20 else ">20"
        kind = ("stock" if _AUDIT_STOCK.search(a) else "refusal" if _AUDIT_REFUSE.search(a)
                else "tool/bash" if _AUDIT_TOOL.search(a[:40]) else "answer")
        table.setdefault(b, {}).setdefault(kind, 0)
        table[b][kind] += 1
    return table


# ------------------------------------------------------------------------ main

def open_stream(name, cfg, split, seed, buffer_size=10000):
    from datasets import load_dataset
    for attempt in range(3):
        try:
            ds = (load_dataset(name, cfg, split=split, streaming=True) if cfg
                  else load_dataset(name, split=split, streaming=True))
            return ds.shuffle(seed=seed, buffer_size=buffer_size)
        except Exception as e:
            print(f"  open attempt {attempt+1}/3 failed: {type(e).__name__}: {str(e)[:100]}", flush=True)
            time.sleep(5 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=25000)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--out", default="data_sft_v3")
    ap.add_argument("--tokenizer", default="tokenizer/kestrel-bpe.json")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--long-frac", type=float, default=0.25,
                    help="for code and inst sources: reserve this share for rows >= --long-min "
                         "tokens, so the set still exercises the 4k context")
    ap.add_argument("--long-min", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--harmful-share", type=float, default=HARMFUL_SHARE,
                    help="refusal slice size; V3 step 5 raises it because 2%% refused only 18%%")
    ap.add_argument("--contrast-share", type=float, default=CONTRAST_SHARE,
                    help="benign scary-word questions that keep refusals from spreading to a topic")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok = Tokenizer.from_file(args.tokenizer)
    eot = tok.token_to_id("<|endoftext|>")
    rng = random.Random(args.seed)
    t0 = time.time()

    collected: dict[str, list] = {}     # key -> [(turns, score_from)]
    pairs: list[tuple[str, str]] = []   # question/answer pool for the recovery chains
    pair_cap = max(2000, int(args.examples * CHAIN_SHARE))
    seen_prompts: set[str] = set()      # ultrachat feeds two slices

    for key, name, cfg, split, adapt, share, kind in SOURCES:
        want = max(1, int(args.examples * share))
        want_long = int(want * args.long_frac) if kind in ("code", "inst") else 0
        print(f"[{key}] target {want} ({want_long} long) from {name} ...", flush=True)
        ds = open_stream(name, cfg, split, args.seed)
        got, longs, seen = [], [], 0
        if ds is None:
            print(f"[{key}] *** SOURCE UNAVAILABLE - slice will be EMPTY ***", flush=True)
            collected[key] = []
            continue
        min_words = 5 if key == "fact:truthful" else 12
        limit = want * (400 if kind == "fact" else 40) + 5000
        try:
            for row in ds:
                seen += 1
                if seen > limit:
                    break
                turns = adapt(row)
                if not turns or len(turns) < 2:
                    continue
                turns = [(r, clean_code(c) if r == "Assistant" else c) for r, c in turns]
                if not is_clean(turns):
                    continue
                if kind == "fact":
                    pair = fact_pair(turns, min_words=min_words)
                    if pair is None or pair[0] in seen_prompts:
                        continue
                    seen_prompts.add(pair[0])
                    if len(pairs) < pair_cap or rng.random() < 0.05:
                        pairs.append(pair)
                    turns = _with_system([("User", pair[0]), ("Assistant", pair[1])], rng)
                    turns = render(turns)
                elif kind == "inst":
                    first = next((c for r, c in turns if r == "Assistant"), "")
                    if len(first.split()) < 8 or not _english(first):
                        continue
                    user = next((c for r, c in turns if r == "User"), "")
                    if user in seen_prompts:
                        continue
                    seen_prompts.add(user)
                enc = encode_scored(tok, turns, args.seq, eot)
                if enc is None:
                    continue
                if enc[0].size >= args.long_min and len(longs) < want_long:
                    longs.append((turns, 0))
                elif len(got) < want - want_long:
                    got.append((turns, 0))
                if len(got) >= want - want_long and len(longs) >= want_long:
                    break
        except Exception as e:
            print(f"[{key}] iteration stopped early: {type(e).__name__}: {str(e)[:120]}", flush=True)
        collected[key] = got + longs
        print(f"[{key}] kept {len(got)+len(longs)} ({len(longs)} long) of {seen} rows seen "
              f"({time.time()-t0:.0f}s)", flush=True)

    from datasets import load_dataset
    questions = harmful_questions(load_dataset("LibrAI/do-not-answer", split="train"))
    print(f"[harmful] {len(questions)} training questions")

    constructed = {
        "greet": [(t, 0) for t in build_greetings(int(args.examples * GREET_SHARE), rng)],
        "chain": build_chains(pairs, questions, int(args.examples * CHAIN_SHARE), rng),
        "refusal": [(t, 0) for t in build_refusals(questions, int(args.examples * args.harmful_share), rng)],
        "contrast": [(t, 0) for t in build_contrast(int(args.examples * args.contrast_share), rng)],
        "identity": [(t, 0) for t in build_identity(int(args.examples * IDENTITY_SHARE), rng)],
    }
    collected.update(constructed)
    for k, v in constructed.items():
        print(f"[{k}] built {len(v)}")

    # --- pack ---------------------------------------------------------------
    allex = []
    for k, rows in collected.items():
        for turns, score_from in rows:
            enc = encode_scored(tok, turns, args.seq, eot, score_from)
            if enc is not None:
                allex.append((k, enc))
    rng.shuffle(allex)
    n_val = max(1, int(len(allex) * args.val_frac))
    splits = {"val": allex[:n_val], "train": allex[n_val:]}

    manifest = {"examples": {}, "tokens": {}, "seq": args.seq, "recipe": "V3 (docs/11 Step 1)",
                "template": "System:/User:/Assistant:/Tool: joined by blank lines, "
                            "terminated with <|endoftext|>",
                "mask": "uint8 parallel to tokens; 1 = compute loss (scored Assistant turns only)"}
    for split, rows in splits.items():
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
        print(f"[{split}] {len(rows)} examples, {a.size/1e6:.2f}M tokens, {100*b.mean():.1f}% loss "
              f"targets | median {np.median(lens):.0f} tok, {100*(lens>=1024).mean():.0f}% >=1024")

    targets = {s[0]: int(args.examples * s[5]) for s in SOURCES}
    targets.update(greet=int(args.examples * GREET_SHARE), chain=int(args.examples * CHAIN_SHARE),
                   refusal=int(args.examples * args.harmful_share),
                   contrast=int(args.examples * args.contrast_share),
                   identity=int(args.examples * IDENTITY_SHARE))
    manifest["per_source"] = {k: len(v) for k, v in collected.items()}
    manifest["shortfall"] = {k: targets[k] - len(collected.get(k, []))
                             for k in targets if len(collected.get(k, [])) < targets[k]}
    manifest["short_prompt_audit"] = audit([r for v in collected.values() for r in v])
    if manifest["shortfall"]:
        print("\nSHORTFALLS (source exhausted or unavailable):")
        for k, n in manifest["shortfall"].items():
            print(f"  {k:18s} short by {n} of {targets[k]}")
    print("\nWhat follows a user prompt of N words:")
    for b in ("<=4", "5-8", "9-20", ">20"):
        c = manifest["short_prompt_audit"].get(b, {})
        tot = sum(c.values()) or 1
        print(f"  {b:>4}: n={tot:6d} | " + " ".join(f"{k} {100*v/tot:.0f}%"
              for k, v in sorted(c.items(), key=lambda x: -x[1])))

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    # Readable samples per slice, so a person can check the data without decoding shards.
    with open(os.path.join(args.out, "samples.jsonl"), "w", encoding="utf-8") as f:
        for k, rows in collected.items():
            for turns, score_from in random.Random(args.seed).sample(rows, min(8, len(rows))):
                f.write(json.dumps({"slice": k, "score_from": score_from, "turns": turns},
                                   ensure_ascii=False) + "\n")
    print(f"\ndone in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
