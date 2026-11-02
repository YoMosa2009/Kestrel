"""Checks for step 4's trainer flags and data rules. Tiny random model; runs on a Colab
runtime before any data is built or A100 time is spent (docs/11).

    python -m scripts.test_step4
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

from kestrel.data import CurriculumLoader, discover_domains
from kestrel.train import parse_mix, wsd_lr_mult
from scripts.prepare_quality import SOURCES, clean, unseen_files


def main():
    # data rules: only the unseen half of a source, and text the tokenizer can take
    names = [f"a/train-{i:05d}.parquet" for i in range(10)] + ["b/x.parquet", "a/README.md"]
    assert unseen_files(names, "a/") == [f"a/train-{i:05d}.parquet" for i in range(5, 10)]
    assert clean("ok \udc9d done") == "ok  done"
    assert len({s["tag"] for s in SOURCES}) == len(SOURCES)
    assert all(s["tag"].startswith("q") for s in SOURCES), "q tags keep new domains out of the old val suite"

    # --mix parses, and rejects a tag with no shards
    assert parse_mix("a=3,b=1.5") == {"a": 3.0, "b": 1.5}

    with tempfile.TemporaryDirectory() as d:
        data = os.path.join(d, "data")
        os.makedirs(data)
        rng = np.random.default_rng(0)
        for name in ("train_old_000", "train_qnew_000", "train_unused_000", "val_old"):
            rng.integers(1, 200, 5000, dtype=np.uint16).tofile(os.path.join(data, name + ".bin"))
        assert set(discover_domains(data)) == {"old", "qnew", "unused"}

        import torch
        loader = CurriculumLoader(data, 16, 8, torch.device("cpu"), stages=[(1.01, parse_mix("old=3,qnew=1"))])
        loader.set_progress(0.5)
        assert loader.stage_weights == {"old": 0.75, "qnew": 0.25}, loader.stage_weights

        # init, stop halfway, resume: the LR must follow one schedule across the interruption
        out = os.path.join(d, "run")
        os.makedirs(out)
        src = os.path.join(d, "src.pt")
        build = (f"import torch; from dataclasses import asdict; from kestrel.config import PRESETS; "
                 f"from kestrel.model import KestrelModel; c=PRESETS['test'](); m=KestrelModel(c); "
                 f"torch.save(dict(model=m.state_dict(), cfg=asdict(c), step=100, tokens_seen=5), {src!r})")
        subprocess.run([sys.executable, "-c", build], check=True)
        total, warmup, decay = 60, 5, 0.3
        cmd = [sys.executable, "-u", "-m", "kestrel.train", "--preset", "test", "--data-dir", data,
               "--curriculum", "--mix", "old=3,qnew=1", "--init-from", src, "--schedule-start", "100",
               "--steps", str(100 + total), "--warmup", str(warmup), "--decay-frac", str(decay),
               "--out", out, "--seq", "16", "--batch", "2", "--accum", "1", "--log-every", "1",
               "--eval-every", "100000", "--device", "cpu", "--muon-lr", "0.001", "--adamw-lr", "0.0001"]
        log = os.path.join(out, "metrics.jsonl")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        t0 = time.time()
        while time.time() - t0 < 300 and p.poll() is None:
            if os.path.exists(log) and sum(1 for _ in open(log)) >= 20:
                json.dump({"command": "stop"}, open(os.path.join(out, "control.json"), "w"))
                break
            time.sleep(0.2)
        first = p.communicate()[0]
        assert "stop requested" in first, first[-1500:]
        stopped = json.load(open(os.path.join(out, "status.json")))["step"]
        assert 100 + 20 <= stopped < 100 + total, stopped

        json.dump({"command": "run"}, open(os.path.join(out, "control.json"), "w"))
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-800:] + r.stderr[-1500:]
        assert "resumed from" in r.stdout, r.stdout[-800:]
        rows = [json.loads(l) for l in open(log)]
        assert rows[-1]["step"] == 100 + total
        after = [x for x in rows if x["step"] > stopped]
        assert after, "nothing logged after the resume"
        for x in rows:
            want = wsd_lr_mult(x["step"] - 1 - 100, total, warmup, decay)
            assert abs(x["lr_mult"] - want) < 1e-3, (x["step"], x["lr_mult"], want)

        # with no --schedule-start, a resume still re-warms (the behaviour earlier runs relied on)
        assert wsd_lr_mult(0, 10, 5, 0.3) == 0.2

        bad = subprocess.run(cmd[:cmd.index("--mix") + 1] + ["old=3,missing=1"] + cmd[cmd.index("--mix") + 2:],
                             capture_output=True, text=True)
        assert bad.returncode != 0 and "missing" in bad.stdout + bad.stderr, "a mix tag with no shards must stop the run"

    print(f"STEP4_TEST_OK | stopped at {stopped}, schedule held through the resume")


if __name__ == "__main__":
    main()
