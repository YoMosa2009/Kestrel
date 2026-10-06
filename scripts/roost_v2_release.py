"""Package a Roost V2 run for download (run on Colab after scripts.roost_v2).

    python -m scripts.roost_v2_release --run /content/v2r4

If the run built V2 (kestrel-nano-v2-weights.pt): export it to .gguf, write PyTorch
reference logits, and check the native C# engine against them (f32 and f16). Then zip
the run directory WITHOUT the 630 MB full checkpoint: results, log, parity reports and
the V2 memory tables (the only tensors V2 changes) -> kestrel-v2-results.zip.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import zipfile


def sh(cmd: str, log: str | None = None):
    print("$", cmd, flush=True)
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    text = out.stdout + out.stderr
    print(text[-4000:], flush=True)
    if log:
        with open(log, "w", encoding="utf-8") as f:
            f.write(text)
    return out.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--zip", default="/content/kestrel-v2-results.zip")
    args = ap.parse_args()
    run = args.run
    w = os.path.join(run, "kestrel-nano-v2-weights.pt")
    if os.path.exists(w):
        f32, f16 = "/content/kestrel-nano-v2-f32.gguf", "/content/kestrel-nano-v2-f16.gguf"
        fix = os.path.join(run, "parity-v2.json")
        sh(f'python -m scripts.export_gguf --ckpt {w} --out {f32} --dtype f32 '
           f'--name "Kestrel-Nano V2" --version V2 --fixture {fix}')
        sh(f'python -m scripts.export_gguf --ckpt {w} --out {f16} --dtype f16 '
           f'--name "Kestrel-Nano V2" --version V2')
        sh(f"md5sum {f16}", os.path.join(run, "v2-f16.md5"))
        dotnet = os.path.expanduser("~/.dotnet/dotnet")
        if not os.path.exists(dotnet):
            sh("curl -sSL https://dot.net/v1/dotnet-install.sh | bash -s -- --channel 8.0 > /dev/null")
        env = "DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1"
        for name, g in (("f32", f32), ("f16", f16)):
            sh(f"{env} {dotnet} run -c Release --project app/Kestrel.Cli -- parity {g} {fix}",
               os.path.join(run, f"parity-v2-{name}.txt"))
    else:
        print("no V2 weights in this run (gate not passed) - packaging results only")
    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(run)):
            if f != "kestrel-nano-v2-weights.pt":
                z.write(os.path.join(run, f), f)
    print(f"{args.zip}: {os.path.getsize(args.zip) / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
