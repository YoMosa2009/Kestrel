#!/usr/bin/env bash
# Finish Kestrel V2 after downloading kestrel-v2-results.zip from Colab into Downloads.
#   bash app/finish_v2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
R=C:/Users/mosaa/KestrelRelease; V2=$R/v2; mkdir -p "$V2"
unzip -o -q "$HOME/Downloads/kestrel-v2-results.zip" -d "$V2/results"
PY=.venv/Scripts/python.exe
$PY -m scripts.assemble_v2 --v1 "$R/v1/kestrel-nano-v1-weights.pt" \
    --pkm "$V2/results/kestrel-nano-v2-pkm-values.pt" --out "$V2/kestrel-nano-v2-weights.pt"
$PY -m scripts.export_gguf --ckpt "$V2/kestrel-nano-v2-weights.pt" --out "$V2/kestrel-nano-v2-f16.gguf" \
    --dtype f16 --name "Kestrel-Nano V2" --version V2
$PY -m scripts.export_gguf --ckpt "$V2/kestrel-nano-v2-weights.pt" --out "$V2/kestrel-nano-v2-q8_0.gguf" \
    --dtype q8_0 --name "Kestrel-Nano V2" --version V2
echo "md5 (Colab-tested f16 was 8f8778f491e1d9c8823955f016cc35b3):"; md5sum "$V2/kestrel-nano-v2-f16.gguf"
B=$R/Kestrel-V2-Windows-x64; rm -rf "$B"; mkdir -p "$B"
cp app/dist/KestrelStudio.exe "$V2/kestrel-nano-v2-f16.gguf" "$B/"
sed 's/KESTREL V1  -  Kestrel Studio + Kestrel-Nano V1/KESTREL V2  -  Kestrel Studio + Kestrel-Nano V2 (Roost: knows who made it)/' \
    "$R/Kestrel-V1-Windows-x64/README.txt" > "$B/README.txt"
(cd "$R" && rm -f Kestrel-V2-Windows-x64.zip && powershell -c "Compress-Archive -Path Kestrel-V2-Windows-x64 -DestinationPath Kestrel-V2-Windows-x64.zip")
git tag -a v2.0.0 -m "Kestrel V2" && git push -q origin v2.0.0
gh release create v2.0.0 -R YoMosa2009/Kestrel --title "Kestrel V2" --latest \
  --notes "Kestrel V2: Kestrel Studio V2 + Kestrel-Nano V2. Roost consolidation fixed: 7 true self-facts consolidated into memory slots, recall 0% -> 100%, regression +0.04%, all non-memory weights bit-identical to V1. Engine parity vs PyTorch: argmax 100%. Details: docs/12-roost-v2.md. Download Kestrel-V2-Windows-x64.zip, unzip, run KestrelStudio.exe." \
  "$R/Kestrel-V2-Windows-x64.zip" "$V2/kestrel-nano-v2-f16.gguf" "$V2/kestrel-nano-v2-q8_0.gguf" \
  "$V2/results/results.json"
echo "V2 released."
