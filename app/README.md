# Kestrel Studio

A Windows app that runs Kestrel models **natively** — no Python, no PyTorch, no GPU
required. Copy `KestrelStudio.exe` to any 64-bit Windows 10/11 PC, import a Kestrel
`.gguf`, and chat.

| project | what |
|---|---|
| `Kestrel.Engine` | .NET 8 library: GGUF reader, byte-level BPE tokenizer, the Kestrel forward pass, sampling, chat sessions, Roost memory |
| `Kestrel.Studio` | the WPF app |
| `Kestrel.Cli` | `kestrel parity / bench / chat` — cross-platform, used to verify the engine against PyTorch on Colab |

## Getting a model file

Kestrel models are exported from a training checkpoint:

```
python -m scripts.export_gguf --ckpt runs/sft/ckpt.pt --out kestrel-nano-q8_0.gguf --dtype q8_0
```

The Colab notebook's stage G does this automatically and writes `Kestrel/release/` on Drive
(`kestrel-nano-f16.gguf`, `kestrel-nano-q8_0.gguf`). The file is a standard GGUF container
with architecture `kestrel`. llama.cpp, LM Studio and Ollama don't implement that
architecture, so use this app.

## What the app exposes

- **Thinking depth (R).** Kestrel's core blocks run R times per token. R=2 is how it
  was trained; R=1 is faster and R=3 spends more compute per word. Changing R re-reads
  the chat into a new state.
- **Sessions.** The GLA blocks keep a fixed-size recurrent state and the attention blocks
  keep a KV cache. A saved `.kses` file holds both exactly, so a session resumes without
  re-reading the transcript.
- **Roost memory.** Facts you teach are stored in `episodic.jsonl` (the same format as
  `kestrel/roost/store.py`). The most relevant ones are recalled into each message, using
  the model's own hidden states as embeddings. The same file can be consolidated into the
  model's memory slots by the Roost sleep job.
- **Inside.** Live heatmaps of the two product-key-memory sites (16,384 slots each),
  showing which slots each token reads.

## Build

```
dotnet build app/Kestrel.slnx -c Release
powershell -ExecutionPolicy Bypass -File app/publish.ps1     # -> app/dist/KestrelStudio.exe
```

Data lives in `%LOCALAPPDATA%\KestrelStudio` (`models\`, `sessions\`, `memory\`, `settings.json`).

## Correctness

The engine is a line-for-line port of `kestrel/model.py`, written in recurrent form: the
GLA chunked scan is replaced by its exact per-token recurrence, and attention uses a KV
cache. `kestrel parity <model.gguf> parity.json` checks it against PyTorch's
reference outputs. It compares tokenizer ids, last-position logits at R=1..3, argmax
agreement at every position, and the Roost embedding.
