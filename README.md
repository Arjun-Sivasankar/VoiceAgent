# VoiceAgent

A low-latency **speech-to-speech** voice agent you can run on your own machine, plus the benchmark harness used to pick its text-to-speech backend.

The pipeline is the whole loop: voice activity detection, streaming speech recognition, a language model, and speech synthesis back into the browser. Around it sit a browser HUD, a loopback-only bridge that lets the agent take allow-listed desktop actions, and a benchmarking rig that measures what each stage actually costs on real hardware.

Most of the interesting work in voice agents is not the model. It is the latency budget, and this repository is a record of measuring it instead of guessing.

## Pipeline

```text
microphone
   │  WebAudio worklet, 16 kHz
   ▼
VAD ── turn detection ──► streaming STT ──► LLM (OpenAI-compatible endpoint)
   ▲                                              │
   │                                              ▼
browser HUD ◄──────── WebSocket ◄──────── TTS ── chunked audio
```

Everything between the browser and the model is local. The HUD talks to the backend over a WebSocket on `127.0.0.1`, audio is captured and played through AudioWorklets, and the desktop bridge accepts only a fixed allow-list of actions on loopback.

## What is in here

| Path | What it is |
|---|---|
| `speech-to-speech/` | Vendored [Hugging Face speech-to-speech](https://github.com/huggingface/speech-to-speech) v1.0.0, Apache-2.0, with local fixes to the request header handling and tool-message flow |
| `bridge/` | Loopback-only Windows bridge for the HUD's allow-listed desktop actions |
| `web/` | Browser client for the realtime WebSocket endpoint |
| `tts-bench/` | Benchmark scripts, notes, results, voice tooling and the small voicepacks |
| `REPLICATION.md` | The full original Windows deployment notes |
| `run-ultron-agent.ps1`, `start-ultron-web.ps1`, `stop-ultron-web.ps1` | Launchers |

`speech-to-speech/` is upstream work, not mine. Treat it as a vendored dependency with local patches; the original project's README, licence and CI live inside that directory and still apply.

## TTS benchmarks

Choosing a TTS backend for a real-time agent comes down to two numbers: how long until the first audio chunk, and how much of a GPU the model holds. `tts-bench/` measures both, across runtime configurations, on a laptop GPU and on the CPU.

Rig: **RTX 5060 Laptop, 8 GB, Windows 11, Python 3.12**. Model: `hexgrad/Kokoro-82M` v1.0. Metrics per run: synthesis time, time to first chunk, real-time factor, characters per second, peak VRAM, RSS and model load time. Each run writes a CSV plus a metadata JSON, so every number below can be regenerated.

| config | RTF short | RTF medium | RTF long | x-realtime | first chunk (medium) | peak VRAM | load |
|---|---|---|---|---|---|---|---|
| CUDA fp32 | 0.027 | 0.033 | 0.036 | ~30x | 0.46 s | ~1.1 GB | 1.6 s |
| CUDA fp32 + TF32 | 0.030 | 0.036 | 0.036 | ~28x | 0.50 s | ~1.1 GB | 1.7 s |
| **CUDA autocast (fp16)** | **0.033** | **0.027** | **0.027** | **~37x** | **0.38 s** | **~0.9 GB** | 2.3 s |
| CPU (16 threads) | 0.703 | 0.355 | 0.328 | ~3x | 4.9 s | — | 1.3 s |
| CPU (4 threads) | 0.388 | 0.324 | 0.322 | ~3x | 4.4 s | — | 0.9 s |

### What the numbers said

- **CUDA autocast is the config to ship.** About 37x realtime, 0.38 s to first audio, under 1 GB peak VRAM. Full fp16 weights crash on an LSTM dtype mismatch in the prosody predictor; autocast is the working path.
- **TF32 buys nothing here.** The workload is LSTM and convolution heavy, not large matmuls, so the tensor-core path has nothing to accelerate.
- **The first eager CUDA call costs about 1.3 s** in kernel autotuning. A warmup synthesis at startup removes it from the user's first turn.
- **Four CPU threads beat sixteen.** The model is small enough that synchronisation overhead dominates; the extra threads make it slower.
- **CPU is not viable interactively.** About 3x realtime sounds fine until you see the 4–9 s wait before the first chunk.

The full notes, including the fine-tuning survey and why the official training code does not exist, are in [`tts-bench/notes/`](tts-bench/notes/).

## Running it

The deployment targets Windows 11 with Python 3.12 and an NVIDIA GPU with CUDA.

```powershell
cd speech-to-speech
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[kokoro]"
cd ..
```

Then either run the backend directly:

```powershell
.\run-ultron-agent.ps1 serve
```

or bring up the whole local stack:

```powershell
.\start-ultron-web.ps1
```

The backend listens on `127.0.0.1:8765`, the HUD on `http://127.0.0.1:7860`, and the desktop bridge on `127.0.0.1:7870`. Stop it with `.\stop-ultron-web.ps1`.

The launcher reads its model credential from `%USERPROFILE%\.local\share\opencode\auth.json`, which must stay outside the repository. Runtime state — virtual environments, model caches, logs, generated audio — is written under `.runtime/` and is ignored.

## Notes and limitations

- This is a working single-machine deployment, not a service. There is no container orchestration, no multi-tenant isolation and no auth beyond binding to loopback.
- The desktop bridge is deliberately narrow: an allow-list of actions, on loopback only. That is what makes it defensible to ship at all.
- The browser HUD is one client. It assumes one person, one microphone, one conversation.
- Benchmark hardware is a laptop GPU. The relative ordering of the configurations should hold elsewhere; the absolute numbers should not be assumed.

## Tracking policy

Source, launchers, documentation, benchmark code, benchmark results and small selected voicepacks are tracked. Do not add API keys, virtual environments, caches, logs, generated audio, or raw voice-reference recordings without checking their rights and provenance first.

## Credits

- [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech) — the pipeline this builds on, Apache-2.0, vendored at v1.0.0
- [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) — the TTS model under test
