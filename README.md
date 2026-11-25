# VoiceAgent

A low-latency **speech-to-speech** voice agent you can run on your own machine, plus the benchmark harness used to pick its text-to-speech backend.

The pipeline is the whole loop: voice activity detection, streaming speech recognition, a language model, and speech synthesis back into the browser. Around it sit a browser HUD, a loopback-only bridge that lets the agent take allow-listed desktop actions, and a benchmarking rig that measures what each stage actually costs on real hardware.

Most of the interesting work in voice agents is not the model. It is the latency budget, and this repository is a record of measuring it instead of guessing.

<!-- demo: docs/demo.gif -->

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

The running configuration: Silero VAD, Parakeet TDT 0.6B for speech recognition on CUDA, an OpenAI-compatible chat-completions endpoint as the language model, and Kokoro-82M for speech synthesis on CUDA.

## What is in here

| Path | What it is |
|---|---|
| `speech-to-speech/` | Vendored [Hugging Face speech-to-speech](https://github.com/huggingface/speech-to-speech) v1.0.0, Apache-2.0, with small local patches (listed below) |
| `bridge/` | Loopback-only Windows bridge for the HUD's allow-listed desktop actions |
| `web/` | Browser HUD and client for the realtime WebSocket endpoint |
| `tts-bench/` | Benchmark scripts, notes, results and voice tooling |
| `REPLICATION.md` | The full original Windows deployment notes |
| `run-voice-agent.ps1`, `start-voice-web.ps1`, `stop-voice-web.ps1` | Launchers |
| `test-*.mjs` | Node scripts that exercise the realtime protocol and tool-call events |

## What's mine vs upstream

`speech-to-speech/` is upstream work, not mine. Treat it as a vendored dependency with local patches; the original project's README, licence and CI live inside that directory and still apply.

| Mine | Upstream or third party |
|---|---|
| `web/index.html`: the HUD (realtime WebSocket client, mic and playback worklets, WebGL orb, latency and token-cost readouts, tools for time, web search, weather and memory, personas, wake word and push-to-talk, image input) | `speech-to-speech/`: the VAD, STT, LLM and TTS pipeline and its realtime server (Hugging Face, Apache-2.0) |
| `bridge/bridge.py`: the allow-listed desktop action bridge | `web/vendor/`: anime.js (MIT) and augmented-ui (BSD-2-Clause) |
| `run-voice-agent.ps1`, `start-voice-web.ps1`, `stop-voice-web.ps1` and the `test-*.mjs` protocol tests | `tts-bench/tools/kokoro-embed/`: Kokoro style-vector extractor by Gyeongmin Kim (kdrkdrkdr/kokoro.embed) |
| `tts-bench/bench/`, `tts-bench/notes/`, `tts-bench/results/` and the clone configs in `tts-bench/tools/kokoro-embed/configs/` | `tts-bench/tools/supertonic3-voice-clone/`: Supertonic-3 style trainer by Saurabh Verma (MIT, see its LICENSE and NOTICE) |
| Local patches inside `speech-to-speech/` (below) | `tts-bench/models/kokoro/voices/bm_george.pt`, `bf_emma.pt`: stock voices from [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0) |

Local patches to the vendored pipeline:

- `LLM/base_openai_compatible_language_model.py`: extra request headers from `S2S_LLM_EXTRA_HEADERS`, needed by the LLM endpoint for session routing.
- `LLM/chat_completions_language_model.py`: drops orphaned `tool` messages from the chat history, which fixed a provider 400 error after tool calls.
- `TTS/qwen3_tts_handler.py`, `scripts/benchmark_tts.py`, `tests/install_smoke.py`: the upstream sample reference clip `TTS/ref_audio.wav` is not shipped here, so these now ask for a user-supplied clip with a clear error instead of assuming it exists.

The full patch text is in `REPLICATION.md`.

## TTS benchmarks

Choosing a TTS backend for a real-time agent comes down to two numbers: how long until the first audio chunk, and how much of a GPU the model holds. `tts-bench/` measures both, across runtime configurations, on a laptop GPU and on the CPU.

Rig: **RTX 5060 Laptop, 8 GB, Windows 11, Python 3.12**. Model: `hexgrad/Kokoro-82M` v1.0. Metrics per run: synthesis time, time to first chunk, real-time factor, characters per second, peak VRAM, RSS and model load time. Each run writes a CSV plus a metadata JSON, so every number below can be regenerated from `tts-bench/results/`.

| config | RTF short | RTF medium | RTF long | x-realtime | first chunk (medium) | peak VRAM | load |
|---|---|---|---|---|---|---|---|
| CUDA fp32 | 0.027 | 0.033 | 0.036 | ~30x | 0.46 s | ~1.1 GB | 1.6 s |
| CUDA fp32 + TF32 | 0.030 | 0.036 | 0.036 | ~28x | 0.50 s | ~1.1 GB | 1.7 s |
| **CUDA autocast (fp16)** | **0.033** | **0.027** | **0.027** | **~37x** | **0.38 s** | **~0.9 GB** | 2.3 s |
| CPU (16 threads) | 0.703 | 0.355 | 0.328 | ~3x | 4.9 s | n/a | 1.3 s |
| CPU (4 threads) | 0.388 | 0.324 | 0.322 | ~3x | 4.4 s | n/a | 0.9 s |

### What the numbers said

- **CUDA autocast is the config to ship.** About 37x realtime, 0.38 s to first audio, under 1 GB peak VRAM. Full fp16 weights crash on an LSTM dtype mismatch in the prosody predictor; autocast is the working path.
- **TF32 buys nothing here.** The workload is LSTM and convolution heavy, not large matmuls, so the tensor-core path has nothing to accelerate.
- **The first eager CUDA call costs about 1.3 s** in kernel autotuning. A warmup synthesis at startup removes it from the user's first turn.
- **Four CPU threads beat sixteen.** The model is small enough that synchronisation overhead dominates; the extra threads make it slower.
- **CPU is not viable interactively.** About 3x realtime sounds fine until you see the 4 to 9 s wait before the first chunk.

The full notes, including the fine-tuning survey, the voice-cloning experiments with speaker-similarity scores, and why the official training code does not exist, are in [`tts-bench/notes/`](tts-bench/notes/).

## Voices

No cloned voice and no reference recording ships with this repository.

- **Default:** the launcher uses the stock Kokoro voice `bm_fable`, which Kokoro downloads from `hexgrad/Kokoro-82M` on first use. The HUD's other two personas use the stock voices `bm_george` and `bf_emma`, bundled in `tts-bench/models/kokoro/voices/` from the same Apache-2.0 model repository.
- **Your own voice:** you can build a Kokoro voicepack from a 3 to 10 second reference clip with `tts-bench/tools/kokoro-embed` (config template: `configs/custom.json`) and convert it with `tts-bench/bench/convert_kokoro_style.py`. Save it as `tts-bench/models/kokoro/voices/custom.pt` and the launcher picks it up automatically. To use any other voice, set `VOICEAGENT_KOKORO_VOICE` to a stock voice name or a `.pt` path.
- **Consent:** the reference recording must be supplied by you, and you need the speaker's consent to clone it. Do not clone actors, public figures or anyone who has not agreed. Both cloning tools state the same terms in their READMEs. Custom voicepacks are git-ignored so they are not committed by accident.

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
.\run-voice-agent.ps1 serve
```

or bring up the whole local stack:

```powershell
.\start-voice-web.ps1
```

The backend listens on `127.0.0.1:8765`, the HUD on `http://127.0.0.1:7860`, and the desktop bridge on `127.0.0.1:7870`. Stop it with `.\stop-voice-web.ps1`. In the HUD, the wake word is "Hey agent"; Space works as push-to-talk.

The launcher reads its model credential from `%USERPROFILE%\.local\share\opencode\auth.json`, which must stay outside the repository. Runtime state (virtual environments, model caches, logs, generated audio) is written under `.runtime/` and is ignored.

## Notes and limitations

- This is a working single-machine deployment, not a service. There is no container orchestration, no multi-tenant isolation and no auth beyond binding to loopback.
- The desktop bridge is deliberately narrow: an allow-list of actions, on loopback only. That is what makes it defensible to ship at all.
- The browser HUD is one client. It assumes one person, one microphone, one conversation.
- The launcher is Windows and PowerShell only, and the LLM endpoint and key path are specific to the original setup; point `--responses_api_base_url` and `--model_name` in `run-voice-agent.ps1` at your own OpenAI-compatible endpoint.
- Benchmark hardware is a laptop GPU. The relative ordering of the configurations should hold elsewhere; the absolute numbers should not be assumed.

## Tracking policy

Source, launchers, documentation, benchmark code, benchmark results and the two stock voicepacks are tracked. Do not add API keys, virtual environments, caches, logs, generated audio, voice-reference recordings or cloned voicepacks.

## License

My own code (everything outside the upstream and third-party paths listed above) is under the MIT License, see [LICENSE](LICENSE). The vendored `speech-to-speech/` directory stays under Apache-2.0 ([`speech-to-speech/LICENSE`](speech-to-speech/LICENSE)), and the other third-party components keep their own licences. See [NOTICE](NOTICE).

## Credits

- [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech): the pipeline this builds on, Apache-2.0, vendored at v1.0.0
- [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M): the TTS model under test and the stock voices
- [kokoro.embed](https://doi.org/10.5281/zenodo.19646514) by Gyeongmin Kim and [supertonic3-voice-clone](https://github.com/saurabhv749/supertonic3-voice-clone) by Saurabh Verma: the voice-style tools used in the experiments
