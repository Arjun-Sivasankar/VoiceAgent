# VoiceAgent

Ultron voice-agent workbench: a low-latency speech-to-speech pipeline with a browser HUD, a local desktop-action bridge, and the custom Kokoro Ultron voice.

The project combines:

- `speech-to-speech/`: vendored [Hugging Face speech-to-speech](https://github.com/huggingface/speech-to-speech) v1.0.0 source with the local OpenCode header and tool-message fixes.
- `bridge/`: loopback-only Windows bridge for the HUD's allow-listed desktop actions.
- `web/`: browser client for the realtime WebSocket endpoint.
- `tts-bench/`: the portable benchmark scripts, notes, results, voice-cloning tools, and small voice assets used to create the Ultron voice.
- `REPLICATION.md`: the detailed original Windows deployment notes.

Large runtime state is intentionally not tracked: Python environments, model caches, pip caches, logs, generated audio, screenshots, and downloaded datasets belong in the ignored `.runtime/` or local cache directories.

## Windows setup

The current deployment targets Windows 11 with Python 3.12 and an NVIDIA GPU with CUDA support.

From the repository root:

```powershell
cd speech-to-speech
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[kokoro]"
cd ..
```

The launcher reads the OpenCode Go credential from `%USERPROFILE%\.local\share\opencode\auth.json`. Keep that file outside the repository.

Run the backend directly:

```powershell
.\run-ultron-agent.ps1 serve
```

Run the complete local HUD stack:

```powershell
.\start-ultron-web.ps1
```

The backend listens on `127.0.0.1:8765`, the HUD on `http://127.0.0.1:7860`, and the desktop bridge on `127.0.0.1:7870`. Stop the stack with:

```powershell
.\stop-ultron-web.ps1
```

The launcher uses `tts-bench/models/kokoro/voices/ultron.pt` and stores downloaded runtime data under `.runtime/`. The base model and dependencies are downloaded on first use.

## Git tracking policy

Source, launchers, documentation, benchmark code, benchmark results, and the small selected voicepacks are tracked here. Do not add API keys, virtual environments, caches, logs, generated audio, or raw voice-reference recordings without reviewing their rights and provenance first.
