# Ultron Voice Agent — Replication Guide

This guide was ported from the original T9 deployment into the `VoiceAgent` repository.
The launchers now resolve paths relative to the repository and keep runtime state under
`.runtime/`; the historical `D:\voice-agent` and `D:\tts-bench` paths below document the
machine on which the original setup was built.

Last updated: 2026-09-13

This document is the single source of truth for rebuilding the Ultron voice agent on another
machine (or after a disk wipe). It covers the cloned pipeline, every modification made to it,
the environment, the web UI, and a changelog.

Maintenance rule: whenever the pipeline, launcher, environment, or web UI changes, add an entry
to the Changelog and update the affected section in the same change.

---

## 1. Architecture

Speech pipeline (upstream: https://github.com/huggingface/speech-to-speech, v1.0.0):

    mic (browser)
      -> WebSocket ws://127.0.0.1:8765/v1/realtime  (OpenAI Realtime protocol)
      -> Silero VAD v5 -> Parakeet TDT 0.6B (STT, CUDA)
      -> OpenCode Go API (LLM, OpenAI-compatible chat completions)
      -> Kokoro-82M with the custom "Ultron" voicepack (TTS, CUDA)
      -> PCM16 audio back over the same WebSocket -> browser playback + HUD

| Component | Location |
|---|---|
| Pipeline clone + venv | `D:\voice-agent\speech-to-speech` |
| Python base runtime | `D:\Python312` |
| Web HUD (browser client) | `D:\voice-agent\web` |
| System bridge (desktop actions) | `D:\voice-agent\bridge\bridge.py` |
| Screenshots (created at runtime) | `D:\voice-agent\screenshots` |
| Launcher / service scripts | `D:\voice-agent\*.ps1` |
| Protocol tests | `D:\voice-agent\test-ws-client.mjs`, `test-tools.mjs` |
| TTS voice / model work | `D:\tts-bench` (see its `notes\kokoro.md`) |
| Shared model cache | `D:\tts-bench\models\hf-cache` |
| Torch hub cache (Silero) | `D:\voice-agent\torch-cache` |
| NLTK data | `D:\voice-agent\nltk_data` |

Ports: backend `8765`, web UI `7860`, system bridge `7870`. All bind to `127.0.0.1` only.

---

## 2. Prerequisites

- Windows 11, NVIDIA GPU with CUDA (built for RTX 5060 Laptop 8 GB, sm_120), driver supporting CUDA 12.8.
- Git, Node.js (only for the protocol test scripts), internet for first model downloads.
- The custom Kokoro voicepack `ultron.pt` (see section 5). It is not derivable from the repo.
- No API key is required for STT/TTS; the LLM uses the OpenCode Go subscription key already on the machine.

---

## 3. Replication steps

### Step 1 — Python 3.12 on D:

The upstream venv format expects Python 3.12. On this machine the base runtime was copied from
`C:\Users\arjun\AppData\Local\Programs\Python\Python312` to `D:\Python312` (the python.org
installer refuses to install a second copy of the same version).

    robocopy "C:\Users\arjun\AppData\Local\Programs\Python\Python312" "D:\Python312" /E /COPY:DAT

Verify: `& D:\Python312\python.exe -c "import sys; print(sys.version)"`

### Step 2 — Clone the pipeline at the pinned commit

    git clone --depth 1 https://github.com/huggingface/speech-to-speech.git D:\voice-agent\speech-to-speech

Built against commit `16d7f98` (v1.0.0, 2026-09-06). If you need the exact tree:

    git fetch --depth 1 origin 16d7f98
    git checkout 16d7f98

### Step 3 — Virtualenv and dependencies

    cd D:\voice-agent\speech-to-speech
    D:\Python312\python.exe -m venv .venv
    set PIP_CACHE_DIR=D:\voice-agent\pip-cache
    .venv\Scripts\python.exe -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
    .venv\Scripts\python.exe -m pip install -e ".[kokoro]"

Notes:
- PyTorch must be installed from the cu128 index first; plain PyPI torch would be CPU-only.
- `[kokoro]` pulls `kokoro>=0.9.2` (installed 0.9.4); misaki/kokoro install `en_core_web_sm`
  into the venv automatically on first use (spacy model for G2P).
- Reference versions: torch 2.11.0+cu128, transformers 5.17.0, kokoro 0.9.4, nano-parakeet 0.2.1,
  faster-qwen3-tts 0.4.0, onnxruntime 1.30.0, numpy 2.5.3, Python 3.12.9.

### Step 4 — Apply the pipeline patches (two files)

Only two upstream files are modified. Re-apply both after any `git pull`.

**Patch A — extra LLM headers** (`src/speech_to_speech/LLM/base_openai_compatible_language_model.py`)

    --- a/src/speech_to_speech/LLM/base_openai_compatible_language_model.py
    +++ b/src/speech_to_speech/LLM/base_openai_compatible_language_model.py
    @@ -3,6 +3,7 @@
     import base64
     import io
     import ipaddress
    +import json
     import logging
     import os
     import wave
    @@ -207,7 +208,16 @@ class BaseOpenAICompatibleHandler(BaseHandler[LLMIn, LLMOut], ABC):
                 and self._is_local_base_url(base_url)
             ):
                 api_key = "none"
    -        self.client = OpenAI(api_key=api_key, base_url=base_url)
    +        extra_headers: dict[str, str] = {}
    +        raw_extra_headers = os.environ.get("S2S_LLM_EXTRA_HEADERS")
    +        if raw_extra_headers:
    +            try:
    +                parsed_headers = json.loads(raw_extra_headers)
    +                if isinstance(parsed_headers, dict):
    +                    extra_headers = {str(key): str(value) for key, value in parsed_headers.items()}
    +            except json.JSONDecodeError:
    +                logger.warning("Ignoring invalid S2S_LLM_EXTRA_HEADERS (not valid JSON)")
    +        self.client = OpenAI(api_key=api_key, base_url=base_url, default_headers=extra_headers or None)
             self._extra_body = self._build_extra_body(base_url, disable_thinking, reasoning_effort)
             self._prefetch_worker_slots = BoundedSemaphore(PREFETCH_PROVIDER_WORKER_LIMIT)
             self._prefetch_workers_lock = Lock()

Why: the OpenCode Go endpoint requires an `x-opencode-session` routing header on every request.
JSON in `S2S_LLM_EXTRA_HEADERS` is parsed and passed as `default_headers`.

**Patch B — sanitize orphaned tool messages** (`src/speech_to_speech/LLM/chat_completions_language_model.py`)

Add a module-level helper before `_chat_messages` and call it at the end of `_chat_messages`
(which now also keeps the existing per-message normalisation):

    def _sanitize_tool_messages(messages):
        paired = []
        for message in messages:
            if message.get("role") == "tool":
                previous = paired[-1] if paired else None
                ids = set()
                if previous is not None and previous.get("role") == "assistant":
                    ids = {call.get("id") for call in (previous.get("tool_calls") or [])}
                if message.get("tool_call_id") not in ids:
                    continue
            paired.append(message)
        for index, message in enumerate(paired):
            calls = message.get("tool_calls")
            if message.get("role") != "assistant" or not calls:
                continue
            responded = set()
            cursor = index + 1
            while cursor < len(paired) and paired[cursor].get("role") == "tool":
                responded.add(paired[cursor].get("tool_call_id"))
                cursor += 1
            remaining = [call for call in calls if call.get("id") in responded]
            if remaining:
                message["tool_calls"] = remaining
            else:
                message.pop("tool_calls", None)
                if not message.get("content"):
                    message["content"] = ""
        return paired

and change the return of `_chat_messages` to `return _sanitize_tool_messages(messages)`.

Why: the realtime layer can leave a `function_call_output` behind when its `function_call`
is rolled back or evicted (a tool result racing a superseding user turn). OpenAI-compatible
providers then reject the whole request with "Messages with role 'tool' must be a response to
a preceding message with 'tool_calls'". The sanitizer drops orphans and prunes unmatched
tool_calls so the conversation keeps working. Verified with `test-tools-followup.mjs`.

To recover after `git pull`: re-apply both patches or keep our copies of the two files.

### Step 5 — Models and caches

No manual model downloads are required. On first run the pipeline fetches:

| Model | Source | Cache |
|---|---|---|
| Kokoro-82M | `hexgrad/Kokoro-82M` | shared HF cache |
| Parakeet TDT 0.6B v3 | `nvidia/parakeet-tdt-0.6b-v3` | shared HF cache |
| Smart Turn v3.2 | `pipecat-ai/smart-turn-v3` (ONNX) | shared HF cache |
| Silero VAD v5 | torch hub `snakers4/silero-vad` | `D:\voice-agent\torch-cache` |
| en_core_web_sm | spaCy | inside the venv |

The shared HF cache is reused from the TTS project to avoid re-downloading Kokoro:

    D:\tts-bench\models\hf-cache

### Step 6 — The Ultron voicepack

- Active file: `D:\tts-bench\models\kokoro\voices\ultron.pt` (shape `[510,1,256]`, float32).
- Produced by `kdrkdrkdr/kokoro.embed` style-vector optimization against a reference clip;
  v2 (deep run) is the current pack, v1 preserved as `ultron_v1.pt`.
- Full provenance, training configs, ECAPA scores: `D:\tts-bench\notes\kokoro.md`.
- The pipeline loads it by absolute path via `--kokoro_voice`; `kokoro`'s `load_single_voice`
  accepts a `.pt` path directly (no copy into the package needed).

### Step 7 — LLM endpoint (OpenCode Go)

- Base URL: `https://opencode.ai/zen/go/v1` (OpenAI-compatible, catalog id `opencode-go`).
- Model: `deepseek-v4.1-flash` with `reasoning_effort=none` for low voice latency.
- Required extra header: `x-opencode-session` (any stable id, e.g. `ultron-voice-agent`).
- Key: read at launch from the OpenCode login store:
  `%USERPROFILE%\.local\share\opencode\auth.json` -> `opencode-go.key`.
  It is never copied or hardcoded.

### Step 8 — Launcher

`D:\voice-agent\run-ultron-agent.ps1` is the authoritative command. Key contents:

    $env:HF_HOME    = "D:\tts-bench\models\hf-cache"
    $env:TORCH_HOME = "D:\voice-agent\torch-cache"
    $env:NLTK_DATA  = "D:\voice-agent\nltk_data"
    $env:PYTHONUTF8 = "1"
    $env:S2S_LLM_EXTRA_HEADERS = '{"x-opencode-session":"ultron-voice-agent"}'
    $opencodeKey = (Get-Content "$env:USERPROFILE\.local\share\opencode\auth.json" -Raw |
                    ConvertFrom-Json).'opencode-go'.key

    & D:\voice-agent\speech-to-speech\.venv\Scripts\speech-to-speech.exe $mode @common

with `$common` flags:

    --stt parakeet-tdt  --parakeet_tdt_device cuda  --parakeet_tdt_language en
    --llm_backend chat-completions
    --responses_api_base_url https://opencode.ai/zen/go/v1
    --responses_api_api_key <key>  --model_name deepseek-v4.1-flash
    --responses_api_reasoning_effort none  --responses_api_stream
    --tts kokoro  --kokoro_device cuda
    --kokoro_voice D:\tts-bench\models\kokoro\voices\ultron.pt
    --kokoro_lang_code b  --kokoro_speed 0.95
    --num_pipelines 2

Modes: no argument = `local` (mic + speakers), or `serve` (API only).
Note: `--kokoro_lang_code b` must match the STT language mapping (`en` -> `b`), otherwise the
handler switches voice on the first turn and falls back to a stock preset.

### Step 9 — Web HUD

Files (all on D):

| File | Purpose |
|---|---|
| `web\index.html` | entire client: WebSocket, mic worklet, playback, HUD, orb, tools, memory, wake word |
| `web\vendor\augmented-ui.min.css` | sci-fi clipped HUD frames (v2.0.0) |
| `web\vendor\anime.umd.min.js` | animation engine (v4.5.0, MIT) for boot, springs, shockwaves |
| `start-ultron-web.ps1` | starts backend + static server, waits, opens browser |
| `stop-ultron-web.ps1` | kills listeners on 8765/7860 |
| `test-ws-client.mjs` | Realtime protocol smoke test (text + audio events) |
| `test-tools.mjs` | tool-call round-trip test |

Serve the HUD (any static server works; mic requires a secure context, so use localhost):

    D:\Python312\python.exe -m http.server 7860 --bind 127.0.0.1 --directory D:\voice-agent\web

Then open `http://127.0.0.1:7860`.

### Step 10 — Verification

    powershell -ExecutionPolicy Bypass -File D:\voice-agent\run-ultron-agent.ps1 serve
    node D:\voice-agent\test-ws-client.mjs    # expect session.updated, transcript, 18+ output_audio deltas, response.done
    node D:\voice-agent\test-tools.mjs        # expect a function_call, then a spoken answer

### Step 11 — System bridge (optional but recommended)

`D:\voice-agent\bridge\bridge.py` is a stdlib-only loopback HTTP service that performs
allow-listed desktop actions for the HUD. Start it via `start-ultron-web.ps1`, or manually:

    D:\Python312\python.exe D:\voice-agent\bridge\bridge.py

Endpoints (all on `http://127.0.0.1:7870`):

| Route | Body | Result |
|---|---|---|
| `GET /health` | - | `{"ok":true}` |
| `POST /action` | `{"action":"system_info"}` | CPU/RAM/GPU/disk summary |
| `POST /action` | `{"action":"open_app","args":{"name":"notepad"}}` | opens an allow-listed app |
| `POST /action` | `{"action":"open_url","args":{"url":"https://..."}}` | opens the default browser |
| `POST /action` | `{"action":"media","args":{"command":"play_pause"}}` | media/volume key |
| `POST /action` | `{"action":"lock"}` | locks the workstation |
| `POST /action` | `{"action":"screenshot"}` | saves a PNG to `D:\voice-agent\screenshots`, returns path |

Security model: loopback bind; CORS granted only to `http://127.0.0.1:7860` /
`http://localhost:7860`; every action requires the `X-Ultron: 1` header. A custom header
forces a CORS preflight that other origins fail, which blocks drive-by localhost CSRF.
There is no arbitrary command execution; the app list is a fixed map in `APP_MAP`.

---

## 4. Environment variables

| Variable | Value | Consumed by |
|---|---|---|
| `HF_HOME` | `D:\tts-bench\models\hf-cache` | pipeline model downloads |
| `TORCH_HOME` | `D:\voice-agent\torch-cache` | Silero VAD via torch hub |
| `NLTK_DATA` | `D:\voice-agent\nltk_data` | LLM sentence tokenizer |
| `PYTHONUTF8` | `1` | console encoding on Windows |
| `S2S_LLM_EXTRA_HEADERS` | `{"x-opencode-session":"ultron-voice-agent"}` | patched LLM client |
| `PIP_CACHE_DIR` | `D:\voice-agent\pip-cache` | installs (keeps C: clean) |

---

## 5. Web HUD feature map

All of the following live in `web\index.html` (single file).

| Feature | Where |
|---|---|
| Realtime WebSocket + session.update (tools, instructions, memory preamble) | `start()`, `ws.onopen` |
| Mic capture (inline AudioWorklet, resampled to 16 kHz PCM16) | `start()` worklet string |
| Playback scheduler with barge-in cancel | `playPcm16`, `clearPlayback` |
| 16 kHz native formats (no format fields) | `session.update` payload |
| WebGL shader orb (simplex noise, fresnel, state modes, shockwave) | `FRAG`, `initGL`, `renderGL` |
| Line-spectrum "Eye" visual mode | `drawEye`, `drawEyePlate` |
| Multi-color arc/tick ring + orbit dots | `drawColorArcs`, `drawColorTicks`, `drawOrbitDots` |
| Ring/tick/spectrum/nodes HUD | `drawRings`, `drawSpectrum`, `drawNodes` |
| VU meters, waveform, corner readouts | `drawMeters`, `drawWaveform`, `drawBrackets` |
| State machine + colors | `STATES`, `setState` |
| Latency HUD (STT/TTS/TOTAL + avg) | `lat`, `drawLatency` |
| JARVIS sound design (WebAudio) | `tone`, `sweep`, `sfx` |
| Tools: time, web search (Wikipedia/DDG), weather (Open-Meteo), remember, recall | `TOOLS`, `execTool`, `webSearch`, `getWeather`, `runToolCall` |
| Memory: facts + history in localStorage, injected into instructions | `loadFacts`, `saveFacts`, `pushHistory`, `memoryPreamble` |
| Wake word "Ultron" (SpeechRecognition) + push-to-talk (Space) | `startWake`, `triggerWake`, `closeGate`, key handlers |
| Visual mode toggle, sound toggle, wake toggle | dock buttons `#vis`, `#sfx`, `#wake` |
| Boot choreography + spring entrances (anime.js) | `bootUI`, `addBubble` |
| Personas: prompt + voicepack + accent hue per persona | `PERSONAS`, `applyPersona`, `#persona` |
| Image input (file picker or Ctrl+V paste, downscaled JPEG) | `sendImageFile`, `downscaleImage`, `addImageBubble` |
| Token/cost HUD (per-turn and session, OpenCode Go price list) | `usage`, `PRICE`, `drawLatency` |
| System control tools through the bridge | `open_app`, `open_url`, `media_control`, `lock_screen`, `screenshot`, `system_info`, `bridgeCall` |
| Bridge health indicator | `checkBridge`, "bridge online/offline" HUD line |
| Center core shifts left while chat is present | `coreShift`, frame loop, `renderGL(t, shiftPx)` |
| Tool-result race guard (wait for active response) | `responseActive`, `continueAfterTool` |
| Tool-call de-duplication (server emits both arguments.done and output_item.done) | `executedCalls`, guard in `runToolCall` |

Tool-call protocol used (OpenAI Realtime GA):
- inbound: `response.output_item.added` (function_call, carries `call_id`), `response.function_call_arguments.delta/done`
- outbound: `conversation.item.create` with `{type:"function_call_output", call_id, output}` then `response.create`

---

## 6. Runbook

    # start everything (backend + HUD) and open the browser
    powershell -ExecutionPolicy Bypass -File D:\voice-agent\start-ultron-web.ps1

    # backend only (mic client or API clients)
    powershell -ExecutionPolicy Bypass -File D:\voice-agent\run-ultron-agent.ps1 serve

    # stop everything
    powershell -ExecutionPolicy Bypass -File D:\voice-agent\stop-ultron-web.ps1

    # logs
    D:\voice-agent\logs\serve.err.log      # pipeline log
    D:\voice-agent\logs\web.err.log        # static server
    D:\voice-agent\logs\bridge.log         # system bridge actions

HUD controls: Enable microphone, Mute, Sound, Wake, Persona, Image, Visual (Eye/Reactor), Clear.
Push-to-talk: hold Space. Attach images with the Image button or Ctrl+V.

---

## 7. Known caveats

- exFAT: the T9 drive cannot store symlinks or junctions. The Supertonic junction in the TTS
  project was materialized, and HF cache symlinks were dereferenced.
- Wake word uses the browser's speech service (Chromium only, needs internet). Push-to-talk is
  the universal fallback.
- The bridge can open apps and lock the machine. It is loopback-only and header-gated; do not
  expose port 7870 on a network. Actions are an explicit allowlist, no arbitrary commands.
- The HUD origin and ports are fixed in several places (bridge CORS allowlist, HUD fetch URL);
  changing the web port requires updating `ALLOWED_ORIGINS` in `bridge.py`.
- Memory lives in the browser profile's localStorage (per origin), not server-side.
- Web search is Wikipedia + DuckDuckGo Instant Answer (CORS-safe, no API key). For richer
  results, add a server-side search proxy with a real search API key.
- The "Ultron" voice is cloned from an actor's performance; keep it private, do not distribute
  or use commercially.
- The LLM costs money via the OpenCode Go subscription; audio replies are short by design.

---

## 8. Changelog

### 2026-09-13 (latest+1) — Duplicate tool-call fix
- The server emits both `response.function_call_arguments.done` and
  `response.output_item.done` for the same function_call (verified with `test-tool-events.mjs`),
  which made the HUD execute every tool twice, send two `function_call_output` items and
  generate two identical replies (the "read out twice" bug).
- Added an `executedCalls` de-duplication guard keyed on `call_id`, and clear it on disconnect.
- Added a bounded retry for transient "response in progress / outputs pending" errors instead
  of surfacing them as a hard error state.

### 2026-09-13 (latest) — Tool-history fix, cleaner HUD, core shift
- Fixed the provider 400 after tool calls: patched `chat_completions_language_model.py` to
  sanitize orphaned `tool` messages (Patch B above) and made the HUD wait for the active
  response before issuing the post-tool `response.create` (`continueAfterTool`).
- Bumped the launcher to `--num_pipelines 2` so a stale client cannot lock out the HUD.
- HUD cleanup: smaller dock buttons with shorter labels, auto-fading hint, waveform only while
  active, calmer tick/arc opacity, toned-down orb core, fewer orbit dots.
- The center core (Eye or Reactor) now shifts smoothly to the left while chat messages are
  present and returns to center when the feed is cleared.

### 2026-09-13 (later) — Bridge, personas, cost HUD, image input
- Added `bridge\bridge.py`: loopback allow-listed desktop control (open apps/urls, media and
  volume keys, lock, screenshot, system info) with `X-Ultron` header gate and origin-locked CORS.
- Wired `start-ultron-web.ps1` / `stop-ultron-web.ps1` for the bridge on port 7870.
- Added six system-control tools to the HUD plus a bridge online/offline indicator.
- Added token/cost tracking from `response.done` usage and a per-turn + session cost HUD
  (prices: input $0.15/M, output $0.60/M, cache read $0.003/M).
- Added personas Ultron / JARVIS / FRIDAY: persona-specific system prompt, Kokoro voicepack
  (ultron.pt, bm_george, bf_emma) and HUD accent hue, cycled with the Persona button and
  persisted in localStorage.
- Added image input: Image button and Ctrl+V paste, downscaled to 1024 px JPEG and sent as
  `input_image` content; renders a thumbnail bubble in the feed.

### 2026-09-13 — Voice agent + HUD
- Cloned `huggingface/speech-to-speech` at `16d7f98` (v1.0.0) into `D:\voice-agent\speech-to-speech`.
- Created venv on `D:\Python312`; installed torch 2.11.0+cu128 and `-e .[kokoro]`.
- Patched `src/speech_to_speech/LLM/base_openai_compatible_language_model.py` for
  `S2S_LLM_EXTRA_HEADERS` (OpenCode Go session header).
- Added `run-ultron-agent.ps1` (Parakeet TDT + OpenCode Go + Kokoro Ultron voice).
- Built `web/index.html`: JARVIS-style HUD with raw Realtime WebSocket client, inline mic
  worklet, playback scheduler, WebGL shader orb and line-spectrum Eye mode.
- Added vendor libs: augmented-ui 2.0.0, anime.js 4.5.0.
- Added `start-ultron-web.ps1`, `stop-ultron-web.ps1`, `test-ws-client.mjs`, `test-tools.mjs`.
- Features added to the HUD: JARVIS sound design, tools (time/web/weather/remember/recall),
  persistent memory preamble + history, wake word with push-to-talk fallback, latency HUD,
  visual mode toggle, boosted tool-calling instructions.
- Removed the rotating radar sweep; scaled the outer ring assembly down ~15%.

### 2026-09-12 — Voice work (tts-bench, prerequisite)
- Kokoro and Supertonic-3 benchmarks on the RTX 5060; both models stored locally.
- Ultron voice cloned for both models; Kokoro deep run produced `ultron.pt` (v2).
- See `D:\tts-bench\notes\kokoro.md` and `notes\supertonic-3.md` for details.

### 2026-09-13 — Storage migration
- Moved `tts-bench` from C: to `D:\tts-bench`; made both venvs self-contained on `D:\Python312`.
- Cleared pip and model caches from C: (about 30 GB freed).
