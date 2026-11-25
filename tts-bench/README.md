# TTS benchmark work

This is the source-and-results portion of the TTS work used by `VoiceAgent`.

Tracked here:

- Kokoro and Supertonic-3 benchmark scripts (`bench/`)
- Benchmark notes and CSV/JSON results (`notes/`, `results/`)
- Third-party voice-style tools (`tools/kokoro-embed`, `tools/supertonic3-voice-clone`) and the clone configs in `tools/kokoro-embed/configs/`
- The stock Kokoro voices `bm_george` and `bf_emma` from hexgrad/Kokoro-82M (Apache-2.0)

Not tracked here:

- Downloaded model weights and Hugging Face caches
- Python virtual environments
- Generated audio, voice-reference recordings and cloned voicepacks
- Tool checkpoints, logs, and local cache data

A custom voicepack, if you build one from a recording you have consent to use, goes in `models/kokoro/voices/custom.pt`; the launcher picks it up automatically. Without it, the stock voice `bm_fable` is used.
