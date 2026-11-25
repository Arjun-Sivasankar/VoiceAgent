# Supertonic 3: local benchmark + finetuning notes

Date: 2026-09-12
Device: RTX 5060 Laptop (8 GB, sm_120) · 16 GB RAM · Win 11 · Python 3.12 venv (`../venv`)
Model: `Supertone/supertonic-3` (99M, ONNX, 31 languages + `na`, 44.1 kHz), pinned revision `724fb5ab`
Stored: `models/supertonic-3/` (385 MB: 4 ONNX graphs + 10 voice styles M1–M5/F1–F5)

## Setup

- `supertonic` 1.3.1 SDK (model dir is fully configurable; loaded offline via `TTS(model_dir=...)`)
- **GPU fix (important):** `onnxruntime-gpu` 1.30 is a CUDA 13 build (`cublas64_13.dll`) and its CUDAExecutionProvider silently does not register against torch's CUDA 12.8 DLLs → all runs fall back to CPU. Fix: `pip install onnxruntime-gpu==1.22.0` (CUDA 12 build) and add `torch\lib` to the DLL path. Verified: `ort sessions=['CUDAExecutionProvider','CPUExecutionProvider']`.
- The SDK hardcodes CPU-only providers; GPU is enabled by setting `supertonic.loader.DEFAULT_ONNX_PROVIDERS = ["CUDAExecutionProvider","CPUExecutionProvider"]` before `TTS(...)` (the bench script does this).
- One-time cost: first GPU inference takes ~40 s (CUDA kernels/graph init). Always warm up before timing.

## Benchmark rig

`bench/bench_supertonic.py` — provider cpu/cuda, sweeps: text length, diffusion steps, all 10 voices, 10 languages, batch 1–8.
Metrics: synth time, RTF, x-realtime, chars/s, RSS, GPU VRAM (sampled via NVML), model load time. CSV + meta JSON in `results/`, WAVs in `audio/supertonic-3/<tag>/`.

## Results (mean of timed runs; `results/supertonic3_matrix_all.csv`)

| condition | GPU (CUDA EP) | CPU (auto threads) |
|---|---|---|
| short (3.1 s audio) | 30x realtime (0.11 s) | 2.8x (1.1 s) |
| medium (13.7 s) | **77x** (0.18 s) | 3.5x (3.9 s) |
| long (47 s) | 63x (0.75 s) | 3.7x (12.9 s) |
| xlong (162 s) | 61x (2.66 s) | 3.8x (43 s) |

- **Diffusion steps ≠ free:** RTF scales linearly — GPU 4 steps 132x, 8 steps 77x, 16 steps 43x, 32 steps 22x; CPU 6.7x/3.4x/1.7x/0.9x. Default is 8.
- **Batching:** GPU 1→79x, 2→101x, 4→119x, 8→122x. Batch-4+ is nearly free throughput-wise.
- **Voices:** all 10 presets within ~±5% of each other (GPU ~75x).
- **Languages:** all 10 tested (en/ko/ja/de/es/fr/hi/ru/pt/it) ~26–38x on GPU short text — overhead-bound, not language-bound.
- **VRAM:** ~1.29 GB peak during synthesis (incl. CUDA context + ORT arena); load 0.7 s (CPU) / 1.7 s (GPU).
- CPU voice-sweep anomaly: M4/M5/F1 showed 2–4x slowdowns from transient system load; primary sweeps were consistent, so use those.

## Kokoro vs Supertonic 3 on this device (same rig style)

| | Kokoro-82M (torch, autocast) | Supertonic 3 (ORT CUDA) |
|---|---|---|
| Speed | ~37x realtime | ~77x (122x batch-8) |
| First audio | 0.38 s, **streaming chunks** | full utterance only |
| Audio | 24 kHz | 44.1 kHz |
| VRAM | ~0.9–1.2 GB | ~1.3 GB |
| Voices | 54 presets + voicepacks | 10 presets + style JSON cloning |
| License | Apache 2.0 | OpenRAIL-M (use restrictions) |

## Finetuning status

- **Official: none.** Supertone states there is no training/fine-tuning code, dataset, or guide — ONNX inference only. GitHub repo archived 2026-07-23; Voice Builder (hosted custom-voice service) shut down 2026-08-31.
- **Community voice cloning (style JSON optimization)** — the realistic route, fits 8 GB easily:
  - `saurabhv749/supertonic3-voice-clone` (MIT): generates `style_ttl` from a reference WAV via gradient-based inverse optimization against WavLM-Large Layer 3 features; `style_dp` frozen. Peak VRAM ~2.6 GB, typical 6–30 min. Paper: "Extracting Voice Styles from Frozen TTS Models via Gradient-Based Inverse Optimization" (Kim 2026).
  - `Saganaki22/supertonic_embeddings_trainer`: Gradio UI, v2+v3, WavLM or ECAPA-guided loss, stop/resume checkpoints.
  - `Fawzan09/voice-builder-for-supertonic-3`: Colab (free T4) notebook, no local GPU needed. Honest caveat: trains timbre/identity only — prosody is inherited from the preset used as starting point, and Supertone's original speaker encoder was never released, so results are "close likeness", not indistinguishable.
- **Full model fine-tune:** possible in research (Self-Purifying Flow Matching, arXiv 2512.17293, 4×A100, 10k iters) but no public training code — not practical locally.
- Trained style JSONs are small files; drop into `models/supertonic-3/voice_styles/` or load via `tts.get_voice_style_from_path(...)` and A/B them with this bench rig.

## Suggested next steps

1. Ears-on comparison: sample WAVs for both models are in `audio/kokoro/` and `audio/supertonic-3/` (same texts).
2. If a custom Supertonic voice is wanted: run `supertonic3-voice-clone` locally (~2.6 GB VRAM, minutes) rather than any cloud service.
3. Re-check ORT GPU on each upgrade: pin `onnxruntime-gpu==1.22.0` until the CUDA 13 toolchain aligns with torch.

## Custom voice clone experiment (2026-09-12)

- Reference: `voice-samples/prepared/custom_reference.wav` (5.83 s, 44.1 kHz, cut from a 21 s recording; prep via `bench/prepare_voice.py`)
- Tool: `tools/supertonic3-voice-clone` in its own venv (Python 3.12, torch 2.9.1+cu128); model read through a junction to `models/supertonic-3`
- Fixes applied: soundfile loader replaces torchaudio.load (torchcodec needs FFmpeg); `PYTHONIOENCODING=utf-8` for the emoji print; `torchcodec` uninstalled
- Training: 3000 steps, lr 2e-4, auto-init from F1 (loss 0.923 → best **0.0431**), 59.2 min on the 5060 (~0.54 s/step), peak VRAM within 8 GB
- Checkpoints every 500 steps in `tools/supertonic3-voice-clone/logs/custom/`; final → `models/supertonic-3/voice_styles/custom.json`
- Independent ECAPA eval (`bench/eval_similarity.py`, 14 sentences): cloned **mean cosine 0.845** (min 0.718, max 0.894, 14/14 ≥ 0.5) vs stock M1 **mean 0.002** — clone clearly targets the reference speaker
- A/B samples: `audio/clones/st3_custom/` vs `audio/clones/st3_M1/`
