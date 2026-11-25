# Kokoro-82M: local benchmark + finetuning notes

Date: 2026-09-12
Device: RTX 5060 Laptop (8 GB, sm_120) · 16 GB RAM · Win 11 · Python 3.12 venv (`../venv`)
Model: `hexgrad/Kokoro-82M` v1.0 (latest official release, Jan 2025; v1.1-zh exists but is Chinese-only experimental)

## Setup

- Model + all 54 voices: `models/kokoro/` (346 MB, loaded fully offline via KModel local config/model paths)
- Kokoro 0.9.4 / misaki 0.9.4 / torch 2.11.0+cu128 / onnxruntime-gpu 1.30.0
- espeak-ng comes bundled via `espeakng-loader` (no system install); misaki auto-installed `en_core_web_sm` on first run

## Benchmark rig

- `bench/bench_kokoro.py` — one runtime config: length sweep (short/medium/long/xlong), voice sweep, optional speed sweep.
  Metrics per run: total synth time, time-to-first-chunk, RTF, x-realtime, chars/s, peak VRAM (torch), RSS, model load time.
  Writes `results/kokoro_<tag>_<ts>.csv` + meta JSON; sample WAVs to `audio/kokoro/<tag>/`.
- `bench/run_kokoro_matrix.py` — runs the matrix (cuda-fp32, cuda-fp32-tf32, cuda-fp16=autocast, cpu, cpu-4t) and aggregates.
- Reproduce: `venv\Scripts\python bench\run_kokoro_matrix.py` (add `--quick` for smoke, `--speed-sweep` for speed params).

## Results (mean of timed runs; aggregated `results/kokoro_matrix_all.csv`)

| config | RTF short | RTF medium | RTF long | RTF xlong | x-realtime | first chunk (medium) | peak VRAM | load |
|---|---|---|---|---|---|---|---|---|
| cuda fp32 | 0.027 | 0.033 | 0.036 | 0.035 | ~30x | 0.46 s | ~1.1 GB | 1.6 s |
| cuda fp32 + TF32 | 0.030 | 0.036 | 0.036 | 0.037 | ~28x | 0.50 s | ~1.1 GB | 1.7 s |
| **cuda autocast (fp16)** | **0.033** | **0.027** | **0.027** | **0.030** | **~37x** | **0.38 s** | **~0.9 GB** | 2.3 s |
| cpu (16 threads) | 0.703 | 0.355 | 0.328 | 0.327 | ~3x | 4.9 s | — | 1.3 s |
| cpu (4 threads) | 0.388 | 0.324 | 0.322 | 0.319 | ~3x | 4.4 s | — | 0.9 s |

### Findings

- **Best config: CUDA autocast** — ~37x realtime, 0.38 s to first audio (medium), <1 GB peak VRAM. Full fp16 weights crash (LSTM dtype mismatch in ProsodyPredictor); autocast is the working path.
- TF32 gives no benefit (workload is LSTM/conv-heavy, not big matmul).
- First eager CUDA call costs ~1.3 s (kernel autotuning) — do a warmup synthesis on startup.
- CPU is ~3x realtime with 4 threads; **4 threads beat 16** (small model, sync overhead). CPU first-chunk is 4–9 s, too slow for interactive use.
- Long-form scales linearly; xlong audio (153 s) in ~4.6 s on GPU.
- Memory grows with utterance length (625 MB short → ~1.16 GB long).

## Finetuning status

- **Official training code does not exist**; hexgrad never released it. All options are community forks of StyleTTS2 (Kokoro's architecture). Voicepack = style tensor `[510, 1, 256]` float32, ~524 KB.

### Full fine-tune (new voice/language), Stage 1 (decoder) + Stage 2 (prosody)

| Project | Notes |
|---|---|
| `semidark/kikiri-tts` (ex kokoro-deutsch) | Reference recipe: dataset prep → weight conversion → S1 → S2 → voicepack extract. Patched StyleTTS2 (PyTorch API fixes, 178-token vocab, 510-token filter). Apache 2.0. Published de voices. |
| `Jeevav62/tts-finetune-recipes` (kokoro-recipe) | Self-contained, bundles training framework + support models + base weights. Docs: ~8 min S1 + ~65 min S2 (10 epochs, no GAN) on 4090 with 1000 clips. |
| `tterrasson/kokoro-french` | Most complete docs: checkpoint→KModel export, per-speaker voicepack extraction, warns to use Stage 1 checkpoint for style encoder (S2 degrades it). |
| `shreyaskarnik/bol-tts-marathi` | End-to-end Marathi example, documents gotchas, vocab slot addition. |

Requirements: 24 kHz mono WAV, 3–25 s clips, IPA phonemization, 1–5 h of target speech (S2), style-matching G2P at train+inference.
VRAM: 12–24 GB recommended. **On our 8 GB 5060**: feasible only with `batch_size=1`, `joint_epoch=99` (disable GAN, saves ~4 GB), reduced `max_len`; Stage 2 with GAN is out of reach locally — use cloud A100 for that. S1 is the lighter stage.

### Lightweight options (fit 8 GB easily)

- **Voice cloning via style-vector optimization** — `kdrkdrkdr/kokoro.embed`: 3–10 s reference WAV, gradient descent on the 256-dim style vector against WavLM layer-3 features (~0.08–0.20 loss = same-speaker territory), no model weights changed. Best bang-for-buck on this GPU.
- **Voice mixing** — average style vectors of existing voices (af = 50/50 bella+sarah). Free, CPU.
- `kvoicewalk` + kokoro-web — iterative voice walking, reported ~20–40 min per 100–300 steps; outputs standard `.pt` voicepacks.
- `amit154154/kokoro_jarvis` — research scripts for coefficient mixing vs full embedding optimization.

### Pipeline for a custom voice on this machine (recommended)

1. Optimize style vector with kokoro.embed (hours, GPU-light).
2. If quality is insufficient, rent a 24 GB+ GPU for kikiri-tts S1+S2 (~1.5 h for 1k clips).
3. Extract voicepack, drop into `models/kokoro/voices/`, benchmark it with this rig before use.

## Custom voice clone experiment (2026-09-12)

- Reference: `voice-samples/prepared/custom_reference.wav` (5.83 s); tool: `tools/kokoro-embed` running in the bench venv (+ librosa)
- Fixes/environment: created `kokoro/voices/` (auto-select crashes without it); WavLM-Large + Kokoro weights cached in `models/hf-cache`
- Training: auto-init from `bm_daniel` (WavLM dist 1.94), lr 2e-4, **early stop at step 5396/20000**, best loss **0.1978** (≤0.20 same-speaker threshold), 36.1 min (~0.37 s/step)
- Style JSON `[1,256]` → standard voicepack via `bench/convert_kokoro_style.py` → `models/kokoro/voices/custom.pt` `[510,1,256]`
- Independent ECAPA eval (`bench/eval_similarity.py`, 14 sentences):
  - kokoro_custom: **mean 0.470** (min 0.357 / max 0.536)
  - am_michael stock baseline: mean 0.244
  - st3_custom (Supertonic, same reference): mean 0.845
- Kokoro's WavLM-space loss converged to the paper's same-speaker range; **Supertonic clone is clearly stronger on this reference** (better identity model + native style vectors). Use Kokoro clone if a specific Kokoro prosody/tempo is wanted.
- Samples: `audio/clones/kokoro_custom/` vs `audio/clones/kokoro_am_michael/`
- **Pacing fix (same day):** default rendering was rushed (~3.4–5.3 s per in-domain line vs stock 4.3–6.3 s). `bench/synth_clone_kokoro.py` now supports `--pause <s>` (silence at sentence boundaries) and `--speed`; paced variants (`--pause 0.4 --speed 0.95`) are in `audio/clones/kokoro_custom_paced/` and `kokoro_am_michael_paced/`. Voice identity unchanged (same style vector).
- **Deep run / v2 (same day, later):** resumed from the v1 checkpoint (step 5000) with `configs/custom_deep.json` (lr 1e-4, early-stop target loss <=0.12). **Plateau-stopped at step 11,846** (best **0.1619** at step 10,346; loss flattened ~0.17 with LR decayed to 3e-6), 64.6 min wall (incl. a ~34 min machine-sleep gap). Only the style vector was optimized; no model weights changed.
  - Packaged `custom_final.json` -> `models/kokoro/voices/custom.pt` (v1 preserved as `custom_v1.pt`).
  - Independent ECAPA eval (14 sentences, same reference): **mean 0.508** (min 0.428 / max 0.572, 9/14 >=0.5) vs v1 0.470 (paced) / 0.470 (plain) and stock am_michael 0.247. Modest but clear identity gain; still well below the Supertonic clone (0.845).
  - Samples: `audio/clones/kokoro_custom_v2/` (rendered `--pause 0.4 --speed 0.95`).
