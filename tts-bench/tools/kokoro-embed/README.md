# Kokoro Voice Style Extractor

Extract voice style embeddings from any WAV file for Kokoro TTS,
without the official (unreleased) style encoder.

Put in a 3~10 second voice sample, and get a voice style JSON that makes Kokoro speak in that voice.

**Requirements:** NVIDIA GPU with 4GB+ VRAM, CUDA support.

## Responsible Use

**This is research code released for academic purposes only.** Voice cloning technology can be misused for serious harm. By using this repository, you agree to the following.

- **Obtain explicit consent** from any speaker whose voice you intend to clone. Cloning a real person's voice without permission may be illegal in your jurisdiction.
- **Do not use this tool** for non-consensual voice impersonation, voice phishing (vishing), fraud, harassment, defamation, generation of misleading political or commercial content, or circumvention of voice-based authentication.
- **Do not target identifiable individuals** (including public figures) without their explicit permission.
- **Disclose synthetic audio** as AI-generated when distributing it, ideally with a watermark or provenance metadata (e.g., C2PA).

The author disclaims all liability for misuse. Evaluation in the accompanying paper uses Common Voice (CC0 license) and does not target identifiable individuals. If you observe misuse of this code, please open an issue.

### How it works:
```
              ┌───────────────────────────────────────────┐
              │         TTS Pipeline (PyTorch)            │
┌───────────┐ │ ┌────────┐  ┌───────────┐  ┌──────────┐  │ ┌─────────┐
│   style   │→│ │  BERT  │→ │ Predictor │→ │ Decoder  │  │→│ gen WAV │
│  vector   │ │ │Encoder │  │  (LSTM)   │  │          │  │ └────┬────┘
└─────┬─────┘ │ └────────┘  └───────────┘  └──────────┘  │      │
      │       └───────────────────────────────────────────┘      │
      │                                                          │
      │                      ┌────────────┐                      │
      │                      │   WavLM    │◄─────────────────────┘
      │                      │  Layer 3   │
      │                      │  (speaker  │◄── target WAV
      │                      │ identity)  │
      │                      └─────┬──────┘
      │                            │
      │      gradient              │ loss
      └────────────────────────────┘
      "update style to be more similar"
```

1. Auto-selects the closest preset style via WavLM Layer 3 distance
2. Synthesizes WAV via TTS, compares with target WAV using WavLM Layer 3 features
3. Updates style vector via gradient descent until convergence (early stop at 0.20)

### Convergence Guide:
Same-speaker baseline loss is **0.08~0.20**. Optimization stops automatically when this threshold is reached, or when no improvement is observed for 1,500 steps (plateau detection).

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Kokoro voice presets
Voice presets are automatically downloaded from [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) on first run. Alternatively, download voice `.bin` files manually and place them in `kokoro/voices/`.

### 3. Prepare your voice sample
Place your WAV file (3~10 seconds, single speaker) in the `wavs/` folder. Any sample rate is fine (auto-resampled).

### 4. Create config
Create `configs/ljs.json`:
```json
{
    "name": "ljs",
    "target_wav": "wavs/ljs.wav",
    "reference_style": "auto",
    "seed": 42,
    "lr": 2e-4,
    "num_steps": 20000,
    "speed": 1.0,
    "save_every": 500,
    "early_stop_loss_threshold": 0.20
}
```

| Parameter | Description |
|-----------|-------------|
| `name` | Name for saving checkpoints and output files |
| `target_wav` | Path to the voice WAV file to clone |
| `reference_style` | `"auto"` (auto-select closest) or preset name (e.g. `"am_adam"`) |
| `seed` | Random seed for reproducibility |
| `lr` | Learning rate. 2e-4 recommended |
| `num_steps` | Max optimization steps (early stopping / plateau detection may stop sooner) |
| `save_every` | Checkpoint save interval |

### 5. Run optimization
```bash
python optimize_style.py ljs
```
Training auto-resumes from the latest checkpoint if interrupted.

### 6. Use the extracted style
See `main.py` for inference example.

## Performance
Evaluated on 44 speakers × 5 utterances = 220 samples:

| | SIM (WavLM) ↑ | SIM (ECAPA) ↑ | SIM (ResNet) ↑ | WER ↓ |
|---|---|---|---|---|
| Nearest preset (no opt.) | 0.730 | 0.046 | 0.051 | 0.39% |
| **Proposed method** | **0.855** | **0.327** | **0.353** | **0.42%** |

## Technical Details

### Native PyTorch (No ONNX Conversion)
Kokoro is a native PyTorch model. Gradient flow is enabled by setting LSTM modules to train mode for cuDNN backward pass. All model parameters remain frozen; only the 256-dim style vector is optimized.

### WavLM Layer 3 Feature Matching
Based on probing analysis by [Chiu et al. (2025)](https://arxiv.org/abs/2501.05310), WavLM Layer 3 best encodes speaker identity. We compare time-averaged feature statistics (mean, std) between generated and target audio. Time-axis averaging reduces content dependency.

### Style Space
- `style` [1, 256] = 256 parameters (optimized)
- First 128 dims: acoustic features (decoder input)
- Last 128 dims: speaker features (predictor input)

### Models Used

| Model | Role |
|-------|------|
| Kokoro (82M) | Text-to-speech (hexgrad/Kokoro-82M) |
| WavLM-Large | Perceptual loss, Layer 3 (microsoft/wavlm-large) |

## Citation

If you use this work, please cite:

```bibtex
@misc{kim2026kokoroembed,
  author       = {Gyeongmin Kim},
  title        = {Extracting Voice Styles from Frozen TTS Models via Gradient-Based Inverse Optimization},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19646514},
  url          = {https://doi.org/10.5281/zenodo.19646514}
}
```

Preprint available on Zenodo: https://doi.org/10.5281/zenodo.19646514

## Looking for help?
If you have any questions, please feel free to open an issue.
