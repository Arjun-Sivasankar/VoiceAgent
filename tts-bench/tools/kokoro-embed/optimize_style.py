"""
Extract voice style embedding from a WAV file for Kokoro TTS.

Core approach:
  - Use native PyTorch Kokoro model (no ONNX conversion needed)
  - Enable gradient flow by setting LSTM modules to train mode
  - Optimize style vector via WavLM Layer 3 feature matching
  - Layer selection based on Chiu et al. (2025) probing analysis
  - Early stopping at same-speaker baseline threshold (0.20)

Usage:
    python optimize_style.py ljs         # uses configs/ljs.json
"""

import json
import os
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import librosa
import soundfile as sf
import time

os.environ.pop('SSL_CERT_FILE', None)
os.environ.pop('CURL_CA_BUNDLE', None)
os.environ.pop('REQUESTS_CA_BUNDLE', None)
import httpx
_orig_client = httpx.Client
class _NoVerifyClient(_orig_client):
    def __init__(self, *args, **kwargs):
        kwargs['verify'] = False
        super().__init__(*args, **kwargs)
httpx.Client = _NoVerifyClient

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KOKORO_SR = 24000


# ===== WavLM perceptual loss =====

def load_wavlm():
    """Load WavLM-Large. Layer 3 best encodes speaker identity (Chiu et al. 2025)."""
    from transformers import WavLMModel
    model = WavLMModel.from_pretrained('microsoft/wavlm-large').to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def wavlm_feature_loss(wavlm, gen_wav, target_features):
    if gen_wav.ndim == 1:
        gen_wav = gen_wav.unsqueeze(0)
    gen_wav_16k = torchaudio.functional.resample(gen_wav, KOKORO_SR, 16000)
    gen_out = wavlm(gen_wav_16k, output_hidden_states=True)
    gen_feat = gen_out.hidden_states[3]
    tgt_mean, tgt_std = target_features
    loss = F.mse_loss(gen_feat.mean(dim=1), tgt_mean) + F.mse_loss(gen_feat.std(dim=1), tgt_std)
    return loss


def extract_wavlm_targets(wavlm, target_wav, sr=44100):
    if target_wav.ndim == 1:
        target_wav = target_wav.unsqueeze(0)
    wav_16k = torchaudio.functional.resample(target_wav, sr, 16000)
    with torch.no_grad():
        out = wavlm(wav_16k, output_hidden_states=True)
    feat = out.hidden_states[3]
    return feat.mean(dim=1), feat.std(dim=1)


# ===== Kokoro TTS =====

def load_kokoro():
    from kokoro import KPipeline
    pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    model = pipeline.model
    for p in model.parameters():
        p.requires_grad_(False)
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)):
            module.train()
    return pipeline, model


def kokoro_forward(model, phonemes_str, ref_s, speed=1.0):
    """Differentiable Kokoro forward pass (bypasses @torch.no_grad)."""
    input_ids = list(filter(lambda i: i is not None,
                            map(lambda p: model.vocab.get(p), phonemes_str)))
    input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(model.device)
    n = input_ids.shape[1]

    input_lengths = torch.full((1,), n, device=model.device, dtype=torch.long)
    text_mask = torch.arange(n, device=model.device).unsqueeze(0)
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))

    bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

    s = ref_s[:, 128:]
    d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
    x, _ = model.predictor.lstm(d)
    dur = model.predictor.duration_proj(x)
    dur = torch.sigmoid(dur).sum(axis=-1) / speed
    pred_dur = torch.round(dur).clamp(min=1).long().squeeze()

    indices = torch.repeat_interleave(torch.arange(n, device=model.device), pred_dur)
    pred_aln = torch.zeros((n, indices.shape[0]), device=model.device)
    pred_aln[indices, torch.arange(indices.shape[0], device=model.device)] = 1
    pred_aln = pred_aln.unsqueeze(0)

    en = d.transpose(-1, -2) @ pred_aln
    F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
    t_en = model.text_encoder(input_ids, input_lengths, text_mask)
    asr = t_en @ pred_aln

    audio = model.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
    return audio


def get_phonemes(pipeline, text):
    for result in pipeline(text, voice='af_bella', speed=1.0):
        return result.phonemes
    return None


# ===== Save style =====

def save_style(path, style, source_file=None, preset=None):
    from datetime import datetime
    style_json = {
        "style": style.cpu().numpy().tolist(),
        "dims": [1, 256],
        "metadata": {
            "source_file": source_file or "unknown",
            "preset": preset or "unknown",
            "extracted_at": datetime.now().isoformat()
        }
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(style_json, f)


# ===== Main =====

def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "ljs"
    if os.path.exists(arg):
        config_path = arg
    elif os.path.exists(f"configs/{arg}.json"):
        config_path = f"configs/{arg}.json"
    else:
        print(f"Config not found: {arg}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    name = cfg["name"]
    target_wav_path = cfg["target_wav"]
    reference_style = cfg.get("reference_style", "auto")
    seed = cfg.get("seed", 42)
    lr = cfg.get("lr", 2e-4)
    num_steps = cfg.get("num_steps", 20000)
    speed = cfg.get("speed", 1.0)
    save_every = cfg.get("save_every", 500)
    threshold = cfg.get("early_stop_loss_threshold", 0.20)

    opt_texts = [
        "The sun sets behind the mountains, painting the sky in shades of pink and orange.",
        "I took a walk in the park this morning, and the sound of birds was so pleasant.",
        "Would you like to sit down and have a cup of tea together?",
        "The wind carries the scent of flowers from the garden across the valley.",
        "Every great journey begins with a single step forward into the unknown.",
    ]

    log_dir = f"logs/{name}"
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)

    # Find latest checkpoint for auto-resume
    def find_latest_checkpoint():
        pattern = os.path.join(log_dir, f"{name}_*.json")
        files = [f for f in glob.glob(pattern) if "train_config" not in f]
        # keep only step-numbered checkpoints (ignore e.g. *_final.json)
        files = [
            f
            for f in files
            if os.path.splitext(os.path.basename(f))[0].split("_")[-1].isdigit()
        ]
        if not files:
            return None, 0
        latest = max(files, key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split("_")[-1]))
        step_num = int(os.path.splitext(os.path.basename(latest))[0].split("_")[-1])
        return latest, step_num

    latest_ckpt, start_step = find_latest_checkpoint()

    print(f"Using device: {DEVICE}")
    print(f"Name: {name}")

    # Load target WAV
    print(f"\nLoading target WAV: {target_wav_path}")
    target_wav, _ = librosa.load(target_wav_path, sr=44100)
    target_wav_t = torch.tensor(target_wav, dtype=torch.float32).to(DEVICE)
    print(f"  Duration: {len(target_wav)/44100:.2f}s")

    # Load models
    print("\nLoading WavLM-Large...")
    wavlm = load_wavlm()

    print("Loading Kokoro...")
    pipeline, kokoro_model = load_kokoro()

    print("Extracting target features (Layer 3)...")
    target_feats = extract_wavlm_targets(wavlm, target_wav_t)

    # Prepare texts
    opt_phonemes = [get_phonemes(pipeline, t) for t in opt_texts]

    # Get token count for style selection
    test_ids = list(filter(lambda i: i is not None,
                           map(lambda p: kokoro_model.vocab.get(p), opt_phonemes[0])))
    n_tokens = len(test_ids) + 2

    # Initialize style
    if latest_ckpt:
        print(f"\nResuming from: {latest_ckpt} (step {start_step})")
        with open(latest_ckpt) as f:
            data = json.load(f)
        ref_s = torch.tensor(data["style"], dtype=torch.float32).to(DEVICE)
        if ref_s.dim() == 1:
            ref_s = ref_s.unsqueeze(0)
        ref_s = ref_s.clone().requires_grad_(True)
        best_preset_name = data.get("preset", data.get("metadata", {}).get("preset", "unknown"))
    elif reference_style == "auto":
        print("\nFinding closest preset...")
        voice_names = [n.replace('.bin', '') for n in sorted(os.listdir("kokoro/voices")) if n.endswith('.bin')]
        if not voice_names:
            voice_names = ['af_bella', 'af_sarah', 'am_adam', 'am_michael', 'bf_alice', 'bm_daniel']

        best_dist = float('inf')
        best_voice = None
        for vn in voice_names:
            try:
                voice = pipeline.load_single_voice(vn)
            except:
                continue
            v_ref = voice[min(n_tokens, 509)].to(DEVICE)
            with torch.no_grad():
                test_wav = kokoro_forward(kokoro_model, opt_phonemes[0], v_ref, speed)
                dist = wavlm_feature_loss(wavlm, test_wav.to(DEVICE), target_feats).item()
            print(f"  {vn}: {dist:.4f}")
            if dist < best_dist:
                best_dist = dist
                best_voice = vn

        print(f"  >> Best: {best_voice} (dist={best_dist:.4f})")
        voice = pipeline.load_single_voice(best_voice)
        ref_s = voice[min(n_tokens, 509)].clone().to(DEVICE).requires_grad_(True)
        best_preset_name = best_voice
    else:
        print(f"\nInitializing from preset: {reference_style}")
        voice = pipeline.load_single_voice(reference_style)
        ref_s = voice[min(n_tokens, 509)].clone().to(DEVICE).requires_grad_(True)
        best_preset_name = reference_style

    print(f"  style: {ref_s.shape}")

    # Optimization
    optimizer = torch.optim.Adam([ref_s], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5, min_lr=1e-7
    )

    end_step = num_steps
    if start_step >= end_step:
        print(f"\nAlready reached target step ({end_step}). Nothing to do.")
        return

    start_time = time.time()
    print(f"\nStarting optimization (step {start_step+1} -> {end_step}, early stop at {threshold})...")

    best_loss = float('inf')
    best_ref_s = ref_s.detach().clone()
    best_step = start_step

    for step in range(start_step, end_step):
        optimizer.zero_grad()
        ph = opt_phonemes[step % len(opt_phonemes)]

        audio = kokoro_forward(kokoro_model, ph, ref_s, speed)
        loss = wavlm_feature_loss(wavlm, audio.to(DEVICE), target_feats)

        loss.backward()
        torch.nn.utils.clip_grad_norm_([ref_s], max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_ref_s = ref_s.detach().clone()
            best_step = step + 1

        if (step + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Step {step+1}/{end_step} | Loss: {loss.item():.4f} | LR: {current_lr:.6f} | Best: {best_loss:.4f}")

        if (step + 1) % save_every == 0:
            ckpt_path = f"{log_dir}/{name}_{step+1:06d}.json"
            save_style(ckpt_path, best_ref_s, target_wav_path, best_preset_name)
            print(f"  >> Checkpoint saved: {ckpt_path}")

        if best_loss <= threshold:
            print(f"  Early stop at step {step+1}: best loss {best_loss:.4f} <= {threshold}")
            break

        if (step + 1) - best_step >= 1500 and step + 1 >= 1500:
            print(f"  Plateau at step {step+1}: best {best_loss:.4f} at step {best_step}")
            break

    # Save final
    final_path = f"{log_dir}/{name}_final.json"
    print(f"\nSaving best style to: {final_path}")
    save_style(final_path, best_ref_s, target_wav_path, best_preset_name)
    elapsed = time.time() - start_time
    print(f"  Done! Best loss: {best_loss:.4f} | Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
