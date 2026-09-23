import torch
import soundfile as sf
from kokoro import KPipeline

NAME = "ljs"
VOICE_STYLE = f"logs/{NAME}/{NAME}_final.json"
TEXT = "Hello, this is a test of the extracted voice style."

pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
model = pipeline.model

# Load optimized style
import json
with open(VOICE_STYLE) as f:
    data = json.load(f)
style = torch.tensor(data["style"], dtype=torch.float32).to(model.device)
if style.dim() == 1:
    style = style.unsqueeze(0)

# Synthesize
from optimize_style import kokoro_forward, get_phonemes

for name_mod, module in model.named_modules():
    if isinstance(module, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)):
        module.train()

phonemes = get_phonemes(pipeline, TEXT)
with torch.no_grad():
    audio = kokoro_forward(model, phonemes, style, 1.0)

sf.write(f"{NAME}.wav", audio.cpu().numpy(), 24000)
print(f"Saved {NAME}.wav")
