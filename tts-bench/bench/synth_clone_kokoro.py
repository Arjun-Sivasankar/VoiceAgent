"""Synthesize the eval sentence set with a Kokoro voice (stock preset or cloned voicepack).

Usage:
    python synth_clone_kokoro.py --voice custom --name kokoro_custom --device cuda
    python synth_clone_kokoro.py --voice am_michael --name kokoro_am_michael
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "kokoro"
SAMPLE_RATE = 24000


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voice", required=True, help="voicepack name in models/kokoro/voices or a .pt path")
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="seconds of silence inserted at sentence boundaries (0 = off)",
    )
    ap.add_argument("--eval-json", default=str(ROOT / "voice-samples" / "eval" / "eval_sentences.json"))
    args = ap.parse_args()

    from kokoro import KModel, KPipeline

    kmodel = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config=str(MODEL_DIR / "config.json"),
        model=str(MODEL_DIR / "kokoro-v1_0.pth"),
    ).to(args.device).eval()
    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=kmodel)

    voice_path = Path(args.voice)
    if not voice_path.exists():
        voice_path = MODEL_DIR / "voices" / f"{args.voice}.pt"
    pack = torch.load(voice_path, weights_only=True)
    print(f"voice: {voice_path} | pack shape {tuple(pack.shape)}")

    texts = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    outdir = ROOT / "audio" / "clones" / args.name
    outdir.mkdir(parents=True, exist_ok=True)

    items = [(f"in{i+1}", t) for i, t in enumerate(texts["in_domain"])]
    items += [(f"n{i+1}", t) for i, t in enumerate(texts["neutral"])]

    for tag, text in items:
        sentences = [text]
        if args.pause > 0:
            sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
        chunks = []
        for si, sentence in enumerate(sentences):
            if si > 0 and args.pause > 0:
                chunks.append(np.zeros(int(args.pause * SAMPLE_RATE), dtype=np.float32))
            for result in pipe(sentence, voice=pack, speed=args.speed):
                if result.audio is not None:
                    chunks.append(result.audio.numpy())
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        path = outdir / f"{tag}_{args.name}.wav"
        sf.write(path, audio, SAMPLE_RATE)
        print(f"{path.name} | {len(audio)/SAMPLE_RATE:.2f}s | {text[:50]}", flush=True)

    print(f"wrote {len(items)} files to {outdir}")


if __name__ == "__main__":
    main()
