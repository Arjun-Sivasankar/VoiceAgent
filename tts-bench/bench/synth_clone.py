"""Synthesize the eval sentence set with a Supertonic style JSON (clone or stock preset).

Usage:
    python synth_clone.py --style <path.json> --name st3_custom --device cuda
"""

import argparse
import json
import os
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "supertonic-3"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--style", required=True, help="path to style JSON")
    ap.add_argument("--name", required=True, help="output subfolder / filename tag")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--speed", type=float, default=1.05)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--eval-json", default=str(ROOT / "voice-samples" / "eval" / "eval_sentences.json"))
    args = ap.parse_args()

    if args.device == "cuda":
        import torch

        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        os.add_dll_directory(lib)
        os.environ["PATH"] = lib + os.pathsep + os.environ["PATH"]
        import supertonic.loader as loader

        loader.DEFAULT_ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    from supertonic import TTS

    tts = TTS(model="supertonic-3", model_dir=str(MODEL_DIR), auto_download=False)
    style = tts.get_voice_style_from_path(args.style)
    texts = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))

    outdir = ROOT / "audio" / "clones" / args.name
    outdir.mkdir(parents=True, exist_ok=True)

    items = [(f"in{i+1}", t) for i, t in enumerate(texts["in_domain"])]
    items += [(f"n{i+1}", t) for i, t in enumerate(texts["neutral"])]

    for tag, text in items:
        wav, dur = tts.synthesize(
            text, voice_style=style, total_steps=args.steps, speed=args.speed, lang=args.lang
        )
        path = outdir / f"{tag}_{args.name}.wav"
        sf.write(path, wav.squeeze(), tts.sample_rate)
        print(f"{path.name} | {wav.shape[-1] / tts.sample_rate:.2f}s | {text[:50]}", flush=True)

    print(f"wrote {len(items)} files to {outdir}")


if __name__ == "__main__":
    main()
