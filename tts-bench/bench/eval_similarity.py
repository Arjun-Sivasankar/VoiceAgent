"""ECAPA speaker-similarity evaluation for generated samples.

Computes cosine similarity between the reference speaker embedding and every
generated WAV, using SpeechBrain ECAPA-VOXCELEB. Run with the tool venv
(tools\\supertonic3-voice-clone\\.venv) which has speechbrain.

Usage:
    python eval_similarity.py --reference <ref.wav> --dirs st3_ultron st3_M1
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_16k(path):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    t = torch.from_numpy(data.T).contiguous()
    if t.shape[0] > 1:
        t = t.mean(dim=0, keepdim=True)
    if sr != 16000:
        t = torchaudio.transforms.Resample(sr, 16000)(t)
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--dirs", nargs="+", required=True, help="subdirs under audio/clones")
    ap.add_argument("--clones-root", default=str(Path(__file__).resolve().parents[1] / "audio" / "clones"))
    args = ap.parse_args()

    from speechbrain.inference.speaker import EncoderClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device}
    )
    clf.eval()

    def emb(path):
        with torch.no_grad():
            e = clf.encode_batch(load_16k(path).to(device)).squeeze()
        return torch.nn.functional.normalize(e, p=2, dim=0)

    ref = emb(args.reference)
    print(f"reference: {args.reference}")
    for name in args.dirs:
        files = sorted(glob.glob(str(Path(args.clones_root) / name / "*.wav")))
        if not files:
            print(f"{name}: no files")
            continue
        sims = [float(torch.dot(ref, emb(f))) for f in files]
        strong = sum(1 for s in sims if s >= 0.5)
        print(
            f"{name}: n={len(sims)} | mean={np.mean(sims):.4f} | min={min(sims):.4f} | "
            f"max={max(sims):.4f} | >=0.5: {strong}/{len(sims)}"
        )


if __name__ == "__main__":
    main()
