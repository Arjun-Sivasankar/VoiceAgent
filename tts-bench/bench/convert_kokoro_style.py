"""Convert a kokoro.embed style JSON ([1,256]) into a standard Kokoro voicepack .pt ([510,1,256]).

Usage:
    python convert_kokoro_style.py --json <style.json> --out models/kokoro/voices/ultron.pt
"""

import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pack-rows", type=int, default=510)
    args = ap.parse_args()

    import json

    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    style = torch.tensor(data["style"], dtype=torch.float32)
    if style.dim() == 2:
        style = style[0]
    assert style.shape == (256,), style.shape
    pack = style.reshape(1, 1, 256).repeat(args.pack_rows, 1, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack, out)
    check = torch.load(out, weights_only=True)
    print(f"wrote {out} | shape {tuple(check.shape)} | dtype {check.dtype}")
    print(f"style source: {data.get('metadata', {}).get('source_file', '?')}")


if __name__ == "__main__":
    main()
