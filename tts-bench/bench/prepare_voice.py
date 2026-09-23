"""Prepare a reference voice sample for style cloning / finetuning.

Splits a monologue at its longest silences, trims and peak-normalizes each
segment, writes 44.1k and 16k mono versions plus a primary 5-10s clip.

Usage:
    python prepare_voice.py <input.wav> --name ultron --segments 4
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF


def find_speech_segments(x, sr, thresh_db=-40.0, min_gap_s=0.25, min_seg_s=0.5):
    frame_s = 0.02
    n = int(sr * frame_s)
    xp = np.pad(x, (0, (-len(x)) % n))
    rms = np.sqrt((xp.reshape(-1, n) ** 2).mean(axis=1))
    active = rms > 10 ** (thresh_db / 20)
    segs = []
    i = 0
    while i < len(active):
        if active[i]:
            j = i
            while j < len(active) and active[j]:
                j += 1
            segs.append([i, j])
            i = j
        else:
            i += 1
    merged = []
    for s in segs:
        if merged and (s[0] - merged[-1][1]) * frame_s < min_gap_s:
            merged[-1][1] = s[1]
        else:
            merged.append(list(s))
    return [(a * frame_s, b * frame_s) for a, b in merged if (b - a) * frame_s >= min_seg_s]


def collapse_to_n(segs, n):
    segs = [list(s) for s in segs]
    while len(segs) > n:
        gaps = [segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)]
        i = int(np.argmin(gaps))
        segs[i][1] = segs[i + 1][1]
        del segs[i + 1]
    return segs


def peak_normalize(x, target_db=-1.0):
    peak = float(np.abs(x).max())
    if peak <= 1e-9:
        return x
    return x * (10 ** (target_db / 20) / peak)


def to_16k(x, sr):
    return AF.resample(torch.from_numpy(x).unsqueeze(0), sr, 16000).squeeze(0).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("--name", default="voice")
    ap.add_argument("--segments", type=int, default=4)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--primary-target-s", type=float, default=8.0)
    args = ap.parse_args()

    src = Path(args.input)
    outdir = Path(args.outdir) if args.outdir else src.parent / "prepared"
    outdir.mkdir(parents=True, exist_ok=True)

    x, sr = sf.read(src)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    dur = len(x) / sr
    print(f"input: {src.name} | {sr} Hz | {dur:.2f} s")

    segs = find_speech_segments(x, sr)
    print(f"detected {len(segs)} speech segments")
    segs = collapse_to_n(segs, args.segments)

    manifest = {"input": str(src), "sample_rate": sr, "duration_s": round(dur, 3), "segments": []}
    written = []
    for i, (start, end) in enumerate(segs, 1):
        a = max(0, int((start - 0.03) * sr))
        b = min(len(x), int((end + 0.03) * sr))
        seg = peak_normalize(x[a:b].copy())
        base = f"{args.name}_s{i}"
        p44 = outdir / f"{base}.wav"
        p16 = outdir / f"{base}_16k.wav"
        sf.write(p44, seg, sr, subtype="PCM_16")
        sf.write(p16, to_16k(seg, sr), 16000, subtype="PCM_16")
        seg_info = {
            "file": str(p44),
            "file_16k": str(p16),
            "start_s": round(start, 3),
            "end_s": round(end, 3),
            "dur_s": round(len(seg) / sr, 3),
        }
        manifest["segments"].append(seg_info)
        written.append((base, seg, seg_info["dur_s"], p44))
        print(f"  {base}: {seg_info['dur_s']:6.2f} s  ({start:.2f}-{end:.2f})")

    primary_parts, primary_len = [], 0.0
    for _, seg, seg_dur, _ in written:
        if primary_len >= args.primary_target_s:
            break
        primary_parts.append(seg)
        primary_len += seg_dur
    primary = peak_normalize(np.concatenate(primary_parts))
    p_primary = outdir / f"{args.name}_primary.wav"
    p_primary_16k = outdir / f"{args.name}_primary_16k.wav"
    sf.write(p_primary, primary, sr, subtype="PCM_16")
    sf.write(p_primary_16k, to_16k(primary, sr), 16000, subtype="PCM_16")
    manifest["primary"] = {
        "file": str(p_primary),
        "file_16k": str(p_primary_16k),
        "dur_s": round(len(primary) / sr, 3),
    }
    (outdir / f"{args.name}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"primary: {len(primary)/sr:.2f} s -> {p_primary.name}")
    print(f"manifest -> {outdir / (args.name + '_manifest.json')}")


if __name__ == "__main__":
    main()
