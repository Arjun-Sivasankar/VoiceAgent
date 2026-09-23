"""Systematic Kokoro-82M (v1.0) benchmark rig.

Measures latency, first-chunk time, RTF, throughput and memory for a single
runtime configuration (device x dtype x threads). Writes long-format CSV +
metadata JSON into results/, sample WAVs into audio/kokoro/<tag>/.

The model is loaded fully from local files in models/kokoro (no network).

Examples:
    python bench_kokoro.py --device cuda --dtype fp32 --tag cuda-fp32
    python bench_kokoro.py --device cuda --dtype fp32 --tf32 --tag cuda-fp32-tf32
    python bench_kokoro.py --device cuda --dtype fp16 --tag cuda-fp16
    python bench_kokoro.py --device cpu --threads 4 --tag cpu-4t --runs 2,2,2,1
"""

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "kokoro"
AUDIO_ROOT = ROOT / "audio" / "kokoro"
RESULTS_DIR = ROOT / "results"
REPO_ID = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24000

for _d in (AUDIO_ROOT, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TEXTS = {
    "short": "The quick brown fox jumps over the lazy dog.",
    "medium": (
        "Speech synthesis has come a long way in a short time. Modern neural "
        "text to speech systems can produce natural sounding voices on consumer "
        "hardware, and this benchmark measures exactly how fast that happens."
    ),
    "long": (
        "Real time factor, or R T F, is the ratio between the time a system "
        "takes to generate an utterance and the duration of the generated audio. "
        "A value below one means the model runs faster than real time. For "
        "interactive applications, the time to the first audio chunk matters just "
        "as much as the total generation speed, because a long wait before "
        "playback starts feels sluggish even if the rest of the sentence renders "
        "quickly. Memory usage is the third pillar of this benchmark. A model "
        "that fits comfortably in graphics memory can be reused across many "
        "requests, while one that spills over into system memory will slow down "
        "under load. Measuring all three together gives a practical picture of "
        "what a deployment can handle."
    ),
    "xlong": (
        "The Time Traveller, for so it will be convenient to speak of him, was "
        "expounding a recondite matter to us. His pale grey eyes shone and "
        "twinkled, and his usually pale face was flushed and animated. The fire "
        "burned brightly, and the soft radiance of the incandescent lights in the "
        "lilies of silver caught the bubbles that flashed and passed in our "
        "glasses. Our chairs, being his patents, embraced and caressed us rather "
        "than submitted to be sat upon, and there was that luxurious after-dinner "
        "atmosphere when thought roams gracefully free of the trammels of "
        "precision. And he put it to us in this way, marking the points with a "
        "lean forefinger, as he sat and lazily admired his earnestness over this "
        "new paradox, as it seemed, and his fecundity.\n"
        "You must follow me carefully. I shall have to controvert one or two "
        "ideas that are almost universally accepted. The geometry, for instance, "
        "they taught you at school is founded on a misconception. Is not that "
        "rather a large thing to expect us to begin upon? said Filby, an "
        "argumentative person with red hair. I do not mean to ask you to accept "
        "anything without reasonable ground for it. You will soon admit as much "
        "as I need from you. You know of course that a mathematical line, a line "
        "of thickness nil, has no real existence. They taught you that? Neither "
        "has a mathematical plane. These things are mere abstractions. That is "
        "all right, said the Psychologist. Nor, having only length, breadth, and "
        "thickness, can a cube have a real existence. There I object, said Filby. "
        "Of course a solid body may exist. All real things must have extension "
        "in three dimensions. Most people are so obstinately wedded to the "
        "common sense view of space that they will not admit a fourth dimension. "
        "Yet a four dimensional space is a very simple matter.\n"
        "The Time Traveller smiled. Are you sure we can move freely in Space? "
        "Right and left we can go, backward and forward we can go, and men have "
        "always gone up and down. I have been a student of mathematics for some "
        "years, and I have never heard of any suggestion of such a thing until "
        "now. Is it not possible that the geometry of four dimensions has been "
        "sketched out? There is no difference between Time and any of the three "
        "dimensions of Space except that our consciousness moves along it. But "
        "this is the point I have been trying to make. It is only another way of "
        "looking at Time. There is no difference between Time and any of the "
        "three dimensions of Space except that our consciousness moves along it. "
        "All this is very clear now, but you have not done anything with it yet. "
        "You have not even begun to make a time machine."
    ),
}

VOICE_SET = ["af_heart", "af_bella", "am_michael", "bf_emma"]
DEFAULT_VOICE = "af_heart"
SPEED_SWEEP = [0.8, 1.0, 1.2]


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def rss_mb():
    return round(psutil.Process().memory_info().rss / 2**20, 1)


def parse_csv_list(value, cast=str):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument(
        "--dtype",
        choices=["fp32", "autocast"],
        default="fp32",
        help="fp32, or cuda autocast (mixed precision, weights stay fp32)",
    )
    p.add_argument("--tag", default=None, help="Config label; defaults to '<device>-<dtype>'")
    p.add_argument("--tf32", action="store_true", help="Allow TF32 matmul/cudnn")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    p.add_argument("--runs", default="4,4,3,3", help="timed runs for short,medium,long,xlong")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--voices", default=",".join(VOICE_SET))
    p.add_argument("--texts", default="short,medium,long,xlong")
    p.add_argument("--speed-sweep", action="store_true")
    p.add_argument("--no-audio", action="store_true", help="Do not write sample WAVs")
    return p.parse_args()


def env_meta(args):
    import kokoro

    meta = {
        "timestamp": now_iso(),
        "tag": args.tag,
        "device": args.device,
        "dtype": args.dtype,
        "tf32": bool(args.tf32),
        "threads": args.threads,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "torch": torch.__version__,
        "kokoro": getattr(kokoro, "__version__", "unknown"),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
    }
    return meta


def build_pipeline(args):
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    np.random.seed(0)

    from kokoro import KModel, KPipeline

    t0 = time.perf_counter()
    kmodel = KModel(
        repo_id=REPO_ID,
        config=str(MODEL_DIR / "config.json"),
        model=str(MODEL_DIR / "kokoro-v1_0.pth"),
    )
    if args.dtype == "autocast" and args.device != "cuda":
        raise ValueError("autocast benchmarking is only supported with --device cuda")
    global _AUTOCAST
    _AUTOCAST = args.dtype == "autocast"
    kmodel = kmodel.to(args.device)
    kmodel.eval()
    if args.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    load_s = round(time.perf_counter() - t0, 3)

    pipe = KPipeline(lang_code="a", repo_id=REPO_ID, model=kmodel)
    return pipe, kmodel, load_s


_AUTOCAST = False


def _autocast_enabled():
    return _AUTOCAST


def _nullcontext():
    import contextlib

    return contextlib.nullcontext()


_voice_cache = {}


def get_voice(name):
    if name not in _voice_cache:
        pack = torch.load(MODEL_DIR / "voices" / f"{name}.pt", weights_only=True)
        _voice_cache[name] = pack
    return _voice_cache[name]


def synth(pipe, text, voice_name, speed, device):
    pack = get_voice(voice_name)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    first_chunk_s = None
    chunks = []
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
    ctx = autocast if (device == "cuda" and _autocast_enabled()) else _nullcontext()
    with ctx:
        for result in pipe(text, voice=pack, speed=speed):
            if first_chunk_s is None:
                first_chunk_s = time.perf_counter() - t0
            if result.audio is not None:
                chunks.append(result.audio.numpy())
    if device == "cuda":
        torch.cuda.synchronize()
    total_s = time.perf_counter() - t0
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    peak_vram_mb = (
        round(torch.cuda.max_memory_allocated() / 2**20, 1) if device == "cuda" else None
    )
    return {
        "synth_s": round(total_s, 4),
        "first_chunk_s": round(first_chunk_s, 4) if first_chunk_s is not None else None,
        "n_chunks": len(chunks),
        "audio_s": round(len(audio) / SAMPLE_RATE, 4),
        "peak_vram_mb": peak_vram_mb,
        "rss_mb": rss_mb(),
        "audio": audio,
    }


def main():
    args = parse_args()
    if args.tag is None:
        args.tag = f"{args.device}-{args.dtype}"
    runs_by_text = dict(zip(["short", "medium", "long", "xlong"], parse_csv_list(args.runs, int)))
    voices = parse_csv_list(args.voices)
    text_ids = parse_csv_list(args.texts)
    audio_dir = AUDIO_ROOT / args.tag
    if not args.no_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{args.tag}] loading model from {MODEL_DIR} ...", flush=True)
    meta = env_meta(args)
    try:
        pipe, kmodel, load_s = build_pipeline(args)
    except Exception as exc:  # e.g. fp16 unsupported
        meta.update({"status": "build_failed", "error": repr(exc)})
        err_path = RESULTS_DIR / f"kokoro_{args.tag}_meta.json"
        err_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[{args.tag}] BUILD FAILED: {exc!r}", flush=True)
        raise SystemExit(2)

    meta["model_load_s"] = load_s
    meta["status"] = "ok"
    print(
        f"[{args.tag}] model loaded in {load_s}s | dtype={args.dtype} "
        f"| threads={torch.get_num_threads()}",
        flush=True,
    )

    rows = []
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    status = "ok"
    error = None

    def record(condition, voice, text_id, speed, run_type, run_idx, metrics, text_len):
        rows.append(
            {
                "tag": args.tag,
                "device": args.device,
                "dtype": args.dtype,
                "tf32": args.tf32,
                "threads": args.threads,
                "condition": condition,
                "voice": voice,
                "text_id": text_id,
                "speed": speed,
                "run_type": run_type,
                "run_idx": run_idx,
                "chars": text_len,
                "synth_s": metrics["synth_s"],
                "first_chunk_s": metrics["first_chunk_s"],
                "n_chunks": metrics["n_chunks"],
                "audio_s": metrics["audio_s"],
                "rtf": round(metrics["synth_s"] / metrics["audio_s"], 4)
                if metrics["audio_s"] > 0
                else None,
                "x_realtime": round(metrics["audio_s"] / metrics["synth_s"], 2)
                if metrics["synth_s"] > 0
                else None,
                "chars_per_s": round(text_len / metrics["synth_s"], 1)
                if metrics["synth_s"] > 0
                else None,
                "peak_vram_mb": metrics["peak_vram_mb"],
                "rss_mb": metrics["rss_mb"],
                "model_load_s": load_s,
                "timestamp": now_iso(),
            }
        )

    def run_condition(condition, voice, text_id, speed=1.0, n_runs=1, save=False):
        text = TEXTS[text_id]
        for i in range(args.warmup + n_runs):
            run_type = "warmup" if i < args.warmup else "timed"
            run_idx = i - args.warmup if run_type == "timed" else i
            m = synth(pipe, text, voice, speed, args.device)
            record(condition, voice, text_id, speed, run_type, run_idx, m, len(text))
            if save and run_type == "timed" and run_idx == 0 and not args.no_audio:
                sf.write(
                    audio_dir / f"{condition}_{voice}_{text_id}_sp{speed}.wav",
                    m["audio"],
                    SAMPLE_RATE,
                )
            print(
                f"[{args.tag}] {condition:10s} voice={voice:10s} text={text_id:6s} "
                f"speed={speed} {run_type}[{run_idx}] "
                f"synth={m['synth_s']:.3f}s audio={m['audio_s']:.2f}s "
                f"rtf={m['synth_s']/m['audio_s'] if m['audio_s'] else float('nan'):.3f} "
                f"first={m['first_chunk_s']:.3f}s"
                + (f" vram={m['peak_vram_mb']}MB" if m["peak_vram_mb"] else ""),
                flush=True,
            )

    try:
        for text_id in text_ids:
            run_condition("length", DEFAULT_VOICE, text_id, n_runs=runs_by_text.get(text_id, 2), save=True)

        for voice in voices:
            run_condition("voice", voice, "medium", n_runs=runs_by_text.get("medium", 2), save=True)

        if args.speed_sweep:
            for speed in SPEED_SWEEP:
                run_condition("speed", DEFAULT_VOICE, "medium", speed=speed, n_runs=2, save=True)
    except Exception as exc:
        status = "error"
        error = repr(exc)
        print(f"[{args.tag}] ERROR during suite: {error}", flush=True)

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / f"kokoro_{args.tag}_{ts}.csv"
    df.to_csv(csv_path, index=False)
    meta["status"] = status
    if error:
        meta["error"] = error
    meta_path = RESULTS_DIR / f"kokoro_{args.tag}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[{args.tag}] wrote {csv_path}", flush=True)
    print(f"[{args.tag}] wrote {meta_path}", flush=True)

    timed = df[df.run_type == "timed"]
    if not timed.empty:
        summary = (
            timed.groupby("condition")[["rtf", "x_realtime", "first_chunk_s", "synth_s"]]
            .mean()
            .round(3)
        )
        print(f"\n[{args.tag}] mean timed-run summary:\n{summary.to_string()}\n", flush=True)


if __name__ == "__main__":
    main()
