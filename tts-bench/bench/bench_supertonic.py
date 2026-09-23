"""Systematic Supertonic-3 benchmark rig (ONNX Runtime, 99M params, 44.1 kHz).

Configs: CPU (official default provider) and optional CUDAExecutionProvider.
Sweeps: text length, diffusion steps, all 10 voices, 10 languages, batch size.
Model is loaded fully from local files in models/supertonic-3 (no network).

Examples:
    python bench_supertonic.py --provider cpu --tag st3-cpu
    python bench_supertonic.py --provider cuda --tag st3-cuda --steps 8,16
    python bench_supertonic.py --provider cpu --threads 4 --tag st3-cpu-4t --quick
"""

import argparse
import json
import platform
import threading
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "supertonic-3"
AUDIO_ROOT = ROOT / "audio" / "supertonic-3"
RESULTS_DIR = ROOT / "results"
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
        "new paradox, as it seemed, and his fecundity. "
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
        "Yet a four dimensional space is a very simple matter. "
        "The Time Traveller smiled. Are you sure we can move freely in Space? "
        "Right and left we can go, backward and forward we can go, and men have "
        "always gone up and down. I have been a student of mathematics for some "
        "years, and I have never heard of any suggestion of such a thing until "
        "now. Is it not possible that the geometry of four dimensions has been "
        "sketched out? There is no difference between Time and any of the three "
        "dimensions of Space except that our consciousness moves along it. But "
        "this is the point I have been trying to make. It is only another way of "
        "looking at Time. All this is very clear now, but you have not done "
        "anything with it yet. You have not even begun to make a time machine."
    ),
}

LANG_TEXTS = {
    "en": "The quick brown fox jumps over the lazy dog.",
    "ko": "안녕하세요. 오늘 날씨가 정말 좋네요.",
    "ja": "こんにちは。今日はとても良い天気ですね。",
    "de": "Guten Tag. Das Wetter ist heute wirklich schön.",
    "es": "Hola. El clima está muy agradable hoy.",
    "fr": "Bonjour. Le temps est très agréable aujourd'hui.",
    "hi": "नमस्ते। आज मौसम बहुत अच्छा है।",
    "ru": "Здравствуйте. Сегодня очень хорошая погода.",
    "pt": "Olá. O tempo está muito agradável hoje.",
    "it": "Ciao. Il tempo è davvero bello oggi.",
}

VOICES = [f"{g}{i}" for g in ("M", "F") for i in range(1, 6)]
STEPS_SWEEP = [4, 8, 16, 32]


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def rss_mb():
    return round(psutil.Process().memory_info().rss / 2**20, 1)


def gpu_used_mb():
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return round(pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**20, 1)
    except Exception:
        return None


class VramSampler:
    def __init__(self, interval=0.02):
        self.interval = interval
        self.peak = None
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            used = gpu_used_mb()
            if used is not None:
                self.peak = used if self.peak is None else max(self.peak, used)
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)


_tts = None
_style = None


def build(provider, threads):
    import os

    import torch

    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(torch_lib):
        os.add_dll_directory(torch_lib)
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ["PATH"]

    import supertonic.loader as loader

    if provider == "cuda":
        loader.DEFAULT_ONNX_PROVIDERS = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    from supertonic import TTS

    t0 = time.perf_counter()
    tts = TTS(
        model="supertonic-3",
        model_dir=str(MODEL_DIR),
        auto_download=False,
        intra_op_num_threads=threads or None,
    )
    load_s = round(time.perf_counter() - t0, 3)
    tts_session_providers = list(tts.model.dp_ort.get_providers())
    return tts, load_s, tts_session_providers


def parse_csv_list(value, cast=str):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--tag", default=None)
    p.add_argument("--threads", type=int, default=0, help="ORT intra-op threads (0 = auto)")
    p.add_argument("--runs", default="3,3,2,1", help="timed runs for short,medium,long,xlong")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--steps", default=",".join(map(str, STEPS_SWEEP)))
    p.add_argument("--voices", default=",".join(VOICES))
    p.add_argument("--langs", default="en,ko,ja,de,es,fr,hi,ru,pt,it")
    p.add_argument("--batch-sizes", default="1,2,4,8")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-audio", action="store_true")
    return p.parse_args()


def env_meta(args):
    import onnxruntime as ort
    import supertonic

    return {
        "timestamp": now_iso(),
        "tag": args.tag,
        "provider": args.provider,
        "threads": args.threads,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "supertonic": supertonic.__version__,
        "onnxruntime": ort.__version__,
        "ort_providers_available": ort.get_available_providers(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
    }


def main():
    args = parse_args()
    if args.tag is None:
        args.tag = f"st3-{args.provider}" + (f"-{args.threads}t" if args.threads else "")
    runs_by_text = dict(zip(["short", "medium", "long", "xlong"], parse_csv_list(args.runs, int)))
    audio_dir = AUDIO_ROOT / args.tag
    if not args.no_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)
    if args.quick:
        args.steps = "8"
        args.voices = "M1,F1"
        args.langs = "en,ko,de"
        args.batch_sizes = "1,4"
        runs_by_text = {"short": 1, "medium": 1, "long": 1, "xlong": 1}

    print(f"[{args.tag}] loading Supertonic-3 from {MODEL_DIR} (provider={args.provider}) ...", flush=True)
    meta = env_meta(args)
    tts, load_s, tts_session_providers = build(args.provider, args.threads)
    meta["model_load_s"] = load_s
    meta["ort_session_providers"] = tts_session_providers
    meta["sample_rate"] = tts.sample_rate
    meta["voice_styles"] = tts.voice_style_names
    vram_after_load = gpu_used_mb() if args.provider == "cuda" else None
    meta["vram_after_load_mb"] = vram_after_load
    print(
        f"[{args.tag}] loaded in {load_s}s | sr={tts.sample_rate} | voices={tts.voice_style_names} "
        f"| ort sessions={tts_session_providers}",
        flush=True,
    )

    rows = []
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    style = tts.get_voice_style("M1")
    cache = {"M1": style}

    def get_style(name):
        if name not in cache:
            cache[name] = tts.get_voice_style(name)
        return cache[name]

    def synth(text, voice="M1", steps=8, speed=1.05, lang="en"):
        use_vram = args.provider == "cuda"
        sampler = VramSampler(0.05) if use_vram else None
        ctx = sampler if use_vram else nullcontext()
        with ctx:
            t0 = time.perf_counter()
            wav, dur = tts.synthesize(
                text, voice_style=get_style(voice), total_steps=steps, speed=speed, lang=lang
            )
            synth_s = time.perf_counter() - t0
        return {
            "synth_s": round(synth_s, 4),
            "audio_s": round(wav.shape[-1] / tts.sample_rate, 4),
            "peak_vram_mb": sampler.peak if use_vram else None,
            "rss_mb": rss_mb(),
            "wav": wav,
        }

    def record(condition, voice, text_id, steps, speed, lang, run_type, run_idx, m, text_len):
        rows.append(
            {
                "tag": args.tag,
                "provider": args.provider,
                "threads": args.threads,
                "condition": condition,
                "voice": voice,
                "text_id": text_id,
                "steps": steps,
                "speed": speed,
                "lang": lang,
                "run_type": run_type,
                "run_idx": run_idx,
                "chars": text_len,
                "synth_s": m["synth_s"],
                "audio_s": m["audio_s"],
                "rtf": round(m["synth_s"] / m["audio_s"], 4) if m["audio_s"] > 0 else None,
                "x_realtime": round(m["audio_s"] / m["synth_s"], 2) if m["synth_s"] > 0 else None,
                "chars_per_s": round(text_len / m["synth_s"], 1) if m["synth_s"] > 0 else None,
                "peak_vram_mb": m["peak_vram_mb"],
                "rss_mb": m["rss_mb"],
                "model_load_s": load_s,
                "timestamp": now_iso(),
            }
        )

    def run_condition(condition, text, voice="M1", steps=8, speed=1.05, lang="en", n_runs=2, save=False, text_id=None):
        for i in range(args.warmup + n_runs):
            run_type = "warmup" if i < args.warmup else "timed"
            run_idx = i - args.warmup if run_type == "timed" else i
            m = synth(text, voice=voice, steps=steps, speed=speed, lang=lang)
            record(condition, voice, text_id or condition, steps, speed, lang, run_type, run_idx, m, len(text))
            if save and run_type == "timed" and run_idx == 0 and not args.no_audio:
                name = f"{condition}_{voice}_{text_id or condition}_s{steps}_{lang}.wav"
                sf.write(audio_dir / name, m["wav"].squeeze(), tts.sample_rate)
            print(
                f"[{args.tag}] {condition:9s} voice={voice:3s} text={str(text_id):7s} steps={steps:2d} "
                f"lang={lang:2s} {run_type}[{run_idx}] synth={m['synth_s']:.3f}s "
                f"audio={m['audio_s']:.2f}s rtf={m['synth_s']/m['audio_s'] if m['audio_s'] else float('nan'):.3f} "
                f"xrt={m['audio_s']/m['synth_s'] if m['synth_s'] else float('nan'):.1f}"
                + (f" vram={m['peak_vram_mb']}MB" if m["peak_vram_mb"] else ""),
                flush=True,
            )

    status, error = "ok", None
    try:
        for text_id in ("short", "medium", "long", "xlong"):
            run_condition("length", TEXTS[text_id], text_id=text_id, n_runs=runs_by_text.get(text_id, 2), save=True)

        for steps in parse_csv_list(args.steps, int):
            run_condition("steps", TEXTS["medium"], steps=steps, n_runs=2, text_id=f"medium-s{steps}", save=(steps == 8))

        for voice in parse_csv_list(args.voices):
            run_condition("voice", TEXTS["medium"], voice=voice, n_runs=2, text_id="medium", save=(voice in ("M1", "F1")))

        for lang in parse_csv_list(args.langs):
            text = LANG_TEXTS.get(lang)
            if not text:
                continue
            try:
                run_condition("lang", text, lang=lang, n_runs=2, text_id=lang, save=True)
            except Exception as exc:
                print(f"[{args.tag}] lang={lang} failed: {exc!r}", flush=True)

        for bsz in parse_csv_list(args.batch_sizes, int):
            from supertonic.core import Style

            text = TEXTS["medium"]
            texts = [text] * bsz
            base_style = get_style("M1")
            batch_style = Style(
                np.repeat(base_style.ttl, bsz, axis=0),
                np.repeat(base_style.dp, bsz, axis=0),
            )
            for i in range(1 + 2):
                run_type = "warmup" if i < 1 else "timed"
                run_idx = i - 1 if run_type == "timed" else i
                use_vram = args.provider == "cuda"
                sampler = VramSampler(0.05) if use_vram else None
                ctx = sampler if use_vram else nullcontext()
                with ctx:
                    t0 = time.perf_counter()
                    wav, dur = tts.model(texts, batch_style, 8, 1.05, "en")
                    synth_s = time.perf_counter() - t0
                audio_s = sum(dur) if hasattr(dur, "__len__") else float(dur)
                m = {
                    "synth_s": round(synth_s, 4),
                    "audio_s": round(float(audio_s), 4),
                    "peak_vram_mb": sampler.peak if use_vram else None,
                    "rss_mb": rss_mb(),
                    "wav": wav,
                }
                record("batch", "M1", f"medium-b{bsz}", 8, 1.05, "en", run_type, run_idx, m, len(text) * bsz)
                print(
                    f"[{args.tag}] batch     bsz={bsz} {run_type}[{run_idx}] synth={m['synth_s']:.3f}s "
                    f"audio={m['audio_s']:.2f}s rtf={m['synth_s']/m['audio_s']:.3f} "
                    f"xrt={m['audio_s']/m['synth_s']:.1f}",
                    flush=True,
                )
    except Exception as exc:
        status, error = "error", repr(exc)
        print(f"[{args.tag}] ERROR: {error}", flush=True)

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / f"supertonic3_{args.tag}_{ts}.csv"
    df.to_csv(csv_path, index=False)
    meta["status"] = status
    if error:
        meta["error"] = error
    meta_path = RESULTS_DIR / f"supertonic3_{args.tag}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[{args.tag}] wrote {csv_path}", flush=True)

    timed = df[df.run_type == "timed"]
    if not timed.empty:
        summary = (
            timed.groupby("condition")[["rtf", "x_realtime", "synth_s"]].mean().round(3)
        )
        print(f"\n[{args.tag}] mean timed-run summary:\n{summary.to_string()}\n", flush=True)


if __name__ == "__main__":
    main()
