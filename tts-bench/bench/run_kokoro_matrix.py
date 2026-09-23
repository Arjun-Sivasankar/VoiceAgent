"""Run the Kokoro benchmark matrix across runtime configs and aggregate results.

Configs (can be filtered with --configs):
    cuda-fp32, cuda-fp32-tf32, cuda-fp16, cpu, cpu-1t, cpu-4t

Each config runs bench_kokoro.py in its own process (clean memory state).
Aggregated CSV -> results/kokoro_matrix_<ts>.csv

Examples:
    python run_kokoro_matrix.py
    python run_kokoro_matrix.py --configs cuda-fp32,cuda-fp16
    python run_kokoro_matrix.py --quick
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench" / "bench_kokoro.py"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    "cuda-fp32": dict(device="cuda", dtype="fp32", runs="4,4,3,3", voices="af_heart,af_bella,am_michael,bf_emma"),
    "cuda-fp32-tf32": dict(device="cuda", dtype="fp32", tf32=True, runs="4,4,3,3", voices="af_heart,af_bella,am_michael,bf_emma"),
    "cuda-fp16": dict(device="cuda", dtype="autocast", runs="4,4,3,3", voices="af_heart,af_bella,am_michael,bf_emma"),
    "cpu": dict(device="cpu", dtype="fp32", runs="1,1,1,1", voices="af_heart"),
    "cpu-4t": dict(device="cpu", dtype="fp32", threads=4, runs="1,1,1,1", voices="af_heart"),
}

QUICK = {
    "cuda-fp32": dict(device="cuda", dtype="fp32", runs="2,2,1,1", voices="af_heart"),
    "cuda-fp16": dict(device="cuda", dtype="autocast", runs="2,2,1,1", voices="af_heart"),
    "cpu": dict(device="cpu", dtype="fp32", runs="1,1,1,1", voices="af_heart"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configs", default=None, help="Comma list, e.g. cuda-fp32,cpu")
    p.add_argument("--quick", action="store_true", help="Tiny smoke matrix")
    p.add_argument("--speed-sweep", action="store_true")
    p.add_argument("--keep-going", action="store_true", help="Continue after a config fails")
    return p.parse_args()


def main():
    args = parse_args()
    pool = QUICK if args.quick else CONFIGS
    names = list(pool.keys())
    if args.configs:
        names = [n.strip() for n in args.configs.split(",") if n.strip()]
        unknown = [n for n in names if n not in pool]
        if unknown:
            print(f"Unknown configs: {unknown}; available: {list(pool)}")
            raise SystemExit(2)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    produced = []
    for name in names:
        cfg = dict(pool[name])
        cmd = [sys.executable, str(BENCH), "--tag", name]
        for key, value in cfg.items():
            if key in ("tf32",):
                if value:
                    cmd.append("--tf32")
            elif key == "threads":
                cmd += ["--threads", str(value)]
            else:
                cmd += [f"--{key}", str(value)]
        if args.speed_sweep:
            cmd.append("--speed-sweep")
        print(f"\n===== [{name}] {' '.join(cmd[1:])} =====", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(ROOT))
        dt = time.perf_counter() - t0
        print(f"===== [{name}] exit={proc.returncode} in {dt:.1f}s =====\n", flush=True)
        if proc.returncode != 0:
            print(f"[{name}] FAILED (exit {proc.returncode})", flush=True)
            if not args.keep_going:
                raise SystemExit(proc.returncode)
            continue
        produced.append(name)

    if not produced:
        print("No configs produced results.")
        raise SystemExit(2)

    csvs = []
    for name in produced:
        matches = sorted(RESULTS_DIR.glob(f"kokoro_{name}_*.csv"))
        if matches:
            csvs.append(matches[-1])
    matrix = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    out = RESULTS_DIR / f"kokoro_matrix_{ts}.csv"
    matrix.to_csv(out, index=False)
    print(f"\nAggregated matrix -> {out}", flush=True)

    timed = matrix[matrix.run_type == "timed"]
    summary = (
        timed.groupby(["tag", "condition"])[
            ["rtf", "x_realtime", "first_chunk_s", "synth_s", "peak_vram_mb"]
        ]
        .mean()
        .round(3)
    )
    print("\nMean timed-run summary:\n" + summary.to_string(), flush=True)
    timed.to_csv(RESULTS_DIR / f"kokoro_matrix_timed_{ts}.csv", index=False)


if __name__ == "__main__":
    main()
