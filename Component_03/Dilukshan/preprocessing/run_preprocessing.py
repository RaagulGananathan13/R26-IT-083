"""
ONE-COMMAND ORCHESTRATOR for the full preprocessing pipeline.
Runs stages 0 -> 5 in dependency order and stops on the first failure.

    python run_preprocessing.py                 # full run, default settings
    python run_preprocessing.py --limit 50      # quick smoke test on 50 videos
    python run_preprocessing.py --denoise median --compress
    python run_preprocessing.py --skip 0,3      # skip stages 0 and 3

Stage dependency graph:
    0 audit  -> 1 labels
    2 keyframes -> 1 labels
    3 norm(estimate) -> 4 cache(refine) -> 5 verify
"""
from __future__ import annotations
import os, sys, time, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

STAGES = [
    ("0", "stage0_audit.py",      []),
    ("2", "stage2_keyframes.py",  []),
    ("1", "stage1_labels.py",     []),
    ("3", "stage3_norm_stats.py", []),
    ("4", "stage4_cache_clips.py", ["--denoise", "--compress", "--max-frames",
                                    "--workers", "--limit", "--resume"]),
    ("5", "stage5_verify.py",     ["--limit"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", default="", help="comma-separated stage ids to skip")
    ap.add_argument("--only", default="", help="comma-separated stage ids to run")
    ap.add_argument("--denoise", default=None)
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    t0 = time.time()
    for sid, script, accepts in STAGES:
        if sid in skip:
            print(f"\n>>> SKIP stage {sid} ({script})"); continue
        if only and sid not in only:
            continue
        cmd = [PY, os.path.join(HERE, script)]
        if "--denoise" in accepts and args.denoise:      cmd += ["--denoise", args.denoise]
        if "--compress" in accepts and args.compress:    cmd += ["--compress"]
        if "--max-frames" in accepts and args.max_frames is not None:
            cmd += ["--max-frames", str(args.max_frames)]
        if "--workers" in accepts and args.workers is not None:
            cmd += ["--workers", str(args.workers)]
        if "--limit" in accepts and args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        if "--resume" in accepts and args.resume:        cmd += ["--resume"]

        print(f"\n{'='*70}\n>>> RUN stage {sid}: {' '.join(cmd)}\n{'='*70}")
        rc = subprocess.call(cmd, cwd=HERE)
        if rc != 0:
            sys.exit(f"[orchestrator] stage {sid} FAILED (exit {rc}). Stopping.")

    print(f"\n[orchestrator] ALL STAGES COMPLETE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
