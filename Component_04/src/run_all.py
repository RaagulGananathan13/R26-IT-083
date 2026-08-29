"""
Component 04 — one-command pipeline.

    python src/run_all.py                 # everything, primary horizon only
    python src/run_all.py --all-horizons  # train every horizon (progressive study)
    python src/run_all.py --skip audit,ablations
    python src/run_all.py --from train_stage1

Each stage is idempotent: rerunning skips work whose artefacts already exist
unless --force is given.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_p for _p in (
    _SRC, os.path.join(_SRC, "core"), os.path.join(_SRC, "data"),
    os.path.join(_SRC, "models"), os.path.join(_SRC, "analysis"),
) if _p not in sys.path]

from config import CFG, DATA_DIR, MODEL_DIR, REPORT_DIR, enable_utf8_stdout
from utils import banner, kv

enable_utf8_stdout()

STAGES = ["preprocess", "split", "audit", "train_stage1", "train_stage2",
          "evaluate", "unified4", "selective", "final_report", "explain",
          "ablations"]


def _exists(*paths) -> bool:
    return all(os.path.exists(p) for p in paths)


def run(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Component 04 pipeline")
    ap.add_argument("--skip", default="", help="comma-separated stages to skip")
    ap.add_argument("--only", default="", help="comma-separated stages to run")
    ap.add_argument("--from", dest="start", default=None, help="start at this stage")
    ap.add_argument("--all-horizons", action="store_true",
                    help="train/evaluate every horizon in config, not just the primary")
    ap.add_argument("--force", action="store_true", help="ignore existing artefacts")
    args = ap.parse_args(argv)

    stages = list(STAGES)
    if args.start:
        stages = stages[stages.index(args.start):]
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        stages = [s for s in stages if s in want]
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    stages = [s for s in stages if s not in skip]

    horizons = CFG.horizons if args.all_horizons else [CFG.primary_horizon]

    banner("COMPONENT 04 — FULL PIPELINE")
    kv("stages", " -> ".join(stages))
    kv("horizons", horizons)
    kv("primary horizon", CFG.primary_horizon)
    kv("raw data", CFG.raw_dir)

    t_all = time.time()
    timings = {}

    for stage in stages:
        banner(f"[{stages.index(stage)+1}/{len(stages)}]  {stage}", ch="~")
        t0 = time.time()
        try:
            if stage == "preprocess":
                done = all(_exists(os.path.join(DATA_DIR, f"features_H{h}.parquet"))
                           for h in CFG.horizons)
                if done and not args.force:
                    kv("status", "already built — skipping (use --force to rebuild)")
                else:
                    import preprocess; preprocess.main()

            elif stage == "split":
                if _exists(os.path.join(DATA_DIR, "split_assignment.parquet")) and not args.force:
                    kv("status", "already built — skipping")
                else:
                    import split; split.main()

            elif stage == "audit":
                import audit_leakage; audit_leakage.main()

            elif stage == "train_stage1":
                import train_stage1
                for h in horizons:
                    if _exists(os.path.join(MODEL_DIR, f"stage1_config_H{h}.json")) \
                            and not args.force:
                        kv(f"H={h}", "already trained — skipping")
                    else:
                        train_stage1.main(h)

            elif stage == "train_stage2":
                import train_stage2
                for h in horizons:
                    if _exists(os.path.join(MODEL_DIR, f"stage2_config_H{h}.json")) \
                            and not args.force:
                        kv(f"H={h}", "already trained — skipping")
                    else:
                        train_stage2.main(h)

            elif stage == "evaluate":
                import evaluate; evaluate.main()

            elif stage == "unified4":
                # UM4 produces the headline all-classes->=75% result; without it
                # a rerun reproduces only the weaker two-stage cascade.
                import unified4; unified4.main()

            elif stage == "selective":
                import selective; selective.main()

            elif stage == "final_report":
                import final_report; final_report.main()

            elif stage == "explain":
                import explain
                for h in horizons:
                    explain.main(h)

            elif stage == "ablations":
                import ablations; ablations.main()

            timings[stage] = time.time() - t0
        except Exception:
            print(f"\n  !! stage '{stage}' failed:\n")
            traceback.print_exc()
            timings[stage] = -1.0
            if stage in ("preprocess", "split"):
                print("\n  This stage is a hard dependency — stopping.")
                return 1

    banner("PIPELINE COMPLETE")
    for s, t in timings.items():
        kv(s, "FAILED" if t < 0 else f"{t/60:.1f} min")
    kv("TOTAL", f"{(time.time()-t_all)/60:.1f} min")
    print(f"\n  Reports : {REPORT_DIR}")
    print(f"  Models  : {MODEL_DIR}")
    print(f"  Figures : {os.path.join(os.path.dirname(REPORT_DIR), 'figures')}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
