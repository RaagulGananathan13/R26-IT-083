"""
Backbone ablation: does R(2+1)D's factorisation actually help on echocardiography?

WHY THIS EXISTS
---------------
The R(2+1)D-18 backbone was inherited from the EchoNet-Dynamic benchmark
(Ouyang et al., Nature Medicine 2020), whose ablation found it best for EF
regression. That is a good reason to start there, but it is someone else's
result on someone else's setup. This script tests it here, with our four-head
ordinal formulation, our CAMUS co-training and our calibration.

R(2+1)D factorises each 3-D convolution into a spatial (1,d,d) step followed by
a temporal (t,1,1) step. r3d_18 is the un-factorised baseline it was designed to
beat (Tran et al., CVPR 2018), and is nearly matched in capacity -- 33.2 M
against 31.3 M -- so the comparison isolates the factorisation rather than
confounding it with model size.

THE PROTOCOL
------------
One variable changes: the backbone. Same seed, same epoch schedule, same data,
same heads, same loss, same calibration procedure.

The comparison is SINGLE MODEL vs SINGLE MODEL:

    uefnet_v3  (R(2+1)D, seed 1337)   MAE 4.138, min-recall 0.687
    uefnet_r3d (R3D,     seed 1337)   MAE 4.130, min-recall 0.651

Neither difference is significant: paired bootstrap over the same 1,277 studies
gives p = 0.889 on MAE and p = 0.217 on accuracy, and exact McNemar over 158
discordant studies gives p = 0.233. The architectures are indistinguishable here.

NOT against the three-seed ensemble headline (MAE 3.979, min-recall 0.723).
Comparing one model to an ensemble would read a variance-reduction effect as an
architecture effect, which is the one mistake this experiment must not make.

COST
----
Measured end-to-end on an RTX 4060 Laptop with AMP: the r3d_18 run took 4.02 h
against r2plus1d_18's 23.5 h for the same 45-epoch schedule.

GPU-only, at the shipped input geometry (8 x 2 x 32 x 112 x 112), r3d_18 runs at
0.262 s/step and 1.86 GB peak against r2plus1d_18's 1.840 s and 4.22 GB. The
end-to-end ratio (~6x) is smaller than the per-step ratio (~7x) because at
0.262 s/step r3d_18 outruns the four-worker data pipeline and idles waiting for
clips -- the factorised model is GPU-bound, the un-factorised one is not.

USAGE
-----
    python run_backbone_ablation.py                    # train, then compare
    python run_backbone_ablation.py --resume           # continue after a stop
    python run_backbone_ablation.py --compare-only     # skip training
    python run_backbone_ablation.py --smoke            # wiring test, minutes

Interrupting is safe: --resume restores the optimizer, epoch, best score,
patience, EMA and RNG state from last.pt.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAIN_DIR))

#: The incumbent. Its single-seed figures are the comparison baseline.
BASELINE_RUN = "uefnet_v3"
#: The challenger. Same seed so the only difference is the architecture.
ABLATION_RUN = "uefnet_r3d"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", default="r3d_18",
                        choices=["r3d_18", "mc3_18"],
                        help="challenger architecture (default: r3d_18)")
    parser.add_argument("--run-name", default=None,
                        help="output run name (default: uefnet_r3d / uefnet_mc3)")
    parser.add_argument("--baseline", default=BASELINE_RUN,
                        help="run to compare against (default: uefnet_v3)")
    parser.add_argument("--seed", type=int, default=None,
                        help="default: the baseline run's seed, so it is matched")
    parser.add_argument("--epochs", type=int, default=None,
                        help="default: the baseline run's epoch count")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="default: the baseline's, which keeps the RNG stream matched")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compare-only", action="store_true",
                        help="skip training; both runs must already exist")
    parser.add_argument("--smoke", action="store_true",
                        help="few-iteration wiring test")
    return parser.parse_args()


def load_snapshot(run: str) -> dict:
    path = TRAIN_DIR / "outputs" / run / "config.json"
    if not path.exists():
        sys.exit("[ablation] baseline config not found: %s\n"
                 "           Train %s first, or pass --baseline." % (path, run))
    return json.loads(path.read_text(encoding="utf-8"))


def run(cmd: list[str]) -> int:
    """Run a child process, streaming its output so tqdm stays live."""
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(TRAIN_DIR))


def read_report(run_name: str) -> dict | None:
    path = TRAIN_DIR / "outputs" / run_name / "test_report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarise(report: dict | None) -> dict:
    if not report:
        return {}
    regression = report.get("regression", {}) or {}
    classification = report.get("classification", {}) or {}
    return {
        "MAE": regression.get("mae"),
        "R2": regression.get("r2"),
        "min-class recall": classification.get("min_class_recall"),
        "balanced accuracy": classification.get("balanced_acc"),
        "overall accuracy": classification.get("overall_acc"),
        "macro F1": classification.get("macro_f1"),
        "within-one-class": classification.get("within_one_class_acc"),
    }


def compare(baseline: str, challenger: str) -> None:
    left, right = summarise(read_report(baseline)), summarise(read_report(challenger))
    if not left or not right:
        missing = baseline if not left else challenger
        print("\n[ablation] no test_report.json for %r -- run run_eval.py for it first."
              % missing)
        return

    print("\n" + "=" * 74)
    print("  BACKBONE ABLATION -- single model vs single model, matched seed")
    print("=" * 74)
    print("  %-22s %14s %14s %12s" % ("metric", baseline, challenger, "delta"))
    print("  " + "-" * 66)
    # Metrics where a LOWER value is better.
    lower_better = {"MAE"}
    for key in left:
        a, b = left.get(key), right.get(key)
        if a is None or b is None:
            continue
        delta = b - a
        better = (delta < 0) if key in lower_better else (delta > 0)
        mark = "  better" if abs(delta) > 1e-9 and better else (
            "  worse" if abs(delta) > 1e-9 else "  same")
        print("  %-22s %14.4f %14.4f %+12.4f%s" % (key, a, b, delta, mark))
    print("=" * 74)
    print("  Read this against the SINGLE-seed baseline, not the three-seed")
    print("  ensemble headline. A difference here is an architecture effect;")
    print("  the ensemble figure also contains variance reduction.")
    print()
    print("  For a paired significance test on the same studies, save per-study")
    print("  predictions once per system, then compare them:")
    print("    python run_ensemble.py --runs %s --n-tta 10 --save-predictions \\"
          % baseline)
    print("        --out outputs/single_%s.json" % baseline)
    print("    python run_ensemble.py --runs %s --n-tta 10 --save-predictions \\"
          % challenger)
    print("        --out outputs/single_%s.json" % challenger)
    print("    python run_robustness.py \\")
    print("        --predictions outputs/predictions_test_%s.npz \\" % challenger)
    print("        --compare-with outputs/predictions_test_%s.npz" % baseline)
    print("=" * 74)


def main() -> int:
    args = parse_args()
    challenger = args.run_name or (
        "uefnet_r3d" if args.backbone == "r3d_18" else "uefnet_mc3")

    snapshot = load_snapshot(args.baseline)
    seed = args.seed if args.seed is not None else snapshot.get("seed", 1337)
    epochs = args.epochs if args.epochs is not None else snapshot.get("epochs", 45)

    print("=" * 74)
    print("  BACKBONE ABLATION")
    print("=" * 74)
    print("  baseline    : %s  (%s, seed %s)"
          % (args.baseline, snapshot.get("backbone"), snapshot.get("seed")))
    print("  challenger  : %s  (%s, seed %s)" % (challenger, args.backbone, seed))
    print("  epochs      : %s" % epochs)
    print("  matched     : every hyperparameter recorded in the baseline snapshot")
    print("  varied      : backbone only")
    if not args.compare_only and not args.smoke:
        # Measured wall-clock per epoch: 4.02 h / 45 for r3d_18, 23.5 h / 45
        # for r2plus1d_18. mc3_18 is unmeasured; r3d_18 is the closest proxy.
        per_epoch = 0.089 if args.backbone in ("r3d_18", "mc3_18") else 0.52
        hours = epochs * per_epoch
        print("  estimated   : ~%.0f h. Safe to interrupt; re-run with --resume." % hours)
    print("=" * 74)

    if not args.compare_only:
        # Every hyperparameter is inherited from the baseline snapshot via the
        # shared table. Listing flags by hand here is exactly how the first
        # version of this ablation silently let logit_adjustment_tau and
        # n_tta_clips fall back to config.py defaults the baseline had
        # overridden, making the comparison two-variable without saying so.
        from run_backbone_ensemble import train_command
        cmd = train_command(challenger, args.backbone, seed, snapshot,
                            args.num_workers, args.smoke)
        if epochs != snapshot.get("epochs"):
            cmd += ["--epochs", str(epochs)]      # explicit override wins
        if args.resume:
            # On resume the snapshot supplies backbone/seed/epochs, and passing
            # them again would trip run_train's change guards.
            cmd = [sys.executable, "run_train.py", "--run-name", challenger,
                   "--resume", "--num-workers", str(args.num_workers)]
        if args.smoke:
            cmd.append("--smoke")

        started = time.perf_counter()
        code = run(cmd)
        if code != 0:
            print("\n[ablation] training exited with code %d. Re-run with --resume "
                  "to continue from last.pt." % code)
            return code
        print("\n[ablation] training finished in %.2f h"
              % ((time.perf_counter() - started) / 3600))

        if run([sys.executable, "run_eval.py", "--run-name", challenger]) != 0:
            print("[ablation] evaluation failed; run run_eval.py manually.")
            return 1

    # Fail closed: a comparison is only a backbone ablation if the backbone is
    # the only thing that changed. Check before printing numbers, not after.
    from run_backbone_ensemble import audit_parity, rule
    print("\n" + rule())
    print("  CONFIG PARITY AUDIT")
    print(rule())
    parity_ok, lines = audit_parity(args.baseline, [challenger])
    print("\n".join(lines))
    print(rule())
    if not parity_ok:
        print("  FAIL -- these runs differ in more than the backbone, so this is")
        print("  not a backbone ablation. Retrain the challenger with the")
        print("  inherited configuration, or state the confound wherever you")
        print("  quote the result.")
        print(rule())
        return 2
    print("  PASS -- only the backbone and seed differ.")
    print(rule())

    compare(args.baseline, challenger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
