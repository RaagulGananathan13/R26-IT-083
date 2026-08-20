"""
Selective prediction evaluation for UEF-Net.
============================================

Grades the studies the model is confident about and defers the rest for
specialist review, then reports recall together with coverage.

    python run_selective.py --runs uefnet_v3 uefnet_v3b uefnet_v3c --n-tta 10

Protocol
--------
1. Ensemble predictions are formed on VAL and TEST with multi-clip TTA.
2. The decision rule (EF calibration + thresholds) is fitted on VAL, as usual.
3. The selective rule -- which uncertainty signal, and what cut-off -- is also
   fitted on VAL, choosing the highest coverage at which every class reaches the
   target recall.
4. Both frozen rules are applied once to TEST.

The test split takes no part in any fitting step.  Coverage is reported beside
every recall figure; a recall quoted without its coverage would be meaningless.
"""
from __future__ import annotations
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from config import CFG


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run names to ensemble")
    ap.add_argument("--n-tta", type=int, default=10)
    ap.add_argument("--target", type=float, default=0.75,
                    help="per-class recall target the selective rule aims for")
    ap.add_argument("--min-coverage", type=float, default=0.60,
                    help="lowest coverage the rule is allowed to fall to")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--plot", action="store_true", help="write the coverage-recall figure")
    return ap.parse_args()


def print_confusion(cm, names):
    cm = np.asarray(cm)
    w = max(len(n) for n in names) + 2
    print(" " * (w + 2) + "".join(f"{n[:8]:>10}" for n in names) + "   (pred)")
    for i, n in enumerate(names):
        print(f"{n:>{w}}  " + "".join(f"{cm[i, j]:>10}" for j in range(len(names))))
    print("  (true, rows)")


def main():
    a = parse_args()
    from run_ensemble import run_predictions, average_predictions
    from engine.calibrate import calibrate, apply_frozen_strategy
    from engine.selective import (fit_selective_rule, apply_selective_rule,
                                  uncertainty_signals, coverage_curve)
    from engine.metrics import classification_metrics, regression_metrics
    from core.common import write_json_atomic

    # ---------------------------------------------------------------- VAL
    print(f"[selective] gathering VAL predictions for {len(a.runs)} run(s) ...")
    val = average_predictions(
        [run_predictions(r, "VAL", a.n_tta, a.device, a.num_workers) for r in a.runs])
    calibration = calibrate(
        val["ef_pred"], val["y_true"], val["ord_pred"], CFG, ef_true=val["ef_true"],
        ord_dist=val.get("ord_dist"), class_dist=val.get("class_dist"),
        pred_std=val.get("ef_pred_std"))
    val_frozen = apply_frozen_strategy(val, calibration, CFG)
    val_pred = val_frozen["operational_pred"]
    val_ef = np.asarray(val_frozen["ef_calibrated"], dtype=np.float64)
    print(f"[selective] VAL-selected decision strategy: {calibration['best_strategy']}")

    rule = fit_selective_rule(val, val_pred, val_ef, CFG,
                              target=a.target, min_coverage=a.min_coverage)
    print(f"[selective] VAL-selected uncertainty signal : {rule['signal']}")
    print(f"[selective] target {a.target:.2f} reached on VAL: "
          f"{'yes' if rule['meets_target_on_validation'] else 'NO'} "
          f"at coverage {rule['validation_coverage']:.3f} "
          f"(min-recall {rule['validation_min_class_recall']:.3f})")

    # --------------------------------------------------------------- TEST
    print(f"[selective] gathering TEST predictions ...")
    test = average_predictions(
        [run_predictions(r, "TEST", a.n_tta, a.device, a.num_workers) for r in a.runs])
    frozen = apply_frozen_strategy(test, calibration, CFG)
    y_pred = frozen["operational_pred"]
    ef_used = np.asarray(frozen["ef_calibrated"], dtype=np.float64)
    y_true = test["y_true"]

    full = classification_metrics(y_true, y_pred, CFG.n_classes)
    reg = regression_metrics(test["ef_true"], ef_used)
    sel = apply_selective_rule(test, y_pred, ef_used, CFG, rule)
    cov = sel["covered"]

    # ------------------------------------------------------------- report
    print("\n=========== SELECTIVE PREDICTION - TEST RESULTS ===========")
    print(f"  runs               : {', '.join(a.runs)}")
    print(f"  decision strategy  : {calibration['best_strategy']}  (frozen on VAL)")
    print(f"  uncertainty signal : {sel['signal']}  (frozen on VAL)")
    print(f"  MAE (all studies)  : {reg['mae']:.3f} EF pts")

    print(f"\n  --- FULL COVERAGE  (100%, every study graded) ---")
    print(f"  n                  : {full['n'] if 'n' in full else len(y_true)}")
    print(f"  overall accuracy   : {full['overall_acc']:.3f}")
    print(f"  balanced accuracy  : {full['balanced_acc']:.3f}")
    print(f"  macro-F1           : {full['macro_f1']:.3f}")
    print(f"  MIN class recall   : {full['min_class_recall']:.3f}")
    for c, name in enumerate(CFG.CLASS_NAMES):
        r = full["per_class_recall"][c]
        print(f"    {'OK ' if (r is not None and r >= a.target) else '<75'} "
              f"{name:<16} {'n/a' if r is None else f'{r:.3f}'}")

    print(f"\n  --- SELECTIVE  (coverage {sel['coverage']*100:.1f}%, "
          f"{sel['n_covered']}/{sel['n_total']} graded, {sel['n_deferred']} deferred) ---")
    print(f"  overall accuracy   : {cov['overall_acc']:.3f}")
    print(f"  balanced accuracy  : {cov['balanced_acc']:.3f}")
    print(f"  macro-F1           : {cov['macro_f1']:.3f}")
    print(f"  MIN class recall   : {sel['min_class_recall']:.3f}  "
          f"({'ALL CLASSES >= ' + format(a.target, '.2f') if sel['min_class_recall'] >= a.target else 'below target'})")
    print("\n  Per-class recall on graded studies:")
    for c, name in enumerate(CFG.CLASS_NAMES):
        r = cov["per_class_recall"][c]
        flag = "OK " if (r is not None and r >= a.target) else "<75"
        print(f"    [{flag}] {name:<16} {'n/a' if r is None else f'{r:.3f}'}")
    print("\n  Confusion matrix (graded studies only):")
    print_confusion(cov["confusion"], CFG.CLASS_NAMES)
    if "deferred_accuracy" in sel:
        print(f"\n  accuracy on the deferred studies: {sel['deferred_accuracy']:.3f}"
              f"   (lower than covered = the rule is selecting the hard cases)")

    # coverage sweep, for the report figure and table
    signals = uncertainty_signals(test, ef_used, CFG.EF_THRESHOLDS)
    curve = coverage_curve(signals[sel["signal"]], y_true, y_pred, CFG.n_classes)
    print("\n  Coverage sweep (TEST):")
    print("    coverage   overall   balanced   min-recall   Sev    Mod    Mild   Norm")
    for m in curve:
        if abs(round(m["coverage"] * 100) % 5) < 1e-6:
            pc = ["  n/a" if r is None else f"{r:.2f}" for r in m["per_class_recall"]]
            print(f"      {m['coverage']*100:5.1f}%    {m['overall_acc']:.3f}     "
                  f"{m['balanced_acc']:.3f}      {m['min_class_recall']:.3f}     "
                  + "  ".join(pc))
    print("===========================================================")

    out = CFG.TRAIN_DIR / "outputs" / "selective_report.json"
    write_json_atomic(out, dict(
        runs=a.runs, n_tta=a.n_tta, target=a.target,
        decision_strategy=calibration["best_strategy"],
        selective_rule={k: v for k, v in rule.items()},
        regression=reg,
        full_coverage=full,
        selective={k: v for k, v in sel.items() if k != "selected"},
        coverage_curve=curve))
    print(f"\n  report -> {out}")

    if a.plot:
        _plot(curve, sel, a.target, CFG)


def _plot(curve, sel, target, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cov = [m["coverage"] * 100 for m in curve]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colours = ["#C00000", "#ED7D31", "#8FAADC", "#2E5C8A"]
    for c, name in enumerate(cfg.CLASS_NAMES):
        ax.plot(cov, [m["per_class_recall"][c] or np.nan for m in curve],
                label=name, color=colours[c], linewidth=1.6)
    ax.plot(cov, [m["min_class_recall"] for m in curve], "k--",
            linewidth=1.8, label="minimum class recall")
    ax.axhline(target, color="#548235", linestyle=":", linewidth=1.6)
    ax.text(51, target + 0.006, f"target {target:.2f}", color="#548235", fontsize=8.5)
    ax.axvline(sel["coverage"] * 100, color="#BF8F00", linestyle="-.", linewidth=1.5)
    ax.text(sel["coverage"] * 100 - 1, 0.52, f"operating point {sel['coverage']*100:.0f}%",
            color="#BF8F00", fontsize=8.5, rotation=90, va="bottom", ha="right")
    ax.set_xlabel("Coverage (% of studies graded)", fontsize=10)
    ax.set_ylabel("Recall", fontsize=10)
    ax.set_xlim(100, 50); ax.set_ylim(0.5, 1.0)
    ax.grid(alpha=0.3, linewidth=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    path = cfg.TRAIN_DIR / "outputs" / "selective_coverage_curve.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  coverage curve -> {path}")


if __name__ == "__main__":
    main()
