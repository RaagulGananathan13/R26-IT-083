"""
Robustness analysis: acquisition-subgroup breakdown and paired significance tests.
=================================================================================

Consumes the per-study prediction files written by ``run_ensemble.py
--save-predictions`` (or ``run_eval.py --save-predictions``), so no re-inference
is required.

    # 1. produce predictions once per system
    python run_ensemble.py --runs uefnet_v3 uefnet_v3b uefnet_v3c --n-tta 10 --save-predictions
    python run_ensemble.py --runs uefnet_v3 --n-tta 10 --save-predictions --out outputs/single.json

    # 2. subgroup robustness for one system
    python run_robustness.py --predictions outputs/predictions_test_uefnet_v3_uefnet_v3b_uefnet_v3c.npz

    # 3. paired significance test between two systems
    python run_robustness.py \
        --predictions outputs/predictions_test_uefnet_v3_uefnet_v3b_uefnet_v3c.npz \
        --compare-with outputs/predictions_test_uefnet_v3.npz

Scope note
----------
EchoNet-Dynamic ships no patient demographics, so **demographic fairness cannot
be assessed on this cohort and is not claimed**.  What is available are
acquisition characteristics (frame rate, spatial resolution, recording length)
and ventricular volumes; systematic failure on one acquisition setting is a real
deployment risk and is what this script measures.
"""
from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from config import CFG


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True,
                    help="npz written by run_ensemble.py --save-predictions")
    ap.add_argument("--compare-with", default=None,
                    help="second npz; enables paired significance testing")
    ap.add_argument("--rule", default="operational", choices=["operational", "clinical"],
                    help="which decision rule's predictions to analyse")
    ap.add_argument("--n-bins", type=int, default=3, help="quantile bins per covariate")
    ap.add_argument("--min-n", type=int, default=30,
                    help="subgroups below this size are flagged underpowered")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def load_predictions(path, rule):
    d = np.load(path, allow_pickle=True)
    key = "y_pred_operational" if rule == "operational" else "y_pred_clinical"
    ef_key = "ef_pred_calibrated" if rule == "operational" else "ef_pred_raw"
    return {
        "file_name": [str(f) for f in d["file_name"]],
        "y_true": d["y_true"].astype(np.int64),
        "ef_true": d["ef_true"].astype(np.float64),
        "y_pred": d[key].astype(np.int64),
        "ef_pred": d[ef_key].astype(np.float64),
        "runs": [str(r) for r in d["runs"]],
        "strategy": str(d["strategy"]),
    }


def build_covariates(file_names):
    """Join acquisition metadata from the preprocessing manifest, in prediction order."""
    man = pd.read_csv(CFG.MANIFEST)
    man["FileName"] = man["FileName"].astype(str)
    man = man.set_index("FileName")
    missing = [f for f in file_names if f not in man.index]
    if missing:
        raise KeyError(f"{len(missing)} predicted studies absent from the manifest, "
                       f"e.g. {missing[:3]}")
    rows = man.loc[file_names]

    cov = {}
    for col, label in (("FPS", "frame_rate_fps"),
                       ("NumberOfFrames", "recording_length_frames"),
                       ("EDV", "end_diastolic_volume_ml"),
                       ("ESV", "end_systolic_volume_ml")):
        if col in rows.columns:
            v = pd.to_numeric(rows[col], errors="coerce").to_numpy(dtype=np.float64)
            if np.isfinite(v).sum() >= 10 and np.nanstd(v) > 0:
                cov[label] = v
    return cov


def main():
    a = parse_args()
    from engine.robustness import subgroup_report, compare_systems

    P = load_predictions(a.predictions, a.rule)
    n = len(P["y_true"])
    print(f"[robustness] system A : {', '.join(P['runs'])}  (strategy {P['strategy']}, "
          f"rule={a.rule}, n={n})")

    report = {"system_a": {"runs": P["runs"], "strategy": P["strategy"], "rule": a.rule, "n": n}}

    # -------------------------------------------------- subgroup robustness
    cov = build_covariates(P["file_name"])
    print(f"[robustness] covariates available: {', '.join(cov) if cov else 'NONE'}")
    print("[robustness] note: EchoNet-Dynamic carries no patient demographics; "
          "demographic fairness is not assessable on this cohort.")

    sub = subgroup_report(P["y_true"], P["y_pred"], P["ef_true"], P["ef_pred"],
                          cov, CFG.n_classes, n_bins=a.n_bins, min_n=a.min_n)
    report["subgroups"] = sub

    print("\n============ ACQUISITION-SUBGROUP ROBUSTNESS ============")
    for name, block in sub.items():
        print(f"\n  {name}")
        print(f"    {'subgroup':>26}  {'n':>5}  {'MAE':>6}  {'acc':>6}  {'bal':>6}  {'min-rec':>7}")
        for g in block["groups"]:
            flag = " *" if g["underpowered"] else "  "
            print(f"    {g['label']:>26}  {g['n']:>5}  {g['mae']:>6.3f}  "
                  f"{g['overall_acc']:>6.3f}  {g['balanced_acc']:>6.3f}  "
                  f"{g['min_class_recall']:>7.3f}{flag}")
        s = block["spread_over_powered_groups"]
        if s:
            print(f"    range across powered subgroups: MAE {s['mae_range']:.3f} "
                  f"({s['mae_min']:.3f}-{s['mae_max']:.3f}) | "
                  f"balanced acc {s['balanced_acc_range']:.3f}")
    print("\n    * = underpowered (n < %d), interval too wide to conclude" % a.min_n)

    # -------------------------------------------------- paired significance
    if a.compare_with:
        Q = load_predictions(a.compare_with, a.rule)
        if len(Q["y_true"]) != n or not np.array_equal(Q["y_true"], P["y_true"]):
            sys.exit("[robustness] the two prediction files describe different studies "
                     "or a different ordering; paired testing requires identical splits")
        print(f"\n[robustness] system B : {', '.join(Q['runs'])}  (strategy {Q['strategy']})")

        cmp = compare_systems(P["y_true"], P["y_pred"], Q["y_pred"],
                              P["ef_true"], P["ef_pred"], Q["ef_pred"],
                              CFG.n_classes, n_boot=a.n_bootstrap, seed=a.seed)
        report["system_b"] = {"runs": Q["runs"], "strategy": Q["strategy"]}
        report["paired_comparison"] = cmp

        print("\n============ PAIRED SIGNIFICANCE (A vs B) ============")
        print(f"  n paired studies    : {cmp['n']}")
        print(f"  MAE       A {cmp['mae_a']:.4f}  vs  B {cmp['mae_b']:.4f}")
        d = cmp["mae_difference"]
        print(f"    difference {d['observed_difference']:+.4f}  "
              f"95% CI [{d['ci_lower']:+.4f}, {d['ci_hi']:+.4f}]  p = {d['p_value_two_sided']:.4f}"
              f"  {'SIGNIFICANT' if d['significant_at_alpha'] else 'not significant'}")
        print(f"  accuracy  A {cmp['accuracy_a']:.4f}  vs  B {cmp['accuracy_b']:.4f}")
        d = cmp["accuracy_difference"]
        print(f"    difference {d['observed_difference']:+.4f}  "
              f"95% CI [{d['ci_lower']:+.4f}, {d['ci_hi']:+.4f}]  p = {d['p_value_two_sided']:.4f}"
              f"  {'SIGNIFICANT' if d['significant_at_alpha'] else 'not significant'}")
        d = cmp["balanced_accuracy_difference"]
        print(f"  balanced accuracy difference {d['observed_difference']:+.4f}  "
              f"95% CI [{d['ci_lower']:+.4f}, {d['ci_hi']:+.4f}]  p = {d['p_value_two_sided']:.4f}"
              f"  {'SIGNIFICANT' if d['significant_at_alpha'] else 'not significant'}")
        m = cmp["mcnemar"]
        print(f"  McNemar (exact)     : A-right/B-wrong {m['b01_a_right_b_wrong']}, "
              f"A-wrong/B-right {m['b10_a_wrong_b_right']}, "
              f"discordant {m['n_discordant']}, p = {m['p_value_two_sided']:.4f}"
              f"  {'SIGNIFICANT' if m['significant_at_0.05'] else 'not significant'}")
    else:
        print("\n[robustness] no --compare-with supplied; skipping paired significance test")

    from core.common import write_json_atomic
    out = a.out or str(CFG.TRAIN_DIR / "outputs" / "robustness_report.json")
    write_json_atomic(out, report)
    print(f"\n  report -> {out}")
    print("=========================================================")


if __name__ == "__main__":
    main()
