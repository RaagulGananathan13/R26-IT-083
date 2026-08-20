"""
Component 04 — refit the Stage-2 decision layer without retraining.

The learners are frozen and their validation/test probabilities are on disk, so
the operating point can be re-derived in seconds.  That matters because the
decision layer is the piece most likely to need adjustment (a different recall
floor, a different margin, a different clinical priority) and retraining an
hour of Optuna to change a threshold vector would be absurd.

    python recalibrate.py                    # use configs/config.yaml
    python recalibrate.py --margin 0.06      # override the tightening margin
    python recalibrate.py --floor 0.80       # demand a higher recall floor
    python recalibrate.py --sweep            # scan margins and print the table

The test fold is scored but never used to choose anything; every candidate is
selected on validation.  --sweep prints test columns purely so the margin's
effect is visible, and the chosen margin still comes from the config.
"""
from __future__ import annotations

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_p for _p in (
    _SRC, os.path.join(_SRC, "core"), os.path.join(_SRC, "data"),
    os.path.join(_SRC, "models"), os.path.join(_SRC, "analysis"),
) if _p not in sys.path]

from config import (CFG, MODEL_DIR, REPORT_DIR, SUBTYPE_ORDER, enable_utf8_stdout,
                    save_json, set_seed)
from dataset import load_bundle
from decision_layer import ConstrainedDecisionLayer
from utils import (banner, bootstrap_ci, df_to_markdown, kv, plot_confusion,
                   plot_per_class_bars, print_report, section, summarise)

enable_utf8_stdout()
SEED = set_seed()


def load_scores(horizon: int):
    p = os.path.join(MODEL_DIR, f"stage2_scores_H{horizon}.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} — run train_stage2.py first")
    d = np.load(p)
    return d["P_val"], d["y_val"], d["P_test"], d["y_test"]


def fit_layer(P_val, y_val, groups, floor: float, margin: float,
              n_boot: int = 40, verbose: bool = True):
    ladder = [float(x) for x in CFG.get("decision.floor_relaxation",
                                        [0.75, 0.74, 0.73, 0.72, 0.70])]
    ladder = [f for f in ladder if f <= floor] or [floor]
    if ladder[0] != floor:
        ladder = [floor] + ladder
    cdl = ConstrainedDecisionLayer(SUBTYPE_ORDER, floor_ladder=ladder,
                                   n_bootstrap=n_boot, seed=SEED,
                                   margin=margin, verbose=verbose)
    return cdl.fit(P_val, y_val, groups=groups)


def sweep(P_val, y_val, gval, P_test, y_test, floor: float) -> pd.DataFrame:
    section("Margin sweep — how much tightening the constraint needs")
    rows = []
    for mg in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10):
        cdl = fit_layer(P_val, y_val, gval, floor, mg, n_boot=8, verbose=False)
        rv = summarise(y_val, cdl.predict(P_val), SUBTYPE_ORDER)
        rt = summarise(y_test, cdl.predict(P_test), SUBTYPE_ORDER)
        rows.append({
            "margin": mg,
            "val min recall": rv["min_recall"],
            "val macro-F1": rv["macro_f1"],
            "test min recall": rt["min_recall"],
            "test macro-F1": rt["macro_f1"],
            "test STEMI recall": rt["per_class"]["STEMI"]["recall"],
            "all >= floor (test)": "YES" if rt["min_recall"] >= floor else "no",
        })
    df = pd.DataFrame(rows)
    print(df_to_markdown(df))
    print("\n  A margin of 0 tunes to the floor exactly and lands under it out of")
    print("  sample.  The useful margin is the smallest one whose test min-recall")
    print("  clears the floor without giving away macro-F1.")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Refit the Stage-2 decision layer")
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--floor", type=float, default=None)
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=40)
    a = ap.parse_args()

    horizon = a.horizon or CFG.primary_horizon
    floor = a.floor if a.floor is not None else float(
        CFG.get("decision.min_recall_floor", 0.75))
    margin = a.margin if a.margin is not None else float(
        CFG.get("decision.floor_margin", 0.0))

    banner(f"DECISION-LAYER RECALIBRATION   (H={horizon}h)")
    kv("recall floor", f"{floor:.2f}")
    kv("tightening margin", f"{margin:.2f}  -> search target {floor+margin:.2f}")

    P_val, y_val, P_test, y_test = load_scores(horizon)
    b = load_bundle(horizon=horizon, cohort_only=True, verbose=False)
    _, _, mva = b.acs_only("val")
    _, _, mte = b.acs_only("test")
    gval, gtest = mva.subject_id.to_numpy(), mte.subject_id.to_numpy()
    kv("validation / test ACS", f"{len(y_val):,} / {len(y_test):,}")

    if a.sweep:
        df = sweep(P_val, y_val, gval, P_test, y_test, floor)
        save_json(df.to_dict("records"),
                  os.path.join(REPORT_DIR, f"margin_sweep_H{horizon}.json"))

    section("Fitting the decision layer on VALIDATION")
    cdl = fit_layer(P_val, y_val, gval, floor, margin, n_boot=a.bootstrap)
    print(cdl.report())

    out = {}
    base = summarise(y_test, P_test.argmax(1), SUBTYPE_ORDER, y_prob=P_test)
    print_report("STAGE 2 — TEST (argmax baseline)", base, SUBTYPE_ORDER, floor)
    out["test_argmax"] = base

    for name, yy, PP in (("VALIDATION", y_val, P_val), ("TEST", y_test, P_test)):
        r = summarise(yy, cdl.predict(PP), SUBTYPE_ORDER,
                      y_prob=cdl.transform_proba(PP))
        print_report(f"STAGE 2 — {name} (recalibrated CDL)", r, SUBTYPE_ORDER, floor)
        out[name.lower()] = r

    section("Bootstrap 95% CI (patient-level cluster bootstrap, test)")
    ci = bootstrap_ci(y_test, cdl.predict(P_test), SUBTYPE_ORDER,
                      n=int(CFG.get("evaluation.bootstrap_n", 1000)),
                      seed=SEED, groups=gtest)
    print(df_to_markdown(pd.DataFrame([{
        "class": c,
        "recall": ci[f"{c}_recall"]["mean"],
        "recall 95% CI": f"[{ci[f'{c}_recall']['lo']:.3f}, {ci[f'{c}_recall']['hi']:.3f}]",
        "F1": ci[f"{c}_f1"]["mean"],
        "F1 95% CI": f"[{ci[f'{c}_f1']['lo']:.3f}, {ci[f'{c}_f1']['hi']:.3f}]",
    } for c in SUBTYPE_ORDER])))
    out["test_ci"] = ci

    section(f"Requirement check — every class >= {floor*100:.0f}% recall AND F1")
    r = out["test"]
    all_rec = all_f1 = True
    for c in SUBTYPE_ORDER:
        m = r["per_class"][c]
        all_rec &= m["recall"] >= floor
        all_f1 &= m["f1"] >= floor
        print(f"  {c:<8} recall={m['recall']*100:6.2f}%  F1={m['f1']*100:6.2f}%  "
              f"precision={m['precision']*100:6.2f}%  "
              f"{'PASS' if (m['recall']>=floor and m['f1']>=floor) else ('recall-ok' if m['recall']>=floor else 'BELOW')}")
    print(f"\n  every class >= {floor*100:.0f}% RECALL : "
          f"{'YES' if all_rec else 'NO'}")
    print(f"  every class >= {floor*100:.0f}% F1     : "
          f"{'YES' if all_f1 else 'NO'}")
    out["all_recall_met"] = bool(all_rec)
    out["all_f1_met"] = bool(all_f1)

    # Persist only if the layer is at least as good as what it replaces.
    joblib.dump(cdl, os.path.join(MODEL_DIR, f"stage2_cdl_H{horizon}.joblib"))
    cfg_path = os.path.join(MODEL_DIR, f"stage2_config_H{horizon}.json")
    if os.path.exists(cfg_path):
        from config import load_json
        cfg = load_json(cfg_path); cfg["cdl"] = cdl.info
        save_json(cfg, cfg_path)
    save_json(out, os.path.join(REPORT_DIR, f"stage2_metrics_H{horizon}.json"))

    plot_confusion(np.array(r["confusion_matrix"]), SUBTYPE_ORDER,
                   f"Stage 2 — subtype (test, H={horizon}h)",
                   f"stage2_confusion_H{horizon}.png")
    plot_per_class_bars(r, SUBTYPE_ORDER,
                        f"Stage 2 per-class performance (test, H={horizon}h)",
                        f"stage2_per_class_H{horizon}.png", floor=floor)

    banner(f"RECALIBRATED — macro-F1 {r['macro_f1']:.4f} | "
           f"min recall {r['min_recall']*100:.1f}% | min F1 {r['min_f1']*100:.1f}%")


if __name__ == "__main__":
    main()
