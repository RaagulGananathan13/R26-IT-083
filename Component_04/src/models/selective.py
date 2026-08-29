"""
Component 04 — Selective prediction with a clinician-referral option (SPR).

A triage aid does not have to answer every case.  Real decision-support systems
issue a recommendation where the evidence supports one and hand the rest back to
the clinician; forcing a label on every patient is a benchmarking convention,
not a clinical requirement.

This module implements selective classification (Chow's rule; El-Yaniv & Wiener,
JMLR 2010).  The model abstains on its least-confident cases and the remainder —
the *covered* set — is scored normally.  Two quantities are reported together,
and neither is meaningful alone:

    coverage   the fraction of patients given an automated diagnosis
    selective  per-class recall / F1 on the covered set
    accuracy

Reporting selective accuracy without coverage would be indefensible: abstaining
on 99% of patients makes any metric look perfect.  Here the abstention threshold
is chosen on VALIDATION as the SMALLEST amount of deferral that lifts every
class over the target, then frozen and applied once to test — so coverage is a
measured cost, not a free parameter.

Confidence is the top-two margin  p(1) - p(2)  rather than max-probability: on
four imbalanced classes the margin separates "confidently No_ACS" from "torn
between NSTEMI and STEMI", which is exactly the distinction that matters here.
"""
from __future__ import annotations

import os
import sys
import warnings
from typing import Dict, Sequence

import numpy as np
import pandas as pd

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_p for _p in (
    _SRC, os.path.join(_SRC, "core"), os.path.join(_SRC, "data"),
    os.path.join(_SRC, "models"), os.path.join(_SRC, "analysis"),
) if _p not in sys.path]
warnings.filterwarnings("ignore")

from config import (CFG, LABEL_ORDER, MODEL_DIR, REPORT_DIR, SUBTYPE_ORDER,
                    enable_utf8_stdout, save_json, set_seed)
from utils import banner, df_to_markdown, kv, print_report, section, summarise

enable_utf8_stdout()
SEED = set_seed()


# --------------------------------------------------------------------------
def confidence(P: np.ndarray) -> np.ndarray:
    """Top-two margin.  Large margin = the model is not torn between classes."""
    S = np.sort(P, axis=1)
    return S[:, -1] - S[:, -2]


def selective_report(y, pred, conf, keep_frac, class_names) -> Dict:
    """Metrics on the most-confident `keep_frac` of cases."""
    n = len(y)
    k = max(int(round(keep_frac * n)), 1)
    idx = np.argsort(-conf)[:k]
    r = summarise(y[idx], pred[idx], class_names)
    r["coverage"] = float(k / n)
    r["n_covered"] = int(k)
    r["n_deferred"] = int(n - k)
    return r


def choose_coverage(y, pred, conf, class_names, floor=0.75, metric="recall",
                    grid=None, n_boot=200, seed=42) -> tuple:
    """
    Largest coverage whose covered set clears the floor on every class — judged
    by the LOWER 5th percentile of a bootstrap on validation, not by the point
    estimate.

    Selecting on the point estimate picks the first coverage that happens to
    clear the floor on ~760 validation cases, and that choice does not survive
    the move to test: it chose 79% coverage and landed at 68% STEMI recall.
    Requiring the bootstrap lower bound to clear the floor buys the margin the
    point estimate lacks, and it sizes that margin from the data rather than
    from a guessed constant.
    """
    rng = np.random.RandomState(seed)
    grid = grid if grid is not None else np.arange(1.00, 0.29, -0.01)
    n = len(y)
    for cov in grid:
        k = max(int(round(cov * n)), 1)
        idx = np.argsort(-conf)[:k]
        yy, pp = y[idx], pred[idx]
        if min((yy == i).sum() for i in range(len(class_names))) < 5:
            continue
        mins = []
        for _ in range(n_boot):
            s = rng.randint(0, k, k)
            if len(np.unique(yy[s])) < len(class_names):
                continue
            r = summarise(yy[s], pp[s], class_names)
            mins.append(min(r["per_class"][c][metric] for c in class_names))
        if not mins:
            continue
        if float(np.percentile(mins, 5)) >= floor:
            return float(cov), selective_report(y, pred, conf, cov, class_names)
    return None, None


# --------------------------------------------------------------------------
def run(name: str, Pv, yv, Pt, yt, class_names: Sequence[str],
        floor: float, out: Dict) -> None:
    section(f"{name}")
    predv, predt = Pv.argmax(1), Pt.argmax(1)
    confv, conft = confidence(Pv), confidence(Pt)

    # coverage/accuracy curve, for the report
    rows = []
    for cov in (1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50):
        r = selective_report(yt, predt, conft, cov, class_names)
        row = {"coverage": cov, "n covered": r["n_covered"],
               "bal. acc": r["balanced_accuracy"], "macro-F1": r["macro_f1"],
               "min recall": r["min_recall"]}
        for c in class_names:
            row[c] = r["per_class"][c]["recall"]
        rows.append(row)
    curve = pd.DataFrame(rows)
    print(df_to_markdown(curve))

    for metric in ("recall", "f1"):
        cov, rv = choose_coverage(yv, predv, confv, class_names, floor, metric)
        if cov is None:
            print(f"\n  [{metric}] no coverage down to 30% clears {floor:.0%} "
                  f"on every class — not attainable by deferral alone.")
            out[f"{name}_{metric}"] = {"attainable": False}
            continue
        rt = selective_report(yt, predt, conft, cov, class_names)
        vals = {c: rt["per_class"][c][metric] for c in class_names}
        ok = min(vals.values()) >= floor
        print(f"\n  [{metric}] validation-selected coverage = {cov:.0%}  "
              f"({rt['n_covered']:,} covered, {rt['n_deferred']:,} deferred)")
        for c in class_names:
            m = rt["per_class"][c]
            print(f"      {c:<9} recall={m['recall']*100:6.2f}%  "
                  f"F1={m['f1']*100:6.2f}%  n={m['support']:,}")
        print(f"      -> every class >= {floor:.0%} on TEST {metric}: "
              f"{'YES' if ok else 'NO'}  (min {min(vals.values())*100:.2f}%)")
        out[f"{name}_{metric}"] = {
            "attainable": bool(ok), "coverage": cov,
            "n_covered": rt["n_covered"], "n_deferred": rt["n_deferred"],
            "test": rt}
    out[f"{name}_curve"] = curve.to_dict("records")


def main() -> None:
    H = CFG.primary_horizon
    floor = float(CFG.get("decision.min_recall_floor", 0.75))
    banner(f"SELECTIVE PREDICTION WITH CLINICIAN REFERRAL   H = {H}h")
    print("  The model answers where the evidence supports an answer and defers")
    print("  the rest.  Coverage is reported with every accuracy figure; a")
    print("  selective metric without its coverage is meaningless.\n")
    out: Dict = {"horizon": H, "floor": floor}

    # ---- Stage 2 (subtyping among ACS) ----
    p2 = os.path.join(MODEL_DIR, f"stage2_scores_H{H}.npz")
    if os.path.exists(p2):
        d = np.load(p2)
        run("Stage 2 — subtyping", d["P_val"], d["y_val"], d["P_test"],
            d["y_test"], SUBTYPE_ORDER, floor, out)

    # ---- Unified four-class, if it has been fitted ----
    p4 = os.path.join(MODEL_DIR, f"um4_scores_H{H}.npz")
    if os.path.exists(p4):
        d = np.load(p4)
        run("Unified 4-class", d["P_val"], d["y_val"], d["P_test"],
            d["y_test"], LABEL_ORDER, floor, out)
    else:
        print("\n  [skip] unified 4-class scores not found — run unified4.py first")

    save_json(out, os.path.join(REPORT_DIR, f"selective_H{H}.json"))
    banner("SELECTIVE PREDICTION COMPLETE")


if __name__ == "__main__":
    main()
