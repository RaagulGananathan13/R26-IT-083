"""Full per-class report for the STEMI territory head.

Recall alone hides half the picture. A class can be recalled well because the
model over-calls it, which costs the other class its precision; on a two-class
problem that trade is invisible unless both are printed side by side. So this
prints precision, recall, specificity, F1, negative predictive value and
support for each wall, with Wilson intervals on every proportion, and the
confusion matrix the numbers come from.

Both configurations are reported, for the reason given in train_territory.py:
three of the features are the ECG cart's own printed read of territory, and the
model's standing depends on whether those are allowed.

USAGE
-----
    python src/models/territory_report.py
    python src/models/territory_report.py --seeds 7
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_territory import (  # noqa: E402
    ECG_TERRITORY, REPORTS, assemble, choose_threshold, fit_ensemble,
    predict, wilson, _out_of_fold,
)

WALLS = ((1, "anterior"), (0, "inferior"))


def block(y_true: np.ndarray, y_pred: np.ndarray, positive: int) -> Dict:
    """One class treated as positive, everything reported against it."""
    tp = int(((y_true == positive) & (y_pred == positive)).sum())
    fn = int(((y_true == positive) & (y_pred != positive)).sum())
    fp = int(((y_true != positive) & (y_pred == positive)).sum())
    tn = int(((y_true != positive) & (y_pred != positive)).sum())

    def ratio(num, den):
        return num / den if den else float("nan")

    recall = ratio(tp, tp + fn)
    precision = ratio(tp, tp + fp)
    specificity = ratio(tn, tn + fp)
    npv = ratio(tn, tn + fn)
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and not math.isnan(precision + recall) else float("nan"))

    return {
        "support": tp + fn, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": recall, "recall_ci": wilson(tp, tp + fn),
        "precision": precision, "precision_ci": wilson(tp, tp + fp),
        "specificity": specificity, "specificity_ci": wilson(tn, tn + fp),
        "npv": npv, "f1": f1,
        # "Accuracy for this class" read as: of every case, how often did the
        # model get this class's call right, either way.
        "class_accuracy": ratio(tp + tn, tp + tn + fp + fn),
    }


def show(name: str, y_true: np.ndarray, y_pred: np.ndarray,
         auroc: float, threshold: float) -> Dict:
    print()
    print("=" * 78)
    print("  %s" % name)
    print("  threshold %.3f (from grouped out-of-fold train+val)   test n = %d"
          % (threshold, len(y_true)))
    print("=" * 78)

    per_class = {}
    print("  %-9s %8s %8s %9s %8s %6s %7s"
          % ("class", "recall", "prec.", "specif.", "F1", "n", "acc."))
    for value, wall in WALLS:
        b = block(y_true, y_pred, value)
        per_class[wall] = b
        print("  %-9s %8.4f %8.4f %9.4f %8.4f %6d %7.4f"
              % (wall, b["recall"], b["precision"], b["specificity"],
                 b["f1"], b["support"], b["class_accuracy"]))

    print()
    print("  95 %% Wilson intervals")
    for value, wall in WALLS:
        b = per_class[wall]
        print("    %-9s recall    [%.3f, %.3f]   (%d/%d)"
              % (wall, b["recall_ci"][0], b["recall_ci"][1], b["tp"], b["support"]))
        print("    %-9s precision [%.3f, %.3f]   (%d/%d)"
              % ("", b["precision_ci"][0], b["precision_ci"][1],
                 b["tp"], b["tp"] + b["fp"]))

    overall = float((y_true == y_pred).mean())
    balanced = float(np.mean([per_class[w]["recall"] for _, w in WALLS]))
    macro_f1 = float(np.mean([per_class[w]["f1"] for _, w in WALLS]))
    macro_p = float(np.mean([per_class[w]["precision"] for _, w in WALLS]))

    print()
    print("  overall accuracy   %.4f" % overall)
    print("  balanced accuracy  %.4f" % balanced)
    print("  macro precision    %.4f" % macro_p)
    print("  macro F1           %.4f" % macro_f1)
    print("  AUROC              %.4f" % auroc)
    print("  minimum recall     %.4f" % min(per_class[w]["recall"] for _, w in WALLS))

    print()
    print("  confusion matrix (rows = truth, columns = predicted)")
    print("  %-12s %10s %10s" % ("", "anterior", "inferior"))
    for value, wall in WALLS:
        row_a = int(((y_true == value) & (y_pred == 1)).sum())
        row_i = int(((y_true == value) & (y_pred == 0)).sum())
        print("  %-12s %10d %10d" % (wall, row_a, row_i))

    return {"per_class": per_class, "overall_accuracy": overall,
            "balanced_accuracy": balanced, "macro_f1": macro_f1,
            "macro_precision": macro_p, "auroc": auroc,
            "threshold": threshold}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=7)
    parser.add_argument("--out", default=os.path.join(REPORTS, "territory_per_class.json"))
    args = parser.parse_args()
    seeds = list(range(args.seeds))

    from sklearn.metrics import roc_auc_score

    df, all_columns = assemble()
    train = df[df["fold"] == "train"]
    val = df[df["fold"] == "val"]
    test = df[df["fold"] == "test"]
    physiology = [c for c in all_columns if c not in ECG_TERRITORY]

    print("STEMI territory head -- per-class report")
    print("  train %d   val %d   test %d" % (len(train), len(val), len(test)))

    results = {}
    for name, columns in (("FULL -- every feature", all_columns),
                          ("PHYSIOLOGY -- no ECG-report territory", physiology)):
        models, _ = fit_ensemble(train, columns, seeds)
        pool = pd.concat([train, val], ignore_index=True)
        threshold = choose_threshold(_out_of_fold(pool, columns, seeds),
                                     pool["territory"].to_numpy(int))

        probabilities = predict(models, test, columns)
        y_true = test["territory"].to_numpy(int)
        y_pred = (probabilities >= threshold).astype(int)
        results[name] = show(name, y_true, y_pred,
                             float(roc_auc_score(y_true, probabilities)), threshold)

    os.makedirs(REPORTS, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=float)
    print()
    print("  written: %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
