"""What happens if the other two STEMI groups are modelled as classes.

The WHO table lists five STEMI codes, so the obvious question is why the head
is binary. The answer is not an opinion about sample size -- it is two
different problems, and this measures both rather than asserting them.

  OTHER SITE    circumflex, lateral, posterior. A real anatomical territory,
                simply rare: 66 stays, which is roughly 10 in the test fold.
                A per-class recall on 10 cases has a Wilson interval about
                60 points wide, so the number would be unreportable rather
                than wrong.

  UNSPECIFIED   I21.3, I21.9, 410.9 -- the coder recording an infarct without
                naming a wall. 199 stays. This is not a fifth territory, it is
                the absence of one, and a model asked to predict it is being
                asked to predict whether the DOCUMENTATION was complete. Any
                accuracy it reaches is accuracy at guessing a coding habit.

Both are trained anyway, because "we measured it and here is what happened"
is a stronger answer at a panel than "we judged it unwise".

USAGE
-----
    python src/models/territory_multiclass.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence, Set

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_territory import (  # noqa: E402
    ANTERIOR, ARTIFACTS, ECG_TERRITORY, ICD_DIR, INFERIOR, NEVER, OTHER_SITE,
    PROCESSED, REPORTS, wilson,
)

UNSPECIFIED = (("I213", "I219"), ("4109",))
NAMES = {0: "inferior", 1: "anterior", 2: "other site", 3: "unspecified"}


def build_multiclass() -> pd.Series:
    """`stay_id -> 0 inferior / 1 anterior / 2 other site / 3 unspecified`."""
    master = pd.read_parquet(os.path.join(PROCESSED, "master_data.parquet"))
    icd = pd.read_csv(os.path.join(ICD_DIR, "diagnoses_icd.csv.gz"),
                      dtype={"icd_code": str, "icd_version": "Int64"})
    icd["icd_code"] = icd["icd_code"].str.strip().str.upper()
    joined = icd.merge(master[["subject_id", "hadm_id", "stay_id", "acs_label"]],
                       on=["subject_id", "hadm_id"], how="inner")

    stemi: Set = set(master.loc[master["acs_label"] == 3, "stay_id"])
    v10 = joined[joined["icd_version"] == 10]
    v9 = joined[joined["icd_version"] == 9]

    def group(spec) -> Set:
        p10, p9 = spec
        a = set(v10.loc[v10["icd_code"].str.startswith(p10, na=False), "stay_id"])
        b = set(v9.loc[v9["icd_code"].str.startswith(p9, na=False), "stay_id"])
        return (a | b) & stemi

    ant, inf = group(ANTERIOR), group(INFERIOR)
    oth, uns = group(OTHER_SITE), group(UNSPECIFIED)

    # A stay carrying two territories is an ambiguity, not a label. Localised
    # codes win over "unspecified", since a named wall is the better evidence.
    label: Dict = {}
    for stay in uns:
        label[stay] = 3
    for stay in oth:
        label[stay] = 2
    for stay in ant ^ inf:
        label[stay] = 1 if stay in ant else 0
    for stay in ant & inf:
        label.pop(stay, None)
    return pd.Series(label, name="territory")


def assemble_multiclass():
    y = build_multiclass()
    feats = pd.read_parquet(os.path.join(ARTIFACTS, "features_H24.parquet"))
    split = pd.read_parquet(os.path.join(ARTIFACTS, "split_assignment.parquet"))
    df = feats[feats["stay_id"].isin(y.index)].copy()
    df["territory"] = df["stay_id"].map(y).astype(int)
    df = df.merge(split[["stay_id", "fold"]], on="stay_id", how="left")
    columns = [c for c in df.columns
               if c not in NEVER and pd.api.types.is_numeric_dtype(df[c])]
    return df, columns


def report(tag: str, df: pd.DataFrame, columns: Sequence[str],
           keep: Sequence[int], seeds: Sequence[int]) -> Dict:
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score

    part = df[df["territory"].isin(keep)].copy()
    train = part[part["fold"] == "train"]
    test = part[part["fold"] == "test"]

    print()
    print("=" * 78)
    print("  %s" % tag)
    print("=" * 78)
    print("  %-13s %7s %6s %6s %7s" % ("class", "train", "val", "test", "total"))
    for value in keep:
        counts = part[part["territory"] == value]["fold"].value_counts()
        print("  %-13s %7d %6d %6d %7d"
              % (NAMES[value], counts.get("train", 0), counts.get("val", 0),
                 counts.get("test", 0), int((part["territory"] == value).sum())))

    models: List = []
    for seed in seeds:
        models.append(lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=7, max_depth=4,
            min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.5, reg_lambda=5.0, reg_alpha=1.0,
            class_weight="balanced", random_state=seed,
            verbose=-1).fit(train[list(columns)], train["territory"]))

    proba = np.mean([m.predict_proba(test[list(columns)]) for m in models], axis=0)
    classes = list(models[0].classes_)
    pred = np.array([classes[i] for i in proba.argmax(axis=1)])
    truth = test["territory"].to_numpy(int)

    print()
    print("  %-13s %8s %10s %8s %7s" % ("class", "recall", "95% CI", "n", "hits"))
    out = {}
    worst = 1.0
    for value in keep:
        mask = truth == value
        total = int(mask.sum())
        hits = int((pred[mask] == value).sum())
        r = hits / total if total else float("nan")
        low, high = wilson(hits, total)
        worst = min(worst, r if total else worst)
        print("  %-13s %8.4f  [%.2f,%.2f] %8d %7d"
              % (NAMES[value], r, low, high, total, hits))
        out[NAMES[value]] = {"recall": r, "ci": [low, high], "n": total, "hits": hits}

    acc = float(accuracy_score(truth, pred))
    bal = float(np.mean([out[NAMES[v]]["recall"] for v in keep]))
    print()
    print("  overall accuracy   %.4f" % acc)
    print("  balanced accuracy  %.4f" % bal)
    print("  weakest class      %.4f" % worst)
    if worst < 0.75:
        print("  -> below 0.75; this configuration does not meet the bar.")
    return {"per_class": out, "accuracy": acc, "balanced_accuracy": bal,
            "weakest": worst}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=7)
    args = parser.parse_args()
    seeds = list(range(args.seeds))

    df, all_columns = assemble_multiclass()
    print("STEMI stays with any territory code: %d" % len(df))

    results = {
        "2-class: anterior vs inferior": report(
            "TWO CLASSES -- anterior vs inferior (the shipped head)",
            df, all_columns, [1, 0], seeds),
        "3-class: + other site": report(
            "THREE CLASSES -- adding 'other site' (circumflex/lateral/posterior)",
            df, all_columns, [1, 0, 2], seeds),
        "4-class: + unspecified": report(
            "FOUR CLASSES -- adding 'unspecified', which names no wall",
            df, all_columns, [1, 0, 2, 3], seeds),
    }

    print()
    print("=" * 78)
    print("  WEAKEST CLASS RECALL, BY CONFIGURATION")
    print("=" * 78)
    for name, r in results.items():
        mark = "OK " if r["weakest"] >= 0.75 else "BELOW"
        print("  %-32s weakest %.4f   accuracy %.4f   %s"
              % (name, r["weakest"], r["accuracy"], mark))

    path = os.path.join(REPORTS, "territory_multiclass.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=float)
    print()
    print("  written: %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
