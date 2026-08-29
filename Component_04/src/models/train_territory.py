"""Anterior vs inferior STEMI territory, from ED-triage data.

WHAT THIS TRAINS
----------------
A binary head over the same 219-feature vector Component 04 already builds,
predicting which wall a STEMI involves. The label comes from the discharge ICD
code -- ICD-10-CM I21.0x/I21.1x by culprit artery, plus ICD-9 410.0/410.1
anterior and 410.2/410.3/410.4 inferior, since this cohort straddles the
transition and half its STEMI stays are coded in each.

WHY IT IS REPORTED TWICE
------------------------
Three of those features -- `ecg_infarct_anterior`, `ecg_infarct_inferior`,
`ecg_territory_count` -- are parsed from the ECG cart's own printed report. The
ICD coder read the same ECG when assigning the code, so those features and the
label share a source. That is not temporal leakage, and the features are
genuinely present at triage, but a model built on them is transcribing an
existing interpretation rather than deriving one.

So both are trained and both are reported:

    FULL         every feature, including the cart's own read
    PHYSIOLOGY   the same minus those three

The gap between them is the honest measure of what the model adds beyond
reading the ECG printout, and hiding it would be the same class of mistake as
the temporal leak this component already retracted an AUROC for.

PROTOCOL
--------
The existing patient-grouped fold assignment is used unchanged, so no
subject_id appears in more than one fold. Three seeds per configuration,
averaged into an ensemble. The operating threshold is chosen on VALIDATION to
maximise the worse of the two class recalls; the test fold is scored once with
that frozen threshold and never consulted while choosing anything.

Wilson intervals accompany every recall, because with roughly a hundred test
cases a point estimate on its own invites more confidence than it can support.

USAGE
-----
    python src/models/train_territory.py
    python src/models/train_territory.py --seeds 5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

COMPONENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED = os.path.join(COMPONENT, "data", "processed")
ARTIFACTS = os.path.join(COMPONENT, "artifacts", "data")
ICD_DIR = os.path.join(COMPONENT, "data", "mimic_icd")
REPORTS = os.path.join(COMPONENT, "artifacts", "reports")

#: ICD-10-CM prefixes and their ICD-9 410.x equivalents, by wall.
ANTERIOR = (("I2101", "I2102", "I2109"), ("4100", "4101"))
INFERIOR = (("I2111", "I2119"), ("4102", "4103", "4104"))
OTHER_SITE = (("I2121", "I2129"), ("4105", "4106", "4108"))

#: Parsed from the ECG cart's report text. See the module docstring.
ECG_TERRITORY = ("ecg_infarct_anterior", "ecg_infarct_inferior", "ecg_territory_count")

#: Never features: identifiers, the source label, and the target.
NEVER = ("subject_id", "hadm_id", "stay_id", "acs_label", "intime",
         "territory", "fold", "ed_los_h")


# ----------------------------------------------------------------- label ---
def build_label() -> pd.Series:
    """`stay_id -> 1 anterior / 0 inferior`, for cleanly-lateralised STEMI."""
    master = pd.read_parquet(os.path.join(PROCESSED, "master_data.parquet"))
    path = os.path.join(ICD_DIR, "diagnoses_icd.csv.gz")
    if not os.path.exists(path):
        raise SystemExit(
            "diagnoses_icd.csv.gz not found in %s. See src/data/icd_subtypes.py"
            % ICD_DIR)

    icd = pd.read_csv(path, dtype={"icd_code": str, "icd_version": "Int64"})
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

    anterior, inferior, other = group(ANTERIOR), group(INFERIOR), group(OTHER_SITE)

    # A stay coded both walls is not a label, it is an ambiguity; excluded
    # rather than assigned to whichever appears first.
    clean = (anterior ^ inferior) - other
    return pd.Series({s: (1 if s in anterior else 0) for s in clean}, name="territory")


def assemble() -> Tuple[pd.DataFrame, List[str]]:
    y = build_label()
    feats = pd.read_parquet(os.path.join(ARTIFACTS, "features_H24.parquet"))
    split = pd.read_parquet(os.path.join(ARTIFACTS, "split_assignment.parquet"))
    split = split[["stay_id", "fold"]]

    df = feats[feats["stay_id"].isin(y.index)].copy()
    df["territory"] = df["stay_id"].map(y).astype(int)
    df = df.merge(split, on="stay_id", how="left")
    if df["fold"].isna().any():
        raise SystemExit("some labelled stays have no fold assignment")

    columns = [c for c in df.columns
               if c not in NEVER and pd.api.types.is_numeric_dtype(df[c])]
    return df, columns


# ------------------------------------------------------------- statistics ---
def wilson(hits: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval. Normal approximation is unusable at n ~ 40."""
    if total == 0:
        return (float("nan"), float("nan"))
    p = hits / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def recalls(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict]:
    out = {}
    for value, name in ((1, "anterior"), (0, "inferior")):
        mask = y_true == value
        total = int(mask.sum())
        hits = int((y_pred[mask] == value).sum())
        low, high = wilson(hits, total)
        out[name] = {"recall": hits / total if total else float("nan"),
                     "n": total, "hits": hits, "ci": [low, high]}
    return out


# ---------------------------------------------------------------- training ---
def _make(seed: int, kind: str, n_pos: int, n_neg: int):
    """One member of the ensemble.

    Capacity is set for the data, not by habit. 475 rows against 219 features
    is a regime where a 500-tree model memorises the training fold: depth is
    capped, leaves must hold real support, and both row and column subsampling
    are aggressive.

    Class weights matter more than they look. The training fold runs 1:1.12
    anterior to inferior while validation and test run 1:1.42 -- the patient-
    grouped split did not preserve the prior. Left uncorrected the model
    inherits the flatter training prior and systematically under-calls
    inferior, which is exactly the recall that came out lowest.
    """
    import lightgbm as lgb

    if kind == "lgb":
        return lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=7, max_depth=4,
            min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.5, reg_lambda=5.0, reg_alpha=1.0,
            class_weight="balanced", random_state=seed, verbose=-1)

    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=3,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.5,
        reg_lambda=5.0, reg_alpha=1.0,
        scale_pos_weight=n_neg / max(1, n_pos),
        random_state=seed, eval_metric="logloss",
        tree_method="hist", verbosity=0)


def fit_ensemble(train: pd.DataFrame, columns: Sequence[str],
                 seeds: Sequence[int]) -> Tuple[List, float]:
    """LightGBM + XGBoost per seed, matching the component's stage-1 blend."""
    try:
        import xgboost  # noqa: F401
        kinds = ("lgb", "xgb")
    except ImportError:
        kinds = ("lgb",)

    X = train[list(columns)]
    y = train["territory"].to_numpy(int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    models: List = []
    started = time.perf_counter()

    for seed in seeds:
        for kind in kinds:
            models.append(_make(seed, kind, n_pos, n_neg).fit(X, y))

    return models, time.perf_counter() - started


def predict(models: Sequence, frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    X = frame[list(columns)]
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def choose_threshold(probabilities: np.ndarray, truth: np.ndarray,
                     objective: str = "min_metric") -> float:
    """The cut that maximises the WORST class metric, on out-of-fold data.

    Accuracy would let the larger class carry the score, so it is not the
    objective. `min_recall` maximises the worse of the two recalls, which stops
    either wall being systematically missed -- but it lets precision drift,
    because a threshold that recalls anterior generously pays for it in
    anterior precision, and on a two-class problem that trade is invisible if
    only recall is watched.

    `min_metric` therefore maximises the worst of all four: both recalls and
    both precisions. It is the operating point at which no per-class number is
    quietly worse than the headline, which is what a reader checking the
    per-class table actually wants.

    Either way the choice is made on grouped out-of-fold predictions over
    train + val. The test fold is never consulted.
    """
    best, best_key = 0.5, (-1.0, -1.0)
    for cut in np.linspace(0.05, 0.95, 181):
        pred = (probabilities >= cut).astype(int)
        r = recalls(truth, pred)
        a, i = r["anterior"]["recall"], r["inferior"]["recall"]
        if math.isnan(a) or math.isnan(i):
            continue

        if objective == "min_recall":
            worst, mean = min(a, i), (a + i) / 2
        else:
            precisions = []
            for value in (1, 0):
                called = int((pred == value).sum())
                hit = int(((pred == value) & (truth == value)).sum())
                precisions.append(hit / called if called else 0.0)
            metrics = [a, i] + precisions
            worst, mean = min(metrics), float(np.mean(metrics))

        key = (worst, mean)
        if key > best_key:
            best, best_key = float(cut), key
    return best


def _out_of_fold(pool: pd.DataFrame, columns: Sequence[str],
                 seeds: Sequence[int]) -> np.ndarray:
    """Out-of-fold probabilities over train+val, grouped by patient.

    Grouped so the same subject_id never sits in both sides of a CV split; an
    ungrouped fold would leak a patient across the boundary and hand back an
    optimistic threshold.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    y = pool["territory"].to_numpy(int)
    groups = pool["subject_id"].to_numpy()
    out = np.zeros(len(pool), dtype=float)

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for tr_idx, te_idx in splitter.split(pool, y, groups):
        fold_train = pool.iloc[tr_idx]
        n_pos = int((y[tr_idx] == 1).sum())
        n_neg = int((y[tr_idx] == 0).sum())
        members = []
        for seed in seeds:
            members.append(_make(seed, "lgb", n_pos, n_neg).fit(
                fold_train[list(columns)], y[tr_idx]))
        out[te_idx] = np.mean(
            [m.predict_proba(pool.iloc[te_idx][list(columns)])[:, 1] for m in members],
            axis=0)
    return out


def evaluate(name: str, models, columns, val, test, train=None,
             seeds: Sequence[int] = (0,)) -> Dict:
    from sklearn.metrics import accuracy_score, roc_auc_score

    # The threshold decides both recalls, and 97 validation cases put a wide
    # interval on it. Out-of-fold predictions over train+val give it several
    # hundred instead, without the test fold being involved in the choice.
    if train is not None:
        pool = pd.concat([train, val], ignore_index=True)
        oof = _out_of_fold(pool, columns, seeds)
        threshold = choose_threshold(oof, pool["territory"].to_numpy(int))
    else:
        p_val = predict(models, val, columns)
        threshold = choose_threshold(p_val, val["territory"].to_numpy(int))

    p_test = predict(models, test, columns)
    y_test = test["territory"].to_numpy(int)
    pred = (p_test >= threshold).astype(int)

    r = recalls(y_test, pred)
    result = {
        "configuration": name,
        "n_features": len(columns),
        "threshold_from_validation": round(threshold, 3),
        "test_n": int(len(y_test)),
        "auroc": float(roc_auc_score(y_test, p_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float((r["anterior"]["recall"] + r["inferior"]["recall"]) / 2),
        "min_recall": float(min(r["anterior"]["recall"], r["inferior"]["recall"])),
        "per_class": r,
    }
    return result


def show(result: Dict) -> None:
    print()
    print("=" * 78)
    print("  %s   (%d features, threshold %.3f chosen on validation)"
          % (result["configuration"], result["n_features"],
             result["threshold_from_validation"]))
    print("=" * 78)
    print("  AUROC              %.4f" % result["auroc"])
    print("  accuracy           %.4f" % result["accuracy"])
    print("  balanced accuracy  %.4f" % result["balanced_accuracy"])
    print("  minimum recall     %.4f" % result["min_recall"])
    print()
    for wall in ("anterior", "inferior"):
        c = result["per_class"][wall]
        print("  %-9s recall %.4f  (%d/%d)   95%% CI [%.3f, %.3f]"
              % (wall, c["recall"], c["hits"], c["n"], c["ci"][0], c["ci"][1]))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", default=os.path.join(REPORTS, "territory_head.json"))
    args = parser.parse_args()
    seeds = list(range(args.seeds))

    df, all_columns = assemble()
    train = df[df["fold"] == "train"]
    val = df[df["fold"] == "val"]
    test = df[df["fold"] == "test"]

    print("STEMI territory head -- anterior (1) vs inferior (0)")
    print("  train %d   val %d   test %d" % (len(train), len(val), len(test)))
    for fold, part in (("train", train), ("val", val), ("test", test)):
        a = int((part["territory"] == 1).sum())
        i = int((part["territory"] == 0).sum())
        print("     %-6s anterior %3d   inferior %3d   ratio 1:%.2f"
              % (fold, a, i, max(a, i) / max(1, min(a, i))))

    physiology = [c for c in all_columns if c not in ECG_TERRITORY]
    results = []

    for name, columns in (("FULL -- every feature", all_columns),
                          ("PHYSIOLOGY -- no ECG-report territory", physiology)):
        models, seconds = fit_ensemble(train, columns, seeds)
        print()
        print("  fitted %d models in %.1f s" % (len(models), seconds))
        result = evaluate(name, models, columns, val, test,
                          train=train, seeds=seeds)
        result["fit_seconds"] = round(seconds, 2)
        results.append(result)
        show(result)

    full, phys = results[0], results[1]
    print()
    print("=" * 78)
    print("  WHAT THE ECG REPORT TEXT IS WORTH")
    print("=" * 78)
    print("  AUROC     full %.4f    physiology %.4f    difference %+.4f"
          % (full["auroc"], phys["auroc"], phys["auroc"] - full["auroc"]))
    print("  accuracy  full %.4f    physiology %.4f    difference %+.4f"
          % (full["accuracy"], phys["accuracy"], phys["accuracy"] - full["accuracy"]))
    print()
    print("  The FULL model includes three features parsed from the ECG cart's")
    print("  printed report. The coder read the same ECG when assigning the ICD")
    print("  code, so those features share a source with the label. Both numbers")
    print("  belong in any write-up; the second is what the model adds beyond")
    print("  reading the printout.")

    os.makedirs(REPORTS, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"seeds": seeds, "results": results}, handle, indent=2)
    print()
    print("  written: %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
