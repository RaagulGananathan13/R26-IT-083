"""Write the train / validation / test folds out as CSV.

WHY THIS EXISTS
---------------
The pipeline never materialises three files per horizon. It keeps one feature
matrix per disclosure horizon and a single `split_assignment.parquet` naming
the fold each ED stay belongs to, and joins them at load time. That is the
right internal design -- one row per stay, one place where the split is
defined, no chance of three files drifting apart -- but it means there is no
`train.csv` to hand to anyone who asks to see the data.

This writes them, from the same assignment table the models were trained
against, so the exported files cannot disagree with what was fitted.

INTEGRITY
---------
The split is grouped by patient, not by stay: a `subject_id` with several ED
visits must land wholly in one fold or the model sees the same patient in
training and in test. That is checked here rather than assumed, because it is
the one property of the split that silently invalidates every reported number
if it is wrong, and it costs a second to verify.

USAGE
-----
    python src/data/export_splits.py                 # all three horizons
    python src/data/export_splits.py --horizon 24    # just the headline one
    python src/data/export_splits.py --out somewhere
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

COMPONENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS = os.path.join(COMPONENT, "artifacts", "data")
FOLDS = ("train", "val", "test")


def load_assignment() -> pd.DataFrame:
    path = os.path.join(ARTIFACTS, "split_assignment.parquet")
    if not os.path.exists(path):
        raise SystemExit("split_assignment.parquet not found at %s" % path)
    return pd.read_parquet(path)


def check_no_patient_crosses_folds(assignment: pd.DataFrame) -> None:
    """A subject_id in two folds means the test score is not a test score."""
    per_subject = assignment.groupby("subject_id")["fold"].nunique()
    leaked = per_subject[per_subject > 1]
    if len(leaked):
        raise SystemExit(
            "%d subject_id values appear in more than one fold. The split is "
            "not patient-grouped and every reported number is suspect. "
            "First few: %s" % (len(leaked), list(leaked.index[:5])))
    print("  patient grouping OK: no subject_id spans two folds")


def export(horizon: int, assignment: pd.DataFrame, out_dir: str) -> None:
    features_path = os.path.join(ARTIFACTS, "features_H%d.parquet" % horizon)
    if not os.path.exists(features_path):
        print("  H=%-2d skipped: %s not present" % (horizon, os.path.basename(features_path)))
        return

    features = pd.read_parquet(features_path)
    merged = features.merge(assignment[["stay_id", "fold"]], on="stay_id", how="left")

    unassigned = int(merged["fold"].isna().sum())
    if unassigned:
        # Silently dropping these would quietly shrink the dataset; say so.
        print("  H=%-2d WARNING: %d rows have no fold and are excluded"
              % (horizon, unassigned))
        merged = merged[merged["fold"].notna()]

    os.makedirs(out_dir, exist_ok=True)
    for fold in FOLDS:
        subset = merged[merged["fold"] == fold].drop(columns=["fold"])
        path = os.path.join(out_dir, "H%d_%s.csv" % (horizon, fold))
        subset.to_csv(path, index=False)
        size = os.path.getsize(path) / 1e6
        positives = ""
        if "acs_label" in subset.columns:
            counts = subset["acs_label"].value_counts().sort_index()
            positives = "   labels " + " ".join(
                "%s=%d" % (k, v) for k, v in counts.items())
        print("  H=%-2d %-5s %7d rows  %4d cols  %7.1f MB%s"
              % (horizon, fold, len(subset), subset.shape[1], size, positives))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--horizon", type=int, choices=[0, 6, 24], default=None,
                        help="Export one horizon only. Default: all three.")
    parser.add_argument("--out", default=os.path.join(ARTIFACTS, "splits_csv"))
    args = parser.parse_args()

    assignment = load_assignment()
    print("split_assignment.parquet: %d stays" % len(assignment))
    for fold, count in assignment["fold"].value_counts().reindex(FOLDS).items():
        print("  %-5s %7d  (%.1f %%)" % (fold, count, 100 * count / len(assignment)))
    check_no_patient_crosses_folds(assignment)

    print()
    print("writing to %s" % args.out)
    for horizon in ([args.horizon] if args.horizon is not None else [0, 6, 24]):
        export(horizon, assignment, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
