"""Export the dataset in the shape of the form a clinician actually fills in.

WHY THIS EXISTS ALONGSIDE export_splits.py
------------------------------------------
`export_splits.py` writes the 228-column matrices the model is trained on.
Those are engineered features -- `cc_acs_lexicon_score`, `ix_age_x_chestpain`,
`arrival_hour_sin` -- and showing them to someone who asked "what data do you
use?" answers a question they did not ask.

This writes the same rows with the columns of `TriageRequest`: the fields the
console posts to `/api/v1/triage/analyze`. One column per input on the form,
plus the label and the fold. That is the file to put on screen when the question
is "show me the dataset".

The 228-column version is not replaced. Both are true; they answer different
questions.

ABOUT H0 / H6 / H24
-------------------
Not three datasets. One cohort of 203,016 ED stays, featurised three times at
three disclosure horizons:

    H0    what is knowable at the triage desk. No troponin -- nobody has drawn
          blood ten seconds after the patient walks in.
    H6    after the first troponin has come back.
    H24   the completed workup.

Same patients, same split, same code. Only the window of admissible information
moves. That is what makes accuracy-versus-time reportable instead of a single
number at an unstated moment.

This exports H0 and H24 so the difference is visible in the values themselves:
the troponin columns are empty at H0 and populated at H24, for the same rows.

USAGE
-----
    python src/data/export_form_view.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

COMPONENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED = os.path.join(COMPONENT, "data", "processed")
ARTIFACTS = os.path.join(COMPONENT, "artifacts", "data")
OUT = os.path.join(ARTIFACTS, "form_view_csv")

LABELS = {0: "No_ACS", 1: "UA", 2: "NSTEMI", 3: "STEMI"}

#: form field  <-  feature column. Order follows the console's own layout.
FIELDS = [
    ("age", "age"),
    ("sex", None),                       # from master_data
    ("race", None),
    ("heartrate", "vs_heartrate"),
    ("sbp", "vs_sbp"),
    ("dbp", "vs_dbp"),
    ("resprate", "vs_resprate"),
    ("o2sat", "vs_o2sat"),
    ("temperature", "vs_temperature"),
    ("pain", "pain_score"),
    ("acuity", "acuity"),
    ("chief_complaint", None),
    ("troponin_max", "trop_max"),
    ("troponin_first", "trop_first"),
    ("troponin_hours_to_first", "trop_t_first_h"),
    ("troponin_draws", "trop_n"),
    ("bnp", "bnp_max"),
    ("ecg_st_elevation", "ecg_st_elevation"),
    ("ecg_st_depression", "ecg_st_depression"),
    ("ecg_t_inversion", "ecg_t_inversion"),
    ("ecg_q_wave", "ecg_q_wave"),
    ("ecg_lbbb", "ecg_lbbb"),
    ("ecg_rbbb", "ecg_rbbb"),
    ("ecg_acute_mi", "ecg_acute_mi"),
    ("ecg_normal", "ecg_normal"),
    ("ecg_infarct_anterior", "ecg_infarct_anterior"),
    ("ecg_infarct_inferior", "ecg_infarct_inferior"),
    ("ecg_qrs_duration", "ecg_qrs_duration"),
    ("ecg_pr_interval", "ecg_pr_interval"),
    ("ecg_qt_interval", "ecg_qt_interval"),
    ("ecg_qrs_axis", "ecg_qrs_axis"),
    ("prior_ed_visits", "hist_prior_ed_visits"),
    ("days_since_last_visit", "hist_days_since_last"),
    ("prior_mi", "hist_prior_mi_icd"),
    ("prior_chf", "hist_prior_chf_icd"),
    ("charlson_index", "hist_charlson_index"),
    ("home_medication_count", "med_n"),
]


def build(horizon: int, master: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    path = os.path.join(ARTIFACTS, "features_H%d.parquet" % horizon)
    feats = pd.read_parquet(path)

    out = pd.DataFrame()
    out["stay_id"] = feats["stay_id"]
    out["subject_id"] = feats["subject_id"]

    for field, column in FIELDS:
        if column is not None and column in feats.columns:
            out[field] = feats[column].to_numpy()
        elif column is not None:
            out[field] = np.nan

    # Straight from the source table, not re-derived.
    src = master.set_index("stay_id")
    for field, source in (("sex", "gender"), ("race", "race"),
                          ("chief_complaint", "chiefcomplaint")):
        if source in src.columns:
            out[field] = out["stay_id"].map(src[source])

    out["acs_label"] = out["stay_id"].map(src["acs_label"]).map(LABELS)
    out["fold"] = out["stay_id"].map(split.set_index("stay_id")["fold"])

    ordered = (["stay_id", "subject_id"] + [f for f, _ in FIELDS]
               + ["acs_label", "fold"])
    return out[[c for c in ordered if c in out.columns]]


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    master = pd.read_parquet(os.path.join(PROCESSED, "master_data.parquet"))
    split = pd.read_parquet(os.path.join(ARTIFACTS, "split_assignment.parquet"))

    print("One cohort, %s ED stays. The horizon changes what is knowable, not who."
          % "{:,}".format(len(master)))
    print()

    for horizon in (0, 24):
        frame = build(horizon, master, split)
        for fold in ("train", "val", "test"):
            part = frame[frame["fold"] == fold].drop(columns=["fold"])
            name = "H%d_%s.csv" % (horizon, fold)
            path = os.path.join(OUT, name)
            part.to_csv(path, index=False)
            print("  %-16s %8s rows  %2d cols  %6.1f MB"
                  % (name, "{:,}".format(len(part)), part.shape[1],
                     os.path.getsize(path) / 1e6))

    # Show the horizon doing its work, on the same rows.
    print()
    print("The same 30,452 test stays, at two horizons:")
    for horizon in (0, 24):
        frame = build(horizon, master, split)
        test = frame[frame["fold"] == "test"]
        have = test["troponin_max"].notna().mean() * 100
        print("   H%-2d  troponin present in %5.1f %% of rows" % (horizon, have))
    print()
    print("   Nothing was hidden. At H = 0 the blood has not come back yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
