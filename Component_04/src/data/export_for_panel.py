"""Export Component 04's data as CSVs a reviewer can actually open.

WHY
---
The pipeline stores everything as Parquet, which is the right format to compute
on and the wrong one to show someone. A panel asking to see the columns and the
values cannot open a Parquet file, and the full training matrices are 92-97 MB
with 228 columns, which no spreadsheet displays usefully either.

So this writes two things:

  samples/     the first N rows of every table, as CSV. Small enough to open,
               complete enough to show what each column holds.
  DATA.md      every column of every table, its type, how many values are
               missing, and an example -- plus the join keys that connect them.

The full training CSVs already exist in `artifacts/data/splits_csv/` and stay
where they are; these are for showing, not for training.

USAGE
-----
    python src/data/export_for_panel.py
    python src/data/export_for_panel.py --rows 1000
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

COMPONENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED = os.path.join(COMPONENT, "data", "processed")
ARTIFACTS = os.path.join(COMPONENT, "artifacts", "data")
OUT = os.path.join(COMPONENT, "data_for_panel")

#: (file, layer, what it is, the key it joins the cohort on)
TABLES = [
    ("data/processed/master_data.parquet", 1,
     "One row per ED stay: demographics, triage vitals, chief complaint, and the "
     "four-class ACS label. This is the cohort; every other table joins to it.",
     "stay_id (and subject_id, hadm_id)"),
    ("data/processed/charlson.parquet", 1,
     "Comorbidity flags per admission, derived upstream from MIMIC's "
     "diagnoses_icd. Supplies the history channel.", "hadm_id"),
    ("data/processed/lab_values.parquet", 1,
     "Troponin and BNP results with their draw times. The laboratory channel, "
     "and the reason the disclosure horizon exists.", "stay_id"),
    ("data/processed/medrecon.parquet", 1,
     "Home medications recorded at triage. The medication channel.", "stay_id"),
    ("data/processed/ecg_records.parquet", 1,
     "Which ECG studies exist for a patient and when they were taken. Used to "
     "decide whether an ECG falls inside the horizon window.", "subject_id"),
    ("data/processed/ecg_measurements.parquet", 1,
     "The ECG cart's own printed report, eighteen text lines per study.",
     "subject_id + study_id"),
    ("data/processed/ecg_numeric.parquet", 1,
     "ECG intervals and axes: QRS duration, PR, QT, P/QRS/T axis.",
     "subject_id + study_id"),
    ("data/processed/lab_discovery.csv", 1,
     "The five MIMIC itemids that count as troponin or BNP. A lookup, not data.",
     "itemid"),
    ("data/processed/verification_report.csv", 1,
     "Cohort counts written when the extraction ran, so the numbers can be "
     "checked against the source.", "-"),
    ("artifacts/data/features_H0.parquet", 2,
     "The 228-column feature matrix at the triage horizon: only what is "
     "knowable at H = 0.", "stay_id"),
    ("artifacts/data/features_H6.parquet", 2,
     "The same cohort re-featurised at H = 6 h, after the first troponin.",
     "stay_id"),
    ("artifacts/data/features_H24.parquet", 2,
     "The same cohort at H = 24 h, the full workup. This is the headline model's "
     "input.", "stay_id"),
    ("artifacts/data/split_assignment.parquet", 2,
     "Which fold each stay belongs to. Grouped by patient, so no subject_id "
     "appears in two folds.", "stay_id"),
    ("artifacts/data/ecg_manifest.parquet", 2,
     "ECG studies matched to ED stays inside the lookback window.",
     "stay_id + study_id"),
    ("artifacts/data/ecg_waveform_features.parquet", 2,
     "Waveform-derived features. NOT used by the trained model -- none of its "
     "40 columns appear in any features_H*.parquet.", "stay_id + study_id"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=500)
    args = parser.parse_args()

    samples = os.path.join(OUT, "samples")
    os.makedirs(samples, exist_ok=True)

    lines = [
        "# Component 04 — the data, and how it connects",
        "",
        "**Temporally-Safe Explainable ACS Triage** · Abishnan J (IT22140234)",
        "",
        "Source: **MIMIC-IV-ED + Hosp + ECG** (PhysioNet, credentialed).",
        "",
        "> The tables here are extracts of credentialed data under a PhysioNet",
        "> data use agreement. They are not redistributable and are gitignored.",
        "",
        "---",
        "",
        "## How the tables connect",
        "",
        "```",
        "  MIMIC-IV (BigQuery)",
        "        |",
        "        v",
        "  LAYER 1  data/processed/          one file per source table",
        "        |                            joined on subject_id / hadm_id / stay_id",
        "        v",
        "  LAYER 2  artifacts/data/          features_H0 / H6 / H24  (228 columns)",
        "        |                            + split_assignment (train/val/test)",
        "        v",
        "  LAYER 3  artifacts/data/splits_csv/   the training data, as CSV",
        "```",
        "",
        "`master_data.parquet` is the spine: one row per ED stay, carrying the",
        "`acs_label`. Every other Layer 1 table joins onto it, and the join key",
        "differs by table because MIMIC records them at different levels — a lab",
        "belongs to a stay, a comorbidity to an admission, an ECG to a patient.",
        "",
        "| Table | Joins on | Reaches |",
        "|---|---|---|",
        "| charlson | `hadm_id` | 100.0 % of admissions |",
        "| medrecon | `stay_id` | 79.7 % of stays |",
        "| ecg_records / measurements / numeric | `subject_id` | 71.7 % of patients |",
        "| lab_values | `stay_id` | 13.1 % of stays |",
        "",
        "**The lab coverage is the point, not a defect.** Only 13.1 % of ED stays",
        "have a troponin or BNP drawn, because most ED patients are not being",
        "worked up for a cardiac cause. A model that required one would silently",
        "exclude everyone else.",
        "",
        "---",
        "",
        "## The three horizons",
        "",
        "The same 203,016 stays are featurised three times, at H = 0, 6 and 24",
        "hours after arrival. Same cohort, same split, same code — only the",
        "window of admissible information changes. That is what makes",
        "accuracy-versus-time reportable instead of a single unstated number.",
        "",
        "---",
        "",
        "## Train / validation / test",
        "",
        "`split_assignment.parquet` holds 203,016 rows of `stay_id, subject_id,",
        "fold`:",
        "",
        "| Fold | Stays | Share |",
        "|---|---|---|",
        "| train | 142,111 | 70 % |",
        "| val | 30,453 | 15 % |",
        "| test | 30,452 | 15 % |",
        "",
        "Grouped by patient: no `subject_id` appears in more than one fold, so a",
        "patient with several ED visits cannot be in training and test at once.",
        "",
        "The full matrices are written as CSV to `artifacts/data/splits_csv/` —",
        "nine files, `H{0,6,24}_{train,val,test}.csv`, regenerated by",
        "`src/data/export_splits.py`.",
        "",
        "---",
        "",
        "## Every table, every column",
        "",
        "Samples of the first %d rows are in `samples/`." % args.rows,
        "",
    ]

    print("writing samples to %s" % samples)
    for rel, layer, purpose, key in TABLES:
        path = os.path.join(COMPONENT, rel)
        name = os.path.basename(rel)
        if not os.path.exists(path):
            print("  %-34s absent, skipped" % name)
            continue

        if path.endswith(".parquet"):
            full = pd.read_parquet(path)
        else:
            full = pd.read_csv(path)

        stem = os.path.splitext(name)[0]
        out_csv = os.path.join(samples, stem + "_sample.csv")
        full.head(args.rows).to_csv(out_csv, index=False)
        print("  %-34s %8s rows x %3d cols -> %s"
              % (name, "{:,}".format(len(full)), len(full.columns),
                 os.path.basename(out_csv)))

        lines += [
            "### `%s`" % name,
            "",
            "*Layer %d.* %s" % (layer, purpose),
            "",
            "**%s rows · %d columns · joins on %s**"
            % ("{:,}".format(len(full)), len(full.columns), key),
            "",
            "| Column | Type | Missing | Example |",
            "|---|---|---|---|",
        ]

        shown = list(full.columns) if len(full.columns) <= 30 else list(full.columns[:30])
        for col in shown:
            series = full[col]
            missing = 100 * series.isna().mean()
            example = series.dropna().iloc[0] if series.notna().any() else ""
            example = str(example).replace("|", "/")[:44]
            lines.append("| `%s` | %s | %.1f %% | %s |"
                         % (col, series.dtype, missing, example))
        if len(full.columns) > 30:
            lines.append("| … | | | *%d more columns — see the sample CSV* |"
                         % (len(full.columns) - 30))
        lines += ["", "Sample: `samples/%s`" % os.path.basename(out_csv), "", "---", ""]

    doc = os.path.join(OUT, "DATA.md")
    with open(doc, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print()
    print("wrote %s" % doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
