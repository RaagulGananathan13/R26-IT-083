"""Is STEMI territory learnable on this cohort? Answer it from the ICD tables.

WHY THIS EXISTS
---------------
The extraction that built `data/processed/` collapsed every acute-coronary ICD
code into a single four-value `acs_label` (No_ACS / UA / NSTEMI / STEMI). Wall
location is in none of the eight extracted tables -- an exhaustive scan for
`^I2[01]` across every object column returns nothing -- so "could we predict
STEMI sub-type?" cannot be answered from what is on disk.

It is answerable from two MIMIC-IV files, and it is a counting exercise rather
than a modelling one: if the smallest class holds single-digit counts in the
test fold, no per-class recall computed from it would mean anything, and that
is worth knowing before any training is attempted.

TWO THINGS THAT MAKE THIS EASY TO GET WRONG
-------------------------------------------
1. WHO ICD-10 and ICD-10-CM are not the same code set. The WHO browser lists
   I21.0 anterior / I21.1 inferior / I21.2 other. MIMIC uses the US clinical
   modification, which subdivides those by CULPRIT ARTERY instead:

       I21.01 left main            I21.11 right coronary
       I21.02 left anterior desc.  I21.19 other artery, inferior wall
       I21.09 other artery, ant.   I21.21 left circumflex
                                   I21.29 other sites

   Matching the WHO codes exactly returns zero anterior and zero inferior
   cases, which looks like missing data and is really a wrong code list.

2. MIMIC-IV straddles the ICD-9 to ICD-10 transition. In this cohort 467 of
   the 941 STEMI stays are coded in ICD-10 and the other 474 in ICD-9 -- almost
   exactly half. Dropping ICD-9 "to avoid mixing vocabularies" therefore throws
   away half the labels, and 410.x carries wall location in its fourth digit
   at least as richly as ICD-10-CM does.

WHAT TO DOWNLOAD
----------------
From PhysioNet (https://physionet.org/content/mimiciv/), the `hosp` module:

    hosp/diagnoses_icd.csv.gz      the codes, per admission
    hosp/d_icd_diagnoses.csv.gz    the dictionary, code -> long title

Put both in `Component_04/data/mimic_icd/` (or pass --icd-dir). Nothing else is
needed: every stay in this cohort has a `hadm_id`.

USAGE
-----
    python src/data/icd_subtypes.py
    python src/data/icd_subtypes.py --primary-only     # seq_num == 1 only
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Set, Tuple

import pandas as pd

COMPONENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED = os.path.join(COMPONENT, "data", "processed")
ARTIFACTS = os.path.join(COMPONENT, "artifacts", "data")

LABELS = {0: "No_ACS", 1: "UA", 2: "NSTEMI", 3: "STEMI"}

#: ICD-10-CM STEMI codes grouped by the wall the artery supplies, and the
#: equivalent ICD-9 410.x prefixes. The ICD-9 fourth digit is the site; the
#: fifth is episode of care, which is not a territory and is ignored.
TERRITORY: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    # name          ICD-10-CM prefixes            ICD-9 prefixes
    "anterior":    (("I2101", "I2102", "I2109"), ("4100", "4101")),
    "inferior":    (("I2111", "I2119"),          ("4102", "4103", "4104")),
    "other site":  (("I2121", "I2129"),          ("4105", "4106", "4108")),
    "unspecified": (("I213", "I219"),            ("4109",)),
}


def _read(icd_dir: str, stem: str) -> pd.DataFrame:
    for name in (stem + ".csv.gz", stem + ".csv"):
        path = os.path.join(icd_dir, name)
        if os.path.exists(path):
            df = pd.read_csv(path, dtype={"icd_code": str, "icd_version": "Int64"})
            df["icd_code"] = df["icd_code"].str.strip().str.upper()
            return df
    raise SystemExit(
        "%s not found in %s%sDownload hosp/%s.csv.gz from PhysioNet."
        % (stem, icd_dir, os.linesep, stem))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--icd-dir", default=os.path.join(COMPONENT, "data", "mimic_icd"))
    parser.add_argument("--primary-only", action="store_true",
                        help="Count only seq_num == 1, the admission's main diagnosis.")
    args = parser.parse_args()

    master = pd.read_parquet(os.path.join(PROCESSED, "master_data.parquet"))
    icd = _read(args.icd_dir, "diagnoses_icd")
    titles = _read(args.icd_dir, "d_icd_diagnoses")

    joined = icd.merge(master[["subject_id", "hadm_id", "stay_id", "acs_label"]],
                       on=["subject_id", "hadm_id"], how="inner")
    if args.primary_only:
        joined = joined[joined["seq_num"] == 1]

    stemi_stays: Set = set(master.loc[master["acs_label"] == 3, "stay_id"])
    v10 = joined[joined["icd_version"] == 10]
    v9 = joined[joined["icd_version"] == 9]

    print("cohort: %d ED stays, all with an admission" % len(master))
    print("labelled STEMI stays: %d" % len(stemi_stays))
    got10 = stemi_stays & set(v10.loc[v10["icd_code"].str.startswith("I21", na=False),
                                      "stay_id"])
    got9 = stemi_stays & set(v9.loc[v9["icd_code"].str.startswith("410", na=False),
                                    "stay_id"])
    print("   coded in ICD-10 (I21*)  : %d" % len(got10))
    print("   coded in ICD-9  (410.x) : %d" % len(got9))
    print("   coded in neither        : %d" % len(stemi_stays - (got10 | got9)))

    title_of = dict(zip(titles["icd_code"] + "|" + titles["icd_version"].astype(str),
                        titles["long_title"]))

    print()
    print("=" * 78)
    print("  STEMI TERRITORY, BOTH VOCABULARIES")
    print("=" * 78)

    groups: Dict[str, Set] = {}
    for name, (p10, p9) in TERRITORY.items():
        a = set(v10.loc[v10["icd_code"].str.startswith(p10, na=False), "stay_id"])
        b = set(v9.loc[v9["icd_code"].str.startswith(p9, na=False), "stay_id"])
        groups[name] = (a | b) & stemi_stays

    split_path = os.path.join(ARTIFACTS, "split_assignment.parquet")
    fold_of: Dict = {}
    if os.path.exists(split_path):
        split = pd.read_parquet(split_path)[["stay_id", "fold"]]
        fold_of = dict(zip(split["stay_id"], split["fold"]))

    print("  %-13s %7s %6s %6s %8s" % ("territory", "train", "val", "test", "total"))
    for name, stays in groups.items():
        folds = pd.Series([fold_of.get(s) for s in stays]).value_counts()
        print("  %-13s %7d %6d %6d %8d"
              % (name, folds.get("train", 0), folds.get("val", 0),
                 folds.get("test", 0), len(stays)))

    # -- the decision ----------------------------------------------------
    both = groups["anterior"] & groups["inferior"]
    clean = (groups["anterior"] ^ groups["inferior"]) - groups["other site"]
    folds = pd.Series([fold_of.get(s) for s in clean]).value_counts()

    print()
    print("=" * 78)
    print("  ANTERIOR vs INFERIOR, THE ONLY SPLIT WITH ENOUGH CASES")
    print("=" * 78)
    print("  coded as both, excluded : %d" % len(both))
    print("  cleanly one of the two  : %d" % len(clean))
    print("     train %d   val %d   test %d"
          % (folds.get("train", 0), folds.get("val", 0), folds.get("test", 0)))

    smallest_test = min(
        sum(1 for s in groups[n] if fold_of.get(s) == "test")
        for n in ("anterior", "inferior"))
    print()
    print("  smallest of the two in the test fold: %d" % smallest_test)
    if smallest_test < 30:
        print("  Too few to report a per-class recall anyone should believe.")
    else:
        print("  Enough to report a per-class recall with a usable interval.")
        print("  The four-way split is not: 'other site' and 'unspecified' are")
        print("  not anatomical answers, they are the absence of one.")

    print()
    print("  Codes behind these groups:")
    for name, (p10, p9) in TERRITORY.items():
        seen = []
        for pref, ver in [(p10, 10), (p9, 9)]:
            frame = v10 if ver == 10 else v9
            sub = frame[frame["icd_code"].str.startswith(pref, na=False)]
            sub = sub[sub["stay_id"].isin(stemi_stays)]
            for code, n in sub.groupby("icd_code")["stay_id"].nunique().items():
                seen.append("%s(%d)" % (code, n))
        print("    %-13s %s" % (name, " ".join(sorted(seen)) or "-"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
