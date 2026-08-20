"""
Component 04 — patient-level grouped, stratified splitting.

The original pipeline used sklearn.train_test_split with stratify=y.  31.1% of
subjects in this cohort have more than one ED stay, so the same patient — with
correlated vitals, the same comorbidity burden and often the same chief
complaint — appeared in both train and test.  That inflates every metric.

Here every subject_id lands in exactly one fold.  Because the rarest class has
only ~700 cases we cannot rely on random group assignment to balance the folds,
so we use a greedy label-histogram balancing pass:

  1. group stays by subject_id and build each subject's label histogram
  2. sort subjects by rarity of their rarest class, then by size (hardest first)
  3. assign each subject to the fold with the largest current deficit against
     its target share, weighted towards the rare classes

This is deterministic, leakage-free, and produces near-exact class proportions.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_p for _p in (
    _SRC, os.path.join(_SRC, "core"), os.path.join(_SRC, "data"),
    os.path.join(_SRC, "models"), os.path.join(_SRC, "analysis"),
) if _p not in sys.path]

from config import CFG, DATA_DIR, LABEL_MAP, REPORT_DIR, enable_utf8_stdout, save_json
from utils import banner, kv, section

enable_utf8_stdout()

FOLDS = ("train", "val", "test")


def grouped_stratified_split(
    df: pd.DataFrame,
    group_col: str = "subject_id",
    label_col: str = "acs_label",
    fractions: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> pd.Series:
    n_classes = int(df[label_col].max()) + 1
    rng = np.random.RandomState(seed)

    # --- per-subject label histogram --------------------------------------
    hist = (
        df.groupby([group_col, label_col]).size()
        .unstack(fill_value=0)
        .reindex(columns=range(n_classes), fill_value=0)
    )
    counts = hist.to_numpy(dtype=np.float64)
    subjects = hist.index.to_numpy()

    totals = counts.sum(axis=0)
    fr = np.asarray(fractions, dtype=np.float64)
    # remaining[f, k] = how many stays of class k fold f still wants
    remaining = np.outer(fr, totals)                        # (3, n_classes)
    remaining_total = fr * totals.sum()

    # Class rarity ranking: UA (739) is rarer than STEMI (941) etc.
    rarity_rank = np.argsort(np.argsort(totals))            # 0 = rarest

    # Process the subjects that carry the rarest classes first, and among
    # those the largest ones first — the hardest placements get first pick of
    # the remaining quota, which is what keeps the tail classes balanced.
    subj_rarest = np.where(
        counts > 0, rarity_rank[None, :], np.iinfo(np.int32).max
    ).min(axis=1)
    jitter = rng.rand(len(subjects))
    order = np.lexsort((jitter, -counts.sum(axis=1), subj_rarest))

    fold_of = np.empty(len(subjects), dtype=np.int8)

    for i in order:
        c = counts[i]
        present = np.flatnonzero(c)
        # the rarest class this subject contributes decides the fold
        k_star = present[np.argmin(rarity_rank[present])]

        cand = np.flatnonzero(remaining[:, k_star] == remaining[:, k_star].max())
        if len(cand) > 1:                       # tie -> largest overall deficit
            sub = remaining_total[cand]
            cand = cand[sub == sub.max()]
        f = int(cand[0]) if len(cand) == 1 else int(rng.choice(cand))

        fold_of[i] = f
        remaining[f] -= c
        remaining_total[f] -= c.sum()

    mapping = pd.Series([FOLDS[f] for f in fold_of], index=subjects)
    return df[group_col].map(mapping)


def verify(df: pd.DataFrame, fold: pd.Series, group_col: str = "subject_id") -> Dict:
    section("Split verification")
    out: Dict = {"folds": {}}

    # 1. no subject appears in two folds
    per_subject_folds = fold.groupby(df[group_col]).nunique()
    overlap = int((per_subject_folds > 1).sum())
    kv("subjects spanning >1 fold", f"{overlap}   " +
       ("[PASS - zero patient leakage]" if overlap == 0 else "[FAIL]"))
    out["patient_overlap"] = overlap
    assert overlap == 0, "patient-level leakage detected"

    # 2. no hadm_id spans folds (a hadm can have >1 ED stay)
    if "hadm_id" in df.columns:
        h = df.dropna(subset=["hadm_id"])
        hover = int((fold[h.index].groupby(h["hadm_id"]).nunique() > 1).sum())
        kv("admissions spanning >1 fold", f"{hover}   " +
           ("[PASS]" if hover == 0 else "[WARN]"))
        out["hadm_overlap"] = hover

    # 3. class proportions
    print()
    print(f"  {'fold':<8}{'n':>10}{'patients':>10}" +
          "".join(f"{LABEL_MAP[k]:>11}" for k in sorted(LABEL_MAP)))
    print("  " + "-" * (28 + 11 * len(LABEL_MAP)))
    for f in FOLDS:
        m = fold == f
        sub = df[m]
        row = f"  {f:<8}{int(m.sum()):>10,}{sub[group_col].nunique():>10,}"
        for k in sorted(LABEL_MAP):
            n = int((sub["acs_label"] == k).sum())
            row += f"{n:>6,} ({n/max(len(sub),1)*100:4.2f}%)".rjust(11)
        print(row)
        out["folds"][f] = {
            "n": int(m.sum()),
            "n_patients": int(sub[group_col].nunique()),
            "labels": {LABEL_MAP[k]: int((sub["acs_label"] == k).sum())
                       for k in sorted(LABEL_MAP)},
        }

    # 4. proportion drift vs the pooled distribution
    pooled = df["acs_label"].value_counts(normalize=True).sort_index()
    drift = 0.0
    for f in FOLDS:
        p = df.loc[fold == f, "acs_label"].value_counts(normalize=True).sort_index()
        drift = max(drift, float((p - pooled).abs().max()))
    kv("\n  max class-proportion drift", f"{drift*100:.4f} pp   " +
       ("[PASS]" if drift < 0.005 else "[CHECK]"))
    out["max_proportion_drift"] = drift
    return out


def main() -> None:
    banner("PATIENT-LEVEL GROUPED STRATIFIED SPLIT")
    fractions = (
        float(CFG.get("split.train", 0.70)),
        float(CFG.get("split.val", 0.15)),
        float(CFG.get("split.test", 0.15)),
    )
    group_col = str(CFG.get("split.group_col", "subject_id"))

    # The split is defined once, on the primary horizon, and reused verbatim by
    # every other horizon so that horizon comparisons are strictly paired.
    ref_path = os.path.join(DATA_DIR, f"features_H{CFG.primary_horizon}.parquet")
    ref = pd.read_parquet(ref_path, columns=["subject_id", "hadm_id", "stay_id",
                                             "acs_label", "in_cohort"])
    kv("reference matrix", os.path.basename(ref_path))
    kv("rows", f"{len(ref):,}")
    kv("patients", f"{ref[group_col].nunique():,}")
    kv("fractions", f"train={fractions[0]:.2f} val={fractions[1]:.2f} test={fractions[2]:.2f}")

    fold = grouped_stratified_split(ref, group_col=group_col,
                                    fractions=fractions, seed=CFG.seed)
    report = verify(ref, fold, group_col=group_col)

    section("Cohort (Intended Use Population) composition per fold")
    print(f"  {'fold':<8}{'n':>10}" + "".join(f"{LABEL_MAP[k]:>10}" for k in sorted(LABEL_MAP)))
    coh = ref[ref.in_cohort == 1]
    cfold = fold[coh.index]
    report["cohort_folds"] = {}
    for f in FOLDS:
        s = coh[cfold == f]
        print(f"  {f:<8}{len(s):>10,}" +
              "".join(f"{int((s.acs_label==k).sum()):>10,}" for k in sorted(LABEL_MAP)))
        report["cohort_folds"][f] = {
            LABEL_MAP[k]: int((s.acs_label == k).sum()) for k in sorted(LABEL_MAP)}

    assign = pd.DataFrame({"stay_id": ref["stay_id"], "subject_id": ref["subject_id"],
                           "fold": fold.to_numpy()})
    out_path = os.path.join(DATA_DIR, "split_assignment.parquet")
    assign.to_parquet(out_path, index=False)
    save_json(report, os.path.join(REPORT_DIR, "split_report.json"))
    print(f"\n  [SAVED] {out_path}")
    banner("SPLIT COMPLETE")


if __name__ == "__main__":
    main()
