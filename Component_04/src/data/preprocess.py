"""
Component 04 — temporally-safe multimodal preprocessing.  Built from scratch.

Anchor:  T0 = ED arrival (edstays.intime).
Rule:    a feature may only use information that physically exists at or
         before T0 + H, where H is the disclosure horizon.

Modalities
  M1 vitals        triage observations, recorded at T0
  M2 demographics  age / sex / race group / arrival hour
  M3 text          chief complaint  (see text_features.py)
  M4 medications   medrecon = home-medication reconciliation, done at arrival
  M5 history       derived ONLY from the patient's strictly-earlier ED stays
  M6 ecg           nearest ECG in [T0 - lookback, T0 + H]
  M7 labs          troponin / natriuretic peptide charted in [T0, T0 + H]

Every modality also emits an availability channel (`*_available`) and, where
meaningful, a latency channel (`*_t_first_h`).  Missingness in an ED record is
informative — a troponin that was ordered is a clinician expressing suspicion —
so we encode it rather than impute it away.
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
import text_features as TF
from utils import banner, kv, section, timer

enable_utf8_stdout()

# --------------------------------------------------------------------------
# Physiological plausibility bounds.  Values outside are set to NaN (recording
# error) rather than clipped, except where clipping is the safer assumption.
# --------------------------------------------------------------------------
VITAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "heartrate":   (20.0, 300.0),
    "sbp":         (40.0, 300.0),
    "dbp":         (15.0, 200.0),
    "resprate":    (4.0, 80.0),
    "o2sat":       (40.0, 100.0),
    "temperature": (90.0, 110.0),   # Fahrenheit in MIMIC-IV-ED
}

RACE_GROUPS: List[Tuple[str, str]] = [
    ("WHITE", "white"),
    ("BLACK", "black"),
    ("HISPANIC", "hispanic"),
    ("ASIAN", "asian"),
    ("AMERICAN INDIAN", "other"),
    ("NATIVE HAWAIIAN", "other"),
    ("PORTUGUESE", "white"),
    ("SOUTH AMERICAN", "hispanic"),
]

MED_CLASSES: Dict[str, str] = {
    "med_antiplatelet":   r"aspirin|clopidogrel|plavix|prasugrel|effient|ticagrelor|brilinta|dipyridamole",
    "med_statin":         r"atorvastatin|rosuvastatin|simvastatin|pravastatin|lovastatin|pitavastatin|statin",
    "med_betablocker":    r"metoprolol|atenolol|carvedilol|propranolol|bisoprolol|nadolol|labetalol|nebivolol",
    "med_acearb":         r"lisinopril|enalapril|ramipril|benazepril|captopril|quinapril|"
                          r"losartan|valsartan|irbesartan|olmesartan|telmisartan|candesartan",
    "med_nitrate":        r"nitroglycerin|nitrostat|isosorbide|imdur|nitrate|ranolazine",
    "med_anticoagulant":  r"warfarin|coumadin|heparin|enoxaparin|lovenox|apixaban|eliquis|"
                          r"rivaroxaban|xarelto|dabigatran|edoxaban",
    "med_ccb":            r"amlodipine|diltiazem|verapamil|nifedipine|felodipine",
    "med_diuretic":       r"furosemide|lasix|torsemide|bumetanide|hydrochlorothiazide|spironolactone",
    "med_antiarrhythmic": r"amiodarone|sotalol|flecainide|digoxin|dofetilide|mexiletine",
    "med_insulin_oha":    r"insulin|metformin|glipizide|glyburide|sitagliptin|empagliflozin|liraglutide",
    "med_ppi":            r"omeprazole|pantoprazole|esomeprazole|famotidine|ranitidine",
    "med_opioid":         r"oxycodone|hydrocodone|morphine|fentanyl|tramadol|hydromorphone",
}

# ECG report findings.  ST-elevation on a triage ECG is the diagnostic
# instrument for STEMI, not leakage: it is acquired within 10 minutes of
# arrival as a standard-of-care quality metric.  Its influence is quantified
# in the ablation study.
ECG_FINDINGS: Dict[str, str] = {
    # --- acuity ------------------------------------------------------------
    # The single most informative token in a MIMIC ECG report for separating
    # STEMI from NSTEMI is the word "acute".  The cart distinguishes an ACUTE
    # injury pattern from an infarct of "age undetermined" (i.e. old), and that
    # distinction is precisely the STEMI/NSTEMI question.  Measured alone it
    # reaches AUROC 0.723 on STEMI-vs-NSTEMI — higher than any ST-segment
    # regex — and fires for 50.3% of STEMI versus 5.8% of NSTEMI.
    "ecg_acute":           r"\bacute\b",
    "ecg_age_undetermined": r"age undetermined",           # marks an OLD infarct
    # "***" is the cart's own critical-finding alert; it is on the printout the
    # clinician holds at the bedside, so it is available at triage.
    "ecg_critical_alert":  r"\*\*\*",
    "ecg_stemi_alert":     r"consider acute st elevation|acute st elevation mi|\bstemi\b",
    "ecg_acute_mi":        r"acute mi\b|acute infarct|acute injury|injury pattern",

    # --- ST segment / repolarisation ---------------------------------------
    "ecg_st_elevation":    r"st elevation|elevated st|st segment elevation|\bste\b",
    "ecg_st_depression":   r"st depression|depressed st|st segment depression",
    "ecg_st_t_abnormal":   r"st & t wave|st-t wave|nonspecific st|repolarization abnormalit",
    "ecg_t_inversion":     r"t wave inversion|inverted t|t-wave inversion|nonspecific t",

    # --- infarct territory (localisation matters clinically) ---------------
    "ecg_infarct_any":     r"infarct",
    "ecg_infarct_inferior": r"inferior infarct",
    "ecg_infarct_anterior": r"anterior infarct|anteroseptal infarct|septal infarct",
    "ecg_infarct_lateral": r"lateral infarct",
    "ecg_infarct_possible": r"possible \w* ?infarct",       # hedged = less likely acute
    "ecg_q_wave":          r"\bq wave|pathologic q",
    "ecg_poor_r":          r"poor r wave progression",

    # --- conduction / rhythm ------------------------------------------------
    "ecg_lbbb":            r"left bundle branch block|\blbbb\b",
    "ecg_rbbb":            r"right bundle branch block|\brbbb\b",
    "ecg_afib":            r"atrial fibrillation|atrial flutter|\bafib\b",
    "ecg_lvh":             r"left ventricular hypertrophy|\blvh\b",
    "ecg_paced":           r"paced|pacemaker",
    "ecg_sinus_tach":      r"sinus tachycardia",
    "ecg_sinus_brady":     r"sinus bradycardia",

    # --- quality ------------------------------------------------------------
    # \b matters: without it "normal ecg" matches inside "ABnormal ecg", which
    # inverts the feature.  It fired for 86% of STEMI before this was fixed.
    "ecg_normal":          r"\bnormal ecg\b|\bnormal electrocardiogram\b|within normal limits",
    "ecg_abnormal":        r"abnormal ecg|abnormal electrocardiogram",
    "ecg_poor_quality":    r"leads are missing|artifact|poor quality|unable to analyze",
}

ECG_NUMERIC_BOUNDS: Dict[str, Tuple[float, float]] = {
    "qrs_duration": (40.0, 300.0),
    "pr_interval":  (60.0, 500.0),
    "qt_interval":  (150.0, 700.0),
    "rr_interval":  (200.0, 3000.0),
    "p_axis":       (-180.0, 180.0),
    "qrs_axis":     (-180.0, 180.0),
    "t_axis":       (-180.0, 180.0),
}
_ECG_SENTINEL = 10000.0   # MIMIC machine error codes: 32767, -29999, 65535 ...


# ==========================================================================
# Loading
# ==========================================================================
def load_raw() -> Dict[str, pd.DataFrame]:
    raw = CFG.raw_dir
    section(f"Loading raw MIMIC-IV-ED tables from {raw}")
    need = {
        "master":   "master_data.parquet",
        "labs":     "lab_values.parquet",
        "meds":     "medrecon.parquet",
        "ecg_rec":  "ecg_records.parquet",
        "ecg_meas": "ecg_measurements.parquet",
        "ecg_num":  "ecg_numeric.parquet",
        "charlson": "charlson.parquet",
    }
    out: Dict[str, pd.DataFrame] = {}
    for key, fname in need.items():
        path = os.path.join(raw, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required table missing: {path}\n"
                f"Set paths.raw_dir in configs/config.yaml to the folder that "
                f"contains the extracted MIMIC-IV-ED parquet files."
            )
        out[key] = pd.read_parquet(path)
        kv(f"{key}", f"{len(out[key]):>10,} rows x {out[key].shape[1]:>2} cols")
    return out


# ==========================================================================
# M0 — cohort assembly & label
# ==========================================================================
def build_index(master: pd.DataFrame) -> pd.DataFrame:
    section("M0  Index encounter table")
    df = master.copy()
    for c in ("subject_id", "hadm_id", "stay_id", "anchor_age", "acs_label"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    df["intime"] = pd.to_datetime(df["intime"])
    df["outtime"] = pd.to_datetime(df["outtime"])
    df["ed_los_h"] = (df["outtime"] - df["intime"]).dt.total_seconds() / 3600.0
    cap = float(CFG.get("temporal.max_ed_los_h", 168))
    df.loc[(df["ed_los_h"] < 0) | (df["ed_los_h"] > cap), "ed_los_h"] = np.nan

    df = df.dropna(subset=["stay_id", "subject_id", "intime", "acs_label"])
    df["acs_label"] = df["acs_label"].astype(int)
    df = df.drop_duplicates(subset="stay_id", keep="first").reset_index(drop=True)

    kv("index encounters", f"{len(df):,}")
    kv("unique patients", f"{df.subject_id.nunique():,}")
    for k, name in LABEL_MAP.items():
        n = int((df.acs_label == k).sum())
        kv(f"  {name}", f"{n:>8,}  ({n/len(df)*100:6.3f}%)")
    return df


# ==========================================================================
# M1 — triage vitals
# ==========================================================================
def _parse_pain(raw: pd.Series) -> pd.DataFrame:
    """
    The `pain` column is free text: 0-10, but also '13', 'uta', 'unable',
    'critical', 'denies', 'u/a', 'c'.  Coercing straight to numeric silently
    discards ~4% of rows and mislabels 'unable to assess' as missing-at-random.
    """
    s = raw.fillna("").astype(str).str.strip().str.lower()
    num = pd.to_numeric(s.str.extract(r"^(\d+(?:\.\d+)?)", expand=False), errors="coerce")

    unable = s.str.contains(r"^(?:u\s*/?\s*t?\s*a|unable|uta|ua|n/?a|c|critical|"
                            r"non ?verbal|sleeping|sedated)$", regex=True)
    denies = s.str.contains(r"denies|none|no pain", regex=True)

    out = pd.DataFrame(index=raw.index)
    out["pain_score"] = num.clip(0, 10)
    out.loc[denies & num.isna(), "pain_score"] = 0.0
    out["pain_unassessable"] = unable.astype(np.int8)
    out["pain_reported"] = out["pain_score"].notna().astype(np.int8)
    out["pain_severe"] = (out["pain_score"] >= 7).fillna(False).astype(np.int8)
    # >10 entries are data-entry noise but flag them; they are real records
    out["pain_out_of_range"] = (num > 10).fillna(False).astype(np.int8)
    return out


def vitals_block(df: pd.DataFrame) -> pd.DataFrame:
    section("M1  Triage vitals")
    out = pd.DataFrame(index=df.index)
    n_oob_total = 0
    for c, (lo, hi) in VITAL_BOUNDS.items():
        v = pd.to_numeric(df[c], errors="coerce")
        oob = ((v < lo) | (v > hi)) & v.notna()
        n_oob_total += int(oob.sum())
        v = v.mask(oob)
        out[c] = v.astype(np.float32)
        out[f"{c}_missing"] = v.isna().astype(np.int8)
    kv("implausible values -> NaN", f"{n_oob_total:,}")

    acuity = pd.to_numeric(df["acuity"], errors="coerce").clip(1, 5)
    out["acuity"] = acuity.astype(np.float32)
    out["acuity_missing"] = acuity.isna().astype(np.int8)
    out["acuity_high"] = (acuity <= 2).fillna(False).astype(np.int8)   # ESI 1-2

    out = pd.concat([out, _parse_pain(df["pain"])], axis=1)

    # --- derived haemodynamics -------------------------------------------
    hr, sbp, dbp = out["heartrate"], out["sbp"], out["dbp"]
    out["shock_index"] = (hr / sbp.replace(0, np.nan)).astype(np.float32)
    out["pulse_pressure"] = (sbp - dbp).astype(np.float32)
    out["map"] = (dbp + (sbp - dbp) / 3.0).astype(np.float32)
    out["rate_pressure_product"] = (hr * sbp / 1000.0).astype(np.float32)
    out["modified_shock_index"] = (hr / out["map"].replace(0, np.nan)).astype(np.float32)
    out["pulse_pressure_narrow"] = (out["pulse_pressure"] < 30).fillna(False).astype(np.int8)

    # --- clinical red flags ----------------------------------------------
    out["vs_tachycardia"] = (hr > 100).fillna(False).astype(np.int8)
    out["vs_bradycardia"] = (hr < 50).fillna(False).astype(np.int8)
    out["vs_hypotension"] = (sbp < 90).fillna(False).astype(np.int8)
    out["vs_hypertension"] = (sbp > 180).fillna(False).astype(np.int8)
    out["vs_hypoxia"] = (out["o2sat"] < 94).fillna(False).astype(np.int8)
    out["vs_tachypnea"] = (out["resprate"] > 22).fillna(False).astype(np.int8)
    out["vs_febrile"] = (out["temperature"] > 100.4).fillna(False).astype(np.int8)
    out["vs_shock_index_high"] = (out["shock_index"] > 0.9).fillna(False).astype(np.int8)
    flags = [c for c in out.columns if c.startswith("vs_")]
    out["vs_redflag_count"] = out[flags].sum(axis=1).astype(np.int8)

    # qSOFA-style instability composite
    out["vs_instability"] = (
        out["vs_hypotension"] * 2 + out["vs_hypoxia"] * 2 +
        out["vs_tachycardia"] + out["vs_tachypnea"] + out["vs_shock_index_high"] * 2
    ).astype(np.int8)

    n_vit = [c for c in VITAL_BOUNDS] + ["acuity"]
    out["vitals_n_recorded"] = out[n_vit].notna().sum(axis=1).astype(np.int8)
    out["vitals_available"] = (out["vitals_n_recorded"] > 0).astype(np.int8)

    kv("features", out.shape[1])
    return out


# ==========================================================================
# M2 — demographics
# ==========================================================================
def demographics_block(df: pd.DataFrame) -> pd.DataFrame:
    section("M2  Demographics")
    out = pd.DataFrame(index=df.index)
    age = pd.to_numeric(df["anchor_age"], errors="coerce").clip(18, 95)
    out["age"] = age.astype(np.float32)
    out["age_missing"] = age.isna().astype(np.int8)
    out["age_over_65"] = (age >= 65).fillna(False).astype(np.int8)
    out["age_over_75"] = (age >= 75).fillna(False).astype(np.int8)
    out["sex_male"] = (df["gender"].astype(str).str.upper() == "M").astype(np.int8)

    race = df["race"].fillna("UNKNOWN").astype(str).str.upper()
    grp = pd.Series("other", index=df.index, dtype=object)
    for token, g in RACE_GROUPS:
        grp = grp.mask(race.str.contains(token, regex=False), g)
    grp = grp.mask(race.str.contains("UNKNOWN|UNABLE|DECLINE|PATIENT DECLINED", regex=True),
                   "unknown")
    for g in ("white", "black", "hispanic", "asian", "other", "unknown"):
        out[f"race_{g}"] = (grp == g).astype(np.int8)

    # Arrival timing — MIMIC shifts calendar dates but preserves time-of-day
    # and day-of-week, so both are legitimate features.
    hour = df["intime"].dt.hour
    out["arrival_hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    out["arrival_hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    out["arrival_overnight"] = hour.isin(range(0, 7)).astype(np.int8)
    out["arrival_weekend"] = (df["intime"].dt.dayofweek >= 5).astype(np.int8)

    # Cardiovascular risk composite (age + sex), a HEART-score analogue
    out["cv_risk_age_sex"] = (
        (age.fillna(age.median()) / 10.0) + out["sex_male"] * 1.0
    ).astype(np.float32)
    kv("features", out.shape[1])
    return out


# ==========================================================================
# M3 — chief complaint text
# ==========================================================================
def text_block(df: pd.DataFrame, rdm_enable: bool) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    section("M3  Chief complaint text")
    norm = TF.normalise(df["chiefcomplaint"])
    masked, rdm_flags = TF.apply_rdm(norm, enable=rdm_enable)
    lex = TF.lexicon_features(masked)

    kv("RDM enabled", rdm_enable)
    kv("records with referral dx", f"{int(rdm_flags.cc_referral_dx.sum()):,} "
                                   f"({rdm_flags.cc_referral_dx.mean()*100:.2f}%)")
    for k, name in LABEL_MAP.items():
        m = df.acs_label == k
        if m.any():
            kv(f"  referral dx | {name}",
               f"{rdm_flags.loc[m, 'cc_referral_dx'].mean()*100:6.2f}%")

    block = pd.concat([lex, rdm_flags], axis=1)
    block["text_available"] = (lex["cc_n_tokens"] > 0).astype(np.int8)
    kv("features (pre-SVD)", block.shape[1])
    return block, masked, norm


# ==========================================================================
# M4 — home medications (medrecon, captured at ED arrival)
# ==========================================================================
def medication_block(df: pd.DataFrame, meds: pd.DataFrame) -> pd.DataFrame:
    section("M4  Home medications (medrecon @ arrival)")
    m = meds[["stay_id", "med_name"]].copy()
    m["stay_id"] = pd.to_numeric(m["stay_id"], errors="coerce").astype("Int64")
    m["med_name"] = m["med_name"].fillna("").astype(str).str.lower()

    stays_with_rec = set(m["stay_id"].dropna().unique())
    flags = pd.DataFrame(index=df.index)
    sid = df["stay_id"]

    for cls, pat in MED_CLASSES.items():
        hit = m.loc[m["med_name"].str.contains(pat, regex=True, na=False), "stay_id"]
        flags[cls] = sid.isin(set(hit.unique())).astype(np.int8)

    n_meds = m.groupby("stay_id").size()
    flags["med_total_count"] = sid.map(n_meds).fillna(0).clip(0, 60).astype(np.float32)
    cardiac = ["med_antiplatelet", "med_statin", "med_betablocker", "med_acearb",
               "med_nitrate", "med_anticoagulant", "med_ccb", "med_antiarrhythmic"]
    flags["med_cardiac_count"] = flags[cardiac].sum(axis=1).astype(np.int8)
    # Secondary-prevention triad => established coronary disease
    flags["med_secondary_prevention"] = (
        (flags["med_antiplatelet"] + flags["med_statin"] + flags["med_betablocker"]) >= 2
    ).astype(np.int8)
    flags["meds_available"] = sid.isin(stays_with_rec).astype(np.int8)

    kv("stays with reconciliation", f"{int(flags.meds_available.sum()):,} "
                                    f"({flags.meds_available.mean()*100:.1f}%)")
    kv("features", flags.shape[1])
    return flags


# ==========================================================================
# M5 — prior-encounter history  (replaces the leaking Charlson join)
# ==========================================================================
def history_block(df: pd.DataFrame, charlson: pd.DataFrame) -> pd.DataFrame:
    """
    The original pipeline joined `charlson` on the CURRENT hadm_id.  Because
    Charlson is computed from that admission's own ICD codes, and the ACS label
    is computed from the same codes, `myocardial_infarct` equals 1.0 for 100%
    of NSTEMI and 100% of STEMI rows.  It is the label.

    Here, all history is reconstructed from encounters strictly earlier than
    T0 of the index visit, so nothing about the current admission can flow in.
    """
    section("M5  Prior-encounter history (strictly < T0)")
    d = df[["subject_id", "hadm_id", "stay_id", "intime", "acs_label"]].copy()
    d = d.sort_values(["subject_id", "intime"]).reset_index(drop=False) \
         .rename(columns={"index": "_orig"})
    g = d.groupby("subject_id", sort=False)

    out = pd.DataFrame(index=d.index)
    out["hist_n_prior_visits"] = g.cumcount().astype(np.float32)
    # shift(1) => value from the previous visit only
    prev_time = g["intime"].shift(1)
    out["hist_days_since_last"] = (
        (d["intime"] - prev_time).dt.total_seconds() / 86400.0
    ).clip(0, 3650).astype(np.float32)
    out["hist_has_prior"] = prev_time.notna().astype(np.int8)

    # prior ACS of any type / by type — cumulative over EARLIER rows only
    for k, name in LABEL_MAP.items():
        if k == 0:
            continue
        ind = (d["acs_label"] == k).astype(int)
        out[f"hist_prior_{name.lower()}"] = (
            g[ind.name].apply(lambda s: s.shift(1).fillna(0).cumsum())
            if False else ind.groupby(d["subject_id"]).transform(
                lambda s: s.shift(1, fill_value=0).cumsum())
        ).astype(np.float32)
    acs_ind = (d["acs_label"] > 0).astype(int)
    out["hist_prior_acs_any"] = acs_ind.groupby(d["subject_id"]).transform(
        lambda s: s.shift(1, fill_value=0).cumsum()).astype(np.float32)

    # 30/365-day revisit intensity
    out["hist_revisit_30d"] = (out["hist_days_since_last"] <= 30).fillna(False).astype(np.int8)
    out["hist_revisit_365d"] = (out["hist_days_since_last"] <= 365).fillna(False).astype(np.int8)
    out["hist_frequent_user"] = (out["hist_n_prior_visits"] >= 5).astype(np.int8)

    # ---- Charlson restricted to PRIOR admissions of the same patient ----
    ch = charlson.copy()
    ch["hadm_id"] = pd.to_numeric(ch["hadm_id"], errors="coerce").astype("Int64")
    ch = ch.drop_duplicates("hadm_id")
    # map hadm_id -> its ED arrival time (only ED-originating admissions known)
    hadm_time = d.dropna(subset=["hadm_id"]).groupby("hadm_id")["intime"].min()
    ch["adm_time"] = ch["hadm_id"].map(hadm_time)
    ch = ch.dropna(subset=["adm_time"])
    ch["subject_id"] = pd.to_numeric(ch["subject_id"], errors="coerce").astype("Int64")

    cols = ["charlson_comorbidity_index", "myocardial_infarct",
            "congestive_heart_failure", "diabetes_without_cc",
            "diabetes_with_cc", "renal_disease"]
    for c in cols:
        ch[c] = pd.to_numeric(ch[c], errors="coerce").fillna(0)

    ch = ch.sort_values(["subject_id", "adm_time"])
    # merge_asof: for each index visit take the most recent PRIOR admission
    left = d[["subject_id", "intime"]].copy()
    left["_row"] = np.arange(len(left))
    left = left.sort_values("intime")
    right = ch[["subject_id", "adm_time"] + cols].sort_values("adm_time")
    merged = pd.merge_asof(
        left, right, left_on="intime", right_on="adm_time", by="subject_id",
        allow_exact_matches=False, direction="backward",
    ).sort_values("_row")

    out["hist_charlson_index"] = merged["charlson_comorbidity_index"].fillna(0).to_numpy(np.float32)
    out["hist_prior_mi_icd"] = merged["myocardial_infarct"].fillna(0).to_numpy(np.float32)
    out["hist_prior_chf_icd"] = merged["congestive_heart_failure"].fillna(0).to_numpy(np.float32)
    out["hist_diabetes"] = (
        (merged["diabetes_without_cc"].fillna(0) + merged["diabetes_with_cc"].fillna(0)) > 0
    ).to_numpy(np.int8)
    out["hist_renal_disease"] = merged["renal_disease"].fillna(0).to_numpy(np.float32)
    out["hist_charlson_available"] = merged["adm_time"].notna().to_numpy(np.int8)

    kv("with >=1 prior ED visit", f"{int(out.hist_has_prior.sum()):,} "
                                  f"({out.hist_has_prior.mean()*100:.1f}%)")
    kv("with prior-admission Charlson", f"{int(out.hist_charlson_available.sum()):,} "
                                        f"({out.hist_charlson_available.mean()*100:.1f}%)")
    kv("leak check P(prior_mi_icd=1 | STEMI)",
       f"{out.loc[(d.acs_label==3).to_numpy(), 'hist_prior_mi_icd'].mean():.4f}"
       "   (was 1.0000 with same-admission join)")

    # restore original row order
    out["_orig"] = d["_orig"].to_numpy()
    out = out.sort_values("_orig").drop(columns="_orig").reset_index(drop=True)
    out.index = df.index
    kv("features", out.shape[1])
    return out


# ==========================================================================
# M6 — ECG within the disclosure horizon
# ==========================================================================
def _prepare_ecg(ecg_rec, ecg_meas, ecg_num) -> pd.DataFrame:
    """One row per study_id: time, numeric intervals, parsed report findings."""
    rec = ecg_rec[["subject_id", "study_id", "ecg_time"]].copy()
    rec["ecg_time"] = pd.to_datetime(rec["ecg_time"])
    for c in ("subject_id", "study_id"):
        rec[c] = pd.to_numeric(rec[c], errors="coerce").astype("Int64")

    num = ecg_num.copy()
    num["study_id"] = pd.to_numeric(num["study_id"], errors="coerce").astype("Int64")
    for c, (lo, hi) in ECG_NUMERIC_BOUNDS.items():
        if c in num.columns:
            v = pd.to_numeric(num[c], errors="coerce").astype(float)
            v = v.mask(v.abs() > _ECG_SENTINEL)
            v = v.mask((v < lo) | (v > hi))
            num[c] = v
    keep_num = ["study_id"] + [c for c in ECG_NUMERIC_BOUNDS if c in num.columns]
    num = num[keep_num].drop_duplicates("study_id")

    meas = ecg_meas.copy()
    meas["study_id"] = pd.to_numeric(meas["study_id"], errors="coerce").astype("Int64")
    rep_cols = [c for c in meas.columns if c.startswith("report_")]
    txt = meas[rep_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    parsed = pd.DataFrame({"study_id": meas["study_id"]})
    for name, pat in ECG_FINDINGS.items():
        parsed[name] = txt.str.contains(pat, regex=True, na=False).astype(np.int8)
    parsed["ecg_n_findings"] = parsed[list(ECG_FINDINGS)].sum(axis=1).astype(np.int8)
    parsed["ecg_report_len"] = txt.str.len().clip(0, 500).astype(np.int16)
    parsed = parsed.drop_duplicates("study_id")

    ecg = rec.merge(num, on="study_id", how="left").merge(parsed, on="study_id", how="left")
    return ecg.dropna(subset=["subject_id", "ecg_time"])


def ecg_block(df: pd.DataFrame, ecg: pd.DataFrame, horizon_h: float,
              lookback_h: float) -> pd.DataFrame:
    """Nearest ECG to T0 within [T0 - lookback, T0 + horizon]."""
    pairs = df[["subject_id", "stay_id", "intime"]].merge(
        ecg, on="subject_id", how="inner")
    pairs["dt_h"] = (pairs["ecg_time"] - pairs["intime"]).dt.total_seconds() / 3600.0
    pairs = pairs[(pairs["dt_h"] >= -lookback_h) & (pairs["dt_h"] <= horizon_h)]

    finding_cols = list(ECG_FINDINGS) + ["ecg_n_findings", "ecg_report_len"]
    numeric_cols = [c for c in ECG_NUMERIC_BOUNDS if c in pairs.columns]

    out = pd.DataFrame(index=df.index)
    if pairs.empty:
        for c in finding_cols:
            out[c] = np.int8(0)
        for c in numeric_cols:
            out[f"ecg_{c}"] = np.float32(np.nan)
        out["ecg_available"] = np.int8(0)
        out["ecg_n_studies"] = np.float32(0)
        out["ecg_dt_first_h"] = np.float32(np.nan)
        return out

    # aggregate ACROSS the window: max for findings (any abnormality seen),
    # value from the EARLIEST study for intervals (closest to triage).
    pairs = pairs.sort_values(["stay_id", "dt_h"])
    first = pairs.groupby("stay_id", sort=False).first()
    agg_max = pairs.groupby("stay_id", sort=False)[list(ECG_FINDINGS)].max()
    n_studies = pairs.groupby("stay_id", sort=False).size().rename("ecg_n_studies")
    dt_first = pairs.groupby("stay_id", sort=False)["dt_h"].min().rename("ecg_dt_first_h")

    sid = df["stay_id"]

    for cls, pat in MED_CLASSES.items():
        hit = m.loc[m["med_name"].str.contains(pat, regex=True, na=False), "stay_id"]
        flags[cls] = sid.isin(set(hit.unique())).astype(np.int8)

    n_meds = m.groupby("stay_id").size()
    flags["med_total_count"] = sid.map(n_meds).fillna(0).clip(0, 60).astype(np.float32)
    cardiac = ["med_antiplatelet", "med_statin", "med_betablocker", "med_acearb",
               "med_nitrate", "med_anticoagulant", "med_ccb", "med_antiarrhythmic"]
    flags["med_cardiac_count"] = flags[cardiac].sum(axis=1).astype(np.int8)
    # Secondary-prevention triad => established coronary disease
    flags["med_secondary_prevention"] = (
        (flags["med_antiplatelet"] + flags["med_statin"] + flags["med_betablocker"]) >= 2
    ).astype(np.int8)
    flags["meds_available"] = sid.isin(stays_with_rec).astype(np.int8)

    kv("stays with reconciliation", f"{int(flags.meds_available.sum()):,} "
                                    f"({flags.meds_available.mean()*100:.1f}%)")
    kv("features", flags.shape[1])
    return flags


# ==========================================================================
# M5 — prior-encounter history  (replaces the leaking Charlson join)
# ==========================================================================
def history_block(df: pd.DataFrame, charlson: pd.DataFrame) -> pd.DataFrame:
    """
    The original pipeline joined `charlson` on the CURRENT hadm_id.  Because
    Charlson is computed from that admission's own ICD codes, and the ACS label
    is computed from the same codes, `myocardial_infarct` equals 1.0 for 100%
    of NSTEMI and 100% of STEMI rows.  It is the label.

    Here, all history is reconstructed from encounters strictly earlier than
    T0 of the index visit, so nothing about the current admission can flow in.
    """
    section("M5  Prior-encounter history (strictly < T0)")
    d = df[["subject_id", "hadm_id", "stay_id", "intime", "acs_label"]].copy()
    d = d.sort_values(["subject_id", "intime"]).reset_index(drop=False) \
         .rename(columns={"index": "_orig"})
    g = d.groupby("subject_id", sort=False)

    out = pd.DataFrame(index=d.index)
    out["hist_n_prior_visits"] = g.cumcount().astype(np.float32)
    # shift(1) => value from the previous visit only
    prev_time = g["intime"].shift(1)
    out["hist_days_since_last"] = (
        (d["intime"] - prev_time).dt.total_seconds() / 86400.0
    ).clip(0, 3650).astype(np.float32)
    out["hist_has_prior"] = prev_time.notna().astype(np.int8)

    # prior ACS of any type / by type — cumulative over EARLIER rows only
    for k, name in LABEL_MAP.items():
        if k == 0:
            continue
        ind = (d["acs_label"] == k).astype(int)
        out[f"hist_prior_{name.lower()}"] = (
            g[ind.name].apply(lambda s: s.shift(1).fillna(0).cumsum())
            if False else ind.groupby(d["subject_id"]).transform(
                lambda s: s.shift(1, fill_value=0).cumsum())
        ).astype(np.float32)
    acs_ind = (d["acs_label"] > 0).astype(int)
    out["hist_prior_acs_any"] = acs_ind.groupby(d["subject_id"]).transform(
        lambda s: s.shift(1, fill_value=0).cumsum()).astype(np.float32)

    # 30/365-day revisit intensity
    out["hist_revisit_30d"] = (out["hist_days_since_last"] <= 30).fillna(False).astype(np.int8)
    out["hist_revisit_365d"] = (out["hist_days_since_last"] <= 365).fillna(False).astype(np.int8)
    out["hist_frequent_user"] = (out["hist_n_prior_visits"] >= 5).astype(np.int8)

    # ---- Charlson restricted to PRIOR admissions of the same patient ----
    ch = charlson.copy()
    ch["hadm_id"] = pd.to_numeric(ch["hadm_id"], errors="coerce").astype("Int64")
    ch = ch.drop_duplicates("hadm_id")
    # map hadm_id -> its ED arrival time (only ED-originating admissions known)
    hadm_time = d.dropna(subset=["hadm_id"]).groupby("hadm_id")["intime"].min()
    ch["adm_time"] = ch["hadm_id"].map(hadm_time)
    ch = ch.dropna(subset=["adm_time"])
    ch["subject_id"] = pd.to_numeric(ch["subject_id"], errors="coerce").astype("Int64")

    cols = ["charlson_comorbidity_index", "myocardial_infarct",
            "congestive_heart_failure", "diabetes_without_cc",
            "diabetes_with_cc", "renal_disease"]
    for c in cols:
        ch[c] = pd.to_numeric(ch[c], errors="coerce").fillna(0)

    ch = ch.sort_values(["subject_id", "adm_time"])
    # merge_asof: for each index visit take the most recent PRIOR admission
    left = d[["subject_id", "intime"]].copy()
    left["_row"] = np.arange(len(left))
    left = left.sort_values("intime")
    right = ch[["subject_id", "adm_time"] + cols].sort_values("adm_time")
    merged = pd.merge_asof(
        left, right, left_on="intime", right_on="adm_time", by="subject_id",
        allow_exact_matches=False, direction="backward",
    ).sort_values("_row")

    out["hist_charlson_index"] = merged["charlson_comorbidity_index"].fillna(0).to_numpy(np.float32)
    out["hist_prior_mi_icd"] = merged["myocardial_infarct"].fillna(0).to_numpy(np.float32)
    out["hist_prior_chf_icd"] = merged["congestive_heart_failure"].fillna(0).to_numpy(np.float32)
    out["hist_diabetes"] = (
        (merged["diabetes_without_cc"].fillna(0) + merged["diabetes_with_cc"].fillna(0)) > 0
    ).to_numpy(np.int8)
    out["hist_renal_disease"] = merged["renal_disease"].fillna(0).to_numpy(np.float32)
    out["hist_charlson_available"] = merged["adm_time"].notna().to_numpy(np.int8)

    kv("with >=1 prior ED visit", f"{int(out.hist_has_prior.sum()):,} "
                                  f"({out.hist_has_prior.mean()*100:.1f}%)")
    kv("with prior-admission Charlson", f"{int(out.hist_charlson_available.sum()):,} "
                                        f"({out.hist_charlson_available.mean()*100:.1f}%)")
    kv("leak check P(prior_mi_icd=1 | STEMI)",
       f"{out.loc[(d.acs_label==3).to_numpy(), 'hist_prior_mi_icd'].mean():.4f}"
       "   (was 1.0000 with same-admission join)")

    # restore original row order
    out["_orig"] = d["_orig"].to_numpy()
    out = out.sort_values("_orig").drop(columns="_orig").reset_index(drop=True)
    out.index = df.index
    kv("features", out.shape[1])
    return out


# ==========================================================================
# M6 — ECG within the disclosure horizon
# ==========================================================================
def _prepare_ecg(ecg_rec, ecg_meas, ecg_num) -> pd.DataFrame:
    """One row per study_id: time, numeric intervals, parsed report findings."""
    rec = ecg_rec[["subject_id", "study_id", "ecg_time"]].copy()
    rec["ecg_time"] = pd.to_datetime(rec["ecg_time"])
    for c in ("subject_id", "study_id"):
        rec[c] = pd.to_numeric(rec[c], errors="coerce").astype("Int64")

    num = ecg_num.copy()
    num["study_id"] = pd.to_numeric(num["study_id"], errors="coerce").astype("Int64")
    for c, (lo, hi) in ECG_NUMERIC_BOUNDS.items():
        if c in num.columns:
            v = pd.to_numeric(num[c], errors="coerce").astype(float)
            v = v.mask(v.abs() > _ECG_SENTINEL)
            v = v.mask((v < lo) | (v > hi))
            num[c] = v
    keep_num = ["study_id"] + [c for c in ECG_NUMERIC_BOUNDS if c in num.columns]
    num = num[keep_num].drop_duplicates("study_id")

    meas = ecg_meas.copy()
    meas["study_id"] = pd.to_numeric(meas["study_id"], errors="coerce").astype("Int64")
    rep_cols = [c for c in meas.columns if c.startswith("report_")]
    txt = meas[rep_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    parsed = pd.DataFrame({"study_id": meas["study_id"]})
    for name, pat in ECG_FINDINGS.items():
        parsed[name] = txt.str.contains(pat, regex=True, na=False).astype(np.int8)
    parsed["ecg_n_findings"] = parsed[list(ECG_FINDINGS)].sum(axis=1).astype(np.int8)
    parsed["ecg_report_len"] = txt.str.len().clip(0, 500).astype(np.int16)
    parsed = parsed.drop_duplicates("study_id")

    ecg = rec.merge(num, on="study_id", how="left").merge(parsed, on="study_id", how="left")
    return ecg.dropna(subset=["subject_id", "ecg_time"])


def ecg_block(df: pd.DataFrame, ecg: pd.DataFrame, horizon_h: float,
              lookback_h: float) -> pd.DataFrame:
    """Nearest ECG to T0 within [T0 - lookback, T0 + horizon]."""
    pairs = df[["subject_id", "stay_id", "intime"]].merge(
        ecg, on="subject_id", how="inner")
    pairs["dt_h"] = (pairs["ecg_time"] - pairs["intime"]).dt.total_seconds() / 3600.0
    pairs = pairs[(pairs["dt_h"] >= -lookback_h) & (pairs["dt_h"] <= horizon_h)]

    finding_cols = list(ECG_FINDINGS) + ["ecg_n_findings", "ecg_report_len"]
    numeric_cols = [c for c in ECG_NUMERIC_BOUNDS if c in pairs.columns]

    out = pd.DataFrame(index=df.index)
    if pairs.empty:
        for c in finding_cols:
            out[c] = np.int8(0)
        for c in numeric_cols:
            out[f"ecg_{c}"] = np.float32(np.nan)
        out["ecg_available"] = np.int8(0)
        out["ecg_n_studies"] = np.float32(0)
        out["ecg_dt_first_h"] = np.float32(np.nan)
        return out

    # --- derived repolarisation measures, per study ------------------------
    # A STEMI is diagnosed from how the ECG EVOLVES, not from one snapshot, so
    # we compute these per study and carry both the earliest value and the
    # change across the window.
    if {"qt_interval", "qrs_duration"} <= set(pairs.columns):
        pairs["_jt"] = pairs["qt_interval"] - pairs["qrs_duration"]
    if {"t_end", "qrs_end"} <= set(pairs.columns):
        pairs["_st_dur"] = pairs["t_end"] - pairs["qrs_end"]
    if "rr_interval" in pairs.columns:
        rr_s = (pairs["rr_interval"] / 1000.0).clip(lower=0.1)
        if "_jt" in pairs:
            pairs["_jtc"] = pairs["_jt"] / np.sqrt(rr_s)
        pairs["_qtc"] = pairs["qt_interval"] / np.sqrt(rr_s)
    if {"qrs_axis", "t_axis"} <= set(pairs.columns):
        # Spatial QRS-T angle: a recognised marker of repolarisation
        # abnormality, and not stated anywhere in the machine report.
        pairs["_qrst_angle"] = (
            (pairs["qrs_axis"] - pairs["t_axis"]).abs() % 360
        ).clip(upper=180)

    # aggregate ACROSS the window: max for findings (any abnormality seen),
    # value from the EARLIEST study for intervals (closest to triage).
    pairs = pairs.sort_values(["stay_id", "dt_h"])
    grp = pairs.groupby("stay_id", sort=False)
    first = grp.first()
    last = grp.last()
    agg_max = grp[list(ECG_FINDINGS)].max()
    n_studies = grp.size().rename("ecg_n_studies")
    dt_first = grp["dt_h"].min().rename("ecg_dt_first_h")
    dt_last = grp["dt_h"].max().rename("ecg_dt_last_h")

    sid = df["stay_id"]
    for c in ECG_FINDINGS:
        out[c] = sid.map(agg_max[c]).fillna(0).astype(np.int8)
    out["ecg_n_findings"] = sid.map(first["ecg_n_findings"]).fillna(0).astype(np.int8)
    out["ecg_report_len"] = sid.map(first["ecg_report_len"]).fillna(0).astype(np.int16)
    for c in numeric_cols:
        out[f"ecg_{c}"] = sid.map(first[c]).astype(np.float32)

    # QTc (Bazett) and derived conduction flags
    qt, rr = out.get("ecg_qt_interval"), out.get("ecg_rr_interval")
    if qt is not None and rr is not None:
        out["ecg_qtc"] = (qt / np.sqrt((rr / 1000.0).clip(lower=0.1))).astype(np.float32)
        out["ecg_qtc_prolonged"] = (out["ecg_qtc"] > 460).fillna(False).astype(np.int8)
        out["ecg_hr_from_rr"] = (60000.0 / rr.replace(0, np.nan)).astype(np.float32)
    if "ecg_qrs_duration" in out:
        out["ecg_wide_qrs"] = (out["ecg_qrs_duration"] > 120).fillna(False).astype(np.int8)
    if "ecg_qrs_axis" in out:
        out["ecg_left_axis"] = (out["ecg_qrs_axis"] < -30).fillna(False).astype(np.int8)
        out["ecg_right_axis"] = (out["ecg_qrs_axis"] > 100).fillna(False).astype(np.int8)
    if "ecg_pr_interval" in out:
        out["ecg_first_degree_block"] = (out["ecg_pr_interval"] > 200).fillna(False).astype(np.int8)

    out["ecg_n_studies"] = sid.map(n_studies).fillna(0).astype(np.float32)
    out["ecg_dt_first_h"] = sid.map(dt_first).astype(np.float32)
    out["ecg_available"] = (out["ecg_n_studies"] > 0).astype(np.int8)
    out["ecg_immediate"] = (out["ecg_dt_first_h"] <= 0.5).fillna(False).astype(np.int8)

    # STEMI-equivalent composite (ST elevation OR new LBBB) — a textbook rule,
    # included as an explicit interpretable feature rather than a post-hoc boost
    out["ecg_stemi_equivalent"] = (
        (out["ecg_st_elevation"] == 1) | (out["ecg_lbbb"] == 1)
    ).astype(np.int8)
    out["ecg_ischemic_any"] = (
        out[["ecg_st_elevation", "ecg_st_depression", "ecg_t_inversion",
             "ecg_q_wave", "ecg_infarct_any"]].sum(axis=1) > 0
    ).astype(np.int8)

    # --- acuity composites -------------------------------------------------
    # An infarct pattern that is ACUTE points to STEMI; the same pattern marked
    # "age undetermined" points to an old infarct on the ECG of an NSTEMI (or
    # of a non-cardiac presentation entirely).  Encoding the contrast directly
    # is worth more than leaving the trees to rediscover it from ~660 STEMIs.
    out["ecg_acute_ischemia"] = (
        (out["ecg_acute"] == 1) & (out["ecg_infarct_any"] == 1)
    ).astype(np.int8)
    out["ecg_old_infarct_only"] = (
        (out["ecg_age_undetermined"] == 1) & (out["ecg_acute"] == 0)
    ).astype(np.int8)
    out["ecg_acuity_score"] = (
        out["ecg_acute"] * 3 + out["ecg_critical_alert"] * 2 +
        out["ecg_stemi_alert"] * 3 + out["ecg_acute_mi"] * 2 +
        out["ecg_st_elevation"] * 2 - out["ecg_age_undetermined"] -
        out["ecg_infarct_possible"]
    ).astype(np.float32)
    out["ecg_territory_count"] = (
        out["ecg_infarct_inferior"] + out["ecg_infarct_anterior"] +
        out["ecg_infarct_lateral"]
    ).astype(np.int8)
    return out


# ==========================================================================
# M7 — cardiac biomarkers within the disclosure horizon
# ==========================================================================
def lab_block(df: pd.DataFrame, labs: pd.DataFrame, horizon_h: float) -> pd.DataFrame:
    section(f"M7  Cardiac biomarkers  window = [T0, T0+{horizon_h:g}h]")
    lb = labs.copy()
    lb["stay_id"] = pd.to_numeric(lb["stay_id"], errors="coerce").astype("Int64")
    lb["valuenum"] = pd.to_numeric(lb["valuenum"], errors="coerce")
    lb["charttime"] = pd.to_datetime(lb["charttime"])
    lb = lb[~lb["lab_name"].str.contains("Pleural", na=False)]        # not serum
    lb = lb.dropna(subset=["valuenum", "stay_id"])

    lb = lb.merge(df[["stay_id", "intime"]], on="stay_id", how="inner")
    lb["h"] = (lb["charttime"] - lb["intime"]).dt.total_seconds() / 3600.0
    total_before = len(lb)
    lb = lb[(lb["h"] >= 0) & (lb["h"] <= horizon_h)]
    kv("lab results retained", f"{len(lb):,} / {total_before:,} "
                               f"({len(lb)/max(total_before,1)*100:.1f}%)")

    out = pd.DataFrame(index=df.index)
    sid = df["stay_id"]

    # ---------------- troponin ----------------
    trop = lb[lb["lab_name"].str.contains("Troponin", na=False)].sort_values(["stay_id", "h"])
    if len(trop):
        gp = trop.groupby("stay_id", sort=False)
        first = gp["valuenum"].first()
        second = gp["valuenum"].nth(1) if len(trop) else pd.Series(dtype=float)
        if isinstance(second, pd.DataFrame):
            second = second.set_index("stay_id")["valuenum"]
        vmax = gp["valuenum"].max()
        t_first = gp["h"].first()
        t_last = gp["h"].last()
        n_draw = gp.size()

        out["trop_first"] = sid.map(first).astype(np.float32)
        out["trop_second"] = sid.map(second).astype(np.float32)
        out["trop_max"] = sid.map(vmax).astype(np.float32)
        out["trop_n_draws"] = sid.map(n_draw).fillna(0).astype(np.float32)
        out["trop_t_first_h"] = sid.map(t_first).astype(np.float32)
        out["trop_span_h"] = (sid.map(t_last) - sid.map(t_first)).astype(np.float32)
    else:
        for c in ("trop_first", "trop_second", "trop_max", "trop_t_first_h", "trop_span_h"):
            out[c] = np.float32(np.nan)
        out["trop_n_draws"] = np.float32(0)

    out["trop_available"] = out["trop_first"].notna().astype(np.int8)
    out["trop_serial"] = (out["trop_n_draws"] >= 2).astype(np.int8)
    out["trop_delta"] = (out["trop_second"] - out["trop_first"]).astype(np.float32)
    out["trop_delta_pct"] = (
        out["trop_delta"] / out["trop_first"].replace(0, np.nan)
    ).clip(-10, 50).astype(np.float32)
    out["trop_delta_rate"] = (
        out["trop_delta"] / out["trop_span_h"].replace(0, np.nan)
    ).clip(-50, 50).astype(np.float32)
    # log transform — troponin spans 4 orders of magnitude
    out["trop_log_first"] = np.log1p(out["trop_first"].clip(lower=0)).astype(np.float32)
    out["trop_log_max"] = np.log1p(out["trop_max"].clip(lower=0)).astype(np.float32)
    # Guideline decision points (99th percentile URL for Troponin T = 0.01-0.04)
    for thr, tag in ((0.04, "url"), (0.1, "mod"), (0.5, "high"), (1.0, "vhigh")):
        out[f"trop_gt_{tag}"] = (out["trop_max"] > thr).fillna(False).astype(np.int8)
    out["trop_rising"] = (out["trop_delta"] > 0.01).fillna(False).astype(np.int8)

    # ---------------- natriuretic peptide ----------------
    bnp = lb[lb["lab_name"].str.contains("BNP|natriuretic|proBNP", case=False,
                                         na=False, regex=True)].sort_values(["stay_id", "h"])
    if len(bnp):
        gb = bnp.groupby("stay_id", sort=False)
        out["bnp_first"] = sid.map(gb["valuenum"].first()).astype(np.float32)
        out["bnp_max"] = sid.map(gb["valuenum"].max()).astype(np.float32)
        out["bnp_t_first_h"] = sid.map(gb["h"].first()).astype(np.float32)
    else:
        for c in ("bnp_first", "bnp_max", "bnp_t_first_h"):
            out[c] = np.float32(np.nan)
    out["bnp_available"] = out["bnp_first"].notna().astype(np.int8)
    out["bnp_log_max"] = np.log1p(out["bnp_max"].clip(lower=0)).astype(np.float32)
    out["bnp_gt_400"] = (out["bnp_max"] > 400).fillna(False).astype(np.int8)

    # ---------------- workup-intensity signal ----------------
    # "A biomarker was ordered" is a clinician expressing suspicion at triage —
    # legitimate, and one of the strongest available signals.
    out["labs_any_available"] = ((out["trop_available"] + out["bnp_available"]) > 0).astype(np.int8)
    out["labs_workup_intensity"] = (
        out["trop_n_draws"].fillna(0) + out["bnp_available"] + out["trop_serial"] * 2
    ).astype(np.float32)

    for k, name in LABEL_MAP.items():
        m = (df.acs_label == k).to_numpy()
        kv(f"  troponin coverage | {name}", f"{out.loc[m, 'trop_available'].mean()*100:6.2f}%")
    kv("features", out.shape[1])
    return out


# ==========================================================================
# Cross-modal interaction features
# ==========================================================================
def interaction_block(v: pd.DataFrame, d: pd.DataFrame, t: pd.DataFrame,
                      e: pd.DataFrame, l: pd.DataFrame) -> pd.DataFrame:
    """
    Explicit clinical interactions.  Trees can approximate these, but with
    ~5k positives out of 200k the sample budget is far better spent giving the
    model the interactions a cardiologist already knows.
    """
    out = pd.DataFrame(index=v.index)
    age = d["age"].fillna(d["age"].median())
    out["ix_age_x_chestpain"] = (age * t["cc_chest_pain"]).astype(np.float32)
    out["ix_male_x_chestpain"] = (d["sex_male"] * t["cc_chest_pain"]).astype(np.float32)
    out["ix_shock_x_chestpain"] = (v["shock_index"].fillna(0) * t["cc_chest_pain"]).astype(np.float32)

    zero = pd.Series(0, index=v.index)
    ste = e["ecg_st_elevation"] if "ecg_st_elevation" in e else zero
    stdep = e["ecg_st_depression"] if "ecg_st_depression" in e else zero
    acute = e["ecg_acute"] if "ecg_acute" in e else zero
    alert = e["ecg_critical_alert"] if "ecg_critical_alert" in e else zero
    out["ix_ste_x_chestpain"] = (ste * t["cc_chest_pain"]).astype(np.int8)
    out["ix_ste_x_trop_high"] = (ste * l["trop_gt_high"]).astype(np.int8)
    out["ix_stdep_x_trop_url"] = (stdep * l["trop_gt_url"]).astype(np.int8)

    # The STEMI/NSTEMI decision in practice: is the ECG ACUTE, and how high is
    # the troponin?  Acute ECG + very high troponin => STEMI; raised troponin
    # with a non-acute ECG => NSTEMI.  Both interactions are given explicitly.
    out["ix_acute_x_trop_vhigh"] = (acute * l["trop_gt_vhigh"]).astype(np.int8)
    out["ix_acute_x_trop_log"] = (acute * l["trop_log_max"].fillna(0)).astype(np.float32)
    out["ix_alert_x_trop_log"] = (alert * l["trop_log_max"].fillna(0)).astype(np.float32)
    out["ix_nonacute_x_trop_pos"] = (
        (1 - acute) * l["trop_gt_url"]).astype(np.int8)          # NSTEMI signature
    out["ix_acute_x_ste"] = (acute * ste).astype(np.int8)

    # HEART-score analogue (History, ECG, Age, Risk factors, Troponin), 0-10
    hist_pt = (t["cc_chest_pain"] * 2 + t["cc_radiation"] + t["cc_diaphoresis"]).clip(0, 2)
    ecg_pt = (ste * 2 + stdep * 2 + acute * 2 +
              e.get("ecg_t_inversion", zero)).clip(0, 2)
    age_pt = (age >= 65).astype(int) * 2 + ((age >= 45) & (age < 65)).astype(int)
    trop_pt = (l["trop_gt_high"] * 2 + l["trop_gt_url"]).clip(0, 2)
    out["ix_heart_score"] = (hist_pt + ecg_pt + age_pt + trop_pt).astype(np.float32)
    out["ix_heart_high"] = (out["ix_heart_score"] >= 7).astype(np.int8)

    # Modality-availability fingerprint: which channels this patient has
    out["ix_modalities_present"] = (
        v["vitals_available"] + t["text_available"] +
        e.get("ecg_available", pd.Series(0, index=v.index)) +
        l["trop_available"] + l["bnp_available"]
    ).astype(np.int8)
    return out


# ==========================================================================
# Orchestration
# ==========================================================================
MODALITY_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "vitals":        ("heartrate", "sbp", "dbp", "resprate", "o2sat", "temperature",
                      "acuity", "pain", "shock", "pulse_pressure", "map",
                      "rate_pressure", "modified_shock", "vs_", "vitals_"),
    "demographics":  ("age", "sex_", "race_", "arrival_", "cv_risk"),
    "text":          ("cc_", "text_"),
    "medications":   ("med_", "meds_"),
    "history":       ("hist_",),
    "ecg":           ("ecg_",),
    "labs":          ("trop_", "bnp_", "labs_"),
    "interaction":   ("ix_",),
}


def assign_modality(col: str) -> str:
    for mod, prefixes in MODALITY_PREFIXES.items():
        if any(col.startswith(p) for p in prefixes):
            return mod
    return "other"


def build_features(horizon_h: int, raw: Dict[str, pd.DataFrame] | None = None,
                   rdm_enable: bool | None = None) -> Tuple[pd.DataFrame, Dict]:
    """Full feature matrix at one disclosure horizon."""
    banner(f"PREPROCESSING  |  disclosure horizon H = {horizon_h}h")
    raw = raw or load_raw()
    rdm_enable = CFG.get("text.rdm_enable", True) if rdm_enable is None else rdm_enable

    idx = build_index(raw["master"])
    lookback = float(CFG.get("temporal.ecg_lookback_h", 1.0))

    with timer("M1 vitals"):
        v = vitals_block(idx)
    with timer("M2 demographics"):
        d = demographics_block(idx)
    with timer("M3 text"):
        t, masked_text, norm_text = text_block(idx, rdm_enable)
    with timer("M4 medications"):
        med = medication_block(idx, raw["meds"])
    with timer("M5 history"):
        h = history_block(idx, raw["charlson"])
    with timer("M6 ECG"):
        section(f"M6  ECG  window = [T0-{lookback:g}h, T0+{horizon_h:g}h]")
        ecg_tbl = _prepare_ecg(raw["ecg_rec"], raw["ecg_meas"], raw["ecg_num"])
        e = ecg_block(idx, ecg_tbl, horizon_h, lookback)
        for k, name in LABEL_MAP.items():
            m = (idx.acs_label == k).to_numpy()
            kv(f"  ECG coverage | {name}", f"{e.loc[m, 'ecg_available'].mean()*100:6.2f}%")
        kv("features", e.shape[1])
    with timer("M7 labs"):
        l = lab_block(idx, raw["labs"], horizon_h)
    with timer("cross-modal interactions"):
        ix = interaction_block(v, d, t, e, l)

    X = pd.concat([v, d, t, med, h, e, l, ix], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]

    meta = idx[["subject_id", "hadm_id", "stay_id", "acs_label", "intime",
                "ed_los_h"]].copy()
    meta["chiefcomplaint_raw"] = idx["chiefcomplaint"].fillna("")
    meta["chiefcomplaint_norm"] = norm_text
    meta["chiefcomplaint_model"] = masked_text

    # ---------------- Intended Use Population ----------------
    section("Cohort selection — Intended Use Population")
    if CFG.get("cohort.enable", True):
        ecg_win = float(CFG.get("cohort.ecg_within_h", 3.0))
        e_cohort = ecg_block(idx, ecg_tbl, ecg_win, lookback)["ecg_available"] == 1
        cardiac_cc = TF.is_cardiac_presentation(norm_text)
        keep = cardiac_cc | e_cohort
        if CFG.get("cohort.require_triage_vitals", True):
            keep &= (v["vitals_available"] == 1)
        meta["in_cohort"] = keep.astype(np.int8)
        kv("criterion", f"cardiac chief complaint OR ECG within {ecg_win:g}h")
        kv("cohort size", f"{int(keep.sum()):,} / {len(keep):,} "
                          f"({keep.mean()*100:.1f}% of ED)")
        for k, name in LABEL_MAP.items():
            m = (idx.acs_label == k).to_numpy()
            kv(f"  {name} retained", f"{keep[m].mean()*100:6.2f}%  "
                                     f"({int(keep[m].sum()):,}/{int(m.sum()):,})")
        prev = (idx.acs_label[keep.to_numpy()] > 0).mean()
        kv("ACS prevalence in cohort", f"{prev*100:.2f}%  (full ED: "
                                       f"{(idx.acs_label>0).mean()*100:.2f}%)")
    else:
        meta["in_cohort"] = np.int8(1)

    modality_map = {c: assign_modality(c) for c in X.columns}
    info = {
        "horizon_h": horizon_h,
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "rdm_enabled": bool(rdm_enable),
        "modality_counts": pd.Series(modality_map).value_counts().to_dict(),
        "modality_map": modality_map,
        "feature_names": list(X.columns),
        "cohort_size": int(meta["in_cohort"].sum()),
        "label_counts": {LABEL_MAP[k]: int((idx.acs_label == k).sum())
                         for k in LABEL_MAP},
        "cohort_label_counts": {
            LABEL_MAP[k]: int(((idx.acs_label == k) & (meta.in_cohort == 1)).sum())
            for k in LABEL_MAP},
    }

    section("Feature matrix")
    kv("rows x features", f"{X.shape[0]:,} x {X.shape[1]}")
    for mod, n in sorted(info["modality_counts"].items(), key=lambda kv_: -kv_[1]):
        kv(f"  {mod}", f"{n:>3} features")
    kv("overall NaN rate", f"{X.isna().to_numpy().mean()*100:.2f}%")
    return pd.concat([meta.reset_index(drop=True), X.reset_index(drop=True)], axis=1), info


def main() -> None:
    raw = load_raw()
    summary = {}
    for h in CFG.horizons:
        df, info = build_features(h, raw=raw)
        out = os.path.join(DATA_DIR, f"features_H{h}.parquet")
        df.to_parquet(out, index=False)
        save_json(info, os.path.join(DATA_DIR, f"features_H{h}_info.json"))
        summary[f"H{h}"] = {k: v for k, v in info.items()
                            if k not in ("modality_map", "feature_names")}
        print(f"\n  [SAVED] {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
    save_json(summary, os.path.join(REPORT_DIR, "preprocessing_summary.json"))
    banner("PREPROCESSING COMPLETE")


if __name__ == "__main__":
    main()
