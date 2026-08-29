"""
Component 04 — single-patient inference with explanation.

Two entry points:

    python predict.py --demo                 # four worked examples
    python predict.py --json patient.json    # your own patient
    python predict.py --stay-id 31234567     # replay a real test-fold encounter

The clinical input is a flat dictionary.  Every field is optional: the model was
built with missingness-aware encoding, so an absent modality is represented as
absent rather than imputed to a population average, and the prediction degrades
gracefully instead of silently assuming an average troponin.

    {
      "age": 68, "sex": "M",
      "heartrate": 104, "sbp": 96, "dbp": 62, "resprate": 22,
      "o2sat": 94, "temperature": 98.1, "pain": 8, "acuity": 2,
      "chief_complaint": "Chest pain radiating to left arm, diaphoresis",
      "troponin": [0.9, 3.4],          # serial values, ED order
      "troponin_hours": [1.0, 4.0],    # hours after arrival
      "bnp": 850,
      "ecg": {"st_elevation": true, "acute": true, "qrs_duration": 96},
      "home_medications": ["aspirin", "atorvastatin", "metoprolol"],
      "prior_ed_visits": 2, "prior_acs": 0
    }
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd


@contextlib.contextmanager
def _quiet():
    """Silence the pipeline's per-module reporting for single-row inference."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield

_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_p for _p in (
    _SRC, os.path.join(_SRC, "core"), os.path.join(_SRC, "data"),
    os.path.join(_SRC, "models"), os.path.join(_SRC, "analysis"),
) if _p not in sys.path]

from config import CFG, DATA_DIR, LABEL_ORDER, enable_utf8_stdout
import preprocess as PP
import text_features as TF
from inference import ACSPredictor
from utils import banner, kv, section

enable_utf8_stdout()

RISK = [
    (0.80, "CRITICAL", "Immediate cardiology activation. If ST-elevation: cath lab now."),
    (0.50, "HIGH", "Urgent ECG review, serial troponin, admit for monitoring."),
    (0.20, "MODERATE", "Serial troponin at 0/3h, observation, risk stratify."),
    (0.05, "LOW", "Consider alternative diagnoses; single troponin may suffice."),
    (0.00, "MINIMAL", "ACS unlikely on current evidence; pursue other causes."),
]


def risk_band(p: float):
    for thr, name, action in RISK:
        if p >= thr:
            return name, action
    return "MINIMAL", RISK[-1][2]


# --------------------------------------------------------------------------
def build_row(pt: Dict) -> pd.DataFrame:
    """Turn a flat clinical dictionary into one model-ready feature row."""
    now = pd.Timestamp("2150-01-01 12:00:00")
    idx = pd.DataFrame([{
        "subject_id": 1, "hadm_id": 1, "stay_id": 1,
        "gender": str(pt.get("sex", "")).upper()[:1],
        "race": pt.get("race", "UNKNOWN"),
        "intime": now, "outtime": now + pd.Timedelta(hours=6),
        "anchor_age": pt.get("age", np.nan),
        "temperature": pt.get("temperature", np.nan),
        "heartrate": pt.get("heartrate", np.nan),
        "resprate": pt.get("resprate", np.nan),
        "o2sat": pt.get("o2sat", np.nan),
        "sbp": pt.get("sbp", np.nan), "dbp": pt.get("dbp", np.nan),
        "pain": pt.get("pain", np.nan), "acuity": pt.get("acuity", np.nan),
        "chiefcomplaint": pt.get("chief_complaint", ""),
        "acs_label": 0, "ed_los_h": 6.0,
    }])

    with _quiet():
        v = PP.vitals_block(idx)
        d = PP.demographics_block(idx)
    norm = TF.normalise(idx["chiefcomplaint"])
    masked, rdm = TF.apply_rdm(norm, enable=bool(CFG.get("text.rdm_enable", True)))
    t = pd.concat([TF.lexicon_features(masked), rdm], axis=1)
    t["text_available"] = (t["cc_n_tokens"] > 0).astype(np.int8)

    # --- medications ---------------------------------------------------
    meds = [str(m).lower() for m in pt.get("home_medications", [])]
    med = pd.DataFrame(index=idx.index)
    for cls, pat in PP.MED_CLASSES.items():
        med[cls] = int(any(pd.Series(meds).str.contains(pat, regex=True).any()
                           for _ in [0])) if meds else 0
    med["med_total_count"] = float(len(meds))
    cardiac = ["med_antiplatelet", "med_statin", "med_betablocker", "med_acearb",
               "med_nitrate", "med_anticoagulant", "med_ccb", "med_antiarrhythmic"]
    med["med_cardiac_count"] = med[cardiac].sum(axis=1).astype(np.int8)
    med["med_secondary_prevention"] = (
        med[["med_antiplatelet", "med_statin", "med_betablocker"]].sum(axis=1) >= 2
    ).astype(np.int8)
    med["meds_available"] = np.int8(1 if meds else 0)

    # --- prior history --------------------------------------------------
    h = pd.DataFrame(index=idx.index)
    h["hist_n_prior_visits"] = float(pt.get("prior_ed_visits", 0))
    h["hist_days_since_last"] = float(pt.get("days_since_last_visit", np.nan))
    h["hist_has_prior"] = np.int8(1 if pt.get("prior_ed_visits", 0) else 0)
    for name in ("ua", "nstemi", "stemi"):
        h[f"hist_prior_{name}"] = 0.0
    h["hist_prior_acs_any"] = float(pt.get("prior_acs", 0))
    h["hist_revisit_30d"] = np.int8(0); h["hist_revisit_365d"] = np.int8(0)
    h["hist_frequent_user"] = np.int8(1 if pt.get("prior_ed_visits", 0) >= 5 else 0)
    h["hist_charlson_index"] = float(pt.get("charlson_index", 0))
    h["hist_prior_mi_icd"] = float(pt.get("prior_mi", 0))
    h["hist_prior_chf_icd"] = float(pt.get("prior_chf", 0))
    h["hist_diabetes"] = np.int8(pt.get("diabetes", 0))
    h["hist_renal_disease"] = float(pt.get("renal_disease", 0))
    h["hist_charlson_available"] = np.int8(1 if pt.get("charlson_index") else 0)

    # --- ECG -------------------------------------------------------------
    ecg_in = pt.get("ecg") or {}
    e = pd.DataFrame(index=idx.index)
    for f in PP.ECG_FINDINGS:
        e[f] = np.int8(bool(ecg_in.get(f.replace("ecg_", ""), False)))
    e["ecg_n_findings"] = np.int8(sum(int(e[f].iloc[0]) for f in PP.ECG_FINDINGS))
    e["ecg_report_len"] = np.int16(60 if ecg_in else 0)
    for c in PP.ECG_NUMERIC_BOUNDS:
        e[f"ecg_{c}"] = np.float32(ecg_in.get(c, np.nan))
    qt, rr = e.get("ecg_qt_interval"), e.get("ecg_rr_interval")
    e["ecg_qtc"] = (qt / np.sqrt((rr / 1000.0).clip(lower=0.1))).astype(np.float32)
    e["ecg_qtc_prolonged"] = (e["ecg_qtc"] > 460).fillna(False).astype(np.int8)
    e["ecg_hr_from_rr"] = (60000.0 / rr.replace(0, np.nan)).astype(np.float32)
    e["ecg_wide_qrs"] = (e["ecg_qrs_duration"] > 120).fillna(False).astype(np.int8)
    e["ecg_left_axis"] = (e["ecg_qrs_axis"] < -30).fillna(False).astype(np.int8)
    e["ecg_right_axis"] = (e["ecg_qrs_axis"] > 100).fillna(False).astype(np.int8)
    e["ecg_first_degree_block"] = (e["ecg_pr_interval"] > 200).fillna(False).astype(np.int8)
    e["ecg_n_studies"] = np.float32(1 if ecg_in else 0)
    e["ecg_dt_first_h"] = np.float32(ecg_in.get("hours_after_arrival", 0.2)
                                     if ecg_in else np.nan)
    e["ecg_available"] = np.int8(1 if ecg_in else 0)
    e["ecg_immediate"] = np.int8(1 if ecg_in and e["ecg_dt_first_h"].iloc[0] <= 0.5 else 0)
    e["ecg_stemi_equivalent"] = np.int8(bool(e["ecg_st_elevation"].iloc[0] or
                                             e["ecg_lbbb"].iloc[0]))
    e["ecg_ischemic_any"] = np.int8(int(e[["ecg_st_elevation", "ecg_st_depression",
                                           "ecg_t_inversion", "ecg_q_wave",
                                           "ecg_infarct_any"]].sum(axis=1).iloc[0] > 0))
    e["ecg_acute_ischemia"] = np.int8(int(e["ecg_acute"].iloc[0] and
                                          e["ecg_infarct_any"].iloc[0]))
    e["ecg_old_infarct_only"] = np.int8(int(e["ecg_age_undetermined"].iloc[0] and
                                            not e["ecg_acute"].iloc[0]))
    e["ecg_acuity_score"] = np.float32(
        e["ecg_acute"].iloc[0] * 3 + e["ecg_critical_alert"].iloc[0] * 2 +
        e["ecg_stemi_alert"].iloc[0] * 3 + e["ecg_acute_mi"].iloc[0] * 2 +
        e["ecg_st_elevation"].iloc[0] * 2 - e["ecg_age_undetermined"].iloc[0] -
        e["ecg_infarct_possible"].iloc[0])
    e["ecg_territory_count"] = np.int8(int(e[["ecg_infarct_inferior",
                                              "ecg_infarct_anterior",
                                              "ecg_infarct_lateral"]].sum(axis=1).iloc[0]))

    # --- biomarkers ------------------------------------------------------
    trop = [float(x) for x in (pt.get("troponin") or [])]
    th = [float(x) for x in (pt.get("troponin_hours") or list(range(len(trop))))]
    l = pd.DataFrame(index=idx.index)
    l["trop_first"] = np.float32(trop[0] if trop else np.nan)
    l["trop_second"] = np.float32(trop[1] if len(trop) > 1 else np.nan)
    l["trop_max"] = np.float32(max(trop) if trop else np.nan)
    l["trop_n_draws"] = np.float32(len(trop))
    l["trop_t_first_h"] = np.float32(th[0] if trop else np.nan)
    l["trop_span_h"] = np.float32((th[-1] - th[0]) if len(trop) > 1 else np.nan)
    l["trop_available"] = np.int8(1 if trop else 0)
    l["trop_serial"] = np.int8(1 if len(trop) >= 2 else 0)
    l["trop_delta"] = (l["trop_second"] - l["trop_first"]).astype(np.float32)
    l["trop_delta_pct"] = (l["trop_delta"] / l["trop_first"].replace(0, np.nan)).astype(np.float32)
    l["trop_delta_rate"] = (l["trop_delta"] / l["trop_span_h"].replace(0, np.nan)).astype(np.float32)
    l["trop_log_first"] = np.log1p(l["trop_first"].clip(lower=0)).astype(np.float32)
    l["trop_log_max"] = np.log1p(l["trop_max"].clip(lower=0)).astype(np.float32)
    for thr, tag in ((0.04, "url"), (0.1, "mod"), (0.5, "high"), (1.0, "vhigh")):
        l[f"trop_gt_{tag}"] = (l["trop_max"] > thr).fillna(False).astype(np.int8)
    l["trop_rising"] = (l["trop_delta"] > 0.01).fillna(False).astype(np.int8)
    bnp = pt.get("bnp")
    l["bnp_first"] = np.float32(bnp if bnp is not None else np.nan)
    l["bnp_max"] = l["bnp_first"]
    l["bnp_t_first_h"] = np.float32(1.0 if bnp is not None else np.nan)
    l["bnp_available"] = np.int8(1 if bnp is not None else 0)
    l["bnp_log_max"] = np.log1p(l["bnp_max"].clip(lower=0)).astype(np.float32)
    l["bnp_gt_400"] = (l["bnp_max"] > 400).fillna(False).astype(np.int8)
    l["labs_any_available"] = np.int8(int(l["trop_available"].iloc[0] or
                                          l["bnp_available"].iloc[0]))
    l["labs_workup_intensity"] = np.float32(len(trop) + l["bnp_available"].iloc[0] +
                                            l["trop_serial"].iloc[0] * 2)

    ix = PP.interaction_block(v, d, t, e, l)
    return pd.concat([v, d, t, med, h, e, l, ix], axis=1)


# --------------------------------------------------------------------------
def report(pt: Dict, pred: ACSPredictor) -> Dict:
    X = build_row(pt)
    # text-SVD columns come from the fitted embedder
    import joblib
    from config import MODEL_DIR
    emb_path = os.path.join(MODEL_DIR, f"text_embedder_H{pred.horizon}.joblib")
    if os.path.exists(emb_path):
        emb = joblib.load(emb_path)
        norm = TF.normalise(pd.Series([pt.get("chief_complaint", "")]))
        masked, _ = TF.apply_rdm(norm, enable=bool(CFG.get("text.rdm_enable", True)))
        Z = emb.transform(masked); Z.index = X.index
        X = pd.concat([X, Z], axis=1)

    out = pred.explain_row(X, 0, pt.get("chief_complaint", ""))
    band, action = risk_band(out["p_acs"])
    out["risk_level"] = band
    out["recommended_action"] = action

    section(f"PATIENT: {pt.get('label', 'unnamed')}")
    cc = pt.get("chief_complaint", "(none)")
    kv("chief complaint", cc[:60])
    kv("age / sex", f"{pt.get('age','?')} / {pt.get('sex','?')}")
    kv("vitals", f"HR {pt.get('heartrate','-')}  BP {pt.get('sbp','-')}/"
                 f"{pt.get('dbp','-')}  SpO2 {pt.get('o2sat','-')}  "
                 f"RR {pt.get('resprate','-')}")
    kv("troponin", pt.get("troponin", "not ordered"))
    kv("ECG", pt.get("ecg", "not performed"))
    print()
    kv(">> PREDICTION", f"{out['prediction']}")
    kv(">> P(ACS)", f"{out['p_acs']*100:.1f}%")
    kv(">> RISK", f"{band}")
    kv(">> ACTION", action)
    print("\n  Four-class probabilities:")
    for c in LABEL_ORDER:
        p = out["probabilities"][c]
        print(f"    {c:<8} {p*100:6.2f}%  " + "#" * int(p * 44))
    print("\n  Subtype distribution given ACS:")
    for c, p in out["subtype_probabilities"].items():
        print(f"    {c:<8} {p*100:6.2f}%  " + "#" * int(p * 44))
    if out.get("text_attribution"):
        print("\n  Chief-complaint findings:")
        for tok in out["text_attribution"][:6]:
            flag = "  [NEGATED]" if tok["negated"] else ""
            print(f"    '{tok['term']}' -> {tok['category']} "
                  f"(weight {tok['weight']:+.1f}){flag}")
    return out


DEMO: List[Dict] = [
    {"label": "Anterior STEMI", "age": 61, "sex": "M", "heartrate": 108, "sbp": 92,
     "dbp": 58, "resprate": 24, "o2sat": 93, "temperature": 98.2, "pain": 9,
     "acuity": 1,
     "chief_complaint": "Crushing chest pain radiating to left arm with diaphoresis",
     "troponin": [1.2, 6.8], "troponin_hours": [0.8, 3.5],
     "ecg": {"st_elevation": True, "acute": True, "critical_alert": True,
             "infarct_any": True, "infarct_anterior": True, "qrs_duration": 98,
             "hours_after_arrival": 0.15},
     "home_medications": ["aspirin", "atorvastatin"], "prior_ed_visits": 1},
    {"label": "NSTEMI", "age": 74, "sex": "F", "heartrate": 92, "sbp": 138,
     "dbp": 80, "resprate": 20, "o2sat": 96, "temperature": 98.6, "pain": 6,
     "acuity": 2, "chief_complaint": "Chest pressure and shortness of breath",
     "troponin": [0.28, 0.51], "troponin_hours": [1.2, 4.0], "bnp": 620,
     "ecg": {"st_depression": True, "t_inversion": True, "age_undetermined": True,
             "infarct_any": True, "qrs_duration": 104, "hours_after_arrival": 0.4},
     "home_medications": ["metoprolol", "lisinopril", "atorvastatin"],
     "prior_ed_visits": 3, "diabetes": 1},
    {"label": "Unstable angina", "age": 58, "sex": "M", "heartrate": 78, "sbp": 146,
     "dbp": 88, "resprate": 16, "o2sat": 98, "temperature": 98.4, "pain": 5,
     "acuity": 2,
     "chief_complaint": "Chest pain on exertion, resolved at rest",
     "troponin": [0.01], "troponin_hours": [1.5],
     "ecg": {"normal": True, "qrs_duration": 88, "hours_after_arrival": 0.3},
     "home_medications": ["aspirin"], "prior_ed_visits": 0},
    {"label": "Non-cardiac (no tests ordered)", "age": 34, "sex": "F",
     "heartrate": 84, "sbp": 118, "dbp": 74, "resprate": 16, "o2sat": 99,
     "temperature": 99.1, "pain": 4, "acuity": 3,
     "chief_complaint": "Abdominal pain and nausea, denies chest pain",
     "home_medications": [], "prior_ed_visits": 0},
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Component 04 single-patient inference")
    ap.add_argument("--json", help="path to a patient JSON file")
    ap.add_argument("--demo", action="store_true", help="run the built-in examples")
    ap.add_argument("--stay-id", type=int, help="replay a real encounter by stay_id")
    ap.add_argument("--horizon", type=int, default=None)
    a = ap.parse_args()

    banner("COMPONENT 04 — ACS TRIAGE DECISION SUPPORT")
    pred = ACSPredictor.load(a.horizon)
    kv("model horizon", f"H = {pred.horizon}h after ED arrival")
    kv("Stage 1 threshold", f"{pred.stage1_cfg['threshold']:.4f}")
    kv("features", len(pred.features))

    if a.stay_id:
        df = pd.read_parquet(os.path.join(DATA_DIR, f"features_H{pred.horizon}.parquet"))
        row = df[df.stay_id == a.stay_id]
        if row.empty:
            print(f"\n  stay_id {a.stay_id} not found"); return
        X = row[[c for c in pred.features if c in row.columns]]
        out = pred.explain_row(X, 0, row.iloc[0].get("chiefcomplaint_raw", ""))
        section(f"stay_id {a.stay_id}")
        kv("true label", LABEL_ORDER[int(row.iloc[0].acs_label)])
        kv("predicted", out["prediction"])
        kv("P(ACS)", f"{out['p_acs']*100:.1f}%")
        for c in LABEL_ORDER:
            print(f"    {c:<8} {out['probabilities'][c]*100:6.2f}%")
        return

    patients = DEMO if (a.demo or not a.json) else [json.load(open(a.json, encoding="utf-8"))]
    results = [report(p, pred) for p in patients]

    if a.demo or not a.json:
        section("Summary")
        print(f"  {'patient':<32}{'predicted':>12}{'P(ACS)':>10}{'risk':>12}")
        print("  " + "-" * 66)
        for p, r in zip(patients, results):
            print(f"  {p['label']:<32}{r['prediction']:>12}"
                  f"{r['p_acs']*100:>9.1f}%{r['risk_level']:>12}")
        print("\n  Note the last case: no troponin and no ECG were ordered.  The model")
        print("  reports on what exists rather than imputing a population-average")
        print("  biomarker, which is the missingness-aware design working as intended.")


if __name__ == "__main__":
    main()
