"""
Component 04 — ST-segment measurement from raw 12-lead ECG waveforms.

Why this module exists
----------------------
STEMI is *defined* by ST-segment elevation measured in millivolts.  MIMIC-IV-ED
ships the ECG cart's text report, not the signal, and that report names ST
elevation in only 41% of true STEMI cases.  Every feature-engineering attempt on
the text plateaued at a STEMI-vs-NSTEMI F1 of ~0.66, and the measured Bayes
bound for that subproblem is 91.04% accuracy — the separating information simply
is not in the text.

This module recovers it from the waveform, following the ESC/AHA Fourth
Universal Definition of Myocardial Infarction:

  * baseline    isoelectric level taken from the PR segment (the 20 ms window
                ending at QRS onset), per beat, per lead
  * J point     end of the QRS complex
  * ST level    amplitude at J + 60 ms relative to that baseline
  * criterion   >= 0.1 mV in two contiguous leads, except V2-V3 where the
                threshold is 0.2 mV (men >= 40), 0.25 mV (men < 40),
                0.15 mV (women)

Beats are detected, measured individually, then aggregated by median across
beats — a single ectopic or noisy beat should not decide a diagnosis.

Contiguity matters as much as amplitude: 0.1 mV in two anatomically adjacent
leads is an infarct, the same 0.1 mV scattered across unrelated leads is noise.
Territories are therefore encoded explicitly.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_p for _p in (
    _SRC, os.path.join(_SRC, "core"), os.path.join(_SRC, "data"),
    os.path.join(_SRC, "models"), os.path.join(_SRC, "analysis"),
) if _p not in sys.path]
warnings.filterwarnings("ignore")

from config import CFG, DATA_DIR, LABEL_MAP, enable_utf8_stdout, save_json
from utils import banner, df_to_markdown, kv, section

enable_utf8_stdout()

WAVE_DIR = os.path.join(DATA_DIR, "ecg_waveforms")
MANIFEST = os.path.join(DATA_DIR, "ecg_manifest.parquet")
OUT = os.path.join(DATA_DIR, "ecg_waveform_features.parquet")

# Standard MIMIC-IV-ECG lead order
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
         "V1", "V2", "V3", "V4", "V5", "V6"]

# Anatomical territories — contiguity is what separates infarct from noise
TERRITORY: Dict[str, List[str]] = {
    "anterior":     ["V1", "V2", "V3", "V4"],
    "lateral":      ["I", "aVL", "V5", "V6"],
    "inferior":     ["II", "III", "aVF"],
    "septal":       ["V1", "V2"],
    "anterolateral": ["V4", "V5", "V6"],
}
# Reciprocal pairs: elevation here + depression there is highly specific
RECIPROCAL = [("inferior", "lateral"), ("lateral", "inferior"),
              ("anterior", "inferior")]

J_OFFSET_MS = 60.0      # ST measured at J + 60 ms
PR_WINDOW_MS = 20.0     # isoelectric baseline window before QRS onset


# --------------------------------------------------------------------------
def remove_baseline(sig12: np.ndarray, fs: float) -> np.ndarray:
    """
    Strip baseline wander with the classic two-stage median filter
    (200 ms then 600 ms), applied per lead.

    This matters more than any other step here.  A 10-second strip routinely
    drifts 0.1-0.2 mV, and measuring ST amplitude against an uncorrected
    baseline reads that drift as elevation — which is exactly what the first
    run did, reporting ">=1mm in 2 leads" for 39% of unstable angina patients,
    a group defined by NOT having significant ST elevation.

    A high-pass filter is the obvious alternative and the wrong one: at the
    0.5 Hz cut-off NeuroKit's default cleaner uses, the filter distorts the ST
    segment itself.  Diagnostic ECG standards require <= 0.05 Hz for this
    reason, and median filtering sidesteps the trade-off entirely.
    """
    from scipy.ndimage import median_filter

    w1 = max(3, int(round(0.2 * fs)) | 1)      # 200 ms, odd
    w2 = max(3, int(round(0.6 * fs)) | 1)      # 600 ms, odd
    out = np.empty_like(sig12)
    for j in range(sig12.shape[1]):
        x = sig12[:, j]
        base = median_filter(x, size=w1, mode="nearest")
        base = median_filter(base, size=w2, mode="nearest")
        out[:, j] = x - base
    return out


def _delineate(sig: np.ndarray, fs: float) -> dict | None:
    """
    R-peaks plus per-beat QRS onset/offset.

    Tries NeuroKit's discrete-wavelet delineator first, then its peak-based
    method, then falls back to fixed physiological offsets from the R peak.
    The wavelet method throws on short or noisy strips (774 of 13,055 records
    in the first pass), and discarding 6% of the data — disproportionately the
    noisy, sick patients — is its own selection bias.
    """
    import neurokit2 as nk
    try:
        clean = nk.ecg_clean(sig, sampling_rate=fs, method="neurokit")
        _, info = nk.ecg_peaks(clean, sampling_rate=fs)
        rpeaks = np.asarray(info["ECG_R_Peaks"])
        if len(rpeaks) < 3:
            return None
    except Exception:
        return None

    for method in ("dwt", "peak"):
        try:
            _, w = nk.ecg_delineate(clean, rpeaks, sampling_rate=fs, method=method)
            on = np.asarray(w.get("ECG_R_Onsets", []), dtype=float)
            off = np.asarray(w.get("ECG_R_Offsets", []), dtype=float)
            if np.isfinite(on).sum() >= 2 and np.isfinite(off).sum() >= 2:
                return {"rpeaks": rpeaks, "qrs_on": on, "qrs_off": off,
                        "method": method}
        except Exception:
            continue

    # Fallback: fixed offsets from R.  QRS onset ~40 ms before R, offset ~40 ms
    # after — crude, but far better than dropping the record entirely, and the
    # J+60ms window is wide enough to tolerate a few ms of error.
    on = rpeaks - int(round(0.04 * fs))
    off = rpeaks + int(round(0.04 * fs))
    return {"rpeaks": rpeaks, "qrs_on": on.astype(float),
            "qrs_off": off.astype(float), "method": "fallback"}


def _st_levels(sig12: np.ndarray, fs: float, d: dict) -> np.ndarray | None:
    """
    Median ST amplitude (mV) at J+60 ms per lead, baselined on the PR segment.
    sig12 is (n_samples, 12).
    """
    j_off = int(round(J_OFFSET_MS * fs / 1000.0))
    pr_w = int(round(PR_WINDOW_MS * fs / 1000.0))
    n = sig12.shape[0]
    per_beat = []

    for k in range(len(d["rpeaks"])):
        on = d["qrs_on"][k] if k < len(d["qrs_on"]) else np.nan
        off = d["qrs_off"][k] if k < len(d["qrs_off"]) else np.nan
        if not np.isfinite(on) or not np.isfinite(off):
            continue
        on, off = int(on), int(off)
        b0, b1 = on - pr_w, on
        st = off + j_off
        if b0 < 0 or st >= n or b1 <= b0:
            continue
        baseline = sig12[b0:b1].mean(axis=0)        # (12,)
        per_beat.append(sig12[st] - baseline)       # (12,)

    if len(per_beat) < 2:
        return None
    return np.median(np.vstack(per_beat), axis=0)   # (12,) in mV


def _features(st: np.ndarray, sex_male: int | None = None,
              age: float | None = None) -> Dict[str, float]:
    """Turn per-lead ST levels into clinically meaningful features."""
    s = pd.Series(st, index=LEADS)
    f: Dict[str, float] = {}

    for lead in LEADS:
        f[f"wf_st_{lead}"] = float(s[lead])

    f["wf_st_max"] = float(s.max())
    f["wf_st_min"] = float(s.min())
    f["wf_st_absmax"] = float(s.abs().max())
    f["wf_n_leads_elev_1mm"] = int((s >= 0.10).sum())
    f["wf_n_leads_elev_2mm"] = int((s >= 0.20).sum())
    f["wf_n_leads_depr_1mm"] = int((s <= -0.10).sum())

    # --- territory aggregates -------------------------------------------
    for name, leads in TERRITORY.items():
        v = s[leads]
        f[f"wf_st_{name}_max"] = float(v.max())
        f[f"wf_st_{name}_mean"] = float(v.mean())
        f[f"wf_{name}_contig2"] = int((v >= 0.10).sum() >= 2)

    # --- ESC/AHA Fourth Universal Definition ----------------------------
    # V2-V3 carry a higher threshold; everything else 0.1 mV.
    if sex_male is None:
        thr_v23 = 0.20
    elif sex_male == 1:
        thr_v23 = 0.20 if (age is None or age >= 40) else 0.25
    else:
        thr_v23 = 0.15
    v23 = (s[["V2", "V3"]] >= thr_v23).sum() >= 2
    other = [l for l in LEADS if l not in ("V2", "V3")]
    contig_other = any(
        (s[leads] >= 0.10).sum() >= 2
        for leads in TERRITORY.values()
        if len([l for l in leads if l in other]) >= 2
    )
    f["wf_stemi_criteria"] = int(bool(v23 or contig_other))
    f["wf_v23_threshold_used"] = float(thr_v23)

    # --- reciprocal change ----------------------------------------------
    recip = 0
    for a, b in RECIPROCAL:
        if s[TERRITORY[a]].max() >= 0.10 and s[TERRITORY[b]].min() <= -0.05:
            recip = 1
            break
    f["wf_reciprocal_change"] = recip

    # NSTEMI signature: depression / no elevation anywhere
    f["wf_depression_only"] = int(f["wf_n_leads_depr_1mm"] >= 2
                                  and f["wf_n_leads_elev_1mm"] == 0)
    f["wf_st_spread"] = float(s.max() - s.min())
    return f


# --------------------------------------------------------------------------
def process(limit: int | None = None, workers: int = 1) -> pd.DataFrame:
    import wfdb

    if not os.path.exists(MANIFEST):
        raise FileNotFoundError(f"{MANIFEST} — run ecg_fetch.py first")
    man = pd.read_parquet(MANIFEST)

    have = [s for s in man.study_id.astype(str)
            if os.path.exists(os.path.join(WAVE_DIR, s, s + ".hea"))]
    kv("studies in manifest", f"{len(man):,}")
    kv("waveforms on disk", f"{len(have):,}")
    if not have:
        raise RuntimeError("no waveforms downloaded yet — run ecg_fetch.py")
    if limit:
        have = have[:limit]

    meta = man.set_index(man.study_id.astype(str))
    rows, failed = [], 0
    for i, stem in enumerate(have, 1):
        try:
            rec = wfdb.rdrecord(os.path.join(WAVE_DIR, stem, stem))
            sig = np.asarray(rec.p_signal, dtype=np.float64)   # (n, 12) in mV
            fs = float(rec.fs)
            names = [str(x) for x in rec.sig_name]
            idx = [names.index(l) for l in LEADS if l in names]
            if len(idx) != 12:
                failed += 1
                continue
            sig = sig[:, idx]
            sig = np.nan_to_num(sig, nan=0.0)
            # Baseline correction BEFORE any amplitude is read.  Previously the
            # fiducials were found on a cleaned lead II while amplitudes were
            # measured on the raw signal, so drift was counted as elevation.
            sig = remove_baseline(sig, fs)

            lead_ii = sig[:, LEADS.index("II")]
            d = _delineate(lead_ii, fs)
            if d is None:
                failed += 1
                continue
            st = _st_levels(sig, fs, d)
            if st is None:
                failed += 1
                continue

            r = _features(st)
            r["study_id"] = int(stem)
            r["stay_id"] = int(meta.loc[stem, "stay_id"])
            r["wf_n_beats"] = int(len(d["rpeaks"]))
            r["wf_delineation"] = d.get("method", "dwt")
            rows.append(r)
        except KeyboardInterrupt:
            print("\n  interrupted - saving what has been processed so far")
            break
        except Exception:
            failed += 1
        if i % 200 == 0 or i == len(have):
            sys.stdout.write(f"\r  processed {i:,}/{len(have):,}  "
                             f"ok={len(rows):,} failed={failed:,}   ")
            sys.stdout.flush()
    print()

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("no waveform yielded usable measurements")

    # one row per stay: worst (most elevated) ECG in the window
    df = df.sort_values("wf_st_max", ascending=False).drop_duplicates("stay_id")
    df.to_parquet(OUT, index=False)
    kv("\n  usable measurements", f"{len(rows):,}")
    kv("stays covered", f"{df.stay_id.nunique():,}")
    kv("saved", OUT)
    return df


def validate(df: pd.DataFrame) -> None:
    """Does measured ST-elevation actually separate STEMI from NSTEMI?"""
    section("Validation against labels")
    man = pd.read_parquet(MANIFEST)[["stay_id", "acs_label"]].drop_duplicates("stay_id")
    d = df.merge(man, on="stay_id", how="left").dropna(subset=["acs_label"])
    d["acs_label"] = d["acs_label"].astype(int)

    rows = []
    for k, name in LABEL_MAP.items():
        m = d.acs_label == k
        if m.sum() < 5:
            continue
        rows.append({"class": name, "n": int(m.sum()),
                     "mean ST max (mV)": float(d.loc[m, "wf_st_max"].mean()),
                     "P(>=1mm in 2 leads)": float(d.loc[m, "wf_stemi_criteria"].mean()),
                     "mean leads elevated": float(d.loc[m, "wf_n_leads_elev_1mm"].mean())})
    print(df_to_markdown(pd.DataFrame(rows)))

    m = d.acs_label.isin([2, 3])
    if m.sum() > 20:
        from sklearn.metrics import roc_auc_score
        y = (d.loc[m, "acs_label"] == 3).astype(int)
        print()
        for feat in ("wf_st_max", "wf_stemi_criteria", "wf_n_leads_elev_1mm",
                     "wf_st_anterior_max", "wf_st_inferior_max"):
            if feat in d:
                try:
                    auc = roc_auc_score(y, d.loc[m, feat])
                    kv(f"STEMI-vs-NSTEMI AUROC · {feat}", f"{auc:.4f}")
                except Exception:
                    pass
        print("\n  For reference, the best TEXT-derived ECG feature reached 0.723")
        print("  and the full text-based model plateaued at AUROC 0.857.")
        print("  If measured ST-elevation clears that, the ceiling has moved.")


def main() -> None:
    ap = argparse.ArgumentParser(description="ST-segment features from waveforms")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N studies (smoke test)")
    a = ap.parse_args()
    banner("ECG WAVEFORM — ST-SEGMENT MEASUREMENT")
    df = process(limit=a.limit)
    validate(df)
    banner("WAVEFORM FEATURES COMPLETE")
    print(f"  Merge {os.path.basename(OUT)} into preprocess.py on stay_id,")
    print("  then rerun split -> train -> evaluate.")


if __name__ == "__main__":
    main()
