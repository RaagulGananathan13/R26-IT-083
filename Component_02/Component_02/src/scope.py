"""
scope.py — Detection of disease OUTSIDE the model's label space.

THE PROBLEM
-----------
This model has five output units: NORM, MI, STTC, CD, HYP. It has no unit for
atrial fibrillation, flutter, paced rhythm, or ventricular tachycardia. When one
of those walks in, the network does not abstain — softmax has no "none of the
above". It distributes the evidence across the five classes it does have, the
conformal layer converts that into a decision, and the report prints a
statistical guarantee next to it.

Measured on this project's own test fold (audit/13_out_of_scope.py): **113 of 114
recordings documenting atrial fibrillation or flutter received a report carrying
a conformal guarantee, and not one of those guarantees concerned atrial
fibrillation.** Two were reported as a normal ECG. AF is associated with roughly
one ischaemic stroke in four; those two patients are the ones who go home.

Every PTB-XL five-superclass paper inherits this failure. None report it,
because the benchmark only scores the five classes it defined.

THE CHECK
---------
Physiology, not machine learning. Atrial fibrillation is *defined* by an
irregularly irregular ventricular response, so the R-R interval series carries
the signal directly — and the pipeline already detects R peaks to compute heart
rate, so these features cost nothing.

    cv     coefficient of variation of RR
    rmssd  root mean square of successive RR differences
    pnn50  proportion of successive RR differences above 50 ms
    irr    median |ΔRR| normalised by median RR

Measured discrimination for AF/flutter against everything else, PTB-XL test
fold: AUROC 0.912 (irr), 0.907 (cv), 0.892 (pnn50). At a threshold fitted on the
FULL validation fold at a 5% false-positive budget (irr = 0.179): 48.9%
sensitivity, 4.8% FPR on the unseen test fold. Sensitivity is limited by the
false-positive budget, not by the feature: AUROC 0.912 means a looser budget
buys more sensitivity.

This does NOT diagnose atrial fibrillation — the system is not permitted to name
a class it was never trained on. It answers a narrower and more honest question:
*is this rhythm outside the region of ECG space where my guarantee was
calibrated?* If yes, the guarantee is withheld and the record is referred.

The decision threshold is fitted on the VALIDATION fold only, like every other
threshold in this system, and stored in checkpoints/scope.json.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Fallback used when checkpoints/scope.json is absent. Chosen on the validation
# fold at a 5% false-positive rate against non-arrhythmic records.
DEFAULT_IRR_THRESHOLD = 0.179


@dataclass
class ScopeReport:
    in_scope: bool = True
    confidence: float = 0.0          # 0..1, how far past the threshold
    reason: Optional[str] = None
    features: Dict[str, float] = field(default_factory=dict)
    detail: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def rr_features(r_peaks: np.ndarray, fs: int = 500) -> Optional[Dict[str, float]]:
    """R-R variability descriptors. Returns None if there are too few beats."""
    if r_peaks is None or len(r_peaks) < 5:
        return None
    rr = np.diff(np.asarray(r_peaks, dtype=float)) / float(fs)
    rr = rr[(rr > 0.25) & (rr < 2.5)]          # 24-240 bpm physiological window
    if len(rr) < 4:
        return None
    d = np.diff(rr)
    return {
        "cv": float(rr.std() / max(rr.mean(), 1e-9)),
        "rmssd": float(np.sqrt(np.mean(d ** 2))),
        "pnn50": float(np.mean(np.abs(d) > 0.05)),
        "irr": float(np.median(np.abs(d)) / max(np.median(rr), 1e-9)),
        "n_beats": int(len(rr) + 1),
        "mean_rr": float(rr.mean()),
    }


def load_threshold(path: Optional[str] = None) -> float:
    if path and os.path.exists(path):
        try:
            with open(path) as f:
                return float(json.load(f)["irr_threshold"])
        except Exception:
            pass
    return DEFAULT_IRR_THRESHOLD


def assess(r_peaks: np.ndarray, fs: int = 500,
           threshold: Optional[float] = None) -> ScopeReport:
    """Decide whether this rhythm lies outside the model's competence."""
    thr = DEFAULT_IRR_THRESHOLD if threshold is None else threshold
    f = rr_features(r_peaks, fs)
    if f is None:
        return ScopeReport(True, 0.0, None, {},
                           ["too few beats to assess rhythm regularity"])

    if f["irr"] <= thr:
        return ScopeReport(True, 0.0, None, f)

    conf = float(min(1.0, (f["irr"] - thr) / max(thr, 1e-9)))
    detail = [
        f"R-R intervals are irregularly irregular: normalised median |dRR| = "
        f"{f['irr']:.3f} against a threshold of {thr:.3f}; "
        f"{f['pnn50']*100:.0f}% of successive beats differ by more than 50 ms "
        f"(coefficient of variation {f['cv']:.3f}).",
        # Each sentence naming an out-of-scope condition must ALSO carry the
        # disclaimer, so the safety verifier can tell an admission of ignorance
        # from a hallucinated diagnosis. See verify.SCOPE_DISCLAIMERS.
        "This model has no output for, and cannot assess, atrial fibrillation "
        "or atrial flutter; the irregular pattern here is characteristic of "
        "such a rhythm and requires assessment by a clinician.",
    ]
    return ScopeReport(
        in_scope=False,
        confidence=round(conf, 3),
        reason="irregular ventricular response — rhythm outside the label space",
        features={k: round(v, 4) for k, v in f.items()},
        detail=detail,
    )


LIMITATION = (
    "This is a rhythm-regularity screen, not an arrhythmia classifier. It flags "
    "irregular ventricular response (AF, flutter with variable block, frequent "
    "ectopy). It will NOT catch out-of-scope disease that preserves a regular "
    "rhythm — paced rhythms at a fixed rate, monomorphic ventricular "
    "tachycardia, or Brugada and long-QT patterns. Those remain silent failures "
    "and require a different check."
)
