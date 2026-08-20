"""
quality.py — Signal-quality gate and unit/gain normalisation.

Fixes audit finding C-1, the most dangerous defect in the archive:

    all-zero (disconnected leads) -> MI 0.691, STTC 0.602 -> "consistent with
                                     myocardial infarction"
    pure Gaussian noise           -> MI 0.629, CD 0.657
    same ECG in microvolts        -> STTC 1.000, HYP 1.000
    same ECG inverted             -> MI 0.818, CD 0.755

The archive fed anything it was given straight into the classifier. Nothing may
reach the model now without passing these checks. A record that fails is
REFUSED, not diagnosed — refusing is the only safe behaviour.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import List

import numpy as np
from scipy import signal as sps

from .models import LEAD_NAMES, SAMPLING_RATE

# Physiological plausibility bands for a diagnostic 12-lead ECG in mV.
MIN_QRS_AMPLITUDE_MV = 0.05     # below this the trace is essentially flat
MAX_ABS_AMPLITUDE_MV = 20.0     # above this it is saturation or a unit error
TYPICAL_P95_MV = (0.15, 6.0)    # expected band for the 95th pct of |x|
MAX_FLAT_LEADS = 2              # >2 dead leads => cannot interpret 12-lead ECG
MIN_HR_BPM, MAX_HR_BPM = 25.0, 250.0
MIN_DURATION_S, MAX_DURATION_S = 5.0, 60.0


def _load_scope_threshold():
    """R-R irregularity threshold, fitted on the VALIDATION fold by
    audit/13_out_of_scope.py. Falls back to the module default if absent."""
    import json as _json
    fp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "checkpoints", "scope.json")
    try:
        with open(fp) as f:
            return float(_json.load(f)["irr_threshold"])
    except Exception:
        return None


_SCOPE_THRESHOLD = _load_scope_threshold()


@dataclass
class QualityReport:
    acceptable: bool = True
    sqi: float = 1.0                                  # 0..1 overall quality index
    errors: List[str] = field(default_factory=list)   # hard failures -> refuse
    warnings: List[str] = field(default_factory=list) # soft -> report but proceed
    gain_correction: float = 1.0                      # factor applied to reach mV
    flat_leads: List[str] = field(default_factory=list)
    noisy_leads: List[str] = field(default_factory=list)
    heart_rate_bpm: float | None = None
    n_beats: int = 0
    duration_s: float = 0.0
    p95_amplitude_mv: float = 0.0
    # Electrode placement. Deliberately NOT folded into `sqi` or `acceptable`:
    # a reversed recording is high-quality, it is just wired wrong, and the
    # detector's ~4.5% false-positive rate makes automatic refusal the wrong
    # default. It suspends the statistical guarantee instead. See
    # src/electrodes.py and audit/12_electrode_reversal.py.
    electrode_suspect: bool = False
    electrode_reversal: str | None = None
    electrode_confidence: float = 0.0
    electrode_reasons: List[str] = field(default_factory=list)
    # Out-of-scope rhythm. Same policy as electrode reversal: the record is
    # still analysed, but the conformal guarantee is withheld, because the
    # bound was never calibrated on disease the label space cannot express.
    out_of_scope: bool = False
    scope_reason: str | None = None
    scope_confidence: float = 0.0
    scope_detail: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    def summary(self) -> str:
        if not self.acceptable:
            return "REFUSED: " + "; ".join(self.errors)
        if self.warnings:
            return f"Acceptable (SQI {self.sqi:.2f}) with caveats: " + "; ".join(self.warnings)
        return f"Good quality (SQI {self.sqi:.2f})"


# ──────────────────────────────────────────────────────────────────────────
def detect_and_fix_units(x: np.ndarray) -> tuple[np.ndarray, float, str | None]:
    """Infer the amplitude unit and rescale to millivolts.

    A raw ECG in mV has |x| p95 around 0.15-6.0. A microvolt file is ~1000x
    larger, a volt file ~1000x smaller. The archive had no check at all, so a
    microvolt upload produced 'STTC 100%, HYP 100%'.
    """
    p95 = float(np.percentile(np.abs(x), 95))
    if p95 <= 0:
        return x, 1.0, None
    if TYPICAL_P95_MV[0] <= p95 <= TYPICAL_P95_MV[1]:
        return x, 1.0, None
    # ONLY genuine unit conventions (powers of 1000) are accepted, and the
    # rescaled amplitude must land comfortably inside the band, not on its edge.
    # An arbitrary factor that happens to fit is a gain FAULT, not a unit, and
    # must be refused: a x50 recording rescaled by 0.01 lands at the very edge of
    # the band and would otherwise be silently "corrected" into a wrong report.
    safe_lo, safe_hi = TYPICAL_P95_MV[0] * 1.5, TYPICAL_P95_MV[1] / 1.5
    for factor, note in ((1e-3, "microvolts"), (1e3, "volts"), (1e-6, "nanovolts")):
        if safe_lo <= p95 * factor <= safe_hi:
            return x * factor, factor, f"input appeared to be in {note}; rescaled to mV"
    return x, 1.0, (f"amplitude is implausible for a diagnostic ECG (95th percentile "
                    f"|x| = {p95:.4g}) and does not match any standard unit "
                    f"convention; the recorder gain cannot be established")


def detect_r_peaks(lead_ii: np.ndarray, fs: int = SAMPLING_RATE) -> np.ndarray:
    """Lightweight Pan-Tompkins-style QRS detector (no external dependency)."""
    nyq = fs / 2.0
    lo, hi = max(5.0 / nyq, 1e-4), min(15.0 / nyq, 0.99)
    b, a = sps.butter(2, [lo, hi], btype="band")
    filt = sps.filtfilt(b, a, lead_ii)
    diff = np.diff(filt, prepend=filt[0])
    sq = diff ** 2
    win = max(int(0.12 * fs), 1)
    integ = np.convolve(sq, np.ones(win) / win, mode="same")
    if integ.max() <= 0:
        return np.array([], dtype=int)
    peaks, _ = sps.find_peaks(integ,
                              height=0.25 * np.mean(integ) + 0.1 * np.max(integ),
                              distance=int(0.25 * fs))     # 240 bpm ceiling
    return peaks


def assess(signal_raw: np.ndarray, fs: int = SAMPLING_RATE,
           lead_names: List[str] | None = None) -> tuple[np.ndarray, QualityReport]:
    """Gate an ECG before it is allowed near the classifier.

    Args:
        signal_raw: (n_samples, 12) in whatever unit the file used.
        fs:         sampling rate of `signal_raw`.
    Returns:
        (signal in mV, QualityReport). If `report.acceptable` is False the
        caller MUST NOT classify — see pipeline.analyse().
    """
    rep = QualityReport()
    lead_names = lead_names or LEAD_NAMES
    x = np.asarray(signal_raw, dtype=np.float64)

    # ── shape ────────────────────────────────────────────────────────────
    if x.ndim != 2:
        rep.acceptable = False
        rep.errors.append(f"expected a 2-D array, got shape {x.shape}")
        return x.astype(np.float32), rep
    if x.shape[0] < x.shape[1]:                    # (12, N) supplied
        x = x.T
    if x.shape[1] != 12:
        rep.acceptable = False
        rep.errors.append(f"expected 12 leads, got {x.shape[1]}")
        return x.astype(np.float32), rep

    rep.duration_s = x.shape[0] / float(fs)
    if not (MIN_DURATION_S <= rep.duration_s <= MAX_DURATION_S):
        rep.acceptable = False
        rep.errors.append(
            f"recording is {rep.duration_s:.1f}s; only {MIN_DURATION_S:.0f}-"
            f"{MAX_DURATION_S:.0f}s is supported (the archive silently resampled "
            f"any length to 10s, distorting heart rate)")
        return x.astype(np.float32), rep

    # ── finite values ────────────────────────────────────────────────────
    if not np.isfinite(x).all():
        n_bad = int((~np.isfinite(x)).sum())
        if n_bad > 0.001 * x.size:
            rep.acceptable = False
            rep.errors.append(f"{n_bad} non-finite samples")
            return np.nan_to_num(x).astype(np.float32), rep
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        rep.warnings.append(f"interpolated {n_bad} non-finite samples")

    # ── flat / dead leads ────────────────────────────────────────────────
    # Checked BEFORE unit inference, using a RELATIVE criterion. Dead leads drag
    # the amplitude percentile down, so running the unit check first made a
    # record with 8 disconnected leads look like a gain error instead. A
    # relative criterion is also unit-independent by construction.
    lead_std = x.std(axis=0)
    lead_ptp = np.ptp(x, axis=0)
    live_ptp = lead_ptp[lead_ptp > 0]
    ref_ptp = float(np.median(live_ptp)) if live_ptp.size else 0.0
    flat = [lead_names[i] for i in range(12)
            if lead_std[i] < 1e-9 or (ref_ptp > 0 and lead_ptp[i] < 0.02 * ref_ptp)]
    rep.flat_leads = flat
    if len(flat) == 12:
        rep.acceptable = False
        rep.errors.append("every lead is flat — the electrodes are disconnected "
                          "or this is not an ECG")
        return x.astype(np.float32), rep
    if len(flat) > MAX_FLAT_LEADS:
        rep.acceptable = False
        rep.errors.append(f"{len(flat)} flat/disconnected leads ({', '.join(flat)}); "
                          f"at most {MAX_FLAT_LEADS} can be tolerated")
        return x.astype(np.float32), rep
    if flat:
        rep.warnings.append(f"flat lead(s): {', '.join(flat)} — interpretation of "
                            f"the affected territory is unreliable")

    # ── units / gain ─────────────────────────────────────────────────────
    # Inferred from the LIVE leads only.
    live_idx = [i for i in range(12) if lead_names[i] not in flat]
    _, gain, unit_note = detect_and_fix_units(x[:, live_idx])
    x = x * gain
    rep.gain_correction = gain
    if unit_note:
        if gain == 1.0:
            rep.acceptable = False
            rep.errors.append(unit_note)
            return x.astype(np.float32), rep
        rep.warnings.append(unit_note)
    rep.p95_amplitude_mv = float(np.percentile(np.abs(x[:, live_idx]), 95))

    # ── amplitude plausibility ───────────────────────────────────────────
    if np.abs(x).max() > MAX_ABS_AMPLITUDE_MV:
        rep.acceptable = False
        rep.errors.append(f"peak amplitude {np.abs(x).max():.1f} mV exceeds the "
                          f"physiological limit of {MAX_ABS_AMPLITUDE_MV} mV "
                          f"(saturation or gain error)")
        return x.astype(np.float32), rep

    # ── high-frequency noise ─────────────────────────────────────────────
    nyq = fs / 2.0
    noisy = []
    if nyq > 45:
        bh, ah = sps.butter(3, min(45.0 / nyq, 0.99), btype="high")
        for i in range(12):
            if lead_names[i] in flat:
                continue
            hf = sps.filtfilt(bh, ah, x[:, i])
            total = float(np.var(x[:, i]))
            if total > 0 and float(np.var(hf)) / total > 0.55:
                noisy.append(lead_names[i])
    rep.noisy_leads = noisy
    if len(noisy) >= 8:
        rep.acceptable = False
        rep.errors.append(f"{len(noisy)} of 12 leads are dominated by high-frequency "
                          f"noise — this does not look like a cardiac signal")
        return x.astype(np.float32), rep
    if noisy:
        rep.warnings.append(f"noisy lead(s): {', '.join(noisy)}")

    # ── rhythm plausibility ──────────────────────────────────────────────
    lead_ii = x[:, 1] if lead_names[1] not in flat else x[:, int(np.argmax(lead_std))]
    peaks = detect_r_peaks(lead_ii, fs)
    rep.n_beats = int(len(peaks))
    if len(peaks) >= 2:
        rr = np.diff(peaks) / float(fs)
        rr = rr[rr > 0]
        if len(rr):
            rep.heart_rate_bpm = float(60.0 / np.median(rr))
    if rep.n_beats < 3:
        rep.acceptable = False
        rep.errors.append(f"only {rep.n_beats} QRS complexes detected in "
                          f"{rep.duration_s:.0f}s — no interpretable rhythm")
        return x.astype(np.float32), rep
    if rep.heart_rate_bpm is not None and not (MIN_HR_BPM <= rep.heart_rate_bpm <= MAX_HR_BPM):
        rep.warnings.append(f"implausible heart rate ({rep.heart_rate_bpm:.0f} bpm) — "
                            f"QRS detection may have failed")

    # ── electrode placement ──────────────────────────────────────────────
    # Runs last so it never interferes with the checks above. It records a
    # suspicion and a warning; it does not change `acceptable` or `sqi`, so
    # existing behaviour is unchanged for every record it does not flag.
    try:
        from .electrodes import detect as _detect_electrodes
        er = _detect_electrodes(x)
        if er.suspected:
            rep.electrode_suspect = True
            rep.electrode_reversal = er.reversal
            rep.electrode_confidence = er.confidence
            rep.electrode_reasons = er.reasons
            rep.warnings.append(
                f"possible {er.reversal} limb-electrode reversal "
                f"(confidence {er.confidence:.2f}) — verify electrode placement; "
                f"the statistical guarantees do not apply to a mis-acquired record")
    except Exception:
        pass          # detection is advisory; never let it break the gate

    # ── scope: is this rhythm inside the model's label space? ────────────
    try:
        from . import scope as _scope
        sr = _scope.assess(peaks, fs, _SCOPE_THRESHOLD)
        if not sr.in_scope:
            rep.out_of_scope = True
            rep.scope_reason = sr.reason
            rep.scope_confidence = sr.confidence
            rep.scope_detail = sr.detail
            rep.warnings.append(
                "rhythm appears outside the model's five diagnostic classes "
                "(irregularly irregular R-R) — statistical guarantees withheld")
    except Exception:
        pass

    # ── composite SQI ────────────────────────────────────────────────────
    penalties = (0.20 * len(flat) + 0.08 * len(noisy)
                 + (0.15 if rep.gain_correction != 1.0 else 0.0)
                 + (0.10 if rep.heart_rate_bpm is None else 0.0))
    rep.sqi = float(np.clip(1.0 - penalties, 0.0, 1.0))
    if rep.sqi < 0.5:
        rep.warnings.append(f"low overall signal quality (SQI {rep.sqi:.2f}); "
                            f"treat the interpretation with caution")

    return x.astype(np.float32), rep
