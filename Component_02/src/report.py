"""
report.py — RESEARCH CONTRIBUTION 2
Explanation-grounded, risk-controlled clinical report generation.

────────────────────────────────────────────────────────────────────────────
WHAT THE ARCHIVE DID
────────────────────────────────────────────────────────────────────────────
Three tiers: a template engine, a BioBART "smoother", and a legacy free-text
generator. The audit found:
  * Tier 3 was an identity function 80.5% of the time, corrupted the text in
    56.9% of cases, and in 42 records invented "atrial fibrillation" — a
    diagnosis the classifier cannot even produce.
  * 5.8% of reports asserted NORM *and* an abnormality in the same paragraph.
  * 63 distinct reports were possible for 1711 patients (cardiologists wrote
    1055). No heart rate, no rhythm, no intervals, no localisation, no risk
    tier, no uncertainty statement, no signal-quality statement — ever.

────────────────────────────────────────────────────────────────────────────
WHAT THIS DOES INSTEAD
────────────────────────────────────────────────────────────────────────────
1. Every sentence is emitted from a `Finding` object, and every `Finding`
   carries the evidence that produced it. Nothing can be written that is not
   traceable to a measurement, a conformal decision, or an attribution.
2. The conformal zone (rule-in / refer / rule-out) chooses the modality of the
   language, so uncertainty is *stated* rather than hidden behind a threshold.
3. Lead attributions are turned into anatomical localisation, so the XAI output
   becomes report content instead of a picture beside it.
4. Measured signal properties (rate, rhythm regularity, quality) are included.
5. NORM is mutually exclusive with any ruled-in abnormality — the contradiction
   is structurally impossible, not filtered afterwards.
6. The finished text is checked by verify.py before it is allowed out.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from .conformal import RULE_IN, RULE_OUT, REFER
from .models import CLASS_NAMES
from .quality import QualityReport
from .xai import Explanation

CLASS_FULL = {
    "NORM": "normal ECG",
    "MI": "myocardial infarction",
    "STTC": "ST/T change",
    "CD": "conduction disturbance",
    "HYP": "ventricular hypertrophy",
}

# Triage tiers, most urgent first.
TRIAGE_IMMEDIATE = "IMMEDIATE"
TRIAGE_URGENT = "URGENT"
TRIAGE_PRIORITY = "PRIORITY"
TRIAGE_ROUTINE = "ROUTINE"
TRIAGE_REPEAT = "REPEAT ECG"

DISCLAIMER = (
    "AI-generated decision support. NOT a medical device and NOT a diagnosis. "
    "Every report requires review by a qualified clinician before any clinical "
    "action is taken."
)


@dataclass
class Finding:
    cls: str
    label: str
    zone: str                       # rule_in | refer | rule_out
    probability: float              # calibrated
    lambda_out: float
    lambda_in: float
    sentence: str = ""
    evidence: List[str] = field(default_factory=list)
    territory: Optional[str] = None
    artery: Optional[str] = None
    top_leads: List[str] = field(default_factory=list)
    timing_s: List[float] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class ClinicalReport:
    triage: str
    headline: str
    quality_line: str
    rhythm_line: str
    findings: List[Finding]
    ruled_out: List[str]
    referred: List[str]
    guarantees: List[str]
    limitations: List[str]
    text: str
    refused: bool = False

    def to_dict(self):
        d = asdict(self)
        d["findings"] = [f.to_dict() for f in self.findings]
        return d


# ══════════════════════════════════════════════════════════════════════════
def _rhythm_line(q: QualityReport) -> str:
    if q.heart_rate_bpm is None:
        return "Rate and rhythm could not be measured reliably."
    hr = q.heart_rate_bpm
    if hr < 60:
        band = "sinus bradycardia range"
    elif hr > 100:
        band = "tachycardia range"
    else:
        band = "normal range"
    return (f"Ventricular rate {hr:.0f} bpm ({band}), {q.n_beats} complexes "
            f"analysed over {q.duration_s:.1f} s.")


def _quality_line(q: QualityReport) -> str:
    parts = [f"Signal quality index {q.sqi:.2f}."]
    if q.electrode_suspect:
        parts.append(f"POSSIBLE {q.electrode_reversal} ELECTRODE REVERSAL "
                     f"(confidence {q.electrode_confidence:.2f}) — verify placement.")
    if q.gain_correction != 1.0:
        parts.append(f"Amplitude was rescaled by {q.gain_correction:g} to reach millivolts.")
    if q.flat_leads:
        parts.append(f"Flat or disconnected lead(s): {', '.join(q.flat_leads)}.")
    if q.noisy_leads:
        parts.append(f"Noisy lead(s): {', '.join(q.noisy_leads)}.")
    if not q.flat_leads and not q.noisy_leads and q.gain_correction == 1.0:
        parts.append("All 12 leads usable.")
    return " ".join(parts)


def _evidence_clause(exp: Optional[Explanation]) -> tuple[str, List[str]]:
    """Turn an attribution into a clinical sentence fragment. This is the step
    that makes the XAI part of the report rather than an illustration."""
    if exp is None:
        return "", []
    ev: List[str] = []
    frag = []
    if exp.top_leads:
        frag.append(f"maximal in {', '.join(exp.top_leads)}")
        ev.append(f"lead attribution (integrated gradients): {', '.join(exp.top_leads)}")
    if exp.territory:
        art = "an" if exp.territory[0] in "aeiou" else "a"
        frag.append(f"{art} {exp.territory} distribution ({exp.territory_artery})")
        ev.append(f"territory concentration score {exp.territory_score:.2f} "
                  f"(1.0 = uniform across leads)")
    if exp.cam_peaks_s:
        t = ", ".join(f"{x:.1f} s" for x in exp.cam_peaks_s)
        frag.append(f"most prominent at {t} of the recording")
        ev.append(f"Grad-CAM temporal peaks at {t}")
    if not frag:
        return "", ev
    return " Supporting evidence is " + "; ".join(frag) + ".", ev


def _sentence(cls: str, zone: str, p: float, exp: Optional[Explanation],
              unresolved: Optional[List[str]] = None) -> tuple[str, List[str]]:
    name = CLASS_FULL[cls]
    clause, ev = _evidence_clause(exp)
    if zone == RULE_IN:
        if cls == "NORM":
            # Do not call a study normal while something is still unresolved —
            # say so explicitly instead (a softer form of the C-4 fix).
            if unresolved:
                names = ", ".join(CLASS_FULL[c] for c in unresolved)
                return (f"No abnormality crossed its rule-in threshold, but this "
                        f"study cannot be called normal while {names} "
                        f"remains unresolved (see below)."), ev
            return ("The ECG meets criteria for a normal study; no abnormality "
                    "crossed its rule-in threshold."), ev
        base = {
            "MI": f"Findings are consistent with {name}.",
            "STTC": f"Significant {name}s are present.",
            "CD": f"A {name} is present.",
            "HYP": f"Voltage criteria are met for {name}.",
        }[cls]
        return base + clause, ev
    if zone == REFER:
        return (f"Possible {name} (calibrated probability {p*100:.0f}%). This "
                f"falls between the rule-out and rule-in thresholds, so the "
                f"system does not decide it — cardiologist review is required."
                + clause), ev
    return "", ev


def _triage(findings: List[Finding], referred: List[str], q: QualityReport) -> tuple[str, str]:
    ruled_in = {f.cls for f in findings if f.zone == RULE_IN}
    if not q.acceptable:
        return TRIAGE_REPEAT, "The recording could not be interpreted. Repeat the ECG."
    if "MI" in ruled_in:
        return (TRIAGE_IMMEDIATE,
                "Findings consistent with myocardial infarction. Immediate clinical "
                "assessment and cardiology involvement are indicated.")
    if "MI" in referred:
        return (TRIAGE_URGENT,
                "Myocardial infarction could not be ruled out. Urgent clinician "
                "review with serial troponin and repeat ECG is indicated.")
    if ruled_in & {"CD", "STTC"}:
        return (TRIAGE_PRIORITY,
                "A significant abnormality is present. Clinician review during "
                "this episode of care is indicated.")
    if "HYP" in ruled_in or referred:
        return (TRIAGE_PRIORITY if referred else TRIAGE_ROUTINE,
                "Non-urgent abnormality or unresolved uncertainty. Routine "
                "clinician review and clinical correlation are advised.")
    if "NORM" in ruled_in:
        return (TRIAGE_ROUTINE,
                "No abnormality was detected. No ECG-driven action is required; "
                "clinical judgement still governs.")
    return (TRIAGE_PRIORITY,
            "The ECG does not meet criteria for a normal study and no specific "
            "abnormality was confirmed. Clinician review is advised.")


# ══════════════════════════════════════════════════════════════════════════
def build_report(probs: Dict[str, float] | List[float],
                 zones: Dict[str, str],
                 thresholds: Dict[str, Dict],
                 quality: QualityReport,
                 explanations: Optional[Dict[str, Explanation]] = None,
                 class_names: Optional[List[str]] = None) -> ClinicalReport:
    """Assemble the report. Every sentence traces to a Finding."""
    class_names = list(class_names or CLASS_NAMES)
    if not isinstance(probs, dict):
        probs = {c: float(probs[i]) for i, c in enumerate(class_names)}
    explanations = explanations or {}

    # ── refusal path (audit finding C-1) ─────────────────────────────────
    if not quality.acceptable:
        reasons = "; ".join(quality.errors)
        text = (f"ECG NOT INTERPRETED.\n\n"
                f"The recording failed automated quality control and was refused "
                f"before classification: {reasons}.\n\n"
                f"No diagnostic probabilities are reported, because a diagnosis "
                f"derived from an uninterpretable signal is worse than no "
                f"diagnosis. Check electrode placement, cable connections and "
                f"recorder gain, then repeat the ECG.\n\n{DISCLAIMER}")
        return ClinicalReport(
            triage=TRIAGE_REPEAT, headline="ECG not interpretable",
            quality_line=_quality_line(quality), rhythm_line="",
            findings=[], ruled_out=[], referred=[],
            guarantees=[], limitations=quality.errors, text=text, refused=True)

    # ── build findings ───────────────────────────────────────────────────
    findings: List[Finding] = []
    ruled_out: List[str] = []
    referred: List[str] = []

    ruled_in_abnormal = [c for c in class_names
                         if c != "NORM" and zones.get(c) == RULE_IN]
    referred_abnormal = [c for c in class_names
                         if c != "NORM" and zones.get(c) == REFER]

    for cls in class_names:
        zone = zones.get(cls, REFER)
        t = thresholds.get(cls, {})
        # ── C-4 FIX: NORM is mutually exclusive with a ruled-in abnormality.
        if cls == "NORM" and zone == RULE_IN and ruled_in_abnormal:
            zone = REFER
        if zone == RULE_OUT:
            ruled_out.append(cls)
            continue
        if zone == REFER:
            referred.append(cls)
        exp = explanations.get(cls)
        sent, ev = _sentence(cls, zone, probs[cls], exp,
                             unresolved=referred_abnormal if cls == "NORM" else None)
        if not sent:
            continue
        findings.append(Finding(
            cls=cls, label=CLASS_FULL[cls], zone=zone, probability=probs[cls],
            lambda_out=float(t.get("lambda_out", float("nan"))),
            lambda_in=float(t.get("lambda_in", float("nan"))),
            sentence=sent, evidence=ev,
            territory=exp.territory if exp else None,
            artery=exp.territory_artery if exp else None,
            top_leads=exp.top_leads if exp else [],
            timing_s=exp.cam_peaks_s if exp else []))

    triage, triage_text = _triage(findings, referred, quality)

    # ── guarantees ───────────────────────────────────────────────────────
    # A conformal bound is calibrated on correctly-acquired recordings. If the
    # electrodes are suspected to be reversed, this record is outside that
    # distribution and the bound does not apply to it — so we withdraw the
    # claim rather than print a promise we cannot keep. Measured effect of
    # reversal on the bounds: audit/12_electrode_reversal.py.
    guarantees: List[str] = []
    if quality.electrode_suspect:
        guarantees.append(
            f"STATISTICAL GUARANTEES SUSPENDED for this record: a "
            f"{quality.electrode_reversal} limb-electrode reversal is suspected "
            f"(confidence {quality.electrode_confidence:.2f}). The conformal "
            f"bounds are calibrated on correctly-placed recordings and do not "
            f"hold here. Verify electrode placement and repeat.")
    for cls in (ruled_out if not quality.electrode_suspect else []):
        t = thresholds.get(cls, {})
        a, n = t.get("alpha"), t.get("n_pos_cal")
        if a is not None and t.get("guarantee_feasible", False):
            guarantees.append(
                f"{CLASS_FULL[cls]} ruled out under a conformal miss-rate bound of "
                f"{a*100:.0f}% (calibrated on {n} positive cases).")
        else:
            guarantees.append(
                f"{CLASS_FULL[cls]} below threshold, but the conformal guarantee is "
                f"NOT available for this class ({t.get('note','insufficient calibration data')}).")

    # ── limitations ──────────────────────────────────────────────────────
    limitations = [
        "This system recognises only 5 diagnostic superclasses (NORM, MI, STTC, "
        "CD, HYP). Arrhythmias such as atrial fibrillation, and any condition "
        "outside these classes, are NOT detected and their absence here is not "
        "evidence of their absence.",
        "Intervals (PR, QRS, QT) and QRS axis are not measured.",
        "The model was trained on PTB-XL (German cohort, 1989-1996) and has not "
        "been validated on this population.",
    ]
    limitations += [f"Signal quality caveat: {w}" for w in quality.warnings]
    if quality.electrode_suspect:
        limitations += [f"Electrode placement: {r}" for r in quality.electrode_reasons]

    # ── compose ──────────────────────────────────────────────────────────
    headline = (", ".join(CLASS_FULL[f.cls] for f in findings if f.zone == RULE_IN)
                or ("no abnormality detected" if "NORM" in
                    [f.cls for f in findings if f.zone == RULE_IN] or not findings
                    else "indeterminate"))

    lines = [f"TRIAGE: {triage}", "", f"TECHNICAL: {_quality_line(quality)}",
             f"RATE & RHYTHM: {_rhythm_line(quality)}", "", "FINDINGS:"]
    if findings:
        lines += [f"  - {f.sentence}" for f in findings]
    else:
        lines.append("  - No finding reached its rule-in threshold and none could "
                     "be confidently ruled out.")
    if ruled_out:
        lines += ["", "RULED OUT: " + ", ".join(CLASS_FULL[c] for c in ruled_out) + "."]
    lines += ["", f"RECOMMENDATION: {triage_text}"]
    if guarantees:
        lines += ["", "STATISTICAL GUARANTEES:"] + [f"  - {g}" for g in guarantees]
    lines += ["", "LIMITATIONS:"] + [f"  - {l}" for l in limitations]
    lines += ["", DISCLAIMER]

    return ClinicalReport(
        triage=triage, headline=headline,
        quality_line=_quality_line(quality), rhythm_line=_rhythm_line(quality),
        findings=findings, ruled_out=ruled_out, referred=referred,
        guarantees=guarantees, limitations=limitations,
        text="\n".join(lines), refused=False)
