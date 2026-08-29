"""
verify.py — Finding-preservation verifier (the safety gate for CONTRIBUTION 2).

Fixes audit finding C-3. The archive claimed Tier 3 "cannot hallucinate because
it never sees the raw ECG". That claim confuses two different things: not seeing
the signal prevents *inventing evidence*, it does not prevent a seq2seq model
from adding, dropping or negating a finding while rephrasing. Measured on the
shipped 1711-record dump, Tier 3:
   * dropped a clinical concept present in Tier 2 in 103 records
   * inserted "atrial fibrillation" — a class the model cannot produce — in 42

The only defence that actually holds is to CHECK the output text against the
structured findings, and refuse it if they disagree. That is what this does.

It is the gate, not the generator: any future natural-language layer (a rule
realiser, a local LLM, a fine-tuned paraphraser) must pass through
`verify_paraphrase()` or it does not ship. If verification fails, the
deterministic template text is emitted instead — degraded fluency, never
degraded safety.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Sequence

from .conformal import RULE_IN, REFER
from .report import CLASS_FULL, ClinicalReport

# Surface forms that assert each class. Used for both directions of the check.
CLASS_TERMS: Dict[str, List[str]] = {
    "NORM": ["normal ecg", "normal study", "within normal limits", "no abnormality"],
    "MI": ["myocardial infarction", "infarct", "infarction"],
    "STTC": ["st/t", "st-t", "st segment", "st-segment", "t wave", "t-wave", "repolaris",
             "repolariz", "ischaem", "ischem"],
    "CD": ["conduction disturbance", "conduction delay", "bundle branch",
           "fascicular block", "intraventricular conduction"],
    "HYP": ["hypertrophy", "voltage criteria"],
}

# Diagnoses this system CANNOT produce. Their appearance is definitionally a
# hallucination — the model has no output unit for them.
FORBIDDEN_TERMS: List[str] = [
    "atrial fibrillation", "atrial flutter", "afib", "a-fib",
    "ventricular tachycardia", "ventricular fibrillation", "supraventricular tachycardia",
    "pacemaker", "paced rhythm", "wolff", "brugada", "long qt", "torsade",
    "stemi", "nstemi", "cardiac arrest", "asystole", "heart block degree",
    "pericarditis", "myocarditis", "pulmonary embolism", "hyperkalaemia", "hyperkalemia",
]

# Language that overclaims certainty for a machine-produced interpretation.
OVERCLAIM_TERMS: List[str] = [
    "definitely", "certainly", "confirmed diagnosis", "rule out entirely",
    "no further review", "no clinician review", "guaranteed diagnosis",
]

NEGATORS = ("no ", "not ", "without", "absence of", "ruled out", "excluded",
            "denies", "free of", "negative for", "cannot be", "does not",
            "are not detected", "recognises only", "such as")


@dataclass
class VerificationResult:
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    asserted: List[str] = field(default_factory=list)
    expected: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _clause(text: str, pos: int) -> str:
    """The enclosing clause of a term: bounded by line breaks and sentence marks."""
    start = max(text.rfind("\n", 0, pos), text.rfind(". ", 0, pos),
                text.rfind("- ", 0, pos), 0)
    end = min([e for e in (text.find("\n", pos), text.find(". ", pos))
               if e != -1] or [len(text)])
    return text[start:end + 1]


def _negated(text: str, pos: int) -> bool:
    """True if a negator PRECEDES the term inside its own clause.

    Scope matters in both directions. A fixed character window is too short —
    in 'RULED OUT: normal ECG, ST/T change, conduction disturbance.' the negator
    is far back, and the finding was read as asserted. But a whole-clause scan is
    too coarse — 'The ECG meets criteria for a normal study; no abnormality
    crossed its threshold.' contains 'no ' AFTER the term, and NORM was then read
    as negated. Only preceding text inside the clause counts.
    """
    start = max(text.rfind("\n", 0, pos), text.rfind(". ", 0, pos),
                text.rfind("- ", 0, pos), 0)
    return any(n in text[start:pos] for n in NEGATORS)


# Contexts in which naming an out-of-scope diagnosis is the OPPOSITE of
# hallucinating it: the report is declaring that it cannot assess the condition.
# Without these, the scope warning added in src/scope.py would be rejected as a
# hallucination — the verifier flagging the system for admitting its own limits.
SCOPE_DISCLAIMERS = (
    "not detected", "recognises only", "outside the five", "outside the model",
    "guarantees withheld", "cannot assess", "not assessed", "no output",
    "was never in the label space", "must not be read as excluding",
    "rhythm scope", "requires assessment by a clinician",
)


def _hallucinated_terms(text: str) -> List[str]:
    """Forbidden diagnoses that are ASSERTED (not merely named as unassessable)."""
    low = text.lower()
    hits = []
    for term in FORBIDDEN_TERMS:
        for m in re.finditer(re.escape(term), low):
            if _negated(low, m.start()):
                continue
            clause = _clause(low, m.start())
            if any(d in clause for d in SCOPE_DISCLAIMERS):
                continue
            hits.append(term)
            break
    return hits


def asserted_classes(text: str) -> List[str]:
    """Which classes does this text positively assert (ignoring negated mentions)?"""
    low = text.lower()
    found = []
    for cls, terms in CLASS_TERMS.items():
        for t in terms:
            for m in re.finditer(re.escape(t), low):
                if not _negated(low, m.start()):
                    found.append(cls)
                    break
            if cls in found:
                break
    return found


def verify_report(report: ClinicalReport) -> VerificationResult:
    """Check a generated report against its own structured findings."""
    res = VerificationResult(passed=True)
    if report.refused:
        low = report.text.lower()
        for cls, terms in CLASS_TERMS.items():
            if cls == "NORM":
                continue
            for t in terms:
                if t in low:
                    res.passed = False
                    res.errors.append(
                        f"a refused report must not mention '{t}' ({cls})")
        return res

    expected_in = [f.cls for f in report.findings if f.zone == RULE_IN]
    expected_refer = [f.cls for f in report.findings if f.zone == REFER]
    res.expected = expected_in

    # ── 1. no invented diagnosis ─────────────────────────────────────────
    low = report.text.lower()
    for term in _hallucinated_terms(report.text):
        res.passed = False
        res.errors.append(
            f"HALLUCINATION: '{term}' is asserted but is not one of the 5 "
            f"classes this system can produce")

    # ── 2. contradiction ─────────────────────────────────────────────────
    if "NORM" in expected_in and any(c != "NORM" for c in expected_in):
        res.passed = False
        res.errors.append(
            "CONTRADICTION: the report asserts a normal ECG and an abnormality "
            "simultaneously")

    # ── 3. overclaiming ──────────────────────────────────────────────────
    for term in OVERCLAIM_TERMS:
        if term in low:
            res.warnings.append(f"overclaiming language: '{term}'")

    # ── 4. mandatory safety content ──────────────────────────────────────
    if "not a medical device" not in low:
        res.passed = False
        res.errors.append("MISSING DISCLAIMER: every report must carry it")
    if not report.triage:
        res.passed = False
        res.errors.append("MISSING TRIAGE tier")

    # ── 5. every ruled-in finding must actually appear ────────────────────
    said = asserted_classes(report.text)
    res.asserted = said
    for cls in expected_in:
        if cls not in said:
            res.passed = False
            res.errors.append(
                f"DROPPED FINDING: {CLASS_FULL[cls]} was ruled in but is not "
                f"asserted anywhere in the text")

    # ── 6. referred classes must be flagged as uncertain, not asserted ────
    for cls in expected_refer:
        if not re.search(r"(possible|cannot|could not|review is required|does not decide)",
                         low):
            res.warnings.append(
                f"{CLASS_FULL[cls]} is in the refer zone; the text should mark it "
                f"as unresolved")
    return res


def verify_paraphrase(paraphrase: str, source_report: ClinicalReport) -> VerificationResult:
    """Gate ANY natural-language rewriting of a verified report.

    Bidirectional containment: the paraphrase may not add a class the source did
    not assert, and may not drop one the source did. This is the check the
    archive's Tier 3 never had.
    """
    res = VerificationResult(passed=True)
    src = set(asserted_classes(source_report.text))
    new = set(asserted_classes(paraphrase))
    res.expected, res.asserted = sorted(src), sorted(new)

    for cls in new - src:
        res.passed = False
        res.errors.append(f"ADDED FINDING: paraphrase asserts {CLASS_FULL[cls]}, "
                          f"the verified report did not")
    for cls in src - new:
        res.passed = False
        res.errors.append(f"DROPPED FINDING: paraphrase omits {CLASS_FULL[cls]}")

    low = paraphrase.lower()
    src_halluc = set(_hallucinated_terms(source_report.text))
    for term in _hallucinated_terms(paraphrase):
        if term in src_halluc:
            continue                      # already present in the verified source
        res.passed = False
        res.errors.append(f"HALLUCINATION: paraphrase introduces '{term}'")

    if source_report.triage and source_report.triage.lower() not in low:
        res.warnings.append("paraphrase does not restate the triage tier")
    if "not a medical device" not in low:
        res.passed = False
        res.errors.append("paraphrase dropped the disclaimer")
    return res


def safe_paraphrase(paraphrase: str, source_report: ClinicalReport) -> tuple[str, VerificationResult]:
    """Return the paraphrase only if it verifies; otherwise the source text.

    Degraded fluency is always preferable to a degraded finding.
    """
    res = verify_paraphrase(paraphrase, source_report)
    return (paraphrase if res.passed else source_report.text), res


def batch_verify(reports: Sequence[ClinicalReport]) -> Dict:
    """Corpus-level safety statistics for the thesis evaluation chapter."""
    n = len(reports)
    results = [verify_report(r) for r in reports]
    failed = [r for r in results if not r.passed]
    err_types: Dict[str, int] = {}
    for r in failed:
        for e in r.errors:
            err_types[e.split(":")[0]] = err_types.get(e.split(":")[0], 0) + 1
    return dict(n=n, passed=n - len(failed), failed=len(failed),
                pass_rate=(n - len(failed)) / n if n else 0.0,
                error_types=err_types,
                warning_rate=sum(1 for r in results if r.warnings) / n if n else 0.0)
