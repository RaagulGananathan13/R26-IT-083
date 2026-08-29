"""
Extract an ED triage record from a PDF.

WHY THIS IS DELICATE
--------------------
Component 04 uses missingness-aware encoding: an untested biomarker is the
clinical fact that nobody ordered the test, not a number to impute. That makes
a silent extraction failure worse than a loud one. If a troponin is present in
the document and the parser misses it, the model is told "not ordered" and
returns a materially different answer -- with no error anywhere.

So this extractor is built to be *audited*, not trusted:

  * every field it fills records the exact source text it came from;
  * every field it could not fill is listed explicitly, separated into
    "absent from the document" and "present but unparseable";
  * the caller is handed both, and the UI shows them side by side before the
    prediction is acted on.

It is a regex-and-lexicon parser over text-layer PDFs, not a document AI. It
handles the structured ED summary format the sample documents use. A scanned
image, a handwritten note, or a different hospital's template will extract
little or nothing -- and will say so rather than guessing.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from cvxai.core.errors import InvalidInput

# --------------------------------------------------------------------------
# Field patterns. Each is (canonical_name, regex, converter).
# Written against the sample template but deliberately loose about whitespace,
# separators and unit suffixes, because real exports vary in all three.
# --------------------------------------------------------------------------

_NUM = r"(-?\d+(?:\.\d+)?)"


def _f(value: str) -> float:
    return float(value)


def _i(value: str) -> int:
    return int(round(float(value)))


VITAL_PATTERNS: List[Tuple[str, str, Any]] = [
    # Both the labelled form ("Age: 61") and the prose form ("68 year old",
    # "68-year-old gentleman", "aged 71"), which a label-only pattern misses.
    ("age", r"\bages?d?\b[^\d\n]{0,12}" + _NUM + r"|" + _NUM
            + r"[\s-]*(?:years?|yrs?|y)[\s-]*old", _f),
    ("heartrate", r"(?:heart\s*rate|pulse|\bHR\b)[^\d\n]{0,12}" + _NUM, _f),
    ("resprate", r"(?:resp(?:iratory)?\s*rate|\bRR\b)[^\d\n]{0,12}" + _NUM, _f),
    ("o2sat", r"(?:o2\s*sat|oxygen\s*sat(?:uration)?|spo2)[^\d\n]{0,12}" + _NUM, _f),
    ("temperature", r"(?:temp(?:erature)?)[^\d\n]{0,12}" + _NUM, _f),
    ("pain", r"(?:pain(?:\s*score)?)[^\d\n]{0,12}" + _NUM, _f),
    ("acuity", r"(?:acuity|esi(?:\s*level)?|triage\s*level)[^\d\n]{0,12}" + _NUM, _f),
    ("bnp", r"(?:\bBNP\b|nt-?probnp)[^\d\n]{0,16}" + _NUM, _f),
    ("prior_ed_visits", r"(?:prior\s*ed\s*visits?|previous\s*ed\s*visits?)[^\d\n]{0,12}"
                        + _NUM, _i),
]

#: Blood pressure is written as a pair far more often than as two fields.
BP_PATTERN = r"(?:blood\s*pressure|\bBP\b)[^\d\n]{0,12}(\d{2,3})\s*/\s*(\d{2,3})"

SEX_PATTERN = (r"\b(?:sex|gender)\b[^A-Za-z\n]{0,12}"
               r"(male|female|man|woman|\bM\b|\bF\b)")

#: A line that is a section heading rather than prose. PDF text layers lose
#: styling, so a heading is recognised by shape: short, capitalised, and
#: without sentence punctuation. Used to bound free-text capture, so that a
#: chief complaint stops before the vitals block instead of swallowing it.
HEADING_LINE = (r"[ \t]*[A-Z][A-Za-z()]*(?:[ \t/&+-]+[A-Za-z()0-9]+){0,4}"
                r"[ \t]*:?[ \t]*$")

#: Each troponin result sits on its own line: "Troponin I: 1.2 ng/mL at 0.8 h".
#: The value is NOT at the start of the line, which an anchored pattern misses.
TROPONIN_LINE = r"(?im)^.*(?:troponin|hs-?c?tn[ti]?).*$"
TROPONIN_VALUE = _NUM + r"\s*(?:ng/(?:ml|l)|ug/l|pg/ml)?"
TROPONIN_TIME = r"(?:@|\bat\b|\bafter\b|t\s*=)\s*" + _NUM + r"\s*(?:h|hr|hours?)\b"

#: Negation cues. A finding inside the scope of one of these is an assertion of
#: ABSENCE. Without this, "No ST elevation" sets st_elevation to true and the
#: model is handed a diagnostic ECG the report explicitly ruled out -- a silent,
#: clinically material inversion.
#:
#: Both directions are needed. English puts the cue before the finding ("no ST
#: elevation") or after it ("ST depression is absent", "Q waves not present"),
#: and a backward-only check silently inverts every postfix form.
NEGATION_CUES_BEFORE = (
    r"\bno\b", r"\bnot\b", r"\bdenies\b", r"\bdenied\b", r"\bwithout\b",
    r"\bnegative\s+for\b", r"\bfree\s+of\b", r"\bnil\b",
    r"\bno\s+evidence\s+of\b", r"\bunremarkable\s+for\b", r"\babsence\s+of\b",
)
NEGATION_CUES_AFTER = (
    r"^\W*(?:is|are|was|were)?\s*(?:not\s+(?:present|seen|noted|identified)|absent)\b",
    r"^\W*(?:is|are|was|were)?\s*negative\b",
    r"^\W*(?:has|have)?\s*(?:been\s+)?(?:ruled\s+out|excluded)\b",
    r"^\W*:\s*(?:no|none|negative|absent)\b",
)
NEGATION_WINDOW = 48
#: Postfix cues bind tightly to the finding, so the forward window is short --
#: long enough for "is not present", short enough that the next clause's verb
#: cannot reach back.
NEGATION_WINDOW_AFTER = 34

#: Most serial troponin protocols draw twice (ESC 0/1 h) and rarely more than
#: four times. A document offering many more "troponin" numbers than that is not
#: reporting a result series -- it is a table, a paper, or a price list that
#: happens to contain the word. Uploading a research write-up produced 25
#: readings and a confident STEMI, which is the failure this bound prevents.
MAX_TROPONIN_DRAWS = 6

#: Plausible range in ng/mL, after unit conversion. Normal is below ~0.04; a
#: large infarct peaks in the tens. Anything above this is a number that landed
#: near the word "troponin", not a measurement.
MAX_TROPONIN_NG_ML = 100.0

#: Troponin assay units. MIMIC-IV records troponin I in ng/mL; European
#: high-sensitivity assays report ng/L, which is 1000x smaller. Reading 45 ng/L
#: as 45 ng/mL turns a mild elevation into an implausible one and would drive
#: the prediction hard. Conversion is applied AND announced.
TROPONIN_UNIT_SCALE = {
    "ng/ml": 1.0,
    "ug/l": 1.0,          # microgram/L is numerically identical to ng/mL
    "ng/l": 0.001,
    "pg/ml": 0.001,
}

#: ECG report findings. Matched against the ECG section only, so that a
#: chief complaint mentioning "no chest pain" cannot set an ECG flag.
ECG_FINDINGS: List[Tuple[str, str]] = [
    ("st_elevation", r"\bST[\s-]*elevat"),
    ("st_depression", r"\bST[\s-]*depress"),
    ("t_inversion", r"\bT[\s-]*wave\s*inver|\binverted\s*T"),
    ("q_wave", r"\bpathologic(?:al)?\s*Q[\s-]*wave|\bQ[\s-]*waves?\b"),
    ("lbbb", r"\bLBBB\b|left\s*bundle\s*branch"),
    ("rbbb", r"\bRBBB\b|right\s*bundle\s*branch"),
    ("acute", r"\bacute\b"),
    ("normal", r"\bnormal\s*(?:ecg|ekg|sinus\s*rhythm)"),
    ("critical_alert", r"\bcritical\b"),
    ("stemi_alert", r"\bSTEMI\b"),
    ("acute_mi", r"\bacute\s*(?:MI|myocardial\s*infarct)"),
    ("infarct_any", r"\binfarct"),
    ("infarct_anterior", r"\banterior\b"),
    ("infarct_inferior", r"\binferior\b"),
    ("infarct_lateral", r"\blateral\b"),
    ("age_undetermined", r"\bage\s*undetermined|\bold\s*infarct"),
]

ECG_NUMERIC: List[Tuple[str, str]] = [
    ("qrs_duration", r"\bQRS\b[^\d\n]{0,16}" + _NUM),
    ("pr_interval", r"\bPR\b[^\d\n]{0,16}" + _NUM),
    ("qt_interval", r"\bQTc?\b[^\d\n]{0,16}" + _NUM),
    ("qrs_axis", r"\baxis\b[^\d\n-]{0,16}(-?\d+)"),
]

MEDICATION_LEXICON = [
    "aspirin", "clopidogrel", "ticagrelor", "prasugrel", "warfarin", "apixaban",
    "rivaroxaban", "atorvastatin", "simvastatin", "rosuvastatin", "metoprolol",
    "bisoprolol", "carvedilol", "atenolol", "lisinopril", "ramipril", "enalapril",
    "losartan", "valsartan", "amlodipine", "diltiazem", "verapamil", "furosemide",
    "isosorbide", "nitroglycerin", "glyceryl trinitrate", "metformin", "insulin",
    "digoxin", "amiodarone", "spironolactone",
]

HISTORY_FLAGS: List[Tuple[str, str]] = [
    ("diabetes", r"\bdiabet|\bDM\b|\bT2DM\b"),
    ("prior_mi", r"prior\s*(?:MI|myocardial\s*infarct)|previous\s*(?:MI|infarct)"),
    ("prior_chf", r"\bCHF\b|heart\s*failure|\bHFrEF\b|\bHFpEF\b"),
    ("renal_disease", r"\bCKD\b|renal\s*(?:impair|failure|disease)|dialysis"),
    ("prior_acs", r"prior\s*ACS|previous\s*ACS|prior\s*(?:NSTEMI|STEMI|unstable\s*angina)"),
]


@dataclass
class ExtractedField:
    """One parsed value and the text it came from."""

    name: str
    value: Any
    source_text: str
    confidence: str = "parsed"       # parsed | inferred


@dataclass
class ExtractionResult:
    fields: Dict[str, Any] = field(default_factory=dict)
    evidence: List[ExtractedField] = field(default_factory=list)
    not_found: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    page_count: int = 0
    characters: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fields": self.fields,
            "evidence": [
                {"field": item.name, "value": item.value,
                 "source_text": item.source_text, "confidence": item.confidence}
                for item in self.evidence
            ],
            "not_found": self.not_found,
            "warnings": self.warnings,
            "document": {"pages": self.page_count, "characters": self.characters},
        }


# --------------------------------------------------------------------------
def read_pdf_text(data: bytes) -> Tuple[str, int]:
    """Return (text, page_count). Raises InvalidInput on an unreadable file."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:                 # pragma: no cover
        raise InvalidInput(
            "PDF support requires the `pypdf` package: python -m pip install pypdf"
        ) from exc

    if not data:
        raise InvalidInput("The uploaded PDF is empty.")
    if not data.lstrip()[:5].startswith(b"%PDF"):
        raise InvalidInput(
            "That file is not a PDF (no %PDF header). Export the ED record as a "
            "PDF with a text layer, or use the form instead.")

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:                   # noqa: BLE001
        raise InvalidInput("Could not read the PDF: %s" % exc) from exc

    text = "\n".join(pages)
    if len(text.strip()) < 40:
        raise InvalidInput(
            "This PDF has no extractable text layer -- it is most likely a scan or "
            "a photograph. Optical character recognition is out of scope here; "
            "enter the record on the form instead.",
            {"pages": len(pages), "characters": len(text.strip())})
    return text, len(pages)


def _section(text: str, *headings: str) -> str:
    """Text under the first matching heading, up to the next heading.

    Scoping matters: `acute` inside the ECG section is an ECG finding, while
    `acute` in a history or complaint line is not.
    """
    lines = text.splitlines()
    for heading in headings:
        opener = re.compile(r"^[ \t]*" + heading + r"[ \t]*[:\-]?[ \t]*(.*)$",
                            re.IGNORECASE)
        for index, line in enumerate(lines):
            match = opener.match(line)
            if not match:
                continue
            collected = [match.group(1)] if match.group(1).strip() else []
            for following in lines[index + 1:]:
                if _is_heading(following):
                    break
                collected.append(following)
            body = "\n".join(collected).strip()
            if body:
                return body
    return ""


def _is_heading(line: str) -> bool:
    """Does this line look like a section heading rather than prose?"""
    stripped = line.strip()
    if not stripped or len(stripped) > 44:
        return False
    if stripped.endswith((".", ",", ";")):
        return False
    # "Heart rate: 108 bpm" is a field, not a heading.
    if re.match(r"^[^:]{1,30}:\s*\S", stripped):
        return False
    return bool(re.match(r"^" + HEADING_LINE, stripped))


def _is_negated(text: str, start: int, end: Optional[int] = None) -> bool:
    """Is the finding at `start` inside the scope of a negation cue?

    Checks both directions. Backwards is bounded by the clause start, so a
    negation in the previous sentence cannot reach across. Forwards is bounded
    by a short window and by the clause end.

    "Infarct ... cannot be excluded" is deliberately NOT treated as negated:
    clinically it asserts possibility rather than absence.
    """
    before = text[max(0, start - NEGATION_WINDOW):start]
    for boundary in (".", ";", "\n"):
        position = before.rfind(boundary)
        if position != -1:
            before = before[position + 1:]
    if any(re.search(cue, before, re.IGNORECASE) for cue in NEGATION_CUES_BEFORE):
        return True

    if end is None:
        return False
    # The finding patterns match word stems ("ST depress" out of "ST
    # depression"), so `end` can land mid-word. Advance to the word boundary
    # first, or the forward cue is compared against "ion is absent".
    while end < len(text) and (text[end].isalnum() or text[end] == "-"):
        end += 1
    after = text[end:end + NEGATION_WINDOW_AFTER]
    for boundary in (".", ";", ",", "\n"):
        position = after.find(boundary)
        if position != -1:
            after = after[:position]
    return any(re.search(cue, after, re.IGNORECASE) for cue in NEGATION_CUES_AFTER)


def _snippet(text: str, start: int, end: int, pad: int = 28) -> str:
    fragment = text[max(0, start - pad): min(len(text), end + pad)]
    return " ".join(fragment.split())


def extract_triage_record(data: bytes) -> ExtractionResult:
    """Parse an ED triage PDF into the fields Component 04 consumes."""
    text, page_count = read_pdf_text(data)
    result = ExtractionResult(page_count=page_count, characters=len(text))
    fields = result.fields

    # ---- scalar vitals ------------------------------------------------
    for name, pattern, convert in VITAL_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            result.not_found.append(name)
            continue
        try:
            # An alternation puts the number in whichever branch matched.
            captured = next((group for group in match.groups() if group), None)
            fields[name] = convert(captured)
        except (TypeError, ValueError, StopIteration):
            result.warnings.append(
                "%s matched %r but could not be read as a number" % (name, match.group(0)))
            result.not_found.append(name)
            continue
        result.evidence.append(ExtractedField(
            name, fields[name], _snippet(text, match.start(), match.end())))

    # ---- blood pressure ------------------------------------------------
    match = re.search(BP_PATTERN, text, re.IGNORECASE)
    if match:
        fields["sbp"], fields["dbp"] = float(match.group(1)), float(match.group(2))
        result.evidence.append(ExtractedField(
            "sbp/dbp", "%s/%s" % (match.group(1), match.group(2)),
            _snippet(text, match.start(), match.end())))
    else:
        result.not_found.extend(["sbp", "dbp"])

    # ---- sex ------------------------------------------------------------
    match = re.search(SEX_PATTERN, text, re.IGNORECASE)
    if match:
        token = match.group(1).strip().lower()
        fields["sex"] = "F" if token.startswith(("f", "wom")) else "M"
        result.evidence.append(ExtractedField(
            "sex", fields["sex"], _snippet(text, match.start(), match.end())))
    else:
        result.not_found.append("sex")

    # ---- chief complaint -------------------------------------------------
    complaint_text = _section(
        text, r"chief\s*complaint", r"presenting\s*complaint",
        r"reason\s*for\s*(?:visit|attendance)", r"complaint")
    if complaint_text:
        # In prose exports there is no heading line to stop at, so the section
        # runs on into the next field. Cut at the first inline field label
        # ("Obs:", "Vitals", "HR 96") so vitals text does not end up in the
        # free-text channel that carries 31 % of the model's attribution.
        complaint_text = re.split(
            r"(?i)\b(?:obs|observations|vitals?|triage\s*vitals|examination|"
            r"ecg|troponin|medications?)\b\s*[:\-]?",
            complaint_text, maxsplit=1)[0]
        complaint = " ".join(complaint_text.split())[:400]
        fields["chief_complaint"] = complaint
        result.evidence.append(ExtractedField(
            "chief_complaint", complaint, complaint))
    else:
        result.not_found.append("chief_complaint")
        result.warnings.append(
            "No chief complaint found. At the triage horizon free text carries "
            "31.3 % of the model's attribution, so this materially weakens the "
            "prediction.")

    # ---- troponin --------------------------------------------------------
    values, hours, evidence_text, troponin_warnings = _extract_troponin(text)
    result.warnings.extend(troponin_warnings)
    if values:
        fields["troponin"] = values
        if hours:
            fields["troponin_hours"] = hours
        result.evidence.append(ExtractedField(
            "troponin", values, evidence_text))
    else:
        result.not_found.append("troponin")

    # ---- ECG -------------------------------------------------------------
    ecg_text = _section(text, r"ECG(?:\s*report)?", r"EKG(?:\s*report)?",
                        r"12[\s-]*lead\s*ECG", r"electrocardiogram")
    if ecg_text:
        ecg: Dict[str, Any] = {}
        negated: List[str] = []
        for name, pattern in ECG_FINDINGS:
            found = re.search(pattern, ecg_text, re.IGNORECASE)
            if not found:
                continue
            if _is_negated(ecg_text, found.start(), found.end()):
                negated.append(name)
                continue
            ecg[name] = True
        if negated:
            # Reported, not silently dropped: a reader must be able to see that
            # the document mentioned the finding and ruled it out.
            result.warnings.append(
                "ECG findings stated as ABSENT and therefore not set: %s."
                % ", ".join(sorted(negated)))
        for name, pattern in ECG_NUMERIC:
            found = re.search(pattern, ecg_text, re.IGNORECASE)
            if found:
                try:
                    ecg[name] = float(found.group(1))
                except (TypeError, ValueError):
                    pass
        timing = re.search(r"(?:acquired|obtained|taken|performed)[^\d\n]{0,20}" + _NUM
                           + r"\s*(?:h|hr|hours?|min(?:ute)?s?)", ecg_text, re.IGNORECASE)
        if timing:
            value = float(timing.group(1))
            if re.search(r"min", timing.group(0), re.IGNORECASE):
                value /= 60.0
            ecg["hours_after_arrival"] = round(value, 3)
        if ecg:
            fields["ecg"] = ecg
            result.evidence.append(ExtractedField(
                "ecg", sorted(k for k, v in ecg.items() if v is True),
                " ".join(ecg_text.split())[:220]))
        else:
            result.warnings.append(
                "An ECG section was found but no recognised finding was parsed from it.")
    else:
        result.not_found.append("ecg")

    # ---- medications -----------------------------------------------------
    medication_text = _section(text, r"(?:home\s*)?medications?", r"drug\s*history",
                               r"current\s*medications?") or text
    medications = sorted({name for name in MEDICATION_LEXICON
                          if re.search(r"\b" + re.escape(name), medication_text,
                                       re.IGNORECASE)})
    if medications:
        fields["home_medications"] = medications
        result.evidence.append(ExtractedField(
            "home_medications", medications, ", ".join(medications)))
    else:
        result.not_found.append("home_medications")

    # ---- history flags ---------------------------------------------------
    history_text = _section(text, r"(?:past\s*)?medical\s*history", r"history",
                            r"comorbidit(?:y|ies)", r"background") or text
    for name, pattern in HISTORY_FLAGS:
        found = re.search(pattern, history_text, re.IGNORECASE)
        if found:
            fields[name] = 1
            result.evidence.append(ExtractedField(
                name, 1, _snippet(history_text, found.start(), found.end())))

    # ---- leakage guard ---------------------------------------------------
    # A Charlson index computed from the INDEX admission is leakage channel L1,
    # which alone moves AUROC 0.9665 -> 0.9889. It is never taken from a
    # document, because a PDF cannot say whether it predates this visit.
    if re.search(r"charlson", text, re.IGNORECASE):
        result.warnings.append(
            "The document mentions a Charlson comorbidity index. It is deliberately "
            "NOT extracted: computed from the index admission it is leakage channel "
            "L1, and a document cannot establish that it predates this visit.")

    return result


#: One reading: a number, an optional unit, and an optional draw time. Applied
#: repeatedly across a line, because prose puts several on one line --
#: "Serial troponin: 0.9 at 1 hour, then 4.2 at 4 hours". Parsing only the
#: first silently discards the rise, which is the entire infarct signal.
TROPONIN_READING = (
    _NUM + r"\s*(ng\s*/\s*m?[lL]|ug\s*/\s*[lL]|pg\s*/\s*m[lL])?"
    r"(?:\s*(?:@|\bat\b|\bafter\b|t\s*=)\s*" + _NUM + r"\s*(?:h\b|hr\b|hours?\b|min))?"
)


def _extract_troponin(text: str) -> Tuple[List[float], List[float], str, List[str]]:
    """Serial troponin values with their draw times, if documented.

    Returns (values_ng_per_ml, hours, evidence, warnings).
    """
    lines = [line for line in re.findall(TROPONIN_LINE, text) if line.strip()]
    if not lines:
        return [], [], "", []

    readings: List[Tuple[float, Optional[float]]] = []
    warnings: List[str] = []
    converted_units: set = set()

    for line in lines:
        position = line.lower().find("trop")
        if position == -1:
            position = 0
        if _is_negated(line, position, position + 4):
            continue
        # Strip the assay name first: "Troponin I" and "hs-cTnT" both carry a
        # trailing letter that must not be read as part of a number.
        body = re.sub(r"(?i)(troponin\s*[ITt]?\b|hs-?c?tn[ti]?\b)", " ", line)

        for match in re.finditer(TROPONIN_READING, body, re.IGNORECASE):
            raw_value = float(match.group(1))
            unit = (match.group(2) or "").replace(" ", "").lower()
            hour = float(match.group(3)) if match.group(3) else None
            if hour is not None and re.search(r"min\b", match.group(0), re.IGNORECASE):
                hour /= 60.0

            scale = TROPONIN_UNIT_SCALE.get(unit)
            if unit and scale is None:
                warnings.append(
                    "Troponin unit %r is not recognised; the value was read as ng/mL. "
                    "Check it against the source." % unit)
                scale = 1.0
            elif scale is None:
                scale = 1.0
            if scale != 1.0:
                converted_units.add(unit)
            readings.append((raw_value * scale, hour))

    if not readings:
        return [], [], "", warnings

    # -- plausibility, before anything downstream sees these ---------------
    implausible = [value for value, _ in readings if value > MAX_TROPONIN_NG_ML]
    if implausible:
        readings = [item for item in readings if item[0] <= MAX_TROPONIN_NG_ML]
        warnings.append(
            "Discarded %d value(s) above %.0f ng/mL (%s) as implausible for a "
            "troponin assay. A number that large next to the word 'troponin' is "
            "far more likely to be a table entry than a measurement."
            % (len(implausible), MAX_TROPONIN_NG_ML,
               ", ".join("%.3g" % value for value in implausible[:5])))

    if len(readings) > MAX_TROPONIN_DRAWS:
        # Refusing the whole series, not trimming it. If the document is not
        # reporting a result series then no subset of these numbers is one
        # either, and a plausible-looking pair salvaged from a table would be
        # worse than nothing -- the model would treat it as a real biomarker.
        warnings.append(
            "Found %d apparent troponin readings, more than the %d a serial "
            "protocol produces. This document does not appear to report a "
            "troponin series, so none were used and the biomarker is recorded "
            "as not ordered. Enter it on the form if the record does have one."
            % (len(readings), MAX_TROPONIN_DRAWS))
        return [], [], "", warnings

    if not readings:
        return [], [], "", warnings

    if converted_units:
        warnings.append(
            "Troponin reported in %s was converted to ng/mL (x0.001), the unit this "
            "model was trained on. A high-sensitivity assay in ng/L read as ng/mL "
            "would be a thousand-fold overstatement, so the conversion is applied and "
            "announced rather than assumed."
            % ", ".join(sorted(converted_units)))

    evidence = " | ".join(" ".join(line.split()) for line in lines)[:240]

    # Order by draw time where every reading has one; otherwise keep document
    # order, which is the order the assays were reported.
    if all(hour is not None for _, hour in readings):
        readings.sort(key=lambda item: item[1])
        return ([value for value, _ in readings],
                [hour for _, hour in readings], evidence, warnings)

    # Values without documented timings: keep the values, omit the hours, so
    # the featuriser sees real measurements with unknown timing rather than an
    # invented schedule.
    return [value for value, _ in readings], [], evidence, warnings
