"""
Request model for Component 04 (ED triage record).

Every field is optional by design. The component uses missingness-aware
encoding: an untested biomarker is the clinical fact that nobody ordered the
test, not a number to be imputed to a population average. Requiring a value
here would destroy that signal before the model ever sees it.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

# Direct import, not a forward reference: `common` imports nothing from this
# module, so there is no cycle to break, and a real class keeps the response
# model fully defined without a model_rebuild() call at import time.
from cvxai.schemas.common import Envelope


class ECGReport(BaseModel):
    """Findings from the ECG cart's text report.

    Component 04 consumes MIMIC-IV's ECG *report*, not the waveform. That is a
    stated limitation: ST elevation is recoverable in only ~41 % of STEMI
    cases, which caps STEMI F1 at 0.61.
    """

    st_elevation: bool = False
    st_depression: bool = False
    t_inversion: bool = False
    q_wave: bool = False
    lbbb: bool = False
    rbbb: bool = False
    acute: bool = False
    normal: bool = False
    critical_alert: bool = False
    stemi_alert: bool = False
    acute_mi: bool = False
    infarct_any: bool = False
    infarct_anterior: bool = False
    infarct_inferior: bool = False
    infarct_lateral: bool = False
    infarct_possible: bool = False
    age_undetermined: bool = False

    qrs_duration: Optional[float] = Field(None, description="ms")
    pr_interval: Optional[float] = Field(None, description="ms")
    qt_interval: Optional[float] = Field(None, description="ms")
    rr_interval: Optional[float] = Field(None, description="ms")
    qrs_axis: Optional[float] = Field(None, description="degrees")
    hours_after_arrival: Optional[float] = Field(
        0.2, description="Hours between ED arrival and this ECG.")

    model_config = {"extra": "allow"}


class ExtractionEvidence(BaseModel):
    """One field the PDF parser filled, and the text it came from."""

    field: str
    value: object
    source_text: str
    confidence: str


class ExtractionReport(BaseModel):
    """What the parser found, and just as importantly what it did not.

    Component 04 encodes missingness as signal, so a field the parser missed is
    not a blank to be filled — it is asserted to the model as "not ordered".
    That makes `not_found` as clinically consequential as `fields`, and both
    are surfaced for review before the prediction is acted on.
    """

    fields: Dict = Field(default_factory=dict)
    evidence: List[ExtractionEvidence] = Field(default_factory=list)
    not_found: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    document: Dict = Field(default_factory=dict)


class TriageRequest(BaseModel):
    """One emergency-department presentation, as known at the decision point."""

    label: Optional[str] = Field(None, description="Free-text case label, for display only.")

    age: Optional[float] = Field(None, ge=0, le=120)
    sex: Optional[str] = Field(None, description="M or F")
    race: Optional[str] = None

    heartrate: Optional[float] = None
    sbp: Optional[float] = None
    dbp: Optional[float] = None
    resprate: Optional[float] = None
    o2sat: Optional[float] = None
    temperature: Optional[float] = Field(None, description="degrees Fahrenheit")
    pain: Optional[float] = Field(None, ge=0, le=10)
    acuity: Optional[float] = Field(None, ge=1, le=5, description="ESI level")

    chief_complaint: str = Field("", description="Triage free text.")

    troponin: List[float] = Field(
        default_factory=list, description="Serial values in ED order. Empty = not ordered.")
    troponin_hours: List[float] = Field(
        default_factory=list, description="Hours after arrival for each troponin draw.")
    bnp: Optional[float] = None

    ecg: Optional[ECGReport] = None

    home_medications: List[str] = Field(default_factory=list)
    prior_ed_visits: int = 0
    days_since_last_visit: Optional[float] = None
    prior_acs: int = 0
    prior_mi: int = 0
    prior_chf: int = 0
    diabetes: int = 0
    renal_disease: int = 0
    charlson_index: Optional[float] = Field(
        None,
        description="Supply only if it predates this admission. Joining the INDEX "
                    "admission's comorbidities is leakage channel L1 and moves AUROC "
                    "0.9665 -> 0.9889 on its own.")

    horizon: Optional[int] = Field(
        None,
        description="Disclosure horizon in hours after arrival: 0, 6 or 24. "
                    "Defaults to the service's configured horizon.")

    @model_validator(mode="after")
    def _check_troponin_pairs(self) -> "TriageRequest":
        if self.troponin_hours and len(self.troponin_hours) != len(self.troponin):
            raise ValueError(
                "troponin_hours must have the same length as troponin "
                "(%d values, %d timestamps)" % (len(self.troponin), len(self.troponin_hours)))
        if self.horizon is not None and self.horizon not in (0, 6, 24):
            raise ValueError("horizon must be one of 0, 6 or 24 hours")
        if self.sex is not None and self.sex.strip().upper()[:1] not in ("M", "F", ""):
            raise ValueError("sex must be M or F")
        return self

    def to_component_dict(self) -> Dict:
        """Flatten to the plain dictionary Component 04's featuriser expects."""
        payload: Dict = {
            "label": self.label or "api-request",
            "age": self.age,
            "sex": (self.sex or "").upper()[:1],
            "race": self.race or "UNKNOWN",
            "heartrate": self.heartrate,
            "sbp": self.sbp,
            "dbp": self.dbp,
            "resprate": self.resprate,
            "o2sat": self.o2sat,
            "temperature": self.temperature,
            "pain": self.pain,
            "acuity": self.acuity,
            "chief_complaint": self.chief_complaint or "",
            "troponin": list(self.troponin),
            "troponin_hours": list(self.troponin_hours),
            "bnp": self.bnp,
            "home_medications": list(self.home_medications),
            "prior_ed_visits": self.prior_ed_visits,
            "days_since_last_visit": self.days_since_last_visit,
            "prior_acs": self.prior_acs,
            "prior_mi": self.prior_mi,
            "prior_chf": self.prior_chf,
            "diabetes": self.diabetes,
            "renal_disease": self.renal_disease,
        }
        if self.charlson_index is not None:
            payload["charlson_index"] = self.charlson_index
        if self.ecg is not None:
            payload["ecg"] = {k: v for k, v in self.ecg.model_dump().items()
                              if v is not None and v is not False}
            # hours_after_arrival is meaningful even when zero/false-y.
            if self.ecg.hours_after_arrival is not None:
                payload["ecg"]["hours_after_arrival"] = self.ecg.hours_after_arrival

        # An absent key is how "not measured" is expressed to the featuriser:
        # it reads every optional field as `pt.get(name, np.nan)`, so a key
        # present with the value None reaches float(None) and raises. Stripping
        # Nones hands the component its own missing-value defaults, which is
        # what missingness-aware encoding needs to see.
        return {key: value for key, value in payload.items() if value is not None}


class TriagePdfResponse(BaseModel):
    """The full audit trail for a PDF-driven prediction.

    Three parts, deliberately separate so a reader can check each stage:

      extraction  what the parser found in the document, with source text,
                  and what it could not find
      request     the record actually submitted to the model, after the
                  extracted fields were assembled
      result      the model's answer, in the same envelope every other
                  endpoint returns

    The middle field is the one that makes this reviewable. A parser that
    silently missed a troponin would produce a confident, wrong answer with no
    error anywhere; showing exactly what was submitted makes that visible.
    """

    extraction: ExtractionReport
    request: TriageRequest
    result: Envelope

    review_required: bool = Field(
        True,
        description="Always true. Extraction is a regex-and-lexicon parser over the "
                    "text layer, not a document AI; the assembled record must be "
                    "checked against the source before the answer is used.")
