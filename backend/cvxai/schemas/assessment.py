"""
Multi-modal assessment: what the four components say about one patient.

SCOPE, STATED UP FRONT
----------------------
This is an *aggregation*, not a fusion model. No joint model was trained across
the four modalities, no cross-modal weights were fitted, and no combined
performance figure is claimed -- there is no dataset in this project carrying
all four modalities for the same patient, so such a claim could not be
validated. What the aggregation does is:

  1. run whichever modalities were supplied, independently;
  2. reduce their reliability verdicts to a single worst-case verdict, because
     a chain of evidence is no more trustworthy than its weakest link;
  3. report agreement and disagreement between modalities as observations,
     each traceable to the component that produced it.

Every clinical number in the result belongs to exactly one component and is
reproduced unchanged. The cross-modal layer adds no probabilities of its own.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from cvxai.schemas.common import Actionability, Envelope
from cvxai.schemas.triage import TriageRequest


class AssessmentRequest(BaseModel):
    """JSON body for the multi-modal endpoint.

    Imaging and signal modalities are uploaded through their own endpoints;
    this body carries the tabular modality and the study identifiers, so the
    endpoint accepts a multipart form with optional file parts alongside it.
    """

    patient_id: str = Field("anonymous", description="Local identifier; never a real MRN.")
    cxr_view: Optional[str] = Field(
        None, description="AP or PA. Omitted means the global operating point is used.")
    triage: Optional[TriageRequest] = None


class CrossModalObservation(BaseModel):
    """One factual statement about how two components relate on this patient."""

    kind: str = Field(..., description="concordance | discordance | context")
    components: List[str]
    statement: str
    basis: str = Field(..., description="The specific values the statement rests on.")


class AssessmentSummary(BaseModel):
    actionability: Actionability = Field(
        ..., description="Worst case across the contributing components.")
    actionable_components: List[str] = Field(default_factory=list)
    blocked_components: List[str] = Field(default_factory=list)
    reasons: List[str] = Field(
        default_factory=list,
        description="Every reason any component gave for reduced trust.")
    guarantees: List[str] = Field(default_factory=list)
    headline: str


class AssessmentResponse(BaseModel):
    patient_id: str
    summary: AssessmentSummary
    observations: List[CrossModalObservation] = Field(default_factory=list)
    components: Dict[str, Envelope] = Field(
        default_factory=dict, description="Keyed by component id.")
    skipped: Dict[str, str] = Field(
        default_factory=dict,
        description="Component id -> why no result was produced.")
    elapsed_ms: int = 0
    request_id: str = "-"

    method_note: str = Field(
        default=("Aggregation only. The four components were developed and validated "
                 "independently on separate cohorts; no cross-modal model was trained "
                 "and no joint performance is claimed. The summary verdict is the "
                 "worst per-component verdict, and every clinical figure below is "
                 "reproduced unchanged from the component that produced it. "
                 "The cohorts cannot be joined: Components 02 and 03 come from "
                 "different institutions, countries and decades than the "
                 "MIMIC-derived Components 01 and 04 and share no patient "
                 "identifier. See GET /api/v1/cohorts for the measured figures."))
    disclaimer: str = Field(
        default=("AI-generated decision support from an unvalidated research prototype. "
                 "NOT a medical device and NOT a diagnosis. Every output requires review "
                 "by a qualified clinician."))
