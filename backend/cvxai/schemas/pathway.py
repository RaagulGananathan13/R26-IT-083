"""
The clinical pathway contract.

WHY THIS IS NOT `/assessment`
-----------------------------
`/assessment` runs every supplied modality and reduces four verdicts to the
worst one. It is order-free by design: a chest film and an echo are independent
readings and neither gates the other.

A clinical pathway is the opposite. The order carries the clinical meaning, and
one result can make the next test irrelevant:

  * A STEMI on the ECG stops the workup. Door-to-balloon is a 90-minute target
    and in-hospital mortality is 8 % under it against 20 % over, so nothing --
    not the radiograph, not the biomarker, not the echo -- may be allowed to
    delay reperfusion.
  * A MINIMAL risk band at triage means the chest-pain fast track is not
    entered at all.
  * A pneumothorax on the film means the answer was never ACS.

So this contract models *stages and the routing between them*, not a bag of
results. Every stage records why the pathway moved where it did, and the
`basis` field names the values that decision rests on, so a reader can check
each hop against the component payload carried in the same response.

WHAT IT DOES NOT CLAIM
----------------------
The four components were trained on four separate cohorts -- MIMIC-CXR, PTB-XL,
EchoNet-Dynamic/CAMUS and MIMIC-IV-ED -- and no patient appears in all four.
This is therefore how the components *would* compose clinically, justified
against published guidelines. It is not a validated end-to-end study on one
population, and no joint accuracy is claimed or claimable.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from cvxai.schemas.assessment import CrossModalObservation
from cvxai.schemas.common import Actionability, Envelope


class StageStatus(str, Enum):
    """What became of one stage of the pathway."""

    COMPLETED = "completed"          # ran and produced a result
    NOT_SUPPLIED = "not_supplied"    # reached, but no study was provided
    SKIPPED = "skipped"              # routing bypassed it deliberately
    BLOCKED = "blocked"              # the component could not run
    NOT_REACHED = "not_reached"      # the pathway terminated before it


class Urgency(str, Enum):
    IMMEDIATE = "immediate"          # minutes; a time target applies
    URGENT = "urgent"                # same admission, hours
    ROUTINE = "routine"
    NONE = "none"


class StageRouting(BaseModel):
    """Why the pathway went where it went, and on what evidence."""

    branch: str = Field(..., description="Machine-readable branch identifier.")
    statement: str = Field(..., description="The routing decision in clinical language.")
    basis: str = Field(
        ...,
        description="The values this decision rests on, so it can be checked against "
                    "the stage's own payload.")
    next_stage: Optional[str] = Field(
        None, description="Stage id the pathway advances to. None when it terminates.")
    terminates: bool = Field(
        False, description="True when this branch ends the pathway.")
    urgency: Urgency = Urgency.ROUTINE
    guideline: Optional[str] = Field(
        None, description="The guideline or evidence justifying this hop.")


class PathwayStage(BaseModel):
    """One clinical act, its component, and what followed from it."""

    id: str
    ordinal: int = Field(..., description="Position in the documented pathway, 1-6.")
    clock: str = Field(..., description="Time from arrival, e.g. 'T + 10 min'.")
    component: Optional[str] = Field(None, description="cxr | ecg | echo | triage")
    horizon_h: Optional[int] = Field(
        None, description="Component 04 only: the disclosure horizon this stage serves.")
    title: str
    clinical_act: str = Field(..., description="What the clinician physically does here.")
    question: str = Field(..., description="The question this stage answers.")

    status: StageStatus
    detail: Optional[str] = Field(
        None, description="Why the stage has the status it has, when not completed.")
    routing: Optional[StageRouting] = None
    result: Optional[Envelope] = Field(
        None, description="The component envelope, when the stage ran.")

    deadline: Optional[str] = Field(
        None, description="Hard clinical deadline attached to this stage, where one exists.")


class Disposition(BaseModel):
    """Where the pathway says the patient goes."""

    destination: str = Field(
        ...,
        description="cath_lab | ccu | ward | observation | discharge | non_cardiac | "
                    "indeterminate")
    label: str
    urgency: Urgency
    time_target: Optional[str] = Field(
        None, description="Where a guideline attaches a clock, e.g. door-to-balloon.")
    rationale: List[str] = Field(default_factory=list)
    heart_failure_pathway: bool = Field(
        False,
        description="True when the echo opened a parallel heart-failure pathway "
                    "(EF < 40 %) alongside the ACS one.")


class PathwayResponse(BaseModel):
    """The full traversal: every stage, the routing between them, the endpoint."""

    patient_id: str
    stages: List[PathwayStage]
    disposition: Disposition

    actionability: Actionability = Field(
        ...,
        description="Worst verdict across the stages that actually ran. Apply this "
                    "before using anything below it.")
    actionability_reasons: List[str] = Field(default_factory=list)

    terminated_at: Optional[str] = Field(
        None, description="Stage id where the pathway stopped early, if it did.")
    termination_reason: Optional[str] = None

    observations: List[CrossModalObservation] = Field(
        default_factory=list,
        description="Cross-modal relationships between the stages that ran.")

    stages_completed: int = 0
    stages_total: int = 6
    limits: List[str] = Field(
        default_factory=list,
        description="What this traversal does not establish. Always populated.")

    elapsed_ms: int = 0
    request_id: str = "-"
    disclaimer: str = Field(
        default=("AI-generated decision support from an unvalidated research prototype. "
                 "NOT a medical device and NOT a diagnosis. This pathway is an ordering "
                 "justified against published guidelines, not a standard of care, and "
                 "not a validated end-to-end study. Every output requires review by a "
                 "qualified clinician."))


class PathwayDefinition(BaseModel):
    """The static pathway map, served so a client can render it before any input."""

    stages: List[Dict[str, Any]]
    references: List[Dict[str, str]]


class PathwayContext(BaseModel):
    """Continuation state for a stage-by-stage traversal.

    Produced by the server and handed back unchanged on the next call. The
    client never reads or constructs it, which is the point: the routing rules
    stay in one place. A console that inspected envelopes to decide what came
    next would be a second copy of the pathway, free to disagree with the
    first.
    """

    visited: List[str] = Field(default_factory=list)
    completed: List[str] = Field(
        default_factory=list, description="Stages that actually produced a result.")
    hf_pathway: bool = Field(
        False, description="Set once the echocardiogram measured EF inside HFrEF.")
    mimics: List[str] = Field(
        default_factory=list, description="Radiographic findings needing parallel treatment.")
    verdicts: Dict[str, str] = Field(
        default_factory=dict, description="stage id -> actionability, for the worst-case roll-up.")
    terminated_at: Optional[str] = None
    termination_reason: Optional[str] = None


class StageRunResponse(BaseModel):
    """One stage of a step-by-step traversal, and where it leads."""

    stage: PathwayStage
    context: PathwayContext = Field(
        ..., description="Pass this back on the next call, unmodified.")
    next_stage: Optional[str] = Field(
        None, description="Stage id to run next. None when the pathway has ended.")
    skipped: List[PathwayStage] = Field(
        default_factory=list,
        description="Stages the routing just advanced past, in order. A stepwise "
                    "caller cannot work these out without reimplementing the "
                    "routing rules, and a stage that was deliberately bypassed "
                    "must not be presented as one that has yet to run.")
    finished: bool = Field(
        False, description="True when no further stage should be run.")

    actionability: Actionability = Field(
        ..., description="Worst verdict across every stage that has run so far.")
    disposition: Optional[Disposition] = Field(
        None, description="Present only when `finished`.")
    limits: List[str] = Field(default_factory=list)

    elapsed_ms: int = 0
    request_id: str = "-"
