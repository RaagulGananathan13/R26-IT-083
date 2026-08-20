"""
The cross-modal response contract.

WHY THERE IS A SHARED CONTRACT AT ALL
-------------------------------------
The four components answer different clinical questions on different data, so
their payloads have nothing in common at the level of *findings*. They do have
something in common at the level of *trust*: every one of them was built around
a mechanism that declines to commit when its own evidence is weak.

    Component 01   per-projection operating point + selective deferral
                   (AP films: measured AUROC 0.8224 vs 0.8864 on PA)
    Component 02   quality gate -> refusal, conformal rule-in / rule-out zones,
                   electrode-reversal and out-of-scope guarantee withdrawal,
                   report verification gate
    Component 03   split-conformal EF interval, learned aleatoric sigma plus
                   inter-clip epistemic disagreement
    Component 04   disclosure horizon, constrained decision layer, clinician
                   referral below a validation-chosen confidence

Those mechanisms are the reason the components are worth integrating rather
than merely co-hosting. This module normalises them into one `Reliability`
block and one `Actionability` verdict, so a caller can apply a single rule --
*do not act on a result that is not actionable* -- across all four modalities
without knowing anything about projections, conformal zones or horizons.

The component-native payload is never rewritten. It is returned verbatim under
`raw`, so nothing is lost in translation and every published figure remains
checkable against the component that produced it.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Actionability(str, Enum):
    """How much weight a caller may place on this result.

    Ordered from most to least usable. `rank()` makes the ordering explicit so
    the multi-modal assessment can take the worst case across components.
    """

    ACTIONABLE = "actionable"   # the component stands behind this answer
    CAUTION = "caution"         # answer stands, but measured reliability is reduced
    DEFERRED = "deferred"       # the component declines to commit; refer to a clinician
    WITHHELD = "withheld"       # output suppressed: quality or verification failure
    UNAVAILABLE = "unavailable"  # the component could not run at all

    @classmethod
    def rank(cls, value: "Actionability") -> int:
        order = [cls.ACTIONABLE, cls.CAUTION, cls.DEFERRED, cls.WITHHELD, cls.UNAVAILABLE]
        return order.index(value)

    @classmethod
    def worst(cls, values: List["Actionability"]) -> "Actionability":
        if not values:
            return cls.UNAVAILABLE
        return max(values, key=cls.rank)


class ComponentStatus(str, Enum):
    READY = "ready"             # loaded, weights resident, serving
    AVAILABLE = "available"     # assets present, not yet loaded (lazy)
    UNAVAILABLE = "unavailable"  # root, weights or dependency missing
    FAILED = "failed"           # load was attempted and raised


class Reliability(BaseModel):
    """The component's own statement about how far to trust this result."""

    actionability: Actionability = Field(
        ..., description="Normalised verdict; apply this before using the findings.")
    level: str = Field(
        ..., description="Component-native reliability label, e.g. standard / reduced.")
    reasons: List[str] = Field(
        default_factory=list,
        description="Why the verdict is what it is, in clinical language.")
    guarantees: List[str] = Field(
        default_factory=list,
        description="Statistical guarantees that hold for THIS result. Empty means none.")
    guarantees_void: bool = Field(
        False,
        description="True when a guarantee normally offered does not apply to this record.")
    coverage: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Fraction of studies answered at this operating point. A selective "
                    "metric without its coverage is meaningless.")


class Finding(BaseModel):
    """One clinical statement, with the number behind it."""

    name: str
    present: Optional[bool] = Field(
        None, description="None when the component reports a grade rather than presence.")
    label: Optional[str] = Field(None, description="Categorical grade, where applicable.")
    probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    threshold: Optional[float] = Field(
        None, description="Operating point applied to this finding.")
    value: Optional[float] = Field(None, description="Continuous measurement, e.g. EF %.")
    unit: Optional[str] = None
    interval: Optional[List[float]] = Field(
        None, description="[low, high] prediction interval where the component supplies one.")
    zone: Optional[str] = Field(
        None, description="Conformal decision zone: rule_out / refer / rule_in.")
    evidence: Optional[str] = Field(None, description="What the model attended to.")


class ModelCard(BaseModel):
    """Provenance and measured performance, carried with every response."""

    component_id: str
    component_name: str
    owner: str
    modality: str
    task: str
    dataset: str
    architecture: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    limitations: List[str] = Field(default_factory=list)
    decision_rule: Optional[str] = Field(
        None, description="Which frozen operating point produced this answer.")


class Envelope(BaseModel):
    """The response every component returns, whatever the modality."""

    component: str = Field(..., description="cxr | ecg | echo | triage")
    status: str = Field(..., description="ok | refused | error")
    headline: str = Field(..., description="One-line clinical summary.")
    findings: List[Finding] = Field(default_factory=list)
    reliability: Reliability
    explanation: Dict[str, Any] = Field(
        default_factory=dict,
        description="Saliency, attribution and localisation, as the component reports it.")
    narrative: Optional[str] = Field(
        None, description="Generated or template-grounded report text, where produced.")
    model: ModelCard
    raw: Dict[str, Any] = Field(
        default_factory=dict,
        description="The component-native payload, unmodified.")
    elapsed_ms: int = 0
    request_id: str = "-"

    disclaimer: str = Field(
        default=("AI-generated decision support from an unvalidated research prototype. "
                 "NOT a medical device and NOT a diagnosis. Every output requires review "
                 "by a qualified clinician."))


class ComponentInfo(BaseModel):
    """Registry entry, returned by /components."""

    id: str
    name: str
    owner: str
    modality: str
    task: str
    dataset: str
    status: ComponentStatus
    endpoint: str
    root: Optional[str] = None
    detail: Optional[str] = Field(
        None, description="Why the component is unavailable, when it is.")
    notes: List[str] = Field(
        default_factory=list,
        description="Optional capabilities absent on this install. Informational: "
                    "none of these stops the component serving.")
    model: Optional[ModelCard] = None


class HealthReport(BaseModel):
    service: str
    version: str
    project_id: str
    status: str = Field(..., description="ok when at least one component can serve.")
    device: str
    components: List[ComponentInfo]
    uptime_s: float
