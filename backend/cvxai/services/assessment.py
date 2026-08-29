"""
Cross-modal assessment.

WHAT THIS IS
------------
A patient may arrive with more than one study. This service runs whichever
modalities were supplied, then answers two questions the individual endpoints
cannot:

  1. *Can any of this be acted on?* -- by reducing four independent reliability
     verdicts to the worst one, because a chain of evidence is no stronger than
     its weakest link.
  2. *Do the modalities agree?* -- by reporting concordance and discordance as
     traceable observations.

WHAT THIS IS NOT
----------------
It is not a fusion model. No joint model was trained across the four
modalities, no cross-modal weights were fitted, and no combined accuracy is
claimed. There is no cohort in this project carrying all four modalities for
the same patient, so a joint claim could not be validated even in principle.
Every clinical number below belongs to exactly one component and is reproduced
unchanged.

The observations are rule-based and each one names the values it rests on, so a
reader can check it against the component payloads in the same response.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from cvxai.core.errors import ComponentUnavailable, CvxaiError
from cvxai.core.logging import get_logger, get_request_id
from cvxai.core.registry import ComponentRegistry
from cvxai.schemas.assessment import (
    AssessmentResponse,
    AssessmentSummary,
    CrossModalObservation,
)
from cvxai.schemas.common import Actionability, Envelope
from cvxai.schemas.triage import TriageRequest

log = get_logger("cvxai.assessment")

#: Severity ordering for Component 03's grades, worst first.
_REDUCED_EF_CLASSES = ("Severe(<30)", "Moderate(30-40)", "Mild(40-55)")

#: Component 02 superclasses that indicate ischaemic injury.
_ISCHAEMIC_ECG = ("MI", "STTC")


class AssessmentService:
    """Runs the supplied modalities and aggregates their verdicts."""

    def __init__(self, registry: ComponentRegistry) -> None:
        self.registry = registry

    # ------------------------------------------------------------------
    def run(
        self,
        patient_id: str,
        cxr: Optional[Dict] = None,
        ecg: Optional[Dict] = None,
        echo: Optional[Dict] = None,
        triage: Optional[TriageRequest] = None,
    ) -> AssessmentResponse:
        started = time.perf_counter()
        envelopes: Dict[str, Envelope] = {}
        skipped: Dict[str, str] = {}

        jobs = (
            ("cxr", cxr),
            ("ecg", ecg),
            ("echo", echo),
            ("triage", {"request": triage} if triage is not None else None),
        )
        for component_id, kwargs in jobs:
            if not kwargs:
                skipped[component_id] = "No study supplied for this modality."
                continue
            try:
                adapter = self.registry.get(component_id)
                envelopes[component_id] = adapter.analyze(**kwargs)
            except ComponentUnavailable as exc:
                skipped[component_id] = exc.message
                log.warning("assessment: %s unavailable -- %s", component_id, exc.message)
            except CvxaiError as exc:
                # One modality failing must not lose the others' results.
                skipped[component_id] = "%s: %s" % (exc.code, exc.message)
                log.warning("assessment: %s failed -- %s", component_id, exc.message)

        summary = self._summarise(envelopes, skipped)
        return AssessmentResponse(
            patient_id=patient_id,
            summary=summary,
            observations=self._observe(envelopes),
            components=envelopes,
            skipped=skipped,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            request_id=get_request_id(),
        )

    # ------------------------------------------------------------------
    def _summarise(self, envelopes: Dict[str, Envelope],
                   skipped: Dict[str, str]) -> AssessmentSummary:
        if not envelopes:
            return AssessmentSummary(
                actionability=Actionability.UNAVAILABLE,
                reasons=["No modality produced a result."] + list(skipped.values()),
                headline="No assessment produced.",
            )

        verdicts = [env.reliability.actionability for env in envelopes.values()]
        worst = Actionability.worst(verdicts)

        actionable, blocked, reasons, guarantees = [], [], [], []
        for component_id, envelope in envelopes.items():
            reliability = envelope.reliability
            if reliability.actionability in (Actionability.ACTIONABLE, Actionability.CAUTION):
                actionable.append(component_id)
            else:
                blocked.append(component_id)
            for reason in reliability.reasons:
                reasons.append("[%s] %s" % (component_id, reason))
            for guarantee in reliability.guarantees:
                guarantees.append("[%s] %s" % (component_id, guarantee))

        return AssessmentSummary(
            actionability=worst,
            actionable_components=actionable,
            blocked_components=blocked,
            reasons=reasons,
            guarantees=guarantees,
            headline=self._headline(envelopes, worst),
        )

    @staticmethod
    def _headline(envelopes: Dict[str, Envelope], worst: Actionability) -> str:
        parts = ["%s: %s" % (cid, env.headline) for cid, env in envelopes.items()]
        prefix = {
            Actionability.ACTIONABLE: "All contributing components stand behind their result",
            Actionability.CAUTION: "Reduced reliability in at least one modality",
            Actionability.DEFERRED: "At least one modality declined to commit",
            Actionability.WITHHELD: "At least one modality withheld its output",
            Actionability.UNAVAILABLE: "At least one modality could not run",
        }[worst]
        return "%s. %s" % (prefix, " | ".join(parts))

    # ------------------------------------------------------------------
    def _observe(self, envelopes: Dict[str, Envelope]) -> List[CrossModalObservation]:
        """Factual relationships between modalities, each with its basis."""
        observations: List[CrossModalObservation] = []
        observations += self._cardiomegaly_versus_ef(envelopes)
        observations += self._ecg_versus_triage(envelopes)
        observations += self._horizon_context(envelopes)
        return observations

    @staticmethod
    def _cardiomegaly_versus_ef(envelopes: Dict[str, Envelope]
                                ) -> List[CrossModalObservation]:
        """Radiographic cardiomegaly against measured systolic function.

        Both bear on left-ventricular status, and they are measured from
        different physics -- silhouette geometry against volume change -- so
        agreement is genuinely corroborative and disagreement is informative.
        """
        cxr, echo = envelopes.get("cxr"), envelopes.get("echo")
        if not cxr or not echo:
            return []

        enlarged = next((f for f in cxr.findings if f.name == "Cardiomegaly"), None)
        grade = next((f for f in echo.findings if f.name == "Severity grade"), None)
        ef = next((f for f in echo.findings
                   if f.name == "Left-ventricular ejection fraction"), None)
        if enlarged is None or grade is None or ef is None:
            return []

        reduced = grade.label in _REDUCED_EF_CLASSES
        basis = ("Component 01 cardiomegaly p=%.3f at threshold %.3f; Component 03 "
                 "EF %.1f %% graded %s."
                 % (enlarged.probability or 0.0, enlarged.threshold or 0.0,
                    ef.value or 0.0, grade.label))

        if enlarged.present and reduced:
            return [CrossModalObservation(
                kind="concordance", components=["cxr", "echo"],
                statement="Radiographic cardiomegaly and measured systolic impairment "
                          "agree; two independent modalities point at the same "
                          "left-ventricular abnormality.",
                basis=basis)]
        if enlarged.present and not reduced:
            return [CrossModalObservation(
                kind="discordance", components=["cxr", "echo"],
                statement="An enlarged cardiac silhouette with preserved ejection "
                          "fraction. Cardiomegaly on a radiograph is a geometric "
                          "finding and does not require reduced systolic function; on "
                          "an AP film it may also reflect projection magnification.",
                basis=basis)]
        if not enlarged.present and reduced:
            return [CrossModalObservation(
                kind="discordance", components=["cxr", "echo"],
                statement="Reduced ejection fraction without radiographic "
                          "cardiomegaly. The echocardiogram measures function "
                          "directly and the radiograph does not; do not read the "
                          "normal film as reassurance.",
                basis=basis)]
        return [CrossModalObservation(
            kind="concordance", components=["cxr", "echo"],
            statement="No radiographic cardiomegaly and preserved ejection fraction.",
            basis=basis)]

    @staticmethod
    def _ecg_versus_triage(envelopes: Dict[str, Envelope]) -> List[CrossModalObservation]:
        """The ECG waveform model against the ED triage model.

        Component 02 reads the waveform; Component 04 reads the cart's text
        report plus the rest of the record. They see the same heart through
        different instruments, and Component 04's stated ceiling is precisely
        that it never sees the waveform.
        """
        ecg, triage = envelopes.get("ecg"), envelopes.get("triage")
        if not ecg or not triage:
            return []

        raw_ecg = ecg.raw or {}
        zones = raw_ecg.get("zones") or {}
        ruled_in = [name for name in _ISCHAEMIC_ECG if zones.get(name) == "rule_in"]
        triage_raw = triage.raw or {}
        prediction = triage_raw.get("prediction", "No_ACS")
        p_acs = float(triage_raw.get("p_acs", 0.0))

        basis = ("Component 02 conformal zones %s; Component 04 %s at P(ACS)=%.3f."
                 % (zones or "unavailable", prediction, p_acs))

        if raw_ecg.get("refused"):
            return [CrossModalObservation(
                kind="context", components=["ecg", "triage"],
                statement="The ECG was refused on quality grounds, so it contributes "
                          "nothing here. The triage estimate rests on the remaining "
                          "modalities and the absence of an ECG finding is not "
                          "evidence of a normal ECG.",
                basis=basis)]

        if ruled_in and prediction != "No_ACS":
            return [CrossModalObservation(
                kind="concordance", components=["ecg", "triage"],
                statement="Waveform-level ischaemic change and the ED triage model "
                          "agree on an acute coronary syndrome. These are independent "
                          "instruments: Component 04 never sees the waveform, only the "
                          "cart's text report.",
                basis=basis)]
        if ruled_in and prediction == "No_ACS":
            return [CrossModalObservation(
                kind="discordance", components=["ecg", "triage"],
                statement="The waveform model rules in ischaemic change while the "
                          "triage model does not call ACS. Component 04's stated "
                          "ceiling is exactly this gap -- ST elevation is recoverable "
                          "from the MIMIC text report in only 41 % of STEMI cases.",
                basis=basis)]
        if not ruled_in and prediction != "No_ACS":
            return [CrossModalObservation(
                kind="discordance", components=["ecg", "triage"],
                statement="The triage model calls ACS without waveform-level ischaemic "
                          "change. Unstable angina and early NSTEMI can present with a "
                          "non-diagnostic ECG, so this combination is clinically "
                          "ordinary rather than contradictory.",
                basis=basis)]
        return [CrossModalObservation(
            kind="concordance", components=["ecg", "triage"],
            statement="Neither the waveform model nor the triage model indicates an "
                      "acute coronary syndrome.",
            basis=basis)]

    @staticmethod
    def _horizon_context(envelopes: Dict[str, Envelope]) -> List[CrossModalObservation]:
        """Flag when the triage answer is bounded by information, not by model."""
        triage = envelopes.get("triage")
        if not triage:
            return []
        raw = triage.raw or {}
        horizon = raw.get("horizon_h")
        if horizon is None or int(horizon) >= 24:
            return []
        return [CrossModalObservation(
            kind="context", components=["triage"],
            statement="The triage result is bounded by information availability, not "
                      "by model quality. At this horizon the biomarker has not "
                      "returned, and unstable angina is defined by a normal troponin.",
            basis="Component 04 served at H=%sh; measured UA recall 37.3 %% at H=0, "
                  "58.2 %% at H=6, 80.0 %% at H=24." % horizon)]
