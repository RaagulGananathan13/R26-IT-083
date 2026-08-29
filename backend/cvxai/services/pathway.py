"""
The clinical pathway engine.

WHAT THIS DOES
--------------
Walks one patient through the six stages of `CLINICAL_WORKFLOW.md`, in order,
letting each stage's result decide where the pathway goes next. Component 04 is
not one stage but three -- the same record re-featurised at its H = 0, H = 6 and
H = 24 disclosure horizons -- and the other three components slot into the gaps
between those horizons.

    T + 0 min      04 @ H=0    triage assessment      is this a possible ACS?
    T + 10 min     02          12-lead ECG            infarct pattern right now?
    T + 15-60 min  01          chest radiograph       what else could kill them?
    T + 1-6 h      04 @ H=6    troponin 0 h / 1 h     does the biomarker confirm?
    T + 6-24 h     03          echocardiogram         how well is the pump working?
    T + 24 h       04 @ H=24   workup complete        UA / NSTEMI / STEMI?

WHY ORDER MATTERS HERE AND NOT IN `/assessment`
-----------------------------------------------
`/assessment` is order-free: four independent readings reduced to their worst
verdict. This is a gated traversal, because a result can make the next test
clinically irrelevant. Three branches end the pathway outright:

    ECG rules in MI          reperfusion; door-to-balloon <= 90 min, and
                             in-hospital mortality is 8 % under that target
                             against 20 % over it. Nothing may delay it.
    MINIMAL / LOW at H = 0   the chest-pain fast track is not entered.
    Pneumothorax on film     the answer was never ACS; it needs decompression.

Every hop records the values it rests on in `basis`, so a reader can check the
routing against the component payload carried in the same response rather than
taking the engine's word for it.

MISSING STUDIES ARE NORMAL
--------------------------
A pathway is walked with whatever exists. A stage with no study is marked
`not_supplied` and the traversal continues -- that is the ordinary case at the
bedside, where the echo is hours away. Absence is never read as a negative
finding: a stage that did not run contributes nothing, and the disposition
records that it is resting on an incomplete workup.
"""
from __future__ import annotations

import dataclasses
import threading
import time
from typing import Dict, List, Optional, Tuple

from cvxai.core.errors import ComponentUnavailable, CvxaiError, InvalidInput
from cvxai.core.logging import get_logger, get_request_id
from cvxai.core.registry import ComponentRegistry
from cvxai.schemas.common import Actionability, Envelope
from cvxai.schemas.pathway import (
    Disposition,
    PathwayContext,
    PathwayResponse,
    PathwayStage,
    StageRouting,
    StageRunResponse,
    StageStatus,
    Urgency,
)
from cvxai.schemas.triage import TriageRequest
from cvxai.services.assessment import AssessmentService

log = get_logger("cvxai.pathway")

#: Bands at which the chest-pain fast track is NOT entered at triage (H = 0).
#:
#: Only MINIMAL. `CLINICAL_WORKFLOW.md` groups MINIMAL with LOW here, and that
#: grouping does not survive contact with the model's own output. Measured on
#: the curated demo records at H = 0:
#:
#:     genuine STEMI          LOW, P(ACS) = 0.111
#:     genuine NSTEMI         LOW, P(ACS) = 0.053
#:     unstable angina        LOW, P(ACS) = 0.109
#:     non-cardiac            MINIMAL, P(ACS) = 0.000
#:
#: Exiting on LOW sends all three genuine coronary syndromes home before the
#: ECG. The band boundary sits at 0.05 while cohort prevalence is 5.6 %, so LOW
#: spans "at the base rate" up to "four times the base rate" -- not a rule-out.
#:
#: The stage-1 screen is tuned to negative predictive value, and its published
#: operating point pairs NPV 99.41 % with sensitivity 91.35 %: about one ACS in
#: eleven is already missed there. Widening the exit to LOW compounds that at
#: precisely the moment the guideline mandates an ECG within ten minutes.
#: MINIMAL -- below the base rate -- is the defensible rule-out band.
_FAST_TRACK_EXIT_BANDS = ("MINIMAL",)

#: Bands that support a rule-out at H = 6, once the biomarker has returned.
#:
#: Wider than the stage-1 set, deliberately. Discharging before the troponin and
#: discharging after it are different decisions on different evidence, and the
#: ESC 0/1 h rule-out arm exists precisely because the biomarker licenses one the
#: triage assessment alone does not.
_RULE_OUT_BANDS = ("MINIMAL", "LOW")

#: Component 03 grades that fall inside HFrEF (EF <= 40 %).
_HFREF_GRADES = ("Severe(<30)", "Moderate(30-40)")

#: Component 01 findings that are immediate non-ACS killers. Pneumothorax is
#: separated from the rest because it needs decompression now, not a work-up.
_CRITICAL_MIMIC = "Pneumothorax"
_MIMIC_FINDINGS = ("Pleural Effusion", "Consolidation", "Pneumonia")

#: Component 01 findings that mean the heart is structurally abnormal and needs
#: quantifying, which is the echo's job.
_STRUCTURAL_FINDINGS = ("Cardiomegaly", "Edema")

#: The pathway map. Order is the traversal order; routing may skip forward
#: within it but never backward.
STAGE_ORDER = ("triage_h0", "ecg", "cxr", "triage_h6", "echo", "triage_h24")

STAGE_SPEC: Dict[str, Dict] = {
    "triage_h0": dict(
        ordinal=1, clock="T + 0 min", component="triage", horizon_h=0,
        title="Triage assessment",
        clinical_act="Vitals, chief complaint and history are taken. No test ordered yet.",
        question="Is this patient a possible acute coronary syndrome?",
        deadline=None),
    "ecg": dict(
        ordinal=2, clock="T + 10 min", component="ecg", horizon_h=None,
        title="12-lead ECG",
        clinical_act="A 12-lead ECG is acquired and interpreted.",
        question="Is there an infarct pattern right now?",
        deadline="Within 10 minutes of arrival (2023 ESC ACS guideline). Repeat at "
                 "15-30 minute intervals through the first hour if non-diagnostic."),
    "cxr": dict(
        ordinal=3, clock="T + 15-60 min", component="cxr", horizon_h=None,
        title="Chest radiograph",
        clinical_act="Usually a portable frontal film in the resuscitation bay.",
        question="What else could be killing them? Is the heart enlarged?",
        deadline=None),
    "triage_h6": dict(
        ordinal=4, clock="T + 1-6 h", component="triage", horizon_h=6,
        title="Troponin 0 h / 1 h",
        clinical_act="High-sensitivity troponin at presentation and at one hour.",
        question="Does the biomarker confirm it?",
        deadline=None),
    "echo": dict(
        ordinal=5, clock="T + 6-24 h", component="echo", horizon_h=None,
        title="Echocardiogram",
        clinical_act="Transthoracic echo, apical four-chamber. Needs a sonographer, "
                     "so it is scheduled rather than instant.",
        question="How well is the pump actually working?",
        deadline=None),
    "triage_h24": dict(
        ordinal=6, clock="T + 24 h", component="triage", horizon_h=24,
        title="Workup complete",
        clinical_act="Every feature is now knowable; the full record is re-scored.",
        question="Final subtype: unstable angina, NSTEMI or STEMI?",
        deadline=None),
}

#: Standing limits on any traversal. These are properties of the project, not
#: of a particular patient, so they are always returned.
STANDING_LIMITS = [
    "No component diagnoses anything. Each output is decision support requiring "
    "clinician review.",
    "The four components share no patients. MIMIC-CXR, PTB-XL, "
    "EchoNet-Dynamic/CAMUS and MIMIC-IV-ED are four separate cohorts, so this "
    "pathway is how the components would compose clinically -- justified against "
    "published guidelines -- not a validated end-to-end study on one population.",
    "No joint model was trained across the modalities and no combined accuracy is "
    "claimed. Every figure belongs to exactly one component.",
    "Component 02 recognises five superclasses. Its MI superclass does not separate "
    "STEMI from NSTEMI, and atrial fibrillation and other arrhythmias are outside "
    "the label space entirely.",
    "Component 04's UA/NSTEMI boundary rests on ICD coding rather than adjudicated "
    "labels.",
]


class _HorizonAdapters:
    """One Component 04 adapter per disclosure horizon.

    THE PROBLEM THIS SOLVES
    -----------------------
    Component 04 ships a separately trained model per horizon -- stage1_lgb_H0,
    _H6, _H24, and so on -- and an adapter binds to exactly one set at load
    time. Ask it for another and it refuses, correctly: it has no weights for
    the horizon requested and guessing with the wrong ones would be worse than
    refusing.

    But the pathway is *defined* by three horizons. Stage 1 is H=0, stage 4 is
    H=6 and stage 6 is H=24, and the whole point of the traversal is that UA
    recall climbs 37.3 % -> 58.2 % -> 80.0 % across them as the biomarker
    returns. Serving that from one adapter is not serving it.

    So the pathway holds one adapter per horizon. The service's configured
    adapter is reused from the registry -- it may already be warm -- and the
    other two are built on demand against a settings copy.

    COST
    ----
    Measured on the shipped artifacts: H=0 adds 21.9 MB of model files and H=6
    adds 16.5 MB beyond whichever horizon the service is configured for. They
    are built lazily, so a caller that never runs the pathway pays nothing.

    Note that only H=24 carries UM4, the unified four-class deployment model.
    H=0 and H=6 fall back to the two-stage cascade, which the adapter already
    handles and records in `raw["engine"]` -- so which engine answered is
    visible in the response rather than implied.
    """

    def __init__(self, registry: ComponentRegistry) -> None:
        self._registry = registry
        self._configured = int(registry.settings.triage_horizon)
        self._extra: Dict[int, object] = {}
        self._lock = threading.Lock()

    def get(self, horizon: int):
        horizon = int(horizon)
        if horizon == self._configured:
            # Reuse the registry's instance: it participates in /health and
            # /warm, and may already have its weights resident.
            return self._registry.get("triage")

        with self._lock:
            adapter = self._extra.get(horizon)
            if adapter is None:
                from cvxai.adapters.triage import TriageAdapter

                settings = dataclasses.replace(
                    self._registry.settings, triage_horizon=horizon)
                root = settings.component_roots().get("triage")
                if root is None:
                    raise ComponentUnavailable(
                        "Component 04 is not installed, so horizon H=%d cannot be "
                        "served." % horizon)
                adapter = TriageAdapter(settings, root)
                self._extra[horizon] = adapter
                log.info("pathway: built a Component 04 adapter for H=%d "
                         "(service is configured for H=%d)", horizon, self._configured)
            return adapter


class PathwayService:
    """Traverses the clinical pathway for one patient."""

    def __init__(self, registry: ComponentRegistry, horizons: Optional[object] = None) -> None:
        """`horizons` overrides how Component 04 adapters are resolved per horizon.

        Injected rather than constructed unconditionally so a test can drive the
        routing with synthetic envelopes without three real model loads. Nothing
        in production passes it.
        """
        self.registry = registry
        self._assessment = AssessmentService(registry)
        if horizons is not None:
            self._horizons = horizons
        else:
            # Cached on the registry, not on the service. A service is built per
            # request, so a per-service cache would reload Component 04's H=0 and
            # H=6 weights on every stage of a stage-by-stage traversal -- about
            # three seconds each, repeatedly, for models already in memory.
            existing = getattr(registry, "_pathway_horizons", None)
            if existing is None:
                existing = _HorizonAdapters(registry)
                setattr(registry, "_pathway_horizons", existing)
            self._horizons = existing

    # ------------------------------------------------------------------ #
    #  entry point
    # ------------------------------------------------------------------ #
    def run(
        self,
        patient_id: str,
        cxr: Optional[Dict] = None,
        ecg: Optional[Dict] = None,
        echo: Optional[Dict] = None,
        triage: Optional[TriageRequest] = None,
    ) -> PathwayResponse:
        started = time.perf_counter()

        stages: Dict[str, PathwayStage] = {}
        envelopes: Dict[str, Envelope] = {}
        visited: List[str] = []
        terminated_at: Optional[str] = None
        termination_reason: Optional[str] = None
        hf_pathway = False
        mimic_notes: List[str] = []

        current: Optional[str] = "triage_h0"
        while current is not None:
            spec = STAGE_SPEC[current]
            visited.append(current)

            envelope, status, detail = self._execute(current, spec, cxr, ecg, echo, triage)
            if envelope is not None:
                # Component 04 runs three times; key its envelopes per stage so
                # each horizon keeps its own payload.
                envelopes[current] = envelope

            routing = self._route(current, envelope, status)

            stages[current] = PathwayStage(
                id=current, status=status, detail=detail, routing=routing,
                result=envelope, **spec)

            if current == "echo" and envelope is not None:
                hf_pathway = self._is_hfref(envelope)
            if current == "cxr" and envelope is not None:
                mimic_notes = self._mimic_notes(envelope)

            if routing is not None and routing.terminates:
                terminated_at = current
                termination_reason = routing.statement
                break
            current = routing.next_stage if routing is not None else None

        self._fill_unvisited(stages, visited, terminated_at)

        ordered = [stages[sid] for sid in STAGE_ORDER]
        actionability, reasons = self._aggregate_actionability(envelopes)
        disposition = self._disposition(
            stages, terminated_at, hf_pathway, mimic_notes, envelopes)

        return PathwayResponse(
            patient_id=patient_id,
            stages=ordered,
            disposition=disposition,
            actionability=actionability,
            actionability_reasons=reasons,
            terminated_at=terminated_at,
            termination_reason=termination_reason,
            observations=self._assessment._observe(self._for_observation(envelopes)),
            stages_completed=sum(1 for s in ordered if s.status == StageStatus.COMPLETED),
            stages_total=len(STAGE_ORDER),
            limits=self._limits(ordered),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            request_id=get_request_id(),
        )


    # ------------------------------------------------------------------ #
    #  stage-by-stage traversal
    # ------------------------------------------------------------------ #
    def run_stage(
        self,
        stage_id: str,
        context: Optional[PathwayContext] = None,
        cxr: Optional[Dict] = None,
        ecg: Optional[Dict] = None,
        echo: Optional[Dict] = None,
        triage: Optional[TriageRequest] = None,
    ) -> StageRunResponse:
        """Run exactly one stage and report where it leads.

        WHY THIS EXISTS ALONGSIDE `run`
        -------------------------------
        `run` walks the whole pathway from one payload. That is right when every
        study is already in hand, and wrong when the studies arrive one at a
        time -- which is the ordinary case at the bedside, and the case when the
        pathway is being demonstrated stage by stage with a different person
        supplying each study.

        The routing still comes from here, never from the caller. The client
        hands back the `context` this returned and is not expected to read it;
        a console that inspected envelopes to decide what came next would be a
        second copy of the pathway, free to disagree with this one.
        """
        started = time.perf_counter()
        if stage_id not in STAGE_SPEC:
            raise InvalidInput(
                "Unknown stage %r. Expected one of: %s."
                % (stage_id, ", ".join(STAGE_ORDER)))

        context = (context or PathwayContext()).model_copy(deep=True)
        if context.terminated_at:
            raise InvalidInput(
                "The pathway already ended at %r (%s). Start a new traversal rather "
                "than running a further stage."
                % (context.terminated_at, context.termination_reason or "terminating branch"))

        spec = STAGE_SPEC[stage_id]
        envelope, status, detail = self._execute(stage_id, spec, cxr, ecg, echo, triage)
        routing = self._route(stage_id, envelope, status)

        stage = PathwayStage(
            id=stage_id, status=status, detail=detail, routing=routing,
            result=envelope, **spec)

        # -- accumulate ------------------------------------------------- #
        if stage_id not in context.visited:
            context.visited.append(stage_id)
        if envelope is not None:
            if stage_id not in context.completed:
                context.completed.append(stage_id)
            context.verdicts[stage_id] = envelope.reliability.actionability.value
            if stage_id == "echo":
                context.hf_pathway = self._is_hfref(envelope)
            elif stage_id == "cxr":
                context.mimics = self._mimic_notes(envelope)

        terminates = bool(routing and routing.terminates)
        next_stage = routing.next_stage if routing and not terminates else None
        finished = terminates or next_stage is None
        if finished:
            context.terminated_at = stage_id
            context.termination_reason = routing.statement if routing else None

        # -- roll up ---------------------------------------------------- #
        verdicts = [Actionability(value) for value in context.verdicts.values()]
        actionability = (Actionability.worst(verdicts) if verdicts
                         else Actionability.UNAVAILABLE)

        disposition = None
        if finished:
            disposition = self._disposition_from(
                routing, context.hf_pathway, context.mimics,
                self._absent_titles(context))

        return StageRunResponse(
            stage=stage,
            context=context,
            next_stage=next_stage,
            skipped=self._skipped_between(stage_id, next_stage),
            finished=finished,
            actionability=actionability,
            disposition=disposition,
            limits=self._stepwise_limits(context, finished),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            request_id=get_request_id(),
        )

    @staticmethod
    def _skipped_between(current: str, following: Optional[str]) -> List[PathwayStage]:
        """Stages the routing has just advanced past.

        A radiograph showing an enlarged heart routes straight to the
        echocardiogram, so the troponin stage between them never runs. The
        all-at-once traversal already reports that as `skipped`; the stepwise
        one returned nothing at all for it, and a stage with no record renders
        as one that has simply not happened yet. Deliberately bypassed and
        not-yet-run are different clinical facts and the second is the wrong
        one to show.

        Computed here rather than in the caller on purpose: working it out from
        the stage ids alone means reimplementing the ordering, which is the one
        thing the opaque `context` exists to keep out of clients.
        """
        if not following:
            return []
        try:
            start = STAGE_ORDER.index(current)
            end = STAGE_ORDER.index(following)
        except ValueError:
            return []
        if end <= start + 1:
            return []
        return [
            PathwayStage(
                id=stage_id,
                status=StageStatus.SKIPPED,
                detail="Routing advanced past this stage.",
                routing=None,
                result=None,
                **STAGE_SPEC[stage_id],
            )
            for stage_id in STAGE_ORDER[start + 1:end]
        ]

    @staticmethod
    def _absent_titles(context: PathwayContext) -> List[str]:
        """Stages that were reached but produced nothing."""
        return [STAGE_SPEC[sid]["title"] for sid in context.visited
                if sid not in context.completed]

    @staticmethod
    def _stepwise_limits(context: PathwayContext, finished: bool) -> List[str]:
        limits = list(STANDING_LIMITS)
        absent = PathwayService._absent_titles(context)
        if finished:
            stop = context.terminated_at
            stop_index = STAGE_ORDER.index(stop) if stop in STAGE_ORDER else len(STAGE_ORDER)
            absent = absent + [STAGE_SPEC[sid]["title"] for sid in STAGE_ORDER
                               if sid not in context.visited
                               and STAGE_ORDER.index(sid) > stop_index]
        if absent:
            limits.insert(0,
                          "This traversal is incomplete. The following stages produced "
                          "no evidence and their silence is not a negative finding: %s."
                          % ", ".join(dict.fromkeys(absent)))
        return limits

    # ------------------------------------------------------------------ #
    #  running one stage
    # ------------------------------------------------------------------ #
    def _execute(
        self, stage_id: str, spec: Dict,
        cxr: Optional[Dict], ecg: Optional[Dict], echo: Optional[Dict],
        triage: Optional[TriageRequest],
    ) -> Tuple[Optional[Envelope], StageStatus, Optional[str]]:
        component = spec["component"]

        horizon = spec["horizon_h"]

        if component == "triage":
            if triage is None:
                return None, StageStatus.NOT_SUPPLIED, (
                    "No emergency-department record was supplied, so this horizon "
                    "could not be scored.")
            # The same record, re-featurised at this stage's horizon. The
            # component masks whatever is not yet knowable, which is the entire
            # point of the horizon: at H = 0 the laboratory channel carries
            # exactly 0.0 % attribution because no troponin exists yet.
            kwargs = {"request": triage.model_copy(update={"horizon": horizon})}
        else:
            kwargs = {"cxr": cxr, "ecg": ecg, "echo": echo}[component]
            if not kwargs:
                return None, StageStatus.NOT_SUPPLIED, (
                    "No study was supplied for this stage.")

        try:
            # Component 04 is resolved per horizon: it has different weights for
            # each, and one adapter cannot answer for the other two.
            adapter = (self._horizons.get(horizon) if component == "triage"
                       else self.registry.get(component))
            return adapter.analyze(**kwargs), StageStatus.COMPLETED, None
        except ComponentUnavailable as exc:
            log.warning("pathway: %s unavailable -- %s", stage_id, exc.message)
            return None, StageStatus.BLOCKED, exc.message
        except CvxaiError as exc:
            log.warning("pathway: %s failed -- %s", stage_id, exc.message)
            return None, StageStatus.BLOCKED, "%s: %s" % (exc.code, exc.message)

    # ------------------------------------------------------------------ #
    #  routing
    # ------------------------------------------------------------------ #
    def _route(self, stage_id: str, envelope: Optional[Envelope],
               status: StageStatus) -> Optional[StageRouting]:
        if status != StageStatus.COMPLETED or envelope is None:
            return self._default_routing(stage_id, status)
        return {
            "triage_h0": self._route_triage_h0,
            "ecg": self._route_ecg,
            "cxr": self._route_cxr,
            "triage_h6": self._route_triage_h6,
            "echo": self._route_echo,
            "triage_h24": self._route_triage_h24,
        }[stage_id](envelope)

    @staticmethod
    def _default_routing(stage_id: str, status: StageStatus) -> Optional[StageRouting]:
        """A stage that could not run advances, it does not stop the pathway.

        The distinction that matters clinically: a stage which did not run has
        produced no evidence, and its silence must never be read as a negative
        finding. That is stated here rather than left implicit.
        """
        index = STAGE_ORDER.index(stage_id)
        following = STAGE_ORDER[index + 1] if index + 1 < len(STAGE_ORDER) else None
        word = ("no study was supplied" if status == StageStatus.NOT_SUPPLIED
                else "the component could not run")
        return StageRouting(
            branch="not_evaluated",
            statement="This stage produced no evidence because %s. Its absence is not "
                      "a negative finding and the pathway continues without it." % word,
            basis="Stage status %s." % status.value,
            next_stage=following,
            terminates=following is None,
            urgency=Urgency.ROUTINE)

    # -- stage 1 ------------------------------------------------------- #
    @staticmethod
    def _route_triage_h0(envelope: Envelope) -> StageRouting:
        raw = envelope.raw or {}
        band = str(raw.get("risk_level", "")).upper()
        p_acs = float(raw.get("p_acs") or 0.0)
        referred = bool(raw.get("referred"))
        basis = ("Component 04 at H=0: P(ACS)=%.3f, risk band %s, prediction %s%s."
                 % (p_acs, band or "unknown", raw.get("prediction", "?"),
                    ", clinician referral raised" if referred else ""))

        if band in _FAST_TRACK_EXIT_BANDS and not referred:
            return StageRouting(
                branch="non_cardiac",
                statement="Risk band %s: the chest-pain fast track is not entered. "
                          "Pursue other causes. This operating point is tuned to "
                          "negative predictive value (99.41 %%), which is what a "
                          "rule-out pathway needs." % band,
                basis=basis, next_stage=None, terminates=True, urgency=Urgency.ROUTINE,
                guideline="2021 AHA/ACC chest pain guideline -- initial evaluation.")

        if referred:
            return StageRouting(
                branch="referred_but_ecg_mandated",
                statement="The constrained decision layer declined to commit to a "
                          "subtype and raised a clinician referral. The ECG is "
                          "guideline-mandated within 10 minutes regardless, so the "
                          "pathway proceeds to it.",
                basis=basis, next_stage="ecg", urgency=Urgency.IMMEDIATE,
                guideline="2023 ESC ACS guideline -- ECG within 10 min of arrival.")

        return StageRouting(
            branch="fast_track",
            statement="Risk band %s: enters the chest-pain fast track. The single "
                      "most urgent act is now the ECG." % (band or "elevated"),
            basis=basis, next_stage="ecg", urgency=Urgency.IMMEDIATE,
            guideline="2023 ESC ACS guideline -- ECG within 10 min of arrival.")

    # -- stage 2 ------------------------------------------------------- #
    @staticmethod
    def _route_ecg(envelope: Envelope) -> StageRouting:
        raw = envelope.raw or {}
        zones = raw.get("zones") or {}
        electrode = raw.get("electrode") or {}
        scope = raw.get("scope") or {}
        basis = ("Component 02 zones %s; refused=%s; electrode.suspected=%s; "
                 "scope.outOfScope=%s."
                 % (zones or "none", bool(raw.get("refused")),
                    bool(electrode.get("suspected")), bool(scope.get("outOfScope"))))

        if raw.get("refused"):
            return StageRouting(
                branch="quality_refusal",
                statement="The record failed quality control, so no probabilities "
                          "exist for it. This is not a normal ECG -- it is an "
                          "uninterpretable one. Repeat the ECG. The pathway continues "
                          "to the radiograph, but the ECG question remains open.",
                basis=basis, next_stage="cxr", urgency=Urgency.IMMEDIATE,
                guideline="2023 ESC ACS guideline -- repeat ECG at 15-30 min intervals.")

        if zones.get("MI") == "rule_in":
            return StageRouting(
                branch="mi_rule_in",
                statement="Myocardial infarction ruled in at the conformal threshold. "
                          "The workup stops here and reperfusion is activated; no "
                          "radiograph, biomarker or echo may be allowed to delay it. "
                          "Note that Component 02's MI superclass does not separate "
                          "STEMI from NSTEMI -- confirm ST-elevation on the trace "
                          "before committing to primary PCI.",
                basis=basis, next_stage=None, terminates=True, urgency=Urgency.IMMEDIATE,
                guideline="Door-to-balloon <= 90 min; in-hospital mortality 8 % under "
                          "the target against 20 % over it.")

        if scope.get("outOfScope"):
            return StageRouting(
                branch="out_of_scope_rhythm",
                statement="An irregularly irregular R-R interval places this trace "
                          "outside the five-superclass label space. Arrhythmia has NOT "
                          "been excluded and the five-class result is not a complete "
                          "interpretation.",
                basis=basis, next_stage="cxr", urgency=Urgency.URGENT)

        if electrode.get("suspected"):
            return StageRouting(
                branch="electrode_reversal",
                statement="Limb-electrode reversal is suspected. Probabilities are "
                          "returned but their statistical guarantees are void. Repeat "
                          "the ECG with corrected lead placement.",
                basis=basis, next_stage="cxr", urgency=Urgency.URGENT)

        return StageRouting(
            branch="non_diagnostic",
            statement="No infarct pattern ruled in. Continue the workup, and repeat "
                      "the ECG at 15-30 minute intervals while symptoms persist.",
            basis=basis, next_stage="cxr", urgency=Urgency.ROUTINE,
            guideline="2023 ESC ACS guideline -- serial ECGs through the first hour.")

    # -- stage 3 ------------------------------------------------------- #
    @staticmethod
    def _route_cxr(envelope: Envelope) -> StageRouting:
        present = {f.name for f in envelope.findings if f.present}
        deferred = envelope.reliability.actionability == Actionability.DEFERRED
        basis = ("Component 01 positive findings: %s. Verdict %s."
                 % (", ".join(sorted(present)) or "none",
                    envelope.reliability.actionability.value))

        if _CRITICAL_MIMIC in present:
            return StageRouting(
                branch="critical_mimic",
                statement="Pneumothorax on the film. This is an immediate non-cardiac "
                          "killer requiring decompression; the acute coronary pathway "
                          "is not the answer here.",
                basis=basis, next_stage=None, terminates=True, urgency=Urgency.IMMEDIATE,
                guideline="Chest pain differential -- life-threatening non-cardiac causes.")

        structural = sorted(present & set(_STRUCTURAL_FINDINGS))
        if structural:
            return StageRouting(
                branch="structural_abnormality",
                statement="%s on the radiograph: the heart is structurally abnormal "
                          "and needs quantifying, so the pathway advances directly to "
                          "the echocardiogram." % " and ".join(structural),
                basis=basis, next_stage="echo", urgency=Urgency.URGENT,
                guideline="2021 AHA/ACC chest pain guideline -- Class 1 TTE for "
                          "ventricular function and wall-motion abnormality.")

        mimics = sorted(present & set(_MIMIC_FINDINGS))
        if mimics:
            return StageRouting(
                branch="mimic_flagged",
                statement="%s on the radiograph. Treat the mimic in parallel -- an "
                          "acute coronary syndrome may not be the answer. The pathway "
                          "continues to the biomarker rather than stopping, because "
                          "these findings and an ACS can coexist."
                          % " and ".join(mimics),
                basis=basis, next_stage="triage_h6", urgency=Urgency.URGENT)

        if deferred:
            return StageRouting(
                branch="film_deferred",
                statement="The radiograph fell inside the deferral margin and was not "
                          "called. Operating points are per projection -- the AP margin "
                          "is 0.2247 against PA's 0.0029 -- so a borderline portable "
                          "film is deferred rather than reported.",
                basis=basis, next_stage="triage_h6", urgency=Urgency.ROUTINE)

        return StageRouting(
            branch="no_mimic",
            statement="No non-cardiac killer and no structural abnormality identified. "
                      "Continue to the biomarker.",
            basis=basis, next_stage="triage_h6", urgency=Urgency.ROUTINE)

    # -- stage 4 ------------------------------------------------------- #
    @staticmethod
    def _route_triage_h6(envelope: Envelope) -> StageRouting:
        raw = envelope.raw or {}
        band = str(raw.get("risk_level", "")).upper()
        prediction = str(raw.get("prediction", "No_ACS"))
        p_acs = float(raw.get("p_acs") or 0.0)
        draws = raw.get("troponin_draws") or []
        basis = ("Component 04 at H=6: P(ACS)=%.3f, risk band %s, prediction %s; "
                 "%d troponin draw(s) visible at this horizon."
                 % (p_acs, band or "unknown", prediction, len(draws)))

        if prediction in ("NSTEMI", "STEMI"):
            return StageRouting(
                branch="rule_in",
                statement="Biomarker-supported rule-in. Admit for an invasive strategy, "
                          "and obtain an echocardiogram to assess ventricular function "
                          "and exclude mechanical complications.",
                basis=basis, next_stage="echo", urgency=Urgency.URGENT,
                guideline="ESC 0/1 h algorithm -- rule-in arm.")

        if prediction == "No_ACS" and band in _RULE_OUT_BANDS:
            return StageRouting(
                branch="rule_out",
                statement="Biomarker-supported rule-out. Discharge pathway, backed by "
                          "a measured negative predictive value of 99.41 %.",
                basis=basis, next_stage=None, terminates=True, urgency=Urgency.ROUTINE,
                guideline="ESC 0/1 h algorithm -- rule-out arm; resolves ~59 % of "
                          "patients between the two arms.")

        return StageRouting(
            branch="observe_zone",
            statement="Neither ruled in nor ruled out: the observe zone, where roughly "
                      "40 % of patients land. Guidelines recommend transthoracic echo "
                      "precisely for patients eligible for neither arm.",
            basis=basis, next_stage="echo", urgency=Urgency.URGENT,
            guideline="ESC 0/1 h algorithm -- observe zone; 2021 AHA/ACC Class 1 TTE.")

    # -- stage 5 ------------------------------------------------------- #
    @staticmethod
    def _route_echo(envelope: Envelope) -> StageRouting:
        ef = next((f for f in envelope.findings
                   if f.name == "Left-ventricular ejection fraction"), None)
        grade = next((f for f in envelope.findings if f.name == "Severity grade"), None)
        ef_value = ef.value if ef is not None else None
        interval = ef.interval if ef is not None else None
        basis = ("Component 03: EF %s %s, grade %s."
                 % ("%.1f %%" % ef_value if ef_value is not None else "unavailable",
                    "interval [%.1f, %.1f]" % tuple(interval) if interval else "",
                    grade.label if grade is not None else "unavailable"))

        if grade is not None and grade.label in _HFREF_GRADES:
            return StageRouting(
                branch="hfref",
                statement="Ejection fraction at or below 40 %% (%s) opens the "
                          "heart-failure pathway alongside the acute coronary one. "
                          "Component 03's Severe and Moderate bands sit entirely "
                          "inside HFrEF under the universal definition."
                          % (grade.label if grade else "reduced"),
                basis=basis, next_stage="triage_h24", urgency=Urgency.URGENT,
                guideline="JACC 2021 -- HFrEF <= 40 %, HFmrEF 41-49 %, HFpEF >= 50 %.")

        return StageRouting(
            branch="preserved_function",
            statement="Systolic function is not in the HFrEF range. Note that the Mild "
                      "band (40-55 %) straddles HFmrEF and HFpEF and does not by itself "
                      "separate those two guideline categories.",
            basis=basis, next_stage="triage_h24", urgency=Urgency.ROUTINE)

    # -- stage 6 ------------------------------------------------------- #
    @staticmethod
    def _route_triage_h24(envelope: Envelope) -> StageRouting:
        raw = envelope.raw or {}
        prediction = str(raw.get("prediction", "No_ACS"))
        referred = bool(raw.get("referred"))
        basis = ("Component 04 at H=24 (the deployment configuration): prediction %s, "
                 "P(ACS)=%.3f%s. Measured UA recall rises 37.3 %% -> 58.2 %% -> 80.0 %% "
                 "across H=0/6/24."
                 % (prediction, float(raw.get("p_acs") or 0.0),
                    ", clinician referral raised" if referred else ""))

        if referred:
            return StageRouting(
                branch="final_referral",
                statement="The top-two margin fell below the frozen cut-off, so the "
                          "component returns a clinician referral rather than a "
                          "subtype. Disposition is a clinical decision here, not a "
                          "model output.",
                basis=basis, next_stage=None, terminates=True, urgency=Urgency.URGENT)

        return StageRouting(
            branch="final_subtype_%s" % prediction.lower(),
            statement="Workup complete. Final subtype %s. Unstable angina is defined "
                      "by a normal troponin, so it is only separable from NSTEMI once "
                      "the biomarker has returned -- which is why this stage is last."
                      % prediction,
            basis=basis, next_stage=None, terminates=True,
            urgency=Urgency.URGENT if prediction != "No_ACS" else Urgency.ROUTINE)

    # ------------------------------------------------------------------ #
    #  post-traversal
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fill_unvisited(stages: Dict[str, PathwayStage], visited: List[str],
                        terminated_at: Optional[str]) -> None:
        """Mark stages the traversal never reached.

        `skipped` and `not_reached` are different clinical facts. Skipped means
        the pathway deliberately routed past a stage that was still open;
        not_reached means the pathway had already ended.
        """
        stop_index = STAGE_ORDER.index(terminated_at) if terminated_at else None
        for index, stage_id in enumerate(STAGE_ORDER):
            if stage_id in visited:
                continue
            spec = STAGE_SPEC[stage_id]
            if stop_index is not None and index > stop_index:
                status, detail = StageStatus.NOT_REACHED, (
                    "The pathway ended at an earlier stage, so this was never reached.")
            else:
                status, detail = StageStatus.SKIPPED, (
                    "Routing advanced past this stage.")
            stages[stage_id] = PathwayStage(
                id=stage_id, status=status, detail=detail, routing=None, result=None,
                **spec)

    @staticmethod
    def _for_observation(envelopes: Dict[str, Envelope]) -> Dict[str, Envelope]:
        """Re-key stage envelopes to the component ids the observer expects.

        The observer compares modalities, so Component 04 must appear once. The
        latest horizon that ran is the most informed, so it is the one used.
        """
        out: Dict[str, Envelope] = {}
        for stage_id in ("cxr", "ecg", "echo"):
            if stage_id in envelopes:
                out[stage_id] = envelopes[stage_id]
        for stage_id in ("triage_h24", "triage_h6", "triage_h0"):
            if stage_id in envelopes:
                out["triage"] = envelopes[stage_id]
                break
        return out

    @staticmethod
    def _aggregate_actionability(envelopes: Dict[str, Envelope]
                                 ) -> Tuple[Actionability, List[str]]:
        if not envelopes:
            return Actionability.UNAVAILABLE, [
                "No stage of the pathway produced a result."]
        worst = Actionability.worst(
            [env.reliability.actionability for env in envelopes.values()])
        reasons = ["[%s] %s" % (stage_id, reason)
                   for stage_id, env in envelopes.items()
                   for reason in env.reliability.reasons]
        return worst, reasons

    @staticmethod
    def _is_hfref(envelope: Envelope) -> bool:
        grade = next((f for f in envelope.findings if f.name == "Severity grade"), None)
        return grade is not None and grade.label in _HFREF_GRADES

    @staticmethod
    def _mimic_notes(envelope: Envelope) -> List[str]:
        present = {f.name for f in envelope.findings if f.present}
        return sorted(present & (set(_MIMIC_FINDINGS) | {_CRITICAL_MIMIC}))

    # ------------------------------------------------------------------ #
    def _disposition(self, stages: Dict[str, PathwayStage],
                     terminated_at: Optional[str], hf_pathway: bool,
                     mimic_notes: List[str],
                     envelopes: Dict[str, Envelope]) -> Disposition:
        """Where the traversal says the patient goes, and why."""
        final = stages[terminated_at] if terminated_at else None
        missing = [s.title for s in stages.values()
                   if s.status in (StageStatus.NOT_SUPPLIED, StageStatus.BLOCKED)]
        return self._disposition_from(
            final.routing if final else None, hf_pathway, mimic_notes, missing)

    def _disposition_from(self, final_routing: Optional[StageRouting],
                          hf_pathway: bool, mimic_notes: List[str],
                          missing: List[str]) -> Disposition:
        """The endpoint, derived from the terminating branch alone.

        Split out so the all-at-once traversal and the stage-by-stage one reach
        the same disposition from the same rule. The alternative -- letting the
        stepwise console decide where the patient goes -- would be a second
        implementation of this, free to disagree with the first.
        """
        rationale: List[str] = []
        branch = final_routing.branch if final_routing else None

        if branch == "mi_rule_in":
            return Disposition(
                destination="cath_lab", label="Cardiac catheterisation laboratory",
                urgency=Urgency.IMMEDIATE,
                time_target="Door-to-balloon <= 90 minutes",
                rationale=[final_routing.statement, final_routing.basis],
                heart_failure_pathway=hf_pathway)

        if branch == "critical_mimic":
            return Disposition(
                destination="non_cardiac", label="Non-cardiac emergency -- treat the mimic",
                urgency=Urgency.IMMEDIATE, time_target=None,
                rationale=[final_routing.statement, final_routing.basis],
                heart_failure_pathway=hf_pathway)

        if branch == "non_cardiac":
            return Disposition(
                destination="non_cardiac", label="Non-cardiac pathway",
                urgency=Urgency.ROUTINE, time_target=None,
                rationale=[final_routing.statement, final_routing.basis],
                heart_failure_pathway=False)

        if branch == "rule_out":
            return Disposition(
                destination="discharge", label="Discharge pathway",
                urgency=Urgency.ROUTINE, time_target=None,
                rationale=[final_routing.statement, final_routing.basis],
                heart_failure_pathway=hf_pathway)

        if branch == "final_referral":
            return Disposition(
                destination="indeterminate",
                label="Clinician decision required -- model declined to commit",
                urgency=Urgency.URGENT, time_target=None,
                rationale=[final_routing.statement, final_routing.basis],
                heart_failure_pathway=hf_pathway)

        if branch and branch.startswith("final_subtype_"):
            prediction = branch.removeprefix("final_subtype_").upper()
            destination, label, urgency = {
                "STEMI": ("cath_lab", "Cardiac catheterisation laboratory",
                          Urgency.IMMEDIATE),
                "NSTEMI": ("ccu", "Coronary care unit -- invasive strategy",
                           Urgency.URGENT),
                "UA": ("ward", "Admit -- unstable angina", Urgency.URGENT),
            }.get(prediction, ("discharge", "Discharge pathway", Urgency.ROUTINE))
            rationale = [final_routing.statement, final_routing.basis]
            if hf_pathway:
                rationale.append(
                    "A parallel heart-failure pathway is open: the echocardiogram "
                    "measured an ejection fraction inside the HFrEF range.")
            if mimic_notes:
                rationale.append(
                    "Radiographic findings requiring parallel treatment: %s."
                    % ", ".join(mimic_notes))
            return Disposition(
                destination=destination, label=label, urgency=urgency,
                time_target="Door-to-balloon <= 90 minutes"
                            if destination == "cath_lab" else None,
                rationale=rationale, heart_failure_pathway=hf_pathway)

        # The traversal ran out of supplied studies rather than reaching a
        # decision. Saying so plainly is the safe answer; inventing a
        # disposition from an incomplete workup is not.
        rationale.append(
            "The pathway did not reach a decision point. %s"
            % ("Stages without a result: %s." % ", ".join(missing) if missing
               else "No terminating branch was taken."))
        if hf_pathway:
            rationale.append(
                "The echocardiogram did measure an ejection fraction inside the "
                "HFrEF range, which stands independently of the ACS question.")
        if mimic_notes:
            rationale.append("Radiographic findings on record: %s." % ", ".join(mimic_notes))
        return Disposition(
            destination="indeterminate", label="Workup incomplete",
            urgency=Urgency.URGENT if (hf_pathway or mimic_notes) else Urgency.ROUTINE,
            time_target=None, rationale=rationale, heart_failure_pathway=hf_pathway)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _limits(stages: List[PathwayStage]) -> List[str]:
        limits = list(STANDING_LIMITS)
        absent = [s.title for s in stages
                  if s.status in (StageStatus.NOT_SUPPLIED, StageStatus.BLOCKED,
                                  StageStatus.NOT_REACHED)]
        if absent:
            limits.insert(0,
                          "This traversal is incomplete. The following stages produced "
                          "no evidence and their silence is not a negative finding: %s."
                          % ", ".join(absent))
        return limits
