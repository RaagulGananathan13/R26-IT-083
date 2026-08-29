"""
Tests for the clinical pathway engine.

These drive the traversal with synthetic component envelopes rather than real
weights, because what is under test is the *routing* -- which branch a given
result takes and where the pathway stops. That logic is where a mistake would
be clinically consequential and silent: a pathway that failed to stop on a
ruled-in MI would look entirely normal in the response.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pytest

from cvxai.core.errors import ComponentUnavailable
from cvxai.schemas.common import (
    Actionability,
    Envelope,
    Finding,
    ModelCard,
    Reliability,
)
from cvxai.schemas.pathway import StageStatus, Urgency
from cvxai.schemas.triage import TriageRequest
from cvxai.services.pathway import STAGE_ORDER, PathwayService


# --------------------------------------------------------------------------- #
#  synthetic envelopes
# --------------------------------------------------------------------------- #
def _card(component: str) -> ModelCard:
    return ModelCard(component_id=component, component_name=component, owner="test",
                     modality="test", task="test", dataset="test", architecture="test")


def envelope(component: str, *, findings: Optional[List[Finding]] = None,
             raw: Optional[Dict] = None,
             actionability: Actionability = Actionability.ACTIONABLE) -> Envelope:
    return Envelope(
        component=component, status="ok", headline="synthetic",
        findings=findings or [],
        reliability=Reliability(actionability=actionability, level="standard"),
        model=_card(component), raw=raw or {})


def triage_raw(prediction: str = "No_ACS", p_acs: float = 0.10,
               risk: str = "LOW", referred: bool = False, horizon: int = 0) -> Dict:
    return {"prediction": prediction, "p_acs": p_acs, "risk_level": risk,
            "referred": referred, "horizon_h": horizon, "troponin_draws": []}


class StubAdapter:
    def __init__(self, result):
        self._result = result

    def analyze(self, **_kwargs):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class StubRegistry:
    """Serves a pre-built envelope per component.

    Component 04 is called three times with different horizons, so its entry may
    be a list consumed in order -- which is how the H=0 / H=6 / H=24 branches are
    exercised independently.
    """

    def __init__(self, mapping: Dict):
        self._mapping = dict(mapping)

    def get(self, component_id: str):
        value = self._mapping.get(component_id)
        if value is None:
            raise ComponentUnavailable("%s is not configured in this stub." % component_id)
        if isinstance(value, list):
            if not value:
                raise ComponentUnavailable("%s stub exhausted." % component_id)
            return StubAdapter(value.pop(0))
        return StubAdapter(value)


class StubHorizons:
    """Resolves every horizon to the stub registry's triage adapter.

    Production resolves each horizon to its own adapter because Component 04
    has separate weights per horizon. These tests are about routing, not
    weights, so the horizon is irrelevant here -- what matters is that stage 1,
    stage 4 and stage 6 each get their own envelope, which the list form of
    StubRegistry provides.
    """

    def __init__(self, registry):
        self._registry = registry

    def get(self, _horizon):
        return self._registry.get("triage")


def service(registry) -> PathwayService:
    return PathwayService(registry, horizons=StubHorizons(registry))


def stages_by_id(response) -> Dict:
    return {stage.id: stage for stage in response.stages}


@pytest.fixture
def request_record() -> TriageRequest:
    return TriageRequest(chief_complaint="chest pain", age=61, sex="M")


# --------------------------------------------------------------------------- #
#  stage 1 -- the fast-track gate
# --------------------------------------------------------------------------- #
def test_minimal_risk_terminates_before_any_test(request_record):
    """A MINIMAL band must not enter the chest-pain fast track."""
    registry = StubRegistry({"triage": envelope("triage", raw=triage_raw(
        risk="MINIMAL", p_acs=0.02))})
    result = service(registry).run("p1", triage=request_record)

    assert result.terminated_at == "triage_h0"
    assert result.disposition.destination == "non_cardiac"
    stages = stages_by_id(result)
    assert stages["triage_h0"].status == StageStatus.COMPLETED
    # Everything downstream must be not_reached, never skipped: the pathway
    # ended, it did not route past open stages.
    for stage_id in STAGE_ORDER[1:]:
        assert stages[stage_id].status == StageStatus.NOT_REACHED


def test_low_risk_still_enters_the_fast_track(request_record):
    """LOW must not exit the pathway before the ECG.

    Measured on the curated demo records at H = 0, a genuine STEMI scores LOW
    (P(ACS) = 0.111), as do a genuine NSTEMI (0.053) and an unstable angina
    (0.109). The band boundary is 0.05 against a cohort prevalence of 5.6 %, so
    LOW spans "at the base rate" to "four times the base rate". Exiting there
    sends coronary syndromes home before the guideline-mandated ECG.
    """
    registry = StubRegistry({"triage": envelope("triage", raw=triage_raw(
        risk="LOW", p_acs=0.111, prediction="NSTEMI"))})
    result = service(registry).run("p1", triage=request_record)

    stages = stages_by_id(result)
    assert stages["triage_h0"].routing.branch == "fast_track"
    assert stages["triage_h0"].routing.next_stage == "ecg"
    assert result.terminated_at != "triage_h0"


def test_low_risk_still_supports_rule_out_after_the_biomarker(request_record):
    """The same band means something different once the troponin is back.

    Discharging before the biomarker and discharging after it rest on different
    evidence, so stage 4 keeps the wider rule-out set that stage 1 no longer has.
    """
    registry = StubRegistry({
        "triage": [
            envelope("triage", raw=triage_raw(risk="MODERATE", p_acs=0.30)),
            envelope("triage", raw=triage_raw(risk="LOW", p_acs=0.04, horizon=6)),
        ],
    })
    result = service(registry).run("p1", triage=request_record)
    assert stages_by_id(result)["triage_h6"].routing.branch == "rule_out"
    assert result.disposition.destination == "discharge"


def test_moderate_risk_enters_fast_track(request_record):
    registry = StubRegistry({"triage": envelope("triage", raw=triage_raw(
        risk="MODERATE", p_acs=0.35))})
    result = service(registry).run("p1", triage=request_record)

    stages = stages_by_id(result)
    assert stages["triage_h0"].routing.next_stage == "ecg"
    assert stages["triage_h0"].routing.urgency == Urgency.IMMEDIATE
    # No ECG supplied, so the stage is reached but contributes nothing.
    assert stages["ecg"].status == StageStatus.NOT_SUPPLIED


# --------------------------------------------------------------------------- #
#  stage 2 -- the branch that must stop everything
# --------------------------------------------------------------------------- #
def test_ruled_in_mi_terminates_the_pathway(request_record):
    """Nothing may be allowed to delay reperfusion."""
    registry = StubRegistry({
        "triage": envelope("triage", raw=triage_raw(risk="HIGH", p_acs=0.72)),
        "ecg": envelope("ecg", raw={"zones": {"MI": "rule_in", "NORM": "rule_out"}}),
    })
    result = service(registry).run(
        "p1", triage=request_record, ecg={"dat_bytes": b"", "hea_bytes": b""})

    assert result.terminated_at == "ecg"
    assert result.disposition.destination == "cath_lab"
    assert result.disposition.urgency == Urgency.IMMEDIATE
    assert "90 minutes" in (result.disposition.time_target or "")
    stages = stages_by_id(result)
    for stage_id in ("cxr", "triage_h6", "echo", "triage_h24"):
        assert stages[stage_id].status == StageStatus.NOT_REACHED


def test_refused_ecg_produces_no_probability_and_continues(request_record):
    """A refused ECG is uninterpretable, not normal."""
    registry = StubRegistry({
        "triage": envelope("triage", raw=triage_raw(risk="HIGH", p_acs=0.72)),
        "ecg": envelope("ecg", raw={"refused": True, "zones": {}},
                        actionability=Actionability.WITHHELD),
    })
    result = service(registry).run(
        "p1", triage=request_record, ecg={"dat_bytes": b"", "hea_bytes": b""})

    routing = stages_by_id(result)["ecg"].routing
    assert routing.branch == "quality_refusal"
    assert routing.next_stage == "cxr"
    assert not routing.terminates
    # The worst verdict must propagate: a withheld ECG makes the whole
    # traversal non-actionable.
    assert result.actionability == Actionability.WITHHELD


# --------------------------------------------------------------------------- #
#  stage 3 -- mimics and structural findings
# --------------------------------------------------------------------------- #
def test_pneumothorax_terminates_as_a_non_cardiac_emergency(request_record):
    registry = StubRegistry({
        "triage": envelope("triage", raw=triage_raw(risk="HIGH", p_acs=0.72)),
        "cxr": envelope("cxr", findings=[
            Finding(name="Pneumothorax", present=True, probability=0.91)]),
    })
    result = service(registry).run(
        "p1", triage=request_record, cxr={"image_bytes": b""})

    assert result.terminated_at == "cxr"
    assert result.disposition.destination == "non_cardiac"
    assert result.disposition.urgency == Urgency.IMMEDIATE


def test_cardiomegaly_routes_straight_to_the_echo(request_record):
    """A structurally abnormal heart needs quantifying; the biomarker waits."""
    registry = StubRegistry({
        "triage": envelope("triage", raw=triage_raw(risk="HIGH", p_acs=0.72)),
        "cxr": envelope("cxr", findings=[
            Finding(name="Cardiomegaly", present=True, probability=0.88),
            Finding(name="Pneumothorax", present=False, probability=0.02)]),
        "echo": envelope("echo", findings=[
            Finding(name="Left-ventricular ejection fraction", value=32.0, unit="%"),
            Finding(name="Severity grade", label="Moderate(30-40)")]),
    })
    result = service(registry).run(
        "p1", triage=request_record, cxr={"image_bytes": b""},
        echo={"video_bytes": b"", "filename": "x.avi"})

    stages = stages_by_id(result)
    assert stages["cxr"].routing.next_stage == "echo"
    # Skipped, not not_reached: the pathway routed past an open stage.
    assert stages["triage_h6"].status == StageStatus.SKIPPED
    assert stages["echo"].status == StageStatus.COMPLETED
    assert result.disposition.heart_failure_pathway is True


def test_effusion_flags_a_mimic_without_stopping(request_record):
    registry = StubRegistry({
        "triage": envelope("triage", raw=triage_raw(risk="HIGH", p_acs=0.72)),
        "cxr": envelope("cxr", findings=[
            Finding(name="Pleural Effusion", present=True, probability=0.77)]),
    })
    result = service(registry).run(
        "p1", triage=request_record, cxr={"image_bytes": b""})

    routing = stages_by_id(result)["cxr"].routing
    assert routing.branch == "mimic_flagged"
    assert routing.next_stage == "triage_h6"
    assert not routing.terminates


# --------------------------------------------------------------------------- #
#  stage 4 -- the 0/1 h arms
# --------------------------------------------------------------------------- #
def test_rule_out_at_h6_discharges(request_record):
    registry = StubRegistry({
        "triage": [
            envelope("triage", raw=triage_raw(risk="MODERATE", p_acs=0.30)),
            envelope("triage", raw=triage_raw(risk="LOW", p_acs=0.04, horizon=6)),
        ],
    })
    result = service(registry).run("p1", triage=request_record)

    assert result.terminated_at == "triage_h6"
    assert result.disposition.destination == "discharge"
    assert stages_by_id(result)["triage_h6"].routing.branch == "rule_out"


def test_observe_zone_routes_to_the_echo(request_record):
    registry = StubRegistry({
        "triage": [
            envelope("triage", raw=triage_raw(risk="MODERATE", p_acs=0.30)),
            envelope("triage", raw=triage_raw(risk="MODERATE", p_acs=0.31, horizon=6)),
            envelope("triage", raw=triage_raw(prediction="UA", risk="HIGH",
                                              p_acs=0.66, horizon=24)),
        ],
        "echo": envelope("echo", findings=[
            Finding(name="Left-ventricular ejection fraction", value=58.0, unit="%"),
            Finding(name="Severity grade", label="Normal(>=55)")]),
    })
    result = service(registry).run(
        "p1", triage=request_record, echo={"video_bytes": b"", "filename": "x.avi"})

    stages = stages_by_id(result)
    assert stages["triage_h6"].routing.branch == "observe_zone"
    assert stages["echo"].status == StageStatus.COMPLETED
    assert result.disposition.destination == "ward"       # UA
    assert result.disposition.heart_failure_pathway is False


# --------------------------------------------------------------------------- #
#  stage 6 -- disposition
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("prediction,destination", [
    ("STEMI", "cath_lab"),
    ("NSTEMI", "ccu"),
    ("UA", "ward"),
    ("No_ACS", "discharge"),
])
def test_final_subtype_maps_to_destination(request_record, prediction, destination):
    registry = StubRegistry({
        "triage": [
            envelope("triage", raw=triage_raw(risk="HIGH", p_acs=0.70)),
            envelope("triage", raw=triage_raw(risk="HIGH", p_acs=0.70, horizon=6)),
            envelope("triage", raw=triage_raw(prediction=prediction, risk="HIGH",
                                              p_acs=0.70, horizon=24)),
        ],
    })
    result = service(registry).run("p1", triage=request_record)
    assert result.disposition.destination == destination


def test_referral_at_h24_is_not_a_subtype(request_record):
    registry = StubRegistry({
        "triage": [
            envelope("triage", raw=triage_raw(risk="HIGH", p_acs=0.70)),
            envelope("triage", raw=triage_raw(risk="HIGH", p_acs=0.70, horizon=6)),
            envelope("triage", raw=triage_raw(prediction="NSTEMI", risk="HIGH",
                                              p_acs=0.55, referred=True, horizon=24),
                     actionability=Actionability.DEFERRED),
        ],
    })
    result = service(registry).run("p1", triage=request_record)
    assert result.disposition.destination == "indeterminate"
    assert result.actionability == Actionability.DEFERRED


# --------------------------------------------------------------------------- #
#  robustness
# --------------------------------------------------------------------------- #
def test_blocked_component_does_not_stop_the_traversal(request_record):
    """One dead component must not lose the other stages."""
    registry = StubRegistry({
        "triage": [
            envelope("triage", raw=triage_raw(risk="HIGH", p_acs=0.70)),
            envelope("triage", raw=triage_raw(risk="HIGH", p_acs=0.70, horizon=6)),
            envelope("triage", raw=triage_raw(prediction="NSTEMI", risk="HIGH",
                                              p_acs=0.70, horizon=24)),
        ],
        "ecg": ComponentUnavailable("Component 02 weights not present on this install."),
        "echo": envelope("echo", findings=[
            Finding(name="Left-ventricular ejection fraction", value=61.0, unit="%"),
            Finding(name="Severity grade", label="Normal(>=55)")]),
    })
    result = service(registry).run(
        "p1", triage=request_record, ecg={"dat_bytes": b"", "hea_bytes": b""},
        echo={"video_bytes": b"", "filename": "x.avi"})

    stages = stages_by_id(result)
    assert stages["ecg"].status == StageStatus.BLOCKED
    assert "weights not present" in (stages["ecg"].detail or "")
    # It advanced rather than terminating, and reached a real disposition.
    assert stages["ecg"].routing.next_stage == "cxr"
    assert result.disposition.destination == "ccu"


def test_absent_stages_are_never_read_as_negative_findings(request_record):
    """The limits block must name what produced no evidence."""
    registry = StubRegistry({
        "triage": [
            envelope("triage", raw=triage_raw(risk="HIGH", p_acs=0.70)),
            envelope("triage", raw=triage_raw(risk="HIGH", p_acs=0.70, horizon=6)),
            envelope("triage", raw=triage_raw(prediction="UA", risk="HIGH",
                                              p_acs=0.70, horizon=24)),
        ],
    })
    result = service(registry).run("p1", triage=request_record)

    joined = " ".join(result.limits)
    assert "not a negative finding" in joined
    assert "12-lead ECG" in joined and "Chest radiograph" in joined
    # The standing limits are always present, whatever ran.
    assert any("share no patients" in limit for limit in result.limits)


def test_every_stage_is_always_reported(request_record):
    """The response must carry all six stages whatever happened."""
    registry = StubRegistry({"triage": envelope("triage", raw=triage_raw(
        risk="MINIMAL", p_acs=0.01))})
    result = service(registry).run("p1", triage=request_record)

    assert [stage.id for stage in result.stages] == list(STAGE_ORDER)
    assert [stage.ordinal for stage in result.stages] == [1, 2, 3, 4, 5, 6]
    assert result.stages_total == 6
    assert result.stages_completed == 1


class TestStepwiseReportsSkippedStages:
    """A bypassed stage must be reported, not omitted.

    The all-at-once traversal fills every stage in, so `run()` has always said
    which ones were routed past. The stepwise traversal returned only the stage
    it just ran, and a stage with no record at all renders in the console as one
    that has simply not happened yet -- the pathway rail showed the troponin
    stage sitting at "T + 1-6 h" as though it were still to come, when the
    radiograph had deliberately routed past it to the echocardiogram.

    Deliberately bypassed and not-yet-run are different clinical facts, and a
    stepwise caller cannot tell them apart without reimplementing the stage
    ordering -- which is the one thing the opaque `context` exists to prevent.
    """

    @staticmethod
    def _between(current, following):
        from cvxai.services.pathway import PathwayService
        return [stage.id for stage in PathwayService._skipped_between(current, following)]

    def test_radiograph_to_echo_reports_the_troponin_stage(self):
        """The real case: cardiomegaly routes straight to the echo."""
        assert self._between("cxr", "echo") == ["triage_h6"]

    def test_adjacent_stages_skip_nothing(self):
        assert self._between("cxr", "triage_h6") == []
        assert self._between("triage_h0", "ecg") == []

    def test_termination_skips_nothing(self):
        """Ending is `not_reached`, which is a different fact and not this one."""
        assert self._between("ecg", None) == []

    def test_a_longer_jump_reports_every_stage_it_passed(self):
        assert self._between("triage_h0", "triage_h24") == [
            "ecg", "cxr", "triage_h6", "echo"]

    def test_skipped_stages_are_marked_and_explained(self):
        from cvxai.services.pathway import PathwayService
        skipped = PathwayService._skipped_between("cxr", "echo")
        assert len(skipped) == 1
        stage = skipped[0]
        assert stage.status == StageStatus.SKIPPED
        assert stage.detail
        assert stage.result is None and stage.routing is None
        # The spec fields must survive, or the console cannot label the stage.
        assert stage.ordinal == 4 and stage.title and stage.clock

    def test_the_response_actually_carries_them(self, request_record):
        """End to end through run_stage, not just the helper.

        Guards the wiring as well as the arithmetic: the field existing on the
        schema and the value being computed are two different things, and it was
        the second that was missing.
        """
        registry = StubRegistry({
            "triage": envelope("triage", raw=triage_raw(risk="HIGH", p_acs=0.62)),
            "ecg": envelope("ecg", raw={"zones": {"MI": "rule_out", "NORM": "rule_in"}}),
            "cxr": envelope("cxr", raw={"prediction": "Cardiomegaly", "probability": 0.93},
                            findings=[Finding(name="Cardiomegaly", present=True,
                                              probability=0.93)]),
        })
        svc = service(registry)

        # A stage with no study attached is never evaluated, so the routing
        # branch under test would not fire and the assertion below would pass
        # without exercising anything.
        studies = {
            "ecg": {"dat_bytes": b"x", "hea_bytes": b"y", "record_name": "r"},
            "cxr": {"image_bytes": b"x", "view": "PA"},
        }

        context, radiograph = None, None
        for stage_id in ("triage_h0", "ecg", "cxr"):
            kwargs = {stage_id: studies[stage_id]} if stage_id in studies else {}
            response = svc.run_stage(stage_id=stage_id, context=context,
                                     triage=request_record, **kwargs)
            context = response.context
            radiograph = response

        assert radiograph.stage.routing.branch == "structural_abnormality", (
            "the cardiomegaly branch did not fire, so this test proves nothing")
        assert radiograph.next_stage == "echo"
        assert [s.id for s in radiograph.skipped] == ["triage_h6"], (
            "the radiograph routed past the troponin stage; the response has to "
            "say so or the console shows that stage as still to come")
        assert radiograph.skipped[0].status == StageStatus.SKIPPED
