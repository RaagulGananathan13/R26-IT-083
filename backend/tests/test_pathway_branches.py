"""
Every routing branch, exercised once.

`test_pathway.py` covers the traversal -- where the pathway stops, what the
disposition becomes, how a blocked component is handled. This covers the
routing table itself: that each documented branch in `CLINICAL_WORKFLOW.md` is
reachable, produces the branch id it claims, and terminates or advances to the
stage the document says it should.

The distinction matters. A branch can be individually correct and still be
unreachable because an earlier condition shadows it -- `electrode_reversal`
sits behind `mi_rule_in` and `out_of_scope_rhythm`, and a reordering that made
it dead code would not fail any traversal test.
"""
from __future__ import annotations

import pytest

from cvxai.schemas.common import (
    Actionability,
    Envelope,
    Finding,
    ModelCard,
    Reliability,
)
from cvxai.services.pathway import PathwayService

_CARD = ModelCard(component_id="x", component_name="x", owner="x", modality="x",
                  task="x", dataset="x", architecture="x")


def envelope(component, raw=None, findings=None,
             actionability=Actionability.ACTIONABLE) -> Envelope:
    return Envelope(
        component=component, status="ok", headline="h", findings=findings or [],
        reliability=Reliability(actionability=actionability, level="l"),
        model=_CARD, raw=raw or {})


def triage_raw(**overrides):
    base = {"risk_level": "HIGH", "p_acs": 0.7, "prediction": "No_ACS", "referred": False}
    base.update(overrides)
    return base


def grade(label: str) -> Finding:
    return Finding(name="Severity grade", label=label)


#: (stage, expected branch, terminates, next stage, envelope factory)
CASES = [
    # -- stage 1, the fast-track gate ---------------------------------------
    ("triage_h0", "non_cardiac", True, None,
     lambda: envelope("triage", triage_raw(risk_level="MINIMAL", p_acs=0.01))),
    ("triage_h0", "fast_track", False, "ecg",
     lambda: envelope("triage", triage_raw())),
    ("triage_h0", "referred_but_ecg_mandated", False, "ecg",
     lambda: envelope("triage", triage_raw(referred=True))),

    # -- stage 2, the branch that must stop everything ----------------------
    ("ecg", "quality_refusal", False, "cxr",
     lambda: envelope("ecg", {"refused": True})),
    ("ecg", "mi_rule_in", True, None,
     lambda: envelope("ecg", {"zones": {"MI": "rule_in"}})),
    ("ecg", "out_of_scope_rhythm", False, "cxr",
     lambda: envelope("ecg", {"zones": {"MI": "refer"}, "scope": {"outOfScope": True}})),
    ("ecg", "electrode_reversal", False, "cxr",
     lambda: envelope("ecg", {"zones": {"MI": "refer"}, "electrode": {"suspected": True}})),
    ("ecg", "non_diagnostic", False, "cxr",
     lambda: envelope("ecg", {"zones": {"MI": "rule_out"}})),

    # -- stage 3, mimics and structural findings ----------------------------
    ("cxr", "critical_mimic", True, None,
     lambda: envelope("cxr", findings=[Finding(name="Pneumothorax", present=True)])),
    ("cxr", "structural_abnormality", False, "echo",
     lambda: envelope("cxr", findings=[Finding(name="Cardiomegaly", present=True)])),
    ("cxr", "mimic_flagged", False, "triage_h6",
     lambda: envelope("cxr", findings=[Finding(name="Consolidation", present=True)])),
    ("cxr", "film_deferred", False, "triage_h6",
     lambda: envelope("cxr", findings=[Finding(name="Cardiomegaly", present=False)],
                      actionability=Actionability.DEFERRED)),
    ("cxr", "no_mimic", False, "triage_h6",
     lambda: envelope("cxr", findings=[Finding(name="Cardiomegaly", present=False)])),

    # -- stage 4, the ESC 0/1 h arms ----------------------------------------
    ("triage_h6", "rule_in", False, "echo",
     lambda: envelope("triage", triage_raw(prediction="NSTEMI"))),
    ("triage_h6", "rule_out", True, None,
     lambda: envelope("triage", triage_raw(risk_level="LOW", prediction="No_ACS"))),
    ("triage_h6", "observe_zone", False, "echo",
     lambda: envelope("triage", triage_raw(risk_level="MODERATE"))),

    # -- stage 5, systolic function -----------------------------------------
    ("echo", "hfref", False, "triage_h24",
     lambda: envelope("echo", findings=[grade("Severe(<30)")])),
    ("echo", "preserved_function", False, "triage_h24",
     lambda: envelope("echo", findings=[grade("Normal(>=55)")])),

    # -- stage 6, disposition -----------------------------------------------
    ("triage_h24", "final_referral", True, None,
     lambda: envelope("triage", triage_raw(referred=True))),
    ("triage_h24", "final_subtype_stemi", True, None,
     lambda: envelope("triage", triage_raw(prediction="STEMI"))),
]

_ROUTERS = {
    "triage_h0": PathwayService._route_triage_h0,
    "ecg": PathwayService._route_ecg,
    "cxr": PathwayService._route_cxr,
    "triage_h6": PathwayService._route_triage_h6,
    "echo": PathwayService._route_echo,
    "triage_h24": PathwayService._route_triage_h24,
}


@pytest.mark.parametrize(
    "stage,branch,terminates,next_stage,build",
    CASES,
    ids=[f"{stage}:{branch}" for stage, branch, _t, _n, _b in CASES],
)
def test_branch(stage, branch, terminates, next_stage, build):
    routing = _ROUTERS[stage](build())

    assert routing.branch == branch
    assert routing.terminates is terminates
    assert routing.next_stage == next_stage
    # Every hop must name the values it rests on. A routing decision a reader
    # cannot check against the payload is one that gets trusted for the wrong
    # reasons.
    assert routing.basis.strip()
    assert routing.statement.strip()


def test_every_implemented_branch_is_covered():
    """The table above must not fall behind the router.

    Guards against a branch being added to the engine and silently never
    exercised, which is how a routing rule ends up shipping untested.
    """
    import re
    from pathlib import Path

    source = Path(PathwayService.__module__.replace(".", "/") + ".py")
    if not source.exists():  # installed rather than in-tree
        import cvxai.services.pathway as module
        source = Path(module.__file__)

    implemented = set(re.findall(r'branch="([a-z_0-9]+)"', source.read_text(encoding="utf-8")))
    implemented.discard("not_evaluated")  # the no-result path, covered elsewhere
    covered = {branch for _s, branch, _t, _n, _b in CASES}
    # final_subtype_* is built by interpolation; one representative is enough.
    implemented = {b for b in implemented if not b.startswith("final_subtype")}
    covered = {b for b in covered if not b.startswith("final_subtype")}

    assert implemented <= covered, "branches with no test: %s" % sorted(implemented - covered)
