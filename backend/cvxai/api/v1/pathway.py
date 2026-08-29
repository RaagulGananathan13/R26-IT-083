"""
The clinical pathway endpoint.

Accepts the same multipart payload as `/assessment` and walks it through the
six documented stages instead of running them in parallel. The difference is
gating: here a stage's result decides whether the next stage happens at all.

`GET /pathway/definition` returns the static stage map, so a client can render
the pathway before any study has been uploaded.
"""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile
from starlette.concurrency import run_in_threadpool

from cvxai.api.deps import collect_modalities, registry
from cvxai.core.errors import InvalidInput
from cvxai.core.registry import ComponentRegistry
from cvxai.schemas.pathway import (
    PathwayContext,
    PathwayDefinition,
    PathwayResponse,
    StageRunResponse,
)
from cvxai.services.pathway import STAGE_ORDER, STAGE_SPEC, PathwayService

router = APIRouter(tags=["pathway"])

#: Guideline sources justifying the ordering. Served with the definition so a
#: client never has to hard-code them alongside the stage map.
REFERENCES = [
    {"id": "esc-acs-2023",
     "title": "2023 ESC Guidelines for the management of acute coronary syndromes",
     "journal": "European Heart Journal",
     "url": "https://academic.oup.com/eurheartj/article/44/38/3720/7243210",
     "supports": "12-lead ECG within 10 minutes of arrival; serial ECGs at 15-30 min."},
    {"id": "aha-chest-pain-2021",
     "title": "2021 AHA/ACC Guideline for the Evaluation and Diagnosis of Chest Pain",
     "journal": "Circulation",
     "url": "https://www.ahajournals.org/doi/10.1161/CIR.0000000000001029",
     "supports": "Initial evaluation goals; Class 1 TTE for ventricular function and "
                 "wall-motion abnormality."},
    {"id": "observe-zone",
     "title": "Novel Criteria for the Observe-Zone of the ESC 0/1h-hs-cTnT Algorithm",
     "journal": "Circulation",
     "url": "https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.120.052982",
     "supports": "Rule-out / rule-in / observe zone; roughly 40 % observe."},
    {"id": "jacc-0-1h",
     "title": "Prospective Validation of the 0/1-h Algorithm for Early Diagnosis of "
              "Myocardial Infarction",
     "journal": "JACC",
     "url": "https://www.jacc.org/doi/10.1016/j.jacc.2018.05.040",
     "supports": "The 0/1 h troponin algorithm."},
    {"id": "jacc-hf-ef",
     "title": "Classification of Heart Failure According to Ejection Fraction",
     "journal": "JACC",
     "url": "https://www.jacc.org/doi/10.1016/j.jacc.2021.04.070",
     "supports": "HFrEF <= 40 %, HFmrEF 41-49 %, HFpEF >= 50 %."},
    {"id": "door-to-balloon",
     "title": "Achieving door-to-balloon time <= 90 minutes in STEMI",
     "journal": "PMC",
     "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC13094172/",
     "supports": "Reperfusion time target and its mortality difference."},
]


@router.get("/pathway/definition", response_model=PathwayDefinition,
            summary="The static six-stage pathway map")
def pathway_definition() -> PathwayDefinition:
    """The stage map, independent of any patient.

    Served rather than duplicated client-side so the console and the engine can
    never disagree about what the pathway is.
    """
    return PathwayDefinition(
        stages=[dict(id=stage_id, **STAGE_SPEC[stage_id]) for stage_id in STAGE_ORDER],
        references=REFERENCES,
    )


@router.post("/pathway", response_model=PathwayResponse,
             summary="Walk one patient through the six-stage clinical pathway")
async def pathway(
    patient_id: str = Form("anonymous",
                           description="Local identifier. Never send a real MRN."),
    cxr_file: Optional[UploadFile] = File(None, description="Chest radiograph."),
    cxr_view: Optional[str] = Form(None, description="AP or PA."),
    ecg_dat_file: Optional[UploadFile] = File(None, description="WFDB .dat."),
    ecg_hea_file: Optional[UploadFile] = File(None, description="WFDB .hea."),
    echo_file: Optional[UploadFile] = File(None, description="Echo video or .npy clip."),
    triage_json: Optional[str] = Form(
        None,
        description="JSON object matching the TriageRequest schema. Any `horizon` "
                    "field is ignored: the pathway serves this record at H=0, H=6 "
                    "and H=24 in turn, which is the point of the traversal."),
    reg: ComponentRegistry = Depends(registry),
) -> PathwayResponse:
    """Run the stages in clinical order, letting each one gate the next.

    Supplying a partial set of studies is the ordinary case, not a degraded one.
    A stage with no study is recorded as `not_supplied` and the traversal
    continues; its silence is never read as a negative finding.
    """
    cxr_kwargs, ecg_kwargs, echo_kwargs, triage_request = await collect_modalities(
        cxr_file=cxr_file, cxr_view=cxr_view,
        ecg_dat_file=ecg_dat_file, ecg_hea_file=ecg_hea_file,
        echo_file=echo_file, triage_json=triage_json)

    if triage_request is None:
        # Every branch out of stage 1 is driven by Component 04's risk band, and
        # stages 4 and 6 are Component 04 as well. Without the record the
        # traversal has no spine and would report five empty stages, which
        # looks like a result but is not one.
        raise InvalidInput(
            "The pathway is driven by the emergency-department record: it is stage 1, "
            "stage 4 and stage 6. Supply triage_json. Imaging and waveform studies are "
            "optional and their stages are skipped when absent.")

    service = PathwayService(reg)
    return await run_in_threadpool(
        service.run, patient_id=patient_id, cxr=cxr_kwargs, ecg=ecg_kwargs,
        echo=echo_kwargs, triage=triage_request)


@router.post("/pathway/stage", response_model=StageRunResponse,
             summary="Run one stage of the pathway and report where it leads")
async def pathway_stage(
    stage_id: str = Form(..., description="triage_h0 | ecg | cxr | triage_h6 | echo | triage_h24"),
    context_json: Optional[str] = Form(
        None,
        description="The `context` returned by the previous stage, passed back "
                    "unmodified. Omit on the first stage."),
    triage_json: Optional[str] = Form(
        None,
        description="The ED record. Required for the three Component 04 stages; "
                    "the horizon is set by the stage, not by this field."),
    cxr_file: Optional[UploadFile] = File(None),
    cxr_view: Optional[str] = Form(None, description="AP or PA."),
    ecg_dat_file: Optional[UploadFile] = File(None),
    ecg_hea_file: Optional[UploadFile] = File(None),
    echo_file: Optional[UploadFile] = File(None),
    reg: ComponentRegistry = Depends(registry),
) -> StageRunResponse:
    """Advance the pathway by exactly one stage.

    For walking the pathway as the studies arrive, rather than all at once. The
    caller supplies the study for this stage and hands back the `context` from
    the previous response; the routing decision, and therefore which stage comes
    next, is made here.

    `context` is opaque to the caller by design. Reading it to decide what to do
    next would put a second copy of the routing rules in the client.
    """
    context: Optional[PathwayContext] = None
    if context_json:
        try:
            context = PathwayContext.model_validate(json.loads(context_json))
        except json.JSONDecodeError as exc:
            raise InvalidInput("context_json is not valid JSON: %s" % exc) from exc
        except ValueError as exc:
            raise InvalidInput("context_json failed validation: %s" % exc) from exc

    cxr_kwargs, ecg_kwargs, echo_kwargs, triage_request = await collect_modalities(
        cxr_file=cxr_file, cxr_view=cxr_view,
        ecg_dat_file=ecg_dat_file, ecg_hea_file=ecg_hea_file,
        echo_file=echo_file, triage_json=triage_json)

    service = PathwayService(reg)
    return await run_in_threadpool(
        service.run_stage, stage_id=stage_id, context=context,
        cxr=cxr_kwargs, ecg=ecg_kwargs, echo=echo_kwargs, triage=triage_request)
