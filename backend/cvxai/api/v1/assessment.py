"""
The multi-modal endpoint.

Accepts any subset of the four modalities for one patient in a single
multipart request and returns each component's result plus an aggregated
verdict. Modalities that are absent, unavailable or failing are reported in
`skipped` rather than failing the request -- a chest film should still be read
when the echo loop is down.

This endpoint is deliberately order-free: four independent readings, reduced to
their worst verdict. When the *order* is what carries the clinical meaning --
where one result can make the next test irrelevant -- use `/pathway`, which
walks the same payload through the six gated stages of `CLINICAL_WORKFLOW.md`.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile
from starlette.concurrency import run_in_threadpool

from cvxai.api.deps import collect_modalities, registry
from cvxai.core.errors import InvalidInput
from cvxai.core.registry import ComponentRegistry
from cvxai.schemas.assessment import AssessmentResponse
from cvxai.services.assessment import AssessmentService

router = APIRouter(tags=["assessment"])


@router.post("/assessment", response_model=AssessmentResponse,
             summary="Run every supplied modality for one patient and aggregate")
async def assessment(
    patient_id: str = Form("anonymous",
                           description="Local identifier. Never send a real MRN."),
    cxr_file: Optional[UploadFile] = File(None, description="Chest radiograph."),
    cxr_view: Optional[str] = Form(None, description="AP or PA."),
    ecg_dat_file: Optional[UploadFile] = File(None, description="WFDB .dat."),
    ecg_hea_file: Optional[UploadFile] = File(None, description="WFDB .hea."),
    echo_file: Optional[UploadFile] = File(None, description="Echo video or .npy clip."),
    triage_json: Optional[str] = Form(
        None, description="JSON object matching the TriageRequest schema."),
    reg: ComponentRegistry = Depends(registry),
) -> AssessmentResponse:
    cxr_kwargs, ecg_kwargs, echo_kwargs, triage_request = await collect_modalities(
        cxr_file=cxr_file, cxr_view=cxr_view,
        ecg_dat_file=ecg_dat_file, ecg_hea_file=ecg_hea_file,
        echo_file=echo_file, triage_json=triage_json)

    if not any((cxr_kwargs, ecg_kwargs, echo_kwargs, triage_request)):
        raise InvalidInput(
            "Supply at least one modality: cxr_file, ecg_dat_file + ecg_hea_file, "
            "echo_file, or triage_json.")

    service = AssessmentService(reg)
    return await run_in_threadpool(
        service.run, patient_id=patient_id, cxr=cxr_kwargs, ecg=ecg_kwargs,
        echo=echo_kwargs, triage=triage_request)
