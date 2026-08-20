"""
The multi-modal endpoint.

Accepts any subset of the four modalities for one patient in a single
multipart request and returns each component's result plus an aggregated
verdict. Modalities that are absent, unavailable or failing are reported in
`skipped` rather than failing the request -- a chest film should still be read
when the echo loop is down.
"""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile
from starlette.concurrency import run_in_threadpool

from cvxai.api.deps import read_upload, registry
from cvxai.api.v1.studies import IMAGE_SUFFIXES, VIDEO_SUFFIXES
from cvxai.core.errors import InvalidInput
from cvxai.core.registry import ComponentRegistry
from cvxai.schemas.assessment import AssessmentResponse
from cvxai.schemas.triage import TriageRequest
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
    cxr_kwargs = None
    if cxr_file is not None:
        data = await read_upload(cxr_file, "cxr_file", allowed_suffixes=IMAGE_SUFFIXES)
        cxr_kwargs = {"image_bytes": data, "view": cxr_view,
                      "filename": cxr_file.filename}

    ecg_kwargs = None
    if ecg_dat_file is not None or ecg_hea_file is not None:
        if ecg_dat_file is None or ecg_hea_file is None:
            raise InvalidInput(
                "An ECG needs both files; received only the "
                + ("header" if ecg_dat_file is None else "signal") + ".")
        from pathlib import Path

        dat = await read_upload(ecg_dat_file, "ecg_dat_file", allowed_suffixes=(".dat",))
        hea = await read_upload(ecg_hea_file, "ecg_hea_file", allowed_suffixes=(".hea",))
        dat_stem = Path(ecg_dat_file.filename or "").stem
        hea_stem = Path(ecg_hea_file.filename or "").stem
        if dat_stem != hea_stem:
            raise InvalidInput(
                "The .dat and .hea files must share a base name (%r vs %r)."
                % (dat_stem, hea_stem))
        ecg_kwargs = {"dat_bytes": dat, "hea_bytes": hea,
                      "record_name": dat_stem or "upload", "with_xai": True}

    echo_kwargs = None
    if echo_file is not None:
        data = await read_upload(echo_file, "echo_file", allowed_suffixes=VIDEO_SUFFIXES)
        echo_kwargs = {"video_bytes": data, "filename": echo_file.filename or "upload.avi"}

    triage_request = None
    if triage_json:
        try:
            triage_request = TriageRequest.model_validate(json.loads(triage_json))
        except json.JSONDecodeError as exc:
            raise InvalidInput("triage_json is not valid JSON: %s" % exc) from exc
        except ValueError as exc:
            raise InvalidInput("triage_json failed validation: %s" % exc) from exc

    if not any((cxr_kwargs, ecg_kwargs, echo_kwargs, triage_request)):
        raise InvalidInput(
            "Supply at least one modality: cxr_file, ecg_dat_file + ecg_hea_file, "
            "echo_file, or triage_json.")

    service = AssessmentService(reg)
    return await run_in_threadpool(
        service.run, patient_id=patient_id, cxr=cxr_kwargs, ecg=ecg_kwargs,
        echo=echo_kwargs, triage=triage_request)
