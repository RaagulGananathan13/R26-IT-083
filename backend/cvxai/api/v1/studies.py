"""
The four single-modality endpoints.

Each route validates the upload, hands it to the component's adapter and
returns the shared envelope. Handlers are declared `def`, not `async def`, so
FastAPI runs them in its worker threadpool -- the components are synchronous,
GPU-bound and internally serialised, and blocking the event loop with them
would stall `/health` for every other caller.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile
from starlette.concurrency import run_in_threadpool

from cvxai.api.deps import IMAGE_SUFFIXES, VIDEO_SUFFIXES, read_upload, registry
from cvxai.core.errors import InvalidInput
from cvxai.core.registry import ComponentRegistry
from cvxai.schemas.common import Envelope
from cvxai.schemas.triage import ExtractionReport, TriagePdfResponse, TriageRequest
from cvxai.services.pdf_triage import extract_triage_record

router = APIRouter(tags=["studies"])

# Re-exported: the accepted-suffix tuples live in deps so every route that takes
# an upload agrees on them. Kept importable from here for existing callers.
__all__ = ["router", "IMAGE_SUFFIXES", "VIDEO_SUFFIXES"]


@router.post("/cxr/analyze", response_model=Envelope,
             summary="Component 01 -- chest radiograph")
async def analyze_cxr(
    file: UploadFile = File(..., description="Frontal chest radiograph."),
    view: Optional[str] = Form(
        None,
        description="AP or PA. Omit if unknown -- the global operating point is then "
                    "used. Guessing PA on a bedside film would apply the stricter "
                    "threshold to the patients least able to tolerate a missed "
                    "cardiomegaly."),
    reg: ComponentRegistry = Depends(registry),
) -> Envelope:
    data = await read_upload(file, "file", allowed_suffixes=IMAGE_SUFFIXES)
    adapter = reg.get("cxr")
    return await run_in_threadpool(
        adapter.analyze, image_bytes=data, view=view, filename=file.filename)


@router.post("/ecg/analyze", response_model=Envelope,
             summary="Component 02 -- 12-lead ECG")
async def analyze_ecg(
    dat_file: UploadFile = File(..., description="WFDB signal file (.dat)."),
    hea_file: UploadFile = File(..., description="WFDB header file (.hea)."),
    with_xai: bool = Form(
        True, description="Compute Grad-CAM and integrated gradients. Disabling it "
                          "roughly halves latency."),
    reg: ComponentRegistry = Depends(registry),
) -> Envelope:
    dat = await read_upload(dat_file, "dat_file", allowed_suffixes=(".dat",))
    hea = await read_upload(hea_file, "hea_file", allowed_suffixes=(".hea",))

    dat_stem = Path(dat_file.filename or "").stem
    hea_stem = Path(hea_file.filename or "").stem
    if dat_stem != hea_stem:
        raise InvalidInput(
            "The .dat and .hea files must share a base name (%r vs %r); WFDB resolves "
            "the signal file through the header." % (dat_stem, hea_stem))

    adapter = reg.get("ecg")
    return await run_in_threadpool(
        adapter.analyze, dat_bytes=dat, hea_bytes=hea,
        record_name=dat_stem or "upload", with_xai=with_xai)


@router.post("/echo/analyze", response_model=Envelope,
             summary="Component 03 -- echocardiogram video")
async def analyze_echo(
    file: UploadFile = File(
        ..., description="Apical four-chamber video, or a cached .npy clip array."),
    reg: ComponentRegistry = Depends(registry),
) -> Envelope:
    data = await read_upload(file, "file", allowed_suffixes=VIDEO_SUFFIXES)
    adapter = reg.get("echo")
    return await run_in_threadpool(
        adapter.analyze, video_bytes=data, filename=file.filename or "upload.avi")


@router.post("/triage/analyze-pdf", response_model=TriagePdfResponse,
             summary="Component 04 -- ED triage record from a PDF")
async def analyze_triage_pdf(
    file: UploadFile = File(..., description="ED record as a PDF with a text layer."),
    reg: ComponentRegistry = Depends(registry),
) -> TriagePdfResponse:
    """Parse an ED record, then predict from what was parsed.

    The response carries three separable parts -- what the parser found, the
    record actually submitted, and the model's answer -- because a parser that
    silently misses a troponin produces a confident wrong answer with no error
    anywhere. Extraction is a regex-and-lexicon parser over the text layer, not
    a document AI: it handles structured ED summaries, and reports what it
    could not read rather than guessing.
    """
    data = await read_upload(file, "file", allowed_suffixes=(".pdf",))

    extraction = await run_in_threadpool(extract_triage_record, data)
    try:
        request = TriageRequest.model_validate(extraction.fields)
    except ValueError as exc:
        raise InvalidInput(
            "The record extracted from this PDF is not internally consistent: %s" % exc,
            {"extracted": extraction.fields}) from exc

    adapter = reg.get("triage")
    envelope = await run_in_threadpool(adapter.analyze, request=request)
    return TriagePdfResponse(
        extraction=ExtractionReport.model_validate(extraction.to_dict()),
        request=request,
        result=envelope,
    )


@router.post("/triage/analyze", response_model=Envelope,
             summary="Component 04 -- ED triage record")
async def analyze_triage(
    request: TriageRequest,
    reg: ComponentRegistry = Depends(registry),
) -> Envelope:
    """Every field is optional.

    The component uses missingness-aware encoding: an untested biomarker is the
    clinical fact that nobody ordered the test, not a number to impute. Sending
    a partial record is the intended usage, not a degraded one.
    """
    adapter = reg.get("triage")
    return await run_in_threadpool(adapter.analyze, request=request)
