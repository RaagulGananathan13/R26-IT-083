"""Shared dependencies and upload helpers for the route modules."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from fastapi import Request, UploadFile

from cvxai.core.errors import InvalidInput, PayloadTooLarge
from cvxai.core.registry import ComponentRegistry, get_registry
from cvxai.schemas.triage import TriageRequest
from cvxai.settings import Settings, get_settings

#: Accepted upload extensions. They live here rather than in a route module
#: because more than one route needs them, and a second copy is how two
#: endpoints quietly start accepting different things.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
VIDEO_SUFFIXES = (".avi", ".mp4", ".mov", ".mkv", ".webm", ".npy")


def registry(request: Request) -> ComponentRegistry:
    """The registry attached to the app at startup, or the process singleton."""
    existing = getattr(request.app.state, "registry", None)
    return existing if existing is not None else get_registry()


def settings() -> Settings:
    return get_settings()


async def read_upload(
    upload: Optional[UploadFile],
    field: str,
    *,
    required: bool = True,
    max_bytes: Optional[int] = None,
    allowed_suffixes: Optional[tuple] = None,
) -> bytes:
    """Read an uploaded file with size and extension checks.

    The size check happens after the read because Starlette streams uploads to a
    spooled temporary file; the limit is enforced here rather than trusting a
    client-supplied Content-Length.
    """
    if upload is None:
        if required:
            raise InvalidInput("Missing required file field %r." % field)
        return b""

    if allowed_suffixes:
        name = (upload.filename or "").lower()
        if not name.endswith(tuple(allowed_suffixes)):
            raise InvalidInput(
                "%r must be one of: %s (received %r)."
                % (field, ", ".join(allowed_suffixes), upload.filename),
                {"accepted": list(allowed_suffixes)})

    data = await upload.read()
    await upload.close()

    if not data:
        if required:
            raise InvalidInput("Uploaded file %r is empty." % field)
        return b""

    limit = max_bytes if max_bytes is not None else get_settings().max_upload_bytes
    if len(data) > limit:
        raise PayloadTooLarge(
            "%r is %.1f MB; the limit is %.0f MB."
            % (field, len(data) / 1e6, limit / 1e6),
            {"limit_bytes": limit})
    return data


async def collect_modalities(
    *,
    cxr_file: Optional[UploadFile] = None,
    cxr_view: Optional[str] = None,
    ecg_dat_file: Optional[UploadFile] = None,
    ecg_hea_file: Optional[UploadFile] = None,
    echo_file: Optional[UploadFile] = None,
    triage_json: Optional[str] = None,
) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict], Optional[TriageRequest]]:
    """Parse the shared multi-modality multipart body into adapter kwargs.

    `/assessment` and `/pathway` accept the same payload and differ only in what
    they do with it. Parsing it once here keeps the two endpoints from drifting
    apart in what they will accept -- which is the sort of divergence that shows
    up as a file type working on one route and not the other.

    Returns `(cxr, ecg, echo, triage)`, each None when that modality was absent.
    Callers decide which combinations are acceptable; this function only refuses
    inputs that are internally inconsistent.
    """
    cxr_kwargs: Optional[Dict] = None
    if cxr_file is not None:
        data = await read_upload(cxr_file, "cxr_file", allowed_suffixes=IMAGE_SUFFIXES)
        cxr_kwargs = {"image_bytes": data, "view": cxr_view,
                      "filename": cxr_file.filename}

    ecg_kwargs: Optional[Dict] = None
    if ecg_dat_file is not None or ecg_hea_file is not None:
        if ecg_dat_file is None or ecg_hea_file is None:
            raise InvalidInput(
                "An ECG needs both files; received only the "
                + ("header" if ecg_dat_file is None else "signal") + ".")
        dat = await read_upload(ecg_dat_file, "ecg_dat_file", allowed_suffixes=(".dat",))
        hea = await read_upload(ecg_hea_file, "ecg_hea_file", allowed_suffixes=(".hea",))
        dat_stem = Path(ecg_dat_file.filename or "").stem
        hea_stem = Path(ecg_hea_file.filename or "").stem
        if dat_stem != hea_stem:
            # WFDB resolves the signal file through the header, so mismatched
            # stems silently read the wrong record rather than failing.
            raise InvalidInput(
                "The .dat and .hea files must share a base name (%r vs %r)."
                % (dat_stem, hea_stem))
        ecg_kwargs = {"dat_bytes": dat, "hea_bytes": hea,
                      "record_name": dat_stem or "upload", "with_xai": True}

    echo_kwargs: Optional[Dict] = None
    if echo_file is not None:
        data = await read_upload(echo_file, "echo_file", allowed_suffixes=VIDEO_SUFFIXES)
        echo_kwargs = {"video_bytes": data, "filename": echo_file.filename or "upload.avi"}

    triage_request: Optional[TriageRequest] = None
    if triage_json:
        try:
            triage_request = TriageRequest.model_validate(json.loads(triage_json))
        except json.JSONDecodeError as exc:
            raise InvalidInput("triage_json is not valid JSON: %s" % exc) from exc
        except ValueError as exc:
            raise InvalidInput("triage_json failed validation: %s" % exc) from exc

    return cxr_kwargs, ecg_kwargs, echo_kwargs, triage_request
