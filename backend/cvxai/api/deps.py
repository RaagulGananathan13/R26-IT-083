"""Shared dependencies and upload helpers for the route modules."""
from __future__ import annotations

from typing import Optional

from fastapi import Request, UploadFile

from cvxai.core.errors import InvalidInput, PayloadTooLarge
from cvxai.core.registry import ComponentRegistry, get_registry
from cvxai.settings import Settings, get_settings


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
