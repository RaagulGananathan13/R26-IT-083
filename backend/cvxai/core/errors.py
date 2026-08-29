"""
Error taxonomy and the single JSON error shape the API returns.

The distinction that matters clinically is between *the service failed* and
*the model declined to answer*. Only the first is an error. A refused ECG, a
deferred radiograph and a referred triage case are successful responses with
`status` set accordingly -- they are the components doing their job. Turning
them into HTTP errors would push callers towards retry loops around a safety
mechanism.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class CvxaiError(Exception):
    """Base class for every error this service raises deliberately."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, detail: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"error": self.code, "message": self.message}
        if self.detail:
            payload["detail"] = self.detail
        return payload


class ComponentUnavailable(CvxaiError):
    """The component cannot serve: root missing, weights absent, import failed.

    503 rather than 500: the service is healthy, this capability is not.
    """

    status_code = 503
    code = "component_unavailable"


class ComponentNotFound(CvxaiError):
    """No component is registered under the requested identifier."""

    status_code = 404
    code = "component_not_found"


class InvalidInput(CvxaiError):
    """The submitted study could not be read as the declared modality."""

    status_code = 400
    code = "invalid_input"


class PayloadTooLarge(CvxaiError):
    status_code = 413
    code = "payload_too_large"


class InferenceFailed(CvxaiError):
    """The component raised while analysing a well-formed study."""

    status_code = 500
    code = "inference_failed"
