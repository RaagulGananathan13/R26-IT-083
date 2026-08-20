"""
Application factory.

Run it:

    cd backend
    python run.py                         # development, with reload
    uvicorn cvxai.main:app --port 8000    # equivalent, explicit

Interactive documentation is at /docs, the OpenAPI schema at /openapi.json.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from cvxai import __project_id__, __version__
from cvxai.api.v1 import api_router
from cvxai.core.errors import CvxaiError
from cvxai.core.logging import configure_logging, get_logger, new_request_id, set_request_id
from cvxai.core.registry import get_registry
from cvxai.settings import get_settings

log = get_logger("cvxai")

#: Paths a browser or probe requests on its own initiative. Logging them buries
#: the clinical traffic that matters in polling noise.
QUIET_PATHS = frozenset({
    "/api/v1/health", "/health",
    "/favicon.ico", "/service-worker.js", "/robots.txt",
    "/apple-touch-icon.png", "/apple-touch-icon-precomposed.png",
})

DESCRIPTION = """
One service over the four components of **R26-IT-083 — Explainable AI System for
Cardiovascular Disease Detection and Diagnosis**.

| Component | Modality | Answers |
|---|---|---|
| **01** · Raagul Gananathan | Chest radiograph | Cardiomegaly + 7 co-pathologies, Grad-CAM, draft report |
| **02** · Venushan T | 12-lead ECG | 5 superclasses with conformal rule-in / rule-out triage |
| **03** · Dilukshan Viyapury | Echocardiogram | Ejection fraction + 4-class severity grade |
| **04** · Abishnan J | ED triage record | ACS detection + UA / NSTEMI / STEMI subtyping |

### The shared contract

The four components answer different clinical questions on different data, so
their findings have nothing in common. What they *do* share is that each was
built around a mechanism that declines to commit when its own evidence is weak
— per-projection deferral, conformal refusal, boundary-ambiguity abstention,
clinician referral.

Every response therefore carries a `reliability` block reducing that mechanism
to one `actionability` verdict:

- `actionable` — the component stands behind this result
- `caution` — the result stands, but measured reliability is reduced
- `deferred` — the component declines to commit; refer to a clinician
- `withheld` — output suppressed after a quality or verification failure
- `unavailable` — the component could not run

A caller can apply one rule across all four modalities — *do not act on a
result that is not actionable* — without knowing anything about projections,
conformal zones or disclosure horizons. The component-native payload is
returned unmodified under `raw`.

`POST /api/v1/assessment` runs every supplied modality for one patient and
reduces the verdicts to their worst case. It is an **aggregation, not a fusion
model**: no joint model was trained, and no combined performance is claimed.

> ⚕️ Research prototype. **Not a medical device**, not clinically validated.
> Every output requires review by a qualified clinician.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)

    registry = get_registry(settings)
    app.state.registry = registry
    app.state.started_at = time.time()

    log.info("=" * 72)
    # ASCII only in log lines. A Windows console under the default codepage,
    # and any redirected log file read back as cp1252, renders U+00B7 as mojibake.
    log.info("  cvxai %s  |  project %s", __version__, __project_id__)
    log.info("  device: %s", registry.device())
    for adapter in registry.all():
        detail = adapter.status_detail
        log.info("  %-7s %-12s %s", adapter.id, adapter.status.value,
                 detail if detail else adapter.name)
    log.info("=" * 72)

    if settings.eager_load:
        log.info("eager loading enabled -- warming every serviceable component")
        registry.warm_all()

    yield

    log.info("cvxai shutting down")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="R26-IT-083 · Cardiovascular XAI Platform",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        contact={"name": "Project R26-IT-083"},
        license_info={"name": "MIT (code only; datasets are credentialed)"},
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Attach a correlation id and log the outcome of every request."""
        request_id = request.headers.get("X-Request-ID") or new_request_id()
        set_request_id(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:                      # noqa: BLE001 - logged, then re-raised
            log.exception("unhandled error on %s %s", request.method, request.url.path)
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Elapsed-Ms"] = "%.0f" % elapsed_ms
        if request.url.path not in QUIET_PATHS:
            log.info("%s %s -> %d (%.0f ms)", request.method, request.url.path,
                     response.status_code, elapsed_ms)
        return response

    @app.exception_handler(CvxaiError)
    async def handle_cvxai_error(request: Request, exc: CvxaiError):
        # Deliberate, classified failures. A component declining to answer is
        # NOT one of these -- that is a 200 with actionability set accordingly.
        log.warning("%s: %s", exc.code, exc.message)
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        # A pydantic model_validator that raises ValueError puts the exception
        # object itself in the error's `ctx`, which json.dumps cannot encode.
        # Rendering the whole detail defensively keeps a schema violation a 422
        # instead of turning it into a 500 from the error handler.
        detail = []
        for error in exc.errors():
            detail.append({
                "location": [str(part) for part in error.get("loc", ())],
                "message": str(error.get("msg", "")),
                "type": str(error.get("type", "")),
            })
        return JSONResponse(
            status_code=422,
            content={"error": "validation_error",
                     "message": "The request did not match the schema.",
                     "detail": detail})

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error",
                     "message": "%s: %s" % (type(exc).__name__, exc)})

    app.include_router(api_router)

    @app.get("/", tags=["system"], summary="Service index")
    def index():
        return {
            "service": "cvxai",
            "version": __version__,
            "project": __project_id__,
            "title": "Explainable AI System for Cardiovascular Disease Detection and Diagnosis",
            "docs": "/docs",
            "openapi": "/openapi.json",
            "endpoints": {
                "health": "/api/v1/health",
                "components": "/api/v1/components",
                "chest_radiograph": "/api/v1/cxr/analyze",
                "ecg": "/api/v1/ecg/analyze",
                "echocardiogram": "/api/v1/echo/analyze",
                "ed_triage": "/api/v1/triage/analyze",
                "multi_modal": "/api/v1/assessment",
            },
            "disclaimer": ("Research prototype. Not a medical device, not clinically "
                           "validated. Every output requires clinician review."),
        }

    # Convenience alias so a load balancer's default probe path works.
    @app.get("/health", tags=["system"], include_in_schema=False)
    def health_alias(request: Request):
        return request.app.state.registry.health(__version__, __project_id__)

    # Browsers request these unprompted when someone opens the API root. This
    # is a JSON service with no static assets, so answering "nothing here, stop
    # asking" is more honest than a 404 the browser will retry.
    @app.get("/favicon.ico", include_in_schema=False)
    @app.get("/robots.txt", include_in_schema=False)
    @app.get("/service-worker.js", include_in_schema=False)
    def _browser_probe():
        from fastapi import Response

        return Response(status_code=204)

    return app


app = create_app()
