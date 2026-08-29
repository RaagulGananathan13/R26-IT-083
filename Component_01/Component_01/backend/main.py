"""
The web API. This is the only file the frontend talks to.

Three endpoints:

    POST /predict     upload an image, optionally say whether it is AP or PA
                      returns: prediction, confidence, gradcam_image,
                               report_text, report_text_raw,
                               ground_truth_report, copathologies,
                               view, threshold, threshold_source,
                               reliability, deferral

    GET  /health      is everything loaded, and which report model
    GET  /thresholds  the threshold tables, including per-projection ones

The models are big, so they load once at startup rather than per request.
This is a separate deployment. It doesn't touch the older backend or frontend
folders.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import CORS_ORIGINS, MODEL_STATS
from backend.services.inference import InferenceService

service: InferenceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts, and once more on shutdown.
    global service
    print("=" * 64)
    print("  CardioVision AI v2  -  starting")
    print("=" * 64)
    service = InferenceService()
    print("=" * 64)
    print("  ready  -  POST /predict")
    print("=" * 64)
    yield
    service = None


app = FastAPI(title="CardioVision AI v2", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS,
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.post("/predict")
async def predict(file: UploadFile = File(...), view: str = Form(None)):
    """Run the whole pipeline on one chest X-ray.

    `view` is optional. If you pass "AP" or "PA" we use the threshold fitted for
    that projection. If you leave it out we use the global one instead of
    guessing, because guessing PA on a bedside film would raise the bar on
    exactly the patients we least want to miss.
    """
    # Startup can take a while on CPU, so the frontend may hit us too early.
    if service is None:
        raise HTTPException(503, "Models not loaded yet")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Uploaded file must be an image")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        return service.predict(data, view=view, filename=file.filename)
    except Exception as e:
        raise HTTPException(500, "Inference failed: %s" % e)


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": service is not None,
            "report_generator": getattr(service, "reportgen_stage", None),
            "stats": MODEL_STATS}


@app.get("/thresholds")
async def thresholds():
    """Returns the thresholds we use, so the frontend can display them."""
    if service is None:
        raise HTTPException(503, "Models not loaded yet")
    return {"tables": service.policy.tables,
            "projection_auroc": service.policy.projection_auroc,
            "projection_gap": service.policy.gap}
