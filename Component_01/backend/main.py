"""
FastAPI application - Component_01 v2.

Serves the retrained models behind the Component_1 UI contract:

    POST /predict   file=<image>, view=<"AP"|"PA"|"">
      -> prediction, confidence, gradcam_image, report_text, report_text_raw,
         ground_truth_report, copathologies
         + view, threshold, threshold_source, reliability   (new)

Separate deployment. Nothing in ../../backend, ../../frontend or
../../Component_1 is read or written.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import CORS_ORIGINS, MODEL_STATS
from backend.services.inference import InferenceService

service: InferenceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    """Classify, explain and report on one chest X-ray.

    `view` ("AP" or "PA") selects the projection-specific operating point.
    Optional: omitting it uses the global threshold rather than guessing.
    Guessing PA on a bedside film would under-call cardiomegaly on exactly the
    patients least able to tolerate a missed diagnosis.
    """
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
    """The operating points, including the per-projection ones."""
    if service is None:
        raise HTTPException(503, "Models not loaded yet")
    return {"tables": service.policy.tables,
            "projection_auroc": service.policy.projection_auroc,
            "projection_gap": service.policy.gap}
