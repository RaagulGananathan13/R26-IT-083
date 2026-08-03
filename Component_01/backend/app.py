from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import inference

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    print("Starting up CardioVision API...")
    inference.load_models()
    yield
    # Clean up on shutdown
    print("Shutting down...")

app = FastAPI(title="CardioVision API", lifespan=lifespan)

# Allow React frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact React domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    try:
        contents = await file.read()
        
        # Run inference (Classification + GradCAM + Report Generation)
        result = inference.run_inference(contents, filename=file.filename or "")
        
        return result
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "CardioVision AI Backend is running."}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
