import time
import uuid
from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager

from prodml.api.schemas import PredictionRequest, PredictionResponse
from prodml.predict import DurationPredictor
from prodml.logging_conf import setup_logging, correlation_id_var

logger = setup_logging()
predictor = DurationPredictor()


# load model
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting API and loading model...")
    predictor.load()
    yield
    logger.info("Shutting down API...")


app = FastAPI(lifespan=lifespan)


# Correlation ID
@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    corr_id = str(uuid.uuid4())
    correlation_id_var.set(corr_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = corr_id
    return response


@app.get("/health")
def health_check():
    if predictor.model is not None:
        return {"status": "healthy"}
    raise HTTPException(status_code=503, detail="Model not loaded in memory")


@app.get("/metadata")
def metadata():
    return {
        "model_version": "0.1.0",
        "framework": "scikit-learn",
        "features": ["PU_DO", "trip_distance"],
    }


# prediction link
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    start = time.time()
    try:
        # take data from user
        features = request.model_dump()
        pred = predictor.predict_one(features)
        latency = (time.time() - start) * 1000

        logger.info(f"Prediction served: {pred:.2f} mins")

        return PredictionResponse(
            prediction=pred,
            model_version="0.1.0",
            correlation_id=correlation_id_var.get(),
            latency_ms=latency,
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
