"""
Production FastAPI Model Serving API.
Exposes:
- /health: Health status and model readiness
- /model-info: Model metadata, metrics, and provenance
- /predict: Single patient risk prediction with confidence & risk band
- /predict/batch: Batch prediction endpoint
- /metrics: Prometheus metrics endpoint
"""
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from api.monitoring import (
    PREDICTION_CONFIDENCE_HISTOGRAM,
    PREDICTION_LATENCY_SECONDS,
    PREDICTION_REQUESTS_TOTAL,
)
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PatientFeatures,
    PredictionResponse,
)
from heart.predict import ModelPredictor

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s", "level":"%(levelname)s", "request_id":"%(name)s", "message":"%(message)s"}'
)
logger = logging.getLogger("heart-api")

predictor: ModelPredictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    try:
        predictor = ModelPredictor()
        logger.info(f"Model successfully loaded: {predictor.metadata.get('model_name')}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        predictor = None
    yield

app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Production-ready MLOps API for Heart Disease Risk Prediction (UCI Cleveland Dataset)",
    version="0.1.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus Auto-Instrumentation
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

@app.middleware("http")
async def add_request_id_and_logging(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.time()
    response: Response = await call_next(request)
    duration = time.time() - start_time
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{duration:.4f}s"
    logger.info(f"Path: {request.url.path} Method: {request.method} Status: {response.status_code} Duration: {duration:.4f}s ReqID: {request_id}")
    return response

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(
        status="healthy" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        version="0.1.0"
    )

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Info"])
def model_info():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return ModelInfoResponse(
        model_name=predictor.metadata.get("model_name", "Unknown"),
        metrics=predictor.metadata.get("metrics", {}),
        author=predictor.metadata.get("author", "Deenadayalan C K"),
        dataset=predictor.metadata.get("dataset", "UCI Cleveland")
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(features: PatientFeatures):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    start_time = time.time()
    res = predictor.predict_one(features.model_dump())
    latency = time.time() - start_time

    # Record metrics
    result_class = "Disease" if res["prediction"] == 1 else "Healthy"
    PREDICTION_REQUESTS_TOTAL.labels(result_class=result_class, risk_level=res["risk_level"]).inc()
    PREDICTION_CONFIDENCE_HISTOGRAM.observe(res["confidence"])
    PREDICTION_LATENCY_SECONDS.observe(latency)

    return PredictionResponse(
        prediction=res["prediction"],
        disease_present=res["disease_present"],
        confidence=res["confidence"],
        probability_disease=res["probability_disease"],
        risk_level=res["risk_level"],
        model_version="0.1.0"
    )

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(batch: BatchPredictionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    items = [p.model_dump() for p in batch.patients]
    raw_results = predictor.predict_batch(items)
    responses = [
        PredictionResponse(
            prediction=r["prediction"],
            disease_present=r["disease_present"],
            confidence=r["confidence"],
            probability_disease=r["probability_disease"],
            risk_level=r["risk_level"],
            model_version="0.1.0"
        )
        for r in raw_results
    ]
    return BatchPredictionResponse(
        predictions=responses,
        total_processed=len(responses)
    )
