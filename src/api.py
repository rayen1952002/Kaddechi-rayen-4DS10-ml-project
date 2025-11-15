from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os
from prometheus_client import generate_latest, Counter, Histogram, REGISTRY
import logging
from data_processor import DataProcessor

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'API request latency')

app = FastAPI(
    title="ML Model API",
    description="Machine Learning Model with MLflow & Prometheus",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"
)

# Initialize processor
processor = DataProcessor()

class PredictionRequest(BaseModel):
    features: list

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    model_version: str = "1.0.0"

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    try:
        processor.generate_sample_data(1000, 5)
        processor.train_model()
        processor.save_model("models/model.pkl")
        logging.info("Model trained and saved successfully")
    except Exception as e:
        logging.error(f"Startup error: {e}")

@app.get("/")
async def root():
    REQUEST_COUNT.labels(method='GET', endpoint='/').inc()
    return {"message": "ML Model API is running"}

@app.get("/health")
async def health():
    REQUEST_COUNT.labels(method='GET', endpoint='/health').inc()
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    REQUEST_COUNT.labels(method='POST', endpoint='/predict').inc()
    
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = processor.predict(features)[0]
        confidence = float(np.max(processor.model.predict_proba(features)))
        
        return PredictionResponse(
            prediction=int(prediction),
            confidence=confidence,
            model_version="1.0.0"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/metrics")
async def metrics():
    return generate_latest(REGISTRY)

@app.get("/model/info")
async def model_info():
    REQUEST_COUNT.labels(method='GET', endpoint='/model/info').inc()
    return processor.get_data_summary()
