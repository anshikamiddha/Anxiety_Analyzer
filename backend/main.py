"""
MILESTONE 6: FastAPI Backend for Anxiety Prediction
AI-Based Exam Anxiety Detector
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import time
import logging
from datetime import datetime

from model.predictor import get_predictor, LABEL_MAP

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("anxiety-api")

app = FastAPI(
    title="Exam Anxiety Detector API",
    description=(
        "AI-powered NLP system to classify exam-related anxiety from student text. "
        "Non-diagnostic tool for educational support. "
        "© 2024 Exam Anxiety Detector"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Student's text reflection (10–1000 characters)",
        example="I feel very anxious about the exam tomorrow. I haven't covered everything.",
    )

    @validator("text")
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Text must not be blank or whitespace only.")
        return v.strip()


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(
        ..., min_items=1, max_items=50,
        description="List of 1–50 student text inputs"
    )


class PredictionResult(BaseModel):
    label: str
    confidence: float
    probabilities: dict
    emoji: str
    color: str
    tips: List[str]
    model_type: str
    disclaimer: str


class PredictResponse(BaseModel):
    success: bool
    prediction: PredictionResult
    processing_time_ms: float
    timestamp: str


class BatchPredictResponse(BaseModel):
    success: bool
    total: int
    predictions: List[dict]
    distribution: dict
    processing_time_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    version: str
    timestamp: str


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Anxiety Predictor...")
    predictor = get_predictor()
    logger.info(f"Predictor ready. Model type: {'BERT' if predictor._loaded else 'Rule-Based'}")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Exam Anxiety Detector API is running 🎓",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    predictor = get_predictor()
    return HealthResponse(
        status="healthy",
        model_loaded=predictor._loaded,
        model_type="BERT" if predictor._loaded else "Rule-Based",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_anxiety(request: PredictRequest):
    """
    Analyze a student's text and classify exam anxiety level.

    Returns:
    - **label**: Low Anxiety | Moderate Anxiety | High Anxiety
    - **confidence**: model confidence score (0–1)
    - **probabilities**: per-class probability breakdown
    - **tips**: anxiety-management recommendations
    """
    try:
        start = time.time()
        predictor = get_predictor()
        result = predictor.predict(request.text)
        elapsed = round((time.time() - start) * 1000, 2)

        logger.info(f"Prediction: {result['label']} ({result['confidence']:.2%}) | {elapsed}ms")

        return PredictResponse(
            success=True,
            prediction=PredictionResult(**result),
            processing_time_ms=elapsed,
            timestamp=datetime.utcnow().isoformat(),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error.")


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest):
    """
    Batch analysis for multiple student texts.
    Returns individual predictions and overall anxiety distribution.
    """
    try:
        start = time.time()
        predictor = get_predictor()
        results = predictor.predict_batch(request.texts)
        elapsed = round((time.time() - start) * 1000, 2)

        distribution = {"Low Anxiety": 0, "Moderate Anxiety": 0, "High Anxiety": 0}
        for r in results:
            distribution[r["label"]] += 1

        logger.info(f"Batch prediction: {len(results)} items | {elapsed}ms")

        return BatchPredictResponse(
            success=True,
            total=len(results),
            predictions=results,
            distribution=distribution,
            processing_time_ms=elapsed,
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        logger.error(f"Batch error: {e}")
        raise HTTPException(status_code=500, detail="Internal batch prediction error.")


@app.get("/labels", tags=["Info"])
async def get_labels():
    """Returns available anxiety labels."""
    return {
        "labels": list(LABEL_MAP.values()),
        "description": {
            "Low Anxiety": "Minimal stress; student appears well-prepared and calm.",
            "Moderate Anxiety": "Noticeable stress; student may benefit from support strategies.",
            "High Anxiety": "Significant distress; professional support is recommended.",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
