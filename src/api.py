"""
FastAPI backend for IndustrialAI platform.
Endpoints: /health, /predict, /anomaly, /explain, /maintenance/ask, /history
"""
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from src.config import MODEL_VERSION, API_HOST, API_PORT, CORS_ORIGINS
from src.database import init_db, get_db, save_prediction, get_history
from src.predict import predict, models_ready, load_components as _preload_model
from src.anomaly_detection import detect_anomaly, load_anomaly_detector as _preload_anomaly
from src.preprocessing import load_preprocessor

logger = logging.getLogger(__name__)

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class SensorInput(BaseModel):
    Type: str = Field(..., description="Product type: L, M, or H")
    air_temperature_K: float = Field(..., alias="Air temperature [K]", ge=250.0, le=400.0)
    process_temperature_K: float = Field(..., alias="Process temperature [K]", ge=250.0, le=400.0)
    rotational_speed_rpm: int = Field(..., alias="Rotational speed [rpm]", ge=0, le=10000)
    torque_Nm: float = Field(..., alias="Torque [Nm]", ge=0.0, le=500.0)
    tool_wear_min: int = Field(..., alias="Tool wear [min]", ge=0, le=1000)

    model_config = {"populate_by_name": True}

    @field_validator("Type")
    @classmethod
    def validate_type(cls, v):
        if v not in {"L", "M", "H"}:
            raise ValueError(f"Type must be L, M, or H. Got: {v}")
        return v

    def to_feature_dict(self) -> dict:
        return {
            "Type": self.Type,
            "Air temperature [K]": self.air_temperature_K,
            "Process temperature [K]": self.process_temperature_K,
            "Rotational speed [rpm]": self.rotational_speed_rpm,
            "Torque [Nm]": self.torque_Nm,
            "Tool wear [min]": self.tool_wear_min,
        }


class PredictionResponse(BaseModel):
    request_id: str
    predicted_failure: bool
    failure_probability: float
    decision_threshold: float
    risk_level: str
    model_name: str
    model_version: str
    input_warnings: List[str]
    inference_latency_ms: float
    prediction_timestamp: str
    disclaimer: str


class AnomalyResponse(BaseModel):
    request_id: str
    is_anomaly: bool
    anomaly_score_raw: float
    anomaly_score_normalised: float
    contamination_threshold: float
    interpretation: str


class ExplainResponse(BaseModel):
    request_id: str
    method: str
    feature_contributions: Dict[str, float]
    summary: str
    top_risk_factors: List[str]
    top_protective_factors: List[str]
    disclaimer: str


class MaintenanceQuestion(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    prediction_context: Optional[Dict] = None
    sensor_context: Optional[Dict] = None


class MaintenanceAnswer(BaseModel):
    question: str
    answer: str
    sources: List[str]
    llm_available: bool
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    anomaly_ready: bool
    rag_available: bool
    timestamp: str
    model_version: str


# ── Application startup ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and initialise database on startup."""
    logger.info("Starting IndustrialAI API...")
    init_db()
    # Pre-load models to avoid cold-start latency
    try:
        _preload_model()
        logger.info("Prediction model loaded successfully")
    except FileNotFoundError:
        logger.warning("Prediction model not found — run training pipeline first")
    try:
        _preload_anomaly()
        logger.info("Anomaly detector loaded successfully")
    except FileNotFoundError:
        logger.warning("Anomaly detector not found — run training pipeline first")
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="IndustrialAI API",
    description=(
        "Predictive Maintenance and Failure Analysis API.\n\n"
        "⚠️ Trained on synthetic UCI AI4I 2020 data. "
        "Results are probabilistic estimates, not operational directives."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API and model health. Safe to call at any time — loads no heavy models."""
    from src.anomaly_detection import ANOMALY_MODEL_PATH
    from src.maintenance_rag import rag_ready
    return HealthResponse(
        status="ok",
        model_ready=models_ready(),
        anomaly_ready=ANOMALY_MODEL_PATH.exists(),
        rag_available=rag_ready(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_version=MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_failure(
    sensor_input: SensorInput,
    db: Session = Depends(get_db),
):
    """
    Predict machine failure probability from sensor readings.

    **Inputs:** Sensor readings for one machine record.
    **Output:** Failure class, probability, threshold, risk level, warnings.

    ⚠️ Results are probabilistic model estimates from synthetic training data.
    """
    if not models_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Run: python -m src.train_pipeline",
        )

    request_id = str(uuid.uuid4())
    feature_dict = sensor_input.to_feature_dict()

    try:
        result = predict(feature_dict)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Save to database
    save_prediction(db, {
        **result,
        "request_id": request_id,
        "sensor_inputs": feature_dict,
    })

    return PredictionResponse(request_id=request_id, **result)


@app.post("/anomaly", response_model=AnomalyResponse, tags=["Anomaly Detection"])
async def anomaly_detection(sensor_input: SensorInput):
    """
    Detect anomalies in sensor readings using Isolation Forest.

    An anomaly flag does NOT equal a confirmed failure.
    """
    from src.anomaly_detection import ANOMALY_MODEL_PATH
    if not ANOMALY_MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Anomaly detector not trained yet.")

    request_id = str(uuid.uuid4())
    feature_dict = sensor_input.to_feature_dict()

    try:
        result = detect_anomaly(feature_dict)
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return AnomalyResponse(request_id=request_id, **result)


@app.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
async def explain_prediction(sensor_input: SensorInput):
    """
    Compute SHAP feature contributions for a single prediction.

    Values explain model behaviour — not physical causation.
    """
    if not models_ready():
        raise HTTPException(status_code=503, detail="Model not trained yet.")

    request_id = str(uuid.uuid4())
    feature_dict = sensor_input.to_feature_dict()

    try:
        from src.explainability import explain_prediction as _explain
        import numpy as np
        from src.config import PROCESSED_DATA_DIR

        # Load background data for SHAP
        bg_path = PROCESSED_DATA_DIR / "X_train.npy"
        if bg_path.exists():
            X_background = np.load(bg_path)
        else:
            X_background = np.zeros((1, 8))  # minimal fallback

        result = _explain(feature_dict, X_background)
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return ExplainResponse(
        request_id=request_id,
        method=result.get("method", "unknown"),
        feature_contributions=result.get("feature_contributions", {}),
        summary=result.get("summary", ""),
        top_risk_factors=result.get("top_risk_factors", []),
        top_protective_factors=result.get("top_protective_factors", []),
        disclaimer=result.get("disclaimer", ""),
    )


@app.post("/maintenance/ask", response_model=MaintenanceAnswer, tags=["GenAI Assistant"])
async def maintenance_ask(request: MaintenanceQuestion):
    """
    Ask the GenAI maintenance assistant a question.

    Uses RAG with Chroma + Sentence Transformers. LLM optional.
    Embedding model loads lazily on first call (~300 MB, takes a few seconds).
    Gracefully degrades if vector store or LLM is unavailable.
    """
    try:
        from src.maintenance_rag import ask
        result = ask(
            question=request.question,
            prediction_context=request.prediction_context,
            sensor_context=request.sensor_context,
        )
    except (MemoryError, Exception) as e:
        logger.error(f"RAG error: {type(e).__name__}: {e}")
        # Return a graceful fallback instead of a 500 crash
        result = {
            "question": request.question,
            "answer": (
                "The RAG assistant encountered an error and could not process your question. "
                f"Error: {type(e).__name__}. "
                "This may be a temporary memory issue on the free hosting tier. Please retry."
            ),
            "sources": [],
            "llm_available": False,
            "latency_ms": 0.0,
        }

    return MaintenanceAnswer(
        question=result["question"],
        answer=result["answer"],
        sources=result.get("sources", []),
        llm_available=result.get("llm_available", False),
        latency_ms=result.get("latency_ms", 0.0),
    )


@app.get("/history", tags=["History"])
async def get_prediction_history(
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """
    Get recent prediction history.
    """
    records = get_history(db, limit=limit)
    return {"count": len(records), "records": records}


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=API_HOST, port=API_PORT)
