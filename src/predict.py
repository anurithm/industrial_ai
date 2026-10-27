"""
Prediction functions for the IndustrialAI platform.
Accepts validated sensor readings and returns structured results.
"""
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

from src.config import MODELS_DIR, RANDOM_SEED, MODEL_VERSION
from src.preprocessing import load_preprocessor, preprocess_single
from src.data_loader import validate_single_input

logger = logging.getLogger(__name__)

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
THRESHOLD_PATH = MODELS_DIR / "best_threshold.json"

_model_cache: Optional[Any] = None
_preprocessor_cache: Optional[Any] = None
_threshold_cache: Optional[float] = None
_model_name_cache: Optional[str] = None


def load_model(force_reload: bool = False):
    """Load the best model (with caching). Raises if not found."""
    global _model_cache, _threshold_cache, _model_name_cache
    if _model_cache is None or force_reload:
        if not BEST_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {BEST_MODEL_PATH}. Please run training first:\n"
                "  python -m src.train_pipeline"
            )
        _model_cache = joblib.load(BEST_MODEL_PATH)
        with open(THRESHOLD_PATH) as f:
            info = json.load(f)
        _threshold_cache = info["threshold"]
        _model_name_cache = info.get("model_name", "best")
        logger.info(f"Loaded model: {_model_name_cache}, threshold: {_threshold_cache:.4f}")
    return _model_cache, _threshold_cache, _model_name_cache


def load_components(force_reload: bool = False):
    """Load model and preprocessor (cached)."""
    global _preprocessor_cache
    if _preprocessor_cache is None or force_reload:
        _preprocessor_cache = load_preprocessor()
    model, threshold, model_name = load_model(force_reload)
    return model, _preprocessor_cache, threshold, model_name


def predict(sensor_data: Dict) -> Dict:
    """
    Run failure prediction on a single sensor reading.

    Args:
        sensor_data: dict with keys matching FEATURE_COLUMNS.

    Returns:
        Structured prediction result dict. Never invents results.
    """
    model, preprocessor, threshold, model_name = load_components()

    # Validate input
    is_valid, input_warnings = validate_single_input(sensor_data)

    # Preprocess
    X = preprocess_single(sensor_data, preprocessor)

    # Time inference
    start = time.perf_counter()
    y_prob = model.predict_proba(X)[0, 1]
    latency_ms = (time.perf_counter() - start) * 1000

    y_pred = int(y_prob >= threshold)
    risk_level = _classify_risk(y_prob, threshold)

    return {
        "predicted_failure": bool(y_pred),
        "failure_probability": round(float(y_prob), 4),
        "decision_threshold": round(threshold, 4),
        "risk_level": risk_level,
        "model_name": model_name,
        "model_version": MODEL_VERSION,
        "input_warnings": input_warnings,
        "inference_latency_ms": round(latency_ms, 3),
        "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
        "disclaimer": (
            "This is a probabilistic estimate from a model trained on synthetic data. "
            "Do NOT treat this as confirmation of actual machine failure."
        ),
    }


def _classify_risk(prob: float, threshold: float) -> str:
    """Classify risk level into human-readable tiers."""
    if prob >= threshold * 1.5:
        return "HIGH"
    elif prob >= threshold:
        return "MEDIUM-HIGH"
    elif prob >= threshold * 0.5:
        return "MEDIUM"
    else:
        return "LOW"


def predict_batch(records: list) -> list:
    """Run predictions on a list of sensor readings."""
    return [predict(r) for r in records]


def models_ready() -> bool:
    """Check whether model artifacts exist."""
    return BEST_MODEL_PATH.exists() and THRESHOLD_PATH.exists()
