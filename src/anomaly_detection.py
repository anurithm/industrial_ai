"""
Isolation Forest anomaly detection for the IndustrialAI platform.
Fit on normal operating records from training data only.
"""
import logging
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Tuple

from sklearn.ensemble import IsolationForest
from sklearn.compose import ColumnTransformer

from src.config import MODELS_DIR, RANDOM_SEED, ANOMALY_CONTAMINATION
from src.preprocessing import load_preprocessor

logger = logging.getLogger(__name__)

ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_detector.joblib"

_anomaly_cache = None


def train_anomaly_detector(
    X_train_raw: "pd.DataFrame",
    y_train: np.ndarray,
) -> IsolationForest:
    """
    Fit Isolation Forest on NORMAL (non-failure) training records only.
    Threshold selection: we use the contamination parameter to flag
    approximately `contamination` fraction as anomalous. The contamination
    is set based on the observed failure rate in the dataset.

    Args:
        X_train_raw: raw (unscaled) feature DataFrame for training set
        y_train: training labels (0=normal, 1=failure)

    Returns:
        Fitted IsolationForest model
    """
    # Use only preprocessor for numeric transform
    preprocessor = load_preprocessor()

    # Select only normal (non-failure) records for fitting
    normal_mask = y_train == 0
    X_normal = X_train_raw[normal_mask]

    logger.info(
        f"Training anomaly detector on {normal_mask.sum()} normal records "
        f"({normal_mask.sum()/len(y_train):.1%} of training set)"
    )

    detector = IsolationForest(
        n_estimators=100,
        contamination=ANOMALY_CONTAMINATION,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    X_normal_t = preprocessor.transform(X_normal)
    detector.fit(X_normal_t)

    joblib.dump(detector, ANOMALY_MODEL_PATH)
    logger.info(f"Anomaly detector saved to {ANOMALY_MODEL_PATH}")
    return detector


def load_anomaly_detector(force_reload: bool = False) -> IsolationForest:
    """Load anomaly detector (cached)."""
    global _anomaly_cache
    if _anomaly_cache is None or force_reload:
        if not ANOMALY_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Anomaly detector not found at {ANOMALY_MODEL_PATH}. "
                "Run training pipeline first."
            )
        _anomaly_cache = joblib.load(ANOMALY_MODEL_PATH)
    return _anomaly_cache


def detect_anomaly(sensor_data: Dict) -> Dict:
    """
    Detect anomalies in a single sensor reading.

    Returns a dict with anomaly_score and is_anomaly flag.
    NOTE: An anomaly flag does NOT mean confirmed machine failure.
    """
    detector = load_anomaly_detector()
    preprocessor = load_preprocessor()

    import pandas as pd
    from src.config import FEATURE_COLUMNS
    df = pd.DataFrame([sensor_data])
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    df = df[feature_cols]
    X = preprocessor.transform(df)

    # IsolationForest: -1 = anomaly, 1 = normal
    prediction = detector.predict(X)[0]
    score = detector.score_samples(X)[0]  # more negative = more anomalous

    # Normalise score to 0–1 (1 = most anomalous)
    # score_samples returns negative values; more negative = more anomalous
    # We scale using training distribution if possible; otherwise simple inversion
    anomaly_score_normalised = float(1 / (1 + np.exp(score)))  # sigmoid inversion

    return {
        "is_anomaly": bool(prediction == -1),
        "anomaly_score_raw": round(float(score), 4),
        "anomaly_score_normalised": round(anomaly_score_normalised, 4),
        "contamination_threshold": ANOMALY_CONTAMINATION,
        "interpretation": (
            "Anomaly detected — sensor readings deviate from normal operating patterns. "
            "This is NOT a confirmed failure; investigate further."
            if prediction == -1
            else "Sensor readings appear within normal operating range."
        ),
    }


def evaluate_anomaly_on_labeled(
    X: np.ndarray, y_true: np.ndarray
) -> Dict:
    """Evaluate anomaly detector against labeled test data."""
    detector = load_anomaly_detector()
    predictions = detector.predict(X)  # -1=anomaly, 1=normal
    anomaly_flags = (predictions == -1).astype(int)

    from sklearn.metrics import classification_report
    report = classification_report(y_true, anomaly_flags, output_dict=True)
    return {
        "anomaly_vs_failure_overlap": float(
            np.sum((anomaly_flags == 1) & (y_true == 1)) / max(np.sum(y_true), 1)
        ),
        "classification_report": report,
    }
